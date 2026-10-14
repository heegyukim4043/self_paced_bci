from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.config import ExperimentConfig
from uncued_mi.features import decision_subsample
from uncued_mi.metrics import chance_tpr, evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    operating_point_dict,
    tune_source_operating_points,
)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_sequence(
    cache: dict[str, np.ndarray],
    normalizer: LocationScale,
    step_seconds: float,
) -> ScoredSequence:
    channels = tuple(str(item) for item in cache["channels"])
    scores = tier0_from_features(
        cache["continuous_features"], channels, normalizer
    )
    times, scores = decision_subsample(
        cache["continuous_times"], scores, step_seconds
    )
    return ScoredSequence(
        times=times,
        scores=scores,
        idle_intervals=cache["iti_intervals"],
        command_intervals=cache["command_intervals"],
    )


def fit_normalizer(values: np.ndarray, axis, method: str) -> LocationScale:
    if method == "mean_std":
        return LocationScale.fit(values, axis=axis)
    if method == "median_mad":
        return LocationScale.fit_median_mad(values, axis=axis)
    raise ValueError(method)


def pooled_source_normalizer(caches: list[dict[str, np.ndarray]], method: str) -> LocationScale:
    values = np.concatenate([cache["pre_features"] for cache in caches], axis=0)
    return fit_normalizer(values, axis=0, method=method)


def own_idle_normalizer(cache: dict[str, np.ndarray], method: str = "mean_std") -> LocationScale:
    return fit_normalizer(cache["pre_features"], axis=0, method=method)


def rest_score(cache: dict[str, np.ndarray], key: str, model: LocationScale) -> np.ndarray:
    channels = tuple(str(item) for item in cache["channels"])
    return tier0_from_features(cache[key], channels, model)


def interval_mask(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in np.asarray(intervals).reshape(-1, 2):
        mask |= (times >= start) & (times < stop)
    return mask


def split_intervals(intervals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.asarray(intervals).reshape(-1, 2)
    midpoint = len(intervals) // 2
    return intervals[:midpoint], intervals[midpoint:]


def metrics_dict(metrics) -> dict:
    value = metrics.__dict__.copy()
    value["delay_seconds"] = list(value["delay_seconds"])
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--target-subjects", type=int, nargs="*")
    parser.add_argument(
        "--decisions",
        nargs="+",
        default=["threshold", "ma_dwell", "accumulator"],
    )
    parser.add_argument("--target-fp", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--accumulator-k-quantile", type=float, default=0.9)
    parser.add_argument("--max-accumulation-seconds", type=float, default=5.0)
    parser.add_argument(
        "--normalization", choices=["mean_std", "median_mad"], default="mean_std"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/tier0_loso.json")
    )
    args = parser.parse_args()

    paths = sorted((args.cache_root / f"session{args.session}").glob("sub-*_tier0_features.npz"))
    if len(paths) < 2:
        raise SystemExit("At least two cached subjects are required for LOSO")
    caches = {int(load_cache(path)["subject"]): load_cache(path) for path in paths}
    targets = args.target_subjects or sorted(caches)
    config = ExperimentConfig()
    rows: list[dict] = []

    for target_subject in targets:
        target = caches[target_subject]
        source = [cache for subject, cache in caches.items() if subject != target_subject]
        source_sequences = [
            make_sequence(cache, own_idle_normalizer(cache, args.normalization), args.decision_step)
            for cache in source
        ]
        source_idle_scores = np.concatenate(
            [
                sequence.scores[
                    interval_mask(sequence.times, sequence.idle_intervals)
                ]
                for sequence in source_sequences
            ]
        )
        drift = float(
            np.quantile(source_idle_scores, args.accumulator_k_quantile)
        )
        global_model = pooled_source_normalizer(source, args.normalization)
        alignment_name = (
            "idle_feature_alignment"
            if args.normalization == "mean_std"
            else f"idle_feature_alignment_{args.normalization}"
        )
        target_models = {
            "no_calibration": global_model,
            alignment_name: own_idle_normalizer(
                target, args.normalization
            ),
        }

        for decision in args.decisions:
            decision_drift = drift if decision == "accumulator" else None
            operating_points = tune_source_operating_points(
                source_sequences,
                decision,
                args.target_fp,
                args.decision_step,
                drift=decision_drift,
                ma_seconds=0.5,
                dwell_seconds=0.3,
                refractory_seconds=1.0,
                max_accumulation_seconds=args.max_accumulation_seconds,
                grid_size=args.grid_size,
            )
            aligned_sequence = make_sequence(
                target, own_idle_normalizer(target), args.decision_step
            )
            oracle_idle_cal, oracle_idle_eval = split_intervals(
                target["iti_intervals"]
            )
            _, oracle_command_eval = split_intervals(
                target["command_intervals"]
            )
            oracle_calibration_sequence = ScoredSequence(
                times=aligned_sequence.times,
                scores=aligned_sequence.scores,
                idle_intervals=oracle_idle_cal,
                command_intervals=np.empty((0, 2)),
            )
            oracle_points = tune_source_operating_points(
                [oracle_calibration_sequence],
                decision,
                args.target_fp,
                args.decision_step,
                drift=decision_drift,
                ma_seconds=0.5,
                dwell_seconds=0.3,
                refractory_seconds=1.0,
                max_accumulation_seconds=args.max_accumulation_seconds,
                grid_size=args.grid_size,
            )
            for target_fp in args.target_fp:
                operating_point = operating_points[float(target_fp)]
                oracle_point = oracle_points[float(target_fp)]
                for calibration, model in target_models.items():
                    sequence = make_sequence(target, model, args.decision_step)
                    alarms = apply_decision(
                        sequence,
                        decision,
                        operating_point.threshold,
                        args.decision_step,
                        drift=decision_drift,
                        ma_seconds=0.5,
                        dwell_seconds=0.3,
                        refractory_seconds=1.0,
                        max_accumulation_seconds=args.max_accumulation_seconds,
                    ).alarms
                    iti = evaluate_alarms(
                        sequence.times,
                        alarms,
                        target["iti_intervals"],
                        target["command_intervals"],
                    )

                    post_scores = rest_score(target, "post_features", model)
                    post_times = target["post_times"]
                    post_times, post_scores = decision_subsample(
                        post_times, post_scores, args.decision_step
                    )
                    post_sequence = ScoredSequence(
                        times=post_times,
                        scores=post_scores,
                        idle_intervals=np.asarray(
                            [[post_times[0], post_times[-1] + args.decision_step]]
                        ),
                        command_intervals=np.empty((0, 2)),
                    )
                    post_alarms = apply_decision(
                        post_sequence,
                        decision,
                        operating_point.threshold,
                        args.decision_step,
                        drift=decision_drift,
                        ma_seconds=0.5,
                        dwell_seconds=0.3,
                        refractory_seconds=1.0,
                        max_accumulation_seconds=args.max_accumulation_seconds,
                    ).alarms
                    post = evaluate_alarms(
                        post_times,
                        post_alarms,
                        post_sequence.idle_intervals,
                        np.empty((0, 2)),
                    )
                    fixation_onset = target["fixation_intervals"].copy()
                    fixation_onset[:, 1] = fixation_onset[:, 0] + 1.0
                    fixation = evaluate_alarms(
                        sequence.times,
                        alarms,
                        fixation_onset,
                        np.empty((0, 2)),
                    )
                    oracle_alarms = apply_decision(
                        aligned_sequence,
                        decision,
                        oracle_point.threshold,
                        args.decision_step,
                        drift=decision_drift,
                        ma_seconds=0.5,
                        dwell_seconds=0.3,
                        refractory_seconds=1.0,
                        max_accumulation_seconds=args.max_accumulation_seconds,
                    ).alarms
                    oracle_evaluation = evaluate_alarms(
                        aligned_sequence.times,
                        oracle_alarms,
                        oracle_idle_eval,
                        oracle_command_eval,
                    )
                    chance = chance_tpr(
                        iti.fp_per_minute,
                        float(np.median(
                            target["command_intervals"][:, 1]
                            - target["command_intervals"][:, 0]
                        )),
                    )
                    rows.append(
                        {
                            "subject": target_subject,
                            "session": args.session,
                            "normalization": args.normalization,
                            "decision": decision,
                            "calibration": calibration,
                            "target_fp_setting": target_fp,
                            "operating_point": operating_point_dict(operating_point),
                            "oracle_operating_point": operating_point_dict(oracle_point),
                            "oracle_calibration_fp_per_minute": oracle_point.source_fp_per_minute,
                            "oracle_evaluation": metrics_dict(oracle_evaluation),
                            "oracle_fp_per_minute": oracle_evaluation.fp_per_minute,
                            "oracle_true_positive_rate": oracle_evaluation.true_positive_rate,
                            "oracle_median_delay_seconds": oracle_evaluation.median_delay_seconds,
                            "tpr_fraction_of_oracle": (
                                iti.true_positive_rate
                                / oracle_evaluation.true_positive_rate
                                if oracle_evaluation.true_positive_rate > 0
                                else None
                            ),
                            "chance_tpr": chance,
                            "tpr_over_chance": (
                                iti.true_positive_rate / chance
                                if chance > 0
                                else None
                            ),
                            "fp_ratio_n": (
                                iti.fp_per_minute
                                / operating_point.source_fp_per_minute
                                if operating_point.source_fp_per_minute > 0
                                else None
                            ),
                            "iti": metrics_dict(iti),
                            "post_rest": metrics_dict(post),
                            "fixation_onset": metrics_dict(fixation),
                            "vep_rate_ratio_vs_post_rest": (
                                fixation.fp_per_minute / post.fp_per_minute
                                if post.fp_per_minute > 0
                                else None
                            ),
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
