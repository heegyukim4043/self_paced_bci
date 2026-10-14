from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from uncued_mi.features import decision_subsample, sample_mask
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from run_matched_lda_scaling import (
    auc,
    fit_lda,
    half,
    load_cache,
    rest_normalized_features,
    score,
    temporal_profile,
    tune_ridge,
    xy_from_intervals,
)


def fit_arm(
    caches: dict[int, dict[str, np.ndarray]],
    normalized: dict[int, np.ndarray],
    subjects: list[int],
    ridge: float,
) -> tuple[np.ndarray, ...]:
    x_parts, y_parts = [], []
    for subject in subjects:
        cache = caches[subject]
        x, y = xy_from_intervals(
            normalized[subject],
            cache["continuous_times"],
            cache["command_intervals"],
            cache["iti_intervals"],
        )
        x_parts.append(x)
        y_parts.append(y)
    return fit_lda(np.concatenate(x_parts), np.concatenate(y_parts), ridge)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--target-fp", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--ridge-candidates", type=float, nargs="*")
    parser.add_argument("--shifts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--test-calibration-seconds",
        type=float,
        help="Use only the first N seconds of the evaluation-session pre-rest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cross_session_matched_lda_subset20.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    session_caches = {}
    session_normalized = {}
    for session in (1, 2):
        paths = sorted(
            (args.cache_root / f"session{session}").glob("sub-*_tier0_features.npz")
        )
        caches = {int(load_cache(path)["subject"]): load_cache(path) for path in paths}
        session_caches[session] = caches
        session_normalized[session] = {
            subject: rest_normalized_features(cache)
            for subject, cache in caches.items()
        }
    rows = []
    for train_session, test_session in ((1, 2), (2, 1)):
        train_caches = session_caches[train_session]
        train_normalized = session_normalized[train_session]
        test_caches = session_caches[test_session]
        test_normalized = session_normalized[test_session]
        if args.test_calibration_seconds is not None:
            test_normalized = {}
            for subject, cache in test_caches.items():
                available = len(cache["pre_features"])
                step = float(np.median(np.diff(cache["pre_times"])))
                count = min(
                    available,
                    max(2, int(np.floor(args.test_calibration_seconds / step))),
                )
                calibration = cache["pre_features"][:count]
                mean = np.mean(calibration, axis=0)
                std = np.std(calibration, axis=0, ddof=1)
                std = np.where(std > 1e-8, std, 1.0)
                test_normalized[subject] = (
                    cache["continuous_features"] - mean
                ) / std
        for target_subject in args.subjects:
            source_subjects = [
                subject for subject in sorted(train_caches) if subject != target_subject
            ]
            arm_subjects = {
                "target_cross_session": [target_subject],
                "source_all_53_cross_session": source_subjects,
            }
            arms = {}
            arm_ridges = {}
            for arm, subjects in arm_subjects.items():
                selected = [
                    (
                        subject,
                        train_caches[subject]["command_intervals"],
                        train_caches[subject]["iti_intervals"],
                    )
                    for subject in subjects
                ]
                selected_ridge = (
                    tune_ridge(
                        selected,
                        train_caches,
                        train_normalized,
                        args.ridge_candidates,
                        rng,
                    )
                    if args.ridge_candidates
                    else args.ridge
                )
                arms[arm] = fit_arm(
                    train_caches, train_normalized, subjects, selected_ridge
                )
                arm_ridges[arm] = selected_ridge
            target = test_caches[target_subject]
            command_test = half(target["command_intervals"], "second")
            idle_test = half(target["iti_intervals"], "second")
            for arm, model in arms.items():
                times, scores = decision_subsample(
                    target["continuous_times"],
                    score(test_normalized[target_subject], model),
                    args.decision_step,
                )
                sequence = ScoredSequence(times, scores, idle_test, command_test)
                points = tune_source_operating_points(
                    [sequence],
                    "ma_dwell",
                    args.target_fp,
                    args.decision_step,
                    ma_seconds=0.5,
                    dwell_seconds=0.3,
                    refractory_seconds=1.0,
                    grid_size=args.grid_size,
                )
                command_mask = sample_mask(times, command_test.tolist())
                idle_mask = sample_mask(times, idle_test.tolist())
                test_auc = auc(scores[command_mask], scores[idle_mask])
                for target_fp, point in points.items():
                    decision_result = apply_decision(
                        sequence,
                        "ma_dwell",
                        point.threshold,
                        args.decision_step,
                        ma_seconds=0.5,
                        dwell_seconds=0.3,
                        refractory_seconds=1.0,
                    )
                    observed = evaluate_alarms(
                        times, decision_result.alarms, idle_test, command_test
                    )
                    null = np.empty(args.shifts)
                    for index, offset in enumerate(
                        rng.integers(1, len(decision_result.alarms), size=args.shifts)
                    ):
                        null[index] = evaluate_alarms(
                            times,
                            np.roll(decision_result.alarms, int(offset)),
                            idle_test,
                            command_test,
                        ).true_positive_rate
                    rows.append(
                        {
                            "target_subject": target_subject,
                            "train_session": train_session,
                            "test_session": test_session,
                            "direction": f"s{train_session}_to_s{test_session}",
                            "arm": arm,
                            "selected_ridge": arm_ridges[arm],
                            "test_calibration_seconds": (
                                args.test_calibration_seconds
                                if args.test_calibration_seconds is not None
                                else "full"
                            ),
                            "target_fp_axis": target_fp,
                            "test_window_auc": test_auc,
                            "test_fp_per_minute": observed.fp_per_minute,
                            "observed_tpr": observed.true_positive_rate,
                            "null_tpr_median": float(np.median(null)),
                            "excess_tpr": float(
                                observed.true_positive_rate - np.median(null)
                            ),
                            **temporal_profile(
                                times,
                                scores,
                                decision_result.statistic,
                                point.threshold,
                                command_test,
                                idle_test,
                                args.decision_step,
                            ),
                        }
                    )
    summaries = []
    for direction, arm, target_fp in sorted(
        {(r["direction"], r["arm"], r["target_fp_axis"]) for r in rows}
    ):
        group = [
            r for r in rows
            if (r["direction"], r["arm"], r["target_fp_axis"])
            == (direction, arm, target_fp)
        ]
        summaries.append(
            {
                "direction": direction,
                "arm": arm,
                "target_fp_axis": target_fp,
                "subjects": len(group),
                **{
                    f"{key}_median": float(np.median([r[key] for r in group]))
                    for key in (
                        "test_window_auc",
                        "test_fp_per_minute",
                        "observed_tpr",
                        "null_tpr_median",
                        "excess_tpr",
                        "hit_density_mean",
                        "command_score_ar1_mean",
                        "fragmentation_bursts_per_exceed_second_mean",
                    )
                },
            }
        )
    payload = {"summary": summaries, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
