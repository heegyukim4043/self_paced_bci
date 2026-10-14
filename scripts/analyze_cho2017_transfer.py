"""Replicate Tier-0 operating-point transfer on seam-safe Cho2017 features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.features import decision_subsample
from uncued_mi.operating_point import (
    ScoredSequence,
    aggregate_metrics,
    evaluate_sequence,
    operating_point_dict,
    tune_source_operating_points,
)


STEP = 0.1
TARGET_FPS = (1.0, 2.0, 4.0)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def score(features: np.ndarray, channels: tuple[str, ...], model: LocationScale) -> np.ndarray:
    return tier0_from_features(features, channels, model)


def sequence(
    times: np.ndarray,
    scores: np.ndarray,
    *,
    idle: bool = False,
    command: bool = False,
) -> ScoredSequence:
    times, scores = decision_subsample(times, scores, STEP)
    exposure = np.asarray([[float(times[0]), float(times[-1] + STEP)]])
    return ScoredSequence(
        times=times,
        scores=scores,
        idle_intervals=exposure if idle else np.empty((0, 2)),
        command_intervals=np.asarray([[2.5, 5.0]]) if command else np.empty((0, 2)),
    )


def subject_sequences(
    cache: dict[str, np.ndarray],
    model: LocationScale,
) -> tuple[ScoredSequence, list[ScoredSequence], list[ScoredSequence]]:
    channels = tuple(str(value) for value in cache["channels"])
    rest = sequence(
        cache["rest_evaluation_times"],
        score(cache["rest_evaluation_features"], channels, model),
        idle=True,
    )
    trials = [
        sequence(cache["trial_times"], score(features, channels, model), command=True)
        for features in cache["trial_features"]
    ]
    noises = [
        sequence(cache["noise_times"], score(features, channels, model), idle=True)
        for features in cache["noise_features"]
    ]
    return rest, trials, noises


def metrics_dict(value) -> dict:
    def finite_or_none(number: float):
        return float(number) if np.isfinite(number) else None

    return {
        "fp_per_minute": finite_or_none(value.fp_per_minute),
        "true_positive_rate": finite_or_none(value.true_positive_rate),
        "median_delay_seconds": finite_or_none(value.median_delay_seconds),
        "false_alarms": value.false_alarms,
        "detected_events": value.detected_events,
        "total_events": value.total_events,
        "idle_seconds": value.idle_seconds,
    }


def evaluate_all(sequences: list[ScoredSequence], threshold: float):
    return aggregate_metrics(
        [
            evaluate_sequence(
                item,
                "ma_dwell",
                threshold,
                STEP,
                ma_seconds=0.5,
                dwell_seconds=0.3,
                refractory_seconds=1.0,
            )
            for item in sequences
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/cho2017_tier0")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/cho2017_transfer.json")
    )
    parser.add_argument("--grid-size", type=int, default=80)
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".checkpoint.json")

    paths = sorted(args.cache_root.glob("sub-*_tier0_features.npz"))
    if len(paths) < 2:
        raise ValueError(f"Need at least two Cho2017 caches, found {len(paths)}")
    caches = {int(load_cache(path)["subject"]): load_cache(path) for path in paths}
    subjects = sorted(caches)

    own_models = {
        subject: LocationScale.fit(caches[subject]["rest_calibration_features"], axis=0)
        for subject in subjects
    }
    own_sequences = {
        subject: subject_sequences(caches[subject], own_models[subject])
        for subject in subjects
    }

    rows: list[dict] = []
    if checkpoint.exists():
        rows = json.loads(checkpoint.read_text(encoding="utf-8"))
    completed = {int(row["subject"]) for row in rows}
    for target in subjects:
        if target in completed:
            print(f"checkpoint subject={target}", flush=True)
            continue
        sources = [subject for subject in subjects if subject != target]
        source_training = [
            item
            for subject in sources
            for item in ([own_sequences[subject][0]] + own_sequences[subject][1])
        ]
        operating_points = tune_source_operating_points(
            source_training,
            "ma_dwell",
            TARGET_FPS,
            STEP,
            ma_seconds=0.5,
            dwell_seconds=0.3,
            refractory_seconds=1.0,
            grid_size=args.grid_size,
        )

        pooled_source_model = LocationScale.fit(
            np.concatenate(
                [caches[subject]["rest_calibration_features"] for subject in sources],
                axis=0,
            ),
            axis=0,
        )
        target_conditions = {
            "no_calibration": subject_sequences(caches[target], pooled_source_model),
            "idle_feature_alignment": own_sequences[target],
        }
        for target_fp, operating_point in operating_points.items():
            threshold = operating_point.threshold
            for calibration, (rest, trials, noises) in target_conditions.items():
                rest_metrics = evaluate_all([rest], threshold)
                command_metrics = evaluate_all(trials, threshold)
                noise_rows = []
                for label, noise in zip(caches[target]["noise_labels"], noises):
                    noise_rows.append(
                        {
                            "noise": str(label),
                            **metrics_dict(evaluate_all([noise], threshold)),
                        }
                    )
                rows.append(
                    {
                        "subject": target,
                        "target_fp_setting": target_fp,
                        "calibration": calibration,
                        "operating_point": {
                            key: (
                                float(value)
                                if isinstance(value, float) and np.isfinite(value)
                                else None
                                if isinstance(value, float)
                                else value
                            )
                            for key, value in operating_point_dict(operating_point).items()
                        },
                        "rest": metrics_dict(rest_metrics),
                        "command": metrics_dict(command_metrics),
                        "dead_gate": (
                            rest_metrics.false_alarms == 0
                            and command_metrics.detected_events == 0
                        ),
                        "noise": noise_rows,
                    }
                )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"analyzed subject={target}", flush=True)

    summary = []
    for target_fp in TARGET_FPS:
        for calibration in ("no_calibration", "idle_feature_alignment"):
            subset = [
                row
                for row in rows
                if row["target_fp_setting"] == target_fp
                and row["calibration"] == calibration
            ]
            summary.append(
                {
                    "target_fp_setting": target_fp,
                    "calibration": calibration,
                    "subjects": len(subset),
                    "dead_gate_fraction": float(np.mean([row["dead_gate"] for row in subset])),
                    "median_rest_fp_per_minute": float(
                        np.median([row["rest"]["fp_per_minute"] for row in subset])
                    ),
                    "median_command_tpr": float(
                        np.median([row["command"]["true_positive_rate"] for row in subset])
                    ),
                }
            )

    result = {
        "protocol": {
            "source_operating_point": "LOSO source FP/min on second-half rest",
            "calibration": "first-half rest (~33 s), mean/std feature alignment",
            "evaluation_idle": "held-out second-half rest (~33 s)",
            "decision": "MA 0.5 s + dwell 0.3 s + refractory 1.0 s",
            "mi_positive": "cue +0.5 to +3.0 s",
            "seam_policy": "filter and decision state reset for every 7 s trial",
            "important_limitation": (
                "2 s causal filter warm-up plus 2 s feature window leaves only "
                "trial-relative 4.0-5.0 s observable inside the positive interval"
            ),
            "noise_labels": "source cell order only; semantic names absent from MAT",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
