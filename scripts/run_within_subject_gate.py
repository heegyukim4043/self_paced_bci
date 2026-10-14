from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.features import decision_subsample, sample_mask
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    operating_point_dict,
    tune_source_operating_points,
)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def chronological_split(intervals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    midpoint = len(intervals) // 2
    return intervals[:midpoint], intervals[midpoint:]


def fit_regularized_lda(
    features: np.ndarray,
    times: np.ndarray,
    command_intervals: np.ndarray,
    idle_intervals: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(features, dtype=float).reshape(len(features), -1)
    command = x[sample_mask(times, command_intervals.tolist())]
    idle = x[sample_mask(times, idle_intervals.tolist())]
    location = np.mean(np.r_[command, idle], axis=0)
    scale = np.std(np.r_[command, idle], axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    command_z = (command - location) / scale
    idle_z = (idle - location) / scale
    mean_command = np.mean(command_z, axis=0)
    mean_idle = np.mean(idle_z, axis=0)
    covariance = (
        np.cov(command_z, rowvar=False) * (len(command_z) - 1)
        + np.cov(idle_z, rowvar=False) * (len(idle_z) - 1)
    ) / (len(command_z) + len(idle_z) - 2)
    trace_scale = float(np.trace(covariance) / covariance.shape[0])
    regularized = covariance + ridge * trace_scale * np.eye(covariance.shape[0])
    weight = np.linalg.solve(regularized, mean_command - mean_idle)
    intercept = -0.5 * float((mean_command + mean_idle) @ weight)
    return location, scale, weight, intercept


def score_features(
    features: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    weight: np.ndarray,
    intercept: float,
) -> np.ndarray:
    x = np.asarray(features, dtype=float).reshape(len(features), -1)
    return ((x - location) / scale) @ weight + intercept


def sliced_sequence(
    times: np.ndarray,
    scores: np.ndarray,
    idle_intervals: np.ndarray,
    command_intervals: np.ndarray,
    start: float,
    stop: float,
) -> ScoredSequence:
    keep = (times >= start) & (times < stop)
    return ScoredSequence(
        times=times[keep],
        scores=scores[keep],
        idle_intervals=idle_intervals,
        command_intervals=command_intervals,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument(
        "--decisions", nargs="+", default=["threshold", "ma_dwell"]
    )
    parser.add_argument(
        "--target-fp", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0]
    )
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=160)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/within_subject_gate.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for session in args.sessions:
        paths = sorted(
            (args.cache_root / f"session{session}").glob(
                "sub-*_tier0_features.npz"
            )
        )
        for path in paths:
            cache = load_cache(path)
            if args.subjects and int(cache["subject"]) not in args.subjects:
                continue
            times, feature_values = decision_subsample(
                cache["continuous_times"],
                cache["continuous_features"],
                args.decision_step,
            )
            command_train, command_test = chronological_split(
                cache["command_intervals"]
            )
            idle_train, idle_test = chronological_split(cache["iti_intervals"])
            location, scale, weight, intercept = fit_regularized_lda(
                feature_values,
                times,
                command_train,
                idle_train,
                args.ridge,
            )
            score_values = score_features(
                feature_values, location, scale, weight, intercept
            )
            train_start = min(command_train[0, 0], idle_train[0, 0])
            train_stop = max(command_train[-1, 1], idle_train[-1, 1])
            test_start = min(command_test[0, 0], idle_test[0, 0])
            test_stop = max(command_test[-1, 1], idle_test[-1, 1]) + args.decision_step
            train_sequence = sliced_sequence(
                times,
                score_values,
                idle_train,
                command_train,
                train_start,
                train_stop,
            )
            test_sequence = sliced_sequence(
                times,
                score_values,
                idle_test,
                command_test,
                test_start,
                test_stop,
            )
            for decision in args.decisions:
                points = tune_source_operating_points(
                    [train_sequence],
                    decision,
                    args.target_fp,
                    args.decision_step,
                    ma_seconds=0.5,
                    dwell_seconds=0.3,
                    refractory_seconds=1.0,
                    grid_size=args.grid_size,
                )
                for target_fp in args.target_fp:
                    point = points[float(target_fp)]
                    alarms = apply_decision(
                        test_sequence,
                        decision,
                        point.threshold,
                        args.decision_step,
                        ma_seconds=0.5,
                        dwell_seconds=0.3,
                        refractory_seconds=1.0,
                    ).alarms
                    observed = evaluate_alarms(
                        test_sequence.times,
                        alarms,
                        idle_test,
                        command_test,
                    )
                    null_tpr = np.empty(args.shifts)
                    for shift_index, offset in enumerate(
                        rng.integers(1, len(alarms), size=args.shifts)
                    ):
                        null_tpr[shift_index] = evaluate_alarms(
                            test_sequence.times,
                            np.roll(alarms, int(offset)),
                            idle_test,
                            command_test,
                        ).true_positive_rate
                    null_median = float(np.median(null_tpr))
                    rows.append(
                        {
                            "subject": int(cache["subject"]),
                            "session": session,
                            "model": "within_subject_regularized_lda",
                            "decision": decision,
                            "target_fp_setting": target_fp,
                            "operating_point": operating_point_dict(point),
                            "test_fp_per_minute": observed.fp_per_minute,
                            "test_tpr": observed.true_positive_rate,
                            "null_tpr_median": null_median,
                            "excess_tpr": observed.true_positive_rate - null_median,
                            "observed_over_null": (
                                observed.true_positive_rate / null_median
                                if null_median > 0
                                else None
                            ),
                            "circular_shift_p_value": float(
                                (
                                    1
                                    + np.sum(
                                        null_tpr >= observed.true_positive_rate
                                    )
                                )
                                / (args.shifts + 1)
                            ),
                        }
                    )

    summaries = []
    keys = sorted(
        {
            (row["session"], row["decision"], row["target_fp_setting"])
            for row in rows
        }
    )
    for session, decision, target_fp in keys:
        group = [
            row
            for row in rows
            if (row["session"], row["decision"], row["target_fp_setting"])
            == (session, decision, target_fp)
        ]
        observed = float(np.median([row["test_tpr"] for row in group]))
        null = float(np.median([row["null_tpr_median"] for row in group]))
        summaries.append(
            {
                "session": session,
                "decision": decision,
                "target_fp_setting": target_fp,
                "subjects": len(group),
                "test_fp_per_minute_median": float(
                    np.median([row["test_fp_per_minute"] for row in group])
                ),
                "observed_tpr_median": observed,
                "null_tpr_median": null,
                "excess_tpr_median": float(
                    np.median([row["excess_tpr"] for row in group])
                ),
                "ratio_of_medians": observed / null if null > 0 else None,
                "significant_fraction_p_lt_0_05": float(
                    np.mean(
                        [row["circular_shift_p_value"] < 0.05 for row in group]
                    )
                ),
            }
        )

    payload = finite_or_none({"summary": summaries, "rows": rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
