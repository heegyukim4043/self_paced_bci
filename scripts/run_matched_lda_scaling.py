from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.features import decision_subsample, sample_mask
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    tune_source_operating_points,
)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def half(intervals: np.ndarray, part: str) -> np.ndarray:
    intervals = np.asarray(intervals).reshape(-1, 2)
    midpoint = len(intervals) // 2
    return intervals[:midpoint] if part == "first" else intervals[midpoint:]


def rest_normalized_features(cache: dict[str, np.ndarray]) -> np.ndarray:
    mean = np.mean(cache["pre_features"], axis=0)
    std = np.std(cache["pre_features"], axis=0, ddof=1)
    std = np.where(std > 1e-8, std, 1.0)
    return (cache["continuous_features"] - mean) / std


def xy_from_intervals(
    features: np.ndarray,
    times: np.ndarray,
    command: np.ndarray,
    idle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(len(features), -1)
    command_x = flat[sample_mask(times, command.tolist())]
    idle_x = flat[sample_mask(times, idle.tolist())]
    return np.r_[command_x, idle_x], np.r_[
        np.ones(len(command_x)), np.zeros(len(idle_x))
    ]


def fit_lda(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, ...]:
    location = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - location) / scale
    positive = z[y == 1]
    negative = z[y == 0]
    mean_positive = np.mean(positive, axis=0)
    mean_negative = np.mean(negative, axis=0)
    covariance = (
        np.cov(positive, rowvar=False) * (len(positive) - 1)
        + np.cov(negative, rowvar=False) * (len(negative) - 1)
    ) / (len(positive) + len(negative) - 2)
    trace_scale = float(np.trace(covariance) / covariance.shape[0])
    weight = np.linalg.solve(
        covariance + ridge * trace_scale * np.eye(covariance.shape[0]),
        mean_positive - mean_negative,
    )
    intercept = -0.5 * float((mean_positive + mean_negative) @ weight)
    return location, scale, weight, intercept


def tune_ridge(
    selected: list[tuple[int, np.ndarray, np.ndarray]],
    caches: dict[int, dict[str, np.ndarray]],
    normalized: dict[int, np.ndarray],
    candidates: list[float],
    rng: np.random.Generator,
) -> float:
    if len(selected) == 1:
        subject, command, idle = selected[0]
        command_cut = max(1, int(len(command) * 0.8))
        idle_cut = max(1, int(len(idle) * 0.8))
        train_items = [(subject, command[:command_cut], idle[:idle_cut])]
        valid_items = [(subject, command[command_cut:], idle[idle_cut:])]
    else:
        order = rng.permutation(len(selected))
        valid_count = max(1, int(np.ceil(len(selected) * 0.2)))
        valid_index = set(order[:valid_count].tolist())
        train_items = [item for index, item in enumerate(selected) if index not in valid_index]
        valid_items = [item for index, item in enumerate(selected) if index in valid_index]

    def collect(items):
        x_parts, y_parts = [], []
        for subject, command, idle in items:
            x, y = xy_from_intervals(
                normalized[subject],
                caches[subject]["continuous_times"],
                command,
                idle,
            )
            x_parts.append(x)
            y_parts.append(y)
        return np.concatenate(x_parts), np.concatenate(y_parts)

    train_x, train_y = collect(train_items)
    valid_x, valid_y = collect(valid_items)
    best = None
    for candidate in candidates:
        model = fit_lda(train_x, train_y, candidate)
        values = score(valid_x.reshape(len(valid_x), -1), model)
        value = auc(values[valid_y == 1], values[valid_y == 0])
        key = (value, -candidate)
        if best is None or key > best[0]:
            best = (key, candidate)
    return float(best[1])


def score(features: np.ndarray, model: tuple[np.ndarray, ...]) -> np.ndarray:
    location, scale, weight, intercept = model
    flat = features.reshape(len(features), -1)
    return ((flat - location) / scale) @ weight + intercept


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n_positive = len(positive)
    return float(
        (
            np.sum(ranks[:n_positive])
            - n_positive * (n_positive + 1) / 2
        )
        / (n_positive * len(negative))
    )


def temporal_profile(
    times: np.ndarray,
    raw_scores: np.ndarray,
    statistic: np.ndarray,
    threshold: float,
    command_intervals: np.ndarray,
    idle_intervals: np.ndarray,
    step_seconds: float,
) -> dict[str, float]:
    idle_mask = sample_mask(times, idle_intervals.tolist())
    idle_median = float(np.median(raw_scores[idle_mask]))
    hit_density, ar1_values, fragmentation = [], [], []
    for start, stop in command_intervals:
        keep = (times >= start) & (times < stop)
        above = np.asarray(statistic[keep] >= threshold, dtype=bool)
        hit_density.append(float(np.mean(above)))
        transitions = np.diff(np.r_[False, above, False].astype(int))
        bursts = int(np.count_nonzero(transitions == 1))
        exceed_seconds = float(np.count_nonzero(above) * step_seconds)
        fragmentation.append(bursts / exceed_seconds if exceed_seconds > 0 else 0.0)
        values = np.asarray(raw_scores[keep], dtype=float)
        if len(values) > 2 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
            ar1_values.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]))
    command_mask = sample_mask(times, command_intervals.tolist())
    return {
        "hit_density_mean": float(np.mean(hit_density)),
        "fragmentation_bursts_per_exceed_second_mean": float(
            np.mean(fragmentation)
        ),
        "command_score_ar1_mean": float(np.mean(ar1_values)),
        "command_fraction_above_idle_median": float(
            np.mean(raw_scores[command_mask] > idle_median)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--target-subjects", type=int, nargs="*")
    parser.add_argument(
        "--arms",
        nargs="*",
        choices=[
            "target_50",
            "source_all_53",
            "source_n_matched_50_trials",
            "source_5",
            "source_10",
            "source_20",
            "source_40",
        ],
        help="Optional arm subset; default runs every arm.",
    )
    parser.add_argument("--target-fp", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--ridge-candidates",
        type=float,
        nargs="*",
        help="If provided, select ridge by inner validation independently per arm.",
    )
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/matched_lda_scaling_session1.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    paths = sorted(
        (args.cache_root / f"session{args.session}").glob(
            "sub-*_tier0_features.npz"
        )
    )
    caches = {int(load_cache(path)["subject"]): load_cache(path) for path in paths}
    normalized = {
        subject: rest_normalized_features(cache)
        for subject, cache in caches.items()
    }
    rows = []

    for target_subject, target in sorted(caches.items()):
        if args.target_subjects and target_subject not in args.target_subjects:
            continue
        source_subjects = [subject for subject in sorted(caches) if subject != target_subject]
        target_command_train = half(target["command_intervals"], "first")
        target_idle_train = half(target["iti_intervals"], "first")
        target_command_test = half(target["command_intervals"], "second")
        target_idle_test = half(target["iti_intervals"], "second")
        arm_specs = [("target_50", [target_subject], "target", 0)]
        arm_specs.append(("source_all_53", source_subjects, "all", 0))
        for repeat in range(args.repeats):
            arm_specs.append(("source_n_matched_50_trials", source_subjects, "n50", repeat))
            shuffled = rng.permutation(source_subjects)
            for count in (5, 10, 20, 40):
                arm_specs.append((f"source_{count}", list(shuffled[:count]), "all", repeat))

        for arm, training_subjects, mode, repeat in arm_specs:
            if args.arms and arm not in args.arms:
                continue
            x_parts, y_parts = [], []
            training_intervals = []
            if mode == "target":
                selected = [(target_subject, target_command_train, target_idle_train)]
            elif mode == "n50":
                trial_pool = [
                    (subject, index)
                    for subject in training_subjects
                    for index in range(len(caches[subject]["iti_intervals"]))
                ]
                trial_pick = rng.choice(len(trial_pool), size=50, replace=False)
                selected = []
                picked_subjects = sorted(
                    {trial_pool[index][0] for index in trial_pick}
                )
                for subject in picked_subjects:
                    trial_indices = [
                        trial_pool[index][1]
                        for index in trial_pick
                        if trial_pool[index][0] == subject
                    ]
                    selected.append(
                        (
                            subject,
                            caches[subject]["command_intervals"][trial_indices],
                            caches[subject]["iti_intervals"][trial_indices],
                        )
                    )
            else:
                selected = [
                    (
                        subject,
                        caches[subject]["command_intervals"],
                        caches[subject]["iti_intervals"],
                    )
                    for subject in training_subjects
                ]

            for subject, command_intervals, idle_intervals in selected:
                cache = caches[subject]
                x, y = xy_from_intervals(
                    normalized[subject],
                    cache["continuous_times"],
                    command_intervals,
                    idle_intervals,
                )
                x_parts.append(x)
                y_parts.append(y)
                training_intervals.append(
                    (subject, command_intervals, idle_intervals)
                )
            selected_ridge = (
                tune_ridge(
                    selected,
                    caches,
                    normalized,
                    args.ridge_candidates,
                    rng,
                )
                if args.ridge_candidates
                else args.ridge
            )
            training_x = np.concatenate(x_parts)
            training_y = np.concatenate(y_parts)
            model = fit_lda(training_x, training_y, selected_ridge)

            target_times, target_scores = decision_subsample(
                target["continuous_times"],
                score(normalized[target_subject], model),
                args.decision_step,
            )
            target_sequence = ScoredSequence(
                times=target_times,
                scores=target_scores,
                idle_intervals=target_idle_test,
                command_intervals=target_command_test,
            )
            # Diagnostic score-quality comparison: every arm receives the same
            # held-out target FP axis. This intentionally removes operating-
            # point transfer from the comparison.
            points = tune_source_operating_points(
                [target_sequence],
                "ma_dwell",
                args.target_fp,
                args.decision_step,
                ma_seconds=0.5,
                dwell_seconds=0.3,
                refractory_seconds=1.0,
                grid_size=args.grid_size,
            )
            command_mask = sample_mask(
                target_times, target_command_test.tolist()
            )
            idle_mask = sample_mask(target_times, target_idle_test.tolist())
            test_auc = auc(target_scores[command_mask], target_scores[idle_mask])
            for target_fp, point in points.items():
                decision_result = apply_decision(
                    target_sequence,
                    "ma_dwell",
                    point.threshold,
                    args.decision_step,
                    ma_seconds=0.5,
                    dwell_seconds=0.3,
                    refractory_seconds=1.0,
                )
                alarms = decision_result.alarms
                observed = evaluate_alarms(
                    target_times, alarms, target_idle_test, target_command_test
                )
                null = np.empty(args.shifts)
                for shift_index, offset in enumerate(
                    rng.integers(1, len(alarms), size=args.shifts)
                ):
                    null[shift_index] = evaluate_alarms(
                        target_times,
                        np.roll(alarms, int(offset)),
                        target_idle_test,
                        target_command_test,
                    ).true_positive_rate
                profile = temporal_profile(
                    target_times,
                    target_scores,
                    decision_result.statistic,
                    point.threshold,
                    target_command_test,
                    target_idle_test,
                    args.decision_step,
                )
                rows.append(
                    {
                        "target_subject": target_subject,
                        "session": args.session,
                        "arm": arm,
                        "repeat": repeat,
                        "training_subjects": len(set(training_subjects)),
                        "selected_ridge": selected_ridge,
                        "training_command_to_idle_window_ratio": float(
                            np.sum(training_y == 1) / np.sum(training_y == 0)
                        ),
                        "target_fp_axis": target_fp,
                        "test_window_auc": test_auc,
                        "selected_threshold": point.threshold,
                        "test_fp_per_minute": observed.fp_per_minute,
                        "observed_tpr": observed.true_positive_rate,
                        "null_tpr_median": float(np.median(null)),
                        "excess_tpr": float(
                            observed.true_positive_rate - np.median(null)
                        ),
                        **profile,
                    }
                )

    summaries = []
    for arm, target_fp in sorted(
        {(row["arm"], row["target_fp_axis"]) for row in rows}
    ):
        group = [
            row
            for row in rows
            if row["arm"] == arm and row["target_fp_axis"] == target_fp
        ]
        # Collapse repeated source draws within subject before cohort summaries.
        subjects = sorted({row["target_subject"] for row in group})
        collapsed = []
        for subject in subjects:
            subject_rows = [
                row for row in group if row["target_subject"] == subject
            ]
            collapsed.append(
                {
                    key: float(np.median([row[key] for row in subject_rows]))
                    for key in (
                        "test_window_auc",
                        "test_fp_per_minute",
                        "observed_tpr",
                        "null_tpr_median",
                        "excess_tpr",
                    )
                }
            )
        summaries.append(
            {
                "arm": arm,
                "target_fp_axis": target_fp,
                "subjects": len(subjects),
                "test_window_auc_median": float(
                    np.median([row["test_window_auc"] for row in collapsed])
                ),
                "test_fp_per_minute_median": float(
                    np.median([row["test_fp_per_minute"] for row in collapsed])
                ),
                "observed_tpr_median": float(
                    np.median([row["observed_tpr"] for row in collapsed])
                ),
                "null_tpr_median": float(
                    np.median([row["null_tpr_median"] for row in collapsed])
                ),
                "excess_tpr_median": float(
                    np.median([row["excess_tpr"] for row in collapsed])
                ),
            }
        )
    payload = {"summary": summaries, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
