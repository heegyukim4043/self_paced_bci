"""Within-subject Stieger Riemannian smoke test with cursor-speed control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import moving_average_dwell
from uncued_mi.riemannian import tangent_features

STEP_SECONDS = 0.1


def regularized_reference(covariances: np.ndarray) -> np.ndarray:
    reference = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(reference) / len(reference) * 1e-6), 1e-12)
    return reference + np.eye(len(reference)) * ridge


def grouped_alarm_summary(
    trial_index: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[int, dict[str, float | bool | int]]:
    """Apply the decision layer independently inside every trial cell."""
    output: dict[int, dict[str, float | bool | int]] = {}
    for trial in np.unique(trial_index):
        mask = trial_index == trial
        decision = moving_average_dwell(
            scores[mask],
            threshold,
            STEP_SECONDS,
            ma_seconds=0.5,
            dwell_seconds=0.3,
            refractory_seconds=1.0,
        )
        onset = np.flatnonzero(decision.alarms)
        output[int(trial)] = {
            "alarm_count": int(decision.alarms.sum()),
            "has_alarm": bool(len(onset)),
            "first_alarm_window": int(onset[0]) if len(onset) else -1,
            "exposure_s": float(mask.sum() * STEP_SECONDS),
        }
    return output


def tune_threshold(
    trial_index: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    target_fp: float = 2.0,
) -> tuple[float, float]:
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.999, 240)))
    best: tuple[float, float, float] | None = None
    trial_label = {
        int(trial): int(labels[np.flatnonzero(trial_index == trial)[0]])
        for trial in np.unique(trial_index)
    }
    for threshold in grid:
        summary = grouped_alarm_summary(trial_index, scores, float(threshold))
        rest = [value for trial, value in summary.items() if trial_label[trial] == 0]
        exposure = sum(float(value["exposure_s"]) for value in rest)
        alarms = sum(int(value["alarm_count"]) for value in rest)
        fp = 60.0 * alarms / exposure
        candidate = (abs(fp - target_fp), -float(threshold), fp)
        if best is None or candidate < best:
            best = candidate
            selected = float(threshold)
    assert best is not None
    return selected, float(best[2])


def event_metrics(
    trial_index: np.ndarray,
    labels: np.ndarray,
    runs: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    shifts: int,
    seed: int,
) -> dict:
    summary = grouped_alarm_summary(trial_index, scores, threshold)
    trials = np.asarray(sorted(summary), dtype=int)
    first = np.asarray([np.flatnonzero(trial_index == trial)[0] for trial in trials])
    trial_labels = labels[first].astype(bool)
    trial_runs = runs[first]
    has_alarm = np.asarray([summary[int(trial)]["has_alarm"] for trial in trials])
    alarm_count = np.asarray([summary[int(trial)]["alarm_count"] for trial in trials])
    exposure = np.asarray([summary[int(trial)]["exposure_s"] for trial in trials])
    observed = float(has_alarm[trial_labels].mean())
    fp = float(60.0 * alarm_count[~trial_labels].sum() / exposure[~trial_labels].sum())

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(shifts):
        shifted = trial_labels.copy()
        for run in np.unique(trial_runs):
            indices = np.flatnonzero(trial_runs == run)
            if len(indices) > 1:
                offset = int(rng.integers(1, len(indices)))
                shifted[indices] = np.roll(shifted[indices], offset)
        null.append(float(has_alarm[shifted].mean()))
    null_median = float(np.median(null))
    return {
        "threshold": float(threshold),
        "fp_per_minute": fp,
        "observed_active_trial_tpr": observed,
        "circular_trial_shift_null_tpr": null_median,
        "excess_tpr": observed - null_median,
        "active_trials": int(trial_labels.sum()),
        "intentional_rest_trials": int((~trial_labels).sum()),
        "alarm_trials": int(has_alarm.sum()),
    }


def speed_matched_indices(
    speed: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
    seed: int = 20260728,
) -> np.ndarray:
    """Equal-count active/rest sampling inside pooled speed quantile bins."""
    speed = np.asarray(speed, dtype=float)
    labels = np.asarray(labels, dtype=int)
    edges = np.unique(np.quantile(speed, np.linspace(0, 1, bins + 1)))
    assignments = np.clip(np.digitize(speed, edges[1:-1]), 0, len(edges) - 2)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for index in range(len(edges) - 1):
        active = np.flatnonzero((assignments == index) & (labels == 1))
        rest = np.flatnonzero((assignments == index) & (labels == 0))
        count = min(len(active), len(rest))
        if count:
            selected.extend(
                [rng.choice(active, count, replace=False), rng.choice(rest, count, replace=False)]
            )
    if not selected:
        raise ValueError("no speed-overlap bins contain both states")
    return np.sort(np.concatenate(selected))


def analyze(cache: Path, shifts: int) -> dict:
    with np.load(cache, allow_pickle=False) as data:
        cov = data["covariances"].astype(np.float64)
        trial = data["trial_index"].astype(int)
        label = data["is_active"].astype(int)
        run = data["run_number"].astype(int)
        speed = data["cursor_speed"].astype(float)

    split = 225
    train = trial < split
    test = trial >= split
    rest_train = train & (label == 0)
    reference = regularized_reference(cov[rest_train])
    features = tangent_features(cov, reference)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear"
    ).fit(features[train], label[train])
    raw_score = classifier.decision_function(features)

    # Some published trials have no finite cursor samples during a complete
    # feedback window.  Preserve those EEG windows and make the confound audit
    # explicit: impute from the training speed median and add a missingness
    # indicator.  No locked labels or locked speed statistics are used.
    finite_train_speed = speed[train & np.isfinite(speed)]
    if not len(finite_train_speed):
        raise ValueError("training split has no finite cursor speeds")
    speed_fill = float(np.median(finite_train_speed))
    speed_missing = ~np.isfinite(speed)
    speed_imputed = np.where(speed_missing, speed_fill, speed)
    speed_feature = np.c_[np.log1p(speed_imputed), speed_missing.astype(float)]
    residualizer = LinearRegression().fit(speed_feature[train], raw_score[train])
    residual_score = raw_score - residualizer.predict(speed_feature)
    cursor_classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="liblinear"
    ).fit(speed_feature[train], label[train])
    cursor_score = cursor_classifier.decision_function(speed_feature)

    matched = speed_matched_indices(speed_imputed[test], label[test])
    test_indices = np.flatnonzero(test)
    matched_global = test_indices[matched]

    arms = {}
    for name, score in (("raw", raw_score), ("speed_residualized", residual_score)):
        threshold, training_fp = tune_threshold(
            trial[train], label[train], score[train]
        )
        oracle_threshold, oracle_fp_tuning = tune_threshold(
            trial[test], label[test], score[test]
        )
        arms[name] = {
            "window_auc": float(roc_auc_score(label[test], score[test])),
            "window_pr_auc": float(average_precision_score(label[test], score[test])),
            "speed_matched_window_auc": float(
                roc_auc_score(label[matched_global], score[matched_global])
            ),
            "training_fp_per_minute": training_fp,
            "event": event_metrics(
                trial[test], label[test], run[test], score[test], threshold,
                shifts=shifts, seed=20260728,
            ),
            "oracle_common_fp_event_diagnostic": {
                "uses_locked_rest_labels_for_threshold_only": True,
                "tuning_fp_per_minute": oracle_fp_tuning,
                **event_metrics(
                    trial[test], label[test], run[test], score[test],
                    oracle_threshold, shifts=shifts, seed=20260728,
                ),
            },
        }

    return {
        "cache": str(cache.resolve()),
        "split": {
            "training_trial_indices": "0-224",
            "locked_trial_indices": "225-449",
            "training_windows": int(train.sum()),
            "locked_windows": int(test.sum()),
        },
        "reference": "mean training intentional-rest covariance",
        "cursor_only_window_auc": float(roc_auc_score(label[test], cursor_score[test])),
        "cursor_speed_missing_windows": int(speed_missing.sum()),
        "cursor_speed_training_median_imputation": speed_fill,
        "speed_matched_windows": int(len(matched_global)),
        "arms": arms,
        "interpretation": {
            "raw": "EEG gate with cursor motion left uncontrolled",
            "speed_residualized": (
                "EEG score after removing its training-set linear dependence on "
                "log cursor speed; target direction/layout and engagement remain."
            ),
            "event_null": (
                "Trial labels are circularly shifted within run because each "
                "published cell contains only one feedback state."
            ),
            "oracle_common_fp": (
                "Diagnostic only: the locked intentional-rest labels set the "
                "threshold so signal discrimination can be compared at FP=2. "
                "It is not a deployable calibration result."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/interim/stieger2021/S1_Session_1_covariances.npz"),
    )
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stieger2021_riemannian_smoke.json"),
    )
    args = parser.parse_args()
    report = analyze(args.cache, args.shifts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
