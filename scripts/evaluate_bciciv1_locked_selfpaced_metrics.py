"""Locked three-layer self-paced evaluation on continuous BCI Competition IV-1.

Unlike the Lee2019 pseudo-stream, the evaluation recording is continuous and
has no injected seams. Ground-truth event boundaries come from the original
100 Hz label stream; strict full-window labels are used only for window AUC and
idle false-alarm accounting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.config import ExperimentConfig
from uncued_mi.features import extract_covariances
from uncued_mi.riemannian import tangent_features
from uncued_mi.event_evaluation import (
    circular_shift_onset_null, false_alarms_per_minute, interval_metrics,
    onset_metrics, predicted_intervals, serializable,
)
from evaluate_lee2019_locked_selfpaced_metrics import ma_trace
from run_bciciv1_riemannian_sanity import (
    REAL_SUBJECTS, SFREQ, STEP_SECONDS, WINDOW_SECONDS, calibration_masks,
    contiguous_events, downsample_true_labels, fp_rate, regularized_reference,
    support_labels, tune_threshold,
)

DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)


def ground_truth_intervals(labels: np.ndarray) -> np.ndarray:
    """Return command intervals on the original continuous sample clock."""
    command = np.isfinite(labels) & (labels != 0)
    events = contiguous_events(command)
    return np.asarray(
        [[event[0] / SFREQ, (event[-1] + 1) / SFREQ] for event in events],
        dtype=float,
    ).reshape(-1, 2)


def score_subject(root: Path, subject: str) -> dict:
    calib = loadmat(root / "mat" / f"BCICIV_calib_ds1{subject}.mat",
                    struct_as_record=False, squeeze_me=True)
    evaluation = loadmat(root / "mat" / f"BCICIV_eval_ds1{subject}.mat",
                         struct_as_record=False, squeeze_me=True)
    labels_1000 = loadmat(
        root / "labels" / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat",
        squeeze_me=True,
    )["true_y"]
    labels_100 = downsample_true_labels(labels_1000)
    channels = tuple(str(value) for value in np.atleast_1d(calib["nfo"].clab))
    config = ExperimentConfig(
        native_sfreq=SFREQ, target_sfreq=SFREQ,
        window_seconds=WINDOW_SECONDS, feature_step_seconds=STEP_SECONDS,
        transient_seconds=2.0,
    )
    calib_times, calib_cov = extract_covariances(
        calib["cnt"].astype(np.float64) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels,
    )
    active, idle = calibration_masks(
        calib_times, np.atleast_1d(calib["mrk"].pos), SFREQ
    )
    reference = regularized_reference(calib_cov[idle])
    calib_features = tangent_features(calib_cov, reference)
    model = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="liblinear",
        random_state=20260813,
    ).fit(
        np.r_[calib_features[active], calib_features[idle]],
        np.r_[np.ones(active.sum()), np.zeros(idle.sum())],
    )
    calibration_scores = model.decision_function(calib_features)
    anchor = calibration_scores[idle]
    anchor_mean = float(anchor.mean())
    anchor_variance = float(anchor.var(ddof=1))
    anchor_z = (anchor - anchor_mean) / np.sqrt(anchor_variance)
    threshold, anchor_fp = tune_threshold(anchor_z, np.ones(len(anchor_z), bool))

    eval_times, eval_cov = extract_covariances(
        evaluation["cnt"].astype(np.float64) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels,
    )
    eval_ends = np.rint(eval_times * SFREQ).astype(int)
    strict_labels = support_labels(labels_100, eval_ends, int(SFREQ))
    raw = model.decision_function(tangent_features(eval_cov, reference))
    adapted = ewma_normalize_and_decide(
        raw, initial_mean=anchor_mean, initial_variance=anchor_variance,
        threshold=threshold, step_seconds=STEP_SECONDS, half_life_seconds=30.0,
        update_mode="all", ma_seconds=0.5, dwell_seconds=0.3,
        refractory_seconds=1.0,
    )
    ma = ma_trace(adapted.statistic)
    return {
        "times": eval_times, "raw": raw, "strict_labels": strict_labels,
        "alarms": adapted.alarms, "ma": ma, "threshold": threshold,
        "anchor_fp": anchor_fp, "gt": ground_truth_intervals(labels_100),
    }


def evaluate_subject(root: Path, subject: str, shifts: int) -> dict:
    data = score_subject(root, subject)
    times, alarms = data["times"], data["alarms"]
    labels = data["strict_labels"]
    valid = np.isfinite(labels)
    idle = valid & (labels == 0)
    command = valid & (labels != 0)
    gt = data["gt"]
    onsets = gt[:, 0]
    observed_rows = onset_metrics(times, alarms, onsets, DELTAS)
    observed = {key: value.sensitivity for key, value in observed_rows.items()}
    circular = circular_shift_onset_null(
        times, alarms, onsets, DELTAS, shifts=shifts,
        minimum_shift_seconds=0.1, seed=20260813 + ord(subject),
    )
    actual_fp = false_alarms_per_minute(alarms, idle, STEP_SECONDS)
    naive = {
        str(delta): float(1 - np.exp(-actual_fp * delta / 60.0))
        for delta in DELTAS
    }

    predicted = predicted_intervals(
        times, alarms, data["ma"], data["threshold"],
        allowed=np.ones(len(times), bool),
    )
    loose = interval_metrics(predicted, gt, 0.1, 0.1)
    strict = interval_metrics(predicted, gt, 0.7, 0.7)
    return {
        "subject": subject,
        "window_auc": float(roc_auc_score(command[valid].astype(int), data["raw"][valid])),
        "actual_fp_per_min": actual_fp,
        "anchor_fp_per_min": data["anchor_fp"],
        "command_events": len(gt),
        "valid_idle_seconds": float(idle.sum() * STEP_SECONDS),
        "onset_sensitivity": observed,
        "circular_shift_null": circular,
        "naive_poisson_null": naive,
        "excess_vs_circular": {key: observed[key] - circular[key] for key in observed},
        "circular_minus_naive": {key: circular[key] - naive[key] for key in observed},
        "interval_0.1": serializable(loose),
        "interval_0.7": serializable(strict),
        "predicted_intervals": len(predicted),
    }


def median_summary(rows: list[dict]) -> dict:
    def median(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(value)
        return float(np.median(values))

    return {
        "n": len(rows),
        "window_auc_median": median(["window_auc"]),
        "actual_fp_per_min_median": median(["actual_fp_per_min"]),
        "onset_sensitivity_median": {str(d): median(["onset_sensitivity", str(d)]) for d in DELTAS},
        "circular_shift_null_median": {str(d): median(["circular_shift_null", str(d)]) for d in DELTAS},
        "naive_poisson_null_median": {str(d): median(["naive_poisson_null", str(d)]) for d in DELTAS},
        "excess_vs_circular_median": {str(d): median(["excess_vs_circular", str(d)]) for d in DELTAS},
        "circular_minus_naive_median": {str(d): median(["circular_minus_naive", str(d)]) for d in DELTAS},
        "coverage_median": median(["interval_0.1", "coverage"]),
        "temporal_iou_median": median(["interval_0.1", "temporal_iou"]),
        "strict_f1_median": median(["interval_0.7", "f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/bciciv1"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/processed/bciciv1_locked_selfpaced_metrics.json"))
    args = parser.parse_args()
    rows = []
    for subject in REAL_SUBJECTS:
        rows.append(evaluate_subject(args.root, subject, args.shifts))
        print(f"done subject={subject}", flush=True)
    payload = {
        "protocol": {
            "stream": "original continuous evaluation recording; no concatenation and no seam guard",
            "ground_truth": "onset and interval boundaries from original 100-Hz true_y transitions",
            "window_labels": "strict full-window support, used for AUC and idle FP accounting",
            "gate": "1-s Riemannian TS+LR; EWMA H=30; MA=0.5 s; dwell=0.3 s; refractory=1.0 s",
            "onset_deltas_seconds": list(DELTAS),
            "interval": "release=trigger-0.5 z held 0.3 s; gap merge 0.3 s",
            "null": f"{args.shifts} circular shifts plus naive Poisson comparison",
            "limitation": "n=4; acoustic-cued MI states, not fully self-initiated intention onset",
        },
        "summary": median_summary(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
