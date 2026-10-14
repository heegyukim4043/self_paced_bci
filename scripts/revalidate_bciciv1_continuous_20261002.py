"""Decision-path revalidation on native-continuous BCI IV-1 (n=4).

This deliberately preserves the historical within-subject score model, but
calibrates on the chronological calibration stream and uses a rearmed detector.
It is a simulation audit, not a self-initiated MI performance estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.config import ExperimentConfig
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from uncued_mi.event_evaluation import (
    circular_shift_onset_null, false_alarms_per_minute, interval_metrics,
    onset_metrics, predicted_intervals, serializable,
)
from uncued_mi.features import extract_covariances
from uncued_mi.riemannian import tangent_features
from evaluate_bciciv1_locked_selfpaced_metrics import DELTAS, ground_truth_intervals
from run_bciciv1_riemannian_sanity import (
    REAL_SUBJECTS, SFREQ, STEP_SECONDS, WINDOW_SECONDS, calibration_masks,
    downsample_true_labels, regularized_reference, support_labels, tune_threshold,
)


def normalized_ma(raw: np.ndarray, mean: float, variance: float) -> np.ndarray:
    # In all-score mode, the old detector's alarm decisions do not feed EWMA.
    z = ewma_normalize_and_decide(
        raw, initial_mean=mean, initial_variance=variance, threshold=0.0,
        step_seconds=STEP_SECONDS, half_life_seconds=30.0,
        update_mode="all", ma_seconds=0.5, dwell_seconds=0.3,
        refractory_seconds=1.0,
    ).statistic
    return guard_local_moving_average(z, np.ones(len(z), bool), ma_steps=5)


def rearmed(ma: np.ndarray, threshold: float) -> np.ndarray:
    return guard_rearmed_alarms(
        ma, threshold, np.ones(len(ma), bool), dwell_steps=3,
        refractory_steps=10,
    )


def chronological_threshold(ma: np.ndarray, idle: np.ndarray,
                            target_fp: float = 2.0) -> tuple[float, float]:
    """Tune on full timeline; count only alarms in labeled calibration idle."""
    if len(ma) != len(idle) or not np.any(idle):
        raise ValueError("invalid calibration idle mask")
    candidates = np.unique(np.quantile(ma[np.isfinite(ma) & idle],
                                       np.linspace(0.01, 0.999, 300)))
    exposure = float(np.sum(idle) * STEP_SECONDS)
    rows = []
    for threshold in candidates:
        alarms = rearmed(ma, float(threshold))
        rate = 60.0 * float(np.sum(alarms & idle)) / exposure
        rows.append((abs(rate - target_fp), -float(threshold), float(threshold), rate))
    _, _, threshold, rate = min(rows)
    return threshold, rate


def evaluate_subject(root: Path, subject: str, shifts: int) -> dict:
    calib = loadmat(root / "mat" / f"BCICIV_calib_ds1{subject}.mat",
                    struct_as_record=False, squeeze_me=True)
    evaluation = loadmat(root / "mat" / f"BCICIV_eval_ds1{subject}.mat",
                         struct_as_record=False, squeeze_me=True)
    label_1000 = loadmat(
        root / "labels" / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat",
        squeeze_me=True,
    )["true_y"]
    label_100 = downsample_true_labels(label_1000)
    channels = tuple(str(v) for v in np.atleast_1d(calib["nfo"].clab))
    config = ExperimentConfig(
        native_sfreq=SFREQ, target_sfreq=SFREQ,
        window_seconds=WINDOW_SECONDS, feature_step_seconds=STEP_SECONDS,
        transient_seconds=2.0,
    )
    calib_times, calib_cov = extract_covariances(
        calib["cnt"].astype(np.float64) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels,
    )
    active, idle_calib = calibration_masks(
        calib_times, np.atleast_1d(calib["mrk"].pos), SFREQ,
    )
    reference = regularized_reference(calib_cov[idle_calib])
    calib_features = tangent_features(calib_cov, reference)
    model = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="liblinear",
        random_state=20260813,
    ).fit(
        np.r_[calib_features[active], calib_features[idle_calib]],
        np.r_[np.ones(active.sum()), np.zeros(idle_calib.sum())],
    )
    calib_raw = model.decision_function(calib_features)
    anchor = calib_raw[idle_calib]
    mean, variance = float(np.mean(anchor)), float(np.var(anchor, ddof=1))
    calib_ma = normalized_ma(calib_raw, mean, variance)
    threshold, anchor_fp = chronological_threshold(calib_ma, idle_calib)
    stitched_z = (anchor - mean) / np.sqrt(variance)
    stitched_threshold, stitched_fp = tune_threshold(
        stitched_z, np.ones(len(stitched_z), bool),
    )

    times, cov = extract_covariances(
        evaluation["cnt"].astype(np.float64) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels,
    )
    strict = support_labels(label_100, np.rint(times * SFREQ).astype(int), int(SFREQ))
    raw = model.decision_function(tangent_features(cov, reference))
    ma = normalized_ma(raw, mean, variance)
    alarms = rearmed(ma, threshold)
    constant_alarm_counts = {
        str(level): int(np.sum(rearmed(
            guard_local_moving_average(np.full(len(times), level),
                                       np.ones(len(times), bool), ma_steps=5),
            threshold,
        ))) for level in (threshold - 1.0, threshold, threshold + 1.0)
    }
    assert not any(constant_alarm_counts.values())
    valid = np.isfinite(strict)
    idle = valid & (strict == 0)
    command = valid & (strict != 0)
    old_alarms_stitched = ewma_normalize_and_decide(
        raw, initial_mean=mean, initial_variance=variance,
        threshold=stitched_threshold, step_seconds=STEP_SECONDS,
        half_life_seconds=30.0, update_mode="all", ma_seconds=0.5,
        dwell_seconds=0.3, refractory_seconds=1.0,
    ).alarms
    old_alarms_chrono = ewma_normalize_and_decide(
        raw, initial_mean=mean, initial_variance=variance,
        threshold=threshold, step_seconds=STEP_SECONDS,
        half_life_seconds=30.0, update_mode="all", ma_seconds=0.5,
        dwell_seconds=0.3, refractory_seconds=1.0,
    ).alarms
    rearmed_stitched = rearmed(ma, stitched_threshold)
    ablation_fp = {
        "stitched_old": false_alarms_per_minute(old_alarms_stitched, idle, STEP_SECONDS),
        "stitched_rearmed": false_alarms_per_minute(rearmed_stitched, idle, STEP_SECONDS),
        "chronological_old": false_alarms_per_minute(old_alarms_chrono, idle, STEP_SECONDS),
        "chronological_rearmed": false_alarms_per_minute(alarms, idle, STEP_SECONDS),
    }
    gt = ground_truth_intervals(label_100)
    observed = {
        key: value.sensitivity for key, value in
        onset_metrics(times, alarms, gt[:, 0], DELTAS).items()
    }
    circular = circular_shift_onset_null(
        times, alarms, gt[:, 0], DELTAS, shifts=shifts,
        minimum_shift_seconds=20.0, seed=20261002 + ord(subject),
    )
    predicted = predicted_intervals(
        times, alarms, ma, threshold, allowed=np.ones(len(times), bool),
    )
    loose = interval_metrics(predicted, gt, 0.1, 0.1)
    strict_interval = interval_metrics(predicted, gt, 0.7, 0.7)
    return {
        "subject": subject, "threshold": threshold,
        "stitched_threshold": stitched_threshold,
        "stitched_calibration_fp_per_min": stitched_fp,
        "ablation_actual_fp_per_min": ablation_fp,
        "calibration_idle_seconds": float(idle_calib.sum() * STEP_SECONDS),
        "calibration_fp_per_min": anchor_fp,
        "actual_fp_per_min": false_alarms_per_minute(alarms, idle, STEP_SECONDS),
        "valid_idle_seconds": float(idle.sum() * STEP_SECONDS),
        "window_auc": float(roc_auc_score(command[valid].astype(int), raw[valid])),
        "command_events": len(gt), "alarm_count": int(alarms.sum()),
        "constant_score_alarm_counts": constant_alarm_counts,
        "onset_sensitivity": observed, "circular_shift_null": circular,
        "excess_vs_circular": {k: observed[k] - circular[k] for k in observed},
        "interval_0.1": serializable(loose),
        "interval_0.7": serializable(strict_interval),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/bciciv1"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/bciciv1_continuous_revalidation_20261002.json"))
    args = parser.parse_args()
    rows = []
    for subject in REAL_SUBJECTS:
        rows.append(evaluate_subject(args.root, subject, args.shifts))
        print(f"done subject={subject}", flush=True)
    keys = ("window_auc", "calibration_fp_per_min", "actual_fp_per_min",
            "valid_idle_seconds")
    summary = {key + "_median": float(np.median([r[key] for r in rows])) for key in keys}
    summary["ablation_actual_fp_per_min_median"] = {
        arm: float(np.median([r["ablation_actual_fp_per_min"][arm] for r in rows]))
        for arm in ("stitched_old", "stitched_rearmed", "chronological_old",
                    "chronological_rearmed")
    }
    for metric in ("onset_sensitivity", "circular_shift_null", "excess_vs_circular"):
        summary[metric + "_median"] = {
            str(d): float(np.median([r[metric][str(d)] for r in rows])) for d in DELTAS
        }
    for key, path in (("coverage_median", ("interval_0.1", "coverage")),
                      ("iou_median", ("interval_0.1", "temporal_iou")),
                      ("strict_f1_median", ("interval_0.7", "f1"))):
        summary[key] = float(np.median([r[path[0]][path[1]] for r in rows]))
    payload = {
        "protocol": {
            "scope": "native-continuous acoustic-cued BCI IV-1; descriptive n=4",
            "model": "historical within-subject LR, calibration idle used in LR and threshold",
            "threshold": "chronological calibration timeline, 2 FP/min target; no idle stitching",
            "decision": "EWMA H30, MA5, rearm after below-threshold, dwell3, refractory10; no seam guard",
            "null": f"{args.shifts} circular alarm shifts, minimum 20 s",
            "limitation": "in-sample calibration, cue-contamination unresolved; no self-initiated onset claim",
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
