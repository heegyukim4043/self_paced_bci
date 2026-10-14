"""Exploratory, disjoint-support BCI IV-1 calibration factorial.

Calibration cue periods 0:80 / 80:120 / 120:200 supply model fitting /
normalization anchor / threshold estimation. Threshold-stage idle is either
kept in chronological context or artificially concatenated. Static and causal
EWMA normalization are crossed with that factor on the same frozen model.
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

from uncued_mi.config import ExperimentConfig
from uncued_mi.decision import guard_local_moving_average
from uncued_mi.event_evaluation import (
    circular_shift_onset_null, false_alarms_per_minute, onset_metrics,
)
from uncued_mi.features import extract_covariances
from uncued_mi.riemannian import tangent_features
from evaluate_bciciv1_locked_selfpaced_metrics import ground_truth_intervals
from revalidate_bciciv1_continuous_20261002 import normalized_ma, rearmed, chronological_threshold
from run_bciciv1_riemannian_sanity import (
    REAL_SUBJECTS, SFREQ, STEP_SECONDS, WINDOW_SECONDS, calibration_masks,
    downsample_true_labels, regularized_reference, support_labels,
)

SPLIT = (80, 120, 200)
TARGET_FP = 2.0


def stage_masks(times: np.ndarray, cues: np.ndarray):
    """Partition causal 1-s supports; exclude cross-boundary windows."""
    if len(cues) != SPLIT[-1]:
        raise ValueError(f"expected {SPLIT[-1]} calibration cues")
    cue_times = np.asarray(cues, float) / SFREQ
    boundaries = [
        float((cue_times[SPLIT[0] - 1] + cue_times[SPLIT[0]]) / 2),
        float((cue_times[SPLIT[1] - 1] + cue_times[SPLIT[1]]) / 2),
    ]
    starts = times - WINDOW_SECONDS
    train = times <= boundaries[0]
    anchor = (starts >= boundaries[0]) & (times <= boundaries[1])
    threshold = starts >= boundaries[1]
    assert not np.any(train & anchor) and not np.any(anchor & threshold)
    return train, anchor, threshold, boundaries


def ma_from_raw(raw: np.ndarray, mean: float, variance: float,
                normalization: str) -> np.ndarray:
    if normalization == "ewma":
        return normalized_ma(raw, mean, variance)
    if normalization == "static":
        z = (raw - mean) / np.sqrt(max(variance, 1e-8))
        return guard_local_moving_average(z, np.ones(len(z), bool), ma_steps=5)
    raise ValueError(normalization)


def stitched_threshold(raw: np.ndarray, idle: np.ndarray, mean: float,
                       variance: float, normalization: str):
    """Apply the same normalizer to idle-only sequence; no full-time context."""
    if not np.any(idle):
        raise ValueError("empty idle")
    ma = ma_from_raw(raw[idle], mean, variance, normalization)
    return chronological_threshold(ma, np.ones(len(ma), bool), TARGET_FP)


def subject_row(root: Path, subject: str, shifts: int) -> dict:
    calib = loadmat(root / "mat" / f"BCICIV_calib_ds1{subject}.mat",
                    struct_as_record=False, squeeze_me=True)
    evaluation = loadmat(root / "mat" / f"BCICIV_eval_ds1{subject}.mat",
                         struct_as_record=False, squeeze_me=True)
    label_1000 = loadmat(root / "labels" / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat",
                         squeeze_me=True)["true_y"]
    labels = downsample_true_labels(label_1000)
    channels = tuple(str(v) for v in np.atleast_1d(calib["nfo"].clab))
    config = ExperimentConfig(
        native_sfreq=SFREQ, target_sfreq=SFREQ,
        window_seconds=WINDOW_SECONDS, feature_step_seconds=STEP_SECONDS,
        transient_seconds=2.0,
    )
    calib_times, calib_cov = extract_covariances(
        calib["cnt"].astype(float) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels,
    )
    cues = np.atleast_1d(calib["mrk"].pos)
    active, idle = calibration_masks(calib_times, cues, SFREQ)
    train, anchor, threshold_stage, boundaries = stage_masks(calib_times, cues)
    training_idle = idle & train
    training_active = active & train
    anchor_idle = idle & anchor
    threshold_idle = idle & threshold_stage
    if min(training_idle.sum(), training_active.sum(), anchor_idle.sum(), threshold_idle.sum()) < 10:
        raise ValueError("insufficient disjoint support")
    reference = regularized_reference(calib_cov[training_idle])
    features = tangent_features(calib_cov, reference)
    model = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="liblinear",
        random_state=20261002,
    ).fit(
        np.r_[features[training_active], features[training_idle]],
        np.r_[np.ones(training_active.sum()), np.zeros(training_idle.sum())],
    )
    calib_raw = model.decision_function(features)
    mean = float(np.mean(calib_raw[anchor_idle]))
    variance = float(np.var(calib_raw[anchor_idle], ddof=1))
    # Include all chronological score windows after the second boundary, but
    # threshold tuning counts only strict idle within this segment.
    stage_raw = calib_raw[threshold_stage]
    stage_idle = threshold_idle[threshold_stage]

    eval_times, eval_cov = extract_covariances(
        evaluation["cnt"].astype(float) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels,
    )
    strict = support_labels(labels, np.rint(eval_times * SFREQ).astype(int), int(SFREQ))
    valid = np.isfinite(strict)
    eval_idle = valid & (strict == 0)
    eval_active = valid & (strict != 0)
    raw = model.decision_function(tangent_features(eval_cov, reference))
    gt = ground_truth_intervals(labels)
    arms = {}
    for norm in ("static", "ewma"):
        chrono_ma = ma_from_raw(stage_raw, mean, variance, norm)
        eval_ma = ma_from_raw(raw, mean, variance, norm)
        for context in ("stitched", "chronological"):
            if context == "stitched":
                selected, calibration_fp = stitched_threshold(
                    stage_raw, stage_idle, mean, variance, norm,
                )
            else:
                selected, calibration_fp = chronological_threshold(
                    chrono_ma, stage_idle, TARGET_FP,
                )
            alarm = rearmed(eval_ma, selected)
            observed = onset_metrics(eval_times, alarm, gt[:, 0], (2.0,))["2.0"].sensitivity
            null = circular_shift_onset_null(
                eval_times, alarm, gt[:, 0], (2.0,), shifts=shifts,
                minimum_shift_seconds=20.0,
                seed=20261002 + ord(subject),
            )["2.0"]
            arms[f"{norm}_{context}"] = {
                "threshold": selected,
                "calibration_fp_per_min": calibration_fp,
                "evaluation_fp_per_min": false_alarms_per_minute(
                    alarm, eval_idle, STEP_SECONDS),
                "sen_2s": observed, "circular_null_2s": null,
                "matched_row_excess_2s": observed - null,
                "evaluation_alarm_count": int(alarm.sum()),
            }
    return {
        "subject": subject,
        "boundary_seconds": boundaries,
        "train_active_seconds": float(training_active.sum() * STEP_SECONDS),
        "train_idle_seconds": float(training_idle.sum() * STEP_SECONDS),
        "anchor_idle_seconds": float(anchor_idle.sum() * STEP_SECONDS),
        "threshold_idle_seconds": float(threshold_idle.sum() * STEP_SECONDS),
        "evaluation_idle_seconds": float(eval_idle.sum() * STEP_SECONDS),
        "window_auc": float(roc_auc_score(eval_active[valid].astype(int), raw[valid])),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/bciciv1"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/bciciv1_calibration_factorial_20261002.json"))
    args = parser.parse_args()
    rows = []
    for subject in REAL_SUBJECTS:
        rows.append(subject_row(args.root, subject, args.shifts))
        print(f"done subject={subject}", flush=True)
    names = tuple(rows[0]["arms"])
    summary = {
        name: {
            metric + "_median": float(np.median([r["arms"][name][metric] for r in rows]))
            for metric in ("calibration_fp_per_min", "evaluation_fp_per_min",
                           "sen_2s", "circular_null_2s", "matched_row_excess_2s")
        } for name in names
    }
    payload = {
        "protocol": {
            "scope": "exploratory n=4 native-continuous acoustic-cued evaluation",
            "calibration_split": "first 80 cues LR/reference; next 40 anchor; final 80 threshold",
            "support": "1-s window supports cannot cross split boundaries",
            "factors": "static vs causal all-score EWMA; stitched idle vs chronological full-stream threshold fitting",
            "decision": "MA5 + below-threshold rearm + dwell3 + refractory10",
            "target_fp_per_min": TARGET_FP,
            "null": f"{args.shifts} circular alarm shifts with 20 s minimum",
            "limitation": "80-cue threshold stage has short unique idle, n=4, acoustically instructed onset",
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
