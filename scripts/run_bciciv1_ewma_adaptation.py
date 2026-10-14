"""Static versus causal EWMA operating-point control on real BCIC IV-1 data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression

from run_bciciv1_riemannian_sanity import (
    REAL_SUBJECTS, SFREQ, STEP_SECONDS, WINDOW_SECONDS, apply,
    bootstrap_median, calibration_masks, contiguous_events,
    downsample_true_labels, event_tpr, fp_rate, regularized_reference,
    support_labels, tune_threshold,
)

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.config import ExperimentConfig
from uncued_mi.features import extract_covariances
from uncued_mi.riemannian import tangent_features


def score_subject(root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calib = loadmat(root / "mat" / f"BCICIV_calib_ds1{subject}.mat",
                    struct_as_record=False, squeeze_me=True)
    evaluation = loadmat(root / "mat" / f"BCICIV_eval_ds1{subject}.mat",
                         struct_as_record=False, squeeze_me=True)
    true_y = loadmat(root / "labels" / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat",
                     squeeze_me=True)["true_y"]
    labels_100 = downsample_true_labels(true_y)
    channels = tuple(str(value) for value in np.atleast_1d(calib["nfo"].clab))
    config = ExperimentConfig(native_sfreq=SFREQ, target_sfreq=SFREQ,
                              window_seconds=WINDOW_SECONDS,
                              feature_step_seconds=STEP_SECONDS,
                              transient_seconds=2.0)
    calib_times, calib_cov = extract_covariances(
        calib["cnt"].astype(np.float64) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels)
    active, idle = calibration_masks(calib_times, np.atleast_1d(calib["mrk"].pos), SFREQ)
    reference = regularized_reference(calib_cov[idle])
    calib_features = tangent_features(calib_cov, reference)
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000,
                                    solver="liblinear").fit(
        np.r_[calib_features[active], calib_features[idle]],
        np.r_[np.ones(active.sum()), np.zeros(idle.sum())])
    calibration_scores = classifier.decision_function(calib_features)
    eval_times, eval_cov = extract_covariances(
        evaluation["cnt"].astype(np.float64) * 0.1, SFREQ, channels, config,
        config.sensorimotor_channels)
    eval_ends = np.rint(eval_times * SFREQ).astype(int)
    labels = support_labels(labels_100, eval_ends, int(SFREQ))
    eval_scores = classifier.decision_function(tangent_features(eval_cov, reference))
    return calibration_scores[idle], eval_scores, labels


def metrics(alarms: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    valid = np.isfinite(labels)
    idle = valid & (labels == 0)
    command = valid & (labels != 0)
    observed = event_tpr(alarms, command)
    rng = np.random.default_rng(seed)
    null = np.asarray([event_tpr(np.roll(alarms, int(offset)), command)
                       for offset in rng.integers(1, len(alarms), 500)])
    return {
        "fp_per_minute": fp_rate(alarms, idle),
        "fp_target_ratio": fp_rate(alarms, idle) / 2.0,
        "observed_event_tpr": observed,
        "null_event_tpr": float(np.median(null)),
        "excess_tpr": observed - float(np.median(null)),
        "command_events": len(contiguous_events(command)),
    }


def main() -> None:
    root = ROOT / "data" / "raw" / "bciciv1"
    rows = []
    for subject in REAL_SUBJECTS:
        anchor, scores, labels = score_subject(root, subject)
        mean, variance = float(np.mean(anchor)), float(np.var(anchor, ddof=1))
        anchor_z = (anchor - mean) / np.sqrt(variance)
        threshold, anchor_fp = tune_threshold(anchor_z, np.ones(len(anchor_z), bool))
        static_z = (scores - mean) / np.sqrt(variance)
        static = metrics(apply(static_z, threshold), labels, 20260805 + ord(subject))
        adaptive_result = ewma_normalize_and_decide(
            scores, initial_mean=mean, initial_variance=variance,
            threshold=threshold, step_seconds=STEP_SECONDS,
            half_life_seconds=30.0, update_mode="all",
            ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0)
        adaptive = metrics(adaptive_result.alarms, labels, 20261805 + ord(subject))
        rows.append({"subject": subject, "anchor_fp_per_minute": anchor_fp,
                     "static": static, "ewma_h30": adaptive})
        print(f"done subject={subject}", flush=True)
    summary = {}
    for arm in ("static", "ewma_h30"):
        summary[arm] = {}
        for key in ("fp_per_minute", "fp_target_ratio", "excess_tpr"):
            values = np.asarray([row[arm][key] for row in rows])
            summary[arm][key + "_median"] = float(np.median(values))
            summary[arm][key + "_ci"] = bootstrap_median(values, 20260805, 10_000)
    payload = {
        "subjects": list(REAL_SUBJECTS),
        "protocol": {
            "threshold": "calibration-idle target 2/min on anchor z score",
            "static": "fixed calibration mean/variance",
            "adaptive": "causal all-score EWMA, half-life 30 s; no evaluation labels",
            "limitation": "n=4 sanity check; acoustic-cued MI states, not fully self-initiated onset",
        },
        "summary": summary,
        "rows": rows,
    }
    output = ROOT / "data" / "processed" / "bciciv1_ewma_adaptation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
