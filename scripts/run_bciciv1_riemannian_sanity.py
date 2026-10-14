"""Riemannian 1 s gate on the four real BCI Competition IV-1 subjects."""

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
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.decision import moving_average_dwell
from uncued_mi.features import extract_covariances
from uncued_mi.riemannian import tangent_features

REAL_SUBJECTS = ("a", "b", "f", "g")
SFREQ = 100.0
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.1


def downsample_true_labels(labels_1000: np.ndarray) -> np.ndarray:
    """Conservatively map 1000 Hz labels to the official 100 Hz EEG clock."""
    labels = np.asarray(labels_1000, dtype=float).reshape(-1)
    if len(labels) % 10:
        raise ValueError("1000 Hz label length is not divisible by 10")
    blocks = labels.reshape(-1, 10)
    output = np.full(len(blocks), np.nan)
    valid = ~np.isnan(blocks).any(axis=1)
    finite = blocks[valid]
    if np.any(np.ptp(finite, axis=1) != 0):
        raise ValueError("a finite label block contains a state transition")
    output[valid] = finite[:, 0]
    return output


def support_labels(labels: np.ndarray, end_samples: np.ndarray, window: int) -> np.ndarray:
    """Label a window only when its complete support has one finite state."""
    result = np.full(len(end_samples), np.nan)
    for index, end in enumerate(end_samples):
        values = labels[end - window : end]
        if len(values) == window and np.all(np.isfinite(values)):
            unique = np.unique(values)
            if len(unique) == 1:
                result[index] = unique[0]
    return result


def calibration_masks(
    feature_times: np.ndarray, cue_samples: np.ndarray, sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    active = np.zeros(len(feature_times), dtype=bool)
    idle = np.zeros(len(feature_times), dtype=bool)
    cues = np.asarray(cue_samples, dtype=float) / sfreq
    for index, cue in enumerate(cues):
        # Official scoring ignores the first second after a cue.
        active |= (
            (feature_times - WINDOW_SECONDS >= cue + 1.0 - 1e-9)
            & (feature_times <= cue + 4.0 + 1e-9)
        )
        if index + 1 < len(cues) and cues[index + 1] - cue < 10.0:
            # Cue ends at +4 s. Guard one second after MI and one second before
            # the next visual cue.
            idle |= (
                (feature_times - WINDOW_SECONDS >= cue + 5.0 - 1e-9)
                & (feature_times <= cues[index + 1] - 1.0 + 1e-9)
            )
    return active, idle


def regularized_reference(covariances: np.ndarray) -> np.ndarray:
    reference = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(reference) / len(reference) * 1e-6), 1e-12)
    return reference + np.eye(len(reference)) * ridge


def apply(scores: np.ndarray, threshold: float) -> np.ndarray:
    return moving_average_dwell(
        scores,
        threshold,
        STEP_SECONDS,
        ma_seconds=0.5,
        dwell_seconds=0.3,
        refractory_seconds=1.0,
    ).alarms


def fp_rate(alarms: np.ndarray, idle_mask: np.ndarray) -> float:
    seconds = float(idle_mask.sum() * STEP_SECONDS)
    return 60.0 * float(np.sum(alarms & idle_mask)) / seconds


def tune_threshold(scores: np.ndarray, idle_mask: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.999, 300)))
    rows = [(abs(fp_rate(apply(scores, value), idle_mask) - 2.0), -value, value)
            for value in grid]
    _, _, threshold = min(rows)
    return float(threshold), fp_rate(apply(scores, threshold), idle_mask)


def contiguous_events(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [np.arange(start, stop) for start, stop in zip(starts, stops)]


def event_tpr(alarms: np.ndarray, command_mask: np.ndarray) -> float:
    events = contiguous_events(command_mask)
    return float(np.mean([np.any(alarms[event]) for event in events]))


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    shifts: int,
    seed: int,
) -> dict:
    valid = np.isfinite(labels)
    idle = valid & (labels == 0)
    command = valid & (labels != 0)
    alarms = apply(scores, threshold)
    observed = event_tpr(alarms, command)
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, len(alarms), shifts)
    null = np.asarray([event_tpr(np.roll(alarms, int(offset)), command)
                       for offset in offsets])
    return {
        "fp_per_minute": fp_rate(alarms, idle),
        "observed_event_tpr": observed,
        "null_event_tpr": float(np.median(null)),
        "excess_tpr": observed - float(np.median(null)),
        "command_events": len(contiguous_events(command)),
        "valid_idle_seconds": float(idle.sum() * STEP_SECONDS),
        "alarms": int(alarms.sum()),
        "window_auc": float(roc_auc_score(command[valid].astype(int), scores[valid])),
    }


def analyze_subject(root: Path, subject: str, shifts: int) -> dict:
    calib = loadmat(
        root / "mat" / f"BCICIV_calib_ds1{subject}.mat",
        struct_as_record=False, squeeze_me=True,
    )
    evaluation = loadmat(
        root / "mat" / f"BCICIV_eval_ds1{subject}.mat",
        struct_as_record=False, squeeze_me=True,
    )
    true_y = loadmat(
        root / "labels" / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat",
        squeeze_me=True,
    )["true_y"]
    labels_100 = downsample_true_labels(true_y)
    channels = tuple(str(value) for value in np.atleast_1d(calib["nfo"].clab))
    config = ExperimentConfig(
        native_sfreq=SFREQ,
        target_sfreq=SFREQ,
        window_seconds=WINDOW_SECONDS,
        feature_step_seconds=STEP_SECONDS,
        transient_seconds=2.0,
    )
    calib_times, calib_cov = extract_covariances(
        calib["cnt"].astype(np.float64) * 0.1,
        SFREQ, channels, config, config.sensorimotor_channels,
    )
    active_train, idle_train = calibration_masks(
        calib_times, np.atleast_1d(calib["mrk"].pos), SFREQ
    )
    reference = regularized_reference(calib_cov[idle_train])
    calib_features = tangent_features(calib_cov, reference)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="liblinear"
    ).fit(
        np.r_[calib_features[active_train], calib_features[idle_train]],
        np.r_[np.ones(active_train.sum()), np.zeros(idle_train.sum())],
    )
    calib_scores = classifier.decision_function(calib_features)
    threshold, calibration_fp = tune_threshold(calib_scores, idle_train)

    eval_times, eval_cov = extract_covariances(
        evaluation["cnt"].astype(np.float64) * 0.1,
        SFREQ, channels, config, config.sensorimotor_channels,
    )
    eval_ends = np.rint(eval_times * SFREQ).astype(int)
    eval_labels = support_labels(labels_100, eval_ends, int(SFREQ))
    eval_scores = classifier.decision_function(tangent_features(eval_cov, reference))
    transferred = evaluate(
        eval_scores, eval_labels, threshold, shifts, 20260728 + ord(subject)
    )
    idle_eval = np.isfinite(eval_labels) & (eval_labels == 0)
    oracle_threshold, oracle_tuning_fp = tune_threshold(eval_scores, idle_eval)
    oracle = evaluate(
        eval_scores, eval_labels, oracle_threshold, shifts, 20260728 + ord(subject)
    )
    return {
        "subject": subject,
        "classes": [str(value) for value in np.atleast_1d(calib["nfo"].classes)],
        "calibration": {
            "active_windows": int(active_train.sum()),
            "idle_windows": int(idle_train.sum()),
            "threshold": threshold,
            "fp_per_minute": calibration_fp,
        },
        "transferred": transferred,
        "oracle_common_fp_diagnostic": {
            "uses_evaluation_idle_labels_for_threshold": True,
            "tuning_fp_per_minute": oracle_tuning_fp,
            **oracle,
        },
    }


def bootstrap_median(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/bciciv1"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/bciciv1_riemannian_sanity.json"),
    )
    args = parser.parse_args()
    rows = []
    for subject in REAL_SUBJECTS:
        rows.append(analyze_subject(args.root, subject, args.shifts))
        print(f"done subject={subject}", flush=True)
    summary = {}
    for arm in ("transferred", "oracle_common_fp_diagnostic"):
        excess = np.asarray([row[arm]["excess_tpr"] for row in rows])
        summary[arm] = {
            "fp_per_minute_median": float(
                np.median([row[arm]["fp_per_minute"] for row in rows])
            ),
            "window_auc_median": float(
                np.median([row[arm]["window_auc"] for row in rows])
            ),
            "excess_tpr_median": float(np.median(excess)),
            "excess_tpr_ci": bootstrap_median(excess, 20260728, args.bootstrap),
        }
    payload = {
        "subjects": list(REAL_SUBJECTS),
        "artificial_subjects_excluded": ["c", "d", "e"],
        "protocol": {
            "sampling_rate_hz": SFREQ,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "filter": "CAR->causal 8-30 Hz",
            "channels": list(ExperimentConfig().sensorimotor_channels),
            "evaluation_transition_guard": "official NaN labels plus full-window support",
            "null": f"{args.shifts} circular shifts",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
