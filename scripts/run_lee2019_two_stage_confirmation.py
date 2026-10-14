"""Two-stage deployable safety layer: gate candidate then direction confirmation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, contiguous_intervals, reference, rolling_covariance,
)

TARGETS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
HORIZON_SECONDS = 0.5
AGREEMENT = 0.7
CONFIDENCE = 0.6
GRACE_SECONDS = 5.0


def confirm_candidates(alarms, direction_probability, valid, times):
    horizon = int(round(HORIZON_SECONDS / STEP))
    executed = np.zeros(len(alarms), bool)
    predicted_class = np.full(len(alarms), -1, int)
    for index in np.flatnonzero(alarms):
        stop = index + horizon
        if stop >= len(alarms) or times[stop] < GRACE_SECONDS:
            continue
        support = np.arange(index, stop + 1)
        if not np.all(valid[support]):
            continue
        classes = np.argmax(direction_probability[support], axis=1)
        counts = np.bincount(classes, minlength=2)
        winner = int(np.argmax(counts))
        agreement = counts[winner] / len(support)
        confidence = float(np.mean(direction_probability[support, winner]))
        if agreement >= AGREEMENT and confidence >= CONFIDENCE:
            executed[stop] = True
            predicted_class[stop] = winner
    return executed, predicted_class


def analyze(subject, shifts=100):
    stream_path = Path(f"data/interim/lee2019_pseudo_uncued_raw/session1/sub-{subject:02d}.npz")
    raw_path = Path(f"data/interim/lee2019_four_rest_raw/session1/sub-{subject:02d}.npz")
    cov_path = Path(f"data/interim/lee2019_four_rest/session1/sub-{subject:02d}.npz")
    mil_path = Path(f"data/interim/lee2019_mil_raw/session1/sub-{subject:02d}_mil_raw.npz")
    with np.load(mil_path, allow_pickle=False) as data:
        trial_labels = data["labels"]
    with np.load(cov_path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        trials = data["train_command_trial"]
        iti_trials = data["train_idle_iti_trial"]
        fit_command = trials < 50
        fit_idle = iti_trials < 50
        cmd = tangent_features(data["train_command_covariance"][fit_command], ref)
        iti = tangent_features(data["train_idle_iti_covariance"][fit_idle], ref)
        gate_scaler = StandardScaler().fit(np.r_[cmd, iti])
        gate = LogisticRegression(class_weight="balanced", solver="liblinear",
                                  max_iter=2000, random_state=20260805).fit(
            gate_scaler.transform(np.r_[cmd, iti]),
            np.r_[np.ones(len(cmd)), np.zeros(len(iti))])
        direction_scaler = StandardScaler().fit(cmd)
        direction_y = trial_labels[trials[fit_command]]
        direction = LogisticRegression(class_weight="balanced", solver="liblinear",
                                       max_iter=2000, random_state=20260805).fit(
            direction_scaler.transform(cmd), direction_y)
    with np.load(raw_path, allow_pickle=False) as data:
        anchor_cov, _ = rolling_covariance(data["pre_train"])
    anchor_raw = gate.decision_function(gate_scaler.transform(tangent_features(anchor_cov, ref)))
    anchor_mean, anchor_var = float(anchor_raw.mean()), float(anchor_raw.var(ddof=1))
    anchor_z = (anchor_raw - anchor_mean) / np.sqrt(anchor_var)
    points = tune_source_operating_points([calibration(anchor_z)], "ma_dwell", TARGETS, STEP,
                                          ma_seconds=.5, dwell_seconds=.3,
                                          refractory_seconds=1., grid_size=160)
    with np.load(stream_path, allow_pickle=False) as data:
        signal, events = data["signal"], data["events"]
        event_labels = data["trial_labels"]
    covariance, ends = rolling_covariance(signal)
    features = tangent_features(covariance, ref)
    raw_scores = gate.decision_function(gate_scaler.transform(features))
    direction_probability = direction.predict_proba(direction_scaler.transform(features))
    direction_classes = direction.classes_
    times, starts = ends / SFREQ, (ends - int(WINDOW * SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), bool)
    for seam in seams:
        guard |= (starts < seam + 2.) & (times > seam - 2.)
    strict_mi = np.zeros(len(times), bool)
    for row in events:
        strict_mi |= (starts >= row[6] / SFREQ) & (times <= row[7] / SFREQ)
    idle = ~strict_mi & ~guard
    idle_intervals = contiguous_intervals(idle, times)
    command_intervals = np.asarray([(r[6] / SFREQ, r[7] / SFREQ) for r in events])
    # Direction-information audit independent of the confirmation decision.
    strict_window_correct, strict_window_total = 0, 0
    full_event_correct, full_event_total = 0, 0
    predicted_windows = direction_classes[np.argmax(direction_probability, axis=1)]
    for event, true_label in zip(events, event_labels):
        member = (starts >= event[6] / SFREQ) & (times <= event[7] / SFREQ)
        values = predicted_windows[member]
        strict_window_correct += int(np.sum(values == true_label))
        strict_window_total += len(values)
        if len(values):
            labels, counts = np.unique(values, return_counts=True)
            full_event_correct += int(labels[np.argmax(counts)] == true_label)
            full_event_total += 1
    rng = np.random.default_rng(40260805 + subject)
    output = {}
    for target in TARGETS:
        candidate = ewma_normalize_and_decide(
            raw_scores, initial_mean=anchor_mean, initial_variance=anchor_var,
            threshold=points[target].threshold, step_seconds=STEP,
            half_life_seconds=30., update_mode="all", update_mask=~guard,
            ma_seconds=.5, dwell_seconds=.3, refractory_seconds=1.)
        executed, predicted_index = confirm_candidates(
            candidate.alarms, direction_probability, ~guard, times)
        candidate_result = evaluate_alarms(times, candidate.alarms, idle_intervals, command_intervals)
        executed_result = evaluate_alarms(times, executed, idle_intervals, command_intervals)
        null = np.empty(shifts)
        for i, offset in enumerate(rng.integers(1, len(executed), shifts)):
            null[i] = evaluate_alarms(times, np.roll(executed, int(offset)),
                                      idle_intervals, command_intervals).true_positive_rate
        correct, classified = 0, 0
        candidate_correct, candidate_classified = 0, 0
        for event, true_label in zip(events, event_labels):
            candidate_inside = np.flatnonzero(candidate.alarms &
                                              (times >= event[6] / SFREQ) &
                                              (times <= event[7] / SFREQ))
            if len(candidate_inside):
                candidate_classified += 1
                index = candidate_inside[0]
                candidate_label = direction_classes[np.argmax(direction_probability[index])]
                candidate_correct += int(candidate_label == true_label)
            inside = np.flatnonzero(executed & (times >= event[6] / SFREQ) &
                                    (times <= event[7] / SFREQ))
            if len(inside):
                classified += 1
                predicted_label = direction_classes[predicted_index[inside[0]]]
                correct += int(predicted_label == true_label)
        output[str(target)] = {
            "candidate_fp_per_minute": candidate_result.fp_per_minute,
            "candidate_tpr": candidate_result.true_positive_rate,
            "executed_fp_per_minute": executed_result.fp_per_minute,
            "executed_tpr": executed_result.true_positive_rate,
            "executed_null_tpr_median": float(np.median(null)),
            "executed_excess_tpr": float(executed_result.true_positive_rate - np.median(null)),
            "executed_median_delay_seconds": executed_result.median_delay_seconds,
            "confirmation_acceptance_fraction": float(executed.sum() / candidate.alarms.sum())
            if candidate.alarms.sum() else 0.0,
            "direction_accuracy_on_detected_events": correct / classified if classified else None,
            "direction_classified_events": classified,
            "direction_accuracy_at_true_candidate_onset": (
                candidate_correct / candidate_classified if candidate_classified else None),
            "direction_candidate_events": candidate_classified,
            "direction_accuracy_all_strict_mi_windows": strict_window_correct / strict_window_total,
            "direction_accuracy_full_mi_event_majority": full_event_correct / full_event_total,
        }
    return {"subject": subject, "operating_points": output}


def boot(values, seed):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (10000, len(values)))]
    return np.quantile(np.median(samples, axis=1), [.025, .975]).tolist()


def main():
    rows = [analyze(subject) for subject in range(1, 55)]
    summary = {}
    for target in map(str, TARGETS):
        summary[target] = {key + "_median": float(np.nanmedian([
            row["operating_points"][target][key] for row in rows
            if row["operating_points"][target][key] is not None])) for key in (
                "candidate_fp_per_minute", "candidate_tpr", "executed_fp_per_minute",
                "executed_tpr", "executed_null_tpr_median", "executed_excess_tpr",
                "executed_median_delay_seconds", "confirmation_acceptance_fraction",
                "direction_accuracy_on_detected_events")}
        for key in ("direction_accuracy_at_true_candidate_onset",
                    "direction_accuracy_all_strict_mi_windows",
                    "direction_accuracy_full_mi_event_majority"):
            summary[target][key + "_median"] = float(np.nanmedian([
                row["operating_points"][target][key] for row in rows
                if row["operating_points"][target][key] is not None]))
    common, failures = [], 0
    for row in rows:
        pairs = sorted((row["operating_points"][str(t)]["executed_fp_per_minute"],
                        row["operating_points"][str(t)]["executed_excess_tpr"])
                       for t in TARGETS)
        fp, excess = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
        if fp[0] <= 2 <= fp[-1]: common.append(float(np.interp(2, fp, excess)))
        else: failures += 1
    summary["common_actual_fp2"] = {
        "bracket_failures": failures,
        "executed_excess_tpr_median": float(np.median(common)) if common else None,
        "executed_excess_tpr_ci": boot(common, 20260805) if common else None,
    }
    result = {"subjects": 54, "configuration": {
        "gate": "Riemannian+LR, all-score EWMA H30",
        "confirmation_horizon_seconds": HORIZON_SECONDS,
        "direction_agreement": AGREEMENT, "mean_confidence": CONFIDENCE,
        "session_start_grace_seconds": GRACE_SECONDS,
    }, "summary": summary, "rows": rows}
    output = Path("data/processed/lee2019_two_stage_confirmation.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
