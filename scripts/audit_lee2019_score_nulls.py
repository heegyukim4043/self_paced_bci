"""Leakage/null and EWMA-score distribution audit on prospectively selected sessions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.riemannian import tangent_features
from uncued_mi.operating_point import tune_source_operating_points
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, reference, rolling_covariance,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z, ma_trace, alarms_from_ma

OFFSETS = np.linspace(-4, 4, 61)
SCREEN = 10
TARGET_FP = 4.0
DELTA = 2.0
N_PERM = 10
N_SHIFT = 20


def target_sensitivity(ma, source, allowed, evalmask, idle, times, onsets):
    exposure = idle.sum() * STEP
    finite = ma[np.isfinite(ma) & allowed & evalmask]
    low, high = float(np.min(finite) - 1), float(np.max(finite) + 1)
    best = None
    for _ in range(14):
        threshold = (low + high) / 2
        alarms = alarms_from_ma(ma, threshold, allowed) & evalmask
        fp = float(np.sum(alarms & idle) / (exposure / 60))
        if best is None or abs(fp - TARGET_FP) < abs(best[0] - TARGET_FP):
            best = (fp, alarms)
        if fp > TARGET_FP:
            low = threshold
        else:
            high = threshold
    fp, alarms = best
    sen = float(np.mean([np.any(alarms & (times >= onset) & (times <= onset + DELTA)) for onset in onsets]))
    return sen, fp


def fit_score(command, idle, anchor_features, stream_features, labels=None):
    x = np.r_[command, idle]
    y = np.r_[np.ones(len(command)), np.zeros(len(idle))] if labels is None else labels
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260812)
    model.fit(scaler.transform(x), y)
    anchor = model.decision_function(scaler.transform(anchor_features))
    stream = model.decision_function(scaler.transform(stream_features))
    return anchor, stream


def audit(subject, session, selected, rng):
    cov_path = Path(f"data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz")
    raw_path = Path(f"data/interim/lee2019_four_rest_raw/session{session}/sub-{subject:02d}.npz")
    stream_path = Path(f"data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz")
    with np.load(cov_path, allow_pickle=False) as data:
        ct, it = data["train_command_trial"], data["train_idle_iti_trial"]
        command_cov = data["train_command_covariance"][ct < 50]
        idle_cov = data["train_idle_iti_covariance"][it < 50]
        ref = reference(data["pre_train_covariance"])
    with np.load(raw_path, allow_pickle=False) as data:
        anchor_cov, _ = rolling_covariance(data["pre_train"])
    with np.load(stream_path, allow_pickle=False) as data:
        signal, events = data["signal"], data["events"]
    stream_cov, ends = rolling_covariance(signal)
    times = ends / SFREQ
    starts = (ends - int(WINDOW * SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), bool)
    for seam in seams:
        guard |= (starts < seam + 3) & (times > seam)
    allowed = ~guard
    evalmask = times >= events[SCREEN, 2] / SFREQ
    onsets = events[SCREEN:, 5] / SFREQ + .5
    stops = events[SCREEN:, 5] / SFREQ + 4.
    state = np.zeros(len(times), bool)
    for onset, stop in zip(onsets, stops):
        state |= (times >= onset) & (times <= stop)
    idle_mask = evalmask & allowed & ~state

    command_features = tangent_features(command_cov, ref)
    idle_features = tangent_features(idle_cov, ref)
    anchor_features = tangent_features(anchor_cov, ref)
    stream_features = tangent_features(stream_cov, ref)
    anchor, raw = fit_score(command_features, idle_features, anchor_features, stream_features)
    anchor_z = (anchor - anchor.mean()) / anchor.std(ddof=1)
    source = float(tune_source_operating_points(
        [calibration(anchor_z)], "ma_dwell", [2.], STEP,
        ma_seconds=.5, dwell_seconds=.2, refractory_seconds=1., grid_size=160,
    )[2.].threshold)
    z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
    observed, observed_fp = target_sensitivity(ma_trace(z), source, allowed, evalmask, idle_mask, times, onsets)
    reversed_score = np.nan

    shifts = []
    valid_shifts = np.arange(int(20 / STEP), len(z) - int(20 / STEP))
    if selected:
        reversed_score, _ = target_sensitivity(ma_trace(z[::-1]), source, allowed, evalmask, idle_mask, times, onsets)
        for shift in rng.choice(valid_shifts, size=N_SHIFT, replace=False):
            shifted = np.roll(z, int(shift))
            value, _ = target_sensitivity(ma_trace(shifted), source, allowed, evalmask, idle_mask, times, onsets)
            shifts.append(value)

    labels = np.r_[np.ones(len(command_cov)), np.zeros(len(idle_cov))]
    permuted = []
    for _ in range(N_PERM if selected else 0):
        shuffled = rng.permutation(labels)
        pa, pr = fit_score(command_features, idle_features, anchor_features, stream_features, shuffled)
        paz = (pa - pa.mean()) / max(pa.std(ddof=1), 1e-8)
        psource = float(tune_source_operating_points(
            [calibration(paz)], "ma_dwell", [2.], STEP,
            ma_seconds=.5, dwell_seconds=.2, refractory_seconds=1., grid_size=160,
        )[2.].threshold)
        pz = ewma_z(pr, float(pa.mean()), float(pa.var(ddof=1)), allowed)
        value, _ = target_sensitivity(ma_trace(pz), psource, allowed, evalmask, idle_mask, times, onsets)
        permuted.append(value)

    idle_z = z[idle_mask]
    return {
        "subject": subject, "session": session, "selected": selected,
        "observed_sensitivity": observed, "observed_fp_per_min": observed_fp,
        "time_reversed_sensitivity": reversed_score,
        "circular_shift_mean": float(np.nanmean(shifts)) if shifts else None,
        "training_label_permutation_mean": float(np.nanmean(permuted)) if permuted else None,
        "idle_z_mean": float(np.mean(idle_z)),
        "idle_z_std": float(np.std(idle_z, ddof=1)),
        "idle_z_skew": float(skew(idle_z)),
        "idle_z_excess_kurtosis": float(kurtosis(idle_z)),
        "idle_z_q01_q50_q99": np.quantile(idle_z, [.01, .5, .99]).tolist(),
    }


def summarize(rows):
    output = {}
    for name, group in (("selected", [r for r in rows if r["selected"]]), ("nonselected", [r for r in rows if not r["selected"]])):
        output[name] = {"n": len(group)}
        for key in (
            "observed_sensitivity", "time_reversed_sensitivity", "circular_shift_mean",
            "training_label_permutation_mean", "idle_z_mean", "idle_z_std",
            "idle_z_skew", "idle_z_excess_kurtosis",
        ):
            values = [np.nan if r[key] is None else r[key] for r in group]
            output[name][key + "_median"] = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else None
    return output


def main():
    rng = np.random.default_rng(20260812)
    rows = []
    for session in (1, 2):
        relative = json.loads(Path(f"data/processed/lee2019_prospective_screening_n10_session{session}_relative.json").read_text())
        cutoff = float(relative["cutoff"])
        screen = json.loads(Path(f"data/processed/lee2019_prospective_screening_n10_session{session}.json").read_text())
        flat = {row["subject"]: row["screen_flat_fraction"] for row in screen["rows"]}
        for subject in range(1, 55):
            is_selected = flat[subject] <= cutoff
            # Full null audits are run only on selected sessions; non-selected
            # sessions retain distribution diagnostics and are represented by
            # a deterministic evenly spaced 25-session reference subset.
            if not is_selected and subject % 4 != session % 4:
                continue
            row = audit(subject, session, is_selected, rng)
            rows.append(row)
            print(f"session{session} sub-{subject:02d} selected={is_selected}", flush=True)
    payload = {
        "protocol": {
            "operating_point": "actual FP=4/min; cue+0.5 onset; 2-s tolerance",
            "training_label_permutations": N_PERM,
            "circular_shifts": N_SHIFT,
            "note": "trial-order permutation is a no-op because every trial is MI; score-event circular shift is used instead",
        },
        "summary": summarize(rows), "rows": rows,
    }
    output = Path("data/processed/lee2019_score_null_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
