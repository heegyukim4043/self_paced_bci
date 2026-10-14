"""Causal parallel-transport covariance adaptation smoke test on Lee2019 Arm A."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.riemannian import (
    riemannian_mean, tangent_features, transport_covariance,
)
from scripts.evaluate_lee2019_loso_deployment import (
    SUBJECTS, fit_model, load_covariances, prepare_target, source_reference,
    source_training, tune_anchor,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma, ewma_z, ma_trace,
)
from scripts.run_lee2019_pseudo_uncued_riemannian import reference

WINDOWS = (5.0, 10.0, 30.0)
STEP = 0.1
UPDATE_SECONDS = 1.0


def metrics(times, z, event_threshold, screening_threshold, allowed, events, seed):
    ma = ma_trace(z)
    alarms = alarms_from_ma(ma, event_threshold, allowed)
    screening_alarms = alarms_from_ma(ma, screening_threshold, allowed)
    cues = events[:, 5] / 250.0
    flat = []
    for cue in cues[:10]:
        valid = (times >= cue + 1.5) & (times < cue + 4.0) & allowed
        flat.append(not np.any(screening_alarms[valid]))
    onsets = cues[10:] + .5
    stops = cues[10:] + 4.0
    evalmask = times >= events[10, 2] / 250.0
    command = np.zeros(len(times), bool)
    for onset, stop in zip(onsets, stops):
        command |= (times >= onset) & (times <= stop)
    idle = evalmask & allowed & ~command
    observed = float(np.mean([
        np.any(alarms & (times >= onset) & (times <= onset + 2.0)) for onset in onsets
    ]))
    exposure = idle.sum() * STEP
    actual_fp = float(np.sum(alarms & idle) / (exposure / 60.0))
    rng = np.random.default_rng(seed)
    candidates = np.arange(int(20 / STEP), len(alarms) - int(20 / STEP))
    null = []
    for shift in rng.choice(candidates, min(100, len(candidates)), replace=False):
        shifted = np.roll(alarms, int(shift))
        null.append(float(np.mean([
            np.any(shifted & (times >= onset) & (times <= onset + 2.0))
            for onset in onsets
        ])))
    null_median = float(np.median(null))
    return {
        "screen_flat_fraction": float(np.mean(flat)),
        "actual_fp_per_min": actual_fp,
        "observed_sensitivity_2s": observed,
        "circular_shift_null_2s": null_median,
        "excess_sensitivity_2s": observed - null_median,
    }


def transported_scores(covariances, times, allowed, eligibility, anchor_reference,
                       scaler, model, window_seconds):
    """Score before updating; update reference every second from prior eligible windows."""
    recent = deque()
    current_reference = anchor_reference.copy()
    output = np.empty(len(covariances))
    update_stride = max(1, int(round(UPDATE_SECONDS / STEP)))
    for index, (covariance, time) in enumerate(zip(covariances, times)):
        aligned = transport_covariance(covariance, current_reference, anchor_reference)
        feature = tangent_features(aligned[None], anchor_reference)
        output[index] = model.decision_function(scaler.transform(feature))[0]
        if allowed[index] and eligibility[index]:
            recent.append((float(time), covariance))
        while recent and recent[0][0] < time - window_seconds:
            recent.popleft()
        if (index + 1) % update_stride == 0 and len(recent) >= 5:
            current_reference = riemannian_mean(
                np.asarray([item[1] for item in recent]), max_iter=8, tolerance=1e-5
            )
    return output


def analyze(subject, session, windows=WINDOWS, include_transport_only=True):
    domains = load_covariances(session)
    own_cache = {}
    for source in SUBJECTS:
        own_ref = reference(domains[source]["pre"])
        own_cache[source] = {
            "command": tangent_features(domains[source]["command"], own_ref),
            "idle": tangent_features(domains[source]["idle"], own_ref),
        }
    anchor_cov, stream_cov, events, times, starts, allowed = prepare_target(subject, session)
    anchor_reference = reference(domains[subject]["pre"])
    x_command, x_idle = source_training(
        domains, subject, "A_target_rest", source_reference(domains, subject), own_cache
    )
    scaler, model = fit_model(x_command, x_idle)
    anchor_scores = model.decision_function(
        scaler.transform(tangent_features(anchor_cov, anchor_reference))
    )
    raw = model.decision_function(
        scaler.transform(tangent_features(stream_cov, anchor_reference))
    )
    thresholds = tune_anchor(anchor_scores)
    event_threshold = thresholds[4.0]
    screening_threshold = thresholds[2.0]
    mean, variance = float(anchor_scores.mean()), float(anchor_scores.var(ddof=1))
    baseline_z = ewma_z(raw, mean, variance, allowed)
    baseline_ma = ma_trace(baseline_z)
    # Label-free causal update proxy fixed from baseline Arm A. Current samples
    # enter the buffer only after being scored, and active/high-score periods do not.
    eligibility = np.isfinite(baseline_ma) & (baseline_ma < screening_threshold)
    seed = 20260813 + 100 * session + subject
    arms = {"none_plus_ewma": metrics(times, baseline_z, event_threshold, screening_threshold, allowed, events, seed)}
    for window in windows:
        transported = transported_scores(
            stream_cov, times, allowed, eligibility, anchor_reference,
            scaler, model, window,
        )
        static_z = (transported - mean) / max(np.sqrt(variance), 1e-8)
        adaptive_z = ewma_z(transported, mean, variance, allowed)
        if include_transport_only:
            arms[f"transport_{int(window)}s"] = metrics(times, static_z, event_threshold, screening_threshold, allowed, events, seed)
        arms[f"transport_{int(window)}s_plus_ewma"] = metrics(times, adaptive_z, event_threshold, screening_threshold, allowed, events, seed)
    return {
        "subject": subject, "session": session,
        "event_threshold_fp4": event_threshold,
        "screening_threshold_fp2": screening_threshold,
        "causal_update": "score current covariance first; then admit baseline-inactive allowed covariance; update every 1 s",
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=43)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_parallel_transport_smoke.json"))
    args = parser.parse_args()
    payload = analyze(args.subject, args.session)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
