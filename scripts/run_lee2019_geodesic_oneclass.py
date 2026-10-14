"""Target-pre-rest-only affine-invariant geodesic one-class gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.riemannian import tangent_features
from scripts.evaluate_lee2019_loso_deployment import (
    evaluate_score, prepare_target, tune_anchor,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import alarms_from_ma, ewma_z, ma_trace
from scripts.run_lee2019_pseudo_uncued_riemannian import reference


def geodesic_score(covariances, idle_reference):
    # tangent_features preserves the affine-invariant Frobenius norm via the
    # sqrt(2) off-diagonal vectorization.
    return np.linalg.norm(tangent_features(covariances, idle_reference), axis=1)


def analyze(subject, session):
    anchor_cov, stream_cov, events, times, starts, allowed = prepare_target(subject, session)
    idle_reference = reference(anchor_cov)
    anchor = geodesic_score(anchor_cov, idle_reference)
    raw = geodesic_score(stream_cov, idle_reference)
    thresholds = tune_anchor(anchor)
    z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
    ma = ma_trace(z)
    flat, margin, auc, surface = evaluate_score(
        raw, ma, thresholds, allowed, events, times, starts
    )
    fixed_alarms = alarms_from_ma(ma, thresholds[4.0], allowed)
    onsets = events[10:, 5] / 250.0 + 0.5
    observed = float(np.mean([
        np.any(fixed_alarms & (times >= onset) & (times <= onset + 2.0))
        for onset in onsets
    ]))
    rng = np.random.default_rng(20260814 + 100 * session + subject)
    candidates = np.arange(200, len(fixed_alarms) - 200)
    null = []
    for shift in rng.choice(candidates, min(100, len(candidates)), replace=False):
        shifted = np.roll(fixed_alarms, int(shift))
        null.append(float(np.mean([
            np.any(shifted & (times >= onset) & (times <= onset + 2.0))
            for onset in onsets
        ])))
    null_median = float(np.median(null))
    return {
        "subject": subject, "session": session,
        "screen_flat_fraction": flat, "screen_margin_mean": margin,
        "window_auc": auc, "thresholds": {str(k): v for k, v in thresholds.items()},
        "fixed_fp4_circular_null_2s": null_median,
        "fixed_fp4_excess_2s": observed - null_median,
        "surface": surface,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="*", default=[43, 2])
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_geodesic_oneclass_smoke.json"))
    args = parser.parse_args()
    if len(args.sessions) == 1:
        pairs = [(args.sessions[0], subject) for subject in args.subjects]
    elif len(args.sessions) == len(args.subjects):
        pairs = list(zip(args.sessions, args.subjects))
    else:
        raise ValueError("provide one session for all subjects or one session per subject")
    rows = []
    for session, subject in pairs:
        rows.append(analyze(subject, session)); print(f"done session{session} sub-{subject:02d}")
    payload = {
        "protocol": {
            "labels": "none; target pre-rest only",
            "score": "affine-invariant geodesic distance to target pre-rest covariance mean",
            "decision": "same anchor threshold, EWMA H30, MA/dwell, guard, and locked three-layer evaluation as Arm A",
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
