"""Run an EEG-free autocorrelated-score control through the locked evaluator.

This is a structural falsification control, not an MI-specific or Type-I null.
Each subject/session gets a stationary Gaussian AR(1) score whose mean, scale
and lag-one coefficient use only its pre-train rest score. A separate synthetic
pre-rest anchor selects the threshold once, then independent synthetic future
streams are evaluated with the unchanged seam/guard/event schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.guard_rearm_calibration import tune_guard_rearmed_anchor
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.evaluate_lee2019_rest_sham_a1 import evaluate_raw


def anchor_ar1_parameters(anchor: np.ndarray) -> tuple[float, float, float]:
    anchor = np.asarray(anchor, dtype=float)
    center = anchor - anchor.mean()
    denominator = float(np.dot(center[:-1], center[:-1]))
    phi = float(np.dot(center[:-1], center[1:]) / denominator) if denominator else 0.
    return float(anchor.mean()), float(max(anchor.std(ddof=1), 1e-8)), float(
        np.clip(phi, -.99, .99))


def draw_stationary_ar1(length: int, mean: float, sd: float, phi: float,
                        rng: np.random.Generator) -> np.ndarray:
    innovations = rng.normal(0., sd * np.sqrt(1. - phi * phi), length)
    initial = float(rng.normal(0., sd))
    out, _ = lfilter([1.], [1., -phi], innovations, zi=[phi * initial])
    return mean + out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--limit", type=int, default=54)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--target", choices=("2", "4"), default="2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/"
                        "performance_revalidation_20261001/"
                        "a1_ar1_no_eeg_stage2_20261002")
    args = parser.parse_args()
    if args.draws < 1 or not 1 <= args.limit <= 54:
        raise ValueError("draws must be positive and limit must be 1..54")
    rng = np.random.default_rng(args.seed)
    hit = np.zeros((2, args.limit, args.draws), dtype=np.int16)
    fp = np.zeros_like(hit)
    calibration = []
    for session in (1, 2):
        for subject in range(1, args.limit + 1):
            path = ROOT / "data/interim/lee2019_revalidation_riemannian_scores" \
                / f"session{session}" / f"sub-{subject:02d}.npz"
            with np.load(path, allow_pickle=False) as data:
                anchor = data["anchor_raw_scores"].astype(float)
                times = data["times"].astype(float)
                allowed = data["allowed"].astype(bool)
                events = data["events"].astype(np.int64)
            mean, sd, phi = anchor_ar1_parameters(anchor)
            synthetic_anchor = draw_stationary_ar1(len(anchor), mean, sd, phi, rng)
            synthetic_anchor_z = ewma_z(
                synthetic_anchor, float(synthetic_anchor.mean()),
                float(synthetic_anchor.var(ddof=1)),
                np.ones(len(anchor), dtype=bool),
            )
            point = tune_guard_rearmed_anchor(
                synthetic_anchor_z, float(args.target), .1, ma_steps=5,
                dwell_steps=3, refractory_steps=10,
                admissible_half_width=.5 if args.target == "2" else 1.,
            )
            future = times >= events[20, 2] / 250.
            cues = events[20:, 5] / 250.
            task = np.zeros(len(times), dtype=bool)
            for cue in cues:
                task |= (times >= cue + .5) & (times <= cue + 4.)
            idle = future & allowed & ~task
            onsets = cues + .5
            for draw in range(args.draws):
                raw = draw_stationary_ar1(len(times), mean, sd, phi, rng)
                h, f = evaluate_raw(
                    raw, float(synthetic_anchor.mean()),
                    float(synthetic_anchor.var(ddof=1)), allowed, idle,
                    times, onsets, point.threshold,
                )
                hit[session - 1, subject - 1, draw] = h
                fp[session - 1, subject - 1, draw] = f
            calibration.append({
                "subject": subject, "session": session, "anchor_phi": phi,
                "anchor_seconds": point.anchor_seconds,
                "anchor_alarm_count": point.anchor_alarm_count,
                "anchor_within_target_band": point.within_target_band,
                "threshold": point.threshold,
                "future_idle_seconds": float(idle.sum() * .1),
            })
        print(f"completed session {session}", flush=True)
    summary = {}
    for session in (1, 2):
        section = [r for r in calibration if r["session"] == session]
        exposure = sum(r["future_idle_seconds"] for r in section)
        hit_draw = hit[session - 1].sum(axis=0) / (30 * args.limit)
        fp_draw = fp[session - 1].sum(axis=0) * 60. / exposure
        summary[f"session{session}"] = {
            "subjects": args.limit, "draws": args.draws,
            "anchor_admissible_n": sum(r["anchor_within_target_band"] for r in section),
            "anchor_phi_median": float(np.median([r["anchor_phi"] for r in section])),
            "hit2_fraction_median": float(np.median(hit_draw)),
            "hit2_fraction_draw_quantile_025_975": np.quantile(
                hit_draw, [.025, .975]).tolist(),
            "fp_per_min_pooled_median": float(np.median(fp_draw)),
            "fp_per_min_pooled_draw_quantile_025_975": np.quantile(
                fp_draw, [.025, .975]).tolist(),
            "idle_exposure_seconds": exposure,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output) + ".npz", hits2=hit, fp_count=fp)
    payload = {
        "status": "EEG-free AR(1) structural control, not a biological null or Type-I test",
        "anchor_estimation": "pre-train rest score only; one synthetic anchor per session, fixed across future draws",
        "target_fp_per_min": int(args.target), "seed": args.seed,
        "summary": summary, "calibration": calibration,
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
