"""Measure the observation cost of guard-local below-threshold re-arming.

This is a descriptive evaluator audit, not an MI-specific performance test.
The threshold is the already frozen target-pre-rest anchor threshold. The two
decision arms share the same EEG score, EWMA, guard-local MA and threshold;
only the below-threshold re-arm requirement differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma, ewma_z,
)


def unarmed_valid_samples(ma: np.ndarray, allowed: np.ndarray,
                          threshold: float) -> np.ndarray:
    """Valid MA samples before the first below-threshold sample in a segment."""
    out = np.zeros(len(ma), dtype=bool)
    armed = False
    for index, value in enumerate(ma):
        if not allowed[index] or not np.isfinite(value):
            armed = False
        elif not armed:
            out[index] = True
            if value < threshold:
                armed = True
    return out


def count_hits(alarms: np.ndarray, times: np.ndarray,
               onsets: np.ndarray) -> int:
    return sum(bool(np.any(alarms & (times >= onset) & (times <= onset + 2.)))
               for onset in onsets)


def evaluate_session(score_path: Path, threshold: float,
                     subject: int, session: int, target: str) -> dict:
    with np.load(score_path, allow_pickle=False) as data:
        raw = data["raw_scores"].astype(float)
        anchor = data["anchor_raw_scores"].astype(float)
        times = data["times"].astype(float)
        allowed = data["allowed"].astype(bool)
        events = data["events"].astype(np.int64)
    z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
    ma = guard_local_moving_average(z, allowed, ma_steps=5)
    old = alarms_from_ma(ma, threshold, allowed, dwell=3, refractory=10)
    new = guard_rearmed_alarms(ma, threshold, allowed,
                               dwell_steps=3, refractory_steps=10)
    future = times >= events[20, 2] / 250.
    cues = events[20:, 5] / 250.
    onsets = cues + .5
    task = np.zeros(len(times), dtype=bool)
    collar = np.zeros(len(times), dtype=bool)
    for cue, onset in zip(cues, onsets):
        task |= (times >= cue + .5) & (times <= cue + 4.)
        collar |= (times >= onset) & (times <= onset + 2.)
    idle = future & allowed & ~task
    valid = future & allowed & np.isfinite(ma)
    unarmed = unarmed_valid_samples(ma, allowed, threshold) & future
    high_unarmed = unarmed & (ma >= threshold)
    collar_valid = valid & collar
    blocked_collar_opportunities = 0
    first_valid_high = 0
    no_below_in_collar = 0
    old_hit_new_miss = 0
    for onset in onsets:
        region = valid & (times >= onset) & (times <= onset + 2.)
        indices = np.flatnonzero(region)
        if len(indices):
            first_valid_high += bool(ma[indices[0]] >= threshold)
            no_below_in_collar += bool(np.all(ma[indices] >= threshold))
            blocked_collar_opportunities += bool(np.any(high_unarmed & region))
        old_hit = bool(np.any(old & region))
        new_hit = bool(np.any(new & region))
        old_hit_new_miss += old_hit and not new_hit
    exposure = float(idle.sum() * .1)
    return {
        "subject": subject, "session": session, "target_fp_per_min": float(target),
        "threshold": threshold,
        "future_idle_seconds": exposure,
        "future_valid_ma_seconds": float(valid.sum() * .1),
        "unarmed_valid_ma_seconds": float((unarmed & valid).sum() * .1),
        "high_unarmed_idle_seconds": float((high_unarmed & idle).sum() * .1),
        "high_unarmed_collar_seconds": float((high_unarmed & collar_valid).sum() * .1),
        "first_valid_high_collar_count": first_valid_high,
        "never_below_in_collar_count": no_below_in_collar,
        "high_unarmed_collar_count": blocked_collar_opportunities,
        "old_hit_new_miss_count": old_hit_new_miss,
        "old_guard_local_hits2": count_hits(old & future, times, onsets),
        "rearmed_guard_local_hits2": count_hits(new & future, times, onsets),
        "old_guard_local_fp_count": int(np.sum(old & idle)),
        "rearmed_guard_local_fp_count": int(np.sum(new & idle)),
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for session in (1, 2):
        for target in (2., 4.):
            selected = [r for r in rows if r["session"] == session
                        and r["target_fp_per_min"] == target]
            sums = {key: float(sum(r[key] for r in selected)) for key in (
                "first_valid_high_collar_count", "never_below_in_collar_count",
                "high_unarmed_collar_count", "old_hit_new_miss_count",
                "old_guard_local_hits2", "rearmed_guard_local_hits2",
                "old_guard_local_fp_count", "rearmed_guard_local_fp_count",
                "high_unarmed_idle_seconds", "high_unarmed_collar_seconds",
            )}
            exposure = sum(r["future_idle_seconds"] for r in selected)
            valid = sum(r["future_valid_ma_seconds"] for r in selected)
            blind = sum(r["unarmed_valid_ma_seconds"] for r in selected)
            sums.update({
                "n_sessions": len(selected), "onsets": 30 * len(selected),
                "idle_exposure_seconds": exposure,
                "valid_ma_seconds": valid,
                "unarmed_valid_ma_seconds": blind,
                "unarmed_valid_ma_fraction": blind / valid if valid else None,
                "old_fp_per_min_pooled": sums["old_guard_local_fp_count"] * 60 / exposure,
                "rearmed_fp_per_min_pooled": sums["rearmed_guard_local_fp_count"] * 60 / exposure,
            })
            result[f"session{session}_fp{int(target)}"] = sums
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=54)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/"
                        "performance_revalidation_20261001/"
                        "a0_rearm_blind_time_stage2_20261002.json")
    args = parser.parse_args()
    if not 1 <= args.limit <= 54:
        raise ValueError("limit must be 1..54")
    source = ROOT / "data/processed/performance_revalidation_20261001/"
    observed = json.loads((source / "guard_rearm_riemannian_ablation_108.json")
                          .read_text(encoding="utf-8"))
    by_key = {(int(r["subject"]), int(r["session"])): r
              for r in observed["rows"]}
    rows = []
    for session in (1, 2):
        for subject in range(1, args.limit + 1):
            path = ROOT / "data/interim/lee2019_revalidation_riemannian_scores" \
                / f"session{session}" / f"sub-{subject:02d}.npz"
            for target in ("2.0", "4.0"):
                threshold = float(by_key[(subject, session)]["targets"][target]
                                  ["anchor"]["threshold"])
                row = evaluate_session(path, threshold, subject, session, target)
                expected = by_key[(subject, session)]["targets"][target]
                if (row["rearmed_guard_local_hits2"] != expected["future_onset_hits_2s"]
                        or row["rearmed_guard_local_fp_count"] != expected["future_fp_count"]):
                    raise AssertionError(f"published decision mismatch: {subject}/{session}/{target}")
                rows.append(row)
    payload = {
        "status": "descriptive decision-cost audit; no MI-specific null validation",
        "rule_comparison": "same score, EWMA, guard-local MA, threshold, dwell3 and refractory10; only re-arm differs",
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
