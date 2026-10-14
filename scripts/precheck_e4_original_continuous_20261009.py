"""Geometry-only E4 audit; never opens scores, features, or outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STEP_S = 0.1
N_CAL = 560
N_PER_SUBTYPE = 280
WINDOW_S = 2.0
TOL = 0.005


def following_iti(iti: np.ndarray, cue: float) -> np.ndarray:
    matches = iti[np.isclose(iti[:, 0], cue + 4.5, rtol=0, atol=TOL)]
    if len(matches) > 1:
        raise AssertionError("multiple task ITIs for one trial")
    return matches.reshape(-1, 2)


def in_intervals(ends: np.ndarray, intervals: np.ndarray,
                 *, full_support: bool = False) -> np.ndarray:
    mask = np.zeros(len(ends), dtype=bool)
    for start, stop in np.asarray(intervals, dtype=float).reshape(-1, 2):
        valid = (ends >= start - 1e-3) & (ends < stop - 1e-3)
        if full_support:
            valid &= ends - WINDOW_S >= start - 1e-3
        mask |= valid
    return mask


def arm_intervals(cues: np.ndarray, fixation: np.ndarray, iti: np.ndarray,
                  first: int, last_exclusive: int) -> tuple[np.ndarray, np.ndarray]:
    fix = fixation[first:last_exclusive]
    idle = np.concatenate([
        following_iti(iti, float(cues[i]))
        for i in range(first, last_exclusive)
    ]).reshape(-1, 2)
    return fix, idle


def exposure(times: np.ndarray, fix: np.ndarray, iti: np.ndarray) -> dict:
    mf, mi = in_intervals(times, fix), in_intervals(times, iti)
    sf = in_intervals(times, fix, full_support=True)
    si = in_intervals(times, iti, full_support=True)
    if np.any(mf & mi) or np.any(sf & si):
        raise AssertionError("fixation and ITI overlap")
    chosen = np.flatnonzero(mf | mi)
    if len(chosen) < N_CAL:
        last_56 = None
        gap = None
    else:
        last_56 = float(times[chosen[N_CAL - 1]])
        gap = None  # Filled by caller for prospective arms.
    return {
        "fixation_intervals": len(fix), "iti_intervals": len(iti),
        "raw_interval_seconds": {
            "fixation": float(np.sum(fix[:, 1] - fix[:, 0])),
            "iti": float(np.sum(iti[:, 1] - iti[:, 0])),
        },
        "endpoint_decision_seconds": {
            "fixation": round(float(mf.sum() * STEP_S), 3),
            "iti": round(float(mi.sum() * STEP_S), 3),
            "total": round(float((mf | mi).sum() * STEP_S), 3),
        },
        "strict_support_decision_seconds": {
            "fixation": round(float(sf.sum() * STEP_S), 3),
            "iti": round(float(si.sum() * STEP_S), 3),
            "total": round(float((sf | si).sum() * STEP_S), 3),
        },
        "has_56s_endpoint": bool(len(chosen) >= N_CAL),
        "has_56s_strict_support": bool((sf | si).sum() >= N_CAL),
        "has_28s_each_endpoint": bool(mf.sum() >= N_PER_SUBTYPE and
                                       mi.sum() >= N_PER_SUBTYPE),
        "last_end_of_first_56s": last_56,
        "gap_to_future_first_fixation_s": gap,
    }


def one(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        subject, session = int(data["subject"]), int(data["session"])
        times = np.asarray(data["continuous_times"], dtype=float)
        pre_times = np.asarray(data["pre_times"], dtype=float)
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
        fixation = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
    if len(cues) != 100 or len(fixation) != 100 or not np.all(np.diff(times) > 0):
        raise AssertionError(f"invalid original recording metadata: {path}")
    if not np.allclose(fixation[:, 0], cues - 3, atol=.005, rtol=0):
        raise AssertionError("fixation timing differs from cue-3 s")
    arm_defs = {"T_far": (10, 30), "T_near": (50, 70),
                "evaluation": (70, 100), "F_early": (70, 85),
                "F_late": (85, 100)}
    arms = {}
    future_start = float(fixation[70, 0])
    for name, (first, last) in arm_defs.items():
        fix, idle = arm_intervals(cues, fixation, iti, first, last)
        result = exposure(times, fix, idle)
        if name in ("T_far", "T_near"):
            result["gap_to_future_first_fixation_s"] = (
                None if result["last_end_of_first_56s"] is None
                else future_start - result["last_end_of_first_56s"])
            if result["gap_to_future_first_fixation_s"] is not None and result[
                "gap_to_future_first_fixation_s"] <= 0:
                raise AssertionError("calibration endpoint reaches future evaluation")
        arms[name] = result
    legacy_mask = (pre_times < 58.0 + 1e-3) & (pre_times >= 2.0 - 1e-3)
    # Cached feature grids historically started at either 2 or 4 s. Use the
    # same valid 4-s start for every subject; 560 points end at 59.9 s.
    harmonized_mask = (pre_times >= 4.0 - 1e-3) & (pre_times < 60.0 - 1e-3)
    anchor_n = int(harmonized_mask.sum())
    return {
        "subject": subject, "session": session,
        "anchor": {
            "rest_feature_first_end_s": float(pre_times[0]),
            "rest_feature_last_end_s": float(pre_times[-1]),
            "n_score_ends_legacy_2_to_58_s": int(legacy_mask.sum()),
            "n_score_ends_harmonized_4_to_60_s": anchor_n,
            "has_56s_endpoint": bool(anchor_n >= N_CAL),
        },
        "arms": arms,
    }


def summary(rows: list[dict]) -> dict:
    out = {}
    for session in (1, 2):
        group = [r for r in rows if r["session"] == session]
        if len(group) != 54 or len({r["subject"] for r in group}) != 54:
            raise AssertionError("54 paired subjects per session required")
        per_arm = {}
        for arm in ("T_far", "T_near", "evaluation", "F_early", "F_late"):
            values = [r["arms"][arm] for r in group]
            per_arm[arm] = {
                "n_has_56s_endpoint": sum(r["has_56s_endpoint"] for r in values),
                "n_has_56s_strict": sum(r["has_56s_strict_support"] for r in values),
                "n_has_28s_each_endpoint": sum(r["has_28s_each_endpoint"] for r in values),
                "median_endpoint_seconds": float(np.median([
                    r["endpoint_decision_seconds"]["total"] for r in values])),
                "min_endpoint_seconds": float(np.min([
                    r["endpoint_decision_seconds"]["total"] for r in values])),
                "median_strict_seconds": float(np.median([
                    r["strict_support_decision_seconds"]["total"] for r in values])),
                "median_raw_interval_seconds": float(np.median([
                    sum(r["raw_interval_seconds"].values()) for r in values])),
                "median_iti_endpoint_seconds": float(np.median([
                    r["endpoint_decision_seconds"]["iti"] for r in values])),
                "median_fixation_endpoint_seconds": float(np.median([
                    r["endpoint_decision_seconds"]["fixation"] for r in values])),
                "min_gap_to_future_s": (
                    float(np.min([r["gap_to_future_first_fixation_s"] for r in values]))
                    if arm in ("T_far", "T_near") else None),
            }
        out[str(session)] = {
            "n_anchor_has_56s_endpoint": sum(r["anchor"]["has_56s_endpoint"] for r in group),
            "n_anchor_first_end_2s": sum(
                r["anchor"]["rest_feature_first_end_s"] == 2.0 for r in group),
            "n_anchor_first_end_4s": sum(
                r["anchor"]["rest_feature_first_end_s"] == 4.0 for r in group),
            "arms": per_arm,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/interim/lee2019_tier0"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/e4_original_continuous_precheck_20261009_v2.json"))
    args = parser.parse_args()
    paths = sorted(args.cache_root.glob("session*/sub-*_tier0_features.npz"))
    if len(paths) != 108:
        raise AssertionError(f"expected 108 caches, found {len(paths)}")
    rows = [one(path) for path in paths]
    result = {
        "scope": "E4 metadata-only precheck; no feature values, scores, thresholds or FP outcomes opened",
        "date": "2026-10-09", "n_cal_endpoint_samples": N_CAL,
        "n_per_subtype_for_28s": N_PER_SUBTYPE,
        "anchor_harmonization": "first 560 valid pre-rest score ends at >=4.0 s; excludes historical 2-4 s extra windows",
        "sampling_step_s": STEP_S,
        "endpoint_note": "0.1-s decision-time exposure is not independent 0.1-s EEG evidence",
        "summary": summary(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
