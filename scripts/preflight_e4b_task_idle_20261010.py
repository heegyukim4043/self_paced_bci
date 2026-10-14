"""E4b timestamp-only preflight. Never reads feature or score arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from precheck_e4_original_continuous_20261009 import following_iti
from run_e4_original_continuous_fp_20261009 import STEP, _trial_strata, _union


def geometry(cache_path: Path, e4_geometry: dict) -> dict:
    with np.load(cache_path, allow_pickle=False) as data:
        subject, session = int(data["subject"]), int(data["session"])
        times = np.asarray(data["continuous_times"], dtype=float)
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
        fixation = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
    assert (subject, session) == (e4_geometry["subject"], e4_geometry["session"])
    times = times[times >= 4. - .001]
    assert np.isclose(times[0], 4., atol=.001)
    masks = {}
    for name, ids in {
        "T56": e4_geometry["arms"]["T_near"]["trial_ids"],
        "T_all": list(range(10, 70)),
        "T_79": list(range(10, 80)),
        "T_89": list(range(10, 90)),
        "E_full": list(range(70, 100)),
        "E_updated": list(range(80, 100)),
        "E_70_79": list(range(70, 80)),
        "E_80_89": list(range(80, 90)),
        "E_90_99": list(range(90, 100)),
    }.items():
        masks[name] = _union(_trial_strata(times, cues, fixation, iti, ids), False)
    if np.any(masks["T_all"] & masks["E_full"]):
        raise AssertionError("T-all calibration overlaps future evaluation")
    if np.any(masks["T_79"] & masks["E_updated"]):
        raise AssertionError("first update calibration overlaps primary evaluation")
    if not np.array_equal(masks["E_full"], masks["E_70_79"] | masks["E_updated"]):
        raise AssertionError("future blocks do not partition full evaluation")
    if not np.array_equal(masks["E_updated"], masks["E_80_89"] | masks["E_90_99"]):
        raise AssertionError("updated blocks do not partition primary evaluation")
    if np.any(masks["T_all"] & ~masks["T_79"]) or np.any(masks["T_79"] & ~masks["T_89"]):
        raise AssertionError("cumulative calibration is not nested")
    if not np.isclose(masks["E_full"].sum() * STEP,
                      e4_geometry["future_total_seconds"], atol=.001):
        raise AssertionError("E4 full-future exposure changed")
    if not np.isclose(masks["T56"].sum() * STEP,
                      e4_geometry["arms"]["T_near"]["endpoint_seconds"], atol=.001):
        raise AssertionError("E4 T-near calibration changed")
    boundaries = []
    for trial in (79, 89):
        interval = following_iti(iti, float(cues[trial]))
        if len(interval) != 1:
            raise AssertionError(f"trial {trial} lacks a single following ITI")
        boundary_s = float(interval[0, 1])
        boundary_index = int(np.searchsorted(times, boundary_s, side="left"))
        if not (0 < boundary_index < len(times)):
            raise AssertionError("update outside recording")
        past = masks["T_79" if trial == 79 else "T_89"]
        future = masks["E_80_89" if trial == 79 else "E_90_99"]
        if not np.all(np.flatnonzero(past) < boundary_index):
            raise AssertionError("update uses scores from its future")
        if not np.all(np.flatnonzero(future) >= boundary_index):
            raise AssertionError("update occurs after evaluation block starts")
        boundaries.append({"after_trial": trial, "time_s": boundary_s,
                           "index": boundary_index})
    if not boundaries[0]["index"] < boundaries[1]["index"]:
        raise AssertionError("update order reversed")
    return {"subject": subject, "session": session,
            "seconds": {name: round(float(mask.sum() * STEP), 3)
                        for name, mask in masks.items()},
            "n_trials_T56": len(e4_geometry["arms"]["T_near"]["trial_ids"]),
            "T56_trial_ids": e4_geometry["arms"]["T_near"]["trial_ids"],
            "boundaries": boundaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    parser.add_argument("--e4-preflight", type=Path,
                        default=Path("data/processed/e4_full_trial_blocks_preflight_20261009.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    e4 = json.loads(args.e4_preflight.read_text(encoding="utf-8"))
    if len(e4["rows"]) != 108:
        raise AssertionError("E4 geometry is not full cohort")
    rows = []
    for old in e4["rows"]:
        path = args.cache_root / f"session{old['session']}" / f"sub-{old['subject']:02d}_tier0_features.npz"
        rows.append(geometry(path, old))
    summary = {}
    for session in (1, 2):
        group = [r for r in rows if r["session"] == session]
        if len(group) != 54 or len({r["subject"] for r in group}) != 54:
            raise AssertionError("not 54 independent subjects per session")
        summary[str(session)] = {
            name: {"median_seconds": float(np.median([r["seconds"][name] for r in group])),
                   "min_seconds": float(min(r["seconds"][name] for r in group)),
                   "max_seconds": float(max(r["seconds"][name] for r in group))}
            for name in group[0]["seconds"]}
    result = {"status": "metadata-only; no scores or alarms read",
              "sampling_step_s": STEP, "n_cases": len(rows),
              "primary_windows": {"T_all_minus_T56": "E_full 70-99",
                                  "T_cum_minus_T_all": "E_updated 80-99"},
              "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
