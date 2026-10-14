"""Freeze E4 full-trial calibration blocks using timestamps only.

This script never opens EEG features, scores or alarm outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from precheck_e4_original_continuous_20261009 import following_iti, in_intervals

STEP = 0.1
MIN_STEPS = 560
TASK_STOP_FROM_CUE = 4.0
ITI_START_FROM_CUE = 4.5
EARLY_ITI_SECONDS = 2.0


def trial_masks(times: np.ndarray, cue: float, fix: np.ndarray,
                iti: np.ndarray) -> dict[str, np.ndarray]:
    if len(iti) == 0:
        empty = np.zeros(len(times), dtype=bool)
        return {"fixation": in_intervals(times, fix.reshape(1, 2)),
                "iti_early": empty.copy(), "iti_late": empty.copy()}
    if len(iti) != 1:
        raise AssertionError("one official task ITI per trial at most")
    if not np.isclose(iti[0, 0], cue + ITI_START_FROM_CUE, atol=.005):
        raise AssertionError("unexpected ITI start")
    fixation = in_intervals(times, fix.reshape(1, 2))
    early_stop = min(iti[0, 1], iti[0, 0] + EARLY_ITI_SECONDS)
    early = in_intervals(times, np.array([[iti[0, 0], early_stop]]))
    late = in_intervals(times, np.array([[early_stop, iti[0, 1]]]))
    if np.any(fixation & early) or np.any(early & late):
        raise AssertionError("idle strata overlap")
    return {"fixation": fixation, "iti_early": early, "iti_late": late}


def choose_block(trials: dict[int, dict[str, np.ndarray]], first: int,
                 last: int, backwards: bool, late_only: bool) -> dict:
    ids = range(last, first - 1, -1) if backwards else range(first, last + 1)
    included: list[int] = []
    selected: np.ndarray | None = None
    for trial_id in ids:
        strata = trials[trial_id]
        unit = strata["fixation"] | strata["iti_late"]
        if not late_only:
            unit = unit | strata["iti_early"]
        selected = unit.copy() if selected is None else selected | unit
        included.append(trial_id)
        if selected.sum() >= MIN_STEPS:
            break
    if selected is None or selected.sum() < MIN_STEPS:
        raise AssertionError(f"insufficient task idle: {first}-{last}")
    ordered = sorted(included)
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise AssertionError("selected trial IDs not consecutive")
    components = {name: int(np.sum(selected & np.logical_or.reduce(
        [trials[i][name] for i in ordered]))) for name in
        ("fixation", "iti_early", "iti_late")}
    expected = components["fixation"] + components["iti_late"]
    if not late_only:
        expected += components["iti_early"]
    if expected != int(selected.sum()):
        raise AssertionError("selected task-idle components do not partition block")
    return {
        "trial_ids": ordered,
        "first_trial": ordered[0], "last_trial": ordered[-1],
        "n_trials": len(ordered),
        "endpoint_steps": int(selected.sum()),
        "endpoint_seconds": float(selected.sum() * STEP),
        "components_seconds": {k: float(v * STEP) for k, v in components.items()},
        "first_endpoint_s": None, "last_endpoint_s": None,
    }


def one(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        subject, session = int(data["subject"]), int(data["session"])
        times = np.asarray(data["continuous_times"], dtype=float)
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
        fix = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
        pre_times = np.asarray(data["pre_times"], dtype=float)
    if len(cues) != 100 or len(fix) != 100 or not np.all(np.diff(times) > 0):
        raise AssertionError(f"invalid native chronology: {path}")
    if np.sum((pre_times >= 4 - .001) & (pre_times < 60 - .001)) != MIN_STEPS:
        raise AssertionError("common 4.0-59.9 s anchor unavailable")
    trials = {
        i: trial_masks(times, float(cues[i]), fix[i], following_iti(iti, float(cues[i])))
        for i in list(range(10, 30)) + list(range(50, 100))
    }
    arms = {
        "T_far": choose_block(trials, 10, 29, False, False),
        "T_near": choose_block(trials, 50, 69, True, False),
        "T_far_late_only": choose_block(trials, 10, 29, False, True),
        "T_near_late_only": choose_block(trials, 50, 69, True, True),
    }
    for arm, record in arms.items():
        chosen = np.logical_or.reduce([
            trials[i]["fixation"] | trials[i]["iti_late"] |
            (trials[i]["iti_early"] if not arm.endswith("late_only") else False)
            for i in record["trial_ids"]
        ])
        indices = np.flatnonzero(chosen)
        record["first_endpoint_s"] = float(times[indices[0]])
        record["last_endpoint_s"] = float(times[indices[-1]])
        if record["last_endpoint_s"] >= fix[70, 0]:
            raise AssertionError("calibration block overlaps future")
    future = {k: int(sum(trials[i][k].sum() for i in range(70, 100)))
              for k in ("fixation", "iti_early", "iti_late")}
    if any(v == 0 for v in future.values()):
        raise AssertionError("future stratum empty")
    return {"subject": subject, "session": session, "arms": arms,
            "future_seconds": {k: round(v * STEP, 3) for k, v in future.items()},
            "future_total_seconds": round(sum(future.values()) * STEP, 3),
            "future_late_only_seconds": round((future["fixation"] + future["iti_late"]) * STEP, 3),
            "task_stop_from_cue_s": TASK_STOP_FROM_CUE,
            "nominal_mi_overlap_stops_from_cue_s": TASK_STOP_FROM_CUE + 2.,
            "conservative_early_iti_stops_from_cue_s": ITI_START_FROM_CUE + EARLY_ITI_SECONDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/e4_full_trial_blocks_preflight_20261009.json"))
    args = parser.parse_args()
    paths = sorted(args.cache_root.glob("session*/sub-*_tier0_features.npz"))
    if len(paths) != 108:
        raise AssertionError(f"expected 108 cases, found {len(paths)}")
    rows = [one(path) for path in paths]
    summary = {}
    for session in (1, 2):
        group = [row for row in rows if row["session"] == session]
        if len(group) != 54 or len({r["subject"] for r in group}) != 54:
            raise AssertionError("expected 54 subjects per session")
        summary[str(session)] = {
            arm: {"median_trials": float(np.median([r["arms"][arm]["n_trials"] for r in group])),
                  "trial_range": [int(min(r["arms"][arm]["first_trial"] for r in group)),
                                  int(max(r["arms"][arm]["last_trial"] for r in group))],
                  "median_seconds": float(np.median([r["arms"][arm]["endpoint_seconds"] for r in group])),
                  "min_seconds": float(min(r["arms"][arm]["endpoint_seconds"] for r in group)),
                  "median_fixation_seconds": float(np.median([r["arms"][arm]["components_seconds"]["fixation"] for r in group])),
                  "median_iti_early_seconds": float(np.median([r["arms"][arm]["components_seconds"]["iti_early"] for r in group])),
                  "median_iti_late_seconds": float(np.median([r["arms"][arm]["components_seconds"]["iti_late"] for r in group]))}
            for arm in ("T_far", "T_near", "T_far_late_only", "T_near_late_only")
        }
        summary[str(session)]["future_median_total_seconds"] = float(np.median([
            r["future_total_seconds"] for r in group]))
        summary[str(session)]["future_median_late_only_seconds"] = float(np.median([
            r["future_late_only_seconds"] for r in group]))
    result = {"status": "metadata-only preflight; no features/scores/alarms opened",
              "selection": "minimal full-trial block >=56 s, far forward from 10, near backward from 69",
              "early_iti": "first 2 s from cue+4.5 to cue+6.5; nominal 2-s MI support overlap ends cue+6.0",
              "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
