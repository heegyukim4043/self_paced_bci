"""Audit original-recording prefix negatives without inspecting EEG outcomes.

The replay's seam guard is deliberately not applied to original continuous EEG.
This checks support geometry only; it does not authorize A3 performance claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/lee2019_tier0"
OUTPUT = ROOT / "data/processed/performance_revalidation_20261001/a3_original_prefix_support_audit.json"
STEP = 0.1
WINDOW = 1.0


def window_count(start: float, stop: float) -> int:
    if stop - start < WINDOW - 1e-8:
        return 0
    # Endpoints must lie on the run's 0.1-s grid. Counting an arbitrary
    # interval-width/step quotient overstates exact 1-s fixation support.
    first_end = int(np.ceil((start + WINDOW - 1e-8) / STEP))
    last_end = int(np.floor((stop + 1e-8) / STEP))
    return max(0, last_end - first_end + 1)


def window_covered_seconds(start: float, stop: float) -> float:
    """Length of raw EEG covered by the union of valid 1-s windows."""
    if window_count(start, stop) == 0:
        return 0.0
    first_end = int(np.ceil((start + WINDOW - 1e-8) / STEP))
    last_end = int(np.floor((stop + 1e-8) / STEP))
    return (last_end - first_end) * STEP + WINDOW


def summarize(values: list[int]) -> dict[str, int]:
    return {"min": min(values), "median": int(np.median(values)), "max": max(values),
            "cells_with_zero": sum(value == 0 for value in values)}


def main() -> None:
    rows = []
    for session in (1, 2):
        for subject in range(1, 55):
            source = SOURCE / f"session{session}/sub-{subject:02d}_tier0_features.npz"
            with np.load(source, allow_pickle=False) as data:
                cues = np.asarray(data["cue_samples"], float) / float(data["native_sfreq"])
                official_iti = np.asarray(data["iti_intervals"], float)
            assert len(cues) == 100
            assert np.all(np.diff(cues) > 0)
            future_start = cues[70] - 3.0
            by_k = {}
            for k in (5, 10, 20):
                fixation_windows = 0
                iti_windows = 0
                iti_intervals = 0
                eligible_seconds = 0.0
                covered_seconds = 0.0
                for trial in range(50, 50 + k):
                    # Exclude 0.5 s after fixation onset and the last 1 s
                    # before the cue. Keep fixation as its own subtype: it
                    # remains visually cued and may contain anticipation.
                    fix_start, fix_stop = cues[trial] - 2.5, cues[trial] - 1.0
                    assert fix_stop <= future_start
                    fixation_windows += window_count(fix_start, fix_stop)
                    # An ITI must be a task ITI, not a block break. Match the
                    # legacy raw-run interval [cue+4.5, next_fixation-0.5].
                    original_start = cues[trial] + 4.5
                    match = np.flatnonzero(np.isclose(official_iti[:, 0], original_start, rtol=0, atol=.005))
                    if len(match) != 1:
                        continue
                    # Explicit proposed washouts: 2 s after task end and
                    # 1 s before next fixation. No artificial replay guard.
                    iti_start = cues[trial] + 6.0
                    iti_stop = cues[trial + 1] - 4.0
                    assert iti_start >= cues[trial] + 4.0
                    assert iti_stop < future_start
                    count = window_count(iti_start, iti_stop)
                    iti_windows += count
                    iti_intervals += int(count > 0)
                    eligible_seconds += max(0.0, iti_stop - iti_start)
                    covered_seconds += window_covered_seconds(iti_start, iti_stop)
                by_k[str(k)] = {
                    "fixation_windows": fixation_windows,
                    "iti_windows": iti_windows,
                    "iti_intervals_with_window": iti_intervals,
                    "iti_eligible_seconds": round(eligible_seconds, 3),
                    "iti_window_covered_seconds": round(covered_seconds, 3),
                    "total_windows": fixation_windows + iti_windows,
                }
            rows.append({"subject": subject, "session": session, "by_k": by_k})
    result = {
        "scope": "108-cell original-recording support geometry; no EEG score/performance read",
        "source": "tier0 cue samples and official task-ITI intervals; original continuous train phase",
        "window_seconds": WINDOW,
        "step_seconds": STEP,
        "iti_definition": "task ITI only, cue_i+6.0 to cue_(i+1)-4.0; no replay seam guard",
        "fixation_definition": "cue_i-2.5 to cue_i-1.0; separate secondary subtype",
        "future_support_begins": "cue_70-3.0; all inspected windows stop earlier",
        "summary": {
            str(k): {
                key: summarize([row["by_k"][str(k)][key] for row in rows])
                for key in ("iti_windows", "iti_intervals_with_window", "fixation_windows", "total_windows")
            } for k in (5, 10, 20)
        },
        "time_definitions": {
            "iti_eligible_seconds": "candidate ITI interval length after washouts, including fragments too short for a complete 1-s window",
            "iti_window_covered_seconds": "union of original EEG sample time spanned by valid 1-s windows on the fixed 0.1-s grid",
        },
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
