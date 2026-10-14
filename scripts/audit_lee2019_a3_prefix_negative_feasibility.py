"""Read-only feasibility audit for target-prefix negatives in A3.

No head is fitted and no future performance is consulted. Replayed rest is
excluded because its original support is also used in future idle evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data/interim/lee2019_revalidation_riemannian_scores"
REPLAY = ROOT / "data/interim/lee2019_revalidation_chronological"


def main() -> None:
    rows = []
    for session in (1, 2):
        for subject in range(1, 55):
            name = f"session{session}/sub-{subject:02d}.npz"
            with np.load(SCORES / name, allow_pickle=False) as score, \
                 np.load(REPLAY / name, allow_pickle=False) as replay:
                times = np.asarray(score["times"], float)
                starts = np.asarray(score["starts"], float)
                allowed = np.asarray(score["allowed"], bool)
                events = np.asarray(score["events"], np.int64)
                assert np.array_equal(events, replay["events"])
                # A negative window must lie entirely in the original trial's
                # pre-command support, have no seam/guard crossing, and end
                # before the defined cue+0.5-s task onset.
                negative = np.zeros(len(times), bool)
                unguarded = np.zeros(len(times), bool)
                for event in events[:20]:
                    trial_start = event[4] / 250.
                    onset = event[5] / 250. + .5
                    candidate = (starts >= trial_start) & (times < onset)
                    unguarded |= candidate
                    negative |= candidate & allowed
                rows.append({"subject": subject, "session": session,
                             "prefix_trial_count": 20,
                             "trial_local_candidate_windows_before_guard": int(unguarded.sum()),
                             "trial_local_negative_windows_after_guard": int(negative.sum())})
        print(f"A3 prefix support feasibility S{session} complete", flush=True)
    result = {
        "scope": "A3 feasibility only; no performance data used",
        "definition": "1-s original trial-local negative windows, ending before cue+0.5 s and valid under frozen guard",
        "cells": len(rows),
        "guard_valid_negative_cells": int(sum(
            row["trial_local_negative_windows_after_guard"] > 0 for row in rows)),
        "candidate_windows_before_guard_min_max": [int(min(
            row["trial_local_candidate_windows_before_guard"] for row in rows)),
            int(max(row["trial_local_candidate_windows_before_guard"] for row in rows))],
        "guard_valid_negative_windows_min_max": [int(min(
            row["trial_local_negative_windows_after_guard"] for row in rows)),
            int(max(row["trial_local_negative_windows_after_guard"] for row in rows))],
        "decision": "No A3 head fitting under the frozen guard-valid negative definition; a different prospective negative source must be specified before opening A3.",
        "rows": rows,
    }
    output = ROOT / "data/processed/performance_revalidation_20261001/" \
             "a3_prefix_negative_feasibility.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
