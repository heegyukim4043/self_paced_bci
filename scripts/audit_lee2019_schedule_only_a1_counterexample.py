"""Negative control: a cue/seam-timed score with no EEG or MI information.

This is a synthetic structural counterexample, not a learned EEG model. It
tests what the rest-only sham cannot rule out: post-guard schedule transients.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_rest_sham_a1 import evaluate_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "data/processed/performance_revalidation_20261001"
            / "a1_schedule_only_counterexample_full108.json"
        ),
        help="Destination JSON; set this to preserve an earlier audit result.",
    )
    args = parser.parse_args()
    root = ROOT / "data/interim/lee2019_revalidation_riemannian_scores"
    results = []
    for session in (1, 2):
        for subject in range(1, 55):
            with np.load(root / f"session{session}/sub-{subject:02d}.npz",
                         allow_pickle=False) as score:
                times = score["times"].astype(float)
                allowed = score["allowed"].astype(bool)
                events = score["events"].astype(np.int64)
            raw = np.full(len(times), -1., dtype=float)
            # Trial start is the simulator's seam, NOT observed EEG or MI data.
            # A pulse at 4.6--5.5 s since that seam allows a valid low
            # post-guard MA first, then a deterministic rising edge.
            for trial_start in events[:, 4] / 250.:
                raw[(times >= trial_start + 4.6) &
                    (times <= trial_start + 5.5)] = 1.
            cues = events[20:, 5] / 250.
            onsets = cues + .5
            future = times >= events[20, 2] / 250.
            task = np.zeros(len(times), dtype=bool)
            for cue in cues:
                task |= (times >= cue + .5) & (times <= cue + 4.)
            idle = future & allowed & ~task
            observed = evaluate_raw(raw, -1., .01, allowed, idle,
                                    times, onsets, .5)
            rest_only = evaluate_raw(np.full(len(times), -1.), -1., .01,
                                     allowed, idle, times, onsets, .5)
            results.append({"subject": subject, "session": session,
                            "schedule_only_hits2": int(observed[0]),
                            "schedule_only_fp_count": int(observed[1]),
                            "rest_only_hits2": int(rest_only[0]),
                            "rest_only_fp_count": int(rest_only[1])})
    summary = {
        "status": "synthetic schedule-only structural counterexample; no EEG used",
        "definition": "raw=-1 except +1 from trial seam+4.6 to +5.5s; "
                      "EWMA mean=-1,var=.01; MA5/rearm/dwell3/refractory10; threshold=.5",
        "rows": results,
        "by_session": {
            str(session): {
                "schedule_only_hit_total": sum(x["schedule_only_hits2"] for x in results
                                               if x["session"] == session),
                "schedule_only_fp_total": sum(x["schedule_only_fp_count"] for x in results
                                              if x["session"] == session),
                "rest_only_hit_total": sum(x["rest_only_hits2"] for x in results
                                            if x["session"] == session),
            } for session in (1, 2)
        },
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_session"], indent=2))


if __name__ == "__main__":
    main()
