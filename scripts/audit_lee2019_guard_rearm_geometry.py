"""Optimistic earliest alarm after each future event under fixed guard rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sys import path as syspath

ROOT = Path(__file__).resolve().parents[1]
syspath.insert(0, str(ROOT / "src"))
from uncued_mi.decision import guard_local_moving_average


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path,
        default=ROOT / "data/interim/lee2019_revalidation_riemannian_scores",
    )
    args = parser.parse_args()
    paths = sorted(args.scores.glob("session*/sub-*.npz"))
    if len(paths) != 108:
        raise ValueError(f"expected 108 score caches, found {len(paths)}")
    delays = []
    for file in paths:
        with np.load(file, allow_pickle=False) as data:
            times = data["times"]
            allowed = data["allowed"]
            cues = data["events"][20:, 5] / 250.
        # The constant's value does not matter; this identifies valid MA
        # support only. First valid MA is assumed below threshold, followed
        # by three high MA values: an optimistic structural lower bound.
        valid = np.isfinite(guard_local_moving_average(
            np.ones(len(times)), allowed, ma_steps=5,
        ))
        for cue in cues:
            candidates = np.flatnonzero(valid & (times >= cue) & (times < cue + 4.))
            if not len(candidates) or candidates[0] + 3 >= len(times):
                raise AssertionError(f"no post-guard support: {file}")
            first = int(candidates[0])
            if not allowed[first:first + 4].all():
                raise AssertionError(f"unexpected intervening guard: {file}")
            delays.append(float(times[first + 3] - (cue + .5)))
    delays = np.asarray(delays)
    summary = {
        "n_future_events": int(len(delays)),
        "optimistic_earliest_alarm_delay_min_seconds": float(delays.min()),
        "optimistic_earliest_alarm_delay_median_seconds": float(np.median(delays)),
        "optimistic_earliest_alarm_delay_max_seconds": float(delays.max()),
        "onsets_with_possible_alarm_at_1s": int(np.sum(delays <= 1.)),
        "onsets_with_possible_alarm_at_2s": int(np.sum(delays <= 2.)),
        "note": "assumes immediate below then three high MA samples and no refractory",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
