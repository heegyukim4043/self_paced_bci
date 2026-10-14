"""Independently replay saved observed matched-FP thresholds with scalar code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z


def main():
    folders = ROOT / "data/processed/performance_revalidation_20261001"
    names = ("a1_rest_sham_matched_fp_riemannian_full108",
             "a1_rest_sham_matched_fp4_riemannian_full108")
    saved = []
    for name in names:
        with np.load(folders / (name + ".npz"), allow_pickle=False) as data:
            saved.append({key: data[key].copy() for key in data.files})
    tested = 0
    for session in (1, 2):
        for subject in range(1, 55):
            path = ROOT / "data/interim/lee2019_revalidation_riemannian_scores" / \
                   f"session{session}/sub-{subject:02d}.npz"
            with np.load(path, allow_pickle=False) as cache:
                raw = cache["raw_scores"].astype(float)
                anchor = cache["anchor_raw_scores"].astype(float)
                times = cache["times"].astype(float)
                allowed = cache["allowed"].astype(bool)
                events = cache["events"].astype(np.int64)
            ma = guard_local_moving_average(
                ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed),
                allowed, ma_steps=5,
            )
            future = times >= events[20, 2] / 250.
            onsets = events[20:, 5] / 250. + .5
            task = np.zeros(len(raw), dtype=bool)
            for onset in onsets:
                task |= (times >= onset) & (times <= onset + 3.5)
            idle = future & allowed & ~task
            for data in saved:
                for draw in (0, 500, 999):
                    slot = (session - 1, subject - 1, draw)
                    threshold = float(data["observed_threshold"][slot])
                    if not np.isfinite(threshold):
                        raise AssertionError("unmatched saved draw")
                    alarms = guard_rearmed_alarms(
                        ma, threshold, allowed, dwell_steps=3, refractory_steps=10
                    )
                    actual_fp = int(np.sum(alarms & idle))
                    actual_hits = sum(np.any(
                        alarms & (times >= onset) & (times <= onset + 2.)
                    ) for onset in onsets)
                    expected = (int(data["matched_fp_count"][slot]),
                                int(data["observed_hits2"][slot]))
                    if (actual_fp, actual_hits) != expected:
                        raise AssertionError(
                            f"scalar mismatch S{session} sub-{subject:02d} draw{draw}: "
                            f"{(actual_fp, actual_hits)} != {expected}")
                    tested += 1
    payload = {"status": "pass", "observed_scalar_replays": tested,
               "subjects": 54, "sessions": 2, "targets": [2, 4],
               "draw_indices": [0, 500, 999]}
    output = folders / "a1_matched_fp_saved_scalar_audit.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
