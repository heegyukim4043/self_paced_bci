"""Read-only provenance/length audit before P1 calibration decomposition.

No EEG scores, thresholds, future alarms, or performance outcomes are read.
The unit in the output is a subject-session; fold/source cells are nested
technical observations and must never be treated as independent subjects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data/interim/lee2019_revalidation_chronological"
RIEM = ROOT / "data/interim/lee2019_revalidation_riemannian_scores"
CNN = ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache"
OUTPUT = ROOT / "data/processed/calibration_error_decomposition_20261008/preflight.json"
STEP = 0.1
SFREQ = 250.0


def valid_rest_seconds(events, source_ids, starts, times, allowed, fold, source):
    count = 0
    for index in range(fold * 10, (fold + 1) * 10):
        if int(source_ids[index]) != source:
            continue
        lo = events[index, 2] / SFREQ
        hi = events[index, 3] / SFREQ
        count += int(np.count_nonzero((starts >= lo) & (times <= hi) & allowed))
    return round(count * STEP, 1)


def main():
    rows = []
    anchor_steps = []
    for session in (1, 2):
        for subject in range(1, 55):
            stem = f"session{session}/sub-{subject:02d}.npz"
            with np.load(REPLAY / stem, allow_pickle=False) as replay, \
                 np.load(RIEM / stem, allow_pickle=False) as riem, \
                 np.load(CNN / stem, allow_pickle=False) as cnn:
                events = np.asarray(replay["events"], dtype=np.int64)
                source_ids = np.asarray(replay["rest_source_ids"], dtype=np.int64)
                starts = np.asarray(riem["starts"], dtype=float)
                times = np.asarray(riem["times"], dtype=float)
                allowed = np.asarray(riem["allowed"], dtype=bool)
                assert events.shape == (50, 8) and source_ids.shape == (50,)
                for key in ("events", "starts", "times", "allowed"):
                    assert np.array_equal(riem[key], cnn[key]), (stem, key)
                assert np.array_equal(events, riem["events"]), stem
                assert len(riem["anchor_raw_scores"]) == len(cnn["anchor_scores"])
                anchor_steps.append(len(riem["anchor_raw_scores"]))
                for fold in (2, 3, 4):
                    for source in (0, 1, 2):
                        rows.append({
                            "subject": subject,
                            "session": session,
                            "future_fold": fold,
                            "source_rest_block": source,
                            "eligible_score_seconds": valid_rest_seconds(
                                events, source_ids, starts, times, allowed,
                                fold, source,
                            ),
                        })
    values = np.asarray([x["eligible_score_seconds"] for x in rows])
    count_resolution = {
        str(length): {
            str(target): {
                "expected_alarm_count_at_target": round(length * target / 60, 3),
                "integer_alarm_rates_within_locked_band": [
                    round(count * 60 / length, 3)
                    for count in range(10)
                    if abs(count * 60 / length - target) <= (.5 if target == 2 else 1.)
                ],
            }
            for target in (2, 4)
        }
        for length in (14, 28, 36, 56)
    }
    result = {
        "status": "input feasibility only; no thresholds or performance calculated",
        "subject_sessions": len(anchor_steps),
        "nested_future_fold_source_cells": len(rows),
        "anchor_score_steps_unique": sorted(set(anchor_steps)),
        "eligible_score_seconds": {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
            "cells_below_36_seconds": int(np.sum(values < 36)),
            "cells_below_40_seconds": int(np.sum(values < 40)),
            "cells_below_56_seconds": int(np.sum(values < 56)),
        },
        "integer_count_resolution": count_resolution,
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
