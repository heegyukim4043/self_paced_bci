"""Geometric audit of cue-preserving sham splice in Lee2019 replay.

No EEG-model scores or outcome labels are used beyond event times and allowed
mask. This is a structural diagnostic, not an MI-specificity test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.decision import guard_local_moving_average


def overlaps(start: float, stop: float, fade_start: float, fade_stop: float) -> bool:
    """Half-open support intersection with a cross-fade interval."""
    return start < fade_stop - 1e-9 and stop > fade_start + 1e-9


def audit_event(times: np.ndarray, valid_ma: np.ndarray, cue: float) -> dict:
    onset = cue + .5
    fade_start, fade_stop = onset, onset + .2
    trial_indices = np.flatnonzero(valid_ma & (times >= cue) & (times < cue + 4.))
    if len(trial_indices) == 0:
        raise AssertionError("no post-guard MA support")
    first = int(trial_indices[0])
    # One below-threshold MA to arm, then three above-threshold MA samples.
    earliest = first + 3
    if earliest >= len(times) or not np.all(valid_ma[first:earliest + 1]):
        raise AssertionError("invalid earliest re-armed alarm support")
    possible = np.flatnonzero(valid_ma & (np.arange(len(times)) >= earliest) &
                              (times >= onset) & (times <= onset + 2.))
    if len(possible) == 0:
        raise AssertionError("no eligible SEN@2s alarm time")
    window_overlap = [overlaps(times[i] - 1., times[i], fade_start, fade_stop)
                      for i in possible]
    # MA5 at index i uses score windows i-4..i, each of 1 s causal support.
    ma_support_overlap = [
        overlaps(times[i - 4] - 1., times[i], fade_start, fade_stop)
        for i in possible
    ]
    if possible[0] < 4:
        raise AssertionError("insufficient MA support")
    return {
        "earliest_delay_seconds": float(times[earliest] - onset),
        "possible_alarm_times_1s": int(np.sum(times[possible] <= onset + 1.)),
        "possible_alarm_times_2s": int(len(possible)),
        "raw_window_splice_overlap_2s": int(sum(window_overlap)),
        "ma_input_splice_overlap_2s": int(sum(ma_support_overlap)),
        "earliest_raw_window_splice_overlap": bool(window_overlap[0]),
        "earliest_ma_input_splice_overlap": bool(ma_support_overlap[0]),
    }


def run() -> dict:
    root = ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache"
    paths = sorted(root.glob("session*/sub-*.npz"))
    if len(paths) != 108:
        raise AssertionError(f"expected 108 score caches, got {len(paths)}")
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            times = np.asarray(saved["times"], dtype=float)
            allowed = np.asarray(saved["allowed"], dtype=bool)
            events = np.asarray(saved["events"], dtype=np.int64)
        if len(events) != 50:
            raise AssertionError(f"expected 50 replay trials: {path}")
        valid_ma = np.isfinite(guard_local_moving_average(
            np.zeros(len(times), dtype=float), allowed, ma_steps=5,
        ))
        event_rows = [audit_event(times, valid_ma, cue=float(event[5]) / 250.)
                      for event in events[20:]]
        rows.append({"subject": int(path.stem.split("-")[1]),
                     "session": int(path.parent.name.replace("session", "")),
                     "events": event_rows})
    events = [event for row in rows for event in row["events"]]
    delays = np.asarray([event["earliest_delay_seconds"] for event in events])
    candidates = sum(event["possible_alarm_times_2s"] for event in events)
    result = {
        "scope": "cue-preserving cross-fade geometry in Lee2019 pseudo-uncued simulation",
        "n_subject_sessions": len(rows),
        "n_future_events": len(events),
        "crossfade": "cue+0.5 to cue+0.7 s; 1 s causal score window and MA5 (0.1 s step)",
        "optimistic_earliest_alarm_delay_min_median_max_seconds": [
            float(np.min(delays)), float(np.median(delays)), float(np.max(delays))],
        "events_with_possible_alarm_at_1s": int(sum(
            event["possible_alarm_times_1s"] > 0 for event in events)),
        "events_with_possible_alarm_at_2s": int(sum(
            event["possible_alarm_times_2s"] > 0 for event in events)),
        "events_where_earliest_raw_window_overlaps_splice": int(sum(
            event["earliest_raw_window_splice_overlap"] for event in events)),
        "events_where_earliest_ma_input_overlaps_splice": int(sum(
            event["earliest_ma_input_splice_overlap"] for event in events)),
        "candidate_alarm_times_2s_total": int(candidates),
        "candidate_alarm_times_with_raw_window_splice_overlap": int(sum(
            event["raw_window_splice_overlap_2s"] for event in events)),
        "candidate_alarm_times_with_ma_input_splice_overlap": int(sum(
            event["ma_input_splice_overlap_2s"] for event in events)),
        "limitation": "Window/MA support only. CNN-GRU hidden state can carry the splice effect beyond the 1 s/MA support; no clean EEG null or MI-specificity inference follows.",
        "rows": rows,
    }
    output = ROOT / "data/processed/performance_revalidation_20261001/" \
             "a1_cue_preserving_splice_geometry_20261002.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
