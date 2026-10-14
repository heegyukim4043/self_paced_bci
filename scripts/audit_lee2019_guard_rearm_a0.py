"""Replay the constant-score A0 guard control on chronological Lee2019 masks.

This is a decision-layer structural test, not an EEG performance estimate.
The legacy detector remains unchanged for comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_locked_selfpaced_metrics import alarms_from_ma
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms

SFREQ = 250.0
STEP_SAMPLES = 25
WINDOW_SAMPLES = 250


def audit_one(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        events = data["events"].copy()
        sample_count = len(data["state"])
    ends = np.arange(WINDOW_SAMPLES, sample_count + 1, STEP_SAMPLES)
    times = ends / SFREQ
    starts = (ends - WINDOW_SAMPLES) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), dtype=bool)
    for seam in seams:
        guard |= (starts < seam + 3.0) & (times > seam)
    allowed = ~guard
    future = times >= events[20, 2] / SFREQ

    constant = np.ones(len(times))
    legacy = alarms_from_ma(constant, 0.5, allowed, dwell=3, refractory=10)
    new_ma = guard_local_moving_average(constant, allowed, ma_steps=5)
    rearmed = guard_rearmed_alarms(
        new_ma, 0.5, allowed, dwell_steps=3, refractory_steps=10
    )

    onsets = events[20:, 5] / SFREQ + 0.5
    task = np.zeros(len(times), dtype=bool)
    for event in events[20:]:
        task |= (times >= event[5] / SFREQ + 0.5) & (
            times <= event[5] / SFREQ + 4.0
        )
    idle = future & allowed & ~task
    hits = sum(
        np.any(legacy & (times >= onset) & (times <= onset + 2.0))
        for onset in onsets
    )
    row = {
        "session": int(path.parent.name.replace("session", "")),
        "subject": int(path.stem.replace("sub-", "")),
        "legacy_future_alarm_count": int(np.sum(legacy & future)),
        "legacy_future_onset_hits_2s": int(hits),
        "legacy_future_idle_alarms": int(np.sum(legacy & idle)),
        "new_future_alarm_count": int(np.sum(rearmed & future)),
        "new_future_idle_alarms": int(np.sum(rearmed & idle)),
        "future_event_count": int(len(onsets)),
        "future_idle_seconds": float(np.sum(idle) * 0.1),
    }
    if row["new_future_alarm_count"] != 0:
        raise AssertionError(f"constant score fired under re-arm rule: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "data/interim/lee2019_revalidation_chronological",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/processed/performance_revalidation_20261001/"
        "a0_guard_rearm_constant_control.json",
    )
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("session*/sub-*.npz"))
    if len(paths) != 108:
        raise ValueError(f"expected 108 chronological replays, found {len(paths)}")
    rows = [audit_one(path) for path in paths]
    summary = {
        "n": len(rows),
        "legacy_future_alarm_count_range": [
            min(row["legacy_future_alarm_count"] for row in rows),
            max(row["legacy_future_alarm_count"] for row in rows),
        ],
        "legacy_future_onset_hits_2s_range": [
            min(row["legacy_future_onset_hits_2s"] for row in rows),
            max(row["legacy_future_onset_hits_2s"] for row in rows),
        ],
        "legacy_future_idle_alarms_range": [
            min(row["legacy_future_idle_alarms"] for row in rows),
            max(row["legacy_future_idle_alarms"] for row in rows),
        ],
        "new_constant_alarm_count_total": sum(
            row["new_future_alarm_count"] for row in rows
        ),
    }
    payload = {
        "control": "constant score=1, threshold=0.5; no EEG information",
        "decision": "guard-local 5-sample MA, below-threshold re-arm, dwell=3, refractory=10",
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
