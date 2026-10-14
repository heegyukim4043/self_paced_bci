"""Read-only precheck of existing Lee2019 pseudo-uncued replay geometry."""

import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1] / "data/interim/lee2019_pseudo_uncued_raw"
paths = sorted(root.glob("session*/sub-*.npz"))
if not paths:
    raise SystemExit(f"No replay files under {root}")

rows = []
chronological = 0
for path in paths:
    with np.load(path, allow_pickle=False) as data:
        events = data["events"]
        trial_ids = data["trial_ids"]
        sfreq = float(data["sfreq"])
    assert sfreq == 250 and events.shape == (50, 8), path
    assert np.array_equal(np.sort(trial_ids), np.arange(50, 100)), path
    assert np.array_equal(trial_ids, events[:, 1]), path
    assert np.all(events[:, 6] - events[:, 5] == 125), path
    assert np.all(events[:, 7] - events[:, 5] == 875), path
    assert np.all(events[1:, 2] - events[:-1, 5] == 1000), path
    chronological += int(np.array_equal(trial_ids, np.arange(50, 100)))
    rows.append(events)

events = np.concatenate(rows)
cue = events[:, 5]
# The replay uses a 1-s window ending on a 25-sample grid. The trial seam is
# cue-3 s and the 3-s guard requires window start >= cue.
first_end = np.ceil(cue / 25).astype(np.int64) * 25 + 250
first_dwell3 = first_end + 2 * 25
print(json.dumps({
    "files": len(paths),
    "events": len(events),
    "chronological_replays": chronological,
    "cue_to_positive_start_s": 0.5,
    "cue_to_positive_stop_s": 3.5,
    "cue_to_next_rest_start_s": 4.0,
    "first_allowed_window_end_after_cue_s": [
        float(np.min((first_end - cue) / 250)),
        float(np.max((first_end - cue) / 250)),
    ],
    "earliest_dwell3_alarm_after_pseudo_onset_s": [
        float(np.min((first_dwell3 - cue - 125) / 250)),
        float(np.max((first_dwell3 - cue - 125) / 250)),
    ],
}, indent=2))
