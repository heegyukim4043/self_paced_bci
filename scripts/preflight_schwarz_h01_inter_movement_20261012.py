"""Metadata-only candidate idle exposure in H01 movement runs.

Not an EEG detector or proof that the candidate intervals are behaviourally idle.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/schwarz2020_audit_20261010/H01.mat"
OUTPUT = ROOT / "data/processed/schwarz_h01_inter_movement_preflight_20261012.json"
ONSET_TO_OFFSET = {503587: 534562, 503588: 534563}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    loaded = loadmat(SOURCE, variable_names=["events", "header"], squeeze_me=True, struct_as_record=False)
    ev, h = loaded["events"], loaded["header"]
    codes = np.ravel(ev.codes).astype(int)
    positions = np.ravel(ev.positions).astype(int)
    fs = float(h.sample_rate)
    runs = np.asarray(h.runs, dtype=int)
    detail = []
    for task, start, stop in runs:
        if task not in (3, 4, 7, 8):
            continue
        index = np.flatnonzero((positions >= start) & (positions <= stop) & np.isin(codes, list(ONSET_TO_OFFSET)))
        if len(index) != 40:
            raise AssertionError(f"H01 task {task} has {len(index)} instead of 40 onsets")
        events = []
        for k, i in enumerate(index):
            later = int(index[k + 1]) if k + 1 < len(index) else int(np.searchsorted(positions, stop, side="right"))
            matching = np.flatnonzero(codes[i + 1:later] == ONSET_TO_OFFSET[int(codes[i])])
            if len(matching) != 1:
                raise AssertionError(f"H01 task {task}, movement {k+1}: {len(matching)} matching offsets")
            offset_index = i + 1 + int(matching[0])
            if positions[offset_index] >= (positions[index[k+1]] if k+1<len(index) else stop):
                raise AssertionError("offset outside inter-movement interval")
            events.append({"onset_s": float((positions[i]-1)/fs),
                           "grasp_offset_s": float((positions[offset_index]-1)/fs)})
        blocks = {}
        for post_buffer in (3., 5., 7.):
            durations = []
            for k, current in enumerate(events[:-1]):
                left = current["grasp_offset_s"] + post_buffer + 1.0  # full prior 1-s EEG support
                right = events[k+1]["onset_s"] - 2.0  # instructed pre-movement gaze
                durations.append(max(0., right-left))
            blocks[str(post_buffer)] = {
                "candidate_unique_seconds": float(np.sum(durations)),
                "positive_intervals": int(np.sum(np.asarray(durations)>0)),
                "interval_median_s": float(np.median(durations)),
                "interval_q10_q90_s": [float(v) for v in np.quantile(durations, [.1,.9])],
            }
        detail.append({"task_code": int(task), "run_start_s": float((start-1)/fs),
                       "run_stop_s": float((stop-1)/fs), "movement_onsets": len(events),
                       "first_last_onset_s": [events[0]["onset_s"],events[-1]["onset_s"]],
                       "candidate_by_post_offset_buffer_s": blocks})
    if [r["task_code"] for r in detail] != [3, 4, 7, 8]:
        raise AssertionError("H01 expected movement-run sequence absent")
    result = {
        "status": "H01 metadata-only candidate idle exposure; not labelled rest or EEG FP",
        "input": str(SOURCE.relative_to(ROOT)),
        "method": "Between matched grasp offset and next movement onset, exclude 3/5/7 s after offset, 2 s before next onset, plus 1 s prior EEG support",
        "limitations": ["hand return and feedback have no independent end marker",
                        "gaze start is instructed, not measured", "dry H01 has no EOG channels",
                        "only one subject, executed movement not covert MI"],
        "runs": detail,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
