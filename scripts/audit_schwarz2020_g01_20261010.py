"""Read BNCI 001-2020 sample events/header without an EEG performance test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

ROOT = Path(__file__).resolve().parents[1]


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", choices=("G01", "V01", "H01"), default="G01")
    participant = parser.parse_args().participant
    file = ROOT / "data" / "raw" / "schwarz2020_audit_20261010" / f"{participant}.mat"
    manifest_path = ROOT / "data" / "processed" / f"schwarz2020_{participant.lower()}_manifest_20261010.json"
    out = ROOT / "data" / "processed" / f"schwarz2020_{participant.lower()}_raw_audit_20261010_v6.json"
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sizes = {name: {"shape": list(shape), "type": typ} for name, shape, typ in whosmat(file)}
    if set(sizes) != {"events", "header", "signal"}:
        raise AssertionError(f"Unexpected MATLAB variables: {sizes}")
    n_channels, n_samples = sizes["signal"]["shape"]
    loaded = loadmat(file, variable_names=["events", "header"], squeeze_me=True,
                     struct_as_record=False)
    ev, h = loaded["events"], loaded["header"]
    codes = np.ravel(ev.codes).astype(np.int64)
    pos = np.ravel(ev.positions).astype(np.int64)
    if len(codes) != len(pos) or not np.all(np.diff(pos) >= 0):
        raise AssertionError("event codes and positions not matched or chronological")
    if np.min(pos) < 1 or np.max(pos) > n_samples:
        raise AssertionError("event positions outside EEG matrix")
    fs = int(h.sample_rate)
    counts = {str(int(c)): int(np.sum(codes == c)) for c in np.unique(codes)}
    rest_start = pos[codes == 768]
    rest_end = pos[codes == 769]
    movement_onsets = pos[np.isin(codes, [503587, 503588])]
    movement_gaps = np.diff(movement_onsets) / fs
    largest_gap_indices = np.argsort(movement_gaps)[::-1][:12]
    rest_pairs = []
    if len(rest_start) == len(rest_end):
        for a, b in zip(rest_start, rest_end):
            rest_pairs.append({"onset_sample_1based": int(a), "offset_sample_1based": int(b),
                               "duration_s": float((b-a)/fs)})
    runs = np.asarray(h.runs)
    run_details = []
    if runs.size:
        if runs.ndim != 2 or runs.shape[1] != 3:
            raise AssertionError(f"unexpected header.runs shape: {runs.shape}")
        if (int(runs[0, 1]) != 1 or int(runs[-1, 2]) != n_samples or
                not np.all(runs[1:, 1].astype(np.int64) == runs[:-1, 2].astype(np.int64) + 1)):
            raise AssertionError("header.runs does not partition the stored signal")
        for task_code, start, end in runs.astype(np.int64):
            in_run = (pos >= start) & (pos <= end)
            run_details.append({"task_code": int(task_code),
                                "start_sample_1based": int(start),
                                "end_sample_1based": int(end),
                                "duration_s": float((end-start+1)/fs),
                                "palmar_onsets": int(np.sum(in_run & (codes == 503587))),
                                "lateral_onsets": int(np.sum(in_run & (codes == 503588))),
                                "rest_starts": int(np.sum(in_run & (codes == 768)))})
        if sum(r["palmar_onsets"] for r in run_details) != np.sum(codes == 503587):
            raise AssertionError("palmar events not partitioned by runs")
        if sum(r["lateral_onsets"] for r in run_details) != np.sum(codes == 503588):
            raise AssertionError("lateral events not partitioned by runs")
    result = {
        "status": f"single BNCI {participant} original-file event/header audit; EEG content and sensor timing not verified",
        "source": manifest,
        "variables": sizes,
        "sampling_rate_hz": fs,
        "duration_s": float(n_samples/fs),
        "event_count": int(len(codes)),
        "event_counts_by_code": counts,
        "rest_pairs_chronological": rest_pairs,
        "rest_pairing_count_matches": bool(len(rest_start) == len(rest_end)),
        "movement_onset_counts": {"palmar_503587": int(np.sum(codes == 503587)),
                                  "lateral_503588": int(np.sum(codes == 503588))},
        "movement_onset_first_last_1based": [int(movement_onsets[0]), int(movement_onsets[-1])],
        "movement_onset_first_last_s": [float((movement_onsets[0]-1)/fs),
                                        float((movement_onsets[-1]-1)/fs)],
        "movement_onset_samples_1based": [int(v) for v in movement_onsets],
        "movement_onset_gap_s_quantiles_0_50_90_99_100": [float(v) for v in
            np.quantile(movement_gaps, [0, .5, .9, .99, 1])],
        "largest_movement_onset_gaps": [{"after_movement_number_1based": int(i+1),
                                           "gap_s": float(movement_gaps[i]),
                                           "before_s": float((movement_onsets[i]-1)/fs),
                                           "after_s": float((movement_onsets[i+1]-1)/fs)}
                                          for i in largest_gap_indices],
        "rest_onsets_s": [float((a-1)/fs) for a in rest_start],
        "rest_all_after_last_movement_onset": bool(len(rest_start) and
                                                 np.all(rest_start > movement_onsets[-1])),
        "first_20_events_code_time_s": [[int(c), float((p-1)/fs)]
                                         for c,p in zip(codes[:20],pos[:20])],
        "last_20_events_code_time_s": [[int(c), float((p-1)/fs)]
                                        for c,p in zip(codes[-20:],pos[-20:])],
        "header_fields": list(h._fieldnames),
        "header_runs": _plain(h.runs),
        "run_details": run_details,
        "header_device_type": str(h.device_type),
        "header_eeg_channel_count": int(np.size(h.channels_eeg)),
        "header_eog_channel_count": int(np.size(h.channels_eog)),
        "header_channel_labels": [str(v) for v in np.ravel(h.channels_labels)],
        "limits": ["one participant per system only", "sensor-to-EEG latency not measured",
                   "no EEG-only model or FP/SEN analysis"],
    }
    if n_channels != result["header_eeg_channel_count"] + result["header_eog_channel_count"]:
        raise AssertionError("signal rows do not match EEG+EOG header counts")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"signal_shape": sizes["signal"]["shape"],
                      "counts": counts, "rest_pairs": rest_pairs,
                      "header_runs": result["header_runs"],
                      "largest_movement_onset_gaps": result["largest_movement_onset_gaps"][:5]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
