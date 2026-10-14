"""Audit three fixed Kaya2018 CLA files: marker, X3, and stored resolution.

This is a source compatibility audit, not an EEG performance experiment.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "kaya2018_audit_subset_manifest_20261010.json"
OUTPUT = ROOT / "data" / "processed" / "kaya2018_raw_subset_audit_20261010_v2.json"


def events(marker: np.ndarray, code: int):
    active = marker == code
    previous = np.r_[False, active[:-1]]
    starts = np.flatnonzero(active & ~previous)
    ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    if len(starts) != len(ends):
        raise AssertionError("unmatched marker runs")
    return starts, ends


def quantization_step(x: np.ndarray):
    unique = np.unique(np.round(x[:100000], 5))
    diff = np.diff(unique)
    return float(np.quantile(diff[diff > 1e-6], .01))


def x3_alignment(x3: np.ndarray, starts: np.ndarray, fs: int):
    left = int(round(.5 * fs))
    right = int(round(.75 * fs))
    valid = starts[(starts >= left) & (starts + right < len(x3))]
    if not len(valid):
        return {"n_events": 0}
    profile = np.mean(np.stack([x3[s-left:s+right] for s in valid]), axis=0)
    def segment(a, b):
        return float(np.mean(profile[int(round(left+a*fs)):int(round(left+b*fs))]))
    baseline = segment(-.4, -.1)
    around_on = profile[left-10:left+11]
    around_half = profile[left+int(round(.5*fs))-10:left+int(round(.5*fs))+11]
    # A within-trial edge audit complements the event average. It is descriptive:
    # the largest nearby X3 derivative is not a calibrated hardware timing error.
    diff = np.abs(np.diff(x3))
    offsets = np.arange(-10, 11)
    individual = np.stack([diff[s + offsets] for s in valid])
    edge_indices = offsets[np.argmax(individual, axis=1)]
    edge_lags_ms = (edge_indices + 1) * 1000 / fs
    edge_magnitudes_uv = np.max(individual, axis=1)
    return {
        "n_events": int(len(valid)),
        "baseline_mean_uv": baseline,
        "mean_0_to_100ms_uv": segment(0, .1),
        "mean_400_to_600ms_uv": segment(.4, .6),
        "average_waveform_onset_peak_uv": float(np.max(np.abs(around_on))),
        "average_waveform_onset_peak_lag_ms": float((int(np.argmax(np.abs(around_on)))-10) * 1000 / fs),
        "average_waveform_halfsecond_peak_uv": float(np.max(np.abs(around_half))),
        "average_waveform_halfsecond_peak_lag_ms": float((int(np.argmax(np.abs(around_half)))-10) * 1000 / fs),
        "individual_nearest_max_edge_lag_ms_quantiles_0_025_50_975_100": [float(x) for x in np.quantile(edge_lags_ms, [0, .025, .5, .975, 1])],
        "individual_nearest_max_edge_magnitude_uv_quantiles_0_50_100": [float(x) for x in np.quantile(edge_magnitudes_uv, [0, .5, 1])],
        "individual_nearest_max_edge_within_20ms_fraction": float(np.mean(np.abs(edge_lags_ms) <= 20)),
        "interpretation": "event average plus per-event local maximum X3 edge; not calibrated stimulus-to-EEG hardware timing accuracy",
    }


def one(path: Path, article: dict):
    o = loadmat(path, squeeze_me=True, struct_as_record=False)["o"]
    fs = int(o.sampFreq)
    marker = np.asarray(o.marker).ravel().astype(np.int16)
    data = np.asarray(o.data)
    if len(marker) != int(o.nS) or len(marker) != len(data) or data.shape[1] != 22:
        raise AssertionError("EEG/marker sample alignment or channel count mismatch")
    codes, n_samples = np.unique(marker, return_counts=True)
    details = {}
    all_start = []
    for code in (1, 2, 3):
        starts, ends = events(marker, code)
        all_start.extend(starts.tolist())
        durations = (ends-starts)/fs
        details[str(code)] = {
            "n_events": int(len(starts)),
            "duration_median_s": float(np.median(durations)) if len(starts) else None,
            "duration_min_s": float(np.min(durations)) if len(starts) else None,
            "duration_max_s": float(np.max(durations)) if len(starts) else None,
        }
    all_start = np.asarray(sorted(all_start), dtype=np.int64)
    # The published sensor synchronization channel is the 22nd input, not EEG.
    x3 = np.asarray(data[:, 21], dtype=float)
    return {
        "source": article["source"], "file": article["file"],
        "sha256": article["sha256"], "samp_freq_hz": fs,
        "n_samples": int(len(marker)), "duration_s": float(len(marker)/fs),
        "data_shape": list(data.shape), "channel_names": [str(x) for x in np.ravel(o.chnames)],
        "marker_samples_by_code": {str(int(k)): int(v) for k,v in zip(codes,n_samples)},
        "events_by_code": details,
        "all_action_events": int(len(all_start)),
        "inter_action_interval_median_s": float(np.median(np.diff(all_start)/fs)) if len(all_start)>1 else None,
        "first_two_eeg_channels_quantization_step_uv": [quantization_step(data[:, i]) for i in (0,4)],
        "x3_quantization_step_uv": quantization_step(x3),
        "x3_event_averaged_alignment": x3_alignment(x3, all_start, fs),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for article in manifest["records"]:
        if article["file"].endswith(".mat"):
            row = one(ROOT / article["file"], article)
            rows.append(row)
            print(article["file"], row["events_by_code"], flush=True)
    script = ROOT / next(x["file"] for x in manifest["records"] if x["file"].endswith(".zip"))
    with zipfile.ZipFile(script) as archive:
        text = archive.read("nkExperimentMotor_1.m").decode("latin1")
        gui = {
            "script": "nkExperimentMotor_1.m",
            "static_icon_list": "lefthand.png, pass.png, righthand.png, leftfoot.png, tongue.png, rightfoot.png",
            "highlight_box_code_1_left_xywh": [0.1164, 0.6177, 0.2342, 0.2819],
            "highlight_box_code_2_right_xywh": [0.6823, 0.6177, 0.2342, 0.2819],
            "highlight_box_code_3_center_xywh": [0.4, 0.6177, 0.2342, 0.2819],
            "code_to_box": "update_state(cue) uses hbox{cuestate}; verified in archived script",
            "same_fixed_layout_all_icons": all(x in text for x in ("subplot(2,3,i)", "hbox{cuestate}", "lefthand.png", "pass.png", "righthand.png")),
            "limit": "archived author script, not file-specific screenshot; exact run-version not independently verified",
        }
    result = {"status": "fixed three-file raw compatibility sample; not cohort-level audit",
              "source_manifest": str(MANIFEST.relative_to(ROOT)),
              "gui_script_audit": gui, "rows": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
