"""Cue-locked grand-average ERP and scalp topographies for Lee2019."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
# Keep MNE configuration inside its temporary directory under restricted
# execution environments instead of writing a lock file to the user profile.
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-cue-erp-")
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.data import lee2019_path, load_lee2019_run

GROUPS = {
    "Occipital/PO": ("O1", "Oz", "O2", "PO3", "POz", "PO4"),
    "Sensorimotor": ("C3", "Cz", "C4", "CP3", "CPz", "CP4"),
    "Fronto-central": ("F3", "Fz", "F4", "FC3", "FC4"),
}


def subject_erp(path: Path, tmin=-0.5, tmax=1.0):
    run = load_lee2019_run(path, "train")
    fs = run.sfreq
    left, right = int(round(tmin * fs)), int(round(tmax * fs))
    samples = np.arange(left, right) / fs
    total = np.zeros((len(run.channels), len(samples)), dtype=np.float64)
    count = 0
    for cue in run.cue_samples:
        start, stop = int(cue) + left, int(cue) + right
        if start < 0 or stop > len(run.eeg):
            continue
        epoch = run.eeg[start:stop].T
        epoch = epoch - epoch.mean(axis=0, keepdims=True)  # CAR
        total += epoch
        count += 1
    if not count:
        raise RuntimeError(f"No valid cue epochs in {path}")
    average = total / count
    sos = butter(4, (0.5, 30.0), btype="bandpass", fs=fs, output="sos")
    average = sosfiltfilt(sos, average, axis=-1)
    baseline = (samples >= -0.2) & (samples < 0)
    average -= average[:, baseline].mean(axis=1, keepdims=True)
    decimation = int(round(fs / 250))
    return average[:, ::decimation], samples[::decimation], run.channels, count


def scale_to_microvolts(values):
    # Lee2019 files are commonly stored in microvolts. Handle volt-scaled
    # variants without changing the visual/statistical interpretation.
    scale = 1e6 if np.nanpercentile(np.abs(values), 99) < 0.01 else 1.0
    return values * scale, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/processed/lee2019_cue_erp_subjects.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cue_erp_topography.png"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("data/processed/lee2019_cue_erp_topography.json"),
    )
    args = parser.parse_args()

    erps, counts, identities = [], [], []
    channels = times = None
    for session in args.sessions:
        for subject in args.subjects:
            average, current_times, current_channels, count = subject_erp(
                lee2019_path(args.data_root, subject, session)
            )
            if channels is None:
                channels, times = current_channels, current_times
            if tuple(current_channels) != tuple(channels):
                raise ValueError("Channel order differs across files")
            erps.append(average)
            counts.append(count)
            identities.append((session, subject))
            print(f"cue-erp session={session} subject={subject}", flush=True)
    erps, scale = scale_to_microvolts(np.asarray(erps))
    grand = erps.mean(axis=0)
    sem = erps.std(axis=0, ddof=1) / np.sqrt(len(erps))
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache, subject_erps=erps, grand_average=grand, sem=sem,
        times=times, channels=np.asarray(channels), trial_counts=counts,
        identities=np.asarray(identities), unit="uV",
    )

    info = mne.create_info(list(channels), 250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    topomap_times = (0.10, 0.20, 0.30, 0.50)
    topomap_values = [
        grand[:, np.argmin(np.abs(times - point))] for point in topomap_times
    ]
    vmax = max(np.max(np.abs(value)) for value in topomap_values)

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.15))
    image = None
    for column, (point, values) in enumerate(zip(topomap_times, topomap_values)):
        axis = fig.add_subplot(grid[0, column])
        image, _ = mne.viz.plot_topomap(
            values, info, axes=axis, show=False, contours=6,
            cmap="RdBu_r", vlim=(-vmax, vmax), sensors=True,
        )
        axis.set_title(f"{int(point * 1000)} ms")
    color = fig.colorbar(image, ax=fig.axes[:4], shrink=0.75, location="right")
    color.set_label("Amplitude (µV)")

    axis = fig.add_subplot(grid[1, :])
    group_summary = {}
    palette = ("#2b6cb0", "#c53030", "#2f855a")
    for (label, names), color_value in zip(GROUPS.items(), palette):
        indices = [channels.index(name) for name in names if name in channels]
        subject_trace = erps[:, indices].mean(axis=1)
        trace = subject_trace.mean(axis=0)
        trace_sem = subject_trace.std(axis=0, ddof=1) / np.sqrt(len(erps))
        mask = (times >= -0.2) & (times <= 0.8)
        axis.plot(times[mask], trace[mask], label=label, color=color_value, lw=2)
        axis.fill_between(
            times[mask], (trace - trace_sem)[mask], (trace + trace_sem)[mask],
            color=color_value, alpha=0.18, linewidth=0,
        )
        post = (times >= 0) & (times <= 0.6)
        peak_index = np.argmax(np.abs(trace[post]))
        post_times = times[post]
        post_trace = trace[post]
        group_summary[label] = {
            "channels": [channels[index] for index in indices],
            "signed_peak_uv": float(post_trace[peak_index]),
            "peak_latency_ms": float(post_times[peak_index] * 1000),
        }
    axis.axvline(0, color="black", ls="--", lw=1)
    axis.axhline(0, color="0.5", lw=0.8)
    axis.set(
        xlim=(-0.2, 0.8), xlabel="Time from visual cue onset (s)",
        ylabel="Amplitude (µV)",
        title=f"Lee2019 cue-locked ERP (N={len(erps)}, mean ± SEM)",
    )
    axis.legend(frameon=False, ncol=3)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "subjects_sessions": len(erps),
        "subjects": args.subjects,
        "sessions": args.sessions,
        "trials_total": int(np.sum(counts)),
        "processing": {
            "epoch_seconds": [-0.5, 1.0],
            "baseline_seconds": [-0.2, 0.0],
            "bandpass_hz": [0.5, 30.0],
            "reference": "common average",
            "grand_average": "equal weight per subject-session",
            "unit_scale_applied": scale,
        },
        "topomap_times_ms": [int(value * 1000) for value in topomap_times],
        "groups": group_summary,
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
