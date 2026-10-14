"""Paired spatiotemporal permutation test: visual cue ERP vs resting EEG."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import t as student_t

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-cue-rest-")
import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.data import lee2019_path, load_lee2019_run
from scripts.plot_lee2019_cue_erp_topography import GROUPS


def average_pseudo_epochs(values, fs, anchors, tmin=-0.5, tmax=1.0):
    left, right = int(round(tmin * fs)), int(round(tmax * fs))
    times = np.arange(left, right) / fs
    total = np.zeros((values.shape[1], len(times)), dtype=np.float64)
    count = 0
    for anchor in anchors:
        start, stop = int(anchor) + left, int(anchor) + right
        if start < 0 or stop > len(values):
            continue
        epoch = values[start:stop].T
        epoch -= epoch.mean(axis=0, keepdims=True)
        total += epoch
        count += 1
    average = total / count
    sos = butter(4, (0.5, 30.0), btype="bandpass", fs=fs, output="sos")
    average = sosfiltfilt(sos, average, axis=-1)
    baseline = (times >= -0.2) & (times < 0)
    average -= average[:, baseline].mean(axis=1, keepdims=True)
    decimation = int(round(fs / 250))
    return average[:, ::decimation], times[::decimation], count


def subject_rest_erp(path):
    train = load_lee2019_run(path, "train")
    test = load_lee2019_run(path, "test")
    blocks = (train.pre_rest, train.post_rest, test.pre_rest, test.post_rest)
    fs = train.sfreq
    erps, counts = [], []
    for block in blocks:
        # 25 deterministic anchors per block, away from epoch boundaries.
        anchors = np.linspace(int(fs), len(block) - int(fs), 25).astype(int)
        erp, times, count = average_pseudo_epochs(block, fs, anchors)
        erps.append(erp)
        counts.append(count)
    return np.mean(erps, axis=0), times, train.channels, sum(counts)


def cluster_records(t_obs, clusters, p_values, times, channels):
    records = []
    for cluster, p_value in zip(clusters, p_values):
        if p_value >= 0.05:
            continue
        time_indices, channel_indices = cluster
        unique_times = np.unique(time_indices)
        unique_channels = np.unique(channel_indices)
        cluster_t = np.abs(t_obs[time_indices, channel_indices])
        peak = int(np.argmax(cluster_t))
        records.append({
            "p_value": float(p_value),
            "start_ms": float(times[unique_times.min()] * 1000),
            "stop_ms": float(times[unique_times.max()] * 1000),
            "channel_count": int(len(unique_channels)),
            "channels": [channels[index] for index in unique_channels],
            "peak_channel": channels[int(channel_indices[peak])],
            "peak_time_ms": float(times[int(time_indices[peak])] * 1000),
            "peak_t": float(t_obs[time_indices[peak], channel_indices[peak]]),
            "peak_abs_t": float(cluster_t[peak]),
            "polarity_at_peak": (
                "cue_greater" if t_obs[time_indices[peak], channel_indices[peak]] > 0
                else "cue_lower"
            ),
        })
    return sorted(records, key=lambda item: item["p_value"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument(
        "--cue-cache", type=Path,
        default=Path("data/processed/lee2019_cue_erp_subjects.npz"),
    )
    parser.add_argument(
        "--rest-cache", type=Path,
        default=Path("data/processed/lee2019_rest_pseudoevent_erp_subjects.npz"),
    )
    parser.add_argument("--permutations", type=int, default=2048)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cue_vs_rest_permutation.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_cue_vs_rest_permutation.png"),
    )
    args = parser.parse_args()

    cue_cache = np.load(args.cue_cache, allow_pickle=False)
    cue = cue_cache["subject_erps"]
    channels = cue_cache["channels"].tolist()
    times = cue_cache["times"]
    identities = cue_cache["identities"]
    if args.rest_cache.exists():
        rest_cache = np.load(args.rest_cache, allow_pickle=False)
        rest = rest_cache["subject_erps"]
    else:
        rest, counts = [], []
        for session, subject in identities:
            erp, current_times, current_channels, count = subject_rest_erp(
                lee2019_path(args.data_root, int(subject), int(session))
            )
            if tuple(current_channels) != tuple(channels):
                raise ValueError("Rest channel order differs from cue cache")
            if not np.allclose(current_times, times):
                raise ValueError("Rest and cue time axes differ")
            rest.append(erp)
            counts.append(count)
            print(f"rest-erp session={session} subject={subject}", flush=True)
        rest = np.asarray(rest)
        # Match the microvolt scaling recorded in the cue cache.
        if np.nanpercentile(np.abs(rest), 99) < 0.01:
            rest *= 1e6
        np.savez_compressed(
            args.rest_cache, subject_erps=rest, times=times,
            channels=np.asarray(channels), identities=identities,
            pseudo_event_counts=counts, unit="uV",
        )

    test_mask = (times >= 0) & (times <= 0.6)
    test_times = times[test_mask]
    difference = (cue - rest)[:, :, test_mask].transpose(0, 2, 1)
    info = mne.create_info(channels, 250.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    adjacency, adjacency_channels = mne.channels.find_ch_adjacency(info, "eeg")
    if adjacency_channels != channels:
        raise ValueError("Adjacency channel order mismatch")
    threshold = float(student_t.ppf(0.975, len(difference) - 1))
    t_obs, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        difference, threshold=threshold, n_permutations=args.permutations,
        tail=0, adjacency=adjacency, seed=20260730, n_jobs=1,
        out_type="indices", verbose=False,
    )
    records = cluster_records(t_obs, clusters, p_values, test_times, channels)

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    palette = ("#2b6cb0", "#c53030", "#2f855a")
    roi_summary = {}
    for (label, names), color in zip(GROUPS.items(), palette):
        indices = [channels.index(name) for name in names]
        subject_difference = (cue[:, indices] - rest[:, indices]).mean(axis=1)
        mean = subject_difference.mean(axis=0)
        sem = subject_difference.std(axis=0, ddof=1) / np.sqrt(len(difference))
        display = (times >= -0.2) & (times <= 0.8)
        axis.plot(times[display], mean[display], lw=2, color=color, label=label)
        axis.fill_between(
            times[display], (mean-sem)[display], (mean+sem)[display],
            color=color, alpha=0.18, linewidth=0,
        )
        roi_clusters = [
            item for item in records if set(item["channels"]).intersection(names)
        ]
        roi_summary[label] = {
            "significant_cluster_count": len(roi_clusters),
            "significant_ranges_ms": [
                [item["start_ms"], item["stop_ms"]] for item in roi_clusters
            ],
        }
    axis.axvline(0, color="black", ls="--", lw=1)
    axis.axhline(0, color="0.5", lw=0.8)
    significance_handles = []
    positive_level, negative_level = 0.055, 0.105
    for item in records:
        start, stop = item["start_ms"] / 1000, item["stop_ms"] / 1000
        positive = item["polarity_at_peak"] == "cue_greater"
        color = "#b2182b" if positive else "#2166ac"
        level = positive_level if positive else negative_level
        axis.plot(
            [start, stop], [level, level],
            transform=axis.get_xaxis_transform(), color=color,
            lw=6, solid_capstyle="butt", clip_on=False,
        )
        axis.text(
            start, level + 0.012,
            "Cue > rest" if positive else "Cue < rest",
            transform=axis.get_xaxis_transform(), color=color,
            fontsize=9, va="bottom",
        )
        significance_handles.append(Line2D(
            [0], [0], color=color, lw=6,
            label=(
                f"{'Cue > rest' if positive else 'Cue < rest'} "
                f"cluster (p={item['p_value']:.4g})"
            ),
        ))
    axis.set(
        xlim=(-0.2, 0.8),
        xlabel="Time from visual cue onset (s)",
        ylabel="Cue − resting pseudo-event (µV)",
        title=(
            f"Cue vs resting ERP difference (N={len(difference)}; "
            f"cluster permutation, {args.permutations} permutations)"
        ),
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles + significance_handles, labels + [
            handle.get_label() for handle in significance_handles
        ],
        frameon=False, ncol=3, loc="upper right",
    )
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "subjects_sessions": int(len(difference)),
        "permutations": args.permutations,
        "test_interval_ms": [0, 600],
        "cluster_forming_threshold_t": threshold,
        "multiple_comparison_control": "spatiotemporal cluster permutation",
        "significant_cluster_count": len(records),
        "significant_clusters": records,
        "roi_summary": roi_summary,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
