"""Paper-style Maris & Oostenveld cluster-permutation visualization."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-maris-")
import mne


def main():
    parser = argparse.ArgumentParser()
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
        default=Path("data/processed/lee2019_maris_oostenveld_comparison.png"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("data/processed/lee2019_maris_oostenveld_comparison.json"),
    )
    args = parser.parse_args()
    cue_cache = np.load(args.cue_cache, allow_pickle=False)
    rest_cache = np.load(args.rest_cache, allow_pickle=False)
    cue, rest = cue_cache["subject_erps"], rest_cache["subject_erps"]
    times = cue_cache["times"]
    channels = cue_cache["channels"].tolist()
    test = (times >= 0) & (times <= 0.6)
    test_times = times[test]
    difference = (cue - rest)[:, :, test].transpose(0, 2, 1)

    info = mne.create_info(channels, 250.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    adjacency, names = mne.channels.find_ch_adjacency(info, "eeg")
    if names != channels:
        raise ValueError("Adjacency channel order mismatch")
    threshold = float(student_t.ppf(0.975, len(difference) - 1))
    t_obs, clusters, p_values, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
        difference, threshold=threshold, n_permutations=args.permutations,
        tail=0, adjacency=adjacency, seed=20260730, n_jobs=1,
        out_type="indices", verbose=False,
    )
    significant = np.zeros_like(t_obs, dtype=bool)
    observed_masses, records = [], []
    for cluster, p_value in zip(clusters, p_values):
        time_indices, channel_indices = cluster
        mass = float(np.sum(t_obs[time_indices, channel_indices]))
        if p_value < 0.05:
            significant[time_indices, channel_indices] = True
            observed_masses.append(mass)
            peak = np.argmax(np.abs(t_obs[time_indices, channel_indices]))
            records.append({
                "p_value": float(p_value),
                "cluster_mass": mass,
                "start_ms": float(test_times[np.min(time_indices)] * 1000),
                "stop_ms": float(test_times[np.max(time_indices)] * 1000),
                "peak_channel": channels[int(channel_indices[peak])],
                "peak_time_ms": float(test_times[int(time_indices[peak])] * 1000),
                "peak_t": float(t_obs[time_indices[peak], channel_indices[peak]]),
            })

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.05))
    waveform_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for axis, channel in zip(waveform_axes, ("Oz", "Cz")):
        index = channels.index(channel)
        for values, color, label in (
            (cue[:, index], "#2166ac", "Cue"),
            (rest[:, index], "#666666", "Rest pseudo-event"),
        ):
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            display = (times >= -0.2) & (times <= 0.8)
            axis.plot(times[display], mean[display], color=color, lw=2, label=label)
            axis.fill_between(
                times[display], (mean-sem)[display], (mean+sem)[display],
                color=color, alpha=0.17, linewidth=0,
            )
        channel_sig = significant[:, index]
        for group in np.split(
            np.flatnonzero(channel_sig),
            np.flatnonzero(np.diff(np.flatnonzero(channel_sig)) > 1) + 1
        ) if np.any(channel_sig) else []:
            axis.axvspan(
                test_times[group[0]], test_times[group[-1]],
                color="#f4a3a3", alpha=0.25, linewidth=0,
            )
        axis.axvline(0, color="black", ls="--", lw=1)
        axis.axhline(0, color="0.6", lw=0.7)
        axis.set(
            xlim=(-0.2, 0.8), title=f"{channel}: Cue vs rest",
            xlabel="Time from cue (s)", ylabel="Amplitude (µV)",
        )
        axis.legend(frameon=False)

    null_axis = fig.add_subplot(grid[0, 2:])
    null_values = np.abs(np.asarray(h0).ravel())
    null_axis.hist(null_values, bins=40, color="0.75", edgecolor="white")
    colors = ("#b2182b", "#2166ac")
    for index, mass in enumerate(observed_masses):
        null_axis.axvline(
            abs(mass), color=colors[index % len(colors)], lw=2.5,
            label=f"Observed cluster {index+1}: |mass|={abs(mass):.1f}",
        )
    null_axis.set(
        title="Permutation null distribution of maximum cluster mass",
        xlabel="|maximum cluster mass|", ylabel="Permutation count",
    )
    null_axis.legend(frameon=False)

    topomap_times = (0.10, 0.20, 0.30, 0.50)
    vmax = float(np.max(np.abs(t_obs)))
    image = None
    counts = {}
    for column, point in enumerate(topomap_times):
        axis = fig.add_subplot(grid[1, column])
        time_index = int(np.argmin(np.abs(test_times - point)))
        mask = significant[time_index]
        counts[str(int(point * 1000))] = int(np.sum(mask))
        image, _ = mne.viz.plot_topomap(
            t_obs[time_index], info, axes=axis, show=False,
            cmap="RdBu_r", vlim=(-vmax, vmax), contours=6,
            mask=mask,
            mask_params={
                "marker": "o", "markerfacecolor": "yellow",
                "markeredgecolor": "black", "linewidth": 0,
                "markersize": 5,
            },
        )
        axis.set_title(
            f"{int(point*1000)} ms\n"
            f"significant sensors: {np.sum(mask)}/62"
        )
    bar = fig.colorbar(image, ax=fig.axes[-4:], shrink=0.72, location="right")
    bar.set_label("Paired t-value (Cue − rest)")
    fig.suptitle(
        "Lee2019 Cue vs Rest — Maris & Oostenveld cluster permutation",
        fontsize=17,
    )
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "reference": {
            "authors": "Maris & Oostenveld",
            "year": 2007,
            "doi": "10.1016/j.jneumeth.2007.03.024",
        },
        "subjects": int(len(difference)),
        "permutations": args.permutations,
        "cluster_forming_threshold_t": threshold,
        "significant_clusters": records,
        "significant_sensor_counts_by_time_ms": counts,
        "visual_encoding": (
            "Pink ERP shading and yellow topomap sensors indicate membership "
            "in a cluster with cluster-level p<0.05."
        ),
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
