"""Maris-Oostenveld visualization for post-cue versus pre-cue EEG."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import t as student_t

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-prepost-")
import mne


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/processed/lee2019_cue_erp_subjects.npz"),
    )
    parser.add_argument("--permutations", type=int, default=2048)
    parser.add_argument("--low-hz", type=float)
    parser.add_argument("--high-hz", type=float)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_maris_oostenveld_pre_vs_post.png"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("data/processed/lee2019_maris_oostenveld_pre_vs_post.json"),
    )
    args = parser.parse_args()
    cache = np.load(args.cache, allow_pickle=False)
    erp, times = cache["subject_erps"], cache["times"]
    channels = cache["channels"].tolist()
    if (args.low_hz is None) != (args.high_hz is None):
        raise ValueError("--low-hz and --high-hz must be provided together")
    band_label = "0.5–30 Hz source ERP"
    if args.low_hz is not None:
        sos = butter(
            4, (args.low_hz, args.high_hz), btype="bandpass",
            fs=250, output="sos",
        )
        erp = sosfiltfilt(sos, erp, axis=-1)
        band_label = f"{args.low_hz:g}–{args.high_hz:g} Hz"
    pre = erp[:, :, (times >= -0.5) & (times < 0)]
    post = erp[:, :, (times >= 0) & (times < 0.5)]
    count = min(pre.shape[-1], post.shape[-1])
    pre, post = pre[..., -count:], post[..., :count]
    relative_times = np.arange(count) / 250
    difference = (post - pre).transpose(0, 2, 1)

    info = mne.create_info(channels, 250, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    adjacency, names = mne.channels.find_ch_adjacency(info, "eeg")
    if names != channels:
        raise ValueError("Adjacency channel order mismatch")
    threshold = float(student_t.ppf(0.975, len(difference)-1))
    t_obs, clusters, p_values, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
        difference, threshold=threshold, n_permutations=args.permutations,
        tail=0, adjacency=adjacency, seed=20260731, n_jobs=1,
        out_type="indices", verbose=False,
    )
    significant = np.zeros_like(t_obs, dtype=bool)
    records, masses = [], []
    for cluster, p_value in zip(clusters, p_values):
        ti, ci = cluster
        if p_value >= 0.05:
            continue
        significant[ti, ci] = True
        mass = float(np.sum(t_obs[ti, ci]))
        masses.append(mass)
        peak = np.argmax(np.abs(t_obs[ti, ci]))
        records.append({
            "p_value": float(p_value),
            "cluster_mass": mass,
            "start_ms_within_segment": float(relative_times[ti.min()]*1000),
            "stop_ms_within_segment": float(relative_times[ti.max()]*1000),
            "peak_channel": channels[int(ci[peak])],
            "peak_time_ms_within_segment": float(relative_times[ti[peak]]*1000),
            "peak_t": float(t_obs[ti[peak], ci[peak]]),
        })

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1, 1.05))
    for column, channel in enumerate(("Oz", "Cz")):
        axis = fig.add_subplot(grid[0, column])
        index = channels.index(channel)
        for values, color, label in (
            (post[:, index], "#2166ac", "Post-cue (0 to +500 ms)"),
            (pre[:, index], "#666666", "Pre-cue (−500 to 0 ms)"),
        ):
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            axis.plot(relative_times, mean, color=color, lw=2, label=label)
            axis.fill_between(
                relative_times, mean-sem, mean+sem,
                color=color, alpha=0.17, linewidth=0,
            )
        indices = np.flatnonzero(significant[:, index])
        if len(indices):
            groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
            for group in groups:
                axis.axvspan(
                    relative_times[group[0]], relative_times[group[-1]],
                    color="#f4a3a3", alpha=0.27, linewidth=0,
                )
        axis.axhline(0, color="0.6", lw=0.7)
        axis.set(
            xlim=(0, 0.5), title=f"{channel}: Post-cue vs pre-cue",
            xlabel="Matched time within 500-ms segment (s)",
            ylabel="Amplitude (µV)",
        )
        axis.legend(frameon=False, fontsize=9)

    null_axis = fig.add_subplot(grid[0, 2:])
    null_axis.hist(np.abs(h0), bins=40, color="0.75", edgecolor="white")
    colors = ("#b2182b", "#2166ac")
    for index, mass in enumerate(masses):
        null_axis.axvline(
            abs(mass), color=colors[index % 2], lw=2.5,
            label=f"Observed cluster {index+1}: |mass|={abs(mass):.1f}",
        )
    null_axis.set(
        title="Permutation null distribution of maximum cluster mass",
        xlabel="|maximum cluster mass|", ylabel="Permutation count",
    )
    null_axis.legend(frameon=False)

    plot_times = (0.10, 0.20, 0.30, 0.45)
    vmax, image, sensor_counts = float(np.max(np.abs(t_obs))), None, {}
    for column, point in enumerate(plot_times):
        axis = fig.add_subplot(grid[1, column])
        ti = int(np.argmin(np.abs(relative_times-point)))
        mask = significant[ti]
        sensor_counts[str(int(point*1000))] = int(mask.sum())
        image, _ = mne.viz.plot_topomap(
            t_obs[ti], info, axes=axis, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), contours=6, mask=mask,
            mask_params={
                "marker": "o", "markerfacecolor": "yellow",
                "markeredgecolor": "black", "linewidth": 0, "markersize": 5,
            },
        )
        axis.set_title(
            f"Matched t={int(point*1000)} ms\n"
            f"post +{int(point*1000)} vs pre {int(-500+point*1000)} ms\n"
            f"significant sensors: {mask.sum()}/62"
        )
    bar = fig.colorbar(image, ax=fig.axes[-4:], shrink=0.72, location="right")
    bar.set_label("Paired t-value (Post-cue − pre-cue)")
    fig.suptitle(
        f"Lee2019 Post-cue vs Pre-cue ({band_label}) — "
        "Maris & Oostenveld cluster permutation",
        fontsize=17,
    )
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "reference_doi": "10.1016/j.jneumeth.2007.03.024",
        "subjects": int(len(erp)),
        "comparison": "post cue 0..500 ms versus pre cue -500..0 ms",
        "analysis_band_hz": (
            [args.low_hz, args.high_hz] if args.low_hz is not None
            else [0.5, 30.0]
        ),
        "permutations": args.permutations,
        "cluster_forming_threshold_t": threshold,
        "significant_clusters": records,
        "significant_sensor_counts_by_matched_time_ms": sensor_counts,
        "interpretation_limit": (
            "Pre-cue is late fixation and can contain anticipation."
        ),
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
