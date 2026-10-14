"""Compare far pre-cue, resting, and cue/task ERPs in Lee2019."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import t as student_t

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-three-vep-")
import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.data import lee2019_path, load_lee2019_run
from scripts.analyze_lee2019_rest_trial_stft_clusters import ROIS


def far_precue_erp(path, tmin=-.5, tmax=1.0):
    run = load_lee2019_run(path, "train")
    fs = run.sfreq
    left, right = int(round(tmin*fs)), int(round(tmax*fs))
    times = np.arange(left, right)/fs
    offset = int(round(-2.0*fs))
    epochs = []
    for cue in run.cue_samples:
        anchor = int(cue)+offset
        start, stop = anchor+left, anchor+right
        epoch = run.eeg[start:stop].T
        epoch -= epoch.mean(axis=0, keepdims=True)
        epochs.append(epoch)
    average = np.mean(epochs, axis=0)
    sos = butter(4, (.5, 30.), btype="bandpass", fs=fs, output="sos")
    average = sosfiltfilt(sos, average, axis=-1)
    baseline = (times >= -.2) & (times < 0)
    average -= average[:, baseline].mean(axis=1, keepdims=True)
    down = int(round(fs/250.))
    return average[:, ::down], times[::down], run.channels, len(epochs)


def cluster_test(a, b, times, channels, permutations, seed, evaluation_stop=.5):
    mask = (times >= 0) & (times <= evaluation_stop)
    difference = (a-b)[:, :, mask].transpose(0, 2, 1)
    info = mne.create_info(channels, 250., "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    adjacency, adjacency_channels = mne.channels.find_ch_adjacency(info, "eeg")
    if adjacency_channels != channels:
        raise ValueError("Adjacency channel order mismatch")
    threshold = float(student_t.ppf(.975, len(difference)-1))
    t_obs, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        difference, threshold=threshold, n_permutations=permutations,
        tail=0, adjacency=adjacency, seed=seed, n_jobs=1,
        out_type="indices", verbose=False,
    )
    records = []
    test_times = times[mask]
    for cluster, p in zip(clusters, p_values):
        time_idx, channel_idx = cluster
        peak = int(np.argmax(np.abs(t_obs[time_idx, channel_idx])))
        records.append({
            "p": float(p),
            "start_ms": float(test_times[np.min(time_idx)]*1000),
            "stop_ms": float(test_times[np.max(time_idx)]*1000),
            "channels": [channels[index] for index in np.unique(channel_idx)],
            "peak_channel": channels[int(channel_idx[peak])],
            "peak_ms": float(test_times[int(time_idx[peak])]*1000),
            "peak_t": float(t_obs[time_idx[peak], channel_idx[peak]]),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--cue-cache", type=Path, default=Path("data/processed/lee2019_cue_erp_subjects.npz"))
    parser.add_argument("--rest-cache", type=Path, default=Path("data/processed/lee2019_rest_pseudoevent_erp_subjects.npz"))
    parser.add_argument("--pre-cache", type=Path, default=Path("data/processed/lee2019_far_precue_erp_subjects.npz"))
    parser.add_argument("--permutations", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_three_condition_vep_roi.png"))
    parser.add_argument("--topography", type=Path, default=Path("data/processed/lee2019_three_condition_vep_topography.png"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/lee2019_three_condition_vep.json"))
    args = parser.parse_args()

    cue_data = np.load(args.cue_cache, allow_pickle=False)
    rest_data = np.load(args.rest_cache, allow_pickle=False)
    task = cue_data["subject_erps"]
    rest = rest_data["subject_erps"]
    times = cue_data["times"]
    channels = cue_data["channels"].tolist()
    identities = cue_data["identities"]
    if args.pre_cache.exists():
        pre_data = np.load(args.pre_cache, allow_pickle=False)
        pre = pre_data["subject_erps"]
    else:
        rows, counts = [], []
        for session, subject in identities:
            average, current_times, current_channels, count = far_precue_erp(
                lee2019_path(args.data_root, int(subject), int(session))
            )
            if tuple(current_channels) != tuple(channels) or not np.allclose(current_times, times):
                raise ValueError("Far-pre cache axes differ")
            rows.append(average); counts.append(count)
            print(f"far-pre session={session} subject={subject}", flush=True)
        pre = np.asarray(rows)
        if np.nanpercentile(np.abs(pre), 99) < .01:
            pre *= 1e6
        np.savez_compressed(args.pre_cache, subject_erps=pre, times=times,
                            channels=np.asarray(channels), identities=identities,
                            trial_counts=np.asarray(counts), unit="uV")

    conditions = {"far pre-cue": pre, "resting": rest, "task cue": task}
    contrasts = (("task cue", "far pre-cue"),
                 ("task cue", "resting"),
                 ("far pre-cue", "resting"))
    tests = {}
    for index, (left, right) in enumerate(contrasts):
        tests[f"{left} - {right}"] = cluster_test(
            conditions[left], conditions[right], times, channels,
            args.permutations, 20260801+index,
        )

    # Six ROI waveforms.
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    palette = {"far pre-cue": "#2563eb", "resting": "#6b7280", "task cue": "#dc2626"}
    display = (times >= -.2) & (times <= .5)
    for axis, (roi_name, names) in zip(axes.flat, ROIS.items()):
        indices = [channels.index(name) for name in names]
        for condition_name, values in conditions.items():
            subject_trace = values[:, indices].mean(axis=1)
            mean = subject_trace.mean(0); sem = subject_trace.std(0, ddof=1)/np.sqrt(len(values))
            color = palette[condition_name]
            axis.plot(times[display], mean[display], color=color, lw=2, label=condition_name)
            axis.fill_between(times[display], (mean-sem)[display], (mean+sem)[display], color=color, alpha=.15)
        axis.axvline(0, color="black", ls="--", lw=.8)
        axis.axhline(0, color="0.5", lw=.7)
        axis.set(title=roi_name, xlabel="Local time from condition onset (s)", ylabel="Amplitude (uV)")
        axis.legend(frameon=False)
    fig.suptitle("Lee2019: far pre-cue, four-rest pseudo-event, and task-cue ERP")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Condition topographies at matched local latencies.
    info = mne.create_info(channels, 250., "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    latencies = (.1, .2, .3, .4)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    all_values = [values.mean(0)[:, np.argmin(abs(times-latency))]
                  for values in conditions.values() for latency in latencies]
    vmax = np.percentile(np.abs(np.concatenate(all_values)), 98)
    image = None
    for row, (condition_name, values) in enumerate(conditions.items()):
        grand = values.mean(0)
        for column, latency in enumerate(latencies):
            index = int(np.argmin(abs(times-latency)))
            image, _ = mne.viz.plot_topomap(
                grand[:, index], info, axes=axes[row, column], show=False,
                cmap="RdBu_r", vlim=(-vmax, vmax), sensors=True,
            )
            axes[row, column].set_title(f"{condition_name}: {latency*1000:.0f} ms")
    fig.colorbar(image, ax=axes, shrink=.65, label="Amplitude (uV)")
    fig.suptitle("Matched-latency scalp topographies")
    fig.savefig(args.topography, dpi=220, bbox_inches="tight")
    plt.close(fig)

    alpha = .05/3
    payload = {
        "subjects": len(task),
        "conditions": {
            "far_pre_cue": {"anchor": "cue - 2.0 s", "evaluation_s": [0, .5]},
            "resting": {"anchor": "100 pseudo-events across four rest blocks", "evaluation_s": [0, .5]},
            "task": {"anchor": "visual cue onset", "evaluation_s": [0, .5]},
        },
        "processing": {"bandpass_hz": [.5, 30], "baseline_s": [-.2, 0], "reference": "CAR"},
        "pairwise_cluster_tests": tests,
        "planned_contrast_bonferroni_alpha": alpha,
        "interpretation": "Clusters are spatiotemporal and pairwise; use p < .0167 for the three planned contrasts.",
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
