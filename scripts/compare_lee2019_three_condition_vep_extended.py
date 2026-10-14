"""Three-condition 1-30 Hz ERP comparison over matched 1.5-s intervals."""

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

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-extended-vep-")
import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src"))
from uncued_mi.data import lee2019_path, load_lee2019_run
from scripts.analyze_lee2019_rest_trial_stft_clusters import ROIS
from scripts.compare_lee2019_three_condition_vep import cluster_test


def average_epochs(values, fs, anchors, tmin=-.5, tmax=2.0):
    left, right = int(round(tmin*fs)), int(round(tmax*fs))
    times = np.arange(left, right)/fs
    epochs = []
    for anchor in anchors:
        start, stop = int(anchor)+left, int(anchor)+right
        if start >= 0 and stop <= len(values):
            epoch = values[start:stop].T
            epoch -= epoch.mean(axis=0, keepdims=True)
            epochs.append(epoch)
    average = np.mean(epochs, axis=0)
    sos = butter(4, (1., 30.), btype="bandpass", fs=fs, output="sos")
    average = sosfiltfilt(sos, average, axis=-1)
    baseline = (times >= -.2) & (times < 0)
    average -= average[:, baseline].mean(axis=1, keepdims=True)
    down = int(round(fs/250.))
    return average[:, ::down], times[::down], len(epochs)


def subject_conditions(path):
    train = load_lee2019_run(path, "train")
    test = load_lee2019_run(path, "test")
    fs = train.sfreq
    cues = np.asarray(train.cue_samples)
    task, times, task_count = average_epochs(train.eeg, fs, cues)
    far, _, far_count = average_epochs(train.eeg, fs, cues-int(round(2*fs)))
    blocks = (train.pre_rest, train.post_rest, test.pre_rest, test.post_rest)
    rest_erps, rest_count = [], 0
    for block in blocks:
        anchors = np.linspace(int(.5*fs), len(block)-int(2*fs), 25).astype(int)
        current, _, count = average_epochs(block, fs, anchors)
        rest_erps.append(current); rest_count += count
    return far, np.mean(rest_erps, axis=0), task, times, train.channels, (far_count, rest_count, task_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1,55)))
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=2048)
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_three_condition_vep_1_30_extended"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_three_condition_vep_1_30_extended_roi.png"))
    parser.add_argument("--topography", type=Path, default=Path("data/processed/lee2019_three_condition_vep_1_30_extended_topography.png"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/lee2019_three_condition_vep_1_30_extended.json"))
    args = parser.parse_args()

    rows = {"far pre-cue": [], "resting": [], "task cue": []}
    times = channels = None; counts = []
    for subject in args.subjects:
        cache = args.cache_root/f"session{args.session}"/f"sub-{subject:02d}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as d:
                far, rest, task = d["far"], d["rest"], d["task"]
                current_times, current_channels, count = d["times"], tuple(d["channels"].tolist()), d["counts"]
        else:
            far, rest, task, current_times, current_channels, count = subject_conditions(
                lee2019_path(args.data_root, subject, args.session))
            scale = 1e6 if np.nanpercentile(np.abs(task), 99) < .01 else 1.
            far *= scale; rest *= scale; task *= scale
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, far=far, rest=rest, task=task,
                                times=current_times, channels=np.asarray(current_channels), counts=count)
        if times is None: times, channels = current_times, current_channels
        rows["far pre-cue"].append(far); rows["resting"].append(rest); rows["task cue"].append(task)
        counts.append(count); print(f"extended-vep session={args.session} subject={subject}", flush=True)
    conditions = {name: np.asarray(values) for name, values in rows.items()}

    contrasts = (("task cue", "far pre-cue"), ("task cue", "resting"), ("far pre-cue", "resting"))
    tests = {}
    for index, (left, right) in enumerate(contrasts):
        tests[f"{left} - {right}"] = cluster_test(
            conditions[left], conditions[right], times, list(channels),
            args.permutations, 20260802+index, evaluation_stop=1.5)

    display = (times >= -.2) & (times <= 1.5)
    palette = {"far pre-cue":"#2563eb", "resting":"#6b7280", "task cue":"#dc2626"}
    fig, axes = plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
    for axis, (roi_name, names) in zip(axes.flat, ROIS.items()):
        indices = [list(channels).index(name) for name in names]
        for name, values in conditions.items():
            trace = values[:, indices].mean(1); mean=trace.mean(0); sem=trace.std(0,ddof=1)/np.sqrt(len(trace))
            axis.plot(times[display],mean[display],color=palette[name],lw=2,label=name)
            axis.fill_between(times[display],(mean-sem)[display],(mean+sem)[display],color=palette[name],alpha=.15)
        axis.axvline(0,color="black",ls="--",lw=.8); axis.axhline(0,color=".5",lw=.7)
        axis.set(title=roi_name,xlabel="Local time (s)",ylabel="Amplitude (uV)"); axis.legend(frameon=False)
    fig.suptitle("Lee2019 three-condition ERP: matched 1.5-s intervals, 1-30 Hz")
    args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=220,bbox_inches="tight"); plt.close(fig)

    info=mne.create_info(list(channels),250.,"eeg"); info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    latencies=(.1,.2,.5,1.0,1.5); fig,axes=plt.subplots(3,5,figsize=(17,10),constrained_layout=True)
    vals=[v.mean(0)[:,np.argmin(abs(times-t))] for v in conditions.values() for t in latencies]
    vmax=np.percentile(np.abs(np.concatenate(vals)),98); image=None
    for row,(name,values) in enumerate(conditions.items()):
        grand=values.mean(0)
        for col,t in enumerate(latencies):
            idx=np.argmin(abs(times-t)); image,_=mne.viz.plot_topomap(grand[:,idx],info,axes=axes[row,col],show=False,cmap="RdBu_r",vlim=(-vmax,vmax),sensors=True)
            axes[row,col].set_title(f"{name}: {t:g} s")
    fig.colorbar(image,ax=axes,shrink=.65,label="Amplitude (uV)"); fig.suptitle("Matched-latency scalp topographies, 1-30 Hz")
    fig.savefig(args.topography,dpi=220,bbox_inches="tight"); plt.close(fig)

    payload={"subjects":len(args.subjects),"conditions":{"far_pre_cue":"cue-2.0 to cue-0.5 s","resting":"pseudo-event 0 to 1.5 s","task":"cue to cue+1.5 s"},
             "processing":{"bandpass_hz":[1,30],"baseline_s":[-.2,0],"reference":"CAR"},
             "pairwise_cluster_tests":tests,"planned_contrast_bonferroni_alpha":.05/3}
    args.summary.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__ == "__main__": main()
