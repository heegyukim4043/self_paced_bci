"""Maris-Oostenveld-style STFT comparison of Lee2019 rest and trial EEG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from scipy.signal import resample_poly, stft
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.data import lee2019_path, load_lee2019_run

ROIS = {
    "prefrontal": ("Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8"),
    "frontal": ("F9", "F7", "F3", "Fz", "F4", "F8", "F10"),
    "central": (
        "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    ),
    "parietal": ("P7", "P3", "P1", "Pz", "P2", "P4", "P8"),
    "temporal": (
        "FT9", "FTT9h", "T7", "TTP7h", "TP7", "TP9",
        "FT10", "FTT10h", "T8", "TPP8h", "TP8", "TP10",
    ),
    "occipital": ("PO9", "PO3", "POz", "PO4", "PO10", "O1", "Oz", "O2"),
}
RESTS = ("pre_train", "post_train", "pre_test", "post_test")


def car_resample(values, original_fs, target_fs=250.0):
    eeg = values.astype(np.float64)
    eeg -= eeg.mean(axis=1, keepdims=True)
    if original_fs != target_fs:
        down = int(round(original_fs / target_fs))
        eeg = resample_poly(eeg, 1, down, axis=0)
    return eeg


def epoch_array(signal, centers, before, after):
    epochs = []
    for center in centers:
        start, stop = int(center-before), int(center+after)
        if start >= 0 and stop <= len(signal):
            epochs.append(signal[start:stop].T)
    return np.asarray(epochs)  # epoch x channel x sample


def rest_epochs(signal, length, guard, step):
    starts = np.arange(guard, len(signal)-guard-length+1, step, dtype=int)
    return np.asarray([signal[start:start+length].T for start in starts])


def roi_tfr(epochs, channels, sfreq, nperseg, hop):
    frequencies, times, z = stft(
        epochs, fs=sfreq, window="hann", nperseg=nperseg,
        noverlap=nperseg-hop, nfft=nperseg, boundary=None, padded=False,
        axis=-1,
    )
    keep = (frequencies >= 1) & (frequencies <= 50)
    power = np.abs(z[..., keep, :]) ** 2
    # scipy moves frequency before time at the end for the selected axis.
    output = []
    lookup = {name.casefold(): index for index, name in enumerate(channels)}
    for names in ROIS.values():
        indices = [lookup[name.casefold()] for name in names]
        output.append(np.log(np.maximum(power[:, indices].mean(axis=(0, 1)), 1e-20)))
    return np.asarray(output), frequencies[keep], times


def subject_tfr(path, epoch_seconds, window_seconds, hop_seconds):
    train = load_lee2019_run(path, "train")
    test = load_lee2019_run(path, "test")
    fs = 250.0
    down = int(round(train.sfreq/fs))
    train_signal = car_resample(train.eeg, train.sfreq, fs)
    cue_centers = np.rint(np.asarray(train.cue_samples)/down).astype(int)
    before = int(round(2.5*fs)); after = int(round(1.5*fs))
    trials = epoch_array(train_signal, cue_centers, before, after)
    nperseg = int(round(window_seconds*fs)); hop = int(round(hop_seconds*fs))
    trial_tfr, frequencies, times = roi_tfr(
        trials, train.channels, fs, nperseg, hop
    )
    times = times - 2.5

    rest_sources = {
        "pre_train": (train.pre_rest, train.sfreq),
        "post_train": (train.post_rest, train.sfreq),
        "pre_test": (test.pre_rest, test.sfreq),
        "post_test": (test.post_rest, test.sfreq),
    }
    rest_values = []
    epoch_length = int(round(epoch_seconds*fs))
    guard = int(round(2*fs)); step = int(round(2*fs))
    for name in RESTS:
        values, native_fs = rest_sources[name]
        signal = car_resample(values, native_fs, fs)
        epochs = rest_epochs(signal, epoch_length, guard, step)
        current, rest_frequencies, rest_times = roi_tfr(
            epochs, train.channels, fs, nperseg, hop
        )
        if not np.array_equal(rest_frequencies, frequencies):
            raise AssertionError("Frequency axes differ")
        rest_values.append(current)
    return trial_tfr, np.asarray(rest_values), frequencies, times, len(trials)


def t_stat(values):
    return values.mean(0) / (values.std(0, ddof=1)/np.sqrt(len(values)))


def clusters(statistic, threshold):
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    found = []
    for sign in (1, -1):
        mask = statistic*sign > threshold
        labels, count = label(mask, structure)
        for index in range(1, count+1):
            selected = labels == index
            found.append((float(np.sum(statistic[selected]*sign)), sign, selected))
    return found


def cluster_permutation(differences, permutations, seed):
    # subject x ROI x frequency x time; correction across all ROIs.
    n = len(differences)
    threshold = float(student_t.ppf(.975, n-1))
    observed_t = t_stat(differences)
    observed = [clusters(observed_t[roi], threshold) for roi in range(len(ROIS))]
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for permutation in range(permutations):
        signs = rng.choice((-1., 1.), size=(n, 1, 1, 1))
        statistic = t_stat(differences*signs)
        masses = [item[0] for roi in statistic for item in clusters(roi, threshold)]
        null[permutation] = max(masses, default=0.)
    significant = np.zeros_like(observed_t, dtype=bool)
    records = []
    for roi_index, roi_clusters in enumerate(observed):
        for mass, sign, mask in roi_clusters:
            p = float((1 + np.sum(null >= mass))/(permutations+1))
            if p < .05:
                significant[roi_index] |= mask
                freq_idx, time_idx = np.where(mask)
                records.append({
                    "roi": list(ROIS)[roi_index], "sign": int(sign),
                    "mass": mass, "p": p,
                    "frequency_index_range": [int(freq_idx.min()), int(freq_idx.max())],
                    "time_index_range": [int(time_idx.min()), int(time_idx.max())],
                })
    return observed_t, significant, threshold, null, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--hop", type=float, default=.05)
    parser.add_argument("--permutations", type=int, default=2048)
    parser.add_argument("--subject-cache-root", type=Path, default=Path("data/interim/lee2019_rest_trial_stft_six_roi"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_rest_trial_stft_clusters_six_roi.png"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/lee2019_rest_trial_stft_clusters_six_roi.json"))
    args = parser.parse_args()
    epoch_seconds = 4.0

    trial_rows, rest_rows, counts = [], [], []
    frequencies = times = None
    for subject in args.subjects:
        cache = args.subject_cache_root/f"session{args.session}"/f"sub-{subject:02d}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                trial, rests = data["trial"], data["rests"]
                current_f, current_t = data["frequencies"], data["times"]
                count = int(data["trial_count"])
        else:
            trial, rests, current_f, current_t, count = subject_tfr(
                lee2019_path(args.data_root, subject, args.session),
                epoch_seconds, args.window, args.hop,
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, trial=trial, rests=rests,
                                frequencies=current_f, times=current_t,
                                trial_count=count)
        if frequencies is None:
            frequencies, times = current_f, current_t
        trial_rows.append(trial); rest_rows.append(rests); counts.append(count)
        print(f"stft session={args.session} subject={subject}", flush=True)
    trials = np.asarray(trial_rows)
    rests = np.asarray(rest_rows)  # subj x rest x ROI x freq x time
    pooled_rest = rests.mean(axis=1)
    differences = trials-pooled_rest
    t_values, significant, sample_threshold, null, records = cluster_permutation(
        differences, args.permutations, 20260801
    )
    for record in records:
        fi, ti = record.pop("frequency_index_range"), record.pop("time_index_range")
        record["frequency_hz"] = [float(frequencies[fi[0]]), float(frequencies[fi[1]])]
        record["time_s"] = [float(times[ti[0]]), float(times[ti[1]])]

    fig, axes = plt.subplots(len(ROIS), 3, figsize=(15, 3.2*len(ROIS)), constrained_layout=True)
    vmax = np.nanpercentile(np.abs(t_values), 99)
    for roi_index, roi_name in enumerate(ROIS):
        raw = differences[:, roi_index].mean(0)
        supra = np.where(np.abs(t_values[roi_index]) >= sample_threshold,
                         t_values[roi_index], np.nan)
        corrected = np.where(significant[roi_index], t_values[roi_index], np.nan)
        for column, (values, title) in enumerate((
            (raw, "(a) mean log-power difference"),
            (supra, "(b) sample threshold"),
            (corrected, "(c) significant clusters"),
        )):
            axis = axes[roi_index, column]
            image = axis.pcolormesh(times, frequencies, values, shading="auto",
                                    cmap="RdBu_r",
                                    vmin=(-vmax if column else None),
                                    vmax=(vmax if column else None))
            axis.axvline(0, color="black", ls="--", lw=.8)
            axis.set(title=f"{roi_name}: {title}", ylabel="Frequency (Hz)")
            if roi_index == len(ROIS)-1:
                axis.set_xlabel("Time from cue (s)")
            fig.colorbar(image, ax=axis, shrink=.75)
    fig.suptitle("Lee2019 trial minus pooled four-rest STFT; cluster permutation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "session": args.session, "subjects": len(args.subjects),
        "trials_total": int(np.sum(counts)), "frequency_hz": [1, 50],
        "epoch_s": [-2.5, 1.5], "reported_stft_center_s": [float(times[0]), float(times[-1])],
        "stft": {"window": "Hann", "window_seconds": args.window,
                 "hop_seconds": args.hop, "frequency_resolution_hz": 1.0},
        "rois": {name: list(channels) for name, channels in ROIS.items()},
        "contrast": "trial minus within-subject mean of four resting blocks",
        "cluster_test": {"permutations": args.permutations,
                         "sample_t_threshold": sample_threshold,
                         "correction_family": "six ROIs, time x frequency clusters"},
        "significant_clusters": records,
        "note": "STFT centers span -2 to +1 s because a 1-s window must fit inside the epoch.",
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
