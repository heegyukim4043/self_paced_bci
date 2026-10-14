"""Compare all four rest blocks with the extended -2..0 s pre-cue period."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-rest-precue-")
import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.data import lee2019_path, load_lee2019_run

BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
RESTS = ("pre_train", "post_train", "pre_test", "post_test")
CENTRAL = ("C3", "Cz", "C4", "CP3", "CP4")
GATE_CHANNELS = ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CP4")


def resample_car(values, original_fs, indices, target_fs=250.0):
    eeg = values.T.astype(np.float64)
    eeg -= eeg.mean(axis=0, keepdims=True)
    eeg = eeg[np.asarray(indices)]
    if original_fs != target_fs:
        down = int(round(original_fs / target_fs))
        eeg = resample_poly(eeg, 1, down, axis=-1)
    return eeg


def log_power_windows(filtered, centers, half):
    squared = filtered * filtered
    cumulative = np.pad(np.cumsum(squared, axis=-1), ((0, 0), (1, 0)))
    output = []
    for center in centers:
        output.append(np.log(np.maximum(
            (cumulative[:, center + half] - cumulative[:, center - half]) / (2 * half),
            1e-20,
        )))
    return np.stack(output, axis=-1)


def subject_values(path, relative_centers, window_seconds):
    train = load_lee2019_run(path, "train")
    test = load_lee2019_run(path, "test")
    lookup = {name.casefold(): index for index, name in enumerate(train.channels)}
    selected_indices = [lookup[name.casefold()] for name in GATE_CHANNELS]
    fs = 250.0
    down = int(round(train.sfreq / fs))
    cues = np.rint(np.asarray(train.cue_samples) / down).astype(int)
    half = int(round(window_seconds * fs / 2))
    offsets = np.rint(relative_centers * fs).astype(int)
    train_eeg = resample_car(train.eeg, train.sfreq, selected_indices, fs)
    rest_raw = {
        "pre_train": resample_car(train.pre_rest, train.sfreq, selected_indices, fs),
        "post_train": resample_car(train.post_rest, train.sfreq, selected_indices, fs),
        "pre_test": resample_car(test.pre_rest, test.sfreq, selected_indices, fs),
        "post_test": resample_car(test.post_rest, test.sfreq, selected_indices, fs),
    }
    pre_by_band, rest_by_band = [], []
    for low, high in BANDS.values():
        sos = butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, train_eeg, axis=-1)
        trials = []
        for cue in cues:
            centers = cue + offsets
            trials.append(log_power_windows(filtered, centers, half))
        pre_by_band.append(np.mean(trials, axis=0))  # channel x time
        current_rests = []
        for name in RESTS:
            rest_filtered = sosfiltfilt(sos, rest_raw[name], axis=-1)
            # Exclude 2 seconds at both ends and use 100-ms-spaced windows.
            centers = np.arange(int(2 * fs) + half, rest_filtered.shape[-1] - int(2 * fs) - half, int(.1 * fs))
            current_rests.append(log_power_windows(rest_filtered, centers, half).mean(axis=-1))
        rest_by_band.append(np.stack(current_rests))  # rest x channel
    return np.stack(pre_by_band), np.stack(rest_by_band), GATE_CHANNELS, len(cues)


def max_t(differences, permutations, seed):
    # subject x band x rest x channel x time
    n = differences.shape[0]
    observed = differences.mean(0) / (differences.std(0, ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    maxima = np.empty(permutations)
    for index in range(permutations):
        signs = rng.choice((-1., 1.), size=(n, 1, 1, 1, 1))
        sample = differences * signs
        statistic = sample.mean(0) / (sample.std(0, ddof=1) / np.sqrt(n))
        maxima[index] = np.nanmax(np.abs(statistic))
    threshold = float(np.quantile(maxima, .95))
    return observed, np.abs(observed) >= threshold, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--window", type=float, default=.2)
    parser.add_argument("--permutations", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_four_rest_vs_precue_mu_beta.png"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/lee2019_four_rest_vs_precue_mu_beta.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/lee2019_four_rest_vs_precue_mu_beta_subjects.npz"))
    parser.add_argument("--subject-cache-root", type=Path,
                        default=Path("data/interim/lee2019_rest_precue_bandpower"))
    args = parser.parse_args()

    relative = np.arange(-1.9, -.099, .1)
    pre_rows, rest_rows, counts = [], [], []
    channels = None
    for subject in args.subjects:
        subject_cache = args.subject_cache_root / f"session{args.session}" / f"sub-{subject:02d}.npz"
        if subject_cache.exists():
            with np.load(subject_cache, allow_pickle=False) as cached:
                pre, rests = cached["pre"], cached["rests"]
                current_channels = tuple(cached["channels"].tolist())
                count = int(cached["trial_count"])
        else:
            pre, rests, current_channels, count = subject_values(
                lee2019_path(args.data_root, subject, args.session), relative, args.window
            )
            subject_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(subject_cache, pre=pre, rests=rests,
                                channels=np.asarray(current_channels), trial_count=count)
        if channels is None:
            channels = current_channels
        if current_channels != channels:
            raise ValueError("Channel order differs")
        pre_rows.append(pre)
        rest_rows.append(rests)
        counts.append(count)
        print(f"rest-precue session={args.session} subject={subject}", flush=True)
    pre_values = np.asarray(pre_rows)  # subj x band x channel x time
    rest_values = np.asarray(rest_rows)  # subj x band x rest x channel
    differences = pre_values[:, :, None] - rest_values[..., None]
    t_values, significant, threshold = max_t(differences, args.permutations, 20260731)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, pre=pre_values, rest_values=rest_values,
                        differences=differences, t_values=t_values,
                        significant=significant, relative_times=relative,
                        channels=np.asarray(channels), rests=np.asarray(RESTS))

    central = [channels.index(name) for name in CENTRAL]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    colors = ("#2563eb", "#16a34a", "#f59e0b", "#9333ea")
    payload_bands = {}
    for band_index, (band_name, hz) in enumerate(BANDS.items()):
        axis = axes[band_index, 0]
        band_payload = {}
        for rest_index, (rest_name, color) in enumerate(zip(RESTS, colors)):
            subject_delta = differences[:, band_index, rest_index, central].mean(axis=1)
            mean = subject_delta.mean(0)
            sem = subject_delta.std(0, ddof=1) / np.sqrt(len(subject_delta))
            axis.plot(relative, mean, color=color, label=rest_name)
            axis.fill_between(relative, mean-sem, mean+sem, color=color, alpha=.12)
            sig = significant[band_index, rest_index, central].any(axis=0)
            axis.scatter(relative[sig], np.full(sig.sum(), -.38 + rest_index*.018),
                         marker="|", color=color, s=35)
            nearest_index = -1
            far_index = 4  # -1.5 s
            band_payload[rest_name] = {
                "central_delta_at_minus_1_5s": float(mean[far_index]),
                "central_delta_at_minus_0_1s": float(mean[nearest_index]),
                "significant_central_times_s": relative[sig].round(2).tolist(),
                "minimum_absolute_central_delta": float(np.min(np.abs(mean))),
                "time_of_minimum_absolute_delta_s": float(relative[np.argmin(np.abs(mean))]),
            }
        axis.axhline(0, color="black", lw=.8)
        axis.axvline(-.5, color="0.5", ls="--", lw=.8)
        axis.set(title=f"{band_name} {hz[0]:g}-{hz[1]:g} Hz: pre-cue minus rest",
                 xlabel="Time from cue (s)", ylabel="Δ log power", xlim=(-1.95, -.05))
        axis.legend(frameon=False, ncol=2)

        # Pooled four-rest gate-channel values at -1.5 and -0.1 s.
        topo_axis = axes[band_index, 1]
        pooled = differences[:, band_index].mean(axis=1)  # subj x chan x time
        chosen = [4, -1]
        inset_values = []
        for pos, idx in enumerate(chosen):
            inset = topo_axis.inset_axes([pos*.5, 0, .48, 1])
            values = pooled[:, :, idx].mean(0)
            inset_values.append(values)
            inset.bar(np.arange(len(channels)), values,
                      color=np.where(values < 0, "#2563eb", "#dc2626"))
            inset.axhline(0, color="black", lw=.8)
            inset.set_xticks(np.arange(len(channels)), channels, rotation=45)
            inset.set_title(f"{relative[idx]:.1f} s")
        topo_axis.set_axis_off()
        topo_axis.set_title("Pooled four-rest Δ log-power topography")
        payload_bands[band_name] = band_payload
    fig.suptitle(
        f"Lee2019: four resting blocks vs extended pre-cue fixation "
        f"(N={len(args.subjects)}, 200-ms windows; max-T FWER .05)"
    )
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "session": args.session, "subjects": len(args.subjects),
        "trials_total": int(np.sum(counts)), "rests": list(RESTS),
        "pre_cue_centers_s": relative.round(2).tolist(),
        "window_seconds": args.window,
        "statistic": {
            "method": "subject-level paired sign-flip max-|t|",
            "permutations": args.permutations,
            "family": "2 bands x 4 rests x 7 gate channels x 19 times",
            "fwer_threshold": threshold,
        },
        "bands": payload_bands,
        "decision_note": (
            "Non-significance is not equivalence; use effect size together with "
            "the corrected test when choosing a rest-to-trial seam."
        ),
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
