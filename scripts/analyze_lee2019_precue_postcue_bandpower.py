"""Paired pre/post-cue mu and beta log-bandpower analysis for Lee2019."""

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
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-bandpower-")
import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.data import lee2019_path, load_lee2019_run

BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}


def subject_difference(path: Path, centers: np.ndarray, window_seconds: float):
    run = load_lee2019_run(path, "train")
    original_fs = float(run.sfreq)
    eeg = run.eeg.T.astype(np.float64)
    eeg -= eeg.mean(axis=0, keepdims=True)  # continuous CAR
    target_fs = 250.0
    if original_fs != target_fs:
        down = int(round(original_fs / target_fs))
        if not np.isclose(original_fs / down, target_fs):
            raise ValueError(f"Unsupported sampling rate {original_fs}")
        eeg = resample_poly(eeg, up=1, down=down, axis=-1)
        cues = np.rint(np.asarray(run.cue_samples) / down).astype(int)
    else:
        cues = np.asarray(run.cue_samples, dtype=int)
    fs = target_fs
    half = int(round(window_seconds * fs / 2))
    offsets = np.rint(centers * fs).astype(int)
    result = []
    for low, high in BANDS.values():
        sos = butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, eeg, axis=-1)
        squared = filtered * filtered
        cumulative = np.pad(
            np.cumsum(squared, axis=-1), ((0, 0), (1, 0)), mode="constant"
        )
        trial_differences = []
        for cue in cues:
            post, pre = [], []
            valid = True
            for offset in offsets:
                post_center = int(cue) + offset
                pre_center = int(cue) - offset
                if pre_center - half < 0 or post_center + half > eeg.shape[-1]:
                    valid = False
                    break
                post_power = (
                    cumulative[:, post_center + half]
                    - cumulative[:, post_center - half]
                ) / (2 * half)
                pre_power = (
                    cumulative[:, pre_center + half]
                    - cumulative[:, pre_center - half]
                ) / (2 * half)
                post.append(np.log(np.maximum(post_power, 1e-20)))
                pre.append(np.log(np.maximum(pre_power, 1e-20)))
            if valid:
                trial_differences.append(np.stack(post) - np.stack(pre))
        # Equal weight per trial within subject; output channel x time.
        result.append(np.mean(trial_differences, axis=0).T)
    return np.stack(result), tuple(run.channels), len(trial_differences)


def max_t_signflip(differences: np.ndarray, permutations: int, seed: int):
    n = differences.shape[0]
    observed = differences.mean(axis=0) / (
        differences.std(axis=0, ddof=1) / np.sqrt(n)
    )
    rng = np.random.default_rng(seed)
    maxima = np.empty(permutations)
    for index in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(n, 1, 1, 1))
        sample = differences * signs
        statistic = sample.mean(axis=0) / (
            sample.std(axis=0, ddof=1) / np.sqrt(n)
        )
        maxima[index] = np.nanmax(np.abs(statistic))
    threshold = float(np.quantile(maxima, 0.95))
    return observed, np.abs(observed) >= threshold, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--window", type=float, default=0.2)
    parser.add_argument("--permutations", type=int, default=4096)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/processed/lee2019_precue_postcue_bandpower_subjects.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_precue_postcue_mu_beta.png"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("data/processed/lee2019_precue_postcue_mu_beta.json"),
    )
    args = parser.parse_args()

    # A 200-ms window remains wholly on one side of the cue at center >=100 ms.
    centers = np.arange(args.window / 2, 0.401, 0.02)
    rows, counts = [], []
    channels = None
    for subject in args.subjects:
        values, current_channels, count = subject_difference(
            lee2019_path(args.data_root, subject, args.session),
            centers, args.window,
        )
        if channels is None:
            channels = current_channels
        if current_channels != channels:
            raise ValueError("Channel order differs")
        rows.append(values)
        counts.append(count)
        print(f"bandpower session={args.session} subject={subject}", flush=True)
    differences = np.asarray(rows)  # subject x band x channel x time
    t_values, significant, threshold = max_t_signflip(
        differences, args.permutations, 20260730
    )

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache, differences=differences, t_values=t_values,
        significant=significant, centers=centers,
        channels=np.asarray(channels), bands=np.asarray(tuple(BANDS)),
        trial_counts=np.asarray(counts),
    )

    info = mne.create_info(list(channels), 250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    selected_ms = (100, 200, 300, 400)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    colors = ("#2563eb", "#dc2626")
    central_names = ("C3", "Cz", "C4", "CP3", "CPz", "CP4")
    central = [channels.index(name) for name in central_names if name in channels]
    summaries = {}
    for band_index, (band_name, band_hz) in enumerate(BANDS.items()):
        subject_trace = differences[:, band_index, central].mean(axis=1)
        trace = subject_trace.mean(axis=0)
        sem = subject_trace.std(axis=0, ddof=1) / np.sqrt(len(subject_trace))
        axis = axes[band_index, 0]
        axis.plot(centers * 1000, trace, color=colors[band_index], lw=2)
        axis.fill_between(
            centers * 1000, trace - sem, trace + sem,
            color=colors[band_index], alpha=0.2,
        )
        central_sig = significant[band_index, central].any(axis=0)
        axis.scatter(
            centers[central_sig] * 1000,
            np.full(central_sig.sum(), axis.get_ylim()[0]),
            marker="|", color="black", s=35,
        )
        axis.axhline(0, color="0.4", lw=0.8)
        axis.set(
            title=f"{band_name} {band_hz[0]:g}-{band_hz[1]:g} Hz: post − pre",
            xlabel="Matched distance from cue (ms)", ylabel="Δ log power",
        )
        topo_indices = [int(np.argmin(np.abs(centers * 1000 - ms))) for ms in selected_ms]
        vmax = max(np.max(np.abs(t_values[band_index, :, idx])) for idx in topo_indices)
        sensor_counts = {}
        for column, (ms, time_index) in enumerate(zip(selected_ms, topo_indices), 1):
            topo_axis = axes[band_index, column]
            mask = significant[band_index, :, time_index]
            mne.viz.plot_topomap(
                t_values[band_index, :, time_index], info, axes=topo_axis,
                show=False, contours=4, cmap="RdBu_r", vlim=(-vmax, vmax),
                sensors=True, mask=mask,
                mask_params=dict(marker="o", markerfacecolor="yellow",
                                 markeredgecolor="black", markersize=4),
            )
            topo_axis.set_title(f"{ms} ms (sig={int(mask.sum())})")
            sensor_counts[str(ms)] = int(mask.sum())
        summaries[band_name] = {
            "band_hz": list(band_hz),
            "central_mean_delta_log_power": {
                str(ms): float(trace[idx])
                for ms, idx in zip(selected_ms, topo_indices)
            },
            "significant_sensor_count": sensor_counts,
            "subjects_with_positive_central_mean_at_300ms": int(
                np.sum(subject_trace[:, topo_indices[2]] > 0)
            ),
        }
    fig.suptitle(
        f"Lee2019 session {args.session}: matched post-cue vs pre-cue bandpower "
        f"(N={len(args.subjects)}, {int(args.window*1000)}-ms windows; max-T FWER .05)"
    )
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "session": args.session,
        "subjects": len(args.subjects),
        "trials_total": int(np.sum(counts)),
        "comparison": "log-power(post +t) minus log-power(pre -t)",
        "window_seconds": args.window,
        "centers_ms": (centers * 1000).round().astype(int).tolist(),
        "processing": {
            "reference": "continuous CAR",
            "resampling_hz": 250,
            "filter": "4th-order zero-phase Butterworth, separately per band",
            "statistic": "paired subject-level sign-flip max-|t|",
            "permutations": args.permutations,
            "family": "2 bands x 62 channels x all time points",
            "max_t_fwer_threshold": threshold,
        },
        "bands": summaries,
        "interpretation_limit": (
            "Descriptive zero-phase bandpower audit, not a causal online estimate."
        ),
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
