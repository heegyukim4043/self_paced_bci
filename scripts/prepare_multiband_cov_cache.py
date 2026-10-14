"""Create compact whitened three-band covariance caches from Lee2019 raw EEG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import channel_indices, lee2019_path, load_lee2019_run
from uncued_mi.features import _causal_band_signal, _causal_downsample, common_average_reference

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
BANDS = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
SFREQ, WINDOW, STEP = 250, 250, 125


def filtered_bands(eeg, channels, native_sfreq, selected):
    referenced = common_average_reference(eeg)[:, channel_indices(channels, selected)]
    return [
        _causal_downsample(
            _causal_band_signal(referenced, native_sfreq, low, high, 4),
            native_sfreq, SFREQ,
        ).astype(np.float32)
        for low, high in BANDS
    ]


def covariances(bands, ends):
    output = np.empty((len(ends), len(bands), bands[0].shape[1], bands[0].shape[1]), np.float32)
    for index, end in enumerate(ends):
        for band, values in enumerate(bands):
            segment = values[end - WINDOW:end]
            segment = segment - segment.mean(axis=0, keepdims=True)
            output[index, band] = segment.T @ segment / (WINDOW - 1)
    return output


def invsqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 1e-8) ** -0.5) @ vectors.T


def whiten(values, transforms):
    return np.einsum("bij,nbjk,bkl->nbil", transforms, values, transforms).astype(np.float32)


def cache_one(raw_path, mil_path, output):
    run = load_lee2019_run(raw_path, "train")
    selected = ExperimentConfig().sensorimotor_channels
    signal_bands = filtered_bands(run.eeg, run.channels, run.sfreq, selected)
    pre_bands = filtered_bands(run.pre_rest, run.channels, run.sfreq, selected)
    with np.load(mil_path) as data:
        positive = []
        for trial in range(50):
            size = int(data["positive_bag_sizes"][trial])
            positive.extend(data["positive_end_samples"][trial, :size])
        iti_mask = data["iti_instance_trials"] < 50
        iti_ends = data["iti_end_samples"][iti_mask]
        pre_ends = np.r_[data["pre_rest_fit_end_samples"], data["pre_rest_validation_end_samples"]]
        stream_ends = np.arange(3 * SFREQ, len(signal_bands[0]) + 1, STEP)
        intervals = {
            "iti_intervals": data["iti_intervals"],
            "iti_trials": data["iti_trials"],
            "command_intervals": data["command_intervals"],
        }
    pre_cov = covariances(pre_bands, pre_ends)
    transforms = np.stack([invsqrt(pre_cov[:, band].mean(axis=0)) for band in range(3)])
    payload = {
        "positive": whiten(covariances(signal_bands, np.asarray(positive)), transforms),
        "iti": whiten(covariances(signal_bands, iti_ends), transforms),
        "rest": whiten(pre_cov, transforms),
        "stream": whiten(covariances(signal_bands, stream_ends), transforms),
        "stream_ends": stream_ends,
        **intervals,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--mil-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/interim/multiband_cov"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    for session in args.sessions:
        for subject in args.subjects:
            output = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not output.exists():
                cache_one(
                    lee2019_path(args.data_root, subject, session),
                    args.mil_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz",
                    output,
                )
            print(f"cached session={session} subject={subject}", flush=True)


if __name__ == "__main__":
    main()
