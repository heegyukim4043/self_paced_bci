"""Prepare seam-safe Tier-0 features from original Cho2017 MATLAB files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.features import extract_log_power

EEG_CHANNELS = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)
NOISE_LABELS = (
    "eye_blinking",
    "eyeball_up_down",
    "eyeball_left_right",
    "jaw_clenching",
    "head_left_right",
)


def trial_features(
    data: np.ndarray,
    sfreq: float,
    n_trials: int,
    config: ExperimentConfig,
    excluded_trials: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    trial_samples = int(round(7.0 * sfreq))
    if data.shape[1] != n_trials * trial_samples:
        raise ValueError(
            f"Expected {n_trials}x{trial_samples} samples, got {data.shape[1]}"
        )
    values = []
    times = None
    for trial in range(n_trials):
        if trial in excluded_trials:
            continue
        segment = data[:64, trial * trial_samples : (trial + 1) * trial_samples].T
        stream = extract_log_power(
            segment,
            sfreq,
            EEG_CHANNELS,
            config,
            output_channels=config.selected_channels,
        )
        values.append(stream.log_power.astype(np.float32))
        if times is None:
            times = stream.times.astype(np.float32)
        elif not np.array_equal(times, stream.times.astype(np.float32)):
            raise AssertionError("Trial feature timestamps differ")
    assert times is not None
    return times, np.stack(values)


def bad_trials(data: np.void, class_index: int) -> set[int]:
    """Return zero-based trials rejected by either Cho2017 quality criterion."""
    bad = data["bad_trial_indices"][0, 0]
    rejected: set[int] = set()
    for field in ("bad_trial_idx_voltage", "bad_trial_idx_mi"):
        per_class = np.asarray(bad[field]).ravel()
        raw = np.asarray(per_class[class_index]).ravel()
        # MATLAB indices are one-based; empty uint8 arrays are common.
        rejected.update(int(value) - 1 for value in raw if int(value) > 0)
    return rejected


def cache_one(source: Path, target: Path) -> None:
    data = loadmat(
        source,
        squeeze_me=False,
        struct_as_record=True,
        verify_compressed_data_integrity=False,
    )["eeg"][0, 0]
    config = ExperimentConfig(native_sfreq=512.0)
    sfreq = float(np.asarray(data["srate"]).squeeze())
    n_trials = int(np.asarray(data["n_imagery_trials"]).squeeze())

    rest = np.asarray(data["rest"], dtype=np.float64)[:64].T
    rest_stream = extract_log_power(
        rest,
        sfreq,
        EEG_CHANNELS,
        config,
        output_channels=config.selected_channels,
    )
    rest_midpoint = rest.shape[0] / sfreq / 2.0
    calibration_mask = rest_stream.times < rest_midpoint
    evaluation_mask = ~calibration_mask

    noise_values = []
    noise_times = None
    for item in np.asarray(data["noise"]).ravel():
        segment = np.asarray(item, dtype=np.float64)[:64].T
        stream = extract_log_power(
            segment,
            sfreq,
            EEG_CHANNELS,
            config,
            output_channels=config.selected_channels,
        )
        noise_values.append(stream.log_power.astype(np.float32))
        if noise_times is None:
            noise_times = stream.times.astype(np.float32)
    assert noise_times is not None

    bad_left = bad_trials(data, 0)
    bad_right = bad_trials(data, 1)
    trial_times, left = trial_features(
        np.asarray(data["imagery_left"]), sfreq, n_trials, config, bad_left
    )
    right_times, right = trial_features(
        np.asarray(data["imagery_right"]), sfreq, n_trials, config, bad_right
    )
    if not np.array_equal(trial_times, right_times):
        raise AssertionError("Left/right trial timestamps differ")

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        subject=np.int16(int(source.stem[1:])),
        sfreq=np.float32(sfreq),
        channels=np.asarray(config.selected_channels),
        bands=np.asarray(rest_stream.bands),
        rest_calibration_features=rest_stream.log_power[calibration_mask].astype(np.float32),
        rest_evaluation_features=rest_stream.log_power[evaluation_mask].astype(np.float32),
        rest_evaluation_times=rest_stream.times[evaluation_mask].astype(np.float32),
        noise_times=noise_times,
        noise_features=np.stack(noise_values),
        # Labels follow the source order documented in the Cho2017 paper.
        noise_labels=np.asarray(NOISE_LABELS),
        trial_times=trial_times,
        trial_features=np.concatenate([left, right], axis=0),
        trial_labels=np.concatenate(
            [
                np.ones(len(left), dtype=np.int8),
                np.full(len(right), 2, dtype=np.int8),
            ]
        ),
        excluded_left_trials=np.asarray(sorted(bad_left), dtype=np.int16),
        excluded_right_trials=np.asarray(sorted(bad_right), dtype=np.int16),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/cho2017"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/interim/cho2017_tier0")
    )
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = args.subjects or [
        subject for subject in range(1, 53) if subject not in {29, 34}
    ]
    for subject in subjects:
        source = args.data_root / f"s{subject:02d}.mat"
        target = args.output_root / f"sub-{subject:02d}_tier0_features.npz"
        if not source.exists():
            print(f"missing {source}", flush=True)
            continue
        if target.exists() and not args.overwrite:
            print(f"exists {target}", flush=True)
            continue
        cache_one(source, target)
        print(f"cached {target}", flush=True)


if __name__ == "__main__":
    main()
