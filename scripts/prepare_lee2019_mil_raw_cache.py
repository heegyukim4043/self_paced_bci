"""Prepare continuous causal raw caches for the preregistered MIL experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import (
    build_intervals,
    channel_indices,
    lee2019_path,
    load_lee2019_run,
)
from uncued_mi.features import (
    _causal_band_signal,
    _causal_downsample,
    common_average_reference,
)

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
WINDOW_SECONDS = 1.0
INSTANCE_STEP_SECONDS = 0.1


def preprocess(
    eeg: np.ndarray,
    channels: tuple[str, ...],
    sfreq: float,
    selected: tuple[str, ...],
    config: ExperimentConfig,
) -> np.ndarray:
    referenced = common_average_reference(eeg)
    indices = channel_indices(channels, selected)
    filtered = _causal_band_signal(
        referenced[:, indices],
        sfreq,
        8.0,
        30.0,
        config.filter_order,
    )
    return _causal_downsample(filtered, sfreq, config.target_sfreq).astype(
        np.float32
    )


def interval_payload(intervals, kind: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [item for item in intervals if item.kind == kind]
    bounds = np.asarray(
        [(item.start, item.stop) for item in selected], dtype=np.float32
    ).reshape(-1, 2)
    trials = np.asarray(
        [item.trial if item.trial is not None else -1 for item in selected],
        dtype=np.int16,
    )
    return bounds, trials


def window_end_samples(
    sample_count: int,
    sfreq: float,
    intervals: np.ndarray,
    *,
    transient_seconds: float,
) -> list[np.ndarray]:
    window = int(round(WINDOW_SECONDS * sfreq))
    step = int(round(INSTANCE_STEP_SECONDS * sfreq))
    first_end = int(round(transient_seconds * sfreq)) + window
    global_ends = np.arange(first_end, sample_count + 1, step, dtype=np.int64)
    result = []
    for start, stop in np.asarray(intervals).reshape(-1, 2):
        starts = global_ends - window
        selected = global_ends[
            (starts / sfreq >= float(start) - 1e-9)
            & (global_ends / sfreq <= float(stop) + 1e-9)
        ]
        if len(selected):
            assert np.min((selected - window) / sfreq) >= float(start) - 1e-9
            assert np.max(selected / sfreq) <= float(stop) + 1e-9
        result.append(selected)
    return result


def flatten_with_trials(
    values: list[np.ndarray], trials: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.concatenate(values) if values else np.empty(0, dtype=np.int64)
    owners = np.concatenate(
        [np.full(len(item), trial, dtype=np.int16) for item, trial in zip(values, trials)]
    )
    return samples, owners


def audit_windows(
    signal: np.ndarray,
    pre: np.ndarray,
    command: np.ndarray,
    command_trials: np.ndarray,
    iti: np.ndarray,
    iti_trials: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict, dict[str, np.ndarray]]:
    positive = window_end_samples(
        len(signal),
        config.target_sfreq,
        command,
        transient_seconds=config.transient_seconds,
    )
    if len(positive) != 100:
        raise AssertionError(f"Expected 100 positive bags, got {len(positive)}")
    bag_sizes = np.asarray([len(item) for item in positive], dtype=np.int16)
    if not set(int(value) for value in np.unique(bag_sizes)).issubset({20, 21}):
        raise AssertionError(
            f"Expected 20-21 instances per positive bag, got {np.unique(bag_sizes)}"
        )
    positive_samples = np.full((len(positive), 21), -1, dtype=np.int64)
    for index, values in enumerate(positive):
        positive_samples[index, : len(values)] = values
    window = int(round(WINDOW_SECONDS * config.target_sfreq))
    for bounds, ends, size in zip(command, positive_samples, bag_sizes):
        ends = ends[:size]
        starts = (ends - window) / config.target_sfreq
        stops = ends / config.target_sfreq
        if np.any(starts < bounds[0] - 1e-9) or np.any(stops > bounds[1] + 1e-9):
            raise AssertionError("Positive window support escaped command bounds")

    iti_values = window_end_samples(
        len(signal),
        config.target_sfreq,
        iti,
        transient_seconds=config.transient_seconds,
    )
    iti_samples, iti_instance_trials = flatten_with_trials(iti_values, iti_trials)
    full_pre = np.asarray([[0.0, len(pre) / config.target_sfreq]], dtype=float)
    pre_samples = window_end_samples(
        len(pre),
        config.target_sfreq,
        full_pre,
        transient_seconds=config.transient_seconds,
    )[0]
    split = int(np.floor(0.8 * len(pre_samples)))
    pre_fit = pre_samples[:split]
    pre_validation = pre_samples[split:]

    def count_trials(low: int, high: int) -> int:
        return int(np.sum((iti_instance_trials >= low) & (iti_instance_trials < high)))

    audit = {
        "window_seconds": WINDOW_SECONDS,
        "instance_step_seconds": INSTANCE_STEP_SECONDS,
        "positive_bags": {
            "fit_trials_0_39": int(np.sum((command_trials >= 0) & (command_trials < 40))),
            "validation_trials_40_49": int(
                np.sum((command_trials >= 40) & (command_trials < 50))
            ),
            "locked_trials_50_99": int(
                np.sum((command_trials >= 50) & (command_trials < 100))
            ),
            "instances_per_bag": sorted(set(int(value) for value in bag_sizes)),
        },
        "negative_instances": {
            "pre_rest_fit": len(pre_fit),
            "pre_rest_validation": len(pre_validation),
            "iti_fit_trials_0_39": count_trials(0, 40),
            "iti_validation_trials_40_49": count_trials(40, 50),
            "iti_locked_trials_50_99": count_trials(50, 100),
        },
        "causal_assertions": {
            "continuous_filter_before_windowing": True,
            "normalization_source": "current-session pre-rest only",
            "positive_support_inside_cue_plus_0p5_to_3p5": True,
            "filter_transient_seconds": config.transient_seconds,
        },
    }
    arrays = {
        "positive_end_samples": positive_samples,
        "positive_bag_sizes": bag_sizes,
        "iti_end_samples": iti_samples,
        "iti_instance_trials": iti_instance_trials,
        "pre_rest_fit_end_samples": pre_fit,
        "pre_rest_validation_end_samples": pre_validation,
    }
    return audit, arrays


def cache_one(source: Path, target: Path, config: ExperimentConfig) -> None:
    run = load_lee2019_run(source, "train")
    intervals = build_intervals(run, config)
    signal = preprocess(
        run.eeg,
        run.channels,
        run.sfreq,
        config.sensorimotor_channels,
        config,
    )
    pre = preprocess(
        run.pre_rest,
        run.channels,
        run.sfreq,
        config.sensorimotor_channels,
        config,
    )
    transient = int(round(config.transient_seconds * config.target_sfreq))
    calibration = pre[transient:]
    mean = calibration.mean(axis=0)
    std = np.maximum(calibration.std(axis=0, ddof=1), 1e-6)
    signal = ((signal - mean) / std).astype(np.float32)
    pre = ((pre - mean) / std).astype(np.float32)

    command, command_trials = interval_payload(intervals, "command")
    iti, iti_trials = interval_payload(intervals, "idle_iti")
    audit, sample_arrays = audit_windows(
        signal,
        pre,
        command,
        command_trials,
        iti,
        iti_trials,
        config,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        signal=signal,
        pre_rest=pre,
        sfreq=np.float32(config.target_sfreq),
        channels=np.asarray(config.sensorimotor_channels),
        command_intervals=command,
        command_trials=command_trials,
        iti_intervals=iti,
        iti_trials=iti_trials,
        labels=run.labels.astype(np.int8),
        preprocessing=np.asarray(
            "CAR(all EEG)->causal Butterworth 8-30Hz->250Hz->pre-rest z-score"
        ),
        audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
        **sample_arrays,
    )
    print(json.dumps(audit, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/lee2019_mi")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/lee2019_mil_raw"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(window_seconds=1.0)
    for session in args.sessions:
        for subject in args.subjects:
            source = lee2019_path(args.data_root, subject, session)
            target = (
                args.output_root
                / f"session{session}"
                / f"sub-{subject:02d}_mil_raw.npz"
            )
            if target.exists() and not args.overwrite:
                print(f"exists {target}", flush=True)
                continue
            cache_one(source, target, config)
            print(f"cached {target}", flush=True)


if __name__ == "__main__":
    main()
