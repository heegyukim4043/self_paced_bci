"""Build source covariances in the same normalized coordinates as A0 replay."""

import argparse
import json
from pathlib import Path

import numpy as np


WINDOW = 250
STEP = 25
TRANSIENT = 500


def covariances_at(signal, ends):
    ends = np.asarray(ends, dtype=np.int64)
    if np.any(ends < WINDOW) or np.any(ends > len(signal)):
        raise ValueError("covariance window exceeds signal")
    return np.asarray([np.cov(signal[end - WINDOW:end], rowvar=False)
                       for end in ends], dtype=np.float32)


def build_one(source, target):
    with np.load(source, allow_pickle=False) as data:
        signal = data["signal"]
        pre = data["pre_rest"]
        command_parts = [np.arange(
            int(round((float(start) + 1.0) * 250)),
            int(round(float(stop) * 250)), STEP,
        ) for start, stop in data["command_intervals"][:50]]
        command_ends = np.concatenate(command_parts)
        command_trials = np.repeat(np.arange(50, dtype=np.int16),
                                   [len(part) for part in command_parts])
        idle_mask = data["iti_instance_trials"] < 50
        idle_ends = data["iti_end_samples"][idle_mask]
        idle_trials = data["iti_instance_trials"][idle_mask]
        channels = data["channels"]
    pre_ends = np.arange(TRANSIENT + WINDOW, len(pre) + 1, STEP)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        train_command_covariance=covariances_at(signal, command_ends),
        train_command_trial=command_trials,
        train_idle_iti_covariance=covariances_at(signal, idle_ends),
        train_idle_iti_trial=idle_trials,
        pre_train_covariance=covariances_at(pre, pre_ends),
        channels=channels,
        metadata_json=np.asarray(json.dumps({
            "source": str(source), "window_samples": WINDOW, "step_samples": STEP,
            "anchor": "pre_train 250-Hz samples [500,15000); per-channel z-score",
            "source_trial_ids": [0, 49], "coordinate_system": "normalized mil_raw/revalidation replay",
            "positive_window_rule": "same interval-to-end rule as CNN-GRU source training; stop exclusive",
        }, sort_keys=True)),
    )
    return len(command_ends), len(idle_ends), len(pre_ends)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/interim/lee2019_revalidation_norm_cov"))
    parser.add_argument("--subjects", type=int, nargs="+", default=[3])
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for session in args.sessions:
        for subject in args.subjects:
            name = f"sub-{subject:02d}"
            source = args.input_root / f"session{session}" / f"{name}_mil_raw.npz"
            target = args.output_root / f"session{session}" / f"{name}.npz"
            if target.exists() and not args.overwrite:
                print(f"exists {target}", flush=True)
                continue
            command, idle, pre = build_one(source, target)
            print(json.dumps({"session": session, "subject": subject,
                              "command": command, "idle": idle, "pre": pre}), flush=True)


if __name__ == "__main__":
    main()
