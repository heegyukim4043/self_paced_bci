from __future__ import annotations

import argparse
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
    validate_mat_container,
)
from uncued_mi.features import extract_log_power


def interval_array(intervals, kind: str) -> np.ndarray:
    values = [(item.start, item.stop) for item in intervals if item.kind == kind]
    return np.asarray(values, dtype=np.float64).reshape(-1, 2)


def cache_one(
    data_root: Path,
    output_root: Path,
    subject: int,
    session: int,
    overwrite: bool,
) -> Path:
    config = ExperimentConfig()
    source = lee2019_path(data_root, subject, session)
    target = (
        output_root
        / f"session{session}"
        / f"sub-{subject:02d}_tier0_features.npz"
    )
    if target.exists() and not overwrite:
        return target

    validate_mat_container(source)
    run = load_lee2019_run(source, phase="train")
    indices = channel_indices(run.channels, config.selected_channels)
    selected_names = tuple(run.channels[index] for index in indices)
    pre = extract_log_power(
        run.pre_rest,
        run.sfreq,
        run.channels,
        config,
        output_channels=selected_names,
    )
    post = extract_log_power(
        run.post_rest,
        run.sfreq,
        run.channels,
        config,
        output_channels=selected_names,
    )
    continuous = extract_log_power(
        run.eeg,
        run.sfreq,
        run.channels,
        config,
        output_channels=selected_names,
    )
    intervals = build_intervals(run, config)

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        subject=np.int16(subject),
        session=np.int8(session),
        source_file=np.asarray(str(source)),
        channels=np.asarray(selected_names),
        bands=np.asarray(pre.bands),
        pre_times=pre.times.astype(np.float32),
        pre_features=pre.log_power.astype(np.float32),
        post_times=post.times.astype(np.float32),
        post_features=post.log_power.astype(np.float32),
        continuous_times=continuous.times.astype(np.float32),
        continuous_features=continuous.log_power.astype(np.float32),
        fixation_intervals=interval_array(intervals, "fixation"),
        command_intervals=interval_array(intervals, "command"),
        iti_intervals=interval_array(intervals, "idle_iti"),
        break_intervals=interval_array(intervals, "idle_break"),
        cue_samples=run.cue_samples,
        labels=run.labels.astype(np.int8),
        native_sfreq=np.float32(run.sfreq),
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--all", action="store_true", help="Process all 54 Lee2019 subjects"
    )
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = list(range(1, 55)) if args.all else args.subjects
    for session in args.sessions:
        for subject in subjects:
            source = lee2019_path(args.data_root, subject, session)
            if not source.exists():
                print(f"skip missing: {source}")
                continue
            target = cache_one(
                args.data_root,
                args.output_root,
                subject,
                session,
                args.overwrite,
            )
            print(f"cached: {target}")


if __name__ == "__main__":
    main()
