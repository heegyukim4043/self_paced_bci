from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat, whosmat

from .config import ExperimentConfig


@dataclass(frozen=True)
class LeeRun:
    eeg: np.ndarray
    cue_samples: np.ndarray
    labels: np.ndarray
    channels: tuple[str, ...]
    sfreq: float
    pre_rest: np.ndarray
    post_rest: np.ndarray


@dataclass(frozen=True)
class Interval:
    kind: str
    start: float
    stop: float
    trial: int | None = None
    label: int | None = None


def lee2019_path(root: Path, subject: int, session: int) -> Path:
    return (
        root
        / f"session{session}"
        / f"s{subject}"
        / f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )


def validate_mat_container(path: Path) -> None:
    variables = {name: (shape, kind) for name, shape, kind in whosmat(path)}
    required = {"EEG_MI_train", "EEG_MI_test"}
    missing = required.difference(variables)
    if missing:
        raise ValueError(f"{path}: missing MATLAB variables {sorted(missing)}")


def _as_channel_names(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(np.squeeze(value).item()) for value in np.ravel(values))


def load_lee2019_run(path: Path, phase: str = "train") -> LeeRun:
    if phase not in {"train", "test"}:
        raise ValueError("phase must be 'train' or 'test'")
    variable = f"EEG_MI_{phase}"
    mat = loadmat(
        path,
        variable_names=[variable],
        squeeze_me=True,
        struct_as_record=False,
        verify_compressed_data_integrity=False,
    )
    if variable not in mat:
        raise ValueError(f"{path}: missing {variable}")
    run = mat[variable]
    required = {"x", "t", "fs", "y_dec", "chan", "pre_rest", "post_rest"}
    missing = required.difference(run._fieldnames)
    if missing:
        raise ValueError(f"{path}:{variable}: missing fields {sorted(missing)}")

    eeg = np.asarray(run.x, dtype=np.float64)
    pre_rest = np.asarray(run.pre_rest, dtype=np.float64)
    post_rest = np.asarray(run.post_rest, dtype=np.float64)
    channels = _as_channel_names(run.chan)
    if eeg.ndim != 2 or eeg.shape[1] != len(channels):
        raise ValueError(f"{path}:{variable}: EEG/channel shape mismatch")
    return LeeRun(
        eeg=eeg,
        cue_samples=np.asarray(run.t, dtype=np.int64).reshape(-1),
        labels=np.asarray(run.y_dec, dtype=np.int64).reshape(-1),
        channels=channels,
        sfreq=float(run.fs),
        pre_rest=pre_rest,
        post_rest=post_rest,
    )


def channel_indices(channels: Iterable[str], requested: Iterable[str]) -> np.ndarray:
    lookup = {name.casefold(): index for index, name in enumerate(channels)}
    missing = [name for name in requested if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"Missing channels: {missing}")
    return np.asarray([lookup[name.casefold()] for name in requested], dtype=np.int64)


def build_intervals(run: LeeRun, config: ExperimentConfig) -> list[Interval]:
    cue_times = run.cue_samples / run.sfreq
    intervals: list[Interval] = []
    for trial, (cue, label) in enumerate(zip(cue_times, run.labels)):
        intervals.append(
            Interval(
                "fixation",
                cue + config.fixation_start_seconds,
                cue + config.fixation_stop_seconds,
                trial,
                int(label),
            )
        )
        intervals.append(
            Interval(
                "command",
                cue + config.mi_start_seconds,
                cue + config.mi_stop_seconds,
                trial,
                int(label),
            )
        )
        if trial + 1 < len(cue_times):
            iti_start = cue + config.task_stop_seconds + config.iti_guard_seconds
            next_fixation = cue_times[trial + 1] + config.fixation_start_seconds
            iti_stop = next_fixation - config.iti_guard_seconds
            if iti_stop > iti_start:
                kind = (
                    "idle_iti"
                    if iti_stop - iti_start <= config.max_task_iti_seconds
                    else "idle_break"
                )
                intervals.append(
                    Interval(kind, iti_start, iti_stop, trial, None)
                )

    duration = run.eeg.shape[0] / run.sfreq
    for interval in intervals:
        if interval.start < 0 or interval.stop > duration or interval.stop <= interval.start:
            raise ValueError(f"Invalid interval: {interval}")
    return intervals


def select_intervals(
    intervals: Iterable[Interval], kind: str
) -> list[tuple[float, float]]:
    return [(item.start, item.stop) for item in intervals if item.kind == kind]
