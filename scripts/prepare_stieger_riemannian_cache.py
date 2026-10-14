"""Build a seam-safe causal covariance cache from original Stieger trial cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.features import (
    _causal_band_signal,
    _causal_downsample,
    common_average_reference,
)
from uncued_mi.riemannian import window_covariances

WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.1
# Trial-cell sample zero corresponds to target time -2 s.  Feedback begins at
# target time +2 s, i.e. 4 s after the first sample in the stored cell.
CELL_FEEDBACK_START_SECONDS = 4.0


def feedback_window_ends(
    trial_length_s: float,
    sfreq: float,
    *,
    window_seconds: float = WINDOW_SECONDS,
    step_seconds: float = STEP_SECONDS,
    feedback_start_seconds: float = CELL_FEEDBACK_START_SECONDS,
) -> np.ndarray:
    """End samples whose complete support lies inside the feedback phase."""
    window = int(round(window_seconds * sfreq))
    step = int(round(step_seconds * sfreq))
    feedback_start = int(round(feedback_start_seconds * sfreq))
    feedback_stop = int(
        np.floor((feedback_start_seconds + float(trial_length_s)) * sfreq + 1e-9)
    )
    first_end = feedback_start + window
    if first_end > feedback_stop:
        return np.empty(0, dtype=np.int64)
    return np.arange(first_end, feedback_stop + 1, step, dtype=np.int64)


def window_cursor_speeds(
    position_x: np.ndarray,
    position_y: np.ndarray,
    native_sfreq: float,
    target_sfreq: float,
    end_samples: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    """Mean cursor speed over each target-clock window."""
    x = _causal_downsample(np.asarray(position_x, dtype=float), native_sfreq, target_sfreq)
    y = _causal_downsample(np.asarray(position_y, dtype=float), native_sfreq, target_sfreq)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    valid_x = np.isfinite(dx)
    valid_y = np.isfinite(dy)
    speed = np.hypot(
        np.where(valid_x, dx, 0.0),
        np.where(valid_y, dy, 0.0),
    ) * float(target_sfreq)
    speed[~(valid_x | valid_y)] = np.nan
    values = np.empty(len(end_samples), dtype=np.float32)
    for index, end in enumerate(end_samples):
        segment = speed[end - window_samples : end]
        finite = segment[np.isfinite(segment)]
        values[index] = np.mean(finite) if finite.size else np.nan
    return values


def selected_indices(labels: list[str], selected: tuple[str, ...]) -> np.ndarray:
    lookup = {label.casefold(): index for index, label in enumerate(labels)}
    missing = [channel for channel in selected if channel.casefold() not in lookup]
    if missing:
        raise ValueError(f"missing sensorimotor channels: {missing}")
    return np.asarray([lookup[channel.casefold()] for channel in selected], dtype=np.int64)


def cache_one(source: Path, target: Path, config: ExperimentConfig) -> dict:
    bci = loadmat(
        source,
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["BCI"],
    )["BCI"]
    data = np.atleast_1d(bci.data)
    trials = np.atleast_1d(bci.TrialData)
    position_x = np.atleast_1d(bci.positionx)
    position_y = np.atleast_1d(bci.positiony)
    labels = [str(value) for value in np.atleast_1d(bci.chaninfo.label)]
    indices = selected_indices(labels, config.sensorimotor_channels)
    native_sfreq = float(bci.SRATE)
    target_sfreq = float(config.target_sfreq)
    window_samples = int(round(WINDOW_SECONDS * target_sfreq))

    covariances: list[np.ndarray] = []
    cursor_speeds: list[np.ndarray] = []
    trial_index: list[np.ndarray] = []
    target_number: list[np.ndarray] = []
    task_number: list[np.ndarray] = []
    run_number: list[np.ndarray] = []
    window_end_from_target_s: list[np.ndarray] = []
    result: list[np.ndarray] = []
    forced_result: list[np.ndarray] = []
    omitted_short = 0
    omitted_artifact = 0
    seam_assertions = 0

    for cell_index, (raw, trial, px, py) in enumerate(
        zip(data, trials, position_x, position_y)
    ):
        if int(trial.artifact):
            omitted_artifact += 1
            continue
        raw = np.asarray(raw, dtype=np.float64).T
        referenced = common_average_reference(raw)
        filtered = _causal_downsample(
            _causal_band_signal(
                referenced[:, indices],
                native_sfreq,
                8.0,
                30.0,
                config.filter_order,
            ),
            native_sfreq,
            target_sfreq,
        )
        ends = feedback_window_ends(float(trial.triallength), target_sfreq)
        if not len(ends):
            omitted_short += 1
            continue

        # Cell starts at target time -2 s. Feedback begins at target time +2 s,
        # hence every feedback window starts >=4 s after filter initialization.
        starts = ends - window_samples
        feedback_start = int(round(CELL_FEEDBACK_START_SECONDS * target_sfreq))
        feedback_stop = int(
            np.floor(
                (CELL_FEEDBACK_START_SECONDS + float(trial.triallength))
                * target_sfreq
                + 1e-9
            )
        )
        assert np.all(starts >= feedback_start)
        assert np.all(ends <= feedback_stop)
        assert np.all(ends <= len(filtered))
        seam_assertions += 1

        count = len(ends)
        covariances.append(window_covariances(filtered, ends, window_samples))
        cursor_speeds.append(
            window_cursor_speeds(
                px, py, native_sfreq, target_sfreq, ends, window_samples
            )
        )
        trial_index.append(np.full(count, cell_index, dtype=np.int16))
        target_number.append(np.full(count, int(trial.targetnumber), dtype=np.int8))
        task_number.append(np.full(count, int(trial.tasknumber), dtype=np.int8))
        run_number.append(np.full(count, int(trial.runnumber), dtype=np.int8))
        window_end_from_target_s.append(
            # Convert cell-relative time back to target-relative time.
            ends.astype(np.float32) / target_sfreq - 2.0
        )
        result.append(np.full(count, float(trial.result), dtype=np.float32))
        forced_result.append(
            np.full(count, int(trial.forcedresult), dtype=np.int8)
        )

    arrays = {
        "covariances": np.concatenate(covariances).astype(np.float32),
        "cursor_speed": np.concatenate(cursor_speeds),
        "trial_index": np.concatenate(trial_index),
        "target_number": np.concatenate(target_number),
        "is_active": (np.concatenate(target_number) != 4).astype(np.int8),
        "task_number": np.concatenate(task_number),
        "run_number": np.concatenate(run_number),
        "window_end_from_target_s": np.concatenate(window_end_from_target_s),
        "result": np.concatenate(result),
        "forced_result": np.concatenate(forced_result),
    }
    unique_trials = np.unique(arrays["trial_index"])
    report = {
        "source": str(source.resolve()),
        "trial_cells_total": int(len(data)),
        "trial_cells_cached": int(len(unique_trials)),
        "trial_cells_omitted_shorter_than_window": int(omitted_short),
        "trial_cells_omitted_artifact": int(omitted_artifact),
        "windows_total": int(len(arrays["covariances"])),
        "active_windows": int(arrays["is_active"].sum()),
        "intentional_rest_windows": int((arrays["is_active"] == 0).sum()),
        "active_trials_cached": int(
            len(np.unique(arrays["trial_index"][arrays["is_active"] == 1]))
        ),
        "intentional_rest_trials_cached": int(
            len(np.unique(arrays["trial_index"][arrays["is_active"] == 0]))
        ),
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "native_sfreq": native_sfreq,
        "target_sfreq": target_sfreq,
        "filter": "CAR(all 62 EEG)->causal Butterworth 8-30Hz->250Hz",
        "filter_warmup_before_feedback_s": 4.0,
        "seam_support_assertions_passed": int(seam_assertions),
        "normalization": "none; covariance reference is fitted downstream",
        "negative_state": "intentional-rest feedback (target 4)",
        "pre_target_idle_excluded_reason": (
            "The cell starts only 2 s before target onset, so strict 2 s causal "
            "filter warm-up plus a complete 1 s window cannot fit before target."
        ),
        "sensorimotor_channels": list(config.sensorimotor_channels),
        "mbsr_subject": int(bci.metadata.MBSRsubject),
        "session_day": int(bci.metadata.day),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **arrays,
        channels=np.asarray(config.sensorimotor_channels),
        audit_json=np.asarray(json.dumps(report, sort_keys=True)),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/stieger2021/S1_Session_1_covariances.npz"),
    )
    args = parser.parse_args()
    report = cache_one(
        args.mat_file,
        args.output,
        ExperimentConfig(window_seconds=WINDOW_SECONDS),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
