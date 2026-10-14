"""Smoke-audit one original Stieger2021 subject-session MAT file.

This deliberately works on the original trial cells.  It never concatenates
cells, so the report preserves seams and the meaning of each within-cell phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.stats import ks_2samp

from uncued_mi.config import ExperimentConfig


def phase_masks(time_ms: np.ndarray, trial_length_s: float) -> dict[str, np.ndarray]:
    """Return masks for the documented within-cell phases."""
    time_s = np.asarray(time_ms, dtype=float) / 1000.0
    feedback_stop = 2.0 + float(trial_length_s)
    return {
        "pre_target_idle": (time_s >= -2.0) & (time_s < 0.0),
        "target_only": (time_s >= 0.0) & (time_s < 2.0),
        "feedback": (time_s >= 2.0) & (time_s < feedback_stop),
        "post_feedback": (time_s >= feedback_stop),
    }


def feedback_speed(
    position_x: np.ndarray,
    position_y: np.ndarray,
    feedback_mask: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Cursor speed during feedback in native position units per second."""
    x = np.asarray(position_x, dtype=float).reshape(-1)
    y = np.asarray(position_y, dtype=float).reshape(-1)
    if x.size != y.size or x.size != feedback_mask.size:
        raise ValueError("cursor/time arrays must have identical lengths")
    dx = np.diff(x)
    dy = np.diff(y)
    valid_x = np.isfinite(dx)
    valid_y = np.isfinite(dy)
    # In 1-D tasks the unused cursor axis is stored as NaN.  Retain the
    # observed component; require both axes only for neither.
    speed = np.hypot(
        np.where(valid_x, dx, 0.0),
        np.where(valid_y, dy, 0.0),
    ) * float(sfreq)
    valid = (
        feedback_mask[1:]
        & feedback_mask[:-1]
        & (valid_x | valid_y)
    )
    return speed[valid]


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: None for key in ("q05", "q25", "median", "q75", "q95")}
    points = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(zip(("q05", "q25", "median", "q75", "q95"), map(float, points)))


def robust_range_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection-over-union of the two empirical 5--95% ranges."""
    a = np.quantile(np.asarray(first, dtype=float), [0.05, 0.95])
    b = np.quantile(np.asarray(second, dtype=float), [0.05, 0.95])
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(intersection / union) if union > 0 else float(a[0] == b[0])


def _field(struct: Any, name: str) -> Any:
    if not hasattr(struct, name):
        raise ValueError(f"missing required field: {name}")
    return getattr(struct, name)


def audit(mat_file: Path) -> dict[str, Any]:
    bci = loadmat(
        mat_file,
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["BCI"],
    )["BCI"]
    required = {
        "data", "time", "positionx", "positiony", "SRATE",
        "TrialData", "metadata", "chaninfo",
    }
    missing = sorted(required.difference(bci._fieldnames))
    if missing:
        raise ValueError(f"BCI is missing fields: {missing}")

    data = np.atleast_1d(bci.data)
    times = np.atleast_1d(bci.time)
    position_x = np.atleast_1d(bci.positionx)
    position_y = np.atleast_1d(bci.positiony)
    trials = np.atleast_1d(bci.TrialData)
    sfreq = float(bci.SRATE)
    if not (len(data) == len(times) == len(position_x) == len(position_y) == len(trials)):
        raise ValueError("trial-cell fields have inconsistent counts")

    labels = [str(label) for label in np.atleast_1d(bci.chaninfo.label)]
    upper_labels = {label.upper() for label in labels}
    selected = ExperimentConfig().sensorimotor_channels
    selected_present = [
        channel for channel in selected if channel.upper() in upper_labels
    ]
    eog_labels = [
        label for label in labels
        if "EOG" in label.upper() or "HEO" in label.upper() or "VEO" in label.upper()
    ]

    phase_seconds = {
        "pre_target_idle": 0.0,
        "target_only": 0.0,
        "feedback": 0.0,
        "post_feedback": 0.0,
    }
    cell_lengths_s: list[float] = []
    feedback_lengths_s: list[float] = []
    target_numbers: list[int] = []
    run_numbers: list[int] = []
    task_numbers: list[int] = []
    artifacts: list[int] = []
    active_speeds: list[np.ndarray] = []
    rest_speeds: list[np.ndarray] = []
    cell_audits: list[dict[str, Any]] = []

    for index, (eeg, time, px, py, trial) in enumerate(
        zip(data, times, position_x, position_y, trials)
    ):
        eeg = np.asarray(eeg)
        time = np.asarray(time, dtype=float).reshape(-1)
        if eeg.ndim != 2 or eeg.shape[0] != len(labels):
            raise ValueError(f"trial {index}: unexpected EEG shape {eeg.shape}")
        if eeg.shape[1] != time.size:
            raise ValueError(f"trial {index}: EEG/time length mismatch")

        trial_length = float(_field(trial, "triallength"))
        target = int(_field(trial, "targetnumber"))
        masks = phase_masks(time, trial_length)
        for phase, mask in masks.items():
            phase_seconds[phase] += float(mask.sum() / sfreq)

        speed = feedback_speed(px, py, masks["feedback"], sfreq)
        (rest_speeds if target == 4 else active_speeds).append(speed)
        cell_lengths_s.append(float(time.size / sfreq))
        feedback_lengths_s.append(trial_length)
        target_numbers.append(target)
        run_numbers.append(int(_field(trial, "runnumber")))
        task_numbers.append(int(_field(trial, "tasknumber")))
        artifacts.append(int(_field(trial, "artifact")))
        if index < 5:
            cell_audits.append(
                {
                    "trial_index": index,
                    "samples": int(time.size),
                    "time_start_ms": float(time[0]),
                    "time_stop_ms": float(time[-1]),
                    "trial_length_s": trial_length,
                    "target_number": target,
                    "run_number": run_numbers[-1],
                    "task_number": task_numbers[-1],
                }
            )

    active_speed = np.concatenate(active_speeds)
    rest_speed = np.concatenate(rest_speeds)
    ks = ks_2samp(active_speed, rest_speed, method="asymp")
    unique_targets, target_counts = np.unique(target_numbers, return_counts=True)
    unique_tasks, task_counts = np.unique(task_numbers, return_counts=True)

    metadata = bci.metadata
    report: dict[str, Any] = {
        "file": str(mat_file.resolve()),
        "schema": {
            "trial_cells": int(len(data)),
            "sampling_rate_hz": sfreq,
            "eeg_channels": int(len(labels)),
            "cell_lengths_s": quantiles(np.asarray(cell_lengths_s)),
            "feedback_lengths_s": quantiles(np.asarray(feedback_lengths_s)),
            "run_count": int(len(np.unique(run_numbers))),
            "trials_per_run": {
                str(run): int(np.sum(np.asarray(run_numbers) == run))
                for run in np.unique(run_numbers)
            },
            "task_counts": {
                str(task): int(count)
                for task, count in zip(unique_tasks, task_counts)
            },
            "target_counts": {
                str(target): int(count)
                for target, count in zip(unique_targets, target_counts)
            },
            "artifact_trials": int(np.sum(artifacts)),
            "first_cells": cell_audits,
        },
        "seam_audit": {
            "continuous_session_matrix": False,
            "independent_trial_cells": True,
            "cell_boundaries_must_be_excluded": True,
            "moabb_concatenation_allowed": False,
        },
        "idle_budget": {
            "session_pre_rest_present": False,
            "initial_causal_idle_s": float(
                phase_masks(np.asarray(times[0]), float(trials[0].triallength))[
                    "pre_target_idle"
                ].sum() / sfreq
            ),
            "total_pre_target_idle_s": phase_seconds["pre_target_idle"],
            "total_intentional_rest_feedback_s": float(
                np.sum(np.asarray(feedback_lengths_s)[np.asarray(target_numbers) == 4])
            ),
            "phase_seconds_all_trials": phase_seconds,
            "natural_idle_interpretation": (
                "Only the fixed cue-preceding -2..0 s interval is present in each "
                "trial cell; independent inter-block/break EEG is absent."
            ),
        },
        "labels_and_channels": {
            "intentional_rest_target_number": 4,
            "intentional_rest_trials": int(np.sum(np.asarray(target_numbers) == 4)),
            "active_trials": int(np.sum(np.asarray(target_numbers) != 4)),
            "channel_labels": labels,
            "sensorimotor_20_present": selected_present,
            "sensorimotor_20_complete": len(selected_present) == len(selected),
            "occipital_channels_excludable": [
                label for label in labels
                if label.upper().startswith(("O", "PO", "CB"))
            ],
            "dedicated_eog_channels": eog_labels,
            "eog_regression_available": bool(eog_labels),
        },
        "cursor_confound": {
            "position_x_present": True,
            "position_y_present": True,
            "velocity_directly_stored": False,
            "velocity_derivable_from_position": True,
            "active_feedback_speed": quantiles(active_speed),
            "rest_feedback_speed": quantiles(rest_speed),
            "speed_5_95_range_iou": robust_range_overlap(active_speed, rest_speed),
            "speed_distribution_ks_statistic": float(ks.statistic),
            "speed_distribution_ks_pvalue": float(ks.pvalue),
            "velocity_matching_feasible": robust_range_overlap(active_speed, rest_speed) > 0,
            "interpretation_limit": (
                "Velocity matching controls motion magnitude only; target direction, "
                "visual layout, task engagement, and feedback remain coupled."
            ),
        },
        "metadata": {
            "mbsr_subject": int(_field(metadata, "MBSRsubject")),
            "session_day": int(_field(metadata, "day")),
            "session_date": int(_field(metadata, "date")),
            "group_stratification_available": True,
        },
        "decision": {
            "raw_mat_parser_required": True,
            "riemannian_transfer_feasible": True,
            "primary_regime_test_interpretable_without_cursor_control": False,
            "primary_regime_test_interpretable_with_velocity_matching": (
                robust_range_overlap(active_speed, rest_speed) > 0
            ),
            "natural_noncontrol_evaluation_complete": False,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stieger2021_smoke_audit.json"),
    )
    args = parser.parse_args()
    report = audit(args.mat_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
