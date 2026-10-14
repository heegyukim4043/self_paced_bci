"""Audit Stieger2021 for the frozen trial-wise CNN-GRU protocol.

The public Stieger files contain independent trial cells, not a continuous
session recording.  This audit therefore never concatenates EEG cells.  It
checks whether each subject-session can support the prespecified chronological
train/locked split, a first-10-active screening prefix, and a future active vs
intentional-rest evaluation set.

The report is a compatibility gate.  It does not train a model or claim that
trial-wise Stieger inference validates Lee2019's long-idle stateful setting.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig


WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.1
FILTER_WARMUP_SECONDS = 2.0
TARGET_ONSET_MS = 0.0
FEEDBACK_ONSET_MS = 2000.0
FIRST_ACTIVE_SCREENING = 10
MIN_FUTURE_ACTIVE = 20
MIN_FUTURE_REST = 10


def _field(struct: Any, name: str) -> Any:
    if not hasattr(struct, name):
        raise ValueError(f"missing required field: {name}")
    return getattr(struct, name)


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in ("min", "q25", "median", "q75", "max")}
    points = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, points)))


def _selected_channel_indices(labels: list[str]) -> tuple[list[int], list[str]]:
    selected = ExperimentConfig().sensorimotor_channels
    lookup = {label.casefold(): index for index, label in enumerate(labels)}
    missing = [channel for channel in selected if channel.casefold() not in lookup]
    indices = [lookup[channel.casefold()] for channel in selected if channel.casefold() in lookup]
    return indices, missing


def complete_pre_event_window_count(
    *,
    cell_start_s: float,
    event_s: float,
    sample_rate_hz: float,
    warmup_seconds: float = FILTER_WARMUP_SECONDS,
    window_seconds: float = WINDOW_SECONDS,
    step_seconds: float = STEP_SECONDS,
) -> int:
    """Number of complete causal windows after warm-up and before the event.

    Window support must start at or after filter warm-up and end no later than
    the feedback onset.  The event-aligned window ending exactly at onset is
    allowed because it contains no feedback samples.
    """
    del sample_rate_hz  # Times are expressed in seconds; retained for API clarity.
    first_end = float(cell_start_s) + float(warmup_seconds) + float(window_seconds)
    last_end = float(event_s)
    if first_end > last_end + 1e-9:
        return 0
    return int(np.floor((last_end - first_end) / float(step_seconds) + 1e-9)) + 1


def chronological_split_counts(
    target_number: np.ndarray,
    artifact: np.ndarray,
    trial_length_s: np.ndarray,
    *,
    split_index: int | None = None,
    first_active: int = FIRST_ACTIVE_SCREENING,
    window_seconds: float = WINDOW_SECONDS,
    minimum_event_seconds: float = STEP_SECONDS,
) -> dict[str, Any]:
    """Return leakage-safe train/screen/future counts for one session."""
    target_number = np.asarray(target_number, dtype=int)
    artifact = np.asarray(artifact, dtype=bool)
    trial_length_s = np.asarray(trial_length_s, dtype=float)
    if not (target_number.size == artifact.size == trial_length_s.size):
        raise ValueError("trial arrays must have equal length")
    total = int(target_number.size)
    split = total // 2 if split_index is None else int(split_index)
    if not 0 < split < total:
        raise ValueError("split index must lie inside the trial sequence")

    index = np.arange(total)
    # Onset windows are allowed to straddle the feedback boundary: a 1-s
    # causal window ending 0.1 s after feedback onset contains 0.9 s of
    # preparation and 0.1 s of feedback. Requiring a full 1-s feedback phase
    # would selectively discard fast target-reaching trials. A trial is onset-
    # usable when it contributes at least one post-onset decision step.
    usable = (
        (~artifact)
        & np.isfinite(trial_length_s)
        & (trial_length_s >= minimum_event_seconds)
    )
    active = target_number != 4
    rest = target_number == 4
    train = index < split
    locked = index >= split
    locked_active_indices = index[locked & usable & active]

    cutoff: int | None = None
    if locked_active_indices.size >= first_active:
        cutoff = int(locked_active_indices[first_active - 1])
        screening = locked & (index <= cutoff)
        future = locked & (index > cutoff)
    else:
        screening = np.zeros(total, dtype=bool)
        future = np.zeros(total, dtype=bool)

    def counts(mask: np.ndarray) -> dict[str, float | int]:
        valid = mask & usable
        return {
            "cells": int(mask.sum()),
            "usable_cells": int(valid.sum()),
            "active_trials": int((valid & active).sum()),
            "intentional_rest_trials": int((valid & rest).sum()),
            "active_feedback_s": float(trial_length_s[valid & active].sum()),
            "intentional_rest_feedback_s": float(trial_length_s[valid & rest].sum()),
        }

    return {
        "split_index": split,
        "training": counts(train),
        "locked_all": counts(locked),
        "screening_first10_active": counts(screening),
        "screening_cutoff_trial_index": cutoff,
        "future_after_screening": counts(future),
        "unusable_artifact_trials": int(artifact.sum()),
        "unusable_feedback_shorter_than_one_step": int(
            ((~artifact) & (trial_length_s < minimum_event_seconds)).sum()
        ),
        "feedback_shorter_than_full_window_trials": int(
            ((~artifact) & (trial_length_s < window_seconds)).sum()
        ),
        "feedback_shorter_than_full_window_active": int(
            ((~artifact) & active & (trial_length_s < window_seconds)).sum()
        ),
        "feedback_shorter_than_full_window_rest": int(
            ((~artifact) & rest & (trial_length_s < window_seconds)).sum()
        ),
    }


def _parse_subject_session(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"S(\d+)_Session_(\d+)\.mat", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unexpected Stieger filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def audit_one(path: Path) -> dict[str, Any]:
    loaded = loadmat(
        path,
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["BCI"],
    )
    bci = loaded["BCI"]
    data = np.atleast_1d(_field(bci, "data"))
    times = np.atleast_1d(_field(bci, "time"))
    trials = np.atleast_1d(_field(bci, "TrialData"))
    labels = [str(value) for value in np.atleast_1d(_field(bci.chaninfo, "label"))]
    sfreq = float(_field(bci, "SRATE"))
    subject, session = _parse_subject_session(path)

    if not (len(data) == len(times) == len(trials)):
        raise ValueError(f"{path.name}: inconsistent trial-cell counts")

    channel_indices, missing_channels = _selected_channel_indices(labels)
    targets: list[int] = []
    artifacts: list[int] = []
    lengths: list[float] = []
    run_numbers: list[int] = []
    task_numbers: list[int] = []
    cell_starts_s: list[float] = []
    cell_stops_s: list[float] = []
    feedback_onsets_s: list[float] = []
    onset_errors_s: list[float] = []
    sample_intervals_s: list[float] = []
    pre_event_window_counts: list[int] = []
    support_failures: list[int] = []
    monotonic_failures: list[int] = []
    shape_failures: list[int] = []

    for cell_index, (eeg, time, trial) in enumerate(zip(data, times, trials)):
        eeg = np.asarray(eeg)
        time_s = np.asarray(time, dtype=float).reshape(-1) / 1000.0
        if eeg.ndim != 2 or eeg.shape[0] != len(labels) or eeg.shape[1] != time_s.size:
            shape_failures.append(cell_index)
            continue
        diffs = np.diff(time_s)
        if not diffs.size or np.any(diffs <= 0):
            monotonic_failures.append(cell_index)
            continue

        trial_length = float(_field(trial, "triallength"))
        feedback_candidates = np.flatnonzero(time_s >= FEEDBACK_ONSET_MS / 1000.0 - 1e-12)
        if not feedback_candidates.size:
            support_failures.append(cell_index)
            actual_feedback = np.nan
        else:
            actual_feedback = float(time_s[feedback_candidates[0]])
            feedback_stop = FEEDBACK_ONSET_MS / 1000.0 + trial_length
            if time_s[-1] + 1.0 / sfreq < feedback_stop:
                support_failures.append(cell_index)

        targets.append(int(_field(trial, "targetnumber")))
        artifacts.append(int(_field(trial, "artifact")))
        lengths.append(trial_length)
        run_numbers.append(int(_field(trial, "runnumber")))
        task_numbers.append(int(_field(trial, "tasknumber")))
        cell_starts_s.append(float(time_s[0]))
        cell_stops_s.append(float(time_s[-1]))
        feedback_onsets_s.append(actual_feedback)
        onset_errors_s.append(actual_feedback - FEEDBACK_ONSET_MS / 1000.0)
        sample_intervals_s.append(float(np.median(diffs)))
        pre_event_window_counts.append(
            complete_pre_event_window_count(
                cell_start_s=float(time_s[0]),
                event_s=actual_feedback,
                sample_rate_hz=sfreq,
            )
            if np.isfinite(actual_feedback)
            else 0
        )

    targets_array = np.asarray(targets, dtype=int)
    artifacts_array = np.asarray(artifacts, dtype=int)
    lengths_array = np.asarray(lengths, dtype=float)
    split = chronological_split_counts(targets_array, artifacts_array, lengths_array)
    future = split["future_after_screening"]
    training = split["training"]

    onset_tolerance_s = 1.0 / sfreq + 1e-9
    assertions = {
        "all_cells_shape_valid": not shape_failures,
        "all_cell_times_monotonic": not monotonic_failures,
        "all_cells_restart_near_minus2s": bool(
            cell_starts_s and np.all(np.abs(np.asarray(cell_starts_s) + 2.0) <= onset_tolerance_s)
        ),
        "feedback_onset_target_plus2s": bool(
            onset_errors_s and np.all(np.abs(np.asarray(onset_errors_s)) <= onset_tolerance_s)
        ),
        "feedback_support_complete": not support_failures,
        "pre_event_complete_window_available": bool(
            pre_event_window_counts and np.min(pre_event_window_counts) >= 1
        ),
        "sensorimotor20_complete": not missing_channels and len(channel_indices) == 20,
        "training_has_active_and_rest": bool(
            training["active_trials"] >= 10 and training["intentional_rest_trials"] >= 10
        ),
        "locked_has_first10_active": split["screening_cutoff_trial_index"] is not None,
        "future_has_minimum_active": future["active_trials"] >= MIN_FUTURE_ACTIVE,
        "future_has_minimum_rest": future["intentional_rest_trials"] >= MIN_FUTURE_REST,
        "trial_cells_are_independent_reset_units": True,
        "cross_cell_windows_forbidden": True,
    }
    compatible = bool(all(assertions.values()))

    report = {
        "file": str(path.resolve()),
        "subject": subject,
        "session": session,
        "sampling_rate_hz": sfreq,
        "channels": {
            "eeg_channel_count": len(labels),
            "sensorimotor20_complete": not missing_channels,
            "sensorimotor20_missing": missing_channels,
            "sensorimotor20_indices": channel_indices,
        },
        "cell_support": {
            "trial_cells": int(len(data)),
            "cell_start_s": _quantiles(cell_starts_s),
            "cell_stop_s": _quantiles(cell_stops_s),
            "sample_interval_s": _quantiles(sample_intervals_s),
            "feedback_length_s": _quantiles(lengths_array),
            "shape_failure_indices": shape_failures,
            "monotonic_time_failure_indices": monotonic_failures,
            "feedback_support_failure_indices": sorted(set(support_failures)),
        },
        "feedback_onset": {
            "documented_target_relative_s": FEEDBACK_ONSET_MS / 1000.0,
            "observed_s": _quantiles(feedback_onsets_s),
            "absolute_error_s": _quantiles(np.abs(onset_errors_s).tolist()),
        },
        "usable_pre_event_context": {
            "cell_start_to_feedback_s": float(FEEDBACK_ONSET_MS / 1000.0 + 2.0),
            "causal_filter_warmup_s": FILTER_WARMUP_SECONDS,
            "post_warmup_pre_feedback_signal_s": float(
                FEEDBACK_ONSET_MS / 1000.0 + 2.0 - FILTER_WARMUP_SECONDS
            ),
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "complete_pre_feedback_window_count": _quantiles(pre_event_window_counts),
            "closest_baseline_definition": (
                "target/preparation-only support ending at feedback onset; "
                "not natural idle"
            ),
            "long_idle_context_available": False,
            "onset_window_support": (
                "Post-onset 1-s windows may straddle preparation and feedback; "
                "a trial needs at least one 0.1-s feedback decision step, not a "
                "full 1-s feedback-only interval."
            ),
        },
        "trial_labels": {
            "target_counts": {
                str(target): int(np.sum(targets_array == target))
                for target in np.unique(targets_array)
            },
            "active_trials_total": int(np.sum(targets_array != 4)),
            "intentional_rest_trials_total": int(np.sum(targets_array == 4)),
            "artifact_trials_total": int(np.sum(artifacts_array != 0)),
            "run_count": int(len(np.unique(run_numbers))),
            "task_count": int(len(np.unique(task_numbers))),
        },
        "chronological_protocol": split,
        "seam_reset_contract": {
            "continuous_session_matrix": False,
            "trial_cells_independent": True,
            "cnn_gru_hidden_state_reset_each_cell": True,
            "ewma_ma_dwell_state_reset_each_cell": True,
            "windows_may_cross_cells": False,
            "expected_reset_units": int(len(data) - np.sum(artifacts_array != 0)),
            "implementation_assertion": (
                "must be enforced and prefix-tested when the Step-2 score cache is built"
            ),
        },
        "assertions": assertions,
        "compatible_for_trialwise_frozen_protocol": compatible,
        "interpretation": (
            "Compatible only with trial-wise preparation-to-feedback validation; "
            "not with Lee2019 long-idle continuous-state deployment."
        ),
    }
    del loaded, bci, data, times, trials
    gc.collect()
    return report


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    compatible = [row for row in reports if row["compatible_for_trialwise_frozen_protocol"]]
    failed_assertions: dict[str, list[str]] = {}
    for row in reports:
        for key, value in row["assertions"].items():
            if not value:
                failed_assertions.setdefault(key, []).append(
                    f"S{row['subject']}_Session_{row['session']}"
                )

    def field(section: str, name: str) -> list[float]:
        return [float(row["chronological_protocol"][section][name]) for row in reports]

    return {
        "protocol": {
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "filter_warmup_seconds": FILTER_WARMUP_SECONDS,
            "feedback_onset_target_relative_s": FEEDBACK_ONSET_MS / 1000.0,
            "chronological_split": "first half training; second half locked",
            "screening": "first 10 usable active trials in locked half",
            "future": "all usable active/rest trials strictly after screening cutoff",
            "minimum_future_active": MIN_FUTURE_ACTIVE,
            "minimum_future_intentional_rest": MIN_FUTURE_REST,
            "onset_usable_minimum_feedback_s": STEP_SECONDS,
            "full_feedback_window_seconds": WINDOW_SECONDS,
        },
        "holdings": {
            "subject_sessions": len(reports),
            "unique_subjects": len({row["subject"] for row in reports}),
            "session1_files": sum(row["session"] == 1 for row in reports),
            "session2_files": sum(row["session"] == 2 for row in reports),
        },
        "compatibility": {
            "passed_subject_sessions": len(compatible),
            "failed_subject_sessions": len(reports) - len(compatible),
            "all_passed": len(compatible) == len(reports),
            "failed_assertions": failed_assertions,
        },
        "count_ranges": {
            "training_active_trials": _quantiles(field("training", "active_trials")),
            "training_rest_trials": _quantiles(field("training", "intentional_rest_trials")),
            "screening_cells_to_first10_active": _quantiles(
                field("screening_first10_active", "cells")
            ),
            "future_active_trials": _quantiles(field("future_after_screening", "active_trials")),
            "future_rest_trials": _quantiles(
                field("future_after_screening", "intentional_rest_trials")
            ),
            "future_rest_feedback_s": _quantiles(
                field("future_after_screening", "intentional_rest_feedback_s")
            ),
            "feedback_shorter_than_full_window_trials": _quantiles(
                [
                    row["chronological_protocol"]["feedback_shorter_than_full_window_trials"]
                    for row in reports
                ]
            ),
            "unusable_feedback_shorter_than_one_step": _quantiles(
                [
                    row["chronological_protocol"]["unusable_feedback_shorter_than_one_step"]
                    for row in reports
                ]
            ),
        },
        "decision": {
            "step2_score_cache_allowed": len(compatible) > 0,
            "eligible_subject_sessions": [
                {"subject": row["subject"], "session": row["session"]}
                for row in compatible
            ],
            "long_idle_stateful_validation_possible": False,
            "valid_scope": "trial-wise partial external validation",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    counts = summary["count_ranges"]
    compatibility = summary["compatibility"]
    lines = [
        "# Stieger2021 CNN-GRU compatibility audit",
        "",
        "## Decision",
        "",
        f"- Passed: **{compatibility['passed_subject_sessions']}/{summary['holdings']['subject_sessions']} subject-sessions**",
        f"- Step-2 trial-wise score cache allowed: **{summary['decision']['step2_score_cache_allowed']}**",
        "- Long-idle continuous-state validation: **not supported by the public trial cells**",
        "- Valid scope: **trial-wise preparation-to-feedback partial external validation**",
        "",
        "## Frozen compatibility contract",
        "",
        "- First half: chronological supervised training",
        "- Second half: locked screening plus future evaluation",
        "- Screening: first 10 usable active trials",
        "- Every trial cell resets CNN-GRU, EWMA, MA, and dwell state",
        "- No window or state may cross a cell seam",
        "- Local baseline is target/preparation-only support, not natural idle",
        "",
        "## Count ranges across subject-sessions",
        "",
        "| Quantity | Min | Median | Max |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Training active trials", "training_active_trials"),
        ("Training intentional-rest trials", "training_rest_trials"),
        ("Screening cells through 10th active", "screening_cells_to_first10_active"),
        ("Future active trials", "future_active_trials"),
        ("Future intentional-rest trials", "future_rest_trials"),
        ("Future intentional-rest feedback seconds", "future_rest_feedback_s"),
        ("Feedback shorter than a full 1-s window", "feedback_shorter_than_full_window_trials"),
        ("Feedback shorter than one 0.1-s step", "unusable_feedback_shorter_than_one_step"),
    ):
        values = counts[key]
        lines.append(
            f"| {label} | {values['min']:.1f} | {values['median']:.1f} | {values['max']:.1f} |"
        )
    lines.extend(["", "## Failures", ""])
    if compatibility["failed_assertions"]:
        for key, sessions in compatibility["failed_assertions"].items():
            lines.append(f"- `{key}`: {', '.join(sessions)}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "Passing this audit does not make Stieger a continuous self-paced stream. "
            "It only permits Step 2 to build cell-local causal CNN-GRU scores without "
            "cross-seam leakage.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/stieger2021"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_compatibility_audit.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("STIEGER_CNN_GRU_COMPATIBILITY_AUDIT_20260921.md"),
    )
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--sessions", type=int, nargs="*")
    args = parser.parse_args()

    paths = sorted(args.raw_root.glob("S*_Session_*.mat"))
    if args.subjects:
        selected = set(args.subjects)
        paths = [path for path in paths if _parse_subject_session(path)[0] in selected]
    if args.sessions:
        selected_sessions = set(args.sessions)
        paths = [path for path in paths if _parse_subject_session(path)[1] in selected_sessions]
    if not paths:
        raise FileNotFoundError(f"no Stieger MAT files found under {args.raw_root}")

    reports: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] auditing {path.name}", flush=True)
        reports.append(audit_one(path))

    payload = {
        "summary": aggregate(reports),
        "subject_sessions": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
