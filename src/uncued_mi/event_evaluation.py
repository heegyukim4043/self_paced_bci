"""Dataset- and model-independent evaluation for asynchronous event detectors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OnsetResult:
    tolerance_seconds: float
    sensitivity: float
    detected_events: int
    total_events: int
    median_latency_seconds: float


@dataclass(frozen=True)
class IntervalResult:
    dtc: float
    gtc: float
    coverage: float
    temporal_iou: float
    fragmentation: float
    precision: float
    recall: float
    f1: float


def _validate(times, alarms, intervals=None):
    times = np.asarray(times, dtype=float)
    alarms = np.asarray(alarms, dtype=bool)
    if times.ndim != 1 or alarms.ndim != 1 or len(times) != len(alarms):
        raise ValueError("times and alarms must be aligned one-dimensional arrays")
    if len(times) and np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if intervals is None:
        return times, alarms
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("every interval must have positive duration")
    return times, alarms, intervals


def onset_metrics(times, alarms, onsets, tolerances: Iterable[float]) -> dict[str, OnsetResult]:
    """Score point alarms against onset timepoints at several post-onset collars."""
    times, alarms = _validate(times, alarms)
    onsets = np.asarray(onsets, dtype=float).reshape(-1)
    alarm_times = times[alarms]
    output = {}
    for tolerance in tolerances:
        tolerance = float(tolerance)
        if tolerance < 0:
            raise ValueError("onset tolerance must be non-negative")
        latencies = []
        for onset in onsets:
            hits = alarm_times[(alarm_times >= onset) & (alarm_times <= onset + tolerance)]
            if len(hits):
                latencies.append(float(hits[0] - onset))
        output[str(tolerance)] = OnsetResult(
            tolerance_seconds=tolerance,
            sensitivity=len(latencies) / len(onsets) if len(onsets) else float("nan"),
            detected_events=len(latencies), total_events=len(onsets),
            median_latency_seconds=float(np.median(latencies)) if latencies else float("nan"),
        )
    return output


def false_alarms_per_minute(alarms, valid_idle_mask, step_seconds: float) -> float:
    alarms = np.asarray(alarms, dtype=bool)
    idle = np.asarray(valid_idle_mask, dtype=bool)
    if alarms.shape != idle.shape:
        raise ValueError("alarms and valid_idle_mask must have the same shape")
    exposure = float(idle.sum() * step_seconds)
    return 60.0 * float(np.sum(alarms & idle)) / exposure if exposure else float("nan")


def predicted_intervals(times, alarms, statistic, trigger_threshold, *, allowed=None,
                        release_drop=0.5, release_hold_seconds=0.3,
                        merge_gap_seconds=0.3) -> np.ndarray:
    """Restore predicted intervals from point alarms and a continuous statistic."""
    times, alarms = _validate(times, alarms)
    statistic = np.asarray(statistic, dtype=float)
    allowed = np.ones(len(times), bool) if allowed is None else np.asarray(allowed, bool)
    if statistic.shape != times.shape or allowed.shape != times.shape:
        raise ValueError("statistic and allowed must align with times")
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    hold = max(1, int(np.ceil(release_hold_seconds / step)))
    release = float(trigger_threshold) - float(release_drop)
    result = []
    index = 0
    while index < len(times):
        candidates = np.flatnonzero(alarms[index:])
        if not len(candidates):
            break
        start = index + int(candidates[0])
        stop_index, below = start + 1, 0
        while stop_index < len(times):
            below = below + 1 if (
                not allowed[stop_index] or not np.isfinite(statistic[stop_index])
                or statistic[stop_index] < release
            ) else 0
            if below >= hold:
                break
            stop_index += 1
        end = min(len(times) - 1, max(start + 1, stop_index - hold + 1))
        result.append([times[start], times[end]])
        index = max(end + 1, stop_index + 1)
    merged = []
    for interval in result:
        if merged and interval[0] - merged[-1][1] <= merge_gap_seconds:
            merged[-1][1] = interval[1]
        else:
            merged.append(interval)
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def _overlap(left, right):
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def interval_metrics(predicted, ground_truth, dtc: float, gtc: float) -> IntervalResult:
    """Binary DTC/GTC event scoring plus continuous coverage and temporal IoU."""
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    ground_truth = np.asarray(ground_truth, dtype=float).reshape(-1, 2)
    _validate(np.array([]), np.array([]), predicted)
    _validate(np.array([]), np.array([]), ground_truth)
    valid = np.asarray([
        sum(_overlap(p, g) for g in ground_truth) / max(p[1] - p[0], 1e-12) >= dtc
        for p in predicted
    ], dtype=bool)
    true_positive, coverage, iou, fragmentation = 0, [], [], []
    for truth in ground_truth:
        overlaps = np.asarray([_overlap(p, truth) for p in predicted])
        covered = min(1.0, overlaps[valid].sum() / (truth[1] - truth[0])) if len(predicted) else 0.0
        coverage.append(covered)
        true_positive += int(covered >= gtc)
        fragmentation.append(int(np.sum(overlaps > 0)))
        iou.append(max([
            _overlap(p, truth) / (max(p[1], truth[1]) - min(p[0], truth[0]))
            for p in predicted
        ] or [0.0]))
    false_positive = int(np.sum(~valid))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / len(ground_truth) if len(ground_truth) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return IntervalResult(float(dtc), float(gtc), float(np.mean(coverage)),
                          float(np.mean(iou)), float(np.mean(fragmentation)),
                          precision, recall, f1)


def circular_shift_onset_null(times, alarms, onsets, tolerances, *, shifts=500,
                              minimum_shift_seconds=20.0, seed=0) -> dict[str, float]:
    """Median onset sensitivity under circular shifts preserving alarm structure."""
    times, alarms = _validate(times, alarms)
    if len(times) < 2:
        raise ValueError("at least two time samples are required")
    step = float(np.median(np.diff(times)))
    minimum = max(1, int(np.ceil(minimum_shift_seconds / step)))
    candidates = np.arange(minimum, len(alarms) - minimum)
    if not len(candidates):
        raise ValueError("recording is too short for the requested minimum shift")
    rng = np.random.default_rng(seed)
    offsets = rng.choice(candidates, min(int(shifts), len(candidates)), replace=False)
    values = {str(float(t)): [] for t in tolerances}
    for offset in offsets:
        row = onset_metrics(times, np.roll(alarms, int(offset)), onsets, tolerances)
        for key, result in row.items():
            values[key].append(result.sensitivity)
    return {key: float(np.median(value)) for key, value in values.items()}


def serializable(results):
    """Convert metric dataclasses or mappings to JSON-ready dictionaries."""
    if hasattr(results, "__dataclass_fields__"):
        return asdict(results)
    if isinstance(results, dict):
        return {key: serializable(value) for key, value in results.items()}
    return results
