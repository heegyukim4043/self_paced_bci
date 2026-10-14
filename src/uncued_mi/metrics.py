from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventMetrics:
    fp_per_minute: float
    true_positive_rate: float
    median_delay_seconds: float
    false_alarms: int
    detected_events: int
    total_events: int
    idle_seconds: float
    delay_seconds: tuple[float, ...] = ()


def chance_tpr(fp_per_minute: float, event_seconds: float) -> float:
    """Poisson reference probability of at least one random alarm in an event."""
    if fp_per_minute < 0 or event_seconds < 0:
        raise ValueError("Rates and durations must be non-negative")
    return float(1.0 - np.exp(-fp_per_minute * event_seconds / 60.0))


def _in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in np.asarray(intervals).reshape(-1, 2):
        mask |= (times >= start) & (times < stop)
    return mask


def evaluate_alarms(
    times: np.ndarray,
    alarms: np.ndarray,
    idle_intervals: np.ndarray,
    command_intervals: np.ndarray,
) -> EventMetrics:
    times = np.asarray(times, dtype=float)
    alarms = np.asarray(alarms, dtype=bool)
    alarm_times = times[alarms]
    idle_mask = _in_intervals(alarm_times, idle_intervals)
    false_alarms = int(np.sum(idle_mask))
    idle_seconds = float(
        np.sum(np.asarray(idle_intervals)[:, 1] - np.asarray(idle_intervals)[:, 0])
    )

    delays: list[float] = []
    for start, stop in np.asarray(command_intervals).reshape(-1, 2):
        hits = alarm_times[(alarm_times >= start) & (alarm_times < stop)]
        if len(hits):
            delays.append(float(hits[0] - start))
    total = len(command_intervals)
    detected = len(delays)
    return EventMetrics(
        fp_per_minute=(
            60.0 * false_alarms / idle_seconds
            if idle_seconds > 0
            else float("nan")
        ),
        true_positive_rate=detected / total if total else float("nan"),
        median_delay_seconds=float(np.median(delays)) if delays else float("nan"),
        false_alarms=false_alarms,
        detected_events=detected,
        total_events=total,
        idle_seconds=idle_seconds,
        delay_seconds=tuple(delays),
    )
