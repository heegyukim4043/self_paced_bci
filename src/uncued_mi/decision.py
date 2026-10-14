from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlarmResult:
    alarms: np.ndarray
    statistic: np.ndarray


def instantaneous(
    scores: np.ndarray,
    threshold: float,
    step_seconds: float,
    refractory_seconds: float = 1.0,
) -> AlarmResult:
    scores = np.asarray(scores, dtype=float)
    above = scores >= threshold
    crossings = above & ~np.r_[False, above[:-1]]
    refractory_steps = max(0, int(np.ceil(refractory_seconds / step_seconds)))
    alarms = np.zeros(len(scores), dtype=bool)
    blocked_until = -1
    for index in np.flatnonzero(crossings):
        if index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
    return AlarmResult(alarms=alarms, statistic=scores.copy())


def moving_average_dwell(
    scores: np.ndarray,
    threshold: float,
    step_seconds: float,
    ma_seconds: float = 0.5,
    dwell_seconds: float = 0.3,
    refractory_seconds: float = 1.0,
) -> AlarmResult:
    scores = np.asarray(scores, dtype=float)
    ma_steps = max(1, int(np.ceil(ma_seconds / step_seconds)))
    dwell_steps = max(1, int(np.ceil(dwell_seconds / step_seconds)))
    kernel = np.ones(ma_steps) / ma_steps
    statistic = np.convolve(scores, kernel, mode="full")[: len(scores)]
    statistic[: ma_steps - 1] = np.nan
    above = statistic >= threshold
    alarms = np.zeros(len(scores), dtype=bool)
    refractory_steps = max(0, int(np.ceil(refractory_seconds / step_seconds)))
    blocked_until = -1
    count = 0
    for index, active in enumerate(above):
        count = count + 1 if active else 0
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
    return AlarmResult(alarms=alarms, statistic=statistic)


def guard_local_moving_average(
    scores: np.ndarray, allowed: np.ndarray, ma_steps: int = 5
) -> np.ndarray:
    """Causal MA that becomes valid only after ``ma_steps`` allowed samples.

    A guard or non-finite score starts a new segment. No MA value may use a
    score from the previous segment, even if that score was computed during a
    suppressed-output period.
    """
    scores = np.asarray(scores, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if scores.ndim != 1 or allowed.shape != scores.shape:
        raise ValueError("scores and allowed must be equally sized 1-D arrays")
    if ma_steps < 1:
        raise ValueError("ma_steps must be positive")
    out = np.full(scores.shape, np.nan)
    segment_start = 0
    for index, score in enumerate(scores):
        if not allowed[index] or not np.isfinite(score):
            segment_start = index + 1
        elif index - segment_start + 1 >= ma_steps:
            out[index] = np.mean(scores[index - ma_steps + 1 : index + 1])
    return out


def guard_rearmed_alarms(
    statistic: np.ndarray,
    threshold: float,
    allowed: np.ndarray,
    dwell_steps: int = 3,
    refractory_steps: int = 10,
) -> np.ndarray:
    """Fire only after a valid below-threshold observation in each segment.

    The first valid value at or above threshold cannot arm the detector.
    Refractory semantics intentionally match the legacy ``count == dwell``
    detector: a run reaching dwell during refractory does not fire later merely
    because refractory ended.
    """
    statistic = np.asarray(statistic, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if statistic.ndim != 1 or allowed.shape != statistic.shape:
        raise ValueError("statistic and allowed must be equally sized 1-D arrays")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if dwell_steps < 1 or refractory_steps < 0:
        raise ValueError("dwell_steps must be positive and refractory_steps nonnegative")
    alarms = np.zeros(statistic.shape, dtype=bool)
    armed = False
    count = 0
    blocked_until = -1
    for index, value in enumerate(statistic):
        if not allowed[index] or not np.isfinite(value):
            armed = False
            count = 0
            continue
        if not armed:
            if value < threshold:
                armed = True
            continue
        count = count + 1 if value >= threshold else 0
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
    return alarms


def cusum_style(
    scores: np.ndarray,
    threshold: float,
    drift: float,
    step_seconds: float,
    refractory_seconds: float = 1.0,
    max_accumulation_seconds: float = 5.0,
) -> AlarmResult:
    scores = np.asarray(scores, dtype=float)
    refractory_steps = max(0, int(np.ceil(refractory_seconds / step_seconds)))
    alarms = np.zeros(len(scores), dtype=bool)
    statistic = np.zeros(len(scores), dtype=float)
    state = 0.0
    blocked = 0
    accumulation_steps = 0
    max_accumulation_steps = max(
        1, int(np.ceil(max_accumulation_seconds / step_seconds))
    )
    for index, score in enumerate(scores):
        if blocked > 0:
            blocked -= 1
            state = 0.0
            accumulation_steps = 0
        else:
            accumulation_steps += 1
            state = max(0.0, state + score - drift)
            if state == 0.0 or accumulation_steps >= max_accumulation_steps:
                if accumulation_steps >= max_accumulation_steps:
                    state = 0.0
                accumulation_steps = 0
            if state >= threshold:
                alarms[index] = True
                state = 0.0
                accumulation_steps = 0
                blocked = refractory_steps
        statistic[index] = state
    return AlarmResult(alarms=alarms, statistic=statistic)


def alarm_onsets(alarms: np.ndarray) -> np.ndarray:
    alarms = np.asarray(alarms, dtype=bool)
    previous = np.r_[False, alarms[:-1]]
    return np.flatnonzero(alarms & ~previous)


def fp_per_minute(alarm_count: int, idle_seconds: float) -> float:
    if idle_seconds <= 0:
        raise ValueError("idle_seconds must be positive")
    return 60.0 * alarm_count / idle_seconds
