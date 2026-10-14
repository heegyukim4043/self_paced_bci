"""Command-blind calibration for the guard-rearmed event detector.

This module deliberately does not reuse the historical source operating-point
selector, whose tie break can inspect command sensitivity. Calibration here
uses only an idle anchor and the exact same decision rule used at evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision import guard_local_moving_average, guard_rearmed_alarms


@dataclass(frozen=True)
class GuardRearmAnchorPoint:
    target_fp_per_min: float
    threshold: float
    anchor_alarm_count: int
    anchor_fp_per_min: float
    anchor_seconds: float
    count_resolution_fp_per_min: float
    within_target_band: bool
    attainable_alarm_counts: tuple[int, ...]


def tune_guard_rearmed_anchor(
    anchor_z: np.ndarray,
    target_fp_per_min: float,
    step_seconds: float,
    ma_steps: int = 5,
    dwell_steps: int = 3,
    refractory_steps: int = 10,
    admissible_half_width: float = 0.5,
) -> GuardRearmAnchorPoint:
    """Select a threshold from idle only; ties choose the highest threshold.

    An exact threshold partition is evaluated at each unique valid MA value,
    plus points below its minimum and above its maximum. Alarm rate need not be
    monotone in threshold for a below-before-high detector, so no interpolation
    or monotonicity assumption is made.
    """
    anchor_z = np.asarray(anchor_z, dtype=float)
    if anchor_z.ndim != 1 or not np.isfinite(anchor_z).all():
        raise ValueError("anchor_z must be a finite 1-D idle score sequence")
    if target_fp_per_min <= 0 or step_seconds <= 0 or admissible_half_width < 0:
        raise ValueError("invalid target, time step, or admissible width")
    allowed = np.ones(len(anchor_z), dtype=bool)
    ma = guard_local_moving_average(anchor_z, allowed, ma_steps=ma_steps)
    finite = np.unique(ma[np.isfinite(ma)])
    if not len(finite):
        raise ValueError("anchor is shorter than its causal moving average")
    below = np.nextafter(float(finite[0]), -np.inf)
    above = np.nextafter(float(finite[-1]), np.inf)
    candidates = np.unique(np.r_[below, finite, above])
    seconds = len(anchor_z) * step_seconds
    rates = []
    counts = []
    for threshold in candidates:
        alarms = guard_rearmed_alarms(
            ma, float(threshold), allowed,
            dwell_steps=dwell_steps, refractory_steps=refractory_steps,
        )
        count = int(alarms.sum())
        counts.append(count)
        rates.append(count * 60.0 / seconds)
    rates = np.asarray(rates)
    counts = np.asarray(counts)
    error = np.abs(rates - target_fp_per_min)
    best = np.flatnonzero(np.isclose(error, error.min(), rtol=0, atol=1e-12))[-1]
    achieved = float(rates[best])
    return GuardRearmAnchorPoint(
        target_fp_per_min=float(target_fp_per_min),
        threshold=float(candidates[best]),
        anchor_alarm_count=int(counts[best]),
        anchor_fp_per_min=achieved,
        anchor_seconds=float(seconds),
        count_resolution_fp_per_min=float(60.0 / seconds),
        within_target_band=bool(
            abs(achieved - target_fp_per_min) <= admissible_half_width
        ),
        attainable_alarm_counts=tuple(sorted(set(counts.tolist()))),
    )
