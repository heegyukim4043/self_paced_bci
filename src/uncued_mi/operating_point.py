from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .decision import (
    AlarmResult,
    cusum_style,
    instantaneous,
    moving_average_dwell,
)
from .metrics import EventMetrics, evaluate_alarms


@dataclass(frozen=True)
class ScoredSequence:
    times: np.ndarray
    scores: np.ndarray
    idle_intervals: np.ndarray
    command_intervals: np.ndarray


@dataclass(frozen=True)
class OperatingPoint:
    decision: str
    threshold: float
    drift: float | None
    source_fp_per_minute: float
    source_true_positive_rate: float
    source_median_delay_seconds: float


def apply_decision(
    sequence: ScoredSequence,
    decision: str,
    threshold: float,
    step_seconds: float,
    drift: float | None = None,
    ma_seconds: float = 0.5,
    dwell_seconds: float = 0.3,
    refractory_seconds: float = 1.0,
    max_accumulation_seconds: float = 5.0,
) -> AlarmResult:
    if decision == "threshold":
        return instantaneous(
            sequence.scores, threshold, step_seconds, refractory_seconds
        )
    if decision == "ma_dwell":
        return moving_average_dwell(
            sequence.scores,
            threshold,
            step_seconds,
            ma_seconds,
            dwell_seconds,
            refractory_seconds,
        )
    if decision == "accumulator":
        if drift is None:
            raise ValueError("drift is required for accumulator")
        return cusum_style(
            sequence.scores,
            threshold,
            drift,
            step_seconds,
            refractory_seconds,
            max_accumulation_seconds,
        )
    raise ValueError(f"Unknown decision layer: {decision}")


def aggregate_metrics(metrics: list[EventMetrics]) -> EventMetrics:
    idle_seconds = sum(item.idle_seconds for item in metrics)
    false_alarms = sum(item.false_alarms for item in metrics)
    detected = sum(item.detected_events for item in metrics)
    total = sum(item.total_events for item in metrics)
    delays = [
        delay
        for item in metrics
        for delay in item.delay_seconds
        if np.isfinite(delay)
    ]
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


def evaluate_sequence(
    sequence: ScoredSequence,
    decision: str,
    threshold: float,
    step_seconds: float,
    drift: float | None = None,
    ma_seconds: float = 0.5,
    dwell_seconds: float = 0.3,
    refractory_seconds: float = 1.0,
    max_accumulation_seconds: float = 5.0,
) -> EventMetrics:
    result = apply_decision(
        sequence,
        decision,
        threshold,
        step_seconds,
        drift,
        ma_seconds,
        dwell_seconds,
        refractory_seconds,
        max_accumulation_seconds,
    )
    return evaluate_alarms(
        sequence.times,
        result.alarms,
        sequence.idle_intervals,
        sequence.command_intervals,
    )


def tune_source_operating_point(
    sequences: list[ScoredSequence],
    decision: str,
    target_fp_per_minute: float,
    step_seconds: float,
    drift: float | None = None,
    ma_seconds: float = 0.5,
    dwell_seconds: float = 0.3,
    refractory_seconds: float = 1.0,
    max_accumulation_seconds: float = 5.0,
    grid_size: int = 200,
) -> OperatingPoint:
    return tune_source_operating_points(
        sequences=sequences,
        decision=decision,
        target_fp_per_minute=[target_fp_per_minute],
        step_seconds=step_seconds,
        drift=drift,
        ma_seconds=ma_seconds,
        dwell_seconds=dwell_seconds,
        refractory_seconds=refractory_seconds,
        max_accumulation_seconds=max_accumulation_seconds,
        grid_size=grid_size,
    )[float(target_fp_per_minute)]


def tune_source_operating_points(
    sequences: list[ScoredSequence],
    decision: str,
    target_fp_per_minute: list[float] | tuple[float, ...],
    step_seconds: float,
    drift: float | None = None,
    ma_seconds: float = 0.5,
    dwell_seconds: float = 0.3,
    refractory_seconds: float = 1.0,
    max_accumulation_seconds: float = 5.0,
    grid_size: int = 80,
) -> dict[float, OperatingPoint]:
    """Evaluate one threshold curve and select all requested FP operating points."""
    targets = sorted({float(value) for value in target_fp_per_minute})
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("target_fp_per_minute values must be positive")
    pooled = np.concatenate([item.scores for item in sequences])
    finite = pooled[np.isfinite(pooled)]
    if not len(finite):
        raise ValueError("No finite source scores")
    if decision == "accumulator":
        low, high = 0.0, max(1.0, float(np.quantile(np.abs(finite), 0.999)) * 50)
        candidates = np.linspace(low, high, grid_size + 1)[1:]
    else:
        probabilities = np.linspace(0.5, 0.9999, grid_size)
        candidates = np.unique(np.quantile(finite, probabilities))

    curve: list[tuple[float, EventMetrics]] = []
    for threshold in candidates:
        metrics = aggregate_metrics(
            [
                evaluate_sequence(
                    item,
                    decision,
                    float(threshold),
                    step_seconds,
                    drift,
                    ma_seconds,
                    dwell_seconds,
                    refractory_seconds,
                    max_accumulation_seconds,
                )
                for item in sequences
            ]
        )
        curve.append((float(threshold), metrics))

    selected: dict[float, OperatingPoint] = {}
    for target in targets:
        threshold, metrics = min(
            curve,
            key=lambda item: (
                abs(item[1].fp_per_minute - target),
                -(
                    item[1].true_positive_rate
                    if np.isfinite(item[1].true_positive_rate)
                    else 0.0
                ),
            ),
        )
        selected[target] = OperatingPoint(
            decision=decision,
            threshold=threshold,
            drift=drift,
            source_fp_per_minute=metrics.fp_per_minute,
            source_true_positive_rate=metrics.true_positive_rate,
            source_median_delay_seconds=metrics.median_delay_seconds,
        )
    return selected


def operating_point_dict(value: OperatingPoint) -> dict:
    return asdict(value)
