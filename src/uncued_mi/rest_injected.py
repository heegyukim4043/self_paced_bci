from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operating_point import ScoredSequence


@dataclass(frozen=True)
class InjectedStreamAudit:
    rest_windows: int
    command_events: int
    guard_windows: int
    rest_chunks: int


def random_partition(
    size: int,
    minimum: int,
    maximum: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Partition an array into non-overlapping chunks without reusing samples."""
    if not (0 < minimum <= maximum):
        raise ValueError("invalid chunk limits")
    chunks: list[tuple[int, int]] = []
    start = 0
    while size - start >= minimum:
        remaining = size - start
        if remaining <= maximum:
            stop = size
        else:
            stop = start + int(rng.integers(minimum, maximum + 1))
            if 0 < size - stop < minimum:
                stop = size - minimum
        chunks.append((start, stop))
        start = stop
    return chunks


def build_rest_injected_stream(
    rest_blocks: list[np.ndarray],
    command_trials: list[np.ndarray],
    *,
    step_seconds: float,
    minimum_rest_seconds: float = 5.0,
    maximum_rest_seconds: float = 11.0,
    guard_seconds: float = 1.0,
    seed: int = 0,
) -> tuple[ScoredSequence, InjectedStreamAudit]:
    """Interleave non-overlapping resting chunks and command score sequences.

    The input is already window-scored. Guard values are deliberately below the
    observed score range, reset the causal decision layer, and are excluded from
    both idle and command exposure.
    """
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if not rest_blocks or not command_trials:
        raise ValueError("rest_blocks and command_trials are required")
    rng = np.random.default_rng(seed)
    minimum = int(np.ceil(minimum_rest_seconds / step_seconds))
    maximum = int(np.floor(maximum_rest_seconds / step_seconds))
    guard = max(1, int(np.ceil(guard_seconds / step_seconds)))
    partitions: list[tuple[int, int, int]] = []
    for block_index, values in enumerate(rest_blocks):
        for start, stop in random_partition(len(values), minimum, maximum, rng):
            partitions.append((block_index, start, stop))
    rng.shuffle(partitions)
    if not partitions:
        raise ValueError("rest blocks are too short")
    trial_order = rng.permutation(len(command_trials))
    event_count = min(len(partitions), len(command_trials))
    partitions = partitions[:event_count]
    trial_order = trial_order[:event_count]
    finite = np.concatenate(
        [np.asarray(x, dtype=float) for x in rest_blocks + command_trials]
    )
    finite = finite[np.isfinite(finite)]
    reset_value = float(np.min(finite) - 10.0 * max(np.std(finite), 1.0))

    scores: list[float] = []
    idle: list[tuple[float, float]] = []
    command: list[tuple[float, float]] = []
    rest_windows = 0
    guard_windows = 0

    def now() -> float:
        return len(scores) * step_seconds

    for (block_index, start, stop), trial_index in zip(
        partitions, trial_order
    ):
        rest = np.asarray(rest_blocks[block_index], dtype=float)[start:stop]
        idle_start = now()
        scores.extend(rest.tolist())
        idle.append((idle_start, now()))
        rest_windows += len(rest)

        scores.extend([reset_value] * guard)
        guard_windows += guard
        event = np.asarray(command_trials[int(trial_index)], dtype=float)
        command_start = now()
        scores.extend(event.tolist())
        command.append((command_start, now()))
        scores.extend([reset_value] * guard)
        guard_windows += guard

    values = np.asarray(scores, dtype=float)
    times = np.arange(len(values), dtype=float) * step_seconds
    return (
        ScoredSequence(
            times=times,
            scores=values,
            idle_intervals=np.asarray(idle, dtype=float),
            command_intervals=np.asarray(command, dtype=float),
        ),
        InjectedStreamAudit(
            rest_windows=rest_windows,
            command_events=event_count,
            guard_windows=guard_windows,
            rest_chunks=len(partitions),
        ),
    )
