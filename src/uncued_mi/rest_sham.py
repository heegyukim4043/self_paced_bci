"""Block-preserving rest-score sham for a memoryless 1-second score model.

This score-level shortcut is valid only when each window score depends on its
own EEG window (for example, fixed Riemannian LR), not for a recurrent model.
It replaces every trial segment, including the prefix, while keeping the
observed rest schedule and all guard positions fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RestScoreShamLibrary:
    runs_by_block: tuple[tuple[np.ndarray, ...], ...]
    command_score_indices: tuple[np.ndarray, ...]
    unique_source_score_counts: tuple[int, ...]
    repeated_source_score_count: int
    max_repeated_score_difference: float

    def draw(self, observed_raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Replace each complete trial's window scores with contiguous rest."""
        sham = np.asarray(observed_raw, dtype=float).copy()
        for indices in self.command_score_indices:
            length = len(indices)
            blocks = [
                block for block, runs in enumerate(self.runs_by_block)
                if any(len(run) >= length for run in runs)
            ]
            if not blocks:
                raise ValueError(f"no unique rest run supports {length} scores")
            block = int(rng.choice(blocks))
            runs = self.runs_by_block[block]
            start_counts = np.asarray([
                max(0, len(run) - length + 1) for run in runs
            ], dtype=int)
            choice = int(rng.integers(start_counts.sum()))
            run_index = int(np.searchsorted(np.cumsum(start_counts), choice, side="right"))
            offset = choice - int(start_counts[:run_index].sum())
            sham[indices] = runs[run_index][offset : offset + length]
        return sham


def build_rest_score_sham_library(
    raw_scores: np.ndarray,
    starts_seconds: np.ndarray,
    ends_seconds: np.ndarray,
    events: np.ndarray,
    rest_source_ids: np.ndarray,
    rest_source_spans: np.ndarray,
    sample_count: int,
    sfreq: float = 250.,
    step_samples: int = 25,
    repeated_tolerance: float = 1e-4,
) -> RestScoreShamLibrary:
    """Index genuine contiguous rest scores by original block/sample support.

    Repeated replay copies are deduplicated by source block and source window
    end sample. The original rest EEG was filtered once before replay, so the
    same source window must produce the same memoryless score each time.
    """
    raw_scores = np.asarray(raw_scores, dtype=float)
    starts = np.rint(np.asarray(starts_seconds) * sfreq).astype(np.int64)
    ends = np.rint(np.asarray(ends_seconds) * sfreq).astype(np.int64)
    events = np.asarray(events, dtype=np.int64)
    rest_source_ids = np.asarray(rest_source_ids, dtype=np.int64)
    rest_source_spans = np.asarray(rest_source_spans, dtype=np.int64)
    if not (raw_scores.ndim == starts.ndim == ends.ndim == 1):
        raise ValueError("scores and window coordinates must be 1-D")
    if not (len(raw_scores) == len(starts) == len(ends)):
        raise ValueError("score/window coordinate lengths differ")
    if events.ndim != 2 or events.shape[1] < 8:
        raise ValueError("expected replay events with sample coordinates")
    if len(events) != len(rest_source_ids) or rest_source_spans.shape != (len(events), 2):
        raise ValueError("rest provenance is incomplete")
    if not np.isfinite(raw_scores).all():
        raise ValueError("all raw scores must be finite")
    if set(rest_source_ids.tolist()) != {0, 1, 2}:
        raise ValueError("expected three post-pretrain original rest blocks")

    source_maps: list[dict[int, float]] = [dict() for _ in range(3)]
    repeated = 0
    max_difference = 0.
    command_indices = []
    for index, event in enumerate(events):
        rest_start, rest_stop, trial_start = map(int, event[[2, 3, 4]])
        source_start, source_stop = map(int, rest_source_spans[index])
        if rest_stop != trial_start or rest_stop - rest_start != source_stop - source_start:
            raise AssertionError("rest replay/source support mismatch")
        rest_mask = (starts >= rest_start) & (ends <= rest_stop)
        block_map = source_maps[int(rest_source_ids[index])]
        for score_index in np.flatnonzero(rest_mask):
            original_end = source_start + int(ends[score_index]) - rest_start
            value = float(raw_scores[score_index])
            if original_end in block_map:
                repeated += 1
                difference = abs(value - block_map[original_end])
                max_difference = max(max_difference, difference)
                if difference > repeated_tolerance:
                    raise AssertionError("repeated original rest score changed")
            else:
                block_map[original_end] = value
        trial_stop = int(events[index + 1, 2]) if index + 1 < len(events) else sample_count
        selected = np.flatnonzero((starts >= trial_start) & (ends <= trial_stop))
        if not len(selected):
            raise AssertionError("trial has no fully supported score windows")
        if np.any(np.diff(selected) != 1):
            raise AssertionError("trial score support is not contiguous")
        command_indices.append(selected)

    runs_by_block = []
    for source_map in source_maps:
        runs_list = []
        # Rest chunks can begin at different EEG-sample phases relative to the
        # 0.1-s replay grid. A single source block therefore contains several
        # interleaved 25-sample grids; split each residue independently.
        for residue in range(step_samples):
            sorted_ends = np.asarray(sorted(
                end for end in source_map if end % step_samples == residue
            ), dtype=np.int64)
            if not len(sorted_ends):
                continue
            cut = np.flatnonzero(np.diff(sorted_ends) != step_samples) + 1
            runs_list.extend(
                np.asarray([source_map[int(end)] for end in group], dtype=float)
                for group in np.split(sorted_ends, cut) if len(group)
            )
        runs = tuple(runs_list)
        runs_by_block.append(runs)
    return RestScoreShamLibrary(
        runs_by_block=tuple(runs_by_block),
        command_score_indices=tuple(command_indices),
        unique_source_score_counts=tuple(len(group) for group in source_maps),
        repeated_source_score_count=repeated,
        max_repeated_score_difference=max_difference,
    )


def ewma_z_many(raw: np.ndarray, mean: float, var: float,
                allowed: np.ndarray, step_seconds: float = .1,
                half_life_seconds: float = 30.) -> np.ndarray:
    """Causal EWMA over independent sham rows; matches scalar evaluation."""
    raw = np.asarray(raw, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if raw.ndim != 2 or allowed.shape != (raw.shape[1],):
        raise ValueError("raw must be [replicate,time], allowed must be [time]")
    alpha = 1. - 2. ** (-step_seconds / half_life_seconds)
    floor = .25 * np.sqrt(max(var, 1e-8))
    running_mean = np.full(raw.shape[0], mean, dtype=float)
    running_var = np.full(raw.shape[0], var, dtype=float)
    out = np.empty_like(raw)
    for index in range(raw.shape[1]):
        x = raw[:, index]
        out[:, index] = (x - running_mean) / np.maximum(
            np.sqrt(np.maximum(running_var, 0.)), floor
        )
        if allowed[index]:
            difference = x - running_mean
            running_mean += alpha * difference
            running_var = (1. - alpha) * (
                running_var + alpha * difference * difference
            )
    return out


def guard_local_ma_many(z: np.ndarray, allowed: np.ndarray,
                        ma_steps: int = 5) -> np.ndarray:
    """Segment-local causal MA over [replicate,time] score arrays."""
    z = np.asarray(z, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if z.ndim != 2 or allowed.shape != (z.shape[1],):
        raise ValueError("z must be [replicate,time], allowed must be [time]")
    if ma_steps < 1 or not np.isfinite(z).all():
        raise ValueError("ma_steps must be positive and z finite")
    out = np.full(z.shape, np.nan)
    changes = np.diff(np.r_[False, allowed, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(changes == 1),
                           np.flatnonzero(changes == -1)):
        if stop - start < ma_steps:
            continue
        cumulative = np.cumsum(z[:, start:stop], axis=1)
        prior = np.concatenate([
            np.zeros((len(z), 1)), cumulative[:, : -ma_steps]
        ], axis=1)
        out[:, start + ma_steps - 1:stop] = (
            cumulative[:, ma_steps - 1:] - prior
        ) / ma_steps
    return out


def guard_rearmed_alarms_many(ma: np.ndarray, threshold: float,
                              allowed: np.ndarray, dwell_steps: int = 3,
                              refractory_steps: int = 10) -> np.ndarray:
    """Vectorized-over-replicates form of the frozen edge-armed detector."""
    ma = np.asarray(ma, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if ma.ndim != 2 or allowed.shape != (ma.shape[1],):
        raise ValueError("ma must be [replicate,time], allowed must be [time]")
    alarms = np.zeros(ma.shape, dtype=bool)
    armed = np.zeros(ma.shape[0], dtype=bool)
    count = np.zeros(ma.shape[0], dtype=np.int32)
    blocked_until = np.full(ma.shape[0], -1, dtype=np.int32)
    for index in range(ma.shape[1]):
        if not allowed[index]:
            armed[:] = False
            count[:] = 0
            continue
        values = ma[:, index]
        valid = np.isfinite(values)
        was_armed = armed.copy()
        below = valid & (values < threshold)
        newly_armed = ~was_armed & below
        armed[~valid] = False
        armed[newly_armed] = True
        count[~valid | below] = 0
        high = valid & was_armed & (values >= threshold)
        count[high] += 1
        fire = high & (count == dwell_steps) & (index > blocked_until)
        alarms[fire, index] = True
        blocked_until[fire] = index + refractory_steps
    return alarms
