from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_IDLE = 0
STATE_FIXATION = 1
STATE_MI = 2


@dataclass(frozen=True)
class PseudoUncuedStream:
    signal: np.ndarray
    state: np.ndarray
    time_seconds: np.ndarray
    events: np.ndarray
    event_names: tuple[str, ...]
    trial_labels: np.ndarray
    trial_ids: np.ndarray
    rest_source_ids: np.ndarray


def fixed_budget_exponential_chunks(
    blocks: list[np.ndarray],
    *,
    chunks_per_block: tuple[int, ...],
    sfreq: float,
    minimum_seconds: float = 5.0,
    seed: int = 0,
    return_spans: bool = False,
) -> tuple[list[np.ndarray], np.ndarray] | tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Partition disjoint rest blocks with exponential relative durations.

    Conditioning exponential draws on each block's fixed duration preserves an
    unpredictable, right-skewed ordering while making total idle exposure
    identical across subjects. Within one call, no sample is reused and no
    chunk crosses a recording-block boundary. Repeated calls can reuse all
    original rest samples.
    """
    if len(blocks) != len(chunks_per_block):
        raise ValueError("chunks_per_block must match blocks")
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    sources: list[int] = []
    spans: list[tuple[int, int]] = []
    minimum = int(round(minimum_seconds * sfreq))
    for source, (block, count) in enumerate(zip(blocks, chunks_per_block)):
        block = np.asarray(block)
        if count < 1 or len(block) < count * minimum:
            raise ValueError("Rest block cannot satisfy the minimum duration")
        extra = len(block) - count * minimum
        weights = rng.exponential(1.0, count)
        allocations = np.floor(extra * weights / weights.sum()).astype(int)
        allocations[-1] += extra - int(allocations.sum())
        lengths = allocations + minimum
        start = 0
        for length in lengths:
            stop = start + int(length)
            chunks.append(block[start:stop])
            sources.append(source)
            spans.append((start, stop))
            start = stop
        assert start == len(block)
    order = rng.permutation(len(chunks))
    result = ([chunks[index] for index in order], np.asarray(sources, dtype=np.int8)[order])
    if return_spans:
        return *result, np.asarray(spans, dtype=np.int64)[order]
    return result


def _draw_rest_chunk(
    blocks: list[np.ndarray],
    duration_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    eligible = [index for index, block in enumerate(blocks) if len(block) >= duration_samples]
    if not eligible:
        raise ValueError("No resting block is long enough for the requested chunk")
    block_id = int(rng.choice(eligible))
    block = blocks[block_id]
    start = int(rng.integers(0, len(block) - duration_samples + 1))
    return np.asarray(block[start : start + duration_samples]), block_id


def build_pseudo_uncued_raw_stream(
    *,
    continuous_trial_signal: np.ndarray,
    cue_samples: np.ndarray,
    labels: np.ndarray,
    rest_blocks: list[np.ndarray],
    sfreq: float,
    trial_indices: np.ndarray,
    fixation_start_seconds: float = -3.0,
    task_stop_seconds: float = 4.0,
    mi_start_seconds: float = 0.5,
    mi_stop_seconds: float = 3.5,
    minimum_rest_seconds: float = 1.0,
    maximum_rest_seconds: float = 4.0,
    seed: int = 0,
    prepared_rest_chunks: list[np.ndarray] | None = None,
    prepared_rest_source_ids: np.ndarray | None = None,
) -> PseudoUncuedStream:
    """Create `random rest -> intact fixation/task trial` raw pseudo-streams.

    Rest/trial seams are intentionally retained as part of the simulation and
    explicitly annotated. MI ground truth is cue+mi_start to cue+mi_stop.
    """
    signal = np.asarray(continuous_trial_signal)
    cue_samples = np.asarray(cue_samples, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    trial_indices = np.asarray(trial_indices, dtype=np.int64).reshape(-1)
    blocks = [np.asarray(block) for block in rest_blocks]
    if signal.ndim != 2 or not blocks or any(block.ndim != 2 for block in blocks):
        raise ValueError("Signals and rest blocks must be channel-by-time matrices")
    channels = signal.shape[1]
    if any(block.shape[1] != channels for block in blocks):
        raise ValueError("Trial and rest channel counts differ")
    if len(cue_samples) != len(labels):
        raise ValueError("cue_samples and labels differ in length")
    if not (0 < minimum_rest_seconds <= maximum_rest_seconds):
        raise ValueError("Invalid resting-duration range")

    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    states: list[np.ndarray] = []
    event_rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    output_labels: list[int] = []
    output_trials: list[int] = []
    output_rest_sources: list[int] = []
    cursor = 0

    if prepared_rest_chunks is not None and len(prepared_rest_chunks) != len(trial_indices):
        raise ValueError("Prepared rest chunks must match trial count")
    for output_id, trial_id in enumerate(trial_indices):
        cue = int(cue_samples[trial_id])
        trial_start = cue + int(round(fixation_start_seconds * sfreq))
        trial_stop = cue + int(round(task_stop_seconds * sfreq))
        if trial_start < 0 or trial_stop > len(signal):
            raise ValueError(f"Trial {trial_id} exceeds continuous recording")

        if prepared_rest_chunks is None:
            rest_seconds = float(rng.uniform(minimum_rest_seconds, maximum_rest_seconds))
            rest_samples = int(round(rest_seconds * sfreq))
            rest, rest_source = _draw_rest_chunk(blocks, rest_samples, rng)
        else:
            rest = np.asarray(prepared_rest_chunks[output_id])
            rest_source = int(prepared_rest_source_ids[output_id])
        rest_start = cursor
        rest_stop = rest_start + len(rest)
        pieces.append(rest)
        states.append(np.full(len(rest), STATE_IDLE, dtype=np.int8))
        cursor = rest_stop

        segment = signal[trial_start:trial_stop]
        fixation_samples = int(round(-fixation_start_seconds * sfreq))
        mi_start = fixation_samples + int(round(mi_start_seconds * sfreq))
        mi_stop = fixation_samples + int(round(mi_stop_seconds * sfreq))
        segment_state = np.full(len(segment), STATE_FIXATION, dtype=np.int8)
        segment_state[mi_start:mi_stop] = STATE_MI
        trial_stream_start = cursor
        cue_stream = trial_stream_start + fixation_samples
        mi_stream_start = trial_stream_start + mi_start
        mi_stream_stop = trial_stream_start + mi_stop
        trial_stream_stop = trial_stream_start + len(segment)
        pieces.append(segment)
        states.append(segment_state)
        cursor = trial_stream_stop

        event_rows.append((
            output_id, int(trial_id), rest_start, rest_stop,
            trial_stream_start, cue_stream, mi_stream_start, mi_stream_stop,
        ))
        output_labels.append(int(labels[trial_id]))
        output_trials.append(int(trial_id))
        output_rest_sources.append(rest_source)

    values = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    state = np.concatenate(states)
    events = np.asarray(event_rows, dtype=np.int64)
    assert len(values) == len(state)
    assert np.all(events[:, 3] == events[:, 4])  # explicit artificial seam
    assert np.all(state[events[:, 2]] == STATE_IDLE)
    assert np.all(state[events[:, 6]] == STATE_MI)
    return PseudoUncuedStream(
        signal=values,
        state=state,
        time_seconds=np.arange(len(values), dtype=np.float64) / sfreq,
        events=events,
        event_names=(
            "stream_trial", "source_trial", "rest_start", "rest_stop",
            "trial_start", "cue_onset", "mi_start", "mi_stop",
        ),
        trial_labels=np.asarray(output_labels, dtype=np.int8),
        trial_ids=np.asarray(output_trials, dtype=np.int16),
        rest_source_ids=np.asarray(output_rest_sources, dtype=np.int8),
    )
