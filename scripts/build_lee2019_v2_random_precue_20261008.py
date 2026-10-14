"""E3 Lee v2 EEG stream construction, before any detector/outcome inspection.

Uses the already causal-filtered, pre-rest-normalized chronological cache. This
is a pseudo-uncued simulation, not a recording of self-initiated MI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SFREQ = 250
FADE = 50  # 0.2 s
SEED = 20261008


def bounded_rest_partition(length: int, rng: np.random.Generator) -> np.ndarray:
    """Partition a 5-s+ rest interval into mostly 4–7-s pieces.

    The 7–8-s interval cannot be partitioned into pieces all in [4, 7] s;
    those intervals stay intact and are flagged by the audit, not discarded.
    """
    low, high = 4 * SFREQ, 7 * SFREQ
    n_min = int(np.ceil(length / high))
    n_max = length // low
    if n_min > n_max:
        return np.asarray([length], dtype=np.int64)
    # Match a 4+U-second trial's mean length (5.5 s), rather than packing the
    # maximum possible number of short rest pieces into every interval.
    n = min(range(n_min, n_max + 1),
            key=lambda count: abs(length / count - 5.5 * SFREQ))
    lengths = np.empty(n, dtype=np.int64)
    remaining = length
    for i in range(n):
        left = n - i - 1
        smallest = max(low, remaining - left * high)
        largest = min(high, remaining - left * low)
        lengths[i] = int(rng.integers(smallest, largest + 1)) if left else remaining
        remaining -= int(lengths[i])
    assert remaining == 0 and lengths.sum() == length
    assert np.all((lengths >= low) & (lengths <= high))
    return lengths


def append_causal_fade(pieces: list[np.ndarray], piece: np.ndarray,
                       cursor: int, fade_intervals: list[tuple[int, int]]) -> int:
    """Blend incoming samples with a tail already observed before the seam."""
    value = np.asarray(piece, dtype=np.float32).copy()
    if pieces:
        if len(pieces[-1]) < FADE or len(value) < FADE:
            raise ValueError("piece too short for causal fade")
        previous_tail = pieces[-1][-FADE:].copy()
        alpha = (np.arange(1, FADE + 1, dtype=np.float32) / FADE)[:, None]
        value[:FADE] = (1 - alpha) * previous_tail + alpha * value[:FADE]
        fade_intervals.append((cursor, cursor + FADE))
    pieces.append(value)
    return cursor + len(value)


def construct(data: dict[str, np.ndarray], subject: int, session: int,
              seed: int = SEED) -> dict[str, np.ndarray]:
    old = np.asarray(data["signal"], dtype=np.float32)
    events = np.asarray(data["events"], dtype=np.int64)
    ids = np.asarray(data["trial_ids"], dtype=np.int64)
    rest_spans = np.asarray(data["rest_source_spans_250hz"], dtype=np.int64)
    source_spans = np.asarray(data["trial_source_spans_250hz"], dtype=np.int64)
    rest_ids = np.asarray(data["rest_source_ids"], dtype=np.int8)
    metadata = json.loads(str(data["metadata_json"]))
    if (old.ndim != 2 or events.shape != (50, 8) or
            not np.array_equal(ids, np.arange(50, 100)) or
            not np.array_equal(events[:, 1], ids) or
            metadata["task_ground_truth_seconds_from_cue"] != [0.5, 4.0]):
        raise ValueError("not a validated chronological Lee replay")
    if float(data["sfreq"]) != SFREQ:
        raise ValueError("expected 250 Hz")
    rng = np.random.default_rng(np.random.SeedSequence([seed, subject, session]))
    pieces: list[np.ndarray] = []
    states: list[np.ndarray] = []
    fades: list[tuple[int, int]] = []
    seams: list[tuple[int, int]] = []  # sample, kind: 0 rest-rest, 1 trial start, 2 trial end
    rows: list[tuple[int, ...]] = []
    trial_spans: list[tuple[int, int]] = []
    crop_samples: list[int] = []
    rest_piece_lengths: list[int] = []
    rest_piece_map: list[tuple[int, int, int, int]] = []
    unmatched_rest: list[int] = []
    cursor = 0
    support_error = 0.0
    for output_id, index in enumerate(range(20, 50)):
        event = events[index]
        rest = old[int(event[2]):int(event[3])]
        lengths = bounded_rest_partition(len(rest), rng)
        if len(lengths) == 1 and not (4 * SFREQ <= lengths[0] <= 7 * SFREQ):
            unmatched_rest.append(output_id)
        rest_start = cursor
        offsets = np.r_[0, np.cumsum(lengths)[:-1]]
        order = rng.permutation(len(lengths))
        # A decoy must be an actual splice of nonadjacent original support.
        # Reversal is always feasible for n>1 if random draws fail.
        for _ in range(20):
            if len(order) == 1 or np.all(offsets[order[1:]] !=
                                         offsets[order[:-1]] + lengths[order[:-1]]):
                break
            order = rng.permutation(len(lengths))
        else:
            order = np.arange(len(lengths) - 1, -1, -1)
        for part_index, part_id in enumerate(order):
            size = int(lengths[part_id])
            offset = int(offsets[part_id])
            part = rest[offset:offset + size]
            if pieces:
                seams.append((cursor, 0 if part_index else 2))
            out_start = cursor
            cursor = append_causal_fade(pieces, part, cursor, fades)
            rest_piece_map.append((out_start, cursor,
                                   int(event[2] + offset), int(event[2] + offset + size)))
            states.append(np.zeros(len(part), dtype=np.int8))
            support_error = max(support_error, float(np.max(np.abs(
                pieces[-1][FADE:] - part[FADE:]))))
            rest_piece_lengths.append(int(size))
        assert int(lengths.sum()) == len(rest)
        rest_stop = cursor
        # Draw at the native 250-Hz grid; U=0 and U=3 s are both allowed.
        u = int(rng.integers(0, 3 * SFREQ + 1))
        cue_old = int(event[5])
        start_old = cue_old - u
        stop_old = cue_old + 4 * SFREQ
        if start_old < int(event[4]) or stop_old > len(old):
            raise AssertionError("trial crop escaped original continuous support")
        trial = old[start_old:stop_old]
        seams.append((cursor, 1))
        trial_start = cursor
        cursor = append_causal_fade(pieces, trial, cursor, fades)
        support_error = max(support_error, float(np.max(np.abs(
            pieces[-1][FADE:] - trial[FADE:]))))
        cue_new = trial_start + u
        onset = cue_new + SFREQ // 2
        mi_stop = cue_new + 4 * SFREQ
        state = np.ones(len(trial), dtype=np.int8)
        state[onset - trial_start:mi_stop - trial_start] = 2
        states.append(state)
        rows.append((output_id, int(ids[index]), rest_start, rest_stop,
                     trial_start, cue_new, onset, mi_stop))
        old_span = source_spans[index]
        trial_spans.append((int(old_span[0] + (start_old - int(event[4]))),
                            int(old_span[1])))
        crop_samples.append(u)
    signal = np.concatenate(pieces)
    state = np.concatenate(states)
    rows_array = np.asarray(rows, dtype=np.int64)
    seam_array = np.asarray(seams, dtype=np.int64)
    assert len(signal) == len(state)
    assert np.array_equal(rows_array[:, 1], np.arange(70, 100))
    assert np.array_equal(rows_array[:, 6] - rows_array[:, 5],
                          np.full(30, SFREQ // 2))
    assert np.array_equal(rows_array[:, 7] - rows_array[:, 5],
                          np.full(30, 4 * SFREQ))
    assert support_error == 0.0
    return {
        "signal": signal, "state": state, "events": rows_array,
        "seams": seam_array, "fade_intervals": np.asarray(fades, dtype=np.int64),
        "trial_ids": ids[20:], "trial_labels": np.asarray(data["trial_labels"])[20:],
        "trial_source_spans_250hz": np.asarray(trial_spans, dtype=np.int64),
        "rest_source_ids": rest_ids[20:],
        "rest_source_spans_250hz": rest_spans[20:],
        "precue_samples": np.asarray(crop_samples, dtype=np.int16),
        "rest_piece_lengths": np.asarray(rest_piece_lengths, dtype=np.int32),
        "rest_piece_source_map": np.asarray(rest_piece_map, dtype=np.int64),
        "unmatched_rest_intervals": np.asarray(unmatched_rest, dtype=np.int16),
        "sfreq": np.float32(SFREQ), "channels": np.asarray(data["channels"]),
        "metadata_json": np.asarray(json.dumps({
            "stage": "E3 G0 stream generation; G1/G2/G3 untested",
            "dataset": "Lee2019 cue-paced pseudo-uncued simulation",
            "subject": subject, "session": session, "seed": seed,
            "seed_derivation": "SeedSequence([seed,subject,session])",
            "trial_ids": [70, 99], "trial_order": "original",
            "input": "causal filtered chronological cache; normalization from pre_train",
            "pre_event_crop_samples": [0, 750], "fade_samples": FADE,
            "rest_reuse": "future 3 folds repartition the same 168 s unique rest",
            "decoy_construction": "within-rest-chunk permutation of source-disjoint 4-7s pieces",
            "seam_kinds": {"0": "rest-rest", "1": "trial-start", "2": "trial-end"},
            "seam_guard": False, "hidden_state_reset": False,
            "source_support_exact_outside_fade": support_error == 0,
            "claim_limit": "not self-initiated onset or deployment performance",
        }, sort_keys=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path,
                        default=Path("data/interim/lee2019_revalidation_chronological"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/interim/lee2019_v2_random_precue_balanced_20261008"))
    parser.add_argument("--subjects", nargs="+", type=int, default=[3])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.all_subjects:
        args.subjects = list(range(1, 55))
    for session in args.sessions:
        for subject in args.subjects:
            source = args.input_root / f"session{session}/sub-{subject:02d}.npz"
            target = args.output_root / f"session{session}/sub-{subject:02d}.npz"
            if target.exists():
                if args.skip_existing:
                    continue
                raise FileExistsError(target)
            with np.load(source, allow_pickle=False) as data:
                result = construct({key: data[key] for key in data.files},
                                   subject, session, args.seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **result)
            print(json.dumps({"target": str(target), "samples": len(result["signal"]),
                              "decoys": int(np.sum(result["seams"][:, 1] == 0)),
                              "unmatched_rest": len(result["unmatched_rest_intervals"])}),
                  flush=True)


if __name__ == "__main__":
    main()
