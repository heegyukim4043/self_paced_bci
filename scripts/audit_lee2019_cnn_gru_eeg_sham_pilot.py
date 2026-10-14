"""EEG-level rest sham for a frozen CNN-GRU on chronological Lee2019 replay.

This is a bounded, single-session diagnostic. It is not the cohort A1 Type-I
validation and must not be interpreted as deployable self-paced performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_lee2019_cnn_gru_loso_arm_a_stream_v2 import (
    SCREEN,
    SFREQ,
    SUBJECTS,
    _cached_source_training,
    _rebuild_exact_target_stream,
    _window_ends,
    predict_signal,
    train_source_pool,
)


def original_rest_library(signal: np.ndarray, events: np.ndarray,
                          source_ids: np.ndarray, spans: np.ndarray,
                          trial_samples: int) -> tuple[tuple[np.ndarray, ...],
                                                       tuple[np.ndarray, ...], float]:
    """Deduplicate replayed rest by original block and exact EEG sample index."""
    blocks: list[np.ndarray] = []
    available: list[np.ndarray] = []
    max_difference = 0.0
    for block_id in range(3):
        rows = np.flatnonzero(source_ids == block_id)
        if not len(rows):
            raise AssertionError(f"missing original rest block {block_id}")
        size = int(np.max(spans[rows, 1]))
        samples = np.zeros((*signal.shape[:2], size), dtype=np.float32)
        covered = np.zeros(size, dtype=bool)
        for row in rows:
            start, stop = map(int, spans[row])
            replay_start, replay_stop = map(int, events[row, 2:4])
            if stop - start != replay_stop - replay_start:
                raise AssertionError("rest provenance length mismatch")
            values = signal[..., replay_start:replay_stop]
            overlap = covered[start:stop]
            if np.any(overlap):
                difference = float(np.max(np.abs(
                    samples[..., start:stop][..., overlap] - values[..., overlap]
                )))
                max_difference = max(max_difference, difference)
                if difference > 1e-5:
                    raise AssertionError("replayed copies of source rest EEG differ")
            samples[..., start:stop] = values
            covered[start:stop] = True
        # A candidate cannot cross any gap in the original rest sample support.
        missing = (~covered).astype(np.int32)
        cumulative = np.r_[0, np.cumsum(missing)]
        starts = np.flatnonzero(cumulative[trial_samples:] ==
                               cumulative[:-trial_samples]).astype(np.int32)
        blocks.append(samples)
        available.append(starts)
    if not any(len(starts) for starts in available):
        raise AssertionError("no contiguous source rest supports a full trial")
    return tuple(blocks), tuple(available), max_difference


def draw_rest_eeg_sham(signal: np.ndarray, events: np.ndarray,
                       blocks: tuple[np.ndarray, ...],
                       starts: tuple[np.ndarray, ...],
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sham = signal.copy()
    source = np.full((len(events), 2), -1, dtype=np.int32)
    for row, event in enumerate(events):
        target_start = int(event[4])
        target_stop = int(events[row + 1, 2]) if row + 1 < len(events) else signal.shape[-1]
        length = target_stop - target_start
        eligible = [i for i, choices in enumerate(starts) if len(choices) and
                    choices[-1] + length <= blocks[i].shape[-1]]
        if not eligible:
            raise AssertionError("no eligible rest block for trial segment")
        block = int(rng.choice(eligible))
        legal = starts[block]
        # The current locked replay uses a constant 7-s trial support.
        if length != 1750:
            raise AssertionError("trial support changed; rebuild source start library")
        offset = int(rng.choice(legal))
        sham[..., target_start:target_stop] = blocks[block][..., offset:offset + length]
        source[row] = block, offset
    return sham, source


def draw_cue_preserving_eeg_sham(
    signal: np.ndarray, events: np.ndarray, blocks: tuple[np.ndarray, ...],
    starts: tuple[np.ndarray, ...], rng: np.random.Generator,
    sfreq: int = 250, preserve_after_cue_seconds: float = .5,
    crossfade_seconds: float = .2,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve pre-cue/cue EEG, then replace task EEG with original rest.

    This is an exploratory cue-preserving control, not a proven MI-specific
    null. The cross-fade reduces but cannot rule out splice transients.
    """
    if preserve_after_cue_seconds < 0 or crossfade_seconds < 0:
        raise ValueError("negative cue-preservation or cross-fade duration")
    sham = signal.copy()
    source = np.full((len(events), 2), -1, dtype=np.int32)
    fade = int(round(crossfade_seconds * sfreq))
    for row, event in enumerate(events):
        trial_start = int(event[4])
        target_start = int(event[5] + round(preserve_after_cue_seconds * sfreq))
        target_stop = int(events[row + 1, 2]) if row + 1 < len(events) else signal.shape[-1]
        length = target_stop - target_start
        trial_length = target_stop - trial_start
        if trial_length != 1750:
            raise AssertionError("trial support changed; rebuild source start library")
        if length <= fade:
            raise AssertionError("no rest-replacement support after cue-preserved prefix")
        eligible = [i for i, choices in enumerate(starts) if len(choices) and
                    choices[-1] + trial_length <= blocks[i].shape[-1]]
        if not eligible:
            raise AssertionError("no eligible original rest for cue-preserving sham")
        block = int(rng.choice(eligible))
        offset = int(rng.choice(starts[block]))
        # The paired rest-only sham assigns ``offset`` to trial_start. Preserve
        # the same rest sample at each later absolute time, not just the same
        # block/start draw (which previously shifted the rest waveform).
        replacement_start = offset + target_start - trial_start
        replacement = blocks[block][..., replacement_start:replacement_start + length]
        if replacement.shape[-1] != length:
            raise AssertionError("rest replacement has incomplete support")
        if fade:
            weight = np.linspace(0., 1., fade, endpoint=True, dtype=np.float32)
            sham[..., target_start:target_start + fade] = (
                signal[..., target_start:target_start + fade] * (1. - weight)
                + replacement[..., :fade] * weight
            )
        sham[..., target_start + fade:target_stop] = replacement[..., fade:]
        source[row] = block, offset
    return sham, source


def draw_rest_rest_splice_eeg_sham(
    signal: np.ndarray, events: np.ndarray, blocks: tuple[np.ndarray, ...],
    starts: tuple[np.ndarray, ...], rng_post: np.random.Generator,
    rng_pre: np.random.Generator, sfreq: int = 250,
    preserve_after_cue_seconds: float = .5, crossfade_seconds: float = .2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join two original-rest trajectories at the cue-preserving splice.

    The post-splice rest draw uses the same RNG sequence and absolute sample
    alignment as rest-only/cue-preserving shams. A separate RNG picks the
    pre-splice rest, so post draws remain paired across sham families.
    """
    if preserve_after_cue_seconds < 0 or crossfade_seconds < 0:
        raise ValueError("negative cue-preservation or cross-fade duration")
    sham = signal.copy()
    post_source = np.full((len(events), 2), -1, dtype=np.int32)
    pre_source = np.full((len(events), 2), -1, dtype=np.int32)
    fade = int(round(crossfade_seconds * sfreq))
    for row, event in enumerate(events):
        trial_start = int(event[4])
        splice_start = int(event[5] + round(preserve_after_cue_seconds * sfreq))
        trial_stop = int(events[row + 1, 2]) if row + 1 < len(events) else signal.shape[-1]
        length = trial_stop - trial_start
        pre_length = splice_start - trial_start
        if length != 1750 or pre_length < 0 or pre_length + fade >= length:
            raise AssertionError("trial/splice support changed")
        eligible = [i for i, choices in enumerate(starts) if len(choices) and
                    choices[-1] + length <= blocks[i].shape[-1]]
        if not eligible:
            raise AssertionError("no eligible original rest for rest-rest splice")
        post_block = int(rng_post.choice(eligible))
        post_offset = int(rng_post.choice(starts[post_block]))
        pre_block = int(rng_pre.choice(eligible))
        pre_offset = int(rng_pre.choice(starts[pre_block]))
        while (pre_block, pre_offset) == (post_block, post_offset):
            pre_block = int(rng_pre.choice(eligible))
            pre_offset = int(rng_pre.choice(starts[pre_block]))
        post = blocks[post_block][..., post_offset:post_offset + length]
        pre = blocks[pre_block][..., pre_offset:pre_offset + length]
        if post.shape[-1] != length or pre.shape[-1] != length:
            raise AssertionError("incomplete rest support")
        sham[..., trial_start:trial_stop] = post
        sham[..., trial_start:splice_start] = pre[..., :pre_length]
        if fade:
            weight = np.linspace(0., 1., fade, endpoint=True, dtype=np.float32)
            sham[..., splice_start:splice_start + fade] = (
                pre[..., pre_length:pre_length + fade] * (1. - weight)
                + post[..., pre_length:pre_length + fade] * weight
            )
        post_source[row] = post_block, post_offset
        pre_source[row] = pre_block, pre_offset
    return sham, post_source, pre_source


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    replay_path = args.stream_root / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    score_path = args.score_root / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    with np.load(replay_path, allow_pickle=False) as replay:
        events = np.asarray(replay["events"], dtype=np.int64)
        source_ids = np.asarray(replay["rest_source_ids"], dtype=np.int64)
        source_spans = np.asarray(replay["rest_source_spans_250hz"], dtype=np.int64)
    with np.load(score_path, allow_pickle=False) as score:
        cached_raw = np.asarray(score["raw_scores"], dtype=np.float32)
        anchor = np.asarray(score["anchor_scores"], dtype=np.float64)
        times = np.asarray(score["times"], dtype=np.float64)
        allowed = np.asarray(score["allowed"], dtype=bool)
        if not np.array_equal(score["events"], events):
            raise AssertionError("chronological score cache/event mismatch")
        if not np.array_equal(score["seeds"], args.seeds):
            raise AssertionError("score cache and model training seeds differ")
    signal, rebuilt_events, target_anchor = _rebuild_exact_target_stream(
        args.data_root, args.stream_root, args.subject, args.session,
    )
    if not np.array_equal(rebuilt_events, events):
        raise AssertionError("rebuilt EEG/event schedule differs")
    ends = _window_ends(signal.shape[-1])
    if not np.array_equal(ends / SFREQ, times):
        raise AssertionError("score/rebuilt EEG window grids differ")
    blocks, starts, source_difference = original_rest_library(
        signal, events, source_ids, source_spans, 1750,
    )
    print("source rest EEG reconstructed; candidate starts:",
          [len(row) for row in starts], "replay copy max difference:",
          source_difference, flush=True)
    source_pool = {
        subject: _cached_source_training(
            args.data_root, args.train_cache_root, args.prepared_source_root,
            subject, args.session, 100,
        ) for subject in SUBJECTS
    }
    train_blocks = [source_pool[subject] for subject in SUBJECTS
                    if subject != args.subject]
    models = []
    observed_seed_scores = []
    anchor_seed_scores = []
    anchor_ends = _window_ends(target_anchor.shape[-1], first_end=3 * SFREQ)
    for index, seed in enumerate(args.seeds, 1):
        print(f"training frozen CNN-GRU seed {index}/{len(args.seeds)}", flush=True)
        model = train_source_pool(train_blocks, seed, args.epochs, device)
        models.append(model)
        observed_seed_scores.append(predict_signal(model, signal, device, ends=ends))
        anchor_seed_scores.append(predict_signal(
            model, target_anchor, device, ends=anchor_ends,
        ))
    observed = np.mean(observed_seed_scores, axis=0)
    anchor_rebuilt = np.mean(anchor_seed_scores, axis=0)
    score_max_difference = float(np.max(np.abs(observed - cached_raw)))
    anchor_max_difference = float(np.max(np.abs(anchor_rebuilt - anchor)))
    if score_max_difference > 1e-5 or anchor_max_difference > 1e-5:
        raise AssertionError("retrained model does not reproduce saved score cache")
    rng = np.random.default_rng(args.seed)
    pre_rng = np.random.default_rng(args.seed + 1)
    source_draws = np.full((args.replicates, len(events), 2), -1, dtype=np.int32)
    pre_source_draws = np.full_like(source_draws, -1)
    sham_scores = np.empty((args.replicates, len(times)), dtype=np.float32)
    for draw in range(args.replicates):
        if args.sham_mode == "rest_only":
            sham_eeg, source = draw_rest_eeg_sham(signal, events, blocks, starts, rng)
        elif args.sham_mode == "cue_preserving":
            sham_eeg, source = draw_cue_preserving_eeg_sham(
                signal, events, blocks, starts, rng,
            )
        else:
            sham_eeg, source, pre_source = draw_rest_rest_splice_eeg_sham(
                signal, events, blocks, starts, rng, pre_rng,
            )
            pre_source_draws[draw] = pre_source
        source_draws[draw] = source
        sham_raw = np.mean([
            predict_signal(model, sham_eeg, device, ends=ends)
            for model in models
        ], axis=0)
        sham_scores[draw] = sham_raw
        print(f"EEG-level sham inference {draw + 1}/{args.replicates}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output) + ".npz", sham_raw_scores=sham_scores,
                        source_draws=source_draws,
                        pre_source_draws=pre_source_draws,
                        observed_raw=observed,
                        anchor_scores=anchor, times=times,
                        allowed=allowed.astype(np.uint8), events=events)
    summary = {
        "scope": "single-session EEG-level sham pilot, not full A1 validation",
        "sham_mode": args.sham_mode,
        "rest_rest_splice_limit": "Rest-to-rest splice is an artifact stress control, not cue-matched no-MI ground truth" if args.sham_mode == "rest_rest_splice" else None,
        "cue_preserving_limit": "Preserves EEG through cue+0.5 s and cross-fades for 0.2 s; splice artifact and cue/MI overlap are not ruled out" if args.sham_mode == "cue_preserving" else None,
        "dataset": "Lee2019 cue-relative chronological pseudo-uncued replay",
        "subject": args.subject,
        "session": args.session,
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "replicates": args.replicates,
        "analysis_stage": "EEG-level inference cache only; matched-FP analysis separate",
        "retrained_observed_score_max_difference": score_max_difference,
        "retrained_anchor_score_max_difference": anchor_max_difference,
        "repeated_original_rest_eeg_max_difference": source_difference,
        "unique_candidate_rest_starts_by_block": [len(row) for row in starts],
        "limitation": "Original rest reused in replay idle and sham task support; causal recurrent model fully re-inferred; no cohort CI or Type-I inference.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260729, 20260730, 20260731])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sham-mode", choices=["rest_only", "cue_preserving",
                                                "rest_rest_splice"],
                        default="rest_only")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/lee2019_mi")
    parser.add_argument("--train-cache-root", type=Path, default=ROOT / "data/interim/lee2019_mil_raw")
    parser.add_argument("--prepared-source-root", type=Path,
                        default=ROOT / "data/interim/lee2019_cnn_gru_loso_prepared_source")
    parser.add_argument("--stream-root", type=Path,
                        default=ROOT / "data/interim/lee2019_revalidation_chronological")
    parser.add_argument("--score-root", type=Path,
                        default=ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/performance_revalidation_20261001/"
                        "a1_cnn_gru_eeg_rest_sham_pilot_sub03_s1_fp2")
    args = parser.parse_args()
    if not 1 <= args.subject <= 54 or args.session not in (1, 2):
        raise ValueError("subject/session out of range")
    if args.replicates < 1:
        raise ValueError("invalid analysis grid")
    run(args)


if __name__ == "__main__":
    main()
