"""Build raw Lee2019 random-rest injected pseudo-uncued test streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path, load_lee2019_run
from uncued_mi.pseudo_uncued import (
    build_pseudo_uncued_raw_stream,
    fixed_budget_exponential_chunks,
)
from scripts.prepare_lee2019_mil_raw_cache import preprocess


def build_one(source: Path, target: Path, subject: int, session: int, seed: int,
              chronological: bool = False) -> dict:
    config = ExperimentConfig(window_seconds=1.0, feature_step_seconds=0.1)
    train = load_lee2019_run(source, "train")
    test = load_lee2019_run(source, "test")
    signal = preprocess(train.eeg, train.channels, train.sfreq,
                        config.sensorimotor_channels, config)
    all_rests = [
        preprocess(values, train.channels, train.sfreq,
                   config.sensorimotor_channels, config)
        for values in (train.pre_rest, train.post_rest, test.pre_rest, test.post_rest)
    ]
    transient = int(round(config.transient_seconds * config.target_sfreq))
    anchor = (all_rests[0][transient:] if chronological
              else all_rests[0][transient:-transient])
    all_rests = [values[transient:-transient] for values in all_rests]
    mean = anchor.mean(axis=0)
    std = np.maximum(anchor.std(axis=0, ddof=1), 1e-6)
    signal = ((signal - mean) / std).astype(np.float32)
    all_rests = [((values - mean) / std).astype(np.float32) for values in all_rests]
    # Strictly disjoint: preTrainRest is anchor-only; the other three blocks
    # supply evaluation idle and are never used to estimate normalization.
    rests = all_rests[1:]
    cue_samples = np.rint(
        train.cue_samples / train.sfreq * config.target_sfreq
    ).astype(np.int64)
    rng = np.random.default_rng(seed + 1000 * session + subject)
    # Five folds cover all 50 locked trials exactly once. Rest recordings are
    # repartitioned in every fold; this averages placement but does not create
    # independent rest observations, so inference remains subject-clustered.
    selected_order = np.arange(50, 100) if chronological else rng.permutation(np.arange(50, 100))
    selected_parts, chunk_parts, source_parts, span_parts = [], [], [], []
    for fold in range(5):
        selected_parts.append(selected_order[10 * fold : 10 * (fold + 1)])
        fold_chunks, fold_sources, fold_spans = fixed_budget_exponential_chunks(
            rests, chunks_per_block=(3, 3, 4), sfreq=config.target_sfreq,
            minimum_seconds=5.0,
            seed=seed + 2000 * session + 100 * fold + subject,
            return_spans=True,
        )
        chunk_parts.extend(fold_chunks)
        source_parts.append(fold_sources)
        span_parts.append(fold_spans)
    selected = np.concatenate(selected_parts)
    rest_chunks = chunk_parts
    rest_sources = np.concatenate(source_parts)
    rest_spans = np.concatenate(span_parts) + transient
    stream = build_pseudo_uncued_raw_stream(
        continuous_trial_signal=signal,
        cue_samples=cue_samples,
        labels=train.labels,
        rest_blocks=rests,
        sfreq=config.target_sfreq,
        trial_indices=selected,
        minimum_rest_seconds=5.0,
        maximum_rest_seconds=60.0,
        seed=seed + 1000 * session + subject,
        prepared_rest_chunks=rest_chunks,
        prepared_rest_source_ids=rest_sources,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "Lee2019",
        "phase": "all 50 train locked trials 50-99, one use each across 5 folds",
        "subject": subject,
        "session": session,
        "seed": seed + 1000 * session + subject,
        "rest_duration_model": "fixed-total conditional exponential; minimum 5 s",
        "rest_repartition_folds": 5,
        "idle_seconds_per_fold": 168.0,
        "idle_seconds_total": 840.0,
        "inference_cluster": "subject (folds/rest reuse are not independent)",
        "trial_support_seconds_from_cue": [-3.0, 4.0],
        "mi_ground_truth_seconds_from_cue": [0.5, 3.5],
        "normalization_anchor": "pre_train only (disjoint from evaluation)",
        "rest_sources": ["post_train", "pre_test", "post_test"],
        "artificial_rest_trial_seam_retained": True,
        "claim_limit": "pseudo-uncued simulation; not self-initiated onset",
    }
    if chronological:
        metadata.update({
            "trial_order": "original source trial indices 50-99",
            "prefix_trial_ids": [50, 69],
            "future_trial_ids": [70, 99],
            "rest_span_coordinates": "preprocessed 250-Hz samples within each named source rest block",
            "normalization_anchor_samples_250hz": [transient, int(len(anchor) + transient)],
            "rest_reuse_warning": "Each of 3 source rest blocks is repartitioned 5 times; prefix and future replay rest share original support. Never fit or calibrate on injected prefix rest.",
            "task_ground_truth_seconds_from_cue": [0.5, 4.0],
            "mi_ground_truth_seconds_from_cue": [0.5, 4.0],
            "training_positive_support_seconds_from_cue": [0.5, 3.5],
            "acquisition_warning": "post_train/pre_test/post_test rest was recorded after train-phase future trials; prospective-in-replay only",
        })
    extra = {}
    if chronological:
        all_trial_spans = np.stack((
            cue_samples - int(round(3 * config.target_sfreq)),
            cue_samples + int(round(4 * config.target_sfreq)),
        ), axis=1)
        extra = {
            "rest_source_spans_250hz": rest_spans,
            "trial_source_spans_250hz": all_trial_spans[selected],
            "all_train_trial_source_spans_250hz": all_trial_spans,
        }
    np.savez_compressed(
        target, signal=stream.signal, state=stream.state,
        time_seconds=stream.time_seconds, events=stream.events,
        event_names=np.asarray(stream.event_names),
        trial_labels=stream.trial_labels, trial_ids=stream.trial_ids,
        rest_source_ids=stream.rest_source_ids,
        sfreq=np.float32(config.target_sfreq),
        channels=np.asarray(config.sensorimotor_channels),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **extra,
    )
    rest_durations = (stream.events[:, 3] - stream.events[:, 2]) / config.target_sfreq
    audit = {
        "samples": len(stream.signal),
        "seconds": len(stream.signal) / config.target_sfreq,
        "channels": stream.signal.shape[1],
        "trials": len(stream.events),
        "rest_seconds_total": float(rest_durations.sum()),
        "rest_seconds_min": float(rest_durations.min()),
        "rest_seconds_max": float(rest_durations.max()),
        "mi_events": int(np.sum(np.diff(np.r_[0, stream.state == 2]) == 1)),
        "seams": len(stream.events),
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--output-root", type=Path,
                        default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chronological", action="store_true",
                        help="preserve source trial order and save original rest sample spans")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = Path("data/interim/lee2019_revalidation_chronological" if args.chronological
                                else "data/interim/lee2019_pseudo_uncued_raw")
    if args.chronological and args.output_root.resolve() == Path("data/interim/lee2019_pseudo_uncued_raw").resolve():
        parser.error("chronological replay must not overwrite historical pseudo-uncued cache")
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            target = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if target.exists() and not args.overwrite:
                print(f"exists {target}", flush=True)
                continue
            audit = build_one(lee2019_path(args.data_root, subject, session), target,
                              subject, session, args.seed, args.chronological)
            rows.append({"subject": subject, "session": session, **audit})
            print(json.dumps(rows[-1], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
