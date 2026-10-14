"""A0 source-support audit; reads only chronology-preserving replay metadata."""

import argparse
import json
from pathlib import Path

import numpy as np


def overlap_samples(a, b):
    return int(np.maximum(0, np.minimum(a[:, None, 1], b[None, :, 1])
                          - np.maximum(a[:, None, 0], b[None, :, 0])).sum())


def audit(path, raw_rest_root=None, trial_root=None):
    with np.load(path, allow_pickle=False) as data:
        events = data["events"]
        ids = data["trial_ids"]
        sources = data["rest_source_ids"]
        rest = data["rest_source_spans_250hz"]
        trials = data["all_train_trial_source_spans_250hz"]
        selected_trials = data["trial_source_spans_250hz"]
        metadata = json.loads(str(data["metadata_json"]))
        sfreq = float(data["sfreq"])
        signal = data["signal"] if raw_rest_root is not None or trial_root is not None else None
    assert sfreq == 250 and events.shape == (50, 8) and trials.shape == (100, 2), path
    assert np.array_equal(ids, np.arange(50, 100)), path
    assert np.array_equal(ids, events[:, 1]), path
    assert np.array_equal(rest[:, 1] - rest[:, 0], events[:, 3] - events[:, 2]), path
    assert np.all(rest[:, 0] >= 500) and np.all(rest[:, 1] > rest[:, 0]), path
    assert np.array_equal(trials[ids], selected_trials), path
    assert np.all(events[:, 6] - events[:, 5] == 125), path
    assert np.all(events[:, 7] - events[:, 5] == 875), path
    assert np.all(events[1:, 2] - events[:-1, 5] == 1000), path
    assert metadata["task_ground_truth_seconds_from_cue"] == [0.5, 4.0], path
    assert metadata["training_positive_support_seconds_from_cue"] == [0.5, 3.5], path
    assert metadata["normalization_anchor_samples_250hz"] == [500, 15000], path
    for fold in range(5):
        for source in range(3):
            spans = rest[10 * fold:10 * (fold + 1)][sources[10 * fold:10 * (fold + 1)] == source]
            spans = spans[np.argsort(spans[:, 0])]
            assert len(spans) in (3, 4) and spans[0, 0] == 500, path
            assert np.array_equal(spans[1:, 0], spans[:-1, 1]), path
            assert spans[-1, 1] - 500 == 56 * 250, path
    source_future = overlap_samples(trials[:50], trials[70:100])
    prefix_future = overlap_samples(trials[50:70], trials[70:100])
    assert source_future == 0 and prefix_future == 0, path
    # Each role repartitions the same three source rest blocks: this overlap is
    # real, so injected prefix rest must never enter model/threshold fitting.
    rest_overlap = sum(overlap_samples(
        rest[:10][sources[:10] == source],
        rest[20:30][sources[20:30] == source],
    ) for source in range(3))
    assert rest_overlap == 168 * 250, path
    stream_stop = int(events[-1, 5] + 1000)
    ends = np.arange(250, stream_stop + 1, 25)
    starts = ends - 250
    valid_future_rest = np.zeros(len(ends), dtype=bool)
    for row in events[20:]:
        valid_future_rest |= ((starts >= row[2] + 3 * 250)
                              & (ends <= row[3]))
    future_raw_rest_seconds = float(np.sum(events[20:, 3] - events[20:, 2]) / 250)
    future_score_idle_seconds = float(valid_future_rest.sum() * 0.1)
    max_rest_diff = None
    if raw_rest_root is not None:
        raw_path = raw_rest_root / path.parent.name / path.name
        with np.load(raw_path, allow_pickle=False) as raw:
            names = ("post_train", "pre_test", "post_test")
            max_rest_diff = max(float(np.max(np.abs(
                signal[int(row[2]):int(row[3])]
                - raw[names[int(source)]][int(span[0]):int(span[1])]
            ))) for row, source, span in zip(events, sources, rest))
        assert max_rest_diff <= 1e-6, (path, max_rest_diff)
    max_trial_diff = None
    if trial_root is not None:
        trial_path = trial_root / path.parent.name / f"{path.stem}_mil_raw.npz"
        with np.load(trial_path, allow_pickle=False) as raw:
            source_signal = raw["signal"]
            max_trial_diff = max(float(np.max(np.abs(
                signal[int(row[4]):int(row[4]) + int(span[1] - span[0])]
                - source_signal[int(span[0]):int(span[1])]
            ))) for row, span in zip(events, selected_trials))
        assert max_trial_diff <= 1e-6, (path, max_trial_diff)
    return {
        "file": str(path), "source_train_vs_future_trial_overlap_samples": source_future,
        "prefix_vs_future_trial_overlap_samples": prefix_future,
        "prefix_vs_future_injected_rest_overlap_samples": rest_overlap,
        "unique_rest_seconds": 168.0, "replayed_rest_seconds": 840.0,
        "future_raw_rest_seconds": future_raw_rest_seconds,
        "future_guard_valid_score_idle_seconds": future_score_idle_seconds,
        "max_rest_cache_abs_diff": max_rest_diff,
        "max_trial_cache_abs_diff": max_trial_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?",
                        default=Path("data/interim/lee2019_revalidation_chronological"))
    parser.add_argument("--check-rest-values", action="store_true")
    parser.add_argument("--check-trial-values", action="store_true")
    parser.add_argument("--expect-files", type=int)
    args = parser.parse_args()
    paths = sorted(args.root.glob("session*/sub-*.npz"))
    if not paths:
        parser.error(f"no replay files under {args.root}")
    if args.expect_files is not None and len(paths) != args.expect_files:
        parser.error(f"expected {args.expect_files} replay files, found {len(paths)}")
    raw_rest_root = Path("data/interim/lee2019_four_rest_raw") if args.check_rest_values else None
    trial_root = Path("data/interim/lee2019_mil_raw") if args.check_trial_values else None
    rows = [audit(path, raw_rest_root, trial_root) for path in paths]
    print(json.dumps({
        "files": len(rows), "source_train_future_trial_overlap_files": sum(
            row["source_train_vs_future_trial_overlap_samples"] > 0 for row in rows),
        "prefix_future_trial_overlap_files": sum(
            row["prefix_vs_future_trial_overlap_samples"] > 0 for row in rows),
        "prefix_future_injected_rest_overlap_seconds_per_file": sorted(set(
            row["prefix_vs_future_injected_rest_overlap_samples"] / 250 for row in rows)),
        "future_guard_valid_score_idle_seconds_range": [
            min(row["future_guard_valid_score_idle_seconds"] for row in rows),
            max(row["future_guard_valid_score_idle_seconds"] for row in rows)],
        "rest_fit_policy": "never use injected prefix rest to fit model or threshold",
        "max_rest_cache_abs_diff": max((row["max_rest_cache_abs_diff"] or 0) for row in rows),
        "max_trial_cache_abs_diff": max((row["max_trial_cache_abs_diff"] or 0) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
