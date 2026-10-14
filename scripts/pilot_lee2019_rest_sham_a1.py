"""Bounded A1 sham-MI pilot for the memoryless Riemannian score pipeline.

This is not the planned >=1000-outer-replicate Type-I/CI-coverage validation.
It checks provenance, complete command replacement, causal reprocessing, and
the effect of fixed pre-rest threshold transfer under a rest-only sham.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from uncued_mi.rest_sham import build_rest_score_sham_library
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z


def metrics(raw: np.ndarray, anchor: np.ndarray, allowed: np.ndarray,
            times: np.ndarray, events: np.ndarray, threshold: float) -> dict:
    z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
    ma = guard_local_moving_average(z, allowed, ma_steps=5)
    alarms = guard_rearmed_alarms(
        ma, threshold, allowed, dwell_steps=3, refractory_steps=10,
    )
    future = times >= events[20, 2] / 250.
    cues = events[20:, 5] / 250.
    onsets = cues + .5
    task = np.zeros(len(times), dtype=bool)
    for cue in cues:
        task |= (times >= cue + .5) & (times <= cue + 4.)
    idle = future & allowed & ~task
    exposure = float(idle.sum() * .1)
    hits = sum(
        np.any(alarms & future & (times >= onset) & (times <= onset + 2.))
        for onset in onsets
    )
    fp = int(np.sum(alarms & idle))
    return {
        "sen2": float(hits / len(onsets)),
        "onset_hits_2s": int(hits),
        "fp_count": fp,
        "fp_per_min": float(fp * 60. / exposure),
        "idle_seconds": exposure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--target", type=str, choices=("2.0", "4.0"), default="2.0")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--scores", type=Path,
        default=ROOT / "data/interim/lee2019_revalidation_riemannian_scores",
    )
    parser.add_argument(
        "--replay", type=Path,
        default=ROOT / "data/interim/lee2019_revalidation_chronological",
    )
    parser.add_argument(
        "--observed", type=Path,
        default=ROOT / "data/processed/performance_revalidation_20261001/"
        "guard_rearm_riemannian_ablation_108.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/processed/performance_revalidation_20261001/"
        "a1_rest_sham_pilot_sub03_s1_fp2.json",
    )
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    score_path = args.scores / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    replay_path = args.replay / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    with np.load(score_path, allow_pickle=False) as score:
        raw = score["raw_scores"].astype(float)
        anchor = score["anchor_raw_scores"].astype(float)
        times = score["times"].astype(float)
        starts = score["starts"].astype(float)
        allowed = score["allowed"].astype(bool)
        events = score["events"].astype(np.int64)
    with np.load(replay_path, allow_pickle=False) as replay:
        if not np.array_equal(replay["events"], events):
            raise AssertionError("score and replay event coordinates differ")
        source_ids = replay["rest_source_ids"].copy()
        source_spans = replay["rest_source_spans_250hz"].copy()
        sample_count = len(replay["state"])
    library = build_rest_score_sham_library(
        raw, starts, times, events, source_ids, source_spans, sample_count,
    )
    replace_mask = np.zeros(len(raw), dtype=bool)
    for indices in library.command_score_indices:
        replace_mask[indices] = True
    task_allowed = np.zeros(len(raw), dtype=bool)
    for event in events:
        cue = event[5] / 250.
        task_allowed |= (times >= cue + .5) & (times <= cue + 4.) & allowed
    if not np.all(replace_mask[task_allowed]):
        raise AssertionError("some allowed task score is not replaced")
    observed_rows = json.loads(args.observed.read_text(encoding="utf-8"))["rows"]
    match = [row for row in observed_rows
             if row["subject"] == args.subject and row["session"] == args.session]
    if len(match) != 1:
        raise AssertionError("observed score result not uniquely identified")
    saved = match[0]["targets"][args.target]
    threshold = float(saved["anchor"]["threshold"])
    observed = metrics(raw, anchor, allowed, times, events, threshold)
    if (observed["fp_count"] != saved["future_fp_count"]
            or observed["onset_hits_2s"] != saved["future_onset_hits_2s"]):
        raise AssertionError("independent pilot evaluator does not reproduce observed metrics")

    rng = np.random.default_rng(args.seed)
    null = []
    for index in range(args.replicates):
        sham = library.draw(raw, rng)
        if not np.array_equal(sham[~replace_mask], raw[~replace_mask]):
            raise AssertionError("sham altered observed rest score schedule")
        null.append(metrics(sham, anchor, allowed, times, events, threshold))
        if (index + 1) % 100 == 0:
            print(f"sham {index + 1}/{args.replicates}", flush=True)
    null_sen = np.asarray([row["sen2"] for row in null])
    null_fp = np.asarray([row["fp_per_min"] for row in null])
    payload = {
        "status": "A1 pipeline/provenance pilot only; not Type-I or CI coverage pass",
        "subject": args.subject, "session": args.session,
        "target_fp_per_min": float(args.target),
        "replicates": args.replicates, "seed": args.seed,
        "threshold": threshold,
        "library": {
            "unique_source_score_counts": library.unique_source_score_counts,
            "repeated_source_score_count": library.repeated_source_score_count,
            "max_repeated_score_difference": library.max_repeated_score_difference,
            "max_contiguous_run_length_by_block": [
                max(len(run) for run in runs) for runs in library.runs_by_block
            ],
            "replaced_trial_segments": len(library.command_score_indices),
            "all_allowed_task_scores_replaced": True,
        },
        "observed": observed,
        "null_sen2_median": float(np.median(null_sen)),
        "null_sen2_range": [float(null_sen.min()), float(null_sen.max())],
        "null_fp_per_min_median": float(np.median(null_fp)),
        "null_fp_per_min_range": [float(null_fp.min()), float(null_fp.max())],
        "null_rows": null,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "null_rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
