"""A1 rest-only score sham audit for the memoryless Riemannian arm.

This is a simulation null, not evidence of motor-imagery specificity. Trial
locations, rest placements, guards and the target pre-rest anchor are fixed.
Only fully supported trial-window scores are replaced by contiguous scores
from the three *unique original* target rest blocks. The replacement is valid
for window-local Riemannian scores, not for a recurrent CNN-GRU score stream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.rest_sham import build_rest_score_sham_library
from scripts.pilot_lee2019_rest_sham_a1 import metrics


@njit(cache=True)
def evaluate_raw(raw, anchor_mean, anchor_var, allowed, idle, times, onsets,
                 threshold):
    alpha = 1. - 2. ** (-.1 / 30.)
    floor = .25 * np.sqrt(max(anchor_var, 1e-8))
    mean = anchor_mean
    var = anchor_var
    recent = np.zeros(5, dtype=np.float64)
    segment_length = 0
    armed = False
    count = 0
    blocked_until = -1
    alarms = np.zeros(len(raw), dtype=np.bool_)
    for index in range(len(raw)):
        x = raw[index]
        z = (x - mean) / max(np.sqrt(max(var, 0.)), floor)
        if allowed[index]:
            difference = x - mean
            mean += alpha * difference
            var = (1. - alpha) * (var + alpha * difference * difference)
            recent[segment_length % 5] = z
            segment_length += 1
            if segment_length >= 5:
                ma = np.sum(recent) / 5.
                if not armed:
                    if ma < threshold:
                        armed = True
                else:
                    if ma >= threshold:
                        count += 1
                        if count == 3 and index > blocked_until:
                            alarms[index] = True
                            blocked_until = index + 10
                    else:
                        count = 0
            else:
                armed = False
                count = 0
        else:
            segment_length = 0
            armed = False
            count = 0
    fp = np.sum(alarms & idle)
    hits = 0
    for onset in onsets:
        hit = False
        for index in range(len(times)):
            if alarms[index] and times[index] >= onset and times[index] <= onset + 2.:
                hit = True
                break
        if hit:
            hits += 1
    return hits, fp


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / n
    denom = 1. + z * z / n
    center = (p + z * z / (2. * n)) / denom
    radius = z * np.sqrt(p * (1. - p) / n + z * z / (4. * n * n)) / denom
    return float(center - radius), float(center + radius)


def split_reference_null_check(null_hits: np.ndarray, draws: int = 1000,
                               seed: int = 20261002) -> dict:
    """Independent-half negative-control test with subject-cluster bootstrap.

    The split makes outer and reference sham draws independent conditional on
    each session's finite rest library. It validates implementation symmetry,
    not the biological adequacy of the rest-only null.
    """
    if null_hits.shape != (2, 54, draws) or draws < 1000 or draws % 2:
        raise ValueError("need two sessions, 54 subjects, >=1000 even sham draws")
    half = draws // 2
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, 54, size=(1000, 54))
    results = {}
    for session in range(2):
        values = null_hits[session].astype(float) / 30.
        excess = np.empty_like(values)
        excess[:, :half] = values[:, :half] - np.median(values[:, half:], axis=1)[:, None]
        excess[:, half:] = values[:, half:] - np.median(values[:, :half], axis=1)[:, None]
        rejected = 0
        covered = 0
        median_bias = []
        for draw in range(draws):
            row = excess[:, draw]
            distribution = np.median(row[boot_indices], axis=1)
            lower, upper = np.quantile(distribution, [.025, .975])
            rejected += bool(lower > 0.)
            covered += bool(lower <= 0. <= upper)
            median_bias.append(float(np.median(row)))
        rejection_ci = wilson_interval(rejected, draws)
        coverage_ci = wilson_interval(covered, draws)
        results[f"session{session + 1}"] = {
            "replicates": draws,
            "rejections_positive": rejected,
            "rejection_fraction": rejected / draws,
            "rejection_wilson95": rejection_ci,
            "zero_covered": covered,
            "coverage_fraction": covered / draws,
            "coverage_wilson95": coverage_ci,
            "outer_median_excess_median": float(np.median(median_bias)),
            "passes_fixed_numeric_gate": rejection_ci[1] <= .075
            and coverage_ci[0] >= .925,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--target", choices=("2.0", "4.0"), default="2.0")
    parser.add_argument("--limit", type=int, default=54)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/"
                        "performance_revalidation_20261001/a1_rest_sham_riemannian_full108")
    args = parser.parse_args()
    if not 1 <= args.limit <= 54 or args.replicates < 1:
        raise ValueError("invalid subject limit or replicate count")
    observed_rows = json.loads((ROOT / "data/processed/performance_revalidation_20261001/"
                                "guard_rearm_riemannian_ablation_108.json").read_text(
                                    encoding="utf-8"))["rows"]
    row_by_key = {(int(row["subject"]), int(row["session"])): row
                  for row in observed_rows}
    null_hits = np.full((2, args.limit, args.replicates), -1, dtype=np.int16)
    null_fp = np.full_like(null_hits, -1)
    provenance = []
    rng = np.random.default_rng(args.seed)
    for session in (1, 2):
        for subject in range(1, args.limit + 1):
            score_path = ROOT / ("data/interim/lee2019_revalidation_riemannian_scores/"
                                 f"session{session}/sub-{subject:02d}.npz")
            replay_path = ROOT / ("data/interim/lee2019_revalidation_chronological/"
                                  f"session{session}/sub-{subject:02d}.npz")
            with np.load(score_path, allow_pickle=False) as score:
                raw = score["raw_scores"].astype(float)
                anchor = score["anchor_raw_scores"].astype(float)
                times = score["times"].astype(float)
                starts = score["starts"].astype(float)
                allowed = score["allowed"].astype(bool)
                events = score["events"].astype(np.int64)
            with np.load(replay_path, allow_pickle=False) as replay:
                if not np.array_equal(replay["events"], events):
                    raise AssertionError("event coordinates differ")
                library = build_rest_score_sham_library(
                    raw, starts, times, events, replay["rest_source_ids"],
                    replay["rest_source_spans_250hz"], len(replay["state"]),
                )
            replaced = np.zeros(len(raw), dtype=bool)
            for indices in library.command_score_indices:
                replaced[indices] = True
            task = np.zeros(len(raw), dtype=bool)
            cues = events[20:, 5] / 250.
            for cue in cues:
                task |= (times >= cue + .5) & (times <= cue + 4.)
            if len(library.command_score_indices) != 50 or not np.all(replaced[task & allowed]):
                raise AssertionError("some allowed task window is not replaced")
            future = times >= events[20, 2] / 250.
            idle = future & allowed & ~task
            onsets = cues + .5
            threshold = float(row_by_key[(subject, session)]["targets"][args.target]
                              ["anchor"]["threshold"])
            expected = metrics(raw, anchor, allowed, times, events, threshold)
            actual = evaluate_raw(raw, float(anchor.mean()), float(anchor.var(ddof=1)),
                                  allowed, idle, times, onsets, threshold)
            if tuple(actual) != (expected["onset_hits_2s"], expected["fp_count"]):
                raise AssertionError(f"accelerated evaluator mismatch: {subject}/{session}")
            for draw in range(args.replicates):
                sham = library.draw(raw, rng)
                if draw == 0 and not np.array_equal(sham[~replaced], raw[~replaced]):
                    raise AssertionError("rest score changed during sham")
                hits, fp = evaluate_raw(sham, float(anchor.mean()),
                                        float(anchor.var(ddof=1)), allowed, idle,
                                        times, onsets, threshold)
                null_hits[session - 1, subject - 1, draw] = hits
                null_fp[session - 1, subject - 1, draw] = fp
            provenance.append({
                "subject": subject, "session": session,
                "unique_source_score_counts": library.unique_source_score_counts,
                "repeated_source_score_count": library.repeated_source_score_count,
                "max_repeated_score_difference": library.max_repeated_score_difference,
                "max_contiguous_run_length_by_block": [
                    max(len(run) for run in runs) for runs in library.runs_by_block],
                "observed_hits2": actual[0], "observed_fp_count": actual[1],
            })
            print(f"completed subject {subject:02d} session {session}: "
                  f"null hits median {np.median(null_hits[session - 1, subject - 1])}, "
                  f"null FP median {np.median(null_fp[session - 1, subject - 1])}",
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output) + ".npz", null_hits2=null_hits,
                        null_fp_count=null_fp)
    summary = {
        "status": "Riemannian rest-only score sham, not MI-specific null",
        "replicates": args.replicates, "seed": args.seed,
        "subjects_per_session": args.limit, "target_fp_per_min": float(args.target),
        "source_unique_rest_seconds": 168,
        "source_rest_reused_in_prefix_and_future": True,
        "provenance": provenance,
    }
    if args.limit == 54 and args.replicates >= 1000 and args.replicates % 2 == 0:
        summary["independent_half_type1_coverage"] = split_reference_null_check(
            null_hits, args.replicates)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "provenance"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
