"""Diagnostic matched-achieved-FP sham comparison on Lee2019 Riemannian scores.

Every sham gets its own threshold sweep. Observed and sham use an identical
achieved idle alarm count selected from their attainable-count intersection.
This uses evaluation idle labels: it is an oracle diagnostic, not deployment.
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

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from uncued_mi.rest_sham import build_rest_score_sham_library
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z


@njit(cache=True)
def compute_guard_local_ma(raw, anchor_mean, anchor_var, allowed):
    alpha = 1. - 2. ** (-.1 / 30.)
    floor = .25 * np.sqrt(max(anchor_var, 1e-8))
    mean = anchor_mean
    var = anchor_var
    recent = np.zeros(5, dtype=np.float64)
    length = 0
    ma = np.full(len(raw), np.nan)
    for index in range(len(raw)):
        x = raw[index]
        z = (x - mean) / max(np.sqrt(max(var, 0.)), floor)
        if allowed[index]:
            difference = x - mean
            mean += alpha * difference
            var = (1. - alpha) * (var + alpha * difference * difference)
            recent[length % 5] = z
            length += 1
            if length >= 5:
                ma[index] = np.sum(recent) / 5.
        else:
            length = 0
    return ma


@njit(cache=True)
def scan_thresholds(ma, thresholds, allowed, idle, hit_label, scan_start):
    """Exact edge-armed detector over each threshold; [threshold, (FP,hits)]."""
    result = np.zeros((len(thresholds), 2), dtype=np.int32)
    for threshold_index in range(len(thresholds)):
        threshold = thresholds[threshold_index]
        armed = False
        count = 0
        blocked_until = -1
        seen = np.zeros(30, dtype=np.bool_)
        for index in range(scan_start, len(ma)):
            value = ma[index]
            if not allowed[index] or not np.isfinite(value):
                armed = False
                count = 0
                continue
            if not armed:
                if value < threshold:
                    armed = True
                continue
            if value >= threshold:
                count += 1
                if count == 3 and index > blocked_until:
                    blocked_until = index + 10
                    if idle[index]:
                        result[threshold_index, 0] += 1
                    label = hit_label[index]
                    if label >= 0 and not seen[label]:
                        seen[label] = True
                        result[threshold_index, 1] += 1
            else:
                count = 0
    return result


def matched_curve(ma, allowed, future, idle, hit_label, n_thresholds):
    valid = ma[allowed & future & np.isfinite(ma)]
    if not len(valid):
        raise ValueError("no valid future MA")
    thresholds = np.unique(np.r_[valid.min() - 1e-8,
                                  np.quantile(valid, np.linspace(0., 1., n_thresholds)),
                                  valid.max() + 1e-8])[::-1].astype(float)
    first_future = int(np.flatnonzero(future)[0])
    prior_guards = np.flatnonzero(~allowed[:first_future])
    scan_start = int(prior_guards[-1]) if len(prior_guards) else 0
    measured = scan_thresholds(ma, thresholds, allowed, idle, hit_label,
                               scan_start)
    peak = int(np.argmax(measured[:, 0]))
    # Retain highest threshold for each count on the conservative upper branch.
    return {int(measured[i, 0]): (float(thresholds[i]), int(measured[i, 1]))
            for i in range(peak, -1, -1)}


def matched_pair(observed_curve, sham_curve, desired_count):
    shared = set(observed_curve) & set(sham_curve)
    if not shared:
        return None
    count = min(shared, key=lambda x: (abs(x - desired_count), x))
    observed_threshold, observed_hits = observed_curve[count]
    sham_threshold, sham_hits = sham_curve[count]
    return count, observed_hits, sham_hits, observed_threshold, sham_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--thresholds", type=int, default=101)
    parser.add_argument("--target-fp", type=float, default=2.)
    parser.add_argument("--limit", type=int, default=54)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/performance_revalidation_20261001/"
                        "a1_rest_sham_matched_fp_riemannian_full108")
    args = parser.parse_args()
    if not (1 <= args.limit <= 54 and args.replicates >= 1 and
            args.thresholds >= 3 and args.target_fp > 0.):
        raise ValueError("invalid analysis arguments")
    counts = np.full((2, args.limit, args.replicates), -1, dtype=np.int16)
    observed_hits = np.full_like(counts, -1)
    sham_hits = np.full_like(counts, -1)
    observed_thresholds = np.full(counts.shape, np.nan)
    sham_thresholds = np.full(counts.shape, np.nan)
    scalar_checks = 0
    rng = np.random.default_rng(args.seed)
    provenance = []
    for session in (1, 2):
        for subject in range(1, args.limit + 1):
            with np.load(ROOT / "data/interim/lee2019_revalidation_riemannian_scores" /
                         f"session{session}/sub-{subject:02d}.npz", allow_pickle=False) as score:
                raw = score["raw_scores"].astype(float)
                anchor = score["anchor_raw_scores"].astype(float)
                times = score["times"].astype(float)
                starts = score["starts"].astype(float)
                allowed = score["allowed"].astype(bool)
                events = score["events"].astype(np.int64)
            with np.load(ROOT / "data/interim/lee2019_revalidation_chronological" /
                         f"session{session}/sub-{subject:02d}.npz", allow_pickle=False) as replay:
                if not np.array_equal(replay["events"], events):
                    raise AssertionError("score/replay event mismatch")
                library = build_rest_score_sham_library(
                    raw, starts, times, events, replay["rest_source_ids"],
                    replay["rest_source_spans_250hz"], len(replay["state"]),
                )
            replaced = np.zeros(len(raw), dtype=bool)
            for indices in library.command_score_indices:
                replaced[indices] = True
            onsets = events[20:, 5] / 250. + .5
            future = times >= events[20, 2] / 250.
            task = np.zeros(len(raw), dtype=bool)
            hit_label = np.full(len(raw), -1, dtype=np.int32)
            for trial, onset in enumerate(onsets):
                cue = onset - .5
                task |= (times >= onset) & (times <= cue + 4.)
                hit_label[(times >= onset) & (times <= onset + 2.)] = trial
            if not np.all(replaced[task & allowed]):
                raise AssertionError("some allowed task score not sham-replaced")
            idle = future & allowed & ~task
            desired = args.target_fp * idle.sum() * .1 / 60.
            mean = float(anchor.mean())
            var = float(anchor.var(ddof=1))
            ma = compute_guard_local_ma(raw, mean, var, allowed)
            if scalar_checks < 3:
                expected = guard_local_moving_average(
                    ewma_z(raw, mean, var, allowed), allowed, ma_steps=5)
                np.testing.assert_allclose(ma, expected, atol=1e-11, rtol=0)
                scalar_checks += 1
            observed_curve = matched_curve(ma, allowed, future, idle, hit_label,
                                           args.thresholds)
            for draw in range(args.replicates):
                sham_raw = library.draw(raw, rng)
                sham_ma = compute_guard_local_ma(sham_raw, mean, var, allowed)
                sham_curve = matched_curve(sham_ma, allowed, future, idle, hit_label,
                                           args.thresholds)
                point = matched_pair(observed_curve, sham_curve, desired)
                if point is None:
                    continue
                count, oh, sh, ot, st = point
                if draw == 0 and scalar_checks < 6:
                    for measured_ma, threshold, hits in ((ma, ot, oh),
                                                          (sham_ma, st, sh)):
                        alarms = guard_rearmed_alarms(
                            measured_ma, threshold, allowed,
                            dwell_steps=3, refractory_steps=10)
                        actual_fp = int(np.sum(alarms & idle))
                        actual_hits = sum(np.any(
                            alarms & (times >= onset) & (times <= onset + 2.))
                            for onset in onsets)
                        if (actual_fp, actual_hits) != (count, hits):
                            raise AssertionError("scalar matched curve mismatch")
                    scalar_checks += 1
                slot = (session - 1, subject - 1, draw)
                counts[slot] = count
                observed_hits[slot] = oh
                sham_hits[slot] = sh
                observed_thresholds[slot] = ot
                sham_thresholds[slot] = st
            provenance.append({"subject": subject, "session": session,
                               "source_unique_score_counts": library.unique_source_score_counts,
                               "max_repeated_source_score_difference":
                                   library.max_repeated_score_difference,
                               "desired_false_alarm_count": float(desired),
                               "observed_attainable_count_max": max(observed_curve),
                               "observed_attainable_count_n": len(observed_curve)})
            row = counts[session - 1, subject - 1]
            print(f"S{session} sub-{subject:02d}: matched {np.sum(row >= 0)}/"
                  f"{args.replicates}; median count {np.median(row[row >= 0]) if np.any(row >= 0) else 'none'}",
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output) + ".npz", matched_fp_count=counts,
                        observed_hits2=observed_hits, sham_hits2=sham_hits,
                        observed_threshold=observed_thresholds,
                        sham_threshold=sham_thresholds)
    summary = {
        "status": "diagnostic oracle: matched achieved FP; not deployable calibration",
        "scope": "Riemannian Arm A memoryless score; Lee2019 pseudo-uncued simulation",
        "replicates": args.replicates,
        "thresholds": args.thresholds,
        "seed": args.seed,
        "subjects_per_session": args.limit,
        "target_fp_per_min": args.target_fp,
        "unmatched_pairs_by_session": [int(np.sum(counts[s] < 0)) for s in range(2)],
        "source_rest_reused_across_prefix_future": True,
        "provenance": provenance,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
