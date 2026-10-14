"""Conditional sham-vs-sham numeric Type-I/coverage audit at matched FP.

Two independent halves of the rest-sham draws are compared at exactly equal
idle alarm count. Conditioning on the finite unique-rest library and on the
observed-score curve's attainable-count selection means this is a *numeric*
symmetry check, not validation of MI specificity or a real-world null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wilson(success: int, total: int):
    z = 1.959963984540054
    p = success / total
    scale = 1. + z * z / total
    center = (p + z * z / (2. * total)) / scale
    radius = z * np.sqrt(p * (1. - p) / total + z * z / (4. * total * total)) / scale
    return [float(center - radius), float(center + radius)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    with np.load(args.input.with_suffix(".npz"), allow_pickle=False) as data:
        count = data["matched_fp_count"].astype(int)
        hit = data["sham_hits2"].astype(int)
    if count.shape != hit.shape or count.shape[0:2] != (2, 54) or count.shape[2] < 1000:
        raise ValueError("need 54 subjects x 2 sessions x >=1000 complete draws")
    half = count.shape[2] // 2
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, 54, size=(args.bootstrap, 54))
    result = {}
    for session in range(2):
        pairs = np.empty((54, args.replicates), dtype=float)
        unsupported = 0
        for subject in range(54):
            for direction in (0, 1):
                if direction == 0:
                    outer_pool = np.arange(0, half)
                    reference_pool = np.arange(half, 2 * half)
                else:
                    outer_pool = np.arange(half, 2 * half)
                    reference_pool = np.arange(0, half)
                reference_by_count = {}
                for index in reference_pool:
                    key = int(count[session, subject, index])
                    if key >= 0:
                        reference_by_count.setdefault(key, []).append(index)
                eligible = [int(index) for index in outer_pool
                            if int(count[session, subject, index]) in reference_by_count]
                unsupported += len(outer_pool) - len(eligible)
                if not eligible:
                    raise AssertionError("no equal-FP cross-half sham pair")
                for repetition in range(direction, args.replicates, 2):
                    outer = int(rng.choice(eligible))
                    inner = int(rng.choice(reference_by_count[
                        int(count[session, subject, outer])]))
                    pairs[subject, repetition] = (
                        hit[session, subject, outer] - hit[session, subject, inner]
                    ) / 30.
        median_values = np.median(pairs, axis=0)
        lower = np.empty(args.replicates)
        upper = np.empty(args.replicates)
        for repetition in range(args.replicates):
            distribution = np.median(pairs[:, repetition][bootstrap_indices], axis=1)
            lower[repetition], upper[repetition] = np.quantile(distribution,
                                                                 [.025, .975])
        positive_rejection = int(np.sum(lower > 0.))
        two_sided_rejection = int(np.sum((lower > 0.) | (upper < 0.)))
        covered = int(np.sum((lower <= 0.) & (upper >= 0.)))
        positive_ci = wilson(positive_rejection, args.replicates)
        two_sided_ci = wilson(two_sided_rejection, args.replicates)
        coverage_ci = wilson(covered, args.replicates)
        result[str(session + 1)] = {
            "replicates": args.replicates,
            "unsupported_cross_half_sham_draws": unsupported,
            "outer_null_median_of_medians": float(np.median(median_values)),
            "positive_rejections": positive_rejection,
            "positive_rejection_wilson95": positive_ci,
            "two_sided_rejections": two_sided_rejection,
            "two_sided_rejection_wilson95": two_sided_ci,
            "zero_covered": covered,
            "coverage_wilson95": coverage_ci,
            "passes_fixed_numeric_gate_two_sided": (
                two_sided_ci[1] <= .075 and coverage_ci[0] >= .925
            ),
        }
    payload = {
        "status": "conditional within-rest-library symmetry check; not MI-specific",
        "input": str(args.input), "seed": args.seed,
        "bootstrap_replicates_per_outer": args.bootstrap,
        "results": result,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
