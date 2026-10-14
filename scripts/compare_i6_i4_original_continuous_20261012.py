"""Descriptive subject-paired I6/I4 oracle comparisons at matched achieved FP.

The same 54 Lee subjects have been repeatedly analysed. These intervals describe
within-cohort uncertainty only; they are not independent confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def indexed(rows):
    out = {}
    for row in rows:
        key = (int(row["subject"]), int(row["session"]), row["arm"], float(row["target_fp"]))
        if key in out:
            raise AssertionError(f"duplicate row: {key}")
        out[key] = row
    return out


def median_ci(values, draws, rng):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    distribution = np.median(values[indices], axis=1)
    return [float(np.quantile(distribution, .025)), float(np.quantile(distribution, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i6", type=Path, default=Path("data/processed/i6_cnn_gru_original_continuous_20261012.json"))
    ap.add_argument("--riemannian", type=Path, default=Path("data/processed/i4_riemannian_original_continuous_20261012.json"))
    ap.add_argument("--tier0", type=Path, default=Path("data/processed/i4_low_fp_original_continuous_20261010.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/i6_i4_matched_achieved_fp_comparison_20261012.json"))
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    i6 = json.loads(args.i6.read_text(encoding="utf-8"))
    riem = json.loads(args.riemannian.read_text(encoding="utf-8"))
    tier = json.loads(args.tier0.read_text(encoding="utf-8"))
    arms = {
        "I6_stateful": indexed(i6["arms"]["stateful"]["rows"]),
        "I6_window_reset": indexed(i6["arms"]["window_reset"]["rows"]),
        "I4_Riemannian": indexed(riem["rows"]),
        "I4_tier0": indexed(tier["rows"]),
    }
    expected = {(s, sess, "future_idle_oracle", fp)
                for s in range(1, 55) for sess in (1, 2) for fp in (1., 2., 4., 8.)}
    for name, rows in arms.items():
        if not expected.issubset(rows):
            raise AssertionError(f"{name} missing {len(expected - rows.keys())} oracle rows")
    rng = np.random.default_rng(20261012)
    pairs = [("I6_stateful", "I4_Riemannian"),
             ("I6_stateful", "I4_tier0"),
             ("I6_stateful", "I6_window_reset")]
    report = []
    for first, second in pairs:
        for sess in (1, 2):
            for fp in (1., 2., 4., 8.):
                # Tier-0's oracle curve cannot attain the same FP8 count in
                # 19/108 cells. Do not silently compare unequal operating points
                # or selectively keep only the achievable subjects.
                if second == "I4_tier0" and fp == 8.:
                    continue
                excess, sen, null = [], [], []
                achieved_counts = []
                for subject in range(1, 55):
                    key = (subject, sess, "future_idle_oracle", fp)
                    a, b = arms[first][key], arms[second][key]
                    if a["future_idle"]["count"] != b["future_idle"]["count"]:
                        raise AssertionError(f"unmatched achieved FP count: {key}")
                    if abs(a["future_idle"]["seconds"] - b["future_idle"]["seconds"]) > 1e-6:
                        raise AssertionError(f"unmatched idle denominator: {key}")
                    achieved_counts.append(a["future_idle"]["count"])
                    excess.append(a["mi"]["excess_2s"] - b["mi"]["excess_2s"])
                    sen.append(a["mi"]["sen_2s"] - b["mi"]["sen_2s"])
                    null.append(a["mi"]["null_2s"] - b["mi"]["null_2s"])
                report.append({
                    "first": first, "second": second, "session": sess,
                    "nominal_fp_per_min": fp, "n_subjects": 54,
                    "future_idle_alarm_counts": sorted(set(achieved_counts)),
                    "median_paired_delta_excess_2s": float(np.median(excess)),
                    "median_paired_delta_excess_2s_ci95": median_ci(excess, args.bootstrap_draws, rng),
                    "median_paired_delta_sen_2s": float(np.median(sen)),
                    "median_paired_delta_null_2s": float(np.median(null)),
                })
    output = {
        "status": "Lee internal exploratory; oracle thresholds fit to future idle; not deployable",
        "comparison": "same subjects, same future idle, same achieved alarm count",
        "difference_caution": "CNN/Riemannian 1-second vs tier-0 2-second feature windows; different source fit",
        "unmatched_fp8_tier0": "excluded: 19/108 tier-0 cells do not attain CNN's oracle FP8 alarm count",
        "bootstrap_seed": 20261012, "bootstrap_draws": args.bootstrap_draws,
        "rows": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    for row in report:
        if row["nominal_fp_per_min"] in (1., 2.):
            print(row["first"], "-", row["second"], "S", row["session"],
                  "FP", row["nominal_fp_per_min"],
                  "delta", round(row["median_paired_delta_excess_2s"], 3),
                  "CI", [round(x, 3) for x in row["median_paired_delta_excess_2s_ci95"]])


if __name__ == "__main__":
    main()
