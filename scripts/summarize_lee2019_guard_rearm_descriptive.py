"""Descriptive operating-point audit; intentionally no null or efficacy test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["rows"]
    if len(rows) != 108:
        raise ValueError(f"expected 108 subject-session rows, found {len(rows)}")
    summary = {}
    for session in (1, 2):
        subset = [row for row in rows if row["session"] == session]
        summary[str(session)] = {}
        for target, bounds in ((2., (1.5, 2.5)), (4., (3., 5.))):
            metrics = [row["targets"][str(target)] for row in subset]
            actual = np.asarray([item["future_fp_per_min"] for item in metrics])
            legacy = np.asarray([
                item["legacy_same_threshold"]["future_fp_count"] * 60.
                / item["future_idle_seconds"] for item in metrics
            ])
            new_hits = sum(item["future_onset_hits_2s"] for item in metrics)
            old_hits = sum(
                item["legacy_same_threshold"]["future_onset_hits_2s"]
                for item in metrics
            )
            edge_hits = sum(
                item["edge_only_same_threshold"]["future_onset_hits_2s"]
                for item in metrics
            )
            events = sum(item["future_onset_count"] for item in metrics)
            high = sum(item["high_at_first_valid_post_guard_ma"]
                       for item in metrics)
            summary[str(session)][str(target)] = {
                "n_sessions": len(metrics),
                "achieved_fp_median": float(np.median(actual)),
                "achieved_fp_iqr": np.quantile(actual, [.25, .75]).tolist(),
                "achieved_fp_in_target_band_n": int(np.sum(
                    (actual >= bounds[0]) & (actual <= bounds[1])
                )),
                "zero_onset_hit_sessions_n": sum(
                    item["future_onset_hits_2s"] == 0 for item in metrics
                ),
                "new_onset_hits_total": new_hits,
                "legacy_same_threshold_onset_hits_total": old_hits,
                "edge_only_same_threshold_onset_hits_total": edge_hits,
                "future_onsets_total": events,
                "high_at_first_valid_ma_total": high,
                "high_at_first_valid_ma_fraction": high / events,
                "legacy_same_threshold_fp_median": float(np.median(legacy)),
                "edge_only_same_threshold_fp_median": float(np.median([
                    item["edge_only_same_threshold"]["future_fp_count"] * 60.
                    / item["future_idle_seconds"] for item in metrics
                ])),
                "same_threshold_comparison_warning": (
                    "legacy and new achieved FP differ; onset hit difference "
                    "is not a matched-FP model comparison"
                ),
            }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
