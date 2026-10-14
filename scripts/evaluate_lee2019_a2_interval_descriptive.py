"""Descriptive A2 interval metrics after the A0 guard-boundary correction.

No null correction is attempted here; this is not a model-promotion result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    interval_metrics, intervals_from_alarm,
)
from scripts.evaluate_lee2019_rest_sham_matched_fp import compute_guard_local_ma
from uncued_mi.decision import guard_rearmed_alarms


def per_trial_union(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    coverage, union_iou, fragmentation = [], [], []
    for start, stop in gt:
        overlaps = np.maximum(0., np.minimum(pred[:, 1], stop) -
                              np.maximum(pred[:, 0], start)) if len(pred) else np.empty(0)
        relevant = overlaps > 0
        common = float(overlaps.sum())
        prediction = float(np.sum(pred[relevant, 1] - pred[relevant, 0]))
        duration = stop - start
        coverage.append(common / duration)
        union_iou.append(common / (duration + prediction - common))
        fragmentation.append(int(relevant.sum()))
    return float(np.mean(coverage)), float(np.mean(union_iou)), float(np.mean(fragmentation))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-fp", action="store_true",
                        help="use evaluation-idle oracle thresholds (diagnostic only)")
    args = parser.parse_args()
    result_root = ROOT / "data/processed/performance_revalidation_20261001"
    source_name = ("a2_cnn_gru_vs_riemannian_chronological_observed_matched_fp.json"
                   if args.matched_fp else "a2_chronological_anchor_operating_points.json")
    operating = json.loads((result_root / source_name).read_text(encoding="utf-8"))
    roots = {"riemannian": ROOT / "data/interim/lee2019_revalidation_riemannian_scores",
             "cnn_gru": ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache"}
    rows = []
    for point in operating["rows"]:
        if args.matched_fp and not point["matched"]:
            continue
        subject, session, target_fp = point["subject"], point["session"], point["target_fp"]
        row = {"subject": subject, "session": session, "target_fp": target_fp}
        for model, root in roots.items():
            with np.load(root / f"session{session}/sub-{subject:02d}.npz",
                         allow_pickle=False) as data:
                raw = np.asarray(data["raw_scores"], float)
                anchor = np.asarray(data["anchor_raw_scores" if model == "riemannian"
                                         else "anchor_scores"], float)
                allowed = np.asarray(data["allowed"], bool)
                times = np.asarray(data["times"], float)
                events = np.asarray(data["events"], np.int64)
            threshold = float(point[f"{model}_threshold"] if args.matched_fp
                              else point[model]["threshold"])
            future = times >= events[20, 2] / 250.
            ma = compute_guard_local_ma(raw, float(anchor.mean()),
                                        float(anchor.var(ddof=1)), allowed)
            alarms = guard_rearmed_alarms(ma, threshold, allowed, 3, 10) & future
            pred = intervals_from_alarm(times, alarms, ma, threshold, allowed)
            gt = np.c_[events[20:, 5] / 250. + .5, events[20:, 5] / 250. + 4.]
            coverage, iou, fragmentation = per_trial_union(pred, gt)
            strict = interval_metrics(pred, gt, .7, .7)
            tolerant = interval_metrics(pred, gt, .1, .1)
            row[model] = {
                "raw_union_coverage": coverage,
                "raw_union_iou": iou,
                "mean_fragmentation": fragmentation,
                "strict_f1_dtc_gtc_0_7": strict["intersection_f1"],
                "tolerant_f1_dtc_gtc_0_1": tolerant["intersection_f1"],
                "predicted_intervals": len(pred),
                "achieved_fp_per_min": (point["achieved_fp_per_min"] if args.matched_fp
                                        else point[model]["future_fp_per_min"]),
            }
        rows.append(row)
    summary = {"scope": "A2 descriptive interval comparison; chronological pseudo-uncued simulation",
               "threshold_source": "evaluation-idle matched-FP diagnostic oracle" if args.matched_fp else "target pre-rest operating threshold",
               "warning": ("Future idle is used to select thresholds; not deployable; no EEG-level sham correction or model superiority claim" if args.matched_fp else
                           "Pre-rest thresholds have unequal achieved FP; no EEG-level sham correction or model superiority claim"),
               "interval_definition": "guard-bounded threshold-active alarm interval; release=trigger-0.5, hold=3 samples, within-segment gap merge<=0.3 s",
               "metrics_definition": "raw_union_iou uses the union of predicted intervals overlapping each GT interval; predictions outside GT enter FP/F1, not that per-trial IoU denominator",
               "rows": rows, "by_session": {}}
    for fp in (2., 4.):
        for session in (1, 2):
            subset = [row for row in rows if row["target_fp"] == fp and row["session"] == session]
            entry = {"subjects": len(subset)}
            for model in roots:
                entry[model] = {metric: float(np.median([row[model][metric] for row in subset]))
                                for metric in ("raw_union_coverage", "raw_union_iou",
                                               "mean_fragmentation", "strict_f1_dtc_gtc_0_7",
                                               "tolerant_f1_dtc_gtc_0_1", "predicted_intervals")}
            paired = np.asarray([
                row["cnn_gru"]["raw_union_coverage"] - row["riemannian"]["raw_union_coverage"]
                for row in subset], float)
            entry["median_paired_delta_cnn_minus_riem_coverage"] = float(np.median(paired))
            rng = np.random.default_rng(20261001 + 100 * int(fp) + session)
            bootstrap = np.median(paired[rng.integers(0, len(paired), (5000, len(paired)))],
                                  axis=1)
            entry["paired_delta_coverage_bootstrap_ci95_descriptive"] = np.quantile(
                bootstrap, [.025, .975]).tolist()
            summary["by_session"][f"FP{fp:g}_S{session}"] = entry
    output = result_root / ("a2_chronological_interval_matched_fp_guard_bounded.json"
                            if args.matched_fp else "a2_chronological_interval_descriptive_guard_bounded.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_session"], indent=2))


if __name__ == "__main__":
    main()
