"""Paired comparison of corrected CNN-GRU LOSO with Riemannian Arm A."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def bootstrap_cluster_delta(a: np.ndarray, b: np.ndarray, seed: int,
                            repeats: int = 10_000) -> dict:
    """Paired median delta CI, resampling subjects as clusters across sessions."""
    delta = np.asarray(a, float) - np.asarray(b, float)
    subjects = np.arange(delta.shape[0])
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        draws[index] = np.nanmedian(delta[selected])
    return {
        "median_delta": float(np.nanmedian(delta)),
        "cluster_bootstrap_95ci": np.nanquantile(draws, [0.025, 0.975]).tolist(),
        "n_subjects": int(delta.shape[0]),
    }


def get_cnn(row: dict, fp: str, mode: str, metric: str) -> float:
    result = row["metrics"][fp][mode]["metrics"]
    if metric.startswith("sensitivity_"):
        return float(result["onset_sensitivity"][metric.removeprefix("sensitivity_")])
    if metric.startswith("excess_"):
        return float(result["circular_shift_excess"][metric.removeprefix("excess_")])
    if metric == "coverage":
        return float(result["interval_0.1"]["coverage"])
    if metric == "temporal_iou":
        return float(result["interval_0.1"]["temporal_iou"])
    if metric == "strict_f1":
        return float(result["interval_0.7"]["intersection_f1"])
    if metric == "actual_fp":
        return float(result["actual_fp_per_min"])
    raise KeyError(metric)


def get_ria(row: dict, fp: str, metric: str) -> float:
    fixed = row["arms"]["A_target_rest"]["surface"][fp]["fixed_threshold"]
    if metric == "actual_fp":
        return float(fixed["actual_fp"])
    if metric.startswith("sensitivity_"):
        return float(fixed["onset_sensitivity"][metric.removeprefix("sensitivity_")])
    if metric.startswith("excess_"):
        return float(fixed["circular_shift_excess"][metric.removeprefix("excess_")])
    if metric == "coverage":
        return float(fixed["coverage"])
    if metric == "temporal_iou":
        return float(fixed["temporal_iou"])
    if metric == "strict_f1":
        return float(fixed["strict_f1"])
    raise KeyError(metric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", type=Path, default=Path(
        "data/processed/lee2019_cnn_gru_loso_arm_a_stream_v2_full54.json"))
    parser.add_argument("--riemannian", type=Path, default=Path(
        "data/processed/lee2019_loso_deployment_dwell03_null500.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/lee2019_cnn_gru_loso_arm_a_stream_v2_comparison.json"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    cnn_doc = json.loads(args.cnn.read_text(encoding="utf-8"))
    ria_doc = json.loads(args.riemannian.read_text(encoding="utf-8"))
    cnn_rows = {(int(r["session"]), int(r["subject"])): r for r in cnn_doc["rows"]}
    ria_rows = {(int(r["session"]), int(r["subject"])): r for r in ria_doc["rows"]}
    expected = {(session, subject) for session in (1, 2) for subject in range(1, 55)}
    if set(cnn_rows) != expected:
        raise ValueError(f"CNN result needs all 108 keys; found {len(cnn_rows)}")
    if set(ria_rows) != expected:
        raise ValueError(f"Riemannian result needs all 108 keys; found {len(ria_rows)}")

    metrics = ["actual_fp", "sensitivity_0.5", "sensitivity_1.0",
               "sensitivity_2.0", "sensitivity_3.0", "sensitivity_4.0",
               "excess_0.5", "excess_1.0", "excess_2.0", "excess_3.0", "excess_4.0",
               "coverage", "temporal_iou", "strict_f1"]
    summary = {}
    paired_rows = []
    for fp in ("2.0", "4.0"):
        summary[fp] = {}
        for mode in ("common_fp_target_pre_rest", "source_transferred"):
            summary[fp][mode] = {}
            for session in (1, 2):
                current = [cnn_rows[(session, subject)] for subject in range(1, 55)]
                baseline = [ria_rows[(session, subject)] for subject in range(1, 55)]
                summary[fp][mode][str(session)] = {
                    "cnn_gru_median": {
                        metric: float(np.nanmedian([get_cnn(r, fp, mode, metric) for r in current]))
                        for metric in metrics
                    },
                    "riemannian_arm_a_median": {
                        metric: float(np.nanmedian([get_ria(r, fp, metric) for r in baseline]))
                        for metric in metrics
                    },
                }
            summary[fp][mode]["paired_delta_cnn_minus_riemannian"] = {}
            for metric in metrics:
                a = np.asarray([
                    [get_cnn(cnn_rows[(session, subject)], fp, mode, metric)
                     for session in (1, 2)]
                    for subject in range(1, 55)
                ])
                b = np.asarray([
                    [get_ria(ria_rows[(session, subject)], fp, metric)
                     for session in (1, 2)]
                    for subject in range(1, 55)
                ])
                result = bootstrap_cluster_delta(a, b, 99300 + int(float(fp) * 10) + len(metric), args.bootstrap)
                summary[fp][mode]["paired_delta_cnn_minus_riemannian"][metric] = result
                paired_rows.append({"nominal_fp": fp, "cnn_threshold_mode": mode,
                                    "metric": metric, **result})

    payload = {
        "comparison_protocol": {
            "paired_unit": "subject cluster; both sessions resampled together",
            "cnn_primary_arm": "common_fp_target_pre_rest (threshold calibrated on target pre-rest only)",
            "cnn_secondary_arm": "source_transferred (source pre-rest only)",
            "riemannian_baseline": "Arm A target pre-rest fixed-threshold values recomputed at dwell 0.3 s with 500 circular shifts",
            "stream_alignment": "CNN event arrays have been asserted exactly equal to locked Riemannian Arm-A stream events",
            "null_comparison": "both models use 500 matched circular-shift offsets per target and nominal FP",
            "latency_grid_note": "Both models report onset sensitivity on the 0.5/1/2/3/4-s grid.",
            "threshold_mode_note": "CNN source-transferred mode is also compared with Riemannian Arm A as a diagnostic; this is not a calibration-mode-matched model-only contrast.",
            "decision_match": "EWMA H30 + MA 0.5 s + dwell 0.3 s + refractory 1 s; first 10 trials excluded; seam guard matched",
        },
        "summary": summary,
        "paired_rows": paired_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps({fp: summary[fp] for fp in summary}, indent=2))
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
