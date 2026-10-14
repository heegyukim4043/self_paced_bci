"""Exploratory K_F 1/2/4-alarm FP4 ladder on frozen Lee2019 scores.

Two distinct future-rest source blocks supply balanced calibration support;
the remaining source block supplies evaluation. Each block's raw EEG is
reused across folds, so the 9 technical cells are never independent subjects.
K_F uses future idle labels and is not a prospective operating point. G1 fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from scripts.decompose_calibration_error_20261008 import (
    ROOT, STEP, _tune_ma, alarms_exact, compress_selected_ma,
    load_case, source_rest_mask,
)
from scripts.audit_p1_quantization_bias_20261008 import make_load_args
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from uncued_mi.decision import guard_local_moving_average

MODELS = ("riemannian", "cnn-gru")
LADDER = ((1, 15), (2, 30), (4, 60))
TARGET_FP = 4.


def balanced_calibration_mask(masks: list[np.ndarray], heldout: int,
                              length_seconds: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Take L/2 seconds from each of two distinct source blocks, in order."""
    if len(masks) != 3 or heldout not in (0, 1, 2):
        raise ValueError("three source masks and one heldout block required")
    selected = np.zeros_like(masks[0], dtype=bool)
    source_ids = tuple(source for source in (0, 1, 2) if source != heldout)
    half_steps = int(round(length_seconds / (2. * STEP)))
    for source in source_ids:
        if np.any(masks[source] & masks[heldout]):
            raise AssertionError("calibration/evaluation raw source overlap")
        indices = np.flatnonzero(masks[source])
        if len(indices) < half_steps:
            raise ValueError(f"source {source}: {len(indices)} < {half_steps} allowed steps")
        selected[indices[:half_steps]] = True
    if np.any(selected & masks[heldout]) or int(selected.sum()) != int(round(length_seconds / STEP)):
        raise AssertionError("balanced calibration support changed")
    return selected, source_ids


def case_rows(model: str, session: int, subject: int, args: argparse.Namespace) -> list[dict]:
    data = load_case(model, session, subject, make_load_args())
    anchor = data["anchor"]
    z = ewma_z(data["raw"], float(anchor.mean()),
               float(anchor.var(ddof=1)), data["allowed"])
    ma = guard_local_moving_average(z, data["allowed"], ma_steps=5)
    rows = []
    alarm_cache = {}
    for fold in (2, 3, 4):
        masks = [source_rest_mask(
            data["events"], data["source_ids"], data["starts"],
            data["times"], data["allowed"], fold, source,
        ) for source in (0, 1, 2)]
        for heldout in (0, 1, 2):
            eval_mask = masks[heldout]
            exposure_seconds = float(eval_mask.sum() * STEP)
            if not (35. <= exposure_seconds <= 50.):
                raise AssertionError(f"unexpected one-block idle exposure: {exposure_seconds}")
            for expected_alarms, length in LADDER:
                selected, sources = balanced_calibration_mask(masks, heldout, length)
                compressed_ma, compressed_allowed = compress_selected_ma(ma, selected)
                point = _tune_ma(compressed_ma, compressed_allowed, length, TARGET_FP)
                threshold = float(point["threshold"])
                if threshold not in alarm_cache:
                    alarm_cache[threshold] = alarms_exact(ma, threshold, data["allowed"])
                count = int(np.count_nonzero(alarm_cache[threshold] & eval_mask))
                actual_fp = count * 60. / exposure_seconds
                attained_rate = float(point["calibration_fp_per_min"])
                rows.append({
                    "model": model, "session": session, "subject": subject,
                    "future_fold": fold, "heldout_rest_source": heldout,
                    "calibration_sources": ",".join(map(str, sources)),
                    "target_fp_per_min": TARGET_FP,
                    "expected_calibration_alarms": expected_alarms,
                    "calibration_seconds": length,
                    "calibration_unique_source_count": 2,
                    "calibration_half_source_seconds": length / 2.,
                    "calibration_alarm_count": int(point["calibration_alarm_count"]),
                    "calibration_attained_fp_per_min": attained_rate,
                    "calibration_target_attained": bool(point["calibration_alarm_count"] == expected_alarms),
                    "threshold": threshold,
                    "evaluation_rest_seconds": exposure_seconds,
                    "evaluation_alarm_count": count,
                    "evaluation_fp_per_min": actual_fp,
                    "signed_fp_rate_error_per_min": actual_fp - attained_rate,
                })
    return rows


def bootstrap_summary(matrix: np.ndarray, draws: int, seed: int) -> dict:
    """Descriptive mean bias and law-of-total-variance at subject level."""
    subject_means = matrix.mean(axis=1)
    within_vars = matrix.var(axis=1, ddof=0)
    between_sd = float(subject_means.std(ddof=0))
    within_sd = float(np.sqrt(within_vars.mean()))
    total_sd = float(matrix.std(ddof=0))
    if not np.isclose(total_sd ** 2, between_sd ** 2 + within_sd ** 2,
                      rtol=1e-10, atol=1e-10):
        raise AssertionError("variance decomposition failed")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(matrix), size=(draws, len(matrix)))
    return {
        "mean_subject_signed_fp_rate_bias": float(subject_means.mean()),
        "mean_bias_subject_bootstrap_ci95": np.quantile(subject_means[ix].mean(axis=1), [.025, .975]).tolist(),
        "median_subject_signed_fp_rate_bias": float(np.median(subject_means)),
        "between_subject_sd": between_sd,
        "between_sd_subject_bootstrap_ci95": np.quantile(subject_means[ix].std(axis=1, ddof=0), [.025, .975]).tolist(),
        "within_subject_cell_sd": within_sd,
        "within_sd_subject_bootstrap_ci95": np.quantile(np.sqrt(within_vars[ix].mean(axis=1)), [.025, .975]).tolist(),
        "cell_sd": total_sd,
    }


def summarize(rows: list[dict], draws: int) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["model"], row["session"], row["subject"],
               row["expected_calibration_alarms"])
        groups[key].append(row)
    output, matrices = [], {}
    for model in MODELS:
        for session in (1, 2):
            for expected, length in LADDER:
                values = []
                fp_values = []
                exposure = []
                attained = []
                counts = Counter()
                for subject in range(1, 55):
                    members = sorted(groups[(model, session, subject, expected)],
                                     key=lambda r: (r["future_fold"], r["heldout_rest_source"]))
                    if len(members) != 9:
                        raise AssertionError("missing fold/heldout technical cells")
                    values.append([r["signed_fp_rate_error_per_min"] for r in members])
                    fp_values.append([r["evaluation_fp_per_min"] for r in members])
                    exposure.extend(r["evaluation_rest_seconds"] for r in members)
                    attained.extend(r["calibration_target_attained"] for r in members)
                    counts.update(r["calibration_alarm_count"] for r in members)
                matrix = np.asarray(values, dtype=float)
                matrices[(model, session, expected)] = matrix
                fp_subject_means = np.asarray(fp_values).mean(axis=1)
                output.append({
                    "model": model, "session": session, "target_fp_per_min": TARGET_FP,
                    "expected_calibration_alarms": expected,
                    "calibration_seconds": length, "subjects": 54,
                    "technical_cells": int(matrix.size),
                    "attained_count_fraction": float(np.mean(attained)),
                    "attained_count_distribution": {str(k): v for k, v in sorted(counts.items())},
                    "one_block_evaluation_seconds_min_median_max": [
                        float(np.min(exposure)), float(np.median(exposure)), float(np.max(exposure))],
                    "median_subject_mean_eval_fp_per_min": float(np.median(fp_subject_means)),
                    "subject_mean_eval_fp_q90": float(np.quantile(fp_subject_means, .9)),
                    **bootstrap_summary(matrix, draws, 20261008 + session * 100
                                        + expected + 1000 * MODELS.index(model)),
                })
    paired = []
    for model in MODELS:
        for session in (1, 2):
            for a, b in ((1, 2), (2, 4), (1, 4)):
                diff = matrices[(model, session, b)].mean(axis=1) - matrices[(model, session, a)].mean(axis=1)
                rng = np.random.default_rng(20261108 + session * 100 + a * 10 + b
                                            + 1000 * MODELS.index(model))
                ix = rng.integers(0, len(diff), (draws, len(diff)))
                paired.append({
                    "model": model, "session": session, "target_fp_per_min": TARGET_FP,
                    "later_minus_earlier_alarms": f"{b}-{a}",
                    "mean_subject_paired_signed_bias_change": float(diff.mean()),
                    "paired_change_subject_bootstrap_ci95": np.quantile(diff[ix].mean(axis=1), [.025, .975]).tolist(),
                })
    return output, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subjects", nargs="+", type=int)
    group.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--sessions", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "data/processed/calibration_error_decomposition_20261008/kf_fp4_event_ladder")
    args = parser.parse_args()
    subjects = list(range(1, 55)) if args.all_subjects else args.subjects
    if any(s < 1 or s > 54 for s in subjects):
        parser.error("subjects must be 1..54")
    rows = []
    for model in args.models:
        for session in args.sessions:
            for subject in subjects:
                rows.extend(case_rows(model, session, subject, args))
                if subject % 10 == 0:
                    print(f"{model} S{session}: {subject}/{subjects[-1]} subjects", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if subjects == list(range(1, 55)) and set(args.models) == set(MODELS) and set(args.sessions) == {1, 2}:
        summary, paired = summarize(rows, args.draws)
        path = args.output_root / "summary.json"
        path.write_text(json.dumps({
            "status": "post-hoc exploratory Lee2019 pseudo-uncued K_F diagnostic; G1 failed",
            "subjects": 54, "technical_cells_per_subject": 9,
            "calibration": "15/30/60s balanced between two source blocks, target FP4",
            "evaluation": "remaining one source block within each future fold",
            "metric": "future FP/min minus attained calibration FP/min; no absolute log error",
            "summary": summary, "paired_bias_change": paired,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {len(rows)} rows and {len(summary)} summary conditions: {path}", flush=True)
    else:
        print(f"saved {len(rows)} smoke rows: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
