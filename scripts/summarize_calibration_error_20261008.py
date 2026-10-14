"""Subject-level summary of exploratory P1 calibration-error rows.

Folds and K_X donors are technical repeats, not independent participants.
Bootstrap resamples subjects within a session; S1 and S2 are kept separate.
This script does not validate G1, EEG-level nulls, or causal deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("K_A", "K_F", "K_S", "K_X")


def subject_rows(input_root: Path, models: list[str], subjects: list[int], sessions: list[int]):
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for model in models:
        for session in sessions:
            for subject in subjects:
                path = input_root / model / f"session{session}" / f"sub-{subject:02d}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (payload["model"], payload["session"], payload["subject"]) != (model, session, subject):
                    raise AssertionError(f"row identity mismatch: {path}")
                for row in payload["rows"]:
                    key = (model, session, subject, int(row["length_seconds"]),
                           float(row["target_fp_per_min"]), row["calibration_kind"])
                    grouped[key].append(row)
    rows = []
    for (model, session, subject, length, target, kind), members in sorted(grouped.items()):
        if {int(row["future_fold"]) for row in members} != {2, 3, 4}:
            raise AssertionError(f"missing future fold: {(model, session, subject, length, target, kind)}")
        if kind == "K_X" and len(members) != 53 * 3 * 3:
            # Three held-out source blocks per fold; 53 donors per block.
            raise AssertionError(f"K_X donor cells missing: {(model, session, subject, length, target)}")
        if kind != "K_X" and len(members) != 3 * 3:
            raise AssertionError(f"calibration cells missing: {(model, session, subject, length, target, kind)}")
        values = {field: np.asarray([row[field] for row in members], dtype=float)
                  for field in ("signed_log_fp_error", "abs_log_fp_error",
                                "future_fp_per_min", "within_future_target_band")}
        rows.append({
            "model": model, "session": session, "subject": subject,
            "length_seconds": length, "target_fp_per_min": target,
            "calibration_resolution_valid": not (length == 14 and target == 2.),
            "calibration_kind": kind, "technical_cells": len(members),
            "mean_signed_log_fp_error": float(values["signed_log_fp_error"].mean()),
            "mean_abs_log_fp_error": float(values["abs_log_fp_error"].mean()),
            "mean_future_fp_per_min": float(values["future_fp_per_min"].mean()),
            "future_fp_q90_across_technical_cells": float(np.quantile(values["future_fp_per_min"], .9)),
            "future_admissible_fraction_across_technical_cells": float(values["within_future_target_band"].mean()),
        })
    return rows


def bootstrap_median(values: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return np.quantile(medians, [.025, .975]).tolist()


def summarize(rows: list[dict], draws: int) -> list[dict]:
    by_condition: dict[tuple, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (row["model"], row["session"], row["length_seconds"], row["target_fp_per_min"])
        by_condition[key][row["subject"]][row["calibration_kind"]] = row
    summary = []
    for (model, session, length, target), subjects in sorted(by_condition.items()):
        sample = sorted(subjects)
        for kind in KINDS:
            available = [subjects[s][kind] for s in sample if kind in subjects[s]]
            if not available:
                continue
            values = np.asarray([item["mean_abs_log_fp_error"] for item in available])
            summary.append({
                "model": model, "session": session, "length_seconds": length,
                "target_fp_per_min": target, "contrast": kind,
                "calibration_resolution_valid": not (length == 14 and target == 2.),
                "subjects": len(available),
                "median_abs_log_fp_error": float(np.median(values)),
                "median_abs_log_fp_error_ci95": bootstrap_median(
                    values, draws, 20261008 + session * 100 + length + int(target * 10)
                    + KINDS.index(kind),
                ),
                "median_subject_mean_fp_per_min": float(np.median([
                    item["mean_future_fp_per_min"] for item in available
                ])),
                "subject_mean_fp_q90": float(np.quantile([
                    item["mean_future_fp_per_min"] for item in available
                ], .9)),
                "mean_subject_admissible_fraction": float(np.mean([
                    item["future_admissible_fraction_across_technical_cells"]
                    for item in available
                ])),
            })
        for name, left, right in (("delta_C_abs_A_minus_F", "K_A", "K_F"),
                                  ("delta_B_abs_X_minus_A", "K_X", "K_A")):
            matched = [s for s in sample if left in subjects[s] and right in subjects[s]]
            if not matched:
                continue
            diffs = np.asarray([
                subjects[s][left]["mean_abs_log_fp_error"]
                - subjects[s][right]["mean_abs_log_fp_error"]
                for s in matched
            ])
            summary.append({
                "model": model, "session": session, "length_seconds": length,
                "target_fp_per_min": target, "contrast": name,
                "calibration_resolution_valid": not (length == 14 and target == 2.),
                "subjects": len(matched),
                "median_paired_difference": float(np.median(diffs)),
                "median_paired_difference_ci95": bootstrap_median(
                    diffs, draws, 20261108 + session * 100 + length + int(target * 10)
                    + (0 if name.startswith("delta_C") else 1),
                ),
            })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("riemannian", "cnn-gru"), default=["riemannian", "cnn-gru"])
    subjects = parser.add_mutually_exclusive_group(required=True)
    subjects.add_argument("--subjects", nargs="+", type=int)
    subjects.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--sessions", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/processed/calibration_error_decomposition_20261008/rows")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/calibration_error_decomposition_20261008")
    args = parser.parse_args()
    args.subjects = list(range(1, 55)) if args.all_subjects else args.subjects
    if args.draws < 100:
        parser.error("at least 100 bootstrap draws required")
    rows = subject_rows(args.input_root, args.models, args.subjects, args.sessions)
    summary = summarize(rows, args.draws)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "subject_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = args.output_root / "summary.json"
    path.write_text(json.dumps({
        "status": "exploratory simulation; G1 failed; K_X is source-to-unseen-target transfer",
        "subjects": args.subjects, "sessions": args.sessions, "models": args.models,
        "bootstrap_draws_subject_level": args.draws,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} subject rows and {len(summary)} summary cells", flush=True)
    print(f"{csv_path}\n{path}", flush=True)


if __name__ == "__main__":
    main()
