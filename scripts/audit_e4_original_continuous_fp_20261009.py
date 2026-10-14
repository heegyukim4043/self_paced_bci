"""Independent row-level arithmetic and distribution audit for E4 FP output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def audit(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = result["rows"]
    if len(rows) != 54 * 2 * 2 * 2 * 3:
        raise AssertionError("missing E4 rows")
    cells = defaultdict(list)
    for row in rows:
        future = row["future"]
        calibration = row["calibration"]
        if not np.isfinite(calibration["threshold"]):
            raise AssertionError("nonfinite threshold")
        if not np.isclose(calibration["calibration_rate"],
                          calibration["calibration_count"] * 60 /
                          calibration["calibration_seconds"]):
            raise AssertionError("calibration rate/count mismatch")
        if not np.isclose(future["fp_per_min"], future["count"] * 60 /
                          future["seconds"]):
            raise AssertionError("future rate/count mismatch")
        strata = future["strata"]
        names = ("fixation", "iti_late") if row["mode"] == "late_only" else (
            "fixation", "iti_early", "iti_late")
        if sum(strata[name]["count"] for name in names) != future["count"]:
            raise AssertionError("future stratified counts do not partition total")
        if not np.isclose(sum(strata[name]["seconds"] for name in names),
                          future["seconds"]):
            raise AssertionError("future stratified exposure does not partition total")
        cells[(row["session"], row["mode"], row["target_fp"], row["arm"])].append(row)
    audit_summary = {}
    for (session, mode, target, arm), group in sorted(cells.items()):
        if len(group) != 54 or len({r["subject"] for r in group}) != 54:
            raise AssertionError("not 54 subject-level rows")
        group.sort(key=lambda r: r["subject"])
        future_fp = np.asarray([r["future"]["fp_per_min"] for r in group])
        original = result["summary"][str(session)][mode][str(target)]["arms"][arm]
        if not np.isclose(np.median(future_fp), original["median_fp_per_min"]):
            raise AssertionError("summary median mismatch")
        if not np.isclose(np.quantile(future_fp, .9), original["q90_fp_per_min"]):
            raise AssertionError("summary q90 mismatch")
        strata_result = {}
        for name in ("fixation", "iti_early", "iti_late"):
            rates = np.asarray([r["future"]["strata"][name]["fp_per_min"]
                                for r in group], dtype=float)
            strata_result[name] = {
                "median_fp_per_min": float(np.nanmedian(rates)),
                "pooled_fp_per_min": float(
                    60 * sum(r["future"]["strata"][name]["count"] for r in group) /
                    sum(r["future"]["strata"][name]["seconds"] for r in group)),
            }
        audit_summary[f"S{session}/{mode}/FP{target:g}/{arm}"] = {
            "future_fp_gt_2x_target_n": int(np.sum(future_fp > 2 * target)),
            "strata": strata_result,
            "median_calibration_threshold": float(np.median([
                r["calibration"]["threshold"] for r in group])),
            "median_candidate_count": float(np.median([
                r["calibration"]["candidate_count"] for r in group])),
        }
    return {"status": "row-level arithmetic and summary audit", "n_rows": len(rows),
            "n_cells": len(cells), "summary": audit_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/e4_original_continuous_fp_20261009_v2.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/e4_original_continuous_fp_audit_20261009_v2.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.input)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"audited {result['n_rows']} rows in {result['n_cells']} cells")


if __name__ == "__main__":
    main()
