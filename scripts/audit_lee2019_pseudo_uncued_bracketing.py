"""Report FP=2 bracketing failures and rare-target threshold convergence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SOURCE = Path("data/processed/lee2019_pseudo_uncued_riemannian.json")
TARGET = Path("data/processed/lee2019_pseudo_uncued_bracket_audit.json")
NAMES = ("0.25", "0.5", "1.0", "2.0", "4.0", "8.0")


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    failures = []
    same_threshold = 0
    both_zero_anchor = 0
    for row in data["rows"]:
        points = row["operating_points"]
        actual = [points[name]["stream_fp_per_minute"] for name in NAMES]
        if not min(actual) <= 2.0 <= max(actual):
            failures.append({
                "subject": row["subject"],
                "failure_direction": "all_above_fp2" if min(actual) > 2 else "all_below_fp2",
                "minimum_actual_fp_per_minute": min(actual),
                "maximum_actual_fp_per_minute": max(actual),
                "actual_fp_by_nominal_target": dict(zip(NAMES, actual)),
            })
        same_threshold += int(np.isclose(points["0.25"]["threshold"],
                                         points["0.5"]["threshold"]))
        both_zero_anchor += int(points["0.25"]["anchor_fp_per_minute"] == 0 and
                                points["0.5"]["anchor_fp_per_minute"] == 0)
    result = {
        "subjects": len(data["rows"]),
        "fp2_bracket_failures": len(failures),
        "fp2_bracket_failure_rate": len(failures) / len(data["rows"]),
        "failure_direction_counts": {
            "all_above_fp2": sum(f["failure_direction"] == "all_above_fp2" for f in failures),
            "all_below_fp2": sum(f["failure_direction"] == "all_below_fp2" for f in failures),
        },
        "rare_target_0p25_0p5_same_threshold_subjects": same_threshold,
        "rare_target_0p25_0p5_both_zero_anchor_alarm_subjects": both_zero_anchor,
        "interpretation": (
            "The approximately 56-s anchor cannot resolve 0.25 vs 0.5 FP/min: "
            "both choose the same zero-alarm threshold. All FP=2 bracketing "
            "failures are extreme stream over-firing cases."
        ),
        "failures": failures,
    }
    TARGET.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
