from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def finite(values):
    return np.asarray([value for value in values if value is not None and np.isfinite(value)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/tier0_loso_summary.json")
    )
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    row_lookup = {
        (
            row["subject"],
            row["session"],
            row["decision"],
            row["calibration"],
            row["target_fp_setting"],
        ): row
        for row in rows
    }
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["session"],
            row["decision"],
            row["calibration"],
            row["target_fp_setting"],
        )
        groups[key].append(row)

    summary = []
    for key, group in sorted(groups.items()):
        n = finite([row["fp_ratio_n"] for row in group])
        fp = finite([row["iti"]["fp_per_minute"] for row in group])
        tpr = finite([row["iti"]["true_positive_rate"] for row in group])
        delay = finite([row["iti"]["median_delay_seconds"] for row in group])
        vep = finite([row["fixation_onset"]["fp_per_minute"] for row in group])
        oracle_fraction = finite([row["tpr_fraction_of_oracle"] for row in group])
        recoveries = []
        if key[2] == "idle_feature_alignment":
            for row in group:
                baseline = row_lookup.get(
                    (
                        row["subject"],
                        row["session"],
                        row["decision"],
                        "no_calibration",
                        row["target_fp_setting"],
                    )
                )
                if baseline is None:
                    continue
                target = row["target_fp_setting"]
                baseline_error = abs(baseline["iti"]["fp_per_minute"] - target)
                aligned_error = abs(row["iti"]["fp_per_minute"] - target)
                oracle_error = abs(row["oracle_fp_per_minute"] - target)
                denominator = baseline_error - oracle_error
                if denominator > 0:
                    recoveries.append(
                        (baseline_error - aligned_error) / denominator
                    )
        summary.append(
            {
                "session": key[0],
                "decision": key[1],
                "calibration": key[2],
                "target_fp_setting": key[3],
                "subjects": len(group),
                "n_median": float(np.median(n)) if len(n) else None,
                "n_iqr": (
                    [float(x) for x in np.quantile(n, [0.25, 0.75])]
                    if len(n)
                    else None
                ),
                "n_gt_3_fraction": float(np.mean(n > 3)) if len(n) else None,
                "fp_per_minute_median": float(np.median(fp)) if len(fp) else None,
                "tpr_median": float(np.median(tpr)) if len(tpr) else None,
                "delay_median_seconds": (
                    float(np.median(delay)) if len(delay) else None
                ),
                "fixation_fp_per_minute_median": (
                    float(np.median(vep)) if len(vep) else None
                ),
                "tpr_fraction_of_oracle_median": (
                    float(np.median(oracle_fraction))
                    if len(oracle_fraction)
                    else None
                ),
                "operating_point_recovery_median": (
                    float(np.median(recoveries)) if recoveries else None
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
