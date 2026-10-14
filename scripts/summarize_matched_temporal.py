from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/matched_lda_scaling_subset20.json"),
    )
    parser.add_argument("--target-fp", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/matched_lda_temporal_summary.json"),
    )
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    output = []
    for arm in sorted({row["arm"] for row in rows}):
        group = [
            row
            for row in rows
            if row["arm"] == arm
            and float(row["target_fp_axis"]) == args.target_fp
        ]
        collapsed = []
        for subject in sorted({row["target_subject"] for row in group}):
            subject_rows = [row for row in group if row["target_subject"] == subject]
            collapsed.append(
                {
                    key: float(np.median([row[key] for row in subject_rows]))
                    for key in (
                        "test_window_auc",
                        "null_tpr_median",
                        "excess_tpr",
                        "hit_density_mean",
                        "command_score_ar1_mean",
                        "command_fraction_above_idle_median",
                        "fragmentation_bursts_per_exceed_second_mean",
                    )
                }
            )
        output.append(
            {
                "arm": arm,
                "target_fp_axis": args.target_fp,
                **{
                    f"{key}_median": float(np.median([row[key] for row in collapsed]))
                    for key in collapsed[0]
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
