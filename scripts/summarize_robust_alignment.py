from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path: Path, calibration: str) -> list[dict]:
    return [
        row for row in json.loads(path.read_text(encoding="utf-8"))
        if row["decision"] == "ma_dwell" and row["calibration"] == calibration
    ]


def summarize(selected: list[dict]) -> list[dict]:
    output = []
    for fp in (1.0, 2.0, 4.0):
        values = [row for row in selected if float(row["target_fp_setting"]) == fp]
        output.append({
            "target_fp": fp,
            "subjects": len(values),
            "fp_per_minute_median": float(np.median([x["iti"]["fp_per_minute"] for x in values])),
            "tpr_median": float(np.median([x["iti"]["true_positive_rate"] for x in values])),
            "dead_gate_rate": float(np.mean([
                x["iti"]["fp_per_minute"] == 0 and x["iti"]["true_positive_rate"] == 0
                for x in values
            ])),
            "source_fp_median": float(np.median([
                x["operating_point"]["source_fp_per_minute"] for x in values
            ])),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--robust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "mean_std": summarize(rows(args.mean, "idle_feature_alignment")),
        "median_mad": summarize(rows(args.robust, "idle_feature_alignment_median_mad")),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
