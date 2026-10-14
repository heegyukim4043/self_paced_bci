from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def select(path: Path) -> dict[tuple[int, str], dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["subject"]), row["calibration"]): row
        for row in rows
        if row["decision"] == "ma_dwell"
        and float(row["target_fp_setting"]) == 2.0
    }


def summarize(rows: dict[tuple[int, str], dict]) -> dict:
    output = {}
    for calibration in ("no_calibration", "idle_feature_alignment"):
        selected = [row for (_, c), row in rows.items() if c == calibration]
        output[calibration] = {
            "subjects": len(selected),
            "iti_fp_per_minute_median": float(
                np.median([row["iti"]["fp_per_minute"] for row in selected])
            ),
            "iti_tpr_median": float(
                np.median([row["iti"]["true_positive_rate"] for row in selected])
            ),
            "dead_gate_rate": float(
                np.mean(
                    [
                        row["iti"]["fp_per_minute"] == 0
                        and row["iti"]["true_positive_rate"] == 0
                        for row in selected
                    ]
                )
            ),
            "source_fp_per_minute_median": float(
                np.median(
                    [row["operating_point"]["source_fp_per_minute"] for row in selected]
                )
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"old": summarize(select(args.old)), "break_fixed": summarize(select(args.new))}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
