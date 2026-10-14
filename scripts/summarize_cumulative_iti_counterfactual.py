from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    blocks = payload["trial_block_curve"]
    output = []
    for session in (1, 2):
        for start in range(1, 101, 10):
            selected = {
                row["condition"]: row
                for row in blocks
                if row["session"] == session and row["trial_block"][0] == start
            }
            static = selected["pre_rest_only"]
            adaptive = selected["pre_rest_plus_past_iti"]
            output.append(
                {
                    "session": session,
                    "trial_block": [start, start + 9],
                    "calibration_seconds": adaptive["calibration_seconds_median"],
                    "delta_fp_per_minute": adaptive["fp_per_minute"]
                    - static["fp_per_minute"],
                    "delta_excess_tpr": adaptive["excess_tpr"]
                    - static["excess_tpr"],
                    "static": static,
                    "adaptive": adaptive,
                }
            )
    summary = {}
    for session in (1, 2):
        values = np.asarray(
            [row["delta_excess_tpr"] for row in output if row["session"] == session]
        )
        fp_values = np.asarray(
            [row["delta_fp_per_minute"] for row in output if row["session"] == session]
        )
        summary[str(session)] = {
            "block_delta_excess_tpr_median": float(np.median(values)),
            "block_delta_excess_tpr_range": [float(np.min(values)), float(np.max(values))],
            "block_delta_excess_tpr_wilcoxon_p": float(wilcoxon(values).pvalue),
            "block_delta_fp_per_minute_median": float(np.median(fp_values)),
        }
    result = {"summary": summary, "blocks": output}
    target = args.output or args.input.with_name(args.input.stem + "_summary.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
