from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session1", type=Path, required=True)
    parser.add_argument("--session2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = []
    for session, path in ((1, args.session1), (2, args.session2)):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for fp in (1.0, 2.0, 4.0):
            for calibration in ("no_calibration", "idle_feature_alignment"):
                selected = [
                    x for x in rows
                    if x["decision"] == "ma_dwell"
                    and x["calibration"] == calibration
                    and float(x["target_fp_setting"]) == fp
                ]
                output.append({
                    "session": session,
                    "target_fp": fp,
                    "calibration": calibration,
                    "subjects": len(selected),
                    "dead_gate_rate": float(np.mean([
                        x["iti"]["fp_per_minute"] == 0
                        and x["iti"]["true_positive_rate"] == 0
                        for x in selected
                    ])),
                    "fp_per_minute_median": float(np.median([
                        x["iti"]["fp_per_minute"] for x in selected
                    ])),
                    "tpr_median": float(np.median([
                        x["iti"]["true_positive_rate"] for x in selected
                    ])),
                    "source_fp_per_minute_median": float(np.median([
                        x["operating_point"]["source_fp_per_minute"] for x in selected
                    ])),
                })
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
