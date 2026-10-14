"""Add pooled Cho2017 noise-stress rates and subject-bootstrap rate ratios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NOISE_NAMES = {
    "noise_1": "eye_blinking",
    "noise_2": "eyeball_up_down",
    "noise_3": "eyeball_left_right",
    "noise_4": "jaw_clenching",
    "noise_5": "head_left_right",
}


def ratio(sample: list[dict], noise_key: str) -> tuple[float, float, float]:
    clean_alarms = sum(row["rest"]["false_alarms"] for row in sample)
    clean_seconds = sum(row["rest"]["idle_seconds"] for row in sample)
    noise = [
        item
        for row in sample
        for item in row["noise"]
        if item["noise"] == noise_key
    ]
    noise_alarms = sum(item["false_alarms"] for item in noise)
    noise_seconds = sum(item["idle_seconds"] for item in noise)
    clean_rate = 60.0 * clean_alarms / clean_seconds
    noise_rate = 60.0 * noise_alarms / noise_seconds
    return noise_rate, clean_rate, noise_rate / clean_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/cho2017_transfer.json")
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    summaries = []
    rng = np.random.default_rng(20260725)

    for target_fp in (1.0, 2.0, 4.0):
        rows = [
            row
            for row in result["rows"]
            if row["target_fp_setting"] == target_fp
            and row["calibration"] == "idle_feature_alignment"
        ]
        for source_name, semantic_name in NOISE_NAMES.items():
            noise_rate, clean_rate, point = ratio(rows, source_name)
            boot = []
            for _ in range(args.bootstrap):
                sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
                try:
                    boot.append(ratio(sampled, source_name)[2])
                except ZeroDivisionError:
                    pass
            summaries.append(
                {
                    "target_fp_setting": target_fp,
                    "noise": semantic_name,
                    "source_cell": source_name,
                    "subjects": len(rows),
                    "pooled_fp_per_minute": noise_rate,
                    "clean_rest_pooled_fp_per_minute": clean_rate,
                    "rate_ratio_vs_clean_rest": point,
                    "rate_ratio_subject_bootstrap_95_ci": [
                        float(value) for value in np.quantile(boot, [0.025, 0.975])
                    ],
                }
            )
    result["protocol"]["noise_labels"] = (
        "Official Cho2017 source order: eye blinking, eye up/down, eye left/right, "
        "jaw clenching, head left/right"
    )
    result["noise_summary"] = summaries
    args.input.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
