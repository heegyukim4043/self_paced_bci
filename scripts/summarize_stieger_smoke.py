"""Summarize the fixed stratified Stieger smoke cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIXED_GROUPS = {"MBSR": (1, 2, 3), "control": (6, 8, 9)}


def bootstrap_median(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def arm_values(rows: list[dict], arm: str, path: tuple[str, ...]) -> np.ndarray:
    values = []
    for row in rows:
        value = row["report"]["arms"][arm]
        for key in path:
            value = value[key]
        values.append(float(value))
    return np.asarray(values)


def summarize_rows(rows: list[dict], bootstrap: int, seed: int) -> dict:
    output = {
        "n": len(rows),
        "subjects": [row["subject"] for row in rows],
        "groups": [row["group"] for row in rows],
    }
    cursor = np.asarray([row["report"]["cursor_only_window_auc"] for row in rows])
    output["cursor_only_auc_median"] = float(np.median(cursor))
    for arm in ("raw", "speed_residualized"):
        window_auc = arm_values(rows, arm, ("window_auc",))
        matched_auc = arm_values(rows, arm, ("speed_matched_window_auc",))
        transferred_fp = arm_values(rows, arm, ("event", "fp_per_minute"))
        transferred_excess = arm_values(rows, arm, ("event", "excess_tpr"))
        oracle_excess = arm_values(
            rows, arm, ("oracle_common_fp_event_diagnostic", "excess_tpr")
        )
        output[arm] = {
            "window_auc_median": float(np.median(window_auc)),
            "speed_matched_window_auc_median": float(np.median(matched_auc)),
            "transferred_fp_per_minute_median": float(np.median(transferred_fp)),
            "transferred_excess_median": float(np.median(transferred_excess)),
            "transferred_excess_ci": bootstrap_median(
                transferred_excess, seed, bootstrap
            ),
            "oracle_fp2_excess_median": float(np.median(oracle_excess)),
            "oracle_fp2_excess_ci": bootstrap_median(
                oracle_excess, seed + 1, bootstrap
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-pattern",
        default="data/processed/stieger2021_riemannian_smoke_S{subject}.json",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stieger2021_stratified_smoke_summary.json"),
    )
    args = parser.parse_args()
    rows = []
    for group, subjects in FIXED_GROUPS.items():
        for subject in subjects:
            path = Path(args.input_pattern.format(subject=subject))
            rows.append(
                {
                    "subject": subject,
                    "group": group,
                    "report": json.loads(path.read_text(encoding="utf-8")),
                }
            )
    summary = {
        "selection_rule": (
            "First three subject IDs available in each group before inspecting "
            "model outcomes."
        ),
        "fixed_groups": {key: list(value) for key, value in FIXED_GROUPS.items()},
        "overall": summarize_rows(rows, args.bootstrap, 20260728),
        "by_group": {
            group: summarize_rows(
                [row for row in rows if row["group"] == group],
                args.bootstrap,
                20260728 + index * 100,
            )
            for index, group in enumerate(FIXED_GROUPS)
        },
        "rows": [
            {
                "subject": row["subject"],
                "group": row["group"],
                "cursor_only_auc": row["report"]["cursor_only_window_auc"],
                "raw_window_auc": row["report"]["arms"]["raw"]["window_auc"],
                "raw_speed_matched_auc": row["report"]["arms"]["raw"][
                    "speed_matched_window_auc"
                ],
                "raw_transferred_fp": row["report"]["arms"]["raw"]["event"][
                    "fp_per_minute"
                ],
                "raw_transferred_excess": row["report"]["arms"]["raw"]["event"][
                    "excess_tpr"
                ],
                "raw_oracle_fp2_excess": row["report"]["arms"]["raw"][
                    "oracle_common_fp_event_diagnostic"
                ]["excess_tpr"],
                "residualized_oracle_fp2_excess": row["report"]["arms"][
                    "speed_residualized"
                ]["oracle_common_fp_event_diagnostic"]["excess_tpr"],
            }
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
