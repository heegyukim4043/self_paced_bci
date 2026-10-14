from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def collapse(rows: list[dict], arm: str, target_fp: float) -> dict[int, float]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm and float(row["target_fp_axis"]) == target_fp
    ]
    return {
        subject: float(
            np.median(
                [
                    row["excess_tpr"]
                    for row in selected
                    if row["target_subject"] == subject
                ]
            )
        )
        for subject in sorted({row["target_subject"] for row in selected})
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/matched_lda_scaling_subset20.json"),
    )
    parser.add_argument("--target-fp", type=float, default=2.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/matched_lda_scaling_subset20_inference.json"),
    )
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    target = collapse(rows, "target_50", args.target_fp)
    rng = np.random.default_rng(args.seed)
    output = []
    for comparator in (
        "source_n_matched_50_trials",
        "source_5",
        "source_10",
        "source_20",
        "source_40",
        "source_all_53",
    ):
        source = collapse(rows, comparator, args.target_fp)
        subjects = sorted(set(target) & set(source))
        differences = np.asarray([target[s] - source[s] for s in subjects])
        boot = np.asarray(
            [
                np.median(
                    differences[
                        rng.integers(0, len(differences), size=len(differences))
                    ]
                )
                for _ in range(args.bootstrap)
            ]
        )
        output.append(
            {
                "target_fp_axis": args.target_fp,
                "comparison": f"target_50_minus_{comparator}",
                "subjects": len(subjects),
                "paired_median_difference": float(np.median(differences)),
                "bootstrap_95_ci": [
                    float(value) for value in np.quantile(boot, [0.025, 0.975])
                ],
                "wilcoxon_p": float(wilcoxon(differences).pvalue),
            }
        )

    repeat_summary = []
    for arm in (
        "source_n_matched_50_trials",
        "source_5",
        "source_10",
        "source_20",
        "source_40",
    ):
        arm_rows = [
            row
            for row in rows
            if row["arm"] == arm
            and float(row["target_fp_axis"]) == args.target_fp
        ]
        repeat_medians = [
            float(
                np.median(
                    [
                        row["excess_tpr"]
                        for row in arm_rows
                        if row["repeat"] == repeat
                    ]
                )
            )
            for repeat in sorted({row["repeat"] for row in arm_rows})
        ]
        repeat_summary.append(
            {
                "arm": arm,
                "repeat_cohort_median_excess": repeat_medians,
                "mean": float(np.mean(repeat_medians)),
                "range": [
                    float(np.min(repeat_medians)),
                    float(np.max(repeat_medians)),
                ],
            }
        )
    payload = {"paired_inference": output, "repeat_summary": repeat_summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
