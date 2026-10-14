from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loso",
        type=Path,
        default=Path(
            "data/processed/detection_circular_shift_null_second_half.json"
        ),
    )
    parser.add_argument(
        "--within",
        type=Path,
        default=Path("data/processed/within_subject_gate.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/loso_vs_within_subject_matched.json"),
    )
    args = parser.parse_args()
    loso = json.loads(args.loso.read_text(encoding="utf-8"))["rows"]
    within = json.loads(args.within.read_text(encoding="utf-8"))["rows"]
    loso_lookup = {
        (
            row["subject"],
            row["session"],
            row["decision"],
            float(row["target_fp_setting"]),
        ): row
        for row in loso
    }
    within_lookup = {
        (
            row["subject"],
            row["session"],
            row["decision"],
            float(row["target_fp_setting"]),
        ): row
        for row in within
    }
    keys = sorted(set(loso_lookup) & set(within_lookup))
    groups = sorted({key[1:] for key in keys})
    rng = np.random.default_rng(args.seed)
    output = []
    for session, decision, target_fp in groups:
        selected = [
            key
            for key in keys
            if key[1:] == (session, decision, target_fp)
        ]
        loso_excess = np.asarray(
            [loso_lookup[key]["observed_minus_null_median"] for key in selected]
        )
        within_excess = np.asarray(
            [within_lookup[key]["excess_tpr"] for key in selected]
        )
        difference = within_excess - loso_excess
        boot = np.asarray(
            [
                np.median(
                    difference[
                        rng.integers(0, len(difference), size=len(difference))
                    ]
                )
                for _ in range(args.bootstrap)
            ]
        )
        try:
            p_value = float(wilcoxon(difference).pvalue)
        except ValueError:
            p_value = 1.0
        output.append(
            {
                "session": session,
                "decision": decision,
                "target_fp_setting": target_fp,
                "subjects": len(selected),
                "loso_excess_tpr_median": float(np.median(loso_excess)),
                "within_excess_tpr_median": float(np.median(within_excess)),
                "paired_within_minus_loso_median": float(np.median(difference)),
                "paired_median_bootstrap_95_ci": [
                    float(value) for value in np.quantile(boot, [0.025, 0.975])
                ],
                "paired_wilcoxon_p_value": p_value,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
