from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rate_ratio(
    fixation_counts: np.ndarray,
    fixation_exposure: np.ndarray,
    rest_counts: np.ndarray,
    rest_exposure: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict:
    def pooled_ratio(index: np.ndarray) -> float:
        fix_rate = np.sum(fixation_counts[index]) / np.sum(fixation_exposure[index])
        rest_rate = np.sum(rest_counts[index]) / np.sum(rest_exposure[index])
        if rest_rate == 0:
            return float("inf") if fix_rate > 0 else 1.0
        return float(fix_rate / rest_rate)

    subjects = np.arange(len(fixation_counts))
    observed = pooled_ratio(subjects)
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [
            pooled_ratio(rng.choice(subjects, size=len(subjects), replace=True))
            for _ in range(bootstrap)
        ]
    )
    finite = boot[np.isfinite(boot)]
    ci = (
        [float(value) for value in np.quantile(finite, [0.025, 0.975])]
        if len(finite)
        else [None, None]
    )
    if observed > 0 and np.isfinite(observed):
        permuted = []
        for _ in range(bootstrap):
            swap = rng.random(len(subjects)) < 0.5
            fix_c = np.where(swap, rest_counts, fixation_counts)
            fix_e = np.where(swap, rest_exposure, fixation_exposure)
            rest_c = np.where(swap, fixation_counts, rest_counts)
            rest_e = np.where(swap, fixation_exposure, rest_exposure)
            fix_rate = np.sum(fix_c) / np.sum(fix_e)
            rest_rate = np.sum(rest_c) / np.sum(rest_e)
            permuted.append(fix_rate / rest_rate if rest_rate > 0 else np.inf)
        permuted = np.asarray(permuted)
        valid = (permuted > 0) & np.isfinite(permuted)
        p_value = float(
            (1 + np.sum(np.abs(np.log(permuted[valid])) >= abs(np.log(observed))))
            / (1 + np.sum(valid))
        )
    else:
        p_value = None
    return {
        "rate_ratio_fixation_vs_post_rest": observed,
        "cluster_bootstrap_95_ci": ci,
        "cluster_bootstrap_p_value": p_value,
        "fixation_events": int(np.sum(fixation_counts)),
        "fixation_exposure_seconds": float(np.sum(fixation_exposure)),
        "post_rest_events": int(np.sum(rest_counts)),
        "post_rest_exposure_seconds": float(np.sum(rest_exposure)),
        "subjects": len(subjects),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/vep_rate_analysis.json")
    )
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in args.inputs:
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = (
                row["session"],
                row["decision"],
                row["calibration"],
                row["target_fp_setting"],
            )
            groups[key].append(row)

    output = []
    for key, rows in sorted(groups.items()):
        if len(rows) != 54:
            raise ValueError(f"{key}: expected 54 subjects, found {len(rows)}")
        result = rate_ratio(
            fixation_counts=np.asarray(
                [row["fixation_onset"]["false_alarms"] for row in rows]
            ),
            fixation_exposure=np.asarray(
                [row["fixation_onset"]["idle_seconds"] for row in rows]
            ),
            rest_counts=np.asarray(
                [row["post_rest"]["false_alarms"] for row in rows]
            ),
            rest_exposure=np.asarray(
                [row["post_rest"]["idle_seconds"] for row in rows]
            ),
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        output.append(
            {
                "session": key[0],
                "decision": key[1],
                "calibration": key[2],
                "target_fp_setting": key[3],
                **result,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {len(output)} VEP rate comparisons to {args.output}")


if __name__ == "__main__":
    main()
