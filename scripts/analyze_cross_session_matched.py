from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon


def hodges_lehmann(values: np.ndarray) -> float:
    walsh = [
        (values[i] + values[j]) / 2
        for i in range(len(values))
        for j in range(i, len(values))
    ]
    return float(np.median(walsh))


def hl_bca_ci(values: np.ndarray, rng: np.random.Generator, n: int) -> list[float]:
    result = bootstrap(
        (values,),
        lambda x: hodges_lehmann(np.asarray(x)),
        n_resamples=n,
        method="BCa",
        random_state=rng,
    )
    return [float(result.confidence_interval.low), float(result.confidence_interval.high)]


def lookup(rows: list[dict], arm: str, target_fp: float) -> dict[int, dict]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm and float(row["target_fp_axis"]) == target_fp
    ]
    return {int(row["target_subject"]): row for row in selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session1",
        type=Path,
        default=Path("data/processed/matched_lda_scaling_subset20.json"),
    )
    parser.add_argument(
        "--session2",
        type=Path,
        default=Path("data/processed/matched_lda_scaling_subset20_session2.json"),
    )
    parser.add_argument(
        "--cross",
        type=Path,
        default=Path("data/processed/cross_session_matched_lda_subset20.json"),
    )
    parser.add_argument("--target-fp", type=float, default=2.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cross_session_matched_inference.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    same1 = lookup(
        json.loads(args.session1.read_text(encoding="utf-8"))["rows"],
        "target_50",
        args.target_fp,
    )
    same2 = lookup(
        json.loads(args.session2.read_text(encoding="utf-8"))["rows"],
        "target_50",
        args.target_fp,
    )
    cross_rows = json.loads(args.cross.read_text(encoding="utf-8"))["rows"]
    output = []
    losses = {}
    for direction, same, train_session, test_session in (
        ("s1_to_s2", same1, 1, 2),
        ("s2_to_s1", same2, 2, 1),
    ):
        directional = [
            row
            for row in cross_rows
            if row["direction"] == direction
            and float(row["target_fp_axis"]) == args.target_fp
        ]
        target = {
            row["target_subject"]: row
            for row in directional
            if row["arm"] == "target_cross_session"
        }
        source = {
            row["target_subject"]: row
            for row in directional
            if row["arm"] == "source_all_53_cross_session"
        }
        subjects = sorted(set(target) & set(source) & set(same))
        gap = np.asarray(
            [target[s]["excess_tpr"] - source[s]["excess_tpr"] for s in subjects]
        )
        loss = np.asarray(
            [same[s]["excess_tpr"] - target[s]["excess_tpr"] for s in subjects]
        )
        losses[direction] = dict(zip(subjects, loss))
        output.append(
            {
                "direction": direction,
                "train_session": train_session,
                "test_session": test_session,
                "target_cross_excess_median": float(
                    np.median([target[s]["excess_tpr"] for s in subjects])
                ),
                "source_cross_excess_median": float(
                    np.median([source[s]["excess_tpr"] for s in subjects])
                ),
                "paired_target_minus_source_hodges_lehmann": hodges_lehmann(gap),
                "target_minus_source_hl_bca_95_ci": hl_bca_ci(
                    gap, rng, args.bootstrap
                ),
                "target_minus_source_wilcoxon_p": float(wilcoxon(gap).pvalue),
                "same_session_minus_cross_session_target_hodges_lehmann": hodges_lehmann(loss),
                "loss_hl_bca_95_ci": hl_bca_ci(loss, rng, args.bootstrap),
                "loss_wilcoxon_p": float(wilcoxon(loss).pvalue),
            }
        )
    subjects = sorted(set(losses["s1_to_s2"]) & set(losses["s2_to_s1"]))
    directional_difference = np.asarray(
        [
            losses["s1_to_s2"][subject] - losses["s2_to_s1"][subject]
            for subject in subjects
        ]
    )
    hypothesis = {
        "contrast": "(S1 same - S1→S2) - (S2 same - S2→S1)",
        "paired_hodges_lehmann": hodges_lehmann(directional_difference),
        "hl_bca_95_ci": hl_bca_ci(
            directional_difference, rng, args.bootstrap
        ),
        "wilcoxon_p": float(wilcoxon(directional_difference).pvalue),
    }
    payload = {"directions": output, "directional_hypothesis": hypothesis}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
