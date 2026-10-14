"""Compare valid Riemannian Arm-A and Arm-C screening sets to an exact null."""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

N_SUBJECTS = 54


def hypergeom_pmf(k: int, n_a: int, n_c: int) -> float:
    if k < max(0, n_a + n_c - N_SUBJECTS) or k > min(n_a, n_c):
        return 0.0
    return comb(n_a, k) * comb(N_SUBJECTS - n_a, n_c - k) / comb(N_SUBJECTS, n_c)


def session_result(a_ids: set[int], c_ids: set[int]) -> dict:
    overlap = len(a_ids & c_ids)
    union = len(a_ids | c_ids)
    expected_overlap = len(a_ids) * len(c_ids) / N_SUBJECTS
    support = range(max(0, len(a_ids) + len(c_ids) - N_SUBJECTS), min(len(a_ids), len(c_ids)) + 1)
    pmf = {k: hypergeom_pmf(k, len(a_ids), len(c_ids)) for k in support}
    expected_jaccard = sum((k / (len(a_ids) + len(c_ids) - k)) * p for k, p in pmf.items())
    return {
        "n_a": len(a_ids), "n_c": len(c_ids), "overlap": overlap,
        "jaccard": overlap / union,
        "expected_overlap": expected_overlap,
        "expected_jaccard": expected_jaccard,
        "p_overlap_le_observed": sum(p for k, p in pmf.items() if k <= overlap),
        "p_overlap_ge_observed": sum(p for k, p in pmf.items() if k >= overlap),
        "a_subjects": sorted(a_ids), "c_subjects": sorted(c_ids),
    }


def main() -> None:
    loso = json.loads(Path("data/processed/lee2019_loso_deployment.json").read_text(encoding="utf-8"))
    within = json.loads(Path("data/processed/lee2019_screening_fp_latency.json").read_text(encoding="utf-8"))
    results = {}
    for session in (1, 2):
        a_ids = set(loso["summary"][str(session)]["A_target_rest"]["selected_subjects"])
        c_ids = {
            int(row["subject"]) for row in within["rows"]
            if int(row["session"]) == session and bool(row.get("selected"))
        }
        results[str(session)] = session_result(a_ids, c_ids)
    observed_total = sum(row["overlap"] for row in results.values())
    pmfs = []
    for row in results.values():
        support = range(max(0, row["n_a"] + row["n_c"] - N_SUBJECTS), min(row["n_a"], row["n_c"]) + 1)
        pmfs.append({k: hypergeom_pmf(k, row["n_a"], row["n_c"]) for k in support})
    combined_p_ge = sum(
        p1 * p2 for k1, p1 in pmfs[0].items() for k2, p2 in pmfs[1].items()
        if k1 + k2 >= observed_total
    )
    payload = {
        "comparison": "Riemannian LOSO Arm A vs valid target-specific Riemannian Arm C",
        "null": "independent uniform sets with observed per-session set sizes; exact hypergeometric",
        "sessions": results,
        "combined": {
            "observed_total_overlap": observed_total,
            "expected_total_overlap": sum(row["expected_overlap"] for row in results.values()),
            "p_total_overlap_ge_observed": combined_p_ge,
        },
        "interpretation_rule": "score dependence requires overlap to differ from the exact random-set null; otherwise it remains unverified",
    }
    output = Path("data/processed/lee2019_screening_arm_a_vs_c_jaccard.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
