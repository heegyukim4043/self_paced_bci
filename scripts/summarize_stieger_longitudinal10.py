"""Paired Session-1 to Session-2 Stieger regime/adaptation smoke summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)
MBSR = {1, 2, 3, 4, 5}
BOOTSTRAPS = 20_000


def median_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (BOOTSTRAPS, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def paired_summary(s1: np.ndarray, s2: np.ndarray, seed: int) -> dict:
    delta = np.asarray(s2, float) - np.asarray(s1, float)
    try:
        pvalue = float(wilcoxon(delta).pvalue)
    except ValueError:
        pvalue = 1.0
    return {
        "session1_median": float(np.median(s1)),
        "session2_median": float(np.median(s2)),
        "paired_delta_median": float(np.median(delta)),
        "paired_delta_bootstrap_95_ci": median_ci(delta, seed),
        "subjects_increased": int(np.sum(delta > 0)),
        "subjects_decreased": int(np.sum(delta < 0)),
        "wilcoxon_p": pvalue,
    }


def main() -> None:
    s1_adapt = json.loads((ROOT / "data" / "processed" /
                           "stieger2021_expansion20_ewma.json").read_text(encoding="utf-8"))
    s2_adapt = json.loads((ROOT / "data" / "processed" /
                           "stieger2021_longitudinal10_session2_ewma.json").read_text(encoding="utf-8"))
    s1_by_subject = {int(row["subject"]): row for row in s1_adapt["rows"]}
    s2_by_subject = {int(row["subject"]): row for row in s2_adapt["rows"]}

    rows = []
    for subject in SUBJECTS:
        smoke1 = json.loads((ROOT / "data" / "processed" /
                             f"stieger2021_riemannian_smoke_S{subject}.json").read_text(encoding="utf-8"))
        smoke2 = json.loads((ROOT / "data" / "processed" /
                             f"stieger2021_riemannian_S{subject}_session2.json").read_text(encoding="utf-8"))
        a1, a2 = s1_by_subject[subject], s2_by_subject[subject]
        rows.append({
            "subject": subject,
            "group": "MBSR" if subject in MBSR else "control",
            "session1": {
                "window_auc": smoke1["arms"]["raw"]["window_auc"],
                "oracle_fp2_excess": smoke1["arms"]["raw"]["oracle_common_fp_event_diagnostic"]["excess_tpr"],
                "static_fp": a1["arms"]["raw"]["static"]["fp_per_minute"],
                "ewma_fp": a1["arms"]["raw"]["ewma_h30"]["fp_per_minute"],
                "ewma_excess": a1["arms"]["raw"]["ewma_h30"]["excess_tpr"],
            },
            "session2": {
                "window_auc": smoke2["arms"]["raw"]["window_auc"],
                "oracle_fp2_excess": smoke2["arms"]["raw"]["oracle_common_fp_event_diagnostic"]["excess_tpr"],
                "static_fp": a2["arms"]["raw"]["static"]["fp_per_minute"],
                "ewma_fp": a2["arms"]["raw"]["ewma_h30"]["fp_per_minute"],
                "ewma_excess": a2["arms"]["raw"]["ewma_h30"]["excess_tpr"],
            },
        })

    metrics = ("window_auc", "oracle_fp2_excess", "static_fp", "ewma_fp", "ewma_excess")
    summary = {}
    for index, metric in enumerate(metrics):
        s1 = np.asarray([row["session1"][metric] for row in rows])
        s2 = np.asarray([row["session2"][metric] for row in rows])
        summary[metric] = paired_summary(s1, s2, 20260808 + index)
        deltas = s2 - s1
        control = deltas[[row["group"] == "control" for row in rows]]
        mbsr = deltas[[row["group"] == "MBSR" for row in rows]]
        summary[metric]["control_delta_median"] = float(np.median(control))
        summary[metric]["mbsr_delta_median"] = float(np.median(mbsr))
        summary[metric]["group_delta_mannwhitney_p"] = float(
            mannwhitneyu(control, mbsr, alternative="two-sided").pvalue
        )

    # Within each session, adaptation must be compared against static.
    adaptation = {}
    for session in ("session1", "session2"):
        static = np.asarray([row[session]["static_fp"] for row in rows])
        adaptive = np.asarray([row[session]["ewma_fp"] for row in rows])
        adaptation[session] = {
            "static_fp_median": float(np.median(static)),
            "ewma_fp_median": float(np.median(adaptive)),
            "subjects_improved": int(np.sum(np.abs(adaptive - 2) < np.abs(static - 2))),
            "paired_wilcoxon_p": float(wilcoxon(np.abs(adaptive - 2), np.abs(static - 2)).pvalue),
        }

    payload = {
        "subjects": list(SUBJECTS),
        "groups": {"MBSR": sorted(MBSR), "control": sorted(set(SUBJECTS) - MBSR)},
        "protocol": {
            "comparison": "paired published Session 1 vs Session 2",
            "hyperparameters": "fixed from Session 1; no Session-2 tuning",
            "ewma_scope": "observed feedback exposure, not continuous wall clock",
            "group_interaction": "exploratory n=5 per group",
        },
        "paired_session_change": summary,
        "within_session_adaptation": adaptation,
        "rows": rows,
    }
    output = ROOT / "data" / "processed" / "stieger2021_longitudinal10_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
