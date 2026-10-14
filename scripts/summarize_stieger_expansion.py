"""Summarize the locked Session-1 Stieger expansion with MBSR stratification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from summarize_stieger_smoke import bootstrap_median, summarize_rows


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = tuple(range(1, 21))


def main() -> None:
    rows = []
    for subject in SUBJECTS:
        cache = ROOT / "data" / "interim" / "stieger2021" / f"S{subject}_Session_1_covariances.npz"
        with np.load(cache, allow_pickle=False) as data:
            audit = json.loads(str(data["audit_json"]))
        group = "MBSR" if int(audit["mbsr_subject"]) else "control"
        report = json.loads((ROOT / "data" / "processed" /
                             f"stieger2021_riemannian_smoke_S{subject}.json").read_text(
                                 encoding="utf-8"))
        rows.append({"subject": subject, "group": group, "report": report})

    overall = summarize_rows(rows, 20_000, 20260806)
    by_group = {
        group: summarize_rows([row for row in rows if row["group"] == group],
                              20_000, 20260806 + index * 100)
        for index, group in enumerate(("control", "MBSR"))
    }
    raw_auc = np.asarray([row["report"]["arms"]["raw"]["window_auc"] for row in rows])
    raw_excess = np.asarray([row["report"]["arms"]["raw"]["event"]["excess_tpr"] for row in rows])
    oracle_excess = np.asarray([
        row["report"]["arms"]["raw"]["oracle_common_fp_event_diagnostic"]["excess_tpr"]
        for row in rows
    ])
    control_auc = raw_auc[[row["group"] == "control" for row in rows]]
    mbsr_auc = raw_auc[[row["group"] == "MBSR" for row in rows]]
    rank = mannwhitneyu(control_auc, mbsr_auc, alternative="two-sided")
    auc_test = wilcoxon(raw_auc - 0.5, alternative="greater")

    payload = {
        "selection": (
            "Subjects 1-20, Session 1, fixed before inspecting subjects 10-20 outcomes; "
            "all available IDs in this contiguous expansion were retained."
        ),
        "groups": {
            "control": [row["subject"] for row in rows if row["group"] == "control"],
            "MBSR": [row["subject"] for row in rows if row["group"] == "MBSR"],
        },
        "overall": overall,
        "by_group": by_group,
        "additional_inference": {
            "raw_window_auc_bootstrap_ci": bootstrap_median(raw_auc, 20260806, 20_000),
            "raw_window_auc_above_chance_wilcoxon_p": float(auc_test.pvalue),
            "control_vs_mbsr_auc_mannwhitney_p": float(rank.pvalue),
            "raw_transferred_excess_positive_subjects": int(np.sum(raw_excess > 0)),
            "raw_oracle_excess_positive_subjects": int(np.sum(oracle_excess > 0)),
        },
        "rows": [{
            "subject": row["subject"],
            "group": row["group"],
            "raw_window_auc": row["report"]["arms"]["raw"]["window_auc"],
            "speed_matched_window_auc": row["report"]["arms"]["raw"]["speed_matched_window_auc"],
            "transferred_fp_per_minute": row["report"]["arms"]["raw"]["event"]["fp_per_minute"],
            "transferred_excess_tpr": row["report"]["arms"]["raw"]["event"]["excess_tpr"],
            "oracle_fp2_excess_tpr": row["report"]["arms"]["raw"][
                "oracle_common_fp_event_diagnostic"]["excess_tpr"],
        } for row in rows],
    }
    output = ROOT / "data" / "processed" / "stieger2021_expansion20_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
