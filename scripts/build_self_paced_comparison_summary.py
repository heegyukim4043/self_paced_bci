"""Build machine-readable harmonized rows for the self-paced comparison table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def median(values):
    return float(np.median(np.asarray(values, float)))


def main():
    dep = json.loads((DATA / "lee2019_loso_deployment.json").read_text())
    rel = json.loads((DATA / "lee2019_auc_event_relation.json").read_text())
    null = {(r["session"], r["subject"]): r["circular_shift_null_2s"] for r in rel["rows"]}
    selected = {
        (session, subject)
        for session in (1, 2)
        for subject in dep["summary"][str(session)]["A_target_rest"]["selected_subjects"]
    }

    def arm_a(rows, name):
        fixed = [r["arms"]["A_target_rest"]["surface"]["4.0"]["fixed_threshold"] for r in rows]
        observed = [x["2.0"] for x in fixed]
        chance = [null[(r["session"], r["subject"])] for r in rows]
        return {
            "model": name, "n": len(rows),
            "window_auc": median([r["arms"]["A_target_rest"]["window_auc"] for r in rows]),
            "actual_fp_per_min": median([x["actual_fp"] for x in fixed]),
            "sensitivity_2s": median(observed), "circular_null_2s": median(chance),
            "excess_2s_matched_median": median(np.asarray(observed)-np.asarray(chance)),
            "coverage": median([x["coverage"] for x in fixed]),
            "temporal_iou": median([x["temporal_iou"] for x in fixed]),
            "strict_f1": median([x["strict_f1"] for x in fixed]),
        }

    rows = dep["rows"]
    output = [
        arm_a(rows, "Riemannian Arm A, all"),
        arm_a([r for r in rows if (r["session"],r["subject"]) in selected], "Riemannian Arm A, selected"),
        arm_a([r for r in rows if (r["session"],r["subject"]) not in selected], "Riemannian Arm A, unselected"),
    ]
    payload = {
        "protocol": "nominal FP4 fixed threshold; 2-s cue-relative pseudo-onset; medians across subject-sessions",
        "screening": {"selected_n": len(selected), "total_n": len(rows), "coverage": len(selected)/len(rows)},
        "rows": output,
    }
    path = DATA / "self_paced_comparison_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
