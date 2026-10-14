"""Fixed random pilot for the target-pre-rest geodesic one-class gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_lee2019_geodesic_oneclass import analyze


def main():
    rng = np.random.default_rng(20260814)
    population = [(session, subject) for session in (1, 2) for subject in range(1, 55)]
    chosen = [population[index] for index in rng.choice(len(population), 10, replace=False)]
    rows = []
    for session, subject in chosen:
        rows.append(analyze(subject, session)); print(f"done session{session} sub-{subject:02d}")
    auc = np.asarray([row["window_auc"] for row in rows])
    fixed = [row["surface"]["4.0"]["fixed_threshold"] for row in rows]
    summary = {
        "n": len(rows),
        "window_auc_median": float(np.median(auc)),
        "window_auc_range": [float(auc.min()), float(auc.max())],
        "auc_below_0.5_n": int(np.sum(auc < .5)),
        "auc_above_0.5_n": int(np.sum(auc > .5)),
        "direction_oracle_auc_median_diagnostic": float(np.median(np.maximum(auc, 1 - auc))),
        "fixed_actual_fp_median": float(np.median([row["actual_fp"] for row in fixed])),
        "fixed_sensitivity_2s_median": float(np.median([row["2.0"] for row in fixed])),
        "fixed_circular_null_2s_median": float(np.median([row["fixed_fp4_circular_null_2s"] for row in rows])),
        "fixed_excess_2s_median": float(np.median([row["fixed_fp4_excess_2s"] for row in rows])),
        "fixed_coverage_median": float(np.median([row["coverage"] for row in fixed])),
        "screen_flat_fraction_median": float(np.median([row["screen_flat_fraction"] for row in rows])),
    }
    payload = {
        "protocol": {
            "selection": "same fixed random 10 subject-sessions as Parallel Transport pilot; seed 20260814",
            "direction": "higher geodesic distance is anomaly/command; no label-based sign flip",
            "direction_oracle": "absolute direction AUC is diagnostic only and non-deployable",
        },
        "chosen": [{"session": s, "subject": u} for s, u in chosen],
        "summary": summary, "rows": rows,
    }
    output = Path("data/processed/lee2019_geodesic_oneclass_pilot.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
