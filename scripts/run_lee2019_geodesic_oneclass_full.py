"""Full paired validation of the target-pre-rest-only geodesic gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_lee2019_geodesic_oneclass import analyze


def cluster_ci(pairs, repeats=10_000, seed=20260814):
    rng = np.random.default_rng(seed)
    by_subject = {u: np.asarray([v for subject, v in pairs if subject == u]) for u in range(1, 55)}
    values = []
    for _ in range(repeats):
        draw = rng.integers(1, 55, 54)
        values.append(float(np.median(np.concatenate([by_subject[int(u)] for u in draw]))))
    return np.quantile(values, [.025, .975]).tolist()


def main():
    rows = []
    for session in (1, 2):
        for subject in range(1, 55):
            rows.append(analyze(subject, session))
            print(f"done session{session} sub-{subject:02d} ({len(rows)}/108)", flush=True)
    arm_a = json.loads(Path("data/processed/lee2019_auc_event_relation.json").read_text(encoding="utf-8"))
    comparator = {(r["session"], r["subject"]): r for r in arm_a["rows"]}
    excess_delta=[]; auc_delta=[]; fp_delta=[]
    for row in rows:
        key=(row["session"],row["subject"]); base=comparator[key]
        excess_delta.append((row["subject"],row["fixed_fp4_excess_2s"]-base["excess_sensitivity_2s"]))
        auc_delta.append((row["subject"],row["window_auc"]-base["window_auc"]))
        # Arm-A actual FP is loaded from the deployment result below.
    deployment=json.loads(Path("data/processed/lee2019_loso_deployment.json").read_text(encoding="utf-8"))
    dep={(r["session"],r["subject"]):r for r in deployment["rows"]}
    for row in rows:
        base=dep[(row["session"],row["subject"])]["arms"]["A_target_rest"]["surface"]["4.0"]["fixed_threshold"]
        geo=row["surface"]["4.0"]["fixed_threshold"]
        fp_delta.append((row["subject"],geo["actual_fp"]-base["actual_fp"]))
    summary={
        "n_subject_sessions":len(rows),
        "geodesic":{
            "window_auc_median":float(np.median([r["window_auc"] for r in rows])),
            "auc_below_0.5_n":int(sum(r["window_auc"]<.5 for r in rows)),
            "actual_fp_median":float(np.median([r["surface"]["4.0"]["fixed_threshold"]["actual_fp"] for r in rows])),
            "sensitivity_2s_median":float(np.median([r["surface"]["4.0"]["fixed_threshold"]["2.0"] for r in rows])),
            "excess_2s_median":float(np.median([r["fixed_fp4_excess_2s"] for r in rows])),
            "coverage_median":float(np.median([r["surface"]["4.0"]["fixed_threshold"]["coverage"] for r in rows])),
            "flat_median":float(np.median([r["screen_flat_fraction"] for r in rows])),
        },
        "paired_vs_arm_a":{
            "excess_delta_median":float(np.median([v for _,v in excess_delta])),
            "excess_delta_cluster_ci":cluster_ci(excess_delta),
            "auc_delta_median":float(np.median([v for _,v in auc_delta])),
            "auc_delta_cluster_ci":cluster_ci(auc_delta,seed=20260815),
            "fp_delta_median":float(np.median([v for _,v in fp_delta])),
            "fp_delta_cluster_ci":cluster_ci(fp_delta,seed=20260816),
        },
    }
    payload={"protocol":{"labels":"target pre-rest only; no command labels","comparator":"LOSO Arm A fixed nominal FP4; same locked last-40 event definition","inference":"paired subject-cluster bootstrap"},"summary":summary,"rows":rows}
    output=Path("data/processed/lee2019_geodesic_oneclass_full.json")
    output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
