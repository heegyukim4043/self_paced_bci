"""Operational top-20% screening summary for the geodesic one-class gate."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def main():
    path=Path("data/processed/lee2019_geodesic_oneclass_full.json")
    payload=json.loads(path.read_text(encoding="utf-8"));selected=[];sessions={}
    arm_a=json.loads(Path("data/processed/lee2019_loso_deployment.json").read_text(encoding="utf-8"))
    for session in (1,2):
        rows=[r for r in payload["rows"] if r["session"]==session]
        ranked=sorted(rows,key=lambda r:(r["screen_flat_fraction"],-r["screen_margin_mean"]))
        chosen=ranked[:11];selected+=chosen
        ids={r["subject"] for r in chosen};aids=set(arm_a["summary"][str(session)]["A_target_rest"]["selected_subjects"])
        sessions[str(session)]={"selected_subjects":sorted(ids),"jaccard_with_arm_a":len(ids&aids)/len(ids|aids),"overlap_with_arm_a":len(ids&aids)}
    fixed=[r["surface"]["4.0"]["fixed_threshold"] for r in selected]
    sessions["combined"]={
        "n":len(selected),
        "window_auc_median":float(np.median([r["window_auc"] for r in selected])),
        "actual_fp_median":float(np.median([r["actual_fp"] for r in fixed])),
        "sensitivity_2s_median":float(np.median([r["2.0"] for r in fixed])),
        "circular_null_2s_median":float(np.median([r["fixed_fp4_circular_null_2s"] for r in selected])),
        "excess_2s_median":float(np.median([r["fixed_fp4_excess_2s"] for r in selected])),
        "coverage_median":float(np.median([r["coverage"] for r in fixed])),
        "flat_median":float(np.median([r["screen_flat_fraction"] for r in selected])),
    }
    payload["screening_summary"]={"rule":"per-session lowest flat fraction, margin tie-break, exactly 11/54",**sessions}
    path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload["screening_summary"],indent=2))


if __name__=="__main__":main()
