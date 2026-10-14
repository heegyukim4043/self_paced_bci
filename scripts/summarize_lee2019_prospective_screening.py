"""Summarize pre-specified relative-20% screening sensitivity."""
import argparse,json
from pathlib import Path
import numpy as np

def ci(values,seed):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),(20000,len(values))) ]
    return np.quantile(np.median(draws,axis=1),[.025,.975]).tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,default=Path("data/processed/lee2019_prospective_screening.json"));p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_prospective_screening_relative.json"));a=p.parse_args();payload=json.loads(a.input.read_text(encoding="utf-8"));rows=payload["rows"]
    cutoff=float(np.quantile([row["screen_flat_fraction"] for row in rows],.2));summary={};index=0
    for dwell in (.1,.2,.3):
        key=f"dwell{dwell:.1f}"
        for selected in (True,False):
            subset=[row for row in rows if (row["screen_flat_fraction"]<=cutoff)==selected]
            get=lambda name:[row["results"][key][name] for row in subset if row["results"][key][name] is not None]
            excess=get("excess");fp=get("actual_fp_per_min");tpr=get("tpr");lat=get("latency_seconds")
            summary[f"{'selected' if selected else 'not_selected'}_{key}"]={"n":len(subset),"excess_median":float(np.median(excess)),"excess_ci95":ci(excess,20261300+index),"actual_fp_median":float(np.median(fp)),"tpr_median":float(np.median(tpr)),"latency_median_seconds":float(np.median(lat))};index+=1
    selected=[row for row in rows if row["screen_flat_fraction"]<=cutoff];delta={}
    for metric in ("excess","latency_seconds","actual_fp_per_min"):
        values=np.asarray([row["results"]["dwell0.2"][metric]-row["results"]["dwell0.3"][metric] for row in selected if row["results"]["dwell0.2"][metric] is not None and row["results"]["dwell0.3"][metric] is not None])
        delta[metric]={"median":float(np.median(values)),"ci95":ci(values,20261400+len(delta))}
    result={"relative_rule":"calibration flat fraction <= cohort 20th percentile; ties included","cutoff":cutoff,"selected_fraction":len(selected)/len(rows),"summary":summary,"selected_dwell0.2_minus0.3":delta};a.output.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
