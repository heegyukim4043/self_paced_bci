"""Cross-session persistence of prospective flat-response screening."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import hypergeom,spearmanr

def ci(values,seed):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),(20000,len(values)))]
    return np.quantile(np.median(draws,axis=1),[.025,.975]).tolist()

def metrics(rows,ids,key,seed):
    values=[rows[i]["results"][key] for i in sorted(ids)]
    excess=[v["excess"] for v in values];lat=[v["latency_seconds"] for v in values if v["latency_seconds"] is not None]
    return {"n":len(ids),"excess_median":float(np.median(excess)),"excess_ci95":ci(excess,seed),"actual_fp_median":float(np.median([v["actual_fp_per_min"] for v in values])),"tpr_median":float(np.median([v["tpr"] for v in values])),"latency_median_seconds":float(np.median(lat))}

def main():
    p=argparse.ArgumentParser();p.add_argument("--session1",type=Path,default=Path("data/processed/lee2019_prospective_screening.json"));p.add_argument("--session2",type=Path,default=Path("data/processed/lee2019_prospective_screening_session2.json"));p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_cross_session_screening.json"));a=p.parse_args();s1={r["subject"]:r for r in json.loads(a.session1.read_text(encoding="utf-8"))["rows"]};s2={r["subject"]:r for r in json.loads(a.session2.read_text(encoding="utf-8"))["rows"]};subjects=sorted(set(s1)&set(s2));x=np.asarray([s1[i]["screen_flat_fraction"] for i in subjects]);y=np.asarray([s2[i]["screen_flat_fraction"] for i in subjects]);rho,pvalue=spearmanr(x,y);c1=float(np.quantile(x,.2));c2=float(np.quantile(y,.2));A={i for i in subjects if s1[i]["screen_flat_fraction"]<=c1};B={i for i in subjects if s2[i]["screen_flat_fraction"]<=c2};overlap=A&B;key="dwell0.2";fixed={i for i in subjects if s2[i]["screen_flat_fraction"]<=c1}
    result={"session_flat_rank_spearman":{"rho":float(rho),"p":float(pvalue)},"relative_cutoffs":{"session1":c1,"session2":c2},"selected_counts":{"session1":len(A),"session2":len(B),"overlap":len(overlap),"jaccard":len(overlap)/len(A|B),"overlap_hypergeometric_p":float(hypergeom.sf(len(overlap)-1,len(subjects),len(A),len(B)))},"session2_performance":{"selected_by_session1":metrics(s2,A,key,20261501),"not_selected_by_session1":metrics(s2,set(subjects)-A,key,20261502),"reselected_in_session2":metrics(s2,B,key,20261503),"stable_overlap":metrics(s2,overlap,key,20261504),"session1_absolute_cutoff_reapplied_in_session2":{"cutoff":c1,**metrics(s2,fixed,key,20261505)}}};a.output.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
