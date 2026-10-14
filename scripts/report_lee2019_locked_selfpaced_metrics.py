from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

P=Path("data/processed/lee2019_locked_selfpaced_metrics.json")
OUT=Path("data/processed/lee2019_locked_selfpaced_metrics")
ARMS=("baseline","shrinkage","multi_idle","soft_veto_0.5")
LABELS=("TS-LR","Shrinkage TS-LR","Multi-idle","Soft veto")
DELTAS=(.5,1.,2.,3.,4.)

def get(rows,arm,path):
    out=[]
    for r in rows:
        x=r["arms"][arm]
        for k in path:x=x[k]
        out.append(np.nan if x is None else x)
    return np.asarray(out,float)

def cluster_ci(rows,arm,path,base="baseline",seed=20260812):
    a=get(rows,arm,path);b=get(rows,base,path);subjects=np.asarray([r["subject"] for r in rows]);unique=np.unique(subjects);rng=np.random.default_rng(seed);v=[]
    for _ in range(20000):
        pick=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(subjects==s) for s in pick]);v.append(np.nanmedian(a[idx]-b[idx]))
    return [float(np.nanmedian(a-b)),*np.quantile(v,[.025,.975]).tolist()]

def main():
    d=json.loads(P.read_text(encoding="utf-8"));rows=d["rows"];OUT.mkdir(parents=True,exist_ok=True)
    report={"n_subject_sessions":len(rows),"paired_delta_vs_baseline":{}}
    for arm in ARMS[1:]:
        report["paired_delta_vs_baseline"][arm]={
            "sensitivity_at_fp2":cluster_ci(rows,arm,["sensitivity_at_fp2"]),
            "partial_auc_fp0_4":cluster_ci(rows,arm,["partial_auc_fp0_4"]),
            "coverage":cluster_ci(rows,arm,["interval_0.1","coverage"]),
        }
    report["medians"]={}
    for arm in ARMS:
        report["medians"][arm]={
            "latency_sensitivity":{str(x):float(np.median(get(rows,arm,["onset",str(x),"sensitivity"]))) for x in DELTAS},
            "sensitivity_at_fp2":float(np.median(get(rows,arm,["sensitivity_at_fp2"]))),
            "pauc":float(np.median(get(rows,arm,["partial_auc_fp0_4"]))),
            "coverage":float(np.median(get(rows,arm,["interval_0.1","coverage"]))),
            "iou":float(np.median(get(rows,arm,["interval_0.1","temporal_iou"]))),
            "f1_tolerant":float(np.median(get(rows,arm,["interval_0.1","intersection_f1"]))),
            "f1_strict":float(np.median(get(rows,arm,["interval_0.7","intersection_f1"]))),
        }
    (OUT/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for arm,label in zip(ARMS,LABELS):ax[0,0].plot(DELTAS,[report["medians"][arm]["latency_sensitivity"][str(x)] for x in DELTAS],marker="o",label=label)
    ax[0,0].set(xlabel="Allowed onset latency (s)",ylabel="Median sensitivity",title="Cue-relative pseudo-onset");ax[0,0].legend();ax[0,0].grid(alpha=.2)
    x=np.arange(4);w=.35
    ax[0,1].bar(x-w/2,[report["medians"][a]["sensitivity_at_fp2"] for a in ARMS],w,label="Sensitivity @ FP=2")
    ax[0,1].bar(x+w/2,[report["medians"][a]["pauc"] for a in ARMS],w,label="pAUC FP 0-4")
    ax[0,1].set_xticks(x,LABELS,rotation=15);ax[0,1].set_title("Threshold-sweep onset metrics");ax[0,1].legend();ax[0,1].grid(axis="y",alpha=.2)
    ax[1,0].bar(x-w/2,[report["medians"][a]["coverage"] for a in ARMS],w,label="Coverage")
    ax[1,0].bar(x+w/2,[report["medians"][a]["iou"] for a in ARMS],w,label="Temporal IoU")
    ax[1,0].set_xticks(x,LABELS,rotation=15);ax[1,0].set_title("Causal interval metrics");ax[1,0].legend();ax[1,0].grid(axis="y",alpha=.2)
    ax[1,1].bar(x-w/2,[report["medians"][a]["f1_tolerant"] for a in ARMS],w,label="DTC/GTC 0.1")
    ax[1,1].bar(x+w/2,[report["medians"][a]["f1_strict"] for a in ARMS],w,label="DTC/GTC 0.7")
    ax[1,1].set_xticks(x,LABELS,rotation=15);ax[1,1].set_title("Intersection-based interval F1");ax[1,1].legend();ax[1,1].grid(axis="y",alpha=.2)
    fig.suptitle("Lee2019 locked self-paced evaluation: 54 subjects × 2 sessions",fontweight="bold")
    fig.savefig(OUT/"locked_metrics_comparison.png",dpi=180)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
