"""Locked dwell x seam-guard diagnosis for the sequential Riemannian gate."""

from __future__ import annotations

import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import SFREQ,STEP,WINDOW,calibration,contiguous_intervals,reference,rolling_covariance

TARGETS=(.25,.5,1.,2.,4.,8.,16.); DWELLS=(.1,.3,.5); GUARDS=(2.,3.)

def interpolation(points,key):
    pairs=sorted((v["fp"],v[key]) for v in points.values());x=np.asarray([a for a,b in pairs]);y=np.asarray([b for a,b in pairs]);u=np.unique(x)
    y=np.asarray([np.median(y[x==z]) for z in u]);x=u
    return None if len(x)<2 or 2<x.min() or 2>x.max() else float(np.interp(2,x,y))

def analyze(subject,stream_root,raw_root,cov_root,shifts):
    with np.load(cov_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d:
        ref=reference(d["pre_train_covariance"]);ct=d["train_command_trial"];it=d["train_idle_iti_trial"]
        cmd=tangent_features(d["train_command_covariance"][ct<50],ref);iti=tangent_features(d["train_idle_iti_covariance"][it<50],ref)
    scaler=StandardScaler().fit(np.r_[cmd,iti]);model=LogisticRegression(class_weight="balanced",solver="liblinear",max_iter=2000,random_state=20260811).fit(scaler.transform(np.r_[cmd,iti]),np.r_[np.ones(len(cmd)),np.zeros(len(iti))])
    with np.load(raw_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d: anchor_cov,_=rolling_covariance(d["pre_train"])
    anchor=model.decision_function(scaler.transform(tangent_features(anchor_cov,ref)));mean,var=float(anchor.mean()),float(anchor.var(ddof=1));z=(anchor-mean)/np.sqrt(var)
    operating={d:tune_source_operating_points([calibration(z)],"ma_dwell",TARGETS,STEP,ma_seconds=.5,dwell_seconds=d,refractory_seconds=1.,grid_size=160) for d in DWELLS}
    with np.load(stream_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d: signal,events=d["signal"],d["events"]
    cov,ends=rolling_covariance(signal);raw=model.decision_function(scaler.transform(tangent_features(cov,ref)));times=ends/SFREQ;starts=(ends-int(WINDOW*SFREQ))/SFREQ
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;commands=np.asarray([(r[6]/SFREQ+WINDOW,r[7]/SFREQ) for r in events]);rng=np.random.default_rng(310000+subject);offsets=rng.integers(1,len(times),shifts)
    out={}
    for guard_width in GUARDS:
        guard=np.zeros(len(times),bool)
        for seam in seams:guard|=(starts<seam+guard_width)&(times>seam)
        strict=np.zeros(len(times),bool)
        for row in events:strict|=(starts>=row[6]/SFREQ)&(times<=row[7]/SFREQ)
        idle=~strict&~guard;idle_intervals=contiguous_intervals(idle,times)
        for dwell in DWELLS:
            points={}
            for target,op in operating[dwell].items():
                decision=ewma_normalize_and_decide(raw,initial_mean=mean,initial_variance=var,threshold=op.threshold,step_seconds=STEP,half_life_seconds=30.,update_mode="all",update_mask=~guard,ma_seconds=.5,dwell_seconds=dwell,refractory_seconds=1.,sigma_floor_fraction=.25)
                obs=evaluate_alarms(times,decision.alarms,idle_intervals,commands)
                null=[evaluate_alarms(times,np.roll(decision.alarms,int(o))&~guard,idle_intervals,commands).true_positive_rate for o in offsets]
                points[str(target)]={"threshold":float(op.threshold),"fp":obs.fp_per_minute,"tpr":obs.true_positive_rate,"null":float(np.median(null)),"excess":float(obs.true_positive_rate-np.median(null)),"detected_events":obs.detected_events}
            out[f"guard{int(guard_width)}_dwell{dwell:.1f}"]={"points":points,**{f"{k}_at_fp2":interpolation(points,k) for k in ("threshold","tpr","null","excess","detected_events")}}
    return {"subject":subject,"conditions":out}

def boot(v,seed):
    v=np.asarray(v);rng=np.random.default_rng(seed);d=v[rng.integers(0,len(v),(10000,len(v)))];return np.quantile(np.median(d,axis=1),[.025,.975]).tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument("--stream-root",type=Path,default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"));p.add_argument("--raw-root",type=Path,default=Path("data/interim/lee2019_four_rest_raw/session1"));p.add_argument("--cov-root",type=Path,default=Path("data/interim/lee2019_four_rest/session1"));p.add_argument("--subjects",type=int,nargs="*");p.add_argument("--shifts",type=int,default=100);p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_dwell_guard_grid.json"));a=p.parse_args();subjects=a.subjects or list(range(1,55));rows=[]
    for i,s in enumerate(subjects,1):print(f"dwell-grid {i}/{len(subjects)} subject={s}",flush=True);rows.append(analyze(s,a.stream_root,a.raw_root,a.cov_root,a.shifts))
    summary={};index=0
    for g in GUARDS:
        for d in DWELLS:
            name=f"guard{int(g)}_dwell{d:.1f}";vals=[r["conditions"][name]["excess_at_fp2"] for r in rows];valid=np.asarray([v for v in vals if v is not None]);tpr=[r["conditions"][name]["tpr_at_fp2"] for r in rows if r["conditions"][name]["tpr_at_fp2"] is not None];null=[r["conditions"][name]["null_at_fp2"] for r in rows if r["conditions"][name]["null_at_fp2"] is not None]
            summary[name]={"bracketed":len(valid),"excess_median":float(np.median(valid)),"excess_ci95":boot(valid,20260900+index),"tpr_median":float(np.median(tpr)),"null_median":float(np.median(null))};index+=1
    payload={"binding_rule":"sustained MI recovery only if guard3 excess CI lower bound > 0; guard2-only gain is transition-assisted","summary":summary,"rows":rows};a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
