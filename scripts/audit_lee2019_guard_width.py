"""Sensitivity of sequential excess detection to rest-trial seam guard width."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, contiguous_intervals, reference, rolling_covariance,
)

TARGETS = (.25, .5, 1., 2., 4., 8., 16.)
GUARDS = (2., 3., 4.)


def interp(points, key):
    pairs = sorted((v["fp"], v[key]) for v in points.values())
    x=np.asarray([a for a,b in pairs]); y=np.asarray([b for a,b in pairs]); u=np.unique(x)
    y=np.asarray([np.median(y[x==z]) for z in u]); x=u
    return None if len(x)<2 or 2<x.min() or 2>x.max() else float(np.interp(2,x,y))


def analyze(subject, stream_root, raw_root, cov_root, shifts):
    with np.load(cov_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d:
        ref=reference(d["pre_train_covariance"]); ct=d["train_command_trial"]; it=d["train_idle_iti_trial"]
        command=tangent_features(d["train_command_covariance"][ct<50],ref)
        iti=tangent_features(d["train_idle_iti_covariance"][it<50],ref)
    scaler=StandardScaler().fit(np.r_[command,iti])
    model=LogisticRegression(class_weight="balanced",solver="liblinear",max_iter=2000,
                             random_state=20260811).fit(
        scaler.transform(np.r_[command,iti]),np.r_[np.ones(len(command)),np.zeros(len(iti))])
    with np.load(raw_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d:
        cov,_=rolling_covariance(d["pre_train"])
    anchor=model.decision_function(scaler.transform(tangent_features(cov,ref)))
    mean,var=float(anchor.mean()),float(anchor.var(ddof=1)); z=(anchor-mean)/np.sqrt(var)
    ops=tune_source_operating_points([calibration(z)],"ma_dwell",TARGETS,STEP,
        ma_seconds=.5,dwell_seconds=.3,refractory_seconds=1.,grid_size=160)
    with np.load(stream_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d:
        signal,events=d["signal"],d["events"]
    cov,ends=rolling_covariance(signal); raw=model.decision_function(scaler.transform(tangent_features(cov,ref)))
    times=ends/SFREQ; starts=(ends-int(WINDOW*SFREQ))/SFREQ
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ
    # Alarm times are window END times.  A complete 1-s window inside the
    # nominal cue+0.5..3.5 support can only end in cue+1.5..3.5.
    command_intervals=np.asarray([(r[6]/SFREQ+WINDOW,r[7]/SFREQ) for r in events])
    rng=np.random.default_rng(220000+subject); offsets=rng.integers(1,len(times),shifts)
    output={}
    for width in GUARDS:
        guard=np.zeros(len(times),bool)
        # Causal seam contamination exists at/after a seam, never before it.
        # A symmetric guard around the following trial->rest seam deletes the
        # preceding MI command interval (100% at width=3 s).  Exclude windows
        # crossing the seam or containing the post-seam warm-up only.
        for seam in seams: guard|=(starts<seam+width)&(times>seam)
        strict=np.zeros(len(times),bool)
        for row in events: strict|=(starts>=row[6]/SFREQ)&(times<=row[7]/SFREQ)
        idle=~strict&~guard; idle_intervals=contiguous_intervals(idle,times)
        points={}
        for target,op in ops.items():
            decision=ewma_normalize_and_decide(raw,initial_mean=mean,initial_variance=var,
                threshold=op.threshold,step_seconds=STEP,half_life_seconds=30.,update_mode="all",
                update_mask=~guard,ma_seconds=.5,dwell_seconds=.3,refractory_seconds=1.,
                sigma_floor_fraction=.25)
            observed=evaluate_alarms(times,decision.alarms,idle_intervals,command_intervals)
            null=[evaluate_alarms(times,np.roll(decision.alarms,int(o)) & ~guard,idle_intervals,
                                  command_intervals).true_positive_rate for o in offsets]
            command_alarm_mask=np.zeros(len(times),bool)
            for a,b in command_intervals: command_alarm_mask|=(times>=a)&(times<b)
            normalized=decision.statistic
            ma_steps=int(np.ceil(.5/STEP)); active=np.zeros(len(times),bool)
            history=[]
            for idx,value in enumerate(normalized):
                if guard[idx]: history=[]; continue
                history.append(float(value))
                if len(history)>ma_steps: history.pop(0)
                active[idx]=len(history)==ma_steps and np.mean(history)>=op.threshold
            run_lengths=[]
            for a,b in command_intervals:
                member=active[(times>=a)&(times<b)]; padded=np.r_[False,member,False]
                changes=np.diff(padded.astype(int)); starts_run=np.flatnonzero(changes==1)
                stops_run=np.flatnonzero(changes==-1)
                run_lengths.extend((stops_run-starts_run).tolist())
            points[str(target)]={"threshold":float(op.threshold),"fp":observed.fp_per_minute,
                                 "tpr":observed.true_positive_rate,"null":float(np.median(null)),
                                 "excess":float(observed.true_positive_rate-np.median(null)),
                                 "detected_events":observed.detected_events,
                                 "raw_command_alarm_count":int(np.sum(decision.alarms&command_alarm_mask)),
                                 "normalized_above_threshold_command_samples":int(np.sum(
                                     (normalized>=op.threshold)&command_alarm_mask)),
                                 "ma_active_command_samples":int(np.sum(active&command_alarm_mask)),
                                 "maximum_ma_active_run_steps":int(max(run_lengths,default=0)),
                                 "idle_seconds":observed.idle_seconds}
        output[str(int(width))]={"points":points,"excess_at_fp2":interp(points,"excess"),
                                 "tpr_at_fp2":interp(points,"tpr"),
                                 "null_at_fp2":interp(points,"null"),
                                 "threshold_at_fp2":interp(points,"threshold"),
                                 "detected_events_at_fp2":interp(points,"detected_events"),
                                 "raw_command_alarms_at_fp2":interp(points,"raw_command_alarm_count"),
                                 "idle_seconds_at_fp2":interp(points,"idle_seconds"),
                                 "guarded_fraction":float(guard.mean())}
    return {"subject":subject,"guards":output}


def bootstrap(values,seed):
    values=np.asarray(values); rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),(10000,len(values)))]
    return np.quantile(np.median(draws,axis=1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stream-root",type=Path,default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"))
    p.add_argument("--raw-root",type=Path,default=Path("data/interim/lee2019_four_rest_raw/session1"))
    p.add_argument("--cov-root",type=Path,default=Path("data/interim/lee2019_four_rest/session1"))
    p.add_argument("--subjects",type=int,nargs="*"); p.add_argument("--shifts",type=int,default=100)
    p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_guard_width_audit.json"))
    a=p.parse_args(); subjects=a.subjects or list(range(1,55)); rows=[]
    for i,s in enumerate(subjects,1):
        print(f"guard audit {i}/{len(subjects)} subject={s}",flush=True)
        rows.append(analyze(s,a.stream_root,a.raw_root,a.cov_root,a.shifts))
    summary={}
    for j,width in enumerate(GUARDS):
        selected=[r["guards"][str(int(width))] for r in rows]
        vals=[v["excess_at_fp2"] for v in selected]
        valid=np.asarray([v for v in vals if v is not None])
        def med(key):
            x=[v[key] for v in selected if v[key] is not None]
            return float(np.median(x))
        summary[str(int(width))]={"bracketed":len(valid),"excess_median":float(np.median(valid)),
                                  "excess_ci95":bootstrap(valid,20260830+j),
                                  "threshold_at_fp2_median":med("threshold_at_fp2"),
                                  "idle_seconds_median":med("idle_seconds_at_fp2"),
                                  "tpr_at_fp2_median":med("tpr_at_fp2"),
                                  "null_at_fp2_median":med("null_at_fp2"),
                                  "detected_events_at_fp2_median":med("detected_events_at_fp2"),
                                  "raw_command_alarms_at_fp2_median":med("raw_command_alarms_at_fp2")}
    a.output.write_text(json.dumps({"summary":summary,"rows":rows},indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
