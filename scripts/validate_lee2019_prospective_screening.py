"""Chronological, source-anchor-only screening validation on held-out stream."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ,STEP,WINDOW,calibration,contiguous_intervals,reference,rolling_covariance)

TARGETS=(.25,.5,1.,2.,4.,8.,16.)
DWELLS=(.1,.2,.3)
SCREEN_TRIALS=20
FLAT_CUTOFF=.25
GUARD_WIDTH=3.
COMMAND_END=4.

def latency(times,alarms,cues,commands):
    values=[]
    for cue,(begin,end) in zip(cues,commands):
        hit=times[alarms&(times>=begin)&(times<end)]
        if len(hit):values.append(float(hit[0]-cue))
    return float(np.median(values)) if values else None

def first_false_alarm(times,alarms,idle):
    """First FP in cumulative valid-idle exposure and wall-clock time."""
    exposure=np.cumsum(idle.astype(float))*STEP
    hits=np.flatnonzero(alarms&idle)
    if len(hits):
        index=int(hits[0]);return True,float(exposure[index]),float(times[index]-times[0])
    return False,float(exposure[-1] if len(exposure) else 0.),float(times[-1]-times[0] if len(times) else 0.)

def analyze(subject,stream_root,raw_root,cov_root,shifts,screen_trials,flat_cutoff):
    with np.load(cov_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d:
        ref=reference(d["pre_train_covariance"]);ct=d["train_command_trial"];it=d["train_idle_iti_trial"]
        command=tangent_features(d["train_command_covariance"][ct<50],ref)
        iti=tangent_features(d["train_idle_iti_covariance"][it<50],ref)
    scaler=StandardScaler().fit(np.r_[command,iti])
    model=LogisticRegression(class_weight="balanced",solver="liblinear",max_iter=2000,
        random_state=20260811).fit(scaler.transform(np.r_[command,iti]),np.r_[np.ones(len(command)),np.zeros(len(iti))])
    with np.load(raw_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d:anchor_cov,_=rolling_covariance(d["pre_train"])
    anchor=model.decision_function(scaler.transform(tangent_features(anchor_cov,ref)))
    mean,var=float(anchor.mean()),float(anchor.var(ddof=1));z=(anchor-mean)/np.sqrt(var)
    operating={}
    for dwell in DWELLS:
        points=tune_source_operating_points([calibration(z)],"ma_dwell",TARGETS,STEP,
            ma_seconds=.5,dwell_seconds=dwell,refractory_seconds=1.,grid_size=160)
        target=min(points,key=lambda value:abs(float(value)-2.))
        operating[dwell]=points[target]

    with np.load(stream_root/f"sub-{subject:02d}.npz",allow_pickle=False) as d:signal,events=d["signal"],d["events"]
    covariance,ends=rolling_covariance(signal);raw=model.decision_function(scaler.transform(tangent_features(covariance,ref)))
    times=ends/SFREQ;starts=(ends-int(WINDOW*SFREQ))/SFREQ;cues=events[:,5]/SFREQ
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;guard=np.zeros(len(times),bool)
    for seam in seams:guard|=(starts<seam+GUARD_WIDTH)&(times>seam)

    decisions={}
    for dwell in DWELLS:
        decisions[dwell]=ewma_normalize_and_decide(raw,initial_mean=mean,initial_variance=var,
            threshold=operating[dwell].threshold,step_seconds=STEP,half_life_seconds=30.,
            update_mode="all",update_mask=~guard,ma_seconds=.5,dwell_seconds=dwell,
            refractory_seconds=1.,sigma_floor_fraction=.25)

    # Screening uses the locked primary dwell=0.3 source-anchor threshold, but
    # flatness itself is a raw normalized-score criterion and does not inspect
    # alarms or future labels.
    primary=decisions[.3];threshold=float(operating[.3].threshold);flat=[]
    for cue in cues[:screen_trials]:
        valid=(times>=cue+1.5)&(times<cue+COMMAND_END)&~guard
        flat.append(not np.any(primary.statistic[valid]>=threshold))
    flat_fraction=float(np.mean(flat));selected=flat_fraction<=flat_cutoff

    eval_begin=events[screen_trials,2]/SFREQ
    eval_mask=times>=eval_begin;eval_times=times[eval_mask];eval_guard=guard[eval_mask]
    eval_starts=starts[eval_mask];eval_cues=cues[screen_trials:]
    positive=np.zeros(len(eval_times),bool)
    for cue in eval_cues:positive|=(eval_starts>=cue+.5)&(eval_times<=cue+COMMAND_END)
    idle=~positive&~eval_guard;idle_intervals=contiguous_intervals(idle,eval_times)
    commands=np.asarray([(cue+1.5,cue+COMMAND_END) for cue in eval_cues])
    rng=np.random.default_rng(640000+subject);offsets=rng.integers(1,len(eval_times),shifts)
    results={}
    for dwell in DWELLS:
        alarms=decisions[dwell].alarms[eval_mask]
        observed=evaluate_alarms(eval_times,alarms,idle_intervals,commands)
        null=[evaluate_alarms(eval_times,np.roll(alarms,int(offset))&~eval_guard,
            idle_intervals,commands).true_positive_rate for offset in offsets]
        fp_observed,fp_exposure,fp_wallclock=first_false_alarm(eval_times,alarms,idle)
        results[f"dwell{dwell:.1f}"]={"source_threshold":float(operating[dwell].threshold),
            "actual_fp_per_min":observed.fp_per_minute,"tpr":observed.true_positive_rate,
            "null_tpr":float(np.median(null)),"excess":float(observed.true_positive_rate-np.median(null)),
            "latency_seconds":latency(eval_times,alarms,eval_cues,commands),
            "detected_trials":int(observed.detected_events),
            "first_fp_observed":fp_observed,"first_fp_idle_exposure_seconds":fp_exposure,
            "first_fp_wallclock_seconds":fp_wallclock}
    return {"subject":subject,"screen_flat_fraction":flat_fraction,"selected":selected,
        "screen_flat_trials":int(sum(flat)),"screen_wallclock_seconds":float(eval_begin),
        "evaluation_trials":len(eval_cues),"results":results}

def boot(values,seed):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),(20000,len(values)))]
    return np.quantile(np.median(draws,axis=1),[.025,.975]).tolist()

def summarize(rows,selected,dwell,index):
    subset=[r for r in rows if r["selected"]==selected];key=f"dwell{dwell:.1f}"
    def values(name):return np.asarray([r["results"][key][name] for r in subset if r["results"][key][name] is not None])
    excess=values("excess");fp=values("actual_fp_per_min");tpr=values("tpr");lat=values("latency_seconds")
    return {"n":len(subset),"excess_median":float(np.median(excess)),"excess_ci95":boot(excess,20261200+index),
        "actual_fp_median":float(np.median(fp)),"actual_fp_iqr":np.quantile(fp,[.25,.75]).tolist(),
        "tpr_median":float(np.median(tpr)),"latency_median_seconds":float(np.median(lat)) if len(lat) else None}

def main():
    p=argparse.ArgumentParser();p.add_argument("--stream-root",type=Path,default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"));p.add_argument("--raw-root",type=Path,default=Path("data/interim/lee2019_four_rest_raw/session1"));p.add_argument("--cov-root",type=Path,default=Path("data/interim/lee2019_four_rest/session1"));p.add_argument("--subjects",type=int,nargs="*");p.add_argument("--shifts",type=int,default=500);p.add_argument("--screen-trials",type=int,default=SCREEN_TRIALS);p.add_argument("--flat-cutoff",type=float,default=FLAT_CUTOFF);p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_prospective_screening.json"));a=p.parse_args();subjects=a.subjects or list(range(1,55));rows=[]
    for i,s in enumerate(subjects,1):print(f"prospective-screen {i}/{len(subjects)} subject={s}",flush=True);rows.append(analyze(s,a.stream_root,a.raw_root,a.cov_root,a.shifts,a.screen_trials,a.flat_cutoff))
    summary={};index=0
    for dwell in DWELLS:
        for selected in (True,False):
            summary[f"{'selected' if selected else 'not_selected'}_dwell{dwell:.1f}"]=summarize(rows,selected,dwell,index);index+=1
    payload={"locked_rule":{"screen_trials":a.screen_trials,"flat_cutoff":a.flat_cutoff,
        "threshold":"source-anchor target FP=2 only","evaluation":f"chronological trials {a.screen_trials}--49 untouched",
        "primary_dwell":.3},"summary":summary,"rows":rows}
    a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
