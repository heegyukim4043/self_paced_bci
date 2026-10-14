"""Locked onset/interval evaluation for Lee2019 pseudo-self-paced gates."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import SFREQ,STEP,WINDOW,calibration,reference,rolling_covariance
from uncued_mi.operating_point import tune_source_operating_points

def shrink(cov: np.ndarray, amount: float = .1) -> np.ndarray:
    """Covariance shrinkage, identical to the former OpenViBE helper."""
    p = cov.shape[-1]
    scale = np.trace(cov, axis1=-2, axis2=-1) / p
    eye = np.eye(p, dtype=cov.dtype)
    return (1 - amount) * cov + amount * scale[..., None, None] * eye

DELTAS=(.5,1.,2.,3.,4.); OFFSETS=np.linspace(-3.,3.,41)
ARMS=("baseline","shrinkage","multi_idle","soft_veto_0.5")

def diag_fit(x): return x.mean(0),np.maximum(x.var(0,ddof=1),.05)
def diag_lp(x,p):
    m,v=p;return -.5*np.sum(np.log(2*np.pi*v)+(x-m)**2/v,axis=1)
def zfix(x,a): return (x-a.mean())/max(a.std(ddof=1),1e-8)

def ma_trace(z,n=5):
    out=np.full(len(z),np.nan);cs=np.r_[0.,np.cumsum(z)]
    out[n-1:]=(cs[n:]-cs[:-n])/n
    return out

def alarms_from_ma(ma,thr,allowed,dwell=2,refractory=10):
    out=np.zeros(len(ma),bool);count=0;blocked=-1
    for i in range(len(ma)):
        if not allowed[i] or not np.isfinite(ma[i]): count=0;continue
        count=count+1 if ma[i]>=thr else 0
        if count==dwell and i>blocked: out[i]=1;blocked=i+refractory
    return out

def ewma_z(raw,mean,var,allowed):
    alpha=1-2**(-STEP/30.);floor=.25*np.sqrt(max(var,1e-8));out=np.empty(len(raw))
    for i,x in enumerate(raw):
        out[i]=(x-mean)/max(np.sqrt(max(var,0)),floor)
        if allowed[i]:
            d=x-mean;mean+=alpha*d;var=(1-alpha)*(var+alpha*d*d)
    return out

def tune(anchor):
    z=zfix(anchor,anchor)
    return float(tune_source_operating_points([calibration(z)],"ma_dwell",[2.],STEP,
        ma_seconds=.5,dwell_seconds=.2,refractory_seconds=1.,grid_size=160)[2.].threshold)

def onset_metrics(times,alarms,onsets,gt_intervals,idle,delta):
    detected=[];lat=[]
    for onset in onsets:
        hit=times[alarms&(times>=onset)&(times<=onset+delta)]
        detected.append(len(hit)>0)
        if len(hit):lat.append(float(hit[0]-onset))
    exposure=idle.sum()*STEP
    fp=float(np.sum(alarms&idle)/(exposure/60)) if exposure else np.nan
    return {"sensitivity":float(np.mean(detected)),"fp_per_min":fp,
            "median_latency":float(np.median(lat)) if lat else None,
            "detected":int(sum(detected)),"events":len(onsets)}

def intervals_from_alarm(times,alarms,ma,thr,allowed,release_drop=.5,hold=3,gap=.3):
    intervals=[];i=0;release=thr-release_drop
    # Segment IDs prevent a short gap merge from bridging a guard/seam.
    segment=np.cumsum(~np.asarray(allowed,bool))
    while i<len(times):
        candidates=np.flatnonzero(alarms[i:])
        if not len(candidates):break
        start=i+int(candidates[0]);j=start+1;below=0
        if not allowed[start]:raise ValueError("alarm inside guard")
        while j<len(times) and allowed[j]:
            below=below+1 if (not np.isfinite(ma[j]) or ma[j]<release) else 0
            if below>=hold:break
            j+=1
        # If a guard is reached first, close at the *last allowed* sample.
        stop=j-hold+1 if below>=hold else j-1
        stop=max(start,min(stop,len(times)-1))
        intervals.append([start,stop])
        i=max(stop+1,j+1)
    merged=[]
    for p in intervals:
        if (merged and segment[p[0]]==segment[merged[-1][1]] and
                times[p[0]]-times[merged[-1][1]]<=gap):merged[-1][1]=p[1]
        else:merged.append(p)
    return np.asarray([[times[a],times[b]] for a,b in merged],float).reshape(-1,2)

def overlap(a,b): return max(0.,min(a[1],b[1])-max(a[0],b[0]))
def interval_metrics(pred,gt,dtc,gtc):
    valid=[]
    for p in pred:
        ov=sum(overlap(p,g) for g in gt);valid.append(ov/max(p[1]-p[0],1e-9)>=dtc)
    valid=np.asarray(valid,bool);tp=0;cover=[];ious=[];frag=[]
    for g in gt:
        ovs=np.asarray([overlap(p,g) for p in pred]);cov=min(1.,ovs[valid].sum()/max(g[1]-g[0],1e-9)) if len(pred) else 0.
        cover.append(cov);tp+=int(cov>=gtc);frag.append(int(np.sum(ovs>0)))
        ious.append(float(max([overlap(p,g)/(max(p[1],g[1])-min(p[0],g[0])) for p in pred] or [0.])))
    fp=int(np.sum(~valid));precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/len(gt)
    return {"coverage":float(np.mean(cover)),"temporal_iou":float(np.mean(ious)),
            "fragmentation":float(np.mean(frag)),"intersection_precision":precision,
            "intersection_recall":recall,"intersection_f1":2*precision*recall/(precision+recall) if precision+recall else 0.}

def confusion(times,alarms,onsets,idle):
    tp=sum(np.any(alarms&(times>=o)&(times<=o+1.)) for o in onsets);fn=len(onsets)-tp
    idx=np.flatnonzero(idle);fp=tn=0;size=int(4/STEP)
    for run in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1):
        for k in range(0,len(run)-size+1,size):
            hit=np.any(alarms[run[k:k+size]]);fp+=int(hit);tn+=int(not hit)
    return [[int(tp),int(fn)],[fp,tn]]

def analyze(subject,session):
    covp=Path(f"data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz")
    rawp=Path(f"data/interim/lee2019_four_rest_raw/session{session}/sub-{subject:02d}.npz")
    streamp=Path(f"data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz")
    with np.load(covp,allow_pickle=False) as d:
        ct,it=d["train_command_trial"],d["train_idle_iti_trial"]
        cc=d["train_command_covariance"][ct<50];ic=d["train_idle_iti_covariance"][it<50]
        fc=d["train_fixation_covariance"][d["train_fixation_trial"]<50];pc=d["pre_train_covariance"]
    with np.load(rawp,allow_pickle=False) as d:ac,_=rolling_covariance(d["pre_train"])
    with np.load(streamp,allow_pickle=False) as d:sig,events=d["signal"],d["events"]
    sc,ends=rolling_covariance(sig);times=ends/SFREQ;starts=(ends-int(WINDOW*SFREQ))/SFREQ
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;guard=np.zeros(len(times),bool)
    for s in seams:guard|=(starts<s+3.)&(times>s)
    allowed=~guard;onsets=events[:,5]/SFREQ+.5;gt=np.c_[onsets,events[:,5]/SFREQ+4.]
    state=np.zeros(len(times),bool)
    for g in gt:state|=(times>=g[0])&(times<=g[1])
    idle=~state&allowed

    def lr_scores(sh=.0):
        train_i=shrink(ic,sh) if sh else ic;train_c=shrink(cc,sh) if sh else cc
        ref=reference(train_i);x0=tangent_features(train_i,ref);x1=tangent_features(train_c,ref)
        ss=StandardScaler().fit(np.r_[x1,x0]);m=LogisticRegression(class_weight="balanced",solver="liblinear",max_iter=2000,random_state=20260812).fit(ss.transform(np.r_[x1,x0]),np.r_[np.ones(len(x1)),np.zeros(len(x0))])
        aa=shrink(ac,sh) if sh else ac;xx=shrink(sc,sh) if sh else sc
        return m.decision_function(ss.transform(tangent_features(aa,ref))),m.decision_function(ss.transform(tangent_features(xx,ref)))
    ba,bs=lr_scores();sa,ss=lr_scores(.1)
    allidle=np.r_[ic,fc,pc];rg=reference(allidle);sg=StandardScaler().fit(tangent_features(allidle,rg))
    cp=diag_fit(sg.transform(tangent_features(cc,rg)));ips=[diag_fit(sg.transform(tangent_features(x,rg))) for x in (ic,fc,pc)]
    def llr(c):
        x=sg.transform(tangent_features(c,rg));return diag_lp(x,cp)-logsumexp(np.c_[tuple(diag_lp(x,p) for p in ips)],axis=1)+np.log(3)
    la,ls=llr(ac),llr(sc);baz,bsz=zfix(ba,ba),zfix(bs,ba);laz,lsz=zfix(la,la),zfix(ls,la)
    scores={"baseline":(ba,bs),"shrinkage":(sa,ss),"multi_idle":(la,ls),"soft_veto_0.5":(baz-.5*np.maximum(0,-laz),bsz-.5*np.maximum(0,-lsz))}
    out={}
    for name,(anchor,stream) in scores.items():
        th=tune(anchor);z=ewma_z(stream,float(anchor.mean()),float(anchor.var(ddof=1)),allowed);ma=ma_trace(z)
        alarms=alarms_from_ma(ma,th,allowed);onset={str(d):onset_metrics(times,alarms,onsets,gt,idle,d) for d in DELTAS}
        pred=intervals_from_alarm(times,alarms,ma,th,allowed)
        curve=[]
        for off in OFFSETS:
            aa=alarms_from_ma(ma,th+off,allowed);m=onset_metrics(times,aa,onsets,gt,idle,1.)
            curve.append([m["fp_per_min"],m["sensitivity"]])
        curve=np.asarray(curve);order=np.argsort(curve[:,0]);x=curve[order,0];y=np.maximum.accumulate(curve[order,1]);grid=np.linspace(0,4,81);yg=np.interp(grid,x,y,left=y[0],right=y[-1]);bracket=bool(x[0]<=2.<=x[-1])
        out[name]={"source_threshold":th,"onset":onset,"fp_sweep_range":[float(x[0]),float(x[-1])],"fp2_bracketed":bracket,"sensitivity_at_fp2":float(np.interp(2.,x,y)) if bracket else None,"partial_auc_fp0_4":float(np.trapezoid(yg,grid)/4),"interval_0.7":interval_metrics(pred,gt,.7,.7),"interval_0.1":interval_metrics(pred,gt,.1,.1),"confusion_onset1s":confusion(times,alarms,onsets,idle),"predicted_intervals":len(pred)}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--sessions",type=int,nargs="*",default=[1,2]);p.add_argument("--subjects",type=int,nargs="*");p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_locked_selfpaced_metrics.json"));a=p.parse_args();subs=a.subjects or list(range(1,55));rows=[]
    for s in a.sessions:
        for i,u in enumerate(subs,1):rows.append({"session":s,"subject":u,"arms":analyze(u,s)});print(f"session{s} [{i}/{len(subs)}] sub-{u:02d}",flush=True)
    summary={}
    for arm in ARMS:
        def vals(path):
            z=[]
            for r in rows:
                x=r["arms"][arm]
                for k in path:x=x[k]
                z.append(np.nan if x is None else x)
            return np.asarray(z,float)
        fp2=vals(["sensitivity_at_fp2"])
        summary[arm]={"n":len(rows),"fp2_bracketed_n":int(np.sum(np.isfinite(fp2))),"onset_sensitivity_1s_median":float(np.median(vals(["onset","1.0","sensitivity"]))),"onset_sensitivity_at_fp2_median":float(np.nanmedian(fp2)),"partial_auc_fp0_4_median":float(np.median(vals(["partial_auc_fp0_4"]))),"coverage_median":float(np.median(vals(["interval_0.1","coverage"]))),"iou_median":float(np.median(vals(["interval_0.1","temporal_iou"])))}
    payload={"protocol":{"onset":"cue+0.5 pseudo-onset","primary_latency_seconds":1.,"interval":"alarm start; release trigger-0.5z held 0.3s; gap merge 0.3s","claim_limit":"cue-relative pseudo-onset"},"summary":summary,"rows":rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
