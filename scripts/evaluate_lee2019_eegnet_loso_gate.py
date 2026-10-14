"""Exact-LOSO EEGNet gate under the locked Arm-A deployment protocol."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_three_model_validation import split_arrays,train_supervised,predict,windows
from scripts.run_lee2019_pseudo_uncued_riemannian import SFREQ,STEP,WINDOW,calibration,rolling_covariance
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z,ma_trace,alarms_from_ma,intervals_from_alarm,interval_metrics
from uncued_mi.operating_point import tune_source_operating_points

SUBJECTS=tuple(range(1,55));TARGETS=(2.,4.);DELTAS=(1.,2.,3.,4.);SCREEN=10;PER_DOMAIN=25

def even(x,n=PER_DOMAIN):
    return x if len(x)<=n else x[np.linspace(0,len(x)-1,n).round().astype(int)]

def load_source_cache(session):
    cache={}
    for u in SUBJECTS:
        with np.load(f'data/interim/lee2019_mil_raw/session{session}/sub-{u:02d}_mil_raw.npz',allow_pickle=False) as d:
            p,r,i=split_arrays(d,0,50,pre='all');cache[u]=(even(p),even(r),even(i))
    return cache

def source_arrays(cache,target):
    ps=[];rs=[];iis=[]
    for u,(p,r,i) in cache.items():
        if u==target:continue
        ps.append(p);rs.append(r);iis.append(i)
    return np.concatenate(ps),np.concatenate(rs),np.concatenate(iis)

def anchor_stream(session,target):
    with np.load(f'data/interim/lee2019_four_rest_raw/session{session}/sub-{target:02d}.npz',allow_pickle=False) as d:pre=d['pre_train'];trans=int(d['transient_samples'])
    ends=np.arange(trans+250,len(pre)+1,25);anchor_windows=windows(pre,ends)
    with np.load(f'data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{target:02d}.npz',allow_pickle=False) as d:sig,events=d['signal'],d['events']
    ends=np.arange(250,len(sig)+1,25);return anchor_windows,windows(sig,ends),ends,events

def auc(pos,neg):
    v=np.r_[pos,neg];r=rankdata(v);n=len(pos);return float((r[:n].sum()-n*(n+1)/2)/(n*len(neg)))

def tune(anchor):
    z=(anchor-anchor.mean())/max(anchor.std(ddof=1),1e-8);p=tune_source_operating_points([calibration(z)],'ma_dwell',list(TARGETS),STEP,ma_seconds=.5,dwell_seconds=.2,refractory_seconds=1.,grid_size=160);return {k:float(v.threshold) for k,v in p.items()}

def metric_at(ma,thr,allowed,evalmask,idle,exposure,gt,times,onsets):
    a=alarms_from_ma(ma,thr,allowed)&evalmask;fp=float(np.sum(a&idle)/(exposure/60));sen={str(d):float(np.mean([np.any(a&(times>=o)&(times<=o+d)) for o in onsets])) for d in DELTAS};pred=intervals_from_alarm(times,a,ma,thr,allowed&evalmask);tol=interval_metrics(pred,gt,.1,.1);strict=interval_metrics(pred,gt,.7,.7);return {'actual_fp':fp,**sen,'coverage':tol['coverage'],'temporal_iou':tol['temporal_iou'],'strict_f1':strict['intersection_f1']},a

def analyze(session,target,epochs,device,cache):
    p,r,i=source_arrays(cache,target);started=time.perf_counter();model,hist=train_supervised(p,r,i,20260815,epochs,device);train_s=time.perf_counter()-started
    aw,sw,ends,events=anchor_stream(session,target);anchor=predict(model,aw,device);raw=predict(model,sw,device);times=ends/SFREQ;starts=(ends-int(WINDOW*SFREQ))/SFREQ
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;guard=np.zeros(len(times),bool)
    for s in seams:guard|=(starts<s+3)&(times>s)
    allowed=~guard;z=ewma_z(raw,float(anchor.mean()),float(anchor.var(ddof=1)),allowed);ma=ma_trace(z);thresholds=tune(anchor);cues=events[:,5]/SFREQ
    flat=[];margin=[]
    for cue in cues[:SCREEN]:
        valid=(times>=cue+1.5)&(times<cue+4)&allowed;finite=ma[valid&np.isfinite(ma)];flat.append(not np.any(finite>=thresholds[2.]));margin.append(float(np.max(finite)-thresholds[2.]) if len(finite) else -np.inf)
    onsets=cues[SCREEN:]+.5;stops=cues[SCREEN:]+4;evalmask=times>=events[SCREEN,2]/SFREQ;state=np.zeros(len(times),bool)
    for a,b in zip(onsets,stops):state|=(times>=a)&(times<=b)
    idle=evalmask&allowed&~state;exposure=idle.sum()*STEP;gt=np.c_[onsets,stops];surface={}
    for fp in TARGETS:surface[str(fp)],_=metric_at(ma,thresholds[fp],allowed,evalmask,idle,exposure,gt,times,onsets)
    pos=np.concatenate([raw[(starts>=a)&(times<=b)&allowed] for a,b in gt]);neg=raw[idle]
    return {'subject':target,'session':session,'screen_flat_fraction':float(np.mean(flat)),'screen_margin_mean':float(np.mean(margin)),'window_auc':auc(pos,neg),'surface':surface,'training_seconds':train_s,'loss_start':hist[0],'loss_end':hist[-1]}

def summarize(rows):
    out={};selected_all=[]
    for s in (1,2):
        rr=[r for r in rows if r['session']==s];ranked=sorted(rr,key=lambda r:(r['screen_flat_fraction'],-r['screen_margin_mean']));sel=ranked[:int(np.ceil(.2*len(ranked)))];selected_all+=sel
        out[str(s)]={'selected_n':len(sel),'selected_subjects':[r['subject'] for r in sel],'screen_cutoff':sel[-1]['screen_flat_fraction']}
    out['combined']={'n':len(selected_all),'window_auc_median':float(np.median([r['window_auc'] for r in selected_all])),'surface':{}}
    for fp in TARGETS:out['combined']['surface'][str(fp)]={k:float(np.median([r['surface'][str(fp)][k] for r in selected_all])) for k in ('actual_fp',*[str(d) for d in DELTAS],'coverage','temporal_iou','strict_f1')}
    return out

def add_nulls(payload,device,nshift=100):
    rng=np.random.default_rng(20260815);rows=payload['rows'];selected={(s,u) for s in (1,2) for u in payload['summary'][str(s)]['selected_subjects']};null=[]
    # Re-score selected models deterministically, then circularly shift z-score before decision.
    caches={s:load_source_cache(s) for s in (1,2)}
    for row in rows:
        if (row['session'],row['subject']) not in selected:continue
        p,r,i=source_arrays(caches[row['session']],row['subject']);model,_=train_supervised(p,r,i,20260815,payload['protocol']['epochs'],device);aw,sw,ends,events=anchor_stream(row['session'],row['subject']);anchor=predict(model,aw,device);raw=predict(model,sw,device);times=ends/SFREQ;starts=(ends-250)/SFREQ;seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;guard=np.zeros(len(times),bool)
        for seam in seams:guard|=(starts<seam+3)&(times>seam)
        allowed=~guard;z=ewma_z(raw,float(anchor.mean()),float(anchor.var(ddof=1)),allowed);thr=tune(anchor)[4.];cues=events[:,5]/SFREQ;onsets=cues[10:]+.5;stops=cues[10:]+4;evalmask=times>=events[10,2]/SFREQ;state=np.zeros(len(times),bool)
        for a,b in zip(onsets,stops):state|=(times>=a)&(times<=b)
        idle=evalmask&allowed&~state;exposure=idle.sum()*STEP;gt=np.c_[onsets,stops];vals=[]
        for shift in rng.integers(200,len(z)-200,nshift):vals.append(metric_at(ma_trace(np.roll(z,int(shift))),thr,allowed,evalmask,idle,exposure,gt,times,onsets)[0]['2.0'])
        null.append({'session':row['session'],'subject':row['subject'],'mean':float(np.mean(vals))})
    payload['circular_shift_null']=null;payload['summary']['combined']['circular_shift_null_median']=float(np.median([r['mean'] for r in null]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--subjects',type=int,nargs='*');p.add_argument('--epochs',type=int,default=10);p.add_argument('--with-nulls',action='store_true');p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_eegnet_loso_gate.json'));a=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');subs=a.subjects or list(SUBJECTS);rows=[]
    for s in (1,2):
        cache=load_source_cache(s)
        for u in subs:rows.append(analyze(s,u,a.epochs,device,cache));print(f'session{s} sub-{u:02d}',flush=True)
    payload={'protocol':{'exact_loso':True,'epochs':a.epochs,'seed':20260815,'source_windows_per_class_per_subject':PER_DOMAIN,'device':str(device),'threshold':'target pre-rest fixed nominal FP=2/4','screening':'arm-specific first 10 trials; exact top 20%'},'summary':summarize(rows) if len(subs)==54 else None,'rows':rows}
    if a.with_nulls and payload['summary'] is not None:add_nulls(payload,device)
    a.output.write_text(json.dumps(payload,indent=2));print(json.dumps(payload['summary'],indent=2))
if __name__=='__main__':main()
