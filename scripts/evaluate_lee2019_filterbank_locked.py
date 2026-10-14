"""Fixed five-band Riemannian gate under the locked onset/interval metrics."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.features import _causal_band_signal
from uncued_mi.riemannian import tangent_features,window_covariances
from scripts.run_lee2019_pseudo_uncued_riemannian import SFREQ,STEP,WINDOW,calibration,reference
from uncued_mi.operating_point import tune_source_operating_points
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (DELTAS,OFFSETS,ewma_z,ma_trace,alarms_from_ma,onset_metrics,intervals_from_alarm,interval_metrics,confusion)
BANDS=((8.,10.),(10.,13.),(13.,18.),(18.,24.),(24.,30.))

def ends(n):return np.arange(3*SFREQ,n+1,int(STEP*SFREQ),dtype=np.int64)
def features(signal,e,refs=None):
    out=[];new=[]
    for i,(lo,hi) in enumerate(BANDS):
        x=_causal_band_signal(signal,SFREQ,lo,hi,4);c=window_covariances(x,e,int(WINDOW*SFREQ));r=reference(c) if refs is None else refs[i];out.append(tangent_features(c,r));new.append(r)
    return np.concatenate(out,axis=1),new
def tune(anchor):
    z=(anchor-anchor.mean())/anchor.std(ddof=1);return float(tune_source_operating_points([calibration(z)],'ma_dwell',[2.],STEP,ma_seconds=.5,dwell_seconds=.2,refractory_seconds=1.,grid_size=160)[2.].threshold)

def analyze(subject,session):
    mp=Path(f'data/interim/lee2019_mil_raw/session{session}/sub-{subject:02d}_mil_raw.npz');rp=Path(f'data/interim/lee2019_four_rest_raw/session{session}/sub-{subject:02d}.npz');sp=Path(f'data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz')
    with np.load(mp,allow_pickle=False) as d:
        train=d['signal'];pe=d['positive_end_samples'][:50].reshape(-1);ie=d['iti_end_samples'][d['iti_instance_trials']<50]
    with np.load(rp,allow_pickle=False) as d:anchor_signal=d['pre_train']
    with np.load(sp,allow_pickle=False) as d:stream=d['signal'];events=d['events']
    ae=ends(len(anchor_signal));anchor_x,refs=features(anchor_signal,ae)
    # Compute only labeled training instances in each band.
    px=[];ix=[]
    for i,(lo,hi) in enumerate(BANDS):
        x=_causal_band_signal(train,SFREQ,lo,hi,4);px.append(tangent_features(window_covariances(x,pe,int(SFREQ)),refs[i]));ix.append(tangent_features(window_covariances(x,ie,int(SFREQ)),refs[i]))
    pos=np.concatenate(px,axis=1);idle=np.concatenate(ix,axis=1);scale=StandardScaler().fit(np.r_[pos,idle]);model=LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=3000,C=.25,random_state=20260812).fit(scale.transform(np.r_[pos,idle]),np.r_[np.ones(len(pos)),np.zeros(len(idle))])
    anchor=model.decision_function(scale.transform(anchor_x));se=ends(len(stream));sx,_=features(stream,se,refs);raw=model.decision_function(scale.transform(sx));times=se/SFREQ;starts=(se-int(WINDOW*SFREQ))/SFREQ
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;guard=np.zeros(len(times),bool)
    for s in seams:guard|=(starts<s+3)&(times>s)
    allowed=~guard;onsets=events[:,5]/SFREQ+.5;gt=np.c_[onsets,events[:,5]/SFREQ+4.];state=np.zeros(len(times),bool)
    for g in gt:state|=(times>=g[0])&(times<=g[1])
    idle_mask=~state&allowed;th=tune(anchor);z=ewma_z(raw,float(anchor.mean()),float(anchor.var(ddof=1)),allowed);ma=ma_trace(z);alarms=alarms_from_ma(ma,th,allowed);pred=intervals_from_alarm(times,alarms,ma,th,allowed)
    onset={str(d):onset_metrics(times,alarms,onsets,gt,idle_mask,d) for d in DELTAS};curve=[]
    for off in OFFSETS:
        a=alarms_from_ma(ma,th+off,allowed);m=onset_metrics(times,a,onsets,gt,idle_mask,1.);curve.append([m['fp_per_min'],m['sensitivity']])
    curve=np.asarray(curve);order=np.argsort(curve[:,0]);x=curve[order,0];y=np.maximum.accumulate(curve[order,1]);bracket=x[0]<=2<=x[-1];grid=np.linspace(0,4,81);yg=np.interp(grid,x,y,left=y[0],right=y[-1])
    return {'subject':subject,'session':session,'source_threshold':th,'onset':onset,'fp2_bracketed':bool(bracket),'sensitivity_at_fp2':float(np.interp(2,x,y)) if bracket else None,'partial_auc_fp0_4':float(np.trapezoid(yg,grid)/4),'interval_0.1':interval_metrics(pred,gt,.1,.1),'interval_0.7':interval_metrics(pred,gt,.7,.7),'confusion_onset1s':confusion(times,alarms,onsets,idle_mask)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--sessions',type=int,nargs='*',default=[1,2]);p.add_argument('--subjects',type=int,nargs='*');p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_filterbank_locked.json'));a=p.parse_args();subs=a.subjects or list(range(1,55));rows=[]
    for s in a.sessions:
        for i,u in enumerate(subs,1):rows.append(analyze(u,s));print(f'session{s} [{i}/{len(subs)}] sub-{u:02d}',flush=True)
    def v(path):
        z=[]
        for r in rows:
            x=r
            for k in path:x=x[k]
            z.append(np.nan if x is None else x)
        return np.asarray(z,float)
    summary={'n':len(rows),'fp2_bracketed_n':int(np.isfinite(v(['sensitivity_at_fp2'])).sum()),'latency_sensitivity':{str(d):float(np.median(v(['onset',str(d),'sensitivity']))) for d in DELTAS},'sensitivity_at_fp2_median':float(np.nanmedian(v(['sensitivity_at_fp2']))),'partial_auc_fp0_4_median':float(np.median(v(['partial_auc_fp0_4']))),'coverage_median':float(np.median(v(['interval_0.1','coverage']))),'iou_median':float(np.median(v(['interval_0.1','temporal_iou']))),'f1_tolerant_median':float(np.median(v(['interval_0.1','intersection_f1']))),'f1_strict_median':float(np.median(v(['interval_0.7','intersection_f1'])))}
    payload={'bands':BANDS,'model':'concatenated per-band tangent features + LR C=0.25','summary':summary,'rows':rows};a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
