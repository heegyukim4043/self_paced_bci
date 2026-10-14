from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import SFREQ,STEP,WINDOW,calibration,reference,rolling_covariance
from uncued_mi.operating_point import tune_source_operating_points
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z,ma_trace,alarms_from_ma,intervals_from_alarm,overlap

def one(subject,session):
    cp=Path(f'data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz');rp=Path(f'data/interim/lee2019_four_rest_raw/session{session}/sub-{subject:02d}.npz');sp=Path(f'data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz')
    with np.load(cp,allow_pickle=False) as d:
        ct,it=d['train_command_trial'],d['train_idle_iti_trial'];ref=reference(d['train_idle_iti_covariance'][it<50]);c=tangent_features(d['train_command_covariance'][ct<50],ref);q=tangent_features(d['train_idle_iti_covariance'][it<50],ref)
    sc=StandardScaler().fit(np.r_[c,q]);m=LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=2000,random_state=20260812).fit(sc.transform(np.r_[c,q]),np.r_[np.ones(len(c)),np.zeros(len(q))])
    with np.load(rp,allow_pickle=False) as d:ac,_=rolling_covariance(d['pre_train'])
    with np.load(sp,allow_pickle=False) as d:sig,events=d['signal'],d['events']
    cov,ends=rolling_covariance(sig);times=ends/SFREQ;starts=(ends-int(WINDOW*SFREQ))/SFREQ;anchor=m.decision_function(sc.transform(tangent_features(ac,ref)));raw=m.decision_function(sc.transform(tangent_features(cov,ref)))
    z0=(anchor-anchor.mean())/anchor.std(ddof=1);th=float(tune_source_operating_points([calibration(z0)],'ma_dwell',[2.],STEP,ma_seconds=.5,dwell_seconds=.2,refractory_seconds=1.,grid_size=160)[2.].threshold)
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;guard=np.zeros(len(times),bool)
    for s in seams:guard|=(starts<s+3)&(times>s)
    z=ewma_z(raw,float(anchor.mean()),float(anchor.var(ddof=1)),~guard);ma=ma_trace(z);alarms=alarms_from_ma(ma,th,~guard);pred=intervals_from_alarm(times,alarms,ma,th,~guard)
    gt=np.c_[events[:,5]/SFREQ+.5,events[:,5]/SFREQ+4.];rows=[]
    for g in gt:
        ov=np.asarray([overlap(p,g) for p in pred]);positive=ov[ov>0];rows.append({'fragments':int(len(positive)),'overlap_seconds':float(positive.sum()),'coverage':float(min(1,positive.sum()/(g[1]-g[0]))),'longest_overlap_seconds':float(positive.max()) if len(positive) else 0.})
    return {'subject':subject,'session':session,'trials':rows,'predicted_intervals':len(pred),'predicted_duration_seconds':[float(b-a) for a,b in pred]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--subjects',type=int,nargs='*');p.add_argument('--sessions',type=int,nargs='*',default=[1,2]);p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_interval_sparsity.json'));a=p.parse_args();subs=a.subjects or list(range(1,55));rows=[]
    for s in a.sessions:
        for i,u in enumerate(subs,1):rows.append(one(u,s));print(f'session{s} [{i}/{len(subs)}] sub-{u:02d}',flush=True)
    trials=[t for r in rows for t in r['trials']];det=[t for t in trials if t['fragments']>0];dur=[d for r in rows for d in r['predicted_duration_seconds']]
    summary={'subject_sessions':len(rows),'trials':len(trials),'any_overlap_trials':len(det),'any_overlap_fraction':len(det)/len(trials),'zero_overlap_fraction':1-len(det)/len(trials),'multi_fragment_trial_fraction':np.mean([t['fragments']>=2 for t in trials]),'fragments_per_trial_mean':float(np.mean([t['fragments'] for t in trials])),'coverage_all_trials_mean':float(np.mean([t['coverage'] for t in trials])),'coverage_detected_trials_median':float(np.median([t['coverage'] for t in det])),'overlap_seconds_detected_median':float(np.median([t['overlap_seconds'] for t in det])),'longest_overlap_seconds_detected_median':float(np.median([t['longest_overlap_seconds'] for t in det])),'predicted_interval_duration_median':float(np.median(dur))}
    payload={'summary':summary,'rows':rows};a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
