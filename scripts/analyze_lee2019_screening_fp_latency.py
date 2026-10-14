"""Screening x FP budget x latency and repeated-attempt analysis."""
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
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma,
    ewma_z,
    interval_metrics,
    intervals_from_alarm,
    ma_trace,
)
DELTAS=(.5,1.,2.,3.,4.);TARGETS=(.5,1.,2.,4.,8.);OFFSETS=np.linspace(-4,4,61);SCREEN=10

def analyze(subject,session,cutoff):
    cp=Path(f'data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz');rp=Path(f'data/interim/lee2019_four_rest_raw/session{session}/sub-{subject:02d}.npz');sp=Path(f'data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz')
    with np.load(cp,allow_pickle=False) as d:
        ct,it=d['train_command_trial'],d['train_idle_iti_trial'];ref=reference(d['pre_train_covariance']);c=tangent_features(d['train_command_covariance'][ct<50],ref);q=tangent_features(d['train_idle_iti_covariance'][it<50],ref)
    scale=StandardScaler().fit(np.r_[c,q]);model=LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=2000,random_state=20260812).fit(scale.transform(np.r_[c,q]),np.r_[np.ones(len(c)),np.zeros(len(q))])
    with np.load(rp,allow_pickle=False) as d:ac,_=rolling_covariance(d['pre_train'])
    with np.load(sp,allow_pickle=False) as d:sig,events=d['signal'],d['events']
    cov,ends=rolling_covariance(sig);times=ends/SFREQ;starts=(ends-int(WINDOW*SFREQ))/SFREQ;anchor=model.decision_function(scale.transform(tangent_features(ac,ref)));raw=model.decision_function(scale.transform(tangent_features(cov,ref)))
    az=(anchor-anchor.mean())/anchor.std(ddof=1);source_points=tune_source_operating_points([calibration(az)],'ma_dwell',list(TARGETS),STEP,ma_seconds=.5,dwell_seconds=.2,refractory_seconds=1.,grid_size=160);source=float(source_points[2.].threshold);fixed_thresholds={k:float(v.threshold) for k,v in source_points.items()}
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ;guard=np.zeros(len(times),bool)
    for s in seams:guard|=(starts<s+3)&(times>s)
    allowed=~guard;z=ewma_z(raw,float(anchor.mean()),float(anchor.var(ddof=1)),allowed);ma=ma_trace(z);onsets=events[:,5]/SFREQ+.5;stops=events[:,5]/SFREQ+4.;eval_begin=events[SCREEN,2]/SFREQ;evalmask=times>=eval_begin
    state=np.zeros(len(times),bool)
    for a,b in zip(onsets[SCREEN:],stops[SCREEN:]):state|=(times>=a)&(times<=b)
    idle=evalmask&~state&allowed;exposure=idle.sum()*STEP
    gt=np.c_[onsets[SCREEN:],stops[SCREEN:]]
    points=[];trial_vectors=[]
    for off in OFFSETS:
        alarms=alarms_from_ma(ma,source+off,allowed)&evalmask;fp=float(np.sum(alarms&idle)/(exposure/60));sens=[];vectors={}
        for delta in DELTAS:
            v=np.asarray([np.any(alarms&(times>=o)&(times<=o+delta)) for o in onsets[SCREEN:]],bool);sens.append(float(v.mean()));vectors[str(delta)]=v
        pred=intervals_from_alarm(times,alarms,ma,source+off,allowed&evalmask)
        tolerant=interval_metrics(pred,gt,.1,.1);strict=interval_metrics(pred,gt,.7,.7)
        mean_duration=float(np.mean(pred[:,1]-pred[:,0])) if len(pred) else 0.
        points.append([fp,*sens,tolerant['coverage'],tolerant['temporal_iou'],tolerant['fragmentation'],tolerant['intersection_f1'],strict['intersection_f1'],mean_duration]);trial_vectors.append(vectors)
    points=np.asarray(points);order=np.argsort(points[:,0]);x=points[order,0];out={}
    for target in TARGETS:
        out[str(target)]={'bracketed':bool(x[0]<=target<=x[-1])}
        for j,delta in enumerate(DELTAS,1):
            y=np.maximum.accumulate(points[order,j]);out[str(target)][str(delta)]=float(np.interp(target,x,y)) if x[0]<=target<=x[-1] else None
        for j,name in enumerate(('coverage','temporal_iou','fragmentation','f1_dtc_gtc_0.1','f1_dtc_gtc_0.7','mean_interval_seconds'),1+len(DELTAS)):
            y=points[order,j]
            out[str(target)][name]=float(np.interp(target,x,y)) if x[0]<=target<=x[-1] else None
        admissible=np.flatnonzero(points[:,0]<=target)
        if len(admissible):
            best_sen=np.max(points[admissible,2]);tied=admissible[np.isclose(points[admissible,2],best_sen)];best=int(tied[np.argmin(points[tied,0])])
            out[str(target)]['deployable']={'actual_fp':float(points[best,0]),**{str(delta):float(points[best,j]) for j,delta in enumerate(DELTAS,1)},'coverage':float(points[best,1+len(DELTAS)]),'temporal_iou':float(points[best,2+len(DELTAS)]),'f1_dtc_gtc_0.7':float(points[best,5+len(DELTAS)])}
        else:out[str(target)]['deployable']=None
        fixed_alarms=alarms_from_ma(ma,fixed_thresholds[target],allowed)&evalmask;fixed_fp=float(np.sum(fixed_alarms&idle)/(exposure/60));fixed_sens={str(delta):float(np.mean([np.any(fixed_alarms&(times>=o)&(times<=o+delta)) for o in onsets[SCREEN:]])) for delta in DELTAS};fixed_pred=intervals_from_alarm(times,fixed_alarms,ma,fixed_thresholds[target],allowed&evalmask);fixed_tol=interval_metrics(fixed_pred,gt,.1,.1);fixed_strict=interval_metrics(fixed_pred,gt,.7,.7);out[str(target)]['fixed_threshold']={'actual_fp':fixed_fp,**fixed_sens,'coverage':fixed_tol['coverage'],'temporal_iou':fixed_tol['temporal_iou'],'f1_dtc_gtc_0.7':fixed_strict['intersection_f1']}
    nearest=int(np.argmin(np.abs(points[:,0]-2.)));vec4=trial_vectors[nearest]['4.0'];vec1=trial_vectors[nearest]['1.0'];fail=~vec4
    return {'subject':subject,'session':session,'selected':None,'cutoff':cutoff,'fp_latency':out,'nearest_fp2':float(points[nearest,0]),'success_1s':vec1.astype(int).tolist(),'success_4s':vec4.astype(int).tolist(),'p_success4':float(vec4.mean()),'p_next_success_given_failure':float(np.mean(vec4[1:][fail[:-1]])) if np.any(fail[:-1]) else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_screening_fp_latency.json'));a=p.parse_args();rows=[]
    for s in (1,2):
        rel=json.load(open(f'data/processed/lee2019_prospective_screening_n10_session{s}_relative.json',encoding='utf-8'));cut=float(rel['cutoff']);screen=json.load(open(f'data/processed/lee2019_prospective_screening_n10_session{s}.json',encoding='utf-8'));flat={r['subject']:r['screen_flat_fraction'] for r in screen['rows']}
        for u in range(1,55):
            r=analyze(u,s,cut);r['selected']=bool(flat[u]<=cut);r['screen_flat_fraction']=float(flat[u]);rows.append(r);print(f'session{s} sub-{u:02d}',flush=True)
    summary={}
    for group,select in [('all',None),('selected',True),('nonselected',False)]:
        rr=rows if select is None else [r for r in rows if r['selected']==select];g={'n':len(rr),'surface':{}}
        for fp in TARGETS:
            g['surface'][str(fp)]={str(d):float(np.nanmedian([r['fp_latency'][str(fp)][str(d)] for r in rr])) for d in DELTAS}
            for metric in ('coverage','temporal_iou','fragmentation','f1_dtc_gtc_0.1','f1_dtc_gtc_0.7','mean_interval_seconds'):
                g['surface'][str(fp)][metric]=float(np.nanmedian([r['fp_latency'][str(fp)][metric] for r in rr]))
            deploy=[r['fp_latency'][str(fp)]['deployable'] for r in rr if r['fp_latency'][str(fp)]['deployable'] is not None]
            g['surface'][str(fp)]['deployable']={key:float(np.nanmedian([v[key] for v in deploy])) for key in ('actual_fp',*[str(d) for d in DELTAS],'coverage','temporal_iou','f1_dtc_gtc_0.7')}
            fixed=[r['fp_latency'][str(fp)]['fixed_threshold'] for r in rr];g['surface'][str(fp)]['fixed_threshold']={key:float(np.nanmedian([v[key] for v in fixed])) for key in ('actual_fp',*[str(d) for d in DELTAS],'coverage','temporal_iou','f1_dtc_gtc_0.7')}
        for n in (1,2,3):
            vals=[]
            for r in rr:
                v=np.asarray(r['success_4s'],bool);groups=[np.any(v[i:i+n]) for i in range(0,len(v)-n+1,n)];vals.extend(groups)
            g[f'empirical_success_within_{n}_attempts']=float(np.mean(vals))
        g['independence_prediction_3_attempts']=float(1-(1-np.mean([r['p_success4'] for r in rr]))**3);g['next_success_given_failure']=float(np.nanmean([r['p_next_success_given_failure'] for r in rr if r['p_next_success_given_failure'] is not None]));summary[group]=g
    # Within-session failure clustering at the FP~2, 4-s condition.
    observed=np.mean([np.mean((~np.asarray(r['success_4s'],bool))[:-1]&(~np.asarray(r['success_4s'],bool))[1:]) for r in rows]);rng=np.random.default_rng(20260812);null=[]
    for _ in range(10000):
        z=[]
        for r in rows:
            f=~np.asarray(r['success_4s'],bool);rng.shuffle(f);z.append(np.mean(f[:-1]&f[1:]))
        null.append(np.mean(z))
    summary['failure_clustering']={'adjacent_failure_pair_observed':float(observed),'permutation_mean':float(np.mean(null)),'p_greater':float((1+np.sum(np.asarray(null)>=observed))/(len(null)+1))}
    payload={'definition':'last 40 trials after locked 10-trial screening; cue+0.5 pseudo-onset','summary':summary,'rows':rows};a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
