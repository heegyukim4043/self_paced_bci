"""Causal 10-trial polarity calibration for cached CNN-GRU scores."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.adaptation import ewma_normalize_and_decide
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import STEP,HALF_LIFE_SECONDS,MA_SECONDS,DWELL_SECONDS,REFRACTORY_SECONDS,normalize_anchor,tune_ma_dwell
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20,load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration

def mask(times,intervals):
    z=np.zeros(len(times),bool)
    for a,b in intervals:z|=(times>=a)&(times<b)
    return z

def screen(sequence,n=10):
    commands=sequence.command_intervals[:n]; idle=sequence.idle_intervals[:n]
    cm=mask(sequence.times,commands);im=mask(sequence.times,idle);y=np.r_[np.ones(cm.sum()),np.zeros(im.sum())];x=np.r_[sequence.scores[cm],sequence.scores[im]]
    auc=float(roc_auc_score(y,x));contrasts=[]
    for (a,b),(c,d) in zip(commands,idle):
        contrasts.append(float(np.median(sequence.scores[(sequence.times>=a)&(sequence.times<b)])-np.median(sequence.scores[(sequence.times>=c)&(sequence.times<d)])))
    rng=np.random.default_rng(20260909+len(x));v=np.asarray(contrasts);draw=np.median(v[rng.integers(0,len(v),(5000,len(v)))],axis=1);ci=np.quantile(draw,[.025,.975]).tolist()
    return auc,float(np.median(v)),ci

def evaluate(sequence,blocks,sign,cutoff,shifts,seed):
    signed=[sign*x for x in blocks];mean,var,z=normalize_anchor(signed);point=tune_ma_dwell([calibration(x) for x in z],160)
    decision=ewma_normalize_and_decide(sign*sequence.scores,initial_mean=mean,initial_variance=var,threshold=point.threshold,step_seconds=STEP,half_life_seconds=HALF_LIFE_SECONDS,update_mode='all',ma_seconds=MA_SECONDS,dwell_seconds=DWELL_SECONDS,refractory_seconds=REFRACTORY_SECONDS,sigma_floor_fraction=.25)
    tail=sequence.times>=cutoff;commands=sequence.command_intervals[10:];idle=mask(sequence.times,sequence.idle_intervals)&tail;alarms=decision.alarms&tail
    sen={}
    for h in (1.,2.):sen[str(h)]=float(np.mean([np.any(alarms&(sequence.times>=a)&(sequence.times<=min(a+h,b))) for a,b in commands]))
    rng=np.random.default_rng(seed);idx=np.flatnonzero(tail);base=alarms[idx];null=[]
    for off in rng.integers(1,len(idx),shifts):
        rolled=np.zeros_like(alarms);rolled[idx]=np.roll(base,int(off));null.append(np.mean([np.any(rolled&(sequence.times>=a)&(sequence.times<=min(a+1,b))) for a,b in commands]))
    null=float(np.median(null));cm=mask(sequence.times,commands)&tail
    auc=float(roc_auc_score(np.r_[np.ones(cm.sum()),np.zeros(idle.sum())],np.r_[sign*sequence.scores[cm],sign*sequence.scores[idle]]))
    return {'sign':int(sign),'threshold':float(point.threshold),'fp_per_min':float(60*np.sum(alarms&idle)/(idle.sum()*STEP)),'sen1':sen['1.0'],'sen2':sen['2.0'],'null1':null,'excess1':sen['1.0']-null,'tail_window_auc':auc}

def boot(v,seed):
    v=np.asarray(v,float);r=np.random.default_rng(seed);d=v[r.integers(0,len(v),(20000,len(v)))];return np.quantile(np.median(d,axis=1),[.025,.975]).tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('data/interim/lee2019_cnn_gru_scores_subset20'));p.add_argument('--subjects',type=int,nargs='+',default=SUBSET20);p.add_argument('--sessions',type=int,nargs='+',default=(1,2));p.add_argument('--shifts',type=int,default=500);p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_cnn_gru_polarity_calibration_subset20.json'));a=p.parse_args();rows=[]
    for session in a.sessions:
      for subject in a.subjects:
        seq,blocks=load_cache(a.root/f'session{session}'/f'sub-{subject:02d}.npz');auc,contrast,ci=screen(seq);cutoff=seq.command_intervals[9][1]
        signs={'baseline':1,'hard_auc_flip':-1 if auc<.5 else 1,'confidence_flip':-1 if ci[1]<0 else 1,'affine_sign':-1 if contrast<0 else 1}
        arms={k:evaluate(seq,blocks,s,cutoff,a.shifts,20262000+1000*session+subject+10*i) for i,(k,s) in enumerate(signs.items())}
        rows.append({'subject':subject,'session':session,'screen_auc':auc,'screen_contrast':contrast,'screen_contrast_ci95':ci,'arms':arms})
    summary={}
    for session in a.sessions:
      rr=[r for r in rows if r['session']==session];summary[str(session)]={'n':len(rr),'screen_auc_median':float(np.median([r['screen_auc'] for r in rr])),'arms':{}}
      for i,name in enumerate(rr[0]['arms']):
        pts=[r['arms'][name] for r in rr];item={'flipped_n':sum(p['sign']<0 for p in pts)}
        for metric in ('fp_per_min','sen1','sen2','null1','excess1','tail_window_auc'):item[metric+'_median']=float(np.median([p[metric] for p in pts]))
        if name!='baseline':
          delta=[r['arms'][name]['excess1']-r['arms']['baseline']['excess1'] for r in rr];item['paired_delta_excess1_median']=float(np.median(delta));item['paired_delta_excess1_ci95']=boot(delta,20262100+100*session+i)
        summary[str(session)]['arms'][name]=item
    payload={'protocol':{'screen_trials':10,'evaluation_trials':40,'causal_split':True,'hard_rule':'flip iff screen AUC < 0.5','confidence_rule':'flip iff upper clustered bootstrap CI of trial contrast < 0','secondary_spacing_not_used':True},'summary':summary,'rows':rows};a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
