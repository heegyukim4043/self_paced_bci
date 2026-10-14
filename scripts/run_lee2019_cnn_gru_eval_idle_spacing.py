"""Frozen CNN-GRU under random 0.75/1.5/3 idle-to-MI duration ratios."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.operating_point import ScoredSequence
from scripts.run_lee2019_cnn_gru import CausalCNNGRU,train_model,predict,bootstrap_median
from scripts.run_lee2019_cnn_gru_idle_ratio import SUBSET20,prepare,build_ratio_stream
from scripts.run_lee2019_stateful_gru import SEEDS,build_no_guard_stream,evaluate

IDLE_MI_RATIOS=(.75,1.50,3.00)

def blocks(values,template,intervals):
    out=[]
    for a,b in intervals:
        m=(template.times>=a)&(template.times<b);out.append(values[m])
    return [x for x in out if len(x)]

def analyze(subject,session,epochs,seeds,device,shifts,repeats,model_class=CausalCNNGRU):
    train_idle,train_cmd,canonical,canonical_template,cals=prepare(ROOT/'data/raw/lee2019_mi',ROOT/'data/interim/lee2019_mil_raw',subject,session)
    eval_idle=blocks(canonical,canonical_template,canonical_template.idle_intervals);eval_cmd=blocks(canonical,canonical_template,canonical_template.command_intervals)
    models=[];cal_scores=[]
    for seed in seeds:
        x,y,w,_=build_no_guard_stream(train_idle,train_cmd,20260700+100*session+subject)
        model=train_model(x,y,w,seed,epochs,device,model_class=model_class);models.append(model);cal_scores.append([predict(model,v,device) for v in cals])
    calibration=[np.mean([z[i] for z in cal_scores],0) for i in range(len(cals))];rows=[]
    for rep in range(repeats):
      comp_seed=202609080+10000*rep+100*session+subject
      for ratio in IDLE_MI_RATIOS:
        x,_,_,template,audit=build_ratio_stream(eval_idle,eval_cmd,ratio,comp_seed,minimum_steps=5)
        score=np.mean([predict(m,x,device) for m in models],0);seq=ScoredSequence(template.times,score,template.idle_intervals,template.command_intervals)
        rng=np.random.default_rng(97000+comp_seed);offsets=rng.integers(1,len(score),shifts);metric=evaluate(seq,calibration,offsets)
        rows.append({'idle_mi_ratio':ratio,'idle_occupancy':ratio/(1+ratio),'repeat':rep,'composition_seed':comp_seed,'audit':audit,'metrics':metric,'idle_seconds':float(sum(b-a for a,b in template.idle_intervals)),'mi_seconds':float(sum(b-a for a,b in template.command_intervals))})
    return {'subject':subject,'session':session,'conditions':rows}

def summarize(rows,bootstrap):
    out={}
    for session in (1,2):
      out[str(session)]={}
      for q in IDLE_MI_RATIOS:
        rr=[c for r in rows if r['session']==session for c in r['conditions'] if c['idle_mi_ratio']==q]
        item={}
        for mode in ('transferred','common_fp'):
          ex=np.asarray([x['metrics'][mode]['excess_1s'] for x in rr]);item[mode]={'fp_median':float(np.median([x['metrics'][mode]['fp'] for x in rr])),'raw_sen1_median':float(np.median([x['metrics'][mode]['raw_tpr_1s'] for x in rr])),'excess_median':float(np.median(ex)),'excess_ci':bootstrap_median(ex,20260908+session+int(q*100),bootstrap)}
        item['idle_occupancy']=q/(1+q);item['idle_seconds_median']=float(np.median([x['idle_seconds'] for x in rr]));item['mi_seconds_median']=float(np.median([x['mi_seconds'] for x in rr]));out[str(session)][str(q)]=item
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--subjects',type=int,nargs='+',default=SUBSET20);p.add_argument('--sessions',type=int,nargs='+',default=(1,2));p.add_argument('--epochs',type=int,default=10);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--repeats',type=int,default=5);p.add_argument('--shifts',type=int,default=500);p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_cnn_gru_eval_idle_mi_ratio_subset20.json'));a=p.parse_args();cp=a.output.with_suffix('.partial.json');rows=json.loads(cp.read_text()) if cp.exists() else [];done={(r['session'],r['subject']) for r in rows};device=torch.device(a.device)
    for session in a.sessions:
      for subject in a.subjects:
        if (session,subject) in done:continue
        print(f'eval-spacing session={session} sub={subject:02d}',flush=True);rows.append(analyze(subject,session,a.epochs,tuple(a.seeds),device,a.shifts,a.repeats));cp.write_text(json.dumps(rows),encoding='utf-8')
    payload={'protocol':{'model_training':'canonical frozen CNN-GRU','evaluation_idle_mi_ratios':IDLE_MI_RATIOS,'minimum_idle_seconds':.5,'random_distribution':'gamma shape=2 normalized to exact total','random_compositions':a.repeats,'paired_trial_order':True,'secondary_only':True},'summary':summarize(rows,a.bootstrap),'rows':rows};a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload['summary'],indent=2))
if __name__=='__main__':main()
