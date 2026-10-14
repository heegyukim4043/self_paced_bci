"""CNN-GRU ablation of random idle/rest-to-MI training sequence ratios."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path,load_lee2019_run
from uncued_mi.operating_point import ScoredSequence
from scripts.run_fbcnet_validation import filter_banks,normalize
from scripts.run_lee2019_cnn_gru import (SFREQ,STEP,make_windows,rest_windows,train_model,predict,bootstrap_median)
from scripts.run_lee2019_stateful_gru import SEEDS,build_no_guard_stream,evaluate

SUBSET20=(2,3,9,10,12,13,14,17,19,20,23,25,26,29,38,40,42,51,53,54)
RATIOS=(.25,.50,.75)

def random_composition(total,n,rng,minimum_steps=1):
    """Positive random lengths summing exactly to total (no fixed trial rhythm)."""
    required=n*minimum_steps
    if total<required: raise ValueError(f'negative-window budget {total} is smaller than minimum {required}')
    weights=rng.gamma(2.,1.,n);extra=rng.multinomial(total-required,weights/weights.sum())
    return extra+minimum_steps

def sample_contiguous(blocks,length,rng):
    eligible=[i for i,x in enumerate(blocks) if len(x)]
    pieces=[];left=int(length)
    while left:
        bi=int(rng.choice(eligible));block=blocks[bi];take=min(left,len(block))
        if len(block)>take:start=int(rng.integers(0,len(block)-take+1))
        else:start=0
        pieces.append(block[start:start+take]);left-=take
    return np.concatenate(pieces)

def build_ratio_stream(idle_blocks,command_trials,ratio,seed,minimum_steps=1):
    rng=np.random.default_rng(seed);order=rng.permutation(len(command_trials))
    commands=[command_trials[int(i)] for i in order];npos=sum(map(len,commands))
    nneg=int(round(npos*ratio));lengths=random_composition(nneg,len(commands),rng,minimum_steps)
    values=[];labels=[];weights=[];idle=[];command=[]
    def now():return len(values)*STEP
    for event,nidle in zip(commands,lengths):
        rest=sample_contiguous(idle_blocks,int(nidle),rng);a=now();values.extend(rest.tolist());labels.extend([0.]*len(rest));weights.extend([1.]*len(rest));idle.append((a,now()))
        a=now();values.extend(event.tolist());labels.extend([1.]*len(event));weights.extend([2. if i<10 else 1. for i in range(len(event))]);command.append((a,now()))
    x=np.asarray(values,np.float32);return x,np.asarray(labels,np.float32),np.asarray(weights,np.float32),ScoredSequence(np.arange(len(x))*STEP,np.zeros(len(x)),np.asarray(idle),np.asarray(command)),{'positive_windows':npos,'negative_windows':nneg,'achieved_ratio':nneg/npos,'rest_lengths_steps':lengths.tolist()}

def prepare(data_root,cache_root,subject,session):
    run=load_lee2019_run(lee2019_path(data_root,subject,session),'train');selected=ExperimentConfig().sensorimotor_channels
    signal=filter_banks(run.eeg,run.channels,run.sfreq,selected);raw=[filter_banks(x,run.channels,run.sfreq,selected) for x in (run.pre_rest,run.post_rest)]
    test=load_lee2019_run(lee2019_path(data_root,subject,session),'test');raw += [filter_banks(x,test.channels,test.sfreq,selected) for x in (test.pre_rest,test.post_rest)]
    signal,pre=normalize(signal,raw[0]);cal=pre[...,int(2*SFREQ):];mean=cal.mean(-1,keepdims=True);std=cal.std(-1,keepdims=True).clip(1e-6);rests=[pre]+[(x-mean)/std for x in raw[1:]]
    with np.load(cache_root/f'session{session}'/f'sub-{subject:02d}_mil_raw.npz') as d:
        cmd=[]
        for start,stop in d['command_intervals']:
            ends=np.arange(int(round((start+1)*SFREQ)),int(round(stop*SFREQ)),int(STEP*SFREQ));cmd.append(make_windows(signal,ends))
        iti=[]
        for trial in range(50):
            mask=d['iti_instance_trials']==trial
            if np.any(mask):iti.append(make_windows(signal,d['iti_end_samples'][mask]))
    split=[rest_windows(x) for x in rests];train_idle=iti+[split[0][0]]
    eval_x,_,_,template=build_no_guard_stream([x[1] for x in split],cmd[50:],20260728+100*session+subject)
    calibration=[x[0] for x in split[1:]]
    return train_idle,cmd[:50],eval_x,template,calibration

def analyze(data_root,cache_root,subject,session,ratio,seeds,epochs,device,shifts):
    idle,cmd,eval_x,template,calibration=prepare(data_root,cache_root,subject,session);seed_eval=[];seed_cal=[];audits=[]
    for seed in seeds:
        x,y,w,_,audit=build_ratio_stream(idle,cmd,ratio,seed+1000*session+subject);audits.append(audit)
        model=train_model(x,y,w,seed,epochs,device);seed_eval.append(predict(model,eval_x,device));seed_cal.append([predict(model,v,device) for v in calibration])
    scores=np.mean(seed_eval,0);cals=[np.mean([v[i] for v in seed_cal],0) for i in range(len(calibration))]
    seq=ScoredSequence(template.times,scores,template.idle_intervals,template.command_intervals);rng=np.random.default_rng(92000+100*session+subject);offsets=rng.integers(1,len(scores),shifts)
    return {'subject':subject,'session':session,'idle_ratio':ratio,'composition_audits':audits,'metrics':evaluate(seq,cals,offsets)}

def summarize(rows,ratios,sessions,repeats):
    out={}
    for session in sessions:
      out[str(session)]={}
      for ratio in ratios:
        rr=[x for x in rows if x['session']==session and np.isclose(x['idle_ratio'],ratio)];item={}
        for mode in ('transferred','common_fp'):
          ex=np.asarray([x['metrics'][mode]['excess_1s'] for x in rr]);item[mode]={'fp_median':float(np.median([x['metrics'][mode]['fp'] for x in rr])),'raw_tpr_1s_median':float(np.median([x['metrics'][mode]['raw_tpr_1s'] for x in rr])),'excess_1s_median':float(np.median(ex)),'excess_1s_ci':bootstrap_median(ex,20260908+session+int(ratio*100),repeats)}
        out[str(session)][str(ratio)]=item
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('data/raw/lee2019_mi'));p.add_argument('--cache-root',type=Path,default=Path('data/interim/lee2019_mil_raw'));p.add_argument('--subjects',type=int,nargs='+',default=SUBSET20);p.add_argument('--sessions',type=int,nargs='+',default=(1,2));p.add_argument('--ratios',type=float,nargs='+',default=RATIOS);p.add_argument('--epochs',type=int,default=10);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--shifts',type=int,default=500);p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_cnn_gru_idle_ratio_subset20.json'));a=p.parse_args();cp=a.output.with_suffix('.partial.json');rows=json.loads(cp.read_text()) if cp.exists() else [];done={(x['session'],x['subject'],x['idle_ratio']) for x in rows};device=torch.device(a.device)
    for session in a.sessions:
      for ratio in a.ratios:
       for subject in a.subjects:
        if (session,subject,ratio) in done:continue
        print(f'cnn-gru ratio={ratio:.2f} session={session} sub={subject:02d}',flush=True);rows.append(analyze(a.data_root,a.cache_root,subject,session,ratio,tuple(a.seeds),a.epochs,device,a.shifts));cp.write_text(json.dumps(rows),encoding='utf-8')
    payload={'protocol':{'idle_ratio_definition':'number of idle+rest training windows / number of MI training windows','ratios':a.ratios,'randomization':'conditional exponential/multinomial total with contiguous idle-block sampling; paired model/composition seeds','evaluation_stream':'identical existing random-rest stream for every arm','seeds':a.seeds},'summary':summarize(rows,a.ratios,a.sessions,a.bootstrap),'rows':rows};a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload['summary'],indent=2))
if __name__=='__main__':main()
