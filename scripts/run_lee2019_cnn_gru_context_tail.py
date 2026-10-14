"""CNN-GRU context randomization and idle-tail ranking ablation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.operating_point import ScoredSequence
from scripts.run_lee2019_cnn_gru import CausalCNNGRU,train_model,predict,bootstrap_median
from scripts.run_lee2019_cnn_gru_idle_ratio import SUBSET20,prepare,sample_contiguous
from scripts.run_lee2019_stateful_gru import SEEDS,build_no_guard_stream,evaluate

ARMS=('baseline_bce','context_bce','context_tail_rank')

def make_pair_batch(idle,commands,indices,rng,device):
    pairs=[]
    for _ in range(2):
        seq=[];labels=[]
        for i in indices:
            cmd=commands[i];ratio=float(rng.choice((.75,1.5,3.0)));mean=max(5,len(cmd)*ratio)
            length=int(np.clip(round(rng.lognormal(np.log(mean)-.125,.5)),5,120));rest=sample_contiguous(idle,length,rng)
            x=np.concatenate([rest,cmd]);y=np.r_[np.zeros(len(rest)),np.ones(len(cmd))];seq.append(x);labels.append(y)
        maxlen=max(map(len,seq));shape=(len(seq),maxlen,*seq[0].shape[1:]);x=np.zeros(shape,np.float32);y=np.zeros((len(seq),maxlen),np.float32);m=np.zeros((len(seq),maxlen),bool)
        for j,(a,b) in enumerate(zip(seq,labels)):x[j,:len(a)]=a;y[j,:len(a)]=b;m[j,:len(a)]=True
        pairs.append((torch.from_numpy(x).to(device),torch.from_numpy(y).to(device),torch.from_numpy(m).to(device)))
    return pairs

def masked_bce(logits,y,mask):
    pos=torch.sum((y==1)&mask).float();neg=torch.sum((y==0)&mask).float();pw=torch.clamp(neg/torch.clamp(pos,min=1),min=1)
    loss=nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pw,reduction='none');return loss[mask].mean()

def auxiliary(logits,y,mask):
    ranks=[]
    for j in range(len(logits)):
        pos=logits[j][(y[j]==1)&mask[j]];neg=logits[j][(y[j]==0)&mask[j]]
        k=max(1,int(np.ceil(.1*len(neg))));hard=torch.topk(neg,k).values.mean();event=torch.logsumexp(pos/.5,0)*.5-.5*np.log(max(len(pos),1));ranks.append(nn.functional.softplus(.5+hard-event))
    return torch.stack(ranks).mean()

def train_paired(idle,commands,seed,epochs,device,use_rank):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=CausalCNNGRU(commands[0].shape[2],32).to(device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);rng=np.random.default_rng(seed+400000)
    for epoch in range(epochs):
        order=rng.permutation(len(commands))
        for start in range(0,len(order),10):
            batch=order[start:start+10];(x1,y1,m1),(x2,y2,m2)=make_pair_batch(idle,commands,batch,rng,device);opt.zero_grad(set_to_none=True);z1,_=model(x1);z2,_=model(x2)
            loss=masked_bce(z1,y1,m1)+masked_bce(z2,y2,m2)
            consistency=[]
            for j,i in enumerate(batch):
                n=len(commands[i]);consistency.append(torch.mean((z1[j][m1[j]][-n:]-z2[j][m2[j]][-n:])**2))
            loss=loss+.1*torch.stack(consistency).mean()
            if use_rank:loss=loss+.2*(auxiliary(z1,y1,m1)+auxiliary(z2,y2,m2))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
    return model

def analyze(subject,session,epochs,seeds,device,shifts):
    idle,cmd,eval_x,template,cals=prepare(ROOT/'data/raw/lee2019_mi',ROOT/'data/interim/lee2019_mil_raw',subject,session);results={}
    for arm in ARMS:
        es=[];cs=[]
        for seed in seeds:
            if arm=='baseline_bce':
                x,y,w,_=build_no_guard_stream(idle,cmd,20260700+100*session+subject);model=train_model(x,y,w,seed,epochs,device)
            else:model=train_paired(idle,cmd,seed,epochs,device,arm=='context_tail_rank')
            es.append(predict(model,eval_x,device));cs.append([predict(model,v,device) for v in cals])
        score=np.mean(es,0);cal=[np.mean([z[i] for z in cs],0) for i in range(len(cals))];seq=ScoredSequence(template.times,score,template.idle_intervals,template.command_intervals);rng=np.random.default_rng(20261100+100*session+subject);off=rng.integers(1,len(score),shifts);results[arm]=evaluate(seq,cal,off)
    return {'subject':subject,'session':session,'arms':results}

def summarize(rows,repeats):
    out={}
    for session in (1,2):
        rr=[r for r in rows if r['session']==session];out[str(session)]={}
        for ai,arm in enumerate(ARMS):
            item={}
            for mode in ('transferred','common_fp'):
                ex=np.array([r['arms'][arm][mode]['excess_1s'] for r in rr]);item[mode]={'fp_median':float(np.median([r['arms'][arm][mode]['fp'] for r in rr])),'sen1_median':float(np.median([r['arms'][arm][mode]['raw_tpr_1s'] for r in rr])),'excess_median':float(np.median(ex)),'excess_ci95':bootstrap_median(ex,20261200+100*session+10*ai+len(mode),repeats)}
                if arm!='baseline_bce':
                    delta=np.array([r['arms'][arm][mode]['excess_1s']-r['arms']['baseline_bce'][mode]['excess_1s'] for r in rr]);item[mode]['paired_delta_excess_median']=float(np.median(delta));item[mode]['paired_delta_excess_ci95']=bootstrap_median(delta,20261300+100*session+10*ai+len(mode),repeats)
            out[str(session)][arm]=item
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--subjects',type=int,nargs='+',default=SUBSET20);p.add_argument('--sessions',type=int,nargs='+',default=(1,2));p.add_argument('--epochs',type=int,default=10);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--shifts',type=int,default=500);p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--output',type=Path,default=Path('data/processed/lee2019_cnn_gru_context_tail_subset20.json'));a=p.parse_args();cp=a.output.with_suffix('.partial.json');rows=json.loads(cp.read_text()) if cp.exists() else [];done={(r['session'],r['subject']) for r in rows};device=torch.device(a.device)
    for session in a.sessions:
      for subject in a.subjects:
        if (session,subject) in done:continue
        print(f'context-tail S{session} sub-{subject:02d}',flush=True);rows.append(analyze(subject,session,a.epochs,tuple(a.seeds),device,a.shifts));cp.write_text(json.dumps(rows),encoding='utf-8')
    payload={'protocol':{'primary_eval':'canonical held-out last 50 trials','arms':ARMS,'context_ratios':(.75,1.5,3.),'context_loss_weight':.1,'tail_rank_weight':.2,'hard_idle_top_fraction':.1,'secondary_spacing_not_used_for_selection':True},'summary':summarize(rows,a.bootstrap),'rows':rows};a.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload['summary'],indent=2))
if __name__=='__main__':main()
