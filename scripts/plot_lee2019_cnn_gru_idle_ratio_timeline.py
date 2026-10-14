"""Time-domain comparison of CNN-GRU training sequence ratios."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.decision import instantaneous
from uncued_mi.operating_point import ScoredSequence
from scripts.run_lee2019_cnn_gru import train_model,predict
from scripts.run_lee2019_cnn_gru_idle_ratio import RATIOS,prepare,build_ratio_stream
from scripts.run_lee2019_stateful_gru import SEEDS,tune_threshold

def mask_intervals(times,intervals):
    out=np.zeros(len(times),bool)
    for a,b in intervals:out|=(times>=a)&(times<b)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,default=3);p.add_argument('--session',type=int,default=1);p.add_argument('--start',type=float,default=0);p.add_argument('--stop',type=float,default=60);p.add_argument('--epochs',type=int,default=10);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--output',type=Path);a=p.parse_args();device=torch.device(a.device)
    idle,cmd,eval_x,template,calibration=prepare(ROOT/'data/raw/lee2019_mi',ROOT/'data/interim/lee2019_mil_raw',a.subject,a.session)
    rows=[]
    for ratio in RATIOS:
        scores=[]
        for seed in SEEDS:
            x,y,w,_,audit=build_ratio_stream(idle,cmd,ratio,seed+1000*a.session+a.subject)
            model=train_model(x,y,w,seed,a.epochs,device);scores.append(predict(model,eval_x,device))
        score=np.mean(scores,0);seq=ScoredSequence(template.times,score,template.idle_intervals,template.command_intervals);point=tune_threshold([seq]);alarms=instantaneous(score,point.threshold,.1,1.).alarms
        rows.append((ratio,score,point.threshold,alarms))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(16,9),sharex=True);view=(template.times>=a.start)&(template.times<=a.stop);idle_mask=mask_intervals(template.times,template.idle_intervals)
    for ax,(ratio,score,thr,alarms) in zip(axes,rows):
        for j,(lo,hi) in enumerate(template.command_intervals):
            if hi>=a.start and lo<=a.stop:ax.axvspan(lo,hi,color='#e98b9a',alpha=.22,label='true MI' if j==0 else None)
        margin=score-thr;ax.plot(template.times[view],margin[view],color='#2878b5',lw=1.2,label='score - common-FP threshold');ax.axhline(0,color='#c82433',ls='--',lw=1.1)
        am=alarms&view;ax.scatter(template.times[am],margin[am],marker='v',s=34,color='black',zorder=5,label='alarm')
        false=(score>=thr)&idle_mask&view;ax.scatter(template.times[false],margin[false],s=9,color='#e15759',alpha=.65,label='idle threshold exceedance')
        ax.set_ylabel('margin');ax.set_title(f'idle+rest / MI training windows = {int(ratio*100)}%   threshold={thr:.2f}',loc='left');ax.grid(axis='y',alpha=.18);ax.legend(loc='upper right',ncol=4,fontsize=8)
    axes[-1].set_xlabel('Time in identical evaluation stream (s)');axes[-1].set_xlim(a.start,a.stop);fig.suptitle(f'CNN-GRU sequence-ratio time-domain comparison — Lee2019 subject {a.subject:02d}, session {a.session}\n(common achieved FP operating point; evaluation stream fixed)');fig.tight_layout(rect=(0,0,1,.94));out=a.output or ROOT/f'data/processed/lee2019_cnn_gru_idle_ratio_timeline_sub{a.subject:02d}_s{a.session}_{a.start:g}-{a.stop:g}s.png';out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180,bbox_inches='tight');print(out)
if __name__=='__main__':main()
