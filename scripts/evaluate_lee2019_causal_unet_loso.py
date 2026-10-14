"""Causal U-Net segmentation gate under the Lee2019 Arm-A protocol.

The model is trained on source-subject 2-s sequences with sample-wise
idle/command masks. At deployment, the last output of each causal 1-s window
forms the continuous score; all calibration, EWMA, decision and evaluation
steps match the locked Arm-A pipeline.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.causal_unet import CausalUNet1D
from uncued_mi.swpc import seed_everything
from uncued_mi.operating_point import tune_source_operating_points
from scripts.run_lee2019_pseudo_uncued_riemannian import SFREQ, STEP, WINDOW, calibration
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    ewma_z, ma_trace, alarms_from_ma, intervals_from_alarm, interval_metrics,
)

SUBJECTS = tuple(range(1, 55)); TARGETS = (2., 4.); DELTAS = (1., 2., 3., 4.)
SCREEN = 10; CHUNK = 500; PER_CLASS = 8; SEED = 20260815


def _slice_pad(x, start, length=CHUNK):
    start = int(start); out = np.zeros((length, x.shape[1]), np.float32)
    a=max(0,start); b=min(len(x),start+length)
    out[a-start:b-start] = x[a:b]
    return out


def subject_sequences(session: int, subject: int):
    path=Path(f"data/interim/lee2019_mil_raw/session{session}/sub-{subject:02d}_mil_raw.npz")
    with np.load(path, allow_pickle=False) as d:
        sig=d["signal"]; pre=d["pre_rest"]; intervals=d["command_intervals"]
        iti=d["iti_intervals"]
    rng=np.random.default_rng(SEED + 1000*session + subject)
    choose=np.linspace(0,len(intervals)-1,PER_CLASS).round().astype(int)
    xs=[]; ys=[]
    for j in choose:
        onset,stop=intervals[j]; onset=int(round(onset*SFREQ)); stop=int(round(stop*SFREQ))
        start=onset-int(.5*SFREQ)
        x=_slice_pad(sig,start); y=np.zeros(CHUNK,np.float32)
        lo=max(0,onset-start); hi=min(CHUNK,stop-start); y[lo:hi]=1
        xs.append(x.T); ys.append(y)
    # Equal negative sequence count; alternate pre-rest and ITI.
    for j in range(PER_CLASS):
        if j % 2 == 0:
            start=int(rng.integers(0,max(1,len(pre)-CHUNK+1))); x=_slice_pad(pre,start)
        else:
            a,b=iti[j % len(iti)]; a=int(round(a*SFREQ)); b=int(round(b*SFREQ))
            high=max(a+1,b-CHUNK+1); start=int(rng.integers(a,high)) if high>a else a
            x=_slice_pad(sig,start)
        xs.append(x.T); ys.append(np.zeros(CHUNK,np.float32))
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32)


def load_source_cache(session):
    return {u:subject_sequences(session,u) for u in SUBJECTS}


def train_loso(cache,target,epochs,device,width=16):
    x=np.concatenate([v[0] for u,v in cache.items() if u!=target]); y=np.concatenate([v[1] for u,v in cache.items() if u!=target])
    seed_everything(SEED); model=CausalUNet1D(x.shape[1],width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    gen=torch.Generator().manual_seed(SEED); history=[]
    for _ in range(epochs):
        order=torch.randperm(len(x),generator=gen).numpy(); losses=[]; model.train()
        for k in range(0,len(order),32):
            idx=order[k:k+32]; xb=torch.from_numpy(x[idx]).to(device); yb=torch.from_numpy(y[idx]).to(device)
            logits=model(xb); pos=yb.sum(); neg=yb.numel()-pos
            pw=(neg/pos.clamp_min(1)).clamp(1,8)
            loss=F.binary_cross_entropy_with_logits(logits,yb,pos_weight=pw)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return model,history


def windows(signal,ends):
    n=int(WINDOW*SFREQ); return np.stack([signal[e-n:e].T for e in ends]).astype(np.float32)


@torch.no_grad()
def predict(model,x,device,batch=128):
    model.eval(); out=[]
    for k in range(0,len(x),batch): out.append(model(torch.from_numpy(x[k:k+batch]).to(device))[:,-1].cpu().numpy())
    return np.concatenate(out)


def anchor_stream(session,target):
    with np.load(f"data/interim/lee2019_four_rest_raw/session{session}/sub-{target:02d}.npz",allow_pickle=False) as d:
        pre=d["pre_train"]; trans=int(d["transient_samples"])
    ae=np.arange(trans+250,len(pre)+1,25)
    with np.load(f"data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{target:02d}.npz",allow_pickle=False) as d:
        sig,events=d["signal"],d["events"]
    ends=np.arange(250,len(sig)+1,25)
    return windows(pre,ae),windows(sig,ends),ends,events


def auc(pos,neg):
    v=np.r_[pos,neg]; r=rankdata(v); n=len(pos)
    return float((r[:n].sum()-n*(n+1)/2)/(n*len(neg)))


def tune(anchor):
    z=(anchor-anchor.mean())/max(anchor.std(ddof=1),1e-8)
    p=tune_source_operating_points([calibration(z)],"ma_dwell",list(TARGETS),STEP,ma_seconds=.5,dwell_seconds=.2,refractory_seconds=1.,grid_size=160)
    return {k:float(v.threshold) for k,v in p.items()}


def metric_at(ma,thr,allowed,evalmask,idle,exposure,gt,times,onsets):
    alarm=alarms_from_ma(ma,thr,allowed)&evalmask
    fp=float(np.sum(alarm&idle)/(exposure/60))
    sen={str(d):float(np.mean([np.any(alarm&(times>=o)&(times<=o+d)) for o in onsets])) for d in DELTAS}
    pred=intervals_from_alarm(times,alarm,ma,thr,allowed&evalmask)
    loose=interval_metrics(pred,gt,.1,.1); strict=interval_metrics(pred,gt,.7,.7)
    return {"actual_fp":fp,**sen,"coverage":loose["coverage"],"temporal_iou":loose["temporal_iou"],"strict_f1":strict["intersection_f1"]}


def analyze(session,target,epochs,device,cache,width,nshift):
    started=time.perf_counter(); model,hist=train_loso(cache,target,epochs,device,width); training=time.perf_counter()-started
    aw,sw,ends,events=anchor_stream(session,target); anchor=predict(model,aw,device); raw=predict(model,sw,device)
    times=ends/SFREQ; starts=(ends-int(WINDOW*SFREQ))/SFREQ
    seams=np.unique(np.r_[events[:,4],events[1:,2]])/SFREQ; guard=np.zeros(len(times),bool)
    for seam in seams: guard|=(starts<seam+3)&(times>seam)
    allowed=~guard; z=ewma_z(raw,float(anchor.mean()),float(anchor.var(ddof=1)),allowed); ma=ma_trace(z); thresholds=tune(anchor); cues=events[:,5]/SFREQ
    flat=[]; margin=[]
    for cue in cues[:SCREEN]:
        valid=(times>=cue+1.5)&(times<cue+4)&allowed; q=ma[valid&np.isfinite(ma)]
        flat.append(not np.any(q>=thresholds[2.])); margin.append(float(np.max(q)-thresholds[2.]) if len(q) else -np.inf)
    onsets=cues[SCREEN:]+.5; stops=cues[SCREEN:]+4; evalmask=times>=events[SCREEN,2]/SFREQ
    state=np.zeros(len(times),bool)
    for a,b in zip(onsets,stops): state|=(times>=a)&(times<=b)
    idle=evalmask&allowed&~state; gt=np.c_[onsets,stops]
    surface={str(fp):metric_at(ma,thresholds[fp],allowed,evalmask,idle,idle.sum()*STEP,gt,times,onsets) for fp in TARGETS}
    rng=np.random.default_rng(SEED+1000*session+target); null=[]
    for shift in rng.integers(200,len(z)-200,nshift):
        null.append(metric_at(ma_trace(np.roll(z,int(shift))),thresholds[4.],allowed,evalmask,idle,idle.sum()*STEP,gt,times,onsets)["2.0"])
    surface["4.0"]["circular_null_2.0"]=float(np.mean(null))
    surface["4.0"]["excess_2.0"]=float(surface["4.0"]["2.0"]-np.mean(null))
    pos=np.concatenate([raw[(starts>=a)&(times<=b)&allowed] for a,b in gt]); neg=raw[idle]
    return {"subject":target,"session":session,"screen_flat_fraction":float(np.mean(flat)),"screen_margin_mean":float(np.mean(margin)),"window_auc":auc(pos,neg),"surface":surface,"training_seconds":training,"loss_start":hist[0],"loss_end":hist[-1]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--subjects",type=int,nargs="*",default=[1]);p.add_argument("--sessions",type=int,nargs="*",default=[1]);p.add_argument("--epochs",type=int,default=5);p.add_argument("--width",type=int,default=16);p.add_argument("--nshift",type=int,default=100);p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_causal_unet_loso_smoke.json"));a=p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for s in a.sessions:
        cache=load_source_cache(s)
        for u in a.subjects:
            row=analyze(s,u,a.epochs,device,cache,a.width,a.nshift);rows.append(row);print(f"session{s} sub-{u:02d}: {row['surface']['4.0']}",flush=True)
    payload={"protocol":{"model":"strictly causal 1-D U-Net segmentation","exact_loso":True,"epochs":a.epochs,"width":a.width,"source_sequences_per_class_per_subject":PER_CLASS,"sequence_seconds":CHUNK/SFREQ,"score":"last sample logit of causal 1-s window","device":str(device),"screening":"first 10 trials; full cohort ranking only in full run"},"rows":rows}
    a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
