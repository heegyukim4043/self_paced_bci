"""Locked supervised short-context tangent-TCN validation."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.riemannian import tangent_features, window_covariances
from uncued_mi.swpc import balanced_gate_loss, seed_everything
from uncued_mi.tangent_tcn import TangentTCN

SUBSET20=(2,3,9,10,12,13,14,17,19,20,23,25,26,29,38,40,42,51,53,54)
SFREQ, WINDOW, STEP, CONTEXT=250,250,.5,3


def sequences(features,times,intervals):
    out=[]
    for start,stop in intervals:
        idx=np.flatnonzero((times-WINDOW/SFREQ>=start-1e-9)&(times<=stop+1e-9))
        for j in range(CONTEXT-1,len(idx)):
            choice=idx[j-CONTEXT+1:j+1]
            if np.all(np.diff(choice)==1): out.append(features[choice])
    return np.asarray(out,dtype=np.float32)


def train(pos,rest,iti,seed,epochs):
    seed_everything(seed); model=TangentTCN(pos.shape[2])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    p,r,i=map(torch.from_numpy,(pos,rest,iti)); g=torch.Generator().manual_seed(seed)
    def draw(x): return x[torch.randint(len(x),(32,),generator=g)]
    for _ in range(epochs):
        model.train()
        for _ in range(max(8,int(np.ceil(len(pos)/64)))):
            opt.zero_grad(set_to_none=True)
            loss=balanced_gate_loss(model(draw(p)),model(draw(r)),model(draw(i)))
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def stream_scores(model,features):
    x=np.stack([features[j-CONTEXT+1:j+1] for j in range(CONTEXT-1,len(features))]).astype(np.float32)
    values=[]
    model.eval()
    for start in range(0,len(x),256): values.append(model(torch.from_numpy(x[start:start+256])).numpy())
    scores=np.full(len(features),np.nan); scores[CONTEXT-1:]=np.concatenate(values)
    scores[:CONTEXT-1]=np.nanmedian(scores)
    return scores


def evaluate(times,scores,idle,command,seed,shifts):
    seq=ScoredSequence(times,scores,idle,command)
    point=tune_source_operating_points([seq],"ma_dwell",[2.],STEP,ma_seconds=.5,dwell_seconds=.3,refractory_seconds=1.,grid_size=160)[2.]
    dec=apply_decision(seq,"ma_dwell",point.threshold,STEP,ma_seconds=.5,dwell_seconds=.3,refractory_seconds=1.)
    obs=evaluate_alarms(times,dec.alarms,idle,command); rng=np.random.default_rng(seed)
    null=[evaluate_alarms(times,np.roll(dec.alarms,int(o)),idle,command).true_positive_rate for o in rng.integers(1,len(dec.alarms),shifts)]
    return {"fp_per_minute":obs.fp_per_minute,"observed_tpr":obs.true_positive_rate,"null_tpr":float(np.median(null)),"excess_tpr":float(obs.true_positive_rate-np.median(null)),"latency":obs.median_delay_seconds}


def analyze(path,session,subject,seeds,epochs,shifts):
    with np.load(path) as d:
        signal,pre=d["signal"],d["pre_rest"]; step=int(STEP*SFREQ)
        pre_ends=np.arange(WINDOW,len(pre)+1,step)
        ref=window_covariances(pre,pre_ends,WINDOW).mean(0); ref+=np.eye(len(ref))*np.trace(ref)/len(ref)*1e-6
        ends=np.arange(3*SFREQ,len(signal)+1,step); times=ends/SFREQ
        feat=tangent_features(window_covariances(signal,ends,WINDOW),ref)
        prefeat=tangent_features(window_covariances(pre,pre_ends,WINDOW),ref); pretimes=pre_ends/SFREQ
        fitcmd=d["command_intervals"][:50]; fititi=d["iti_intervals"][d["iti_trials"]<50]
        scaler=StandardScaler().fit(np.r_[sequences(feat,times,fitcmd).reshape(-1,feat.shape[1]),sequences(feat,times,fititi).reshape(-1,feat.shape[1]),prefeat])
        feat=scaler.transform(feat); prefeat=scaler.transform(prefeat)
        pos=sequences(feat,times,fitcmd); iti=sequences(feat,times,fititi); rest=sequences(prefeat,pretimes,np.asarray([[0.,len(pre)/SFREQ]]))
        idle=d["iti_intervals"][(d["iti_trials"]>=50)&(d["iti_trials"]<100)]; cmd=d["command_intervals"][50:100]
    metrics=[]
    for seed in seeds:
        model=train(pos,rest,iti,seed,epochs)
        metrics.append(evaluate(times,stream_scores(model,feat),idle,cmd,seed+session*1000+subject,shifts))
    return {"session":session,"subject":subject,"seeds":metrics,"counts":{"positive":len(pos),"rest":len(rest),"iti":len(iti)}}


def boot(v,seed,n):
    g=np.random.default_rng(seed); x=v[g.integers(0,len(v),(n,len(v)))]
    return np.quantile(np.median(x,1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache-root",type=Path,default=Path("data/interim/lee2019_mil_raw")); p.add_argument("--subjects",type=int,nargs="+",default=SUBSET20); p.add_argument("--sessions",type=int,nargs="+",default=[1,2]); p.add_argument("--seeds",type=int,nargs="+",default=SEEDS); p.add_argument("--epochs",type=int,default=30); p.add_argument("--shifts",type=int,default=500); p.add_argument("--bootstrap",type=int,default=10000); p.add_argument("--smoke",action="store_true"); p.add_argument("--output",type=Path,default=Path("data/processed/tangent_tcn_validation.json")); a=p.parse_args()
    if a.smoke:a.subjects,a.sessions,a.seeds,a.epochs,a.shifts,a.bootstrap=a.subjects[:1],[1],a.seeds[:1],2,20,100
    rows=[]
    for s in a.sessions:
        for u in a.subjects:
            rows.append(analyze(a.cache_root/f"session{s}"/f"sub-{u:02d}_mil_raw.npz",s,u,a.seeds,a.epochs,a.shifts)); print(f"done session={s} subject={u}",flush=True)
    comp=json.loads(Path("data/processed/mil_riemannian_comparator.json").read_text()); summary=[]
    for s in a.sessions:
        z=[r for r in rows if r["session"]==s]; rm={r["subject"]:r["excess_tpr"] for r in comp["rows"] if r["session"]==s}
        v=np.asarray([np.mean([m["excess_tpr"] for m in r["seeds"]]) for r in z]); b=np.asarray([rm[r["subject"]] for r in z])
        summary.append({"session":s,"excess_median":float(np.median(v)),"excess_ci":boot(v,100+s,a.bootstrap),"minus_riemannian_median":float(np.median(v-b)),"minus_riemannian_ci":boot(v-b,200+s,a.bootstrap),"fp_median":float(np.median([np.mean([m["fp_per_minute"] for m in r["seeds"]]) for r in z]))})
    a.output.write_text(json.dumps({"protocol":{"preregistered":True,"context_points":3,"epochs":a.epochs,"seeds":list(a.seeds)},"summary":summary,"rows":rows},indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
