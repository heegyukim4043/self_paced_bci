"""LOSO source pooling and pre-rest-only session adaptation validation."""

from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence,apply_decision,tune_source_operating_points

SUBSET20=(2,3,9,10,12,13,14,17,19,20,23,25,26,29,38,40,42,51,53,54)
STEP=.5


def load(path):
    with np.load(path) as d:return {k:np.asarray(d[k]) for k in d.files}


def idle_norm(data):
    mean=data["pre"].mean(0);std=data["pre"].std(0,ddof=1).clip(1e-4)
    return mean,std


def transformed_training(domains,adapt):
    xs=[];ys=[]
    for d in domains:
        x=np.r_[d["pos"],d["neg"]]; y=np.r_[np.ones(len(d["pos"])),np.zeros(len(d["neg"]))]
        if adapt:
            mean,std=idle_norm(d);x=(x-mean)/std
        xs.append(x);ys.append(y)
    return np.concatenate(xs),np.concatenate(ys)


def score(train_domains,target,adapt):
    x,y=transformed_training(train_domains,adapt)
    scaler=StandardScaler().fit(x); model=LogisticRegression(C=1.,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=SEEDS[0]).fit(scaler.transform(x),y)
    stream=target["stream"]
    if adapt:
        mean,std=idle_norm(target);stream=(stream-mean)/std
    return model.decision_function(scaler.transform(stream))


def evaluate(target,scores,seed,shifts):
    idle=target["iti_intervals"][(target["iti_trials"]>=50)&(target["iti_trials"]<100)];cmd=target["command_intervals"][50:100];times=target["times"]
    seq=ScoredSequence(times,scores,idle,cmd);point=tune_source_operating_points([seq],"ma_dwell",[2.],STEP,ma_seconds=.5,dwell_seconds=.3,refractory_seconds=1.,grid_size=160)[2.]
    dec=apply_decision(seq,"ma_dwell",point.threshold,STEP,ma_seconds=.5,dwell_seconds=.3,refractory_seconds=1.);obs=evaluate_alarms(times,dec.alarms,idle,cmd);rng=np.random.default_rng(seed)
    null=[evaluate_alarms(times,np.roll(dec.alarms,int(o)),idle,cmd).true_positive_rate for o in rng.integers(1,len(dec.alarms),shifts)]
    return {"fp_per_minute":obs.fp_per_minute,"observed_tpr":obs.true_positive_rate,"null_tpr":float(np.median(null)),"excess_tpr":float(obs.true_positive_rate-np.median(null)),"latency":obs.median_delay_seconds}


def boot(v,seed,n):
    g=np.random.default_rng(seed);x=v[g.integers(0,len(v),(n,len(v)))];return np.quantile(np.median(x,1),[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache-root",type=Path,default=Path("data/interim/cross_subject_tangent"));p.add_argument("--subjects",type=int,nargs="+",default=SUBSET20);p.add_argument("--sessions",type=int,nargs="+",default=[1,2]);p.add_argument("--shifts",type=int,default=500);p.add_argument("--bootstrap",type=int,default=10000);p.add_argument("--comparator",type=Path,default=Path("data/processed/mil_riemannian_comparator.json"));p.add_argument("--smoke",action="store_true");p.add_argument("--output",type=Path,default=Path("data/processed/cross_subject_session_adaptation.json"));a=p.parse_args()
    if a.smoke:a.subjects,a.sessions,a.shifts,a.bootstrap=a.subjects[:3],[1],20,100
    all_data={(s,u):load(a.cache_root/f"session{s}"/f"sub-{u:02d}.npz") for s in [1,2] for u in a.subjects}
    rows=[]
    for s in a.sessions:
        other=3-s
        for u in a.subjects:
            target=all_data[s,u];pooled=[all_data[s,v] for v in a.subjects if v!=u];self_source=[all_data[other,u]]
            arms={}
            for idx,(name,domains,adapt) in enumerate([
                ("pooled_ea",pooled,False),("pooled_adabn",pooled,True),
                ("cross_session_self_ea",self_source,False),("cross_session_self_adabn",self_source,True)]):
                arms[name]=evaluate(target,score(domains,target,adapt),SEEDS[0]+s*1000+u+idx*10,a.shifts)
            rows.append({"session":s,"subject":u,"arms":arms});print(f"done session={s} subject={u}",flush=True)
    comp=json.loads(a.comparator.read_text());summary=[]
    for s in a.sessions:
        z=[r for r in rows if r["session"]==s];rmap={r["subject"]:r["excess_tpr"] for r in comp["rows"] if r["session"]==s};arms={}
        for name in ("pooled_ea","pooled_adabn","cross_session_self_ea","cross_session_self_adabn"):
            v=np.asarray([r["arms"][name]["excess_tpr"] for r in z]);base=np.asarray([rmap[r["subject"]] for r in z])
            arms[name]={"excess_median":float(np.median(v)),"excess_ci":boot(v,100+s,a.bootstrap),"minus_target_specific_median":float(np.median(v-base)),"minus_target_specific_ci":boot(v-base,200+s,a.bootstrap)}
        pa=np.asarray([r["arms"]["pooled_adabn"]["excess_tpr"]-r["arms"]["pooled_ea"]["excess_tpr"] for r in z]);ca=np.asarray([r["arms"]["cross_session_self_adabn"]["excess_tpr"]-r["arms"]["cross_session_self_ea"]["excess_tpr"] for r in z])
        summary.append({"session":s,"arms":arms,"pooled_adabn_minus_ea":{"median":float(np.median(pa)),"ci":boot(pa,300+s,a.bootstrap)},"self_adabn_minus_ea":{"median":float(np.median(ca)),"ci":boot(ca,400+s,a.bootstrap)}})
    a.output.write_text(json.dumps({"protocol":{"preregistered":True,"subjects":list(a.subjects),"adaptation":"target pre-rest diagonal feature AdaBN only"},"summary":summary,"rows":rows},indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
