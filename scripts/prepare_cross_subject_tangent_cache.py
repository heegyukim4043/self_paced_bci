"""Cache target-local tangent features for transfer/adaptation experiments."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from uncued_mi.riemannian import tangent_features,window_covariances

SUBSET20=(2,3,9,10,12,13,14,17,19,20,23,25,26,29,38,40,42,51,53,54)
SFREQ,WINDOW,STEP=250,250,125


def positives(d):
    out=[]
    for trial in range(50):
        size=int(d["positive_bag_sizes"][trial]); out.extend(d["positive_end_samples"][trial,:size])
    return np.asarray(out,dtype=int)


def cache_one(source,target):
    with np.load(source) as d:
        pre=d["pre_rest"]; pre_ends=np.arange(WINDOW,len(pre)+1,STEP)
        ref=window_covariances(pre,pre_ends,WINDOW).mean(0); ref+=np.eye(len(ref))*np.trace(ref)/len(ref)*1e-6
        pos=tangent_features(window_covariances(d["signal"],positives(d),WINDOW),ref).astype(np.float32)
        mask=d["iti_instance_trials"]<50; neg=tangent_features(window_covariances(d["signal"],d["iti_end_samples"][mask],WINDOW),ref).astype(np.float32)
        pre_x=tangent_features(window_covariances(pre,pre_ends,WINDOW),ref).astype(np.float32)
        ends=np.arange(3*SFREQ,len(d["signal"])+1,STEP)
        stream=tangent_features(window_covariances(d["signal"],ends,WINDOW),ref).astype(np.float32)
        payload=dict(pos=pos,neg=neg,pre=pre_x,stream=stream,times=ends/SFREQ,
                     iti_intervals=d["iti_intervals"],iti_trials=d["iti_trials"],
                     command_intervals=d["command_intervals"])
    target.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(target,**payload)


def main():
    p=argparse.ArgumentParser();p.add_argument("--raw-root",type=Path,default=Path("data/interim/lee2019_mil_raw"));p.add_argument("--output-root",type=Path,default=Path("data/interim/cross_subject_tangent"));p.add_argument("--subjects",type=int,nargs="+",default=SUBSET20);p.add_argument("--sessions",type=int,nargs="+",default=[1,2]);a=p.parse_args()
    for s in a.sessions:
        for u in a.subjects:
            target=a.output_root/f"session{s}"/f"sub-{u:02d}.npz"
            if not target.exists():cache_one(a.raw_root/f"session{s}"/f"sub-{u:02d}_mil_raw.npz",target)
            print(f"cached session={s} subject={u}",flush=True)
if __name__=="__main__":main()
