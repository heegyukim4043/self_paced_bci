"""Rest-only null for direction-aligned sequential feature trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path,load_lee2019_run
from scripts.screen_lee2019_sequential_features import (
    REL_ENDS,car_select,downsample,extract_features,peak_mu,
)


def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); q=p[order]*len(p)/np.arange(1,len(p)+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; out=np.empty(len(p)); out[order]=np.minimum(q,1)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",type=Path,default=Path("data/raw/lee2019_mi"))
    ap.add_argument("--screen",type=Path,default=Path("data/processed/lee2019_sequential_feature_screen.json"))
    ap.add_argument("--output",type=Path,default=Path("data/processed/lee2019_directed_feature_rest_null.json"))
    ap.add_argument("--figure",type=Path,default=Path("data/processed/lee2019_directed_feature_rest_null.png"))
    a=ap.parse_args(); screen=json.loads(a.screen.read_text()); config=ExperimentConfig()
    by_subject={int(r["subject"]):r for r in screen["subjects"]}; rows=[]
    baseline=(REL_ENDS>=-.5)&(REL_ENDS<=0); mi=(REL_ENDS>=1.5)&(REL_ENDS<=3.5)
    for idx,subject in enumerate(sorted(by_subject),1):
        train=load_lee2019_run(lee2019_path(a.data_root,subject,1),"train")
        test=load_lee2019_run(lee2019_path(a.data_root,subject,1),"test")
        selected=config.sensorimotor_channels
        pre=downsample(car_select(train.pre_rest,train.channels,selected),train.sfreq)
        mu=peak_mu(pre,selected); pre_features=extract_features(pre,selected,mu)
        epochs=[]
        for native,fs in ((train.post_rest,train.sfreq),(test.pre_rest,test.sfreq),(test.post_rest,test.sfreq)):
            rest=downsample(car_select(native,train.channels,selected),fs)
            guard=int(2*250); rest=rest[guard:-guard]
            length=int(6*250)
            epochs.extend(rest[start:start+length] for start in range(0,len(rest)-length+1,length))
        null_features=[extract_features(epoch,selected,mu) for epoch in epochs]
        features={}
        for name,record in by_subject[subject]["features"].items():
            center=float(np.median(pre_features[name])); scale=max(float(np.median(np.abs(pre_features[name]-center)))*1.4826,1e-8)
            direction=float(record["direction"])
            null=np.asarray([direction*(f[name]-center)/scale for f in null_features])
            null_curve=null.mean(0); observed=np.asarray(record["trajectory"])
            features[name]={
                "observed_delta_z":float(observed[mi].mean()-observed[baseline].mean()),
                "rest_null_delta_z":float(null_curve[mi].mean()-null_curve[baseline].mean()),
                "observed_trajectory":observed.tolist(),"rest_null_trajectory":null_curve.tolist(),
            }
        rows.append({"subject":subject,"rest_epochs":len(epochs),"features":features})
        print(f"directed null {idx}/{len(by_subject)} subject={subject}",flush=True)
    names=list(rows[0]["features"]); summary={}; pvalues=[]
    for name in names:
        observed=np.asarray([r["features"][name]["observed_delta_z"] for r in rows])
        null=np.asarray([r["features"][name]["rest_null_delta_z"] for r in rows]); delta=observed-null
        pv=float(wilcoxon(delta,alternative="greater").pvalue); pvalues.append(pv)
        summary[name]={"observed_delta_z_median":float(np.median(observed)),
                       "rest_null_delta_z_median":float(np.median(null)),
                       "observed_minus_null_median":float(np.median(delta)),
                       "wilcoxon_p_one_sided":pv,
                       "observed_curve":np.mean([r["features"][name]["observed_trajectory"] for r in rows],0).tolist(),
                       "null_curve":np.mean([r["features"][name]["rest_null_trajectory"] for r in rows],0).tolist()}
    for name,q in zip(names,bh(pvalues)):summary[name]["bh_q"]=float(q)
    ordered=sorted(names,key=lambda n:summary[n]["observed_minus_null_median"],reverse=True)
    payload={"ranking":ordered,"summary":summary,"rows":rows}; a.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(4,3,figsize=(13,11),sharex=True)
    for ax,name in zip(axes.flat,ordered[:12]):
        ax.plot(REL_ENDS,summary[name]["observed_curve"],color="#16833b",label="nonresponse trials")
        ax.plot(REL_ENDS,summary[name]["null_curve"],color="0.35",label="rest-only null")
        ax.axvspan(1.5,3.5,color="#a40018",alpha=.1);ax.axvline(0,color="k",ls="--",lw=.7)
        ax.set_title(f"{name}\nobs-null={summary[name]['observed_minus_null_median']:.3f}, q={summary[name]['bh_q']:.3g}",fontsize=9)
    axes.flat[0].legend(fontsize=8); fig.tight_layout();fig.savefig(a.figure,dpi=180);plt.close(fig)
    print(json.dumps({n:summary[n] for n in ordered[:12]},indent=2))


if __name__=="__main__":main()
