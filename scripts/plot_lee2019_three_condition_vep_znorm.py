"""PreTrainRest-SD normalized three-condition ERP plots and statistics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["_MNE_FAKE_HOME_DIR"] = tempfile.mkdtemp(prefix="mne-znorm-vep-")
import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src"))
from uncued_mi.data import lee2019_path, load_lee2019_run
from scripts.analyze_lee2019_rest_trial_stft_clusters import ROIS
from scripts.compare_lee2019_three_condition_vep import cluster_test


def rest_scale(path):
    run = load_lee2019_run(path, "train")
    values = run.pre_rest.astype(np.float64)
    values -= values.mean(axis=1, keepdims=True)
    sos = butter(4, (1., 30.), btype="bandpass", fs=run.sfreq, output="sos")
    values = sosfiltfilt(sos, values, axis=0)
    guard = int(round(2*run.sfreq))
    values = values[guard:-guard]
    scale = values.std(axis=0, ddof=1)
    # Convert the scale to microvolts exactly as for the ERP caches.
    if np.nanpercentile(np.abs(values), 99) < .01:
        scale *= 1e6
    return np.maximum(scale, 1e-9), run.channels


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--condition-cache-root",type=Path,default=Path("data/interim/lee2019_three_condition_vep_1_30_extended/session1"))
    parser.add_argument("--scale-cache-root",type=Path,default=Path("data/interim/lee2019_pretrain_rest_scale_1_30/session1"))
    parser.add_argument("--subjects",type=int,nargs="+",default=list(range(1,55)))
    parser.add_argument("--session",type=int,default=1)
    parser.add_argument("--permutations",type=int,default=2048)
    parser.add_argument("--output",type=Path,default=Path("data/processed/lee2019_three_condition_vep_1_30_znorm_roi.png"))
    parser.add_argument("--oz-output",type=Path,default=Path("data/processed/lee2019_oz_cue_vs_rest_1_30_znorm.png"))
    parser.add_argument("--topography",type=Path,default=Path("data/processed/lee2019_three_condition_vep_1_30_znorm_topography.png"))
    parser.add_argument("--summary",type=Path,default=Path("data/processed/lee2019_three_condition_vep_1_30_znorm.json"))
    args=parser.parse_args()

    rows={"far pre-cue":[],"resting":[],"task cue":[]}; scales=[]
    times=channels=None
    for subject in args.subjects:
        condition_path=args.condition_cache_root/f"sub-{subject:02d}.npz"
        with np.load(condition_path,allow_pickle=False) as d:
            far,rest,task=d["far"],d["rest"],d["task"]
            current_times=d["times"]; current_channels=tuple(d["channels"].tolist())
        scale_path=args.scale_cache_root/f"sub-{subject:02d}.npz"
        if scale_path.exists():
            with np.load(scale_path,allow_pickle=False) as d: scale=d["scale"]
        else:
            scale,scale_channels=rest_scale(lee2019_path(args.data_root,subject,args.session))
            if tuple(scale_channels)!=current_channels: raise ValueError("Scale channel order differs")
            scale_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(scale_path,scale=scale,channels=np.asarray(scale_channels))
        if times is None: times,channels=current_times,current_channels
        divisor=scale[:,None]
        rows["far pre-cue"].append(far/divisor); rows["resting"].append(rest/divisor); rows["task cue"].append(task/divisor)
        scales.append(scale); print(f"z-erp session={args.session} subject={subject}",flush=True)
    conditions={name:np.asarray(values) for name,values in rows.items()}

    contrasts=(("task cue","far pre-cue"),("task cue","resting"),("far pre-cue","resting")); tests={}
    for index,(left,right) in enumerate(contrasts):
        tests[f"{left} - {right}"]=cluster_test(conditions[left],conditions[right],times,list(channels),args.permutations,20260803+index,evaluation_stop=1.5)

    display=(times>=-.2)&(times<=1.5); palette={"far pre-cue":"#2563eb","resting":"#6b7280","task cue":"#dc2626"}
    fig,axes=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
    for axis,(roi,names) in zip(axes.flat,ROIS.items()):
        indices=[list(channels).index(name) for name in names]
        for name,values in conditions.items():
            trace=values[:,indices].mean(1); mean=trace.mean(0); sem=trace.std(0,ddof=1)/np.sqrt(len(trace))
            axis.plot(times[display],mean[display],color=palette[name],lw=2,label=name)
            axis.fill_between(times[display],(mean-sem)[display],(mean+sem)[display],color=palette[name],alpha=.15)
        axis.axvline(0,color="black",ls="--",lw=.8); axis.axhline(0,color=".5",lw=.7)
        axis.set(title=roi,xlabel="Local time (s)",ylabel="Amplitude (preTrainRest SD)"); axis.legend(frameon=False)
    fig.suptitle("Lee2019 three-condition ERP, subject/channel z-normalized by preTrainRest SD")
    args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=220,bbox_inches="tight"); plt.close(fig)

    # Requested Oz cue-versus-rest view with significant cluster spans touching Oz.
    oz=list(channels).index("Oz"); fig,axis=plt.subplots(figsize=(7,6),constrained_layout=True)
    for name,color,label_name in (("task cue","#2563eb","Cue"),("resting","#6b7280","Rest pseudo-event")):
        trace=conditions[name][:,oz]; mean=trace.mean(0); sem=trace.std(0,ddof=1)/np.sqrt(len(trace))
        axis.plot(times[display],mean[display],color=color,lw=2,label=label_name)
        axis.fill_between(times[display],(mean-sem)[display],(mean+sem)[display],color=color,alpha=.18)
    task_rest=tests["task cue - resting"]
    for record in task_rest:
        if record["p"]<.05/3 and "Oz" in record["channels"]:
            axis.axvspan(record["start_ms"]/1000,record["stop_ms"]/1000,color="#ef4444",alpha=.12)
    axis.axvline(0,color="black",ls="--",lw=.8); axis.axhline(0,color=".5",lw=.7)
    axis.set(xlim=(-.2,1.5),title="Oz: Cue vs rest (preTrainRest-SD normalized)",xlabel="Time from condition onset (s)",ylabel="Amplitude (z)")
    axis.legend(frameon=False); fig.savefig(args.oz_output,dpi=220,bbox_inches="tight"); plt.close(fig)

    info=mne.create_info(list(channels),250.,"eeg"); info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    latencies=(.1,.2,.5,1.,1.5); fig,axes=plt.subplots(3,5,figsize=(17,10),constrained_layout=True)
    vals=[v.mean(0)[:,np.argmin(abs(times-t))] for v in conditions.values() for t in latencies]; vmax=np.percentile(np.abs(np.concatenate(vals)),98); image=None
    for row,(name,values) in enumerate(conditions.items()):
        grand=values.mean(0)
        for col,t in enumerate(latencies):
            idx=np.argmin(abs(times-t)); image,_=mne.viz.plot_topomap(grand[:,idx],info,axes=axes[row,col],show=False,cmap="RdBu_r",vlim=(-vmax,vmax),sensors=True)
            axes[row,col].set_title(f"{name}: {t:g} s")
    fig.colorbar(image,ax=axes,shrink=.65,label="Amplitude (preTrainRest SD)"); fig.suptitle("Subject-normalized scalp topographies")
    fig.savefig(args.topography,dpi=220,bbox_inches="tight"); plt.close(fig)

    payload={"subjects":len(args.subjects),"normalization":"per-subject, per-channel preTrainRest 1-30 Hz SD; same divisor for all three conditions",
             "scale_summary_uv":{"median":float(np.median(scales)),"iqr":[float(np.quantile(scales,.25)),float(np.quantile(scales,.75))]},
             "pairwise_cluster_tests":tests,"planned_contrast_bonferroni_alpha":.05/3}
    args.summary.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
