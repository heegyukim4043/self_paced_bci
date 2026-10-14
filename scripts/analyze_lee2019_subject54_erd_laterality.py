"""Offline physiological audit of early/late ERD laterality for Lee2019 sub-54 S1."""
from pathlib import Path
import json, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from uncued_mi.data import lee2019_path,load_lee2019_run

SUBJECT,SESSION=54,1; BANDS={'mu 8-13 Hz':(8,13),'beta 13-30 Hz':(13,30)}
run=load_lee2019_run(lee2019_path(ROOT/'data/raw/lee2019_mi',SUBJECT,SESSION),'train')
sf=float(run.sfreq); channels={x:i for i,x in enumerate(run.channels)}; picks=[channels['C3'],channels['C4']]
time=np.arange(round(-2*sf),round(6*sf))/sf; epochs=[]
for cue in run.cue_samples:
    ix=cue+np.arange(round(-2*sf),round(6*sf)); epochs.append(run.eeg[ix][:,picks].T)
epochs=np.asarray(epochs,float); baseline=(time>=-2)&(time<0); task=(time>=.5)&(time<4); post=(time>=4.5)&(time<6)
fig,axes=plt.subplots(2,3,figsize=(15,9)); summary={}
for row,(name,(lo,hi)) in enumerate(BANDS.items()):
    sos=butter(4,[lo,hi],btype='bandpass',fs=sf,output='sos'); filtered=sosfiltfilt(sos,epochs,axis=-1)
    power=np.abs(hilbert(filtered,axis=-1))**2; base=np.mean(power[...,baseline],axis=-1,keepdims=True)
    erd=10*np.log10(np.maximum(power,1e-20)/np.maximum(base,1e-20)); smooth=max(1,round(.2*sf));kernel=np.ones(smooth)/smooth
    erd=np.apply_along_axis(lambda x:np.convolve(x,kernel,mode='same'),-1,erd)
    trial=np.mean(erd[...,task],axis=-1); post_trial=np.mean(erd[...,post],axis=-1); labels=np.asarray(run.labels)
    # Authoritative MAT `class` field: label 1=right hand (contra C3),
    # label 2=left hand (contra C4). Positive laterality below means stronger
    # contralateral ERD: ipsilateral dB minus contralateral dB.
    laterality=np.where(labels==1,trial[:,1]-trial[:,0],trial[:,0]-trial[:,1])
    summary[name]={}
    for split,idx,color in [('first50',np.arange(50),'#2878b5'),('last50',np.arange(50,100),'#d95f02')]:
        li=laterality[idx]; contra=np.where(labels[idx]==1,trial[idx,0],trial[idx,1]);ipsi=np.where(labels[idx]==1,trial[idx,1],trial[idx,0])
        post_contra=np.where(labels[idx]==1,post_trial[idx,0],post_trial[idx,1]);post_ipsi=np.where(labels[idx]==1,post_trial[idx,1],post_trial[idx,0])
        summary[name][split]={'laterality_median_db':float(np.median(li)),'laterality_mean_db':float(np.mean(li)),'positive_laterality_fraction':float(np.mean(li>0)),'contra_erd_median_db':float(np.median(contra)),'ipsi_erd_median_db':float(np.median(ipsi)),'post_contra_ers_median_db':float(np.median(post_contra)),'post_ipsi_ers_median_db':float(np.median(post_ipsi))}
        axes[row,0].plot(time,np.mean(erd[idx,0],axis=0),color=color,lw=2,label=split)
        axes[row,1].plot(time,np.mean(erd[idx,1],axis=0),color=color,lw=2,label=split)
    positions=[1,2]; data=[laterality[:50],laterality[50:]]
    bp=axes[row,2].boxplot(data,positions=positions,widths=.55,patch_artist=True,showfliers=True)
    for patch,color in zip(bp['boxes'],['#2878b5','#d95f02']):patch.set_facecolor(color);patch.set_alpha(.55)
    axes[row,2].axhline(0,color='black',ls='--',lw=1);axes[row,2].set_xticks(positions,['first 50','last 50']);axes[row,2].set_ylabel('Ipsi-minus-contra ERD strength (dB)')
    for col,ch in enumerate(('C3','C4')):
        axes[row,col].axvspan(.5,4,color='#e98b9a',alpha=.12);axes[row,col].axvspan(4.5,6,color='#a6cee3',alpha=.12);axes[row,col].axhline(0,color='black',lw=.8);axes[row,col].set_title(f'{name}: {ch} ERD/ERS');axes[row,col].set_xlabel('Time from cue (s)');axes[row,col].set_ylabel('Power change (dB)');axes[row,col].grid(alpha=.2)
    axes[row,2].set_title(f'{name}: direction-corrected laterality');axes[row,2].grid(axis='y',alpha=.2)
axes[0,0].legend(frameon=False);fig.suptitle('Lee2019 sub-54 S1: early vs late physiological ERD/laterality\nOffline zero-phase audit; baseline -2 to 0 s, MI 0.5 to 4 s, post-MI 4.5 to 6 s',fontsize=15,fontweight='bold');fig.tight_layout(rect=(0,0,1,.93))
out=ROOT/'data/processed/lee2019_sub54_s1_erd_laterality_first_last.png';fig.savefig(out,dpi=190,bbox_inches='tight')
jout=ROOT/'data/processed/lee2019_sub54_s1_erd_laterality_first_last.json';jout.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));print(out)
