"""Wald-threshold SPRT smoke test on calibrated Gaussian score LLRs."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_lee2019_sequential_onset import (
    RESTS, STEP, fit_score, onset_rates, reference,
)
from uncued_mi.hsmm import GaussianEmission, _normal_logpdf
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features


def wald_alarms(scores, emission, target_fp, beta=.2, refractory_seconds=1.):
    llr=_normal_logpdf(scores,emission.mi_mean,emission.mi_std)-_normal_logpdf(scores,emission.idle_mean,emission.idle_std)
    alpha=min(.99,target_fp*STEP/60.)
    upper=float(np.log((1-beta)/alpha));lower=float(np.log(beta/(1-alpha)))
    alarms=np.zeros(len(scores),bool);state=0.;blocked=-1;refractory=int(np.ceil(refractory_seconds/STEP))
    for i,value in enumerate(llr):
        if i<=blocked:continue
        state+=value
        if state>=upper:alarms[i]=True;state=0.;blocked=i+refractory
        elif state<=lower:state=0.
    return alarms,upper,lower


def analyze(path,subject,session,target_fp=4.,shifts=100):
    with np.load(path,allow_pickle=False) as d:
        pre=d['pre_train_covariance'];split=len(pre)//2;ref=reference(pre[:split])
        command=tangent_features(d['train_command_covariance'],ref);iti=tangent_features(d['train_idle_iti_covariance'],ref)
        rest={n:tangent_features(d[f'{n}_covariance'],ref) for n in RESTS};ct=d['train_command_trial'];it=d['train_idle_iti_trial']
        scaler,model=fit_score(command[ct<50],iti[it<50],rest['pre_train'][:split]);score=lambda x:model.decision_function(scaler.transform(x))
        emission=GaussianEmission.fit(np.r_[score(iti[it<50]),score(rest['pre_train'][:split])],score(command[ct<50]))
        rest_scores={n:score(x) for n,x in rest.items()};evaluation=[v[len(v)//2:] for v in rest_scores.values()]
        cs=score(command);trials=[cs[ct==trial] for trial in range(50,100) if np.any(ct==trial)]
    stream,_=build_rest_injected_stream(evaluation,trials,step_seconds=STEP,minimum_rest_seconds=5,maximum_rest_seconds=11,guard_seconds=1,seed=20260728+100*session+subject)
    alarms,upper,lower=wald_alarms(stream.scores,emission,target_fp);conv=evaluate_alarms(stream.times,alarms,stream.idle_intervals,stream.command_intervals);obs,delay=onset_rates(stream.times,alarms,stream.command_intervals)
    rng=np.random.default_rng(20260814+100*session+subject);null=[]
    for off in rng.integers(1,len(alarms),shifts):null.append(onset_rates(stream.times,np.roll(alarms,int(off)),stream.command_intervals)[0]['2.0'])
    null=float(np.median(null))
    return {'subject':subject,'session':session,'target_fp':target_fp,'alpha_per_step':target_fp*STEP/60,'beta':.2,'upper_A':upper,'lower_B':lower,'actual_fp':conv.fp_per_minute,'sensitivity_2s':obs['2.0'],'null_2s':null,'excess_2s':obs['2.0']-null,'delay':delay}


def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,default=43);p.add_argument('--session',type=int,default=1);a=p.parse_args();path=Path(f'data/interim/lee2019_four_rest/session{a.session}/sub-{a.subject:02d}.npz');row=analyze(path,a.subject,a.session);out=Path('data/processed/lee2019_wald_sprt_smoke.json');out.write_text(json.dumps(row,indent=2));print(json.dumps(row,indent=2))
if __name__=='__main__':main()
