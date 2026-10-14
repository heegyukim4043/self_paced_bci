"""Deployment summaries: KM, coverage/performance, and dwell Pareto."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PERCENTILES=(.1,.2,.3,.4,.5,.6,.8,1.)
DWELLS=(.1,.2,.3)

def km(times,events):
    times=np.asarray(times,float);events=np.asarray(events,bool);survival=1.;xs=[0.];ys=[1.]
    for time in np.unique(times[events]):
        risk=np.sum(times>=time);deaths=np.sum(events&(times==time))
        if risk:survival*=1-deaths/risk
        xs.extend([float(time),float(time)]);ys.extend([ys[-1],survival])
    return np.asarray(xs),np.asarray(ys)

def km_stats(rows,key,ids,time_key):
    values=[rows[i]["results"][key] for i in ids];times=[v[time_key] for v in values];events=[v["first_fp_observed"] for v in values];x,y=km(times,events)
    below=np.flatnonzero(y<=.5);median=float(x[below[0]]) if len(below) else None
    survival={str(point):float(y[np.searchsorted(x,point,side="right")-1]) for point in (30.,60.,120.)}
    return {"n":len(ids),"events":int(sum(events)),"median_seconds":median,"survival_probability":survival,"curve":{"time":x.tolist(),"survival":y.tolist()}}

def selected(rows,percentile):
    values=np.asarray([row["screen_flat_fraction"] for row in rows.values()]);cut=float(np.quantile(values,percentile));ids={i for i,row in rows.items() if row["screen_flat_fraction"]<=cut};return cut,ids

def group_metrics(rows,ids,key):
    vals=[rows[i]["results"][key] for i in ids]
    return {"n":len(ids),"excess_median":float(np.median([v["excess"] for v in vals])),"actual_fp_median":float(np.median([v["actual_fp_per_min"] for v in vals])),"tpr_median":float(np.median([v["tpr"] for v in vals])),"latency_median_seconds":float(np.median([v["latency_seconds"] for v in vals if v["latency_seconds"] is not None]))}

def session_summary(rows):
    cutoff,chosen=selected(rows,.2);other=set(rows)-chosen;coverage=[]
    for percentile in PERCENTILES:
        cut,ids=selected(rows,percentile);coverage.append({"requested_percentile":percentile,"cutoff":cut,"coverage":len(ids)/len(rows),**group_metrics(rows,ids,"dwell0.2")})
    pareto=[]
    for dwell in DWELLS:pareto.append({"dwell":dwell,**group_metrics(rows,chosen,f"dwell{dwell:.1f}")})
    return {"relative20_cutoff":cutoff,"selected_ids":sorted(chosen),"selected":group_metrics(rows,chosen,"dwell0.2"),"not_selected":group_metrics(rows,other,"dwell0.2"),"km_idle_exposure":{"selected":km_stats(rows,"dwell0.2",chosen,"first_fp_idle_exposure_seconds"),"not_selected":km_stats(rows,"dwell0.2",other,"first_fp_idle_exposure_seconds")},"km_wallclock":{"selected":km_stats(rows,"dwell0.2",chosen,"first_fp_wallclock_seconds"),"not_selected":km_stats(rows,"dwell0.2",other,"first_fp_wallclock_seconds")},"coverage_curve":coverage,"dwell_pareto":pareto,"screen_wallclock_minutes":float(np.median([r["screen_wallclock_seconds"] for r in rows.values()])/60)}

def plot(result,path):
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    colors={"session1":"#2166ac","session2":"#b2182b"}
    for session in ("session1","session2"):
        summary=result[session]
        for group,style in (("selected","-"),("not_selected","--")):
            curve=summary["km_idle_exposure"][group]["curve"]
            axes[0].step(curve["time"],curve["survival"],where="post",color=colors[session],ls=style,label=f"{session} {group}")
        coverage=summary["coverage_curve"]
        axes[1].plot([x["coverage"] for x in coverage],[x["excess_median"] for x in coverage],marker="o",color=colors[session],label=session)
        pareto=summary["dwell_pareto"]
        axes[2].plot([x["latency_median_seconds"] for x in pareto],[x["excess_median"] for x in pareto],marker="o",color=colors[session],label=session)
        for point in pareto:axes[2].annotate(f"d={point['dwell']:.1f}",(point["latency_median_seconds"],point["excess_median"]),fontsize=8)
    axes[0].set(xlabel="Valid-idle exposure to first FP (s)",ylabel="FP-free probability",title="First-false-alarm KM");axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Selected session coverage",ylabel="Median excess TPR",title="Coverage-performance trade-off");axes[1].legend()
    axes[2].set(xlabel="Median detection latency (s)",ylabel="Median excess TPR",title="Selected-session dwell Pareto");axes[2].legend()
    fig.savefig(path,dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument("--session1",type=Path,default=Path("data/processed/lee2019_prospective_screening.json"));p.add_argument("--session2",type=Path,default=Path("data/processed/lee2019_prospective_screening_session2.json"));p.add_argument("--output",type=Path,default=Path("data/processed/lee2019_screening_deployment.json"));p.add_argument("--figure",type=Path,default=Path("data/processed/lee2019_screening_deployment.png"));a=p.parse_args();result={}
    for name,path in (("session1",a.session1),("session2",a.session2)):result[name]=session_summary({r["subject"]:r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]})
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8");plot(result,a.figure)
    concise={s:{"screen_minutes":v["screen_wallclock_minutes"],"selected":v["selected"],"km_idle":{g:{k:x for k,x in v["km_idle_exposure"][g].items() if k!="curve"} for g in ("selected","not_selected")},"pareto":v["dwell_pareto"]} for s,v in result.items()};print(json.dumps(concise,indent=2))
if __name__=="__main__":main()
