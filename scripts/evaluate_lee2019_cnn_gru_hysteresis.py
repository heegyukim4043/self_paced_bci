"""Release-threshold hysteresis on cached CNN-GRU + EWMA streams."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.event_evaluation import interval_metrics, predicted_intervals
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS, HALF_LIFE_SECONDS, MA_SECONDS, REFRACTORY_SECONDS, STEP,
    normalize_anchor, tune_ma_dwell,
)
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import interval_mask, metrics
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration

RELEASE_DROPS = (0.0, 0.25, 0.5, 1.0, 1.5)

def causal_ma(values, seconds=MA_SECONDS):
    n = max(1, int(np.ceil(seconds / STEP)))
    result = np.convolve(np.asarray(values, float), np.ones(n)/n, mode="full")[:len(values)]
    result[:n-1] = np.nan
    return result

def active_mask(times, intervals):
    return interval_mask(times, intervals)

def boot(values, seed, repeats=20000):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()

def analyze(path, subject, session, drops, shifts):
    sequence, blocks = load_cache(path)
    mean, variance, anchor_z = normalize_anchor(blocks)
    point = tune_ma_dwell([calibration(x) for x in anchor_z], 160)
    decision = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=point.threshold, step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS, update_mode="all",
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=.25,
    )
    statistic = causal_ma(decision.statistic)
    idle = interval_mask(sequence.times, sequence.idle_intervals)
    gt = np.asarray(sequence.command_intervals, float)
    rng = np.random.default_rng(202609080 + 1000*session + subject)
    offsets = rng.integers(1, len(sequence.times), shifts)
    onset = metrics(sequence, decision.alarms, offsets)
    arms = {}
    for drop in drops:
        pred = predicted_intervals(
            sequence.times, decision.alarms, statistic, point.threshold,
            release_drop=drop, release_hold_seconds=.3, merge_gap_seconds=.3,
        )
        tolerant = interval_metrics(pred, gt, .1, .1)
        strict = interval_metrics(pred, gt, .7, .7)
        active = active_mask(sequence.times, pred)
        arms[str(drop)] = {
            "trigger_threshold": float(point.threshold),
            "release_threshold": float(point.threshold-drop),
            "fp_per_min": onset["fp_per_min"],
            "sen1": onset["observed_sen1"], "null1": onset["null_sen1"],
            "excess1": onset["excess_sen1"],
            "predicted_intervals": int(len(pred)),
            "coverage": tolerant.coverage, "temporal_iou": tolerant.temporal_iou,
            "fragmentation": tolerant.fragmentation,
            "tolerant_f1": tolerant.f1, "strict_f1": strict.f1,
            "idle_false_active_occupancy": float(np.mean(active[idle])) if np.any(idle) else float("nan"),
        }
    return {"subject": subject, "session": session, "arms": arms}

def summarize(rows, drops):
    output = {}
    for session in (1, 2):
        selected = [r for r in rows if r["session"] == session]; base = str(drops[0])
        output[str(session)] = {"n": len(selected), "arms": {}}
        for index, drop in enumerate(drops):
            key = str(drop); points = [r["arms"][key] for r in selected]; item = {}
            for metric in ("fp_per_min", "sen1", "excess1", "coverage", "temporal_iou", "fragmentation", "strict_f1", "idle_false_active_occupancy"):
                vals = [p[metric] for p in points]
                item[metric+"_median"] = float(np.median(vals))
                item[metric+"_ci95"] = boot(vals, 20260908 + 100*session + 10*index + len(metric))
            if drop != drops[0]:
                for metric in ("coverage", "temporal_iou", "strict_f1", "idle_false_active_occupancy"):
                    delta = [r["arms"][key][metric]-r["arms"][base][metric] for r in selected]
                    item["paired_delta_"+metric+"_median"] = float(np.median(delta))
                    item["paired_delta_"+metric+"_ci95"] = boot(delta, 20261008 + 100*session + 10*index + len(metric))
            output[str(session)]["arms"][key] = item
    return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    p.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    p.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    p.add_argument("--drops", type=float, nargs="+", default=RELEASE_DROPS)
    p.add_argument("--shifts", type=int, default=500)
    p.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_hysteresis_subset20.json"))
    a = p.parse_args(); rows = []
    for session in a.sessions:
        for subject in a.subjects:
            rows.append(analyze(a.cnn_root/f"session{session}"/f"sub-{subject:02d}.npz", subject, session, tuple(a.drops), a.shifts))
    payload = {"protocol": {"model": "frozen CNN-GRU + all-score EWMA H30", "entry_alarms": "fixed across arms", "release_drops": a.drops, "release_hold_seconds": .3, "merge_gap_seconds": .3, "primary_comparison": True, "spacing_secondary": "run only after method selection"}, "summary": summarize(rows, tuple(a.drops)), "rows": rows}
    a.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))

if __name__ == "__main__": main()
