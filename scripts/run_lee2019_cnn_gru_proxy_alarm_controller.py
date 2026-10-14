"""Deployable time-proxy alarm-rate controller and EWMA 2x2 ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS, HALF_LIFE_SECONDS, MA_SECONDS, REFRACTORY_SECONDS, STEP,
    normalize_anchor, tune_ma_dwell,
)
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import metrics
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration

TARGET_FP = 2.0


def static_z(scores, mean, variance):
    return (np.asarray(scores, float) - mean) / np.sqrt(max(float(variance), 1e-12))


def proxy_controlled_decision(scores, initial_threshold, gain=.8,
                              proxy_gap_seconds=5., warmup_seconds=30.,
                              lower_offset=-1., upper_offset=3.):
    """Update only when the *prior* alarm-free gap makes the sample eligible."""
    scores = np.asarray(scores, float)
    ma_steps = max(1, int(np.ceil(MA_SECONDS / STEP)))
    dwell_steps = max(1, int(np.ceil(DWELL_SECONDS / STEP)))
    refractory_steps = max(0, int(np.ceil(REFRACTORY_SECONDS / STEP)))
    gap_steps = max(1, int(np.ceil(proxy_gap_seconds / STEP)))
    warmup_steps = max(0, int(np.ceil(warmup_seconds / STEP)))
    expected = TARGET_FP * STEP / 60.
    alarms = np.zeros(len(scores), bool)
    eligible = np.zeros(len(scores), bool)
    thresholds = np.empty(len(scores), float)
    ma = np.full(len(scores), np.nan)
    history, count, blocked_until, last_alarm, offset = [], 0, -1, -10**9, 0.
    for index, score in enumerate(scores):
        threshold = initial_threshold + offset
        thresholds[index] = threshold
        history.append(float(score))
        if len(history) > ma_steps:
            history.pop(0)
        if len(history) == ma_steps:
            ma[index] = float(np.mean(history))
        active = len(history) == ma_steps and ma[index] >= threshold
        count = count + 1 if active else 0
        preeligible = index >= warmup_steps and index > blocked_until and index-last_alarm >= gap_steps
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
            last_alarm = index
        eligible[index] = preeligible
        if preeligible:
            innovation = float(alarms[index]) - expected
            offset = float(np.clip(offset + gain * innovation,
                                   lower_offset, upper_offset))
    return alarms, thresholds, ma, eligible


def analyze(path, subject, session, gain, gap, warmup, shifts):
    sequence, blocks = load_cache(path)
    mean, variance, anchor_z = normalize_anchor(blocks)
    point = tune_ma_dwell([calibration(x) for x in anchor_z], 160)
    ewma = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=point.threshold, step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS, update_mode="all",
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=.25,
    )
    static = ewma_normalize_and_decide(
        static_z(sequence.scores, mean, variance), initial_mean=0., initial_variance=1.,
        threshold=point.threshold, step_seconds=STEP, half_life_seconds=1e12,
        update_mode="all", ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=.25,
    )
    rng = np.random.default_rng(980000 + session*1000 + subject)
    offsets = rng.integers(1, len(sequence.scores), shifts)
    arms = {}
    for norm_name, decision in (("static", static), ("ewma", ewma)):
        arms[f"{norm_name}_only"] = metrics(sequence, decision.alarms, offsets)
        alarms, thresholds, _, eligible = proxy_controlled_decision(
            decision.statistic, point.threshold, gain, gap, warmup)
        result = metrics(sequence, alarms, offsets)
        result.update({
            "threshold_initial": float(point.threshold),
            "threshold_final": float(thresholds[-1]),
            "threshold_offset_median": float(np.median(thresholds-point.threshold)),
            "proxy_eligible_fraction": float(np.mean(eligible)),
        })
        arms[f"{norm_name}_proxy"] = result
    return {"subject": subject, "session": session, "arms": arms}


def boot(values, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (20000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def summarize(rows, session):
    chosen = [x for x in rows if x["session"] == session]
    result = {"session": session, "n": len(chosen), "arms": {}}
    for arm_index, arm in enumerate(chosen[0]["arms"]):
        points = [x["arms"][arm] for x in chosen]
        item = {}
        for key in ("fp_per_min", "observed_sen1", "null_sen1", "excess_sen1"):
            vals = [p[key] for p in points]
            item[f"{key}_median"] = float(np.median(vals))
            item[f"{key}_ci95"] = boot(vals, 20262000+session*100+arm_index*10+len(key))
        item["under_count"] = sum(p["fp_per_min"] < 1.5 for p in points)
        item["admissible_count"] = sum(1.5 <= p["fp_per_min"] <= 2.5 for p in points)
        item["over_count"] = sum(p["fp_per_min"] > 2.5 for p in points)
        if arm.endswith("_proxy"):
            base = arm.replace("_proxy", "_only")
            delta = [x["arms"][arm]["excess_sen1"]-x["arms"][base]["excess_sen1"]
                     for x in chosen]
            item["paired_delta_excess_median"] = float(np.median(delta))
            item["paired_delta_excess_ci95"] = boot(delta, 20262100+session*10+arm_index)
            item["proxy_eligible_fraction_median"] = float(np.median(
                [p["proxy_eligible_fraction"] for p in points]))
        result["arms"][arm] = item
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    p.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    p.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    p.add_argument("--gain", type=float, default=.8)
    p.add_argument("--proxy-gap", type=float, default=5.)
    p.add_argument("--warmup", type=float, default=30.)
    p.add_argument("--shifts", type=int, default=500)
    p.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_proxy_alarm_controller.json"))
    args = p.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(args.cnn_root/f"session{session}"/f"sub-{subject:02d}.npz",
                                subject, session, args.gain, args.proxy_gap,
                                args.warmup, args.shifts))
            print(f"proxy controller session={session} subject={subject}", flush=True)
    summary = [summarize(rows, s) for s in args.sessions]
    payload = {"protocol": vars(args) | {
        "cnn_root": str(args.cnn_root), "output": str(args.output),
        "proxy_definition": "current sample eligible when prior 5 s are alarm/refractory-free; score-independent",
        "pseudo_uncued_warning": True,
    }, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
