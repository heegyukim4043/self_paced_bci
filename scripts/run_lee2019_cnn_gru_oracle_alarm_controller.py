"""Oracle-idle upper bound for direct causal alarm-rate control.

The controller never changes EWMA moments.  It adjusts only a threshold offset
from false-alarm innovations observed in ground-truth idle.  Ground-truth idle
makes this diagnostic and non-deployable; a proxy arm is justified only if this
upper bound improves operating-point calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.metrics import evaluate_alarms
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS, HALF_LIFE_SECONDS, MA_SECONDS, REFRACTORY_SECONDS, STEP,
    normalize_anchor, tune_ma_dwell,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration
from scripts.run_lee2019_stateful_gru import onset_rates


def interval_mask(times, intervals):
    mask = np.zeros(len(times), bool)
    for start, stop in intervals:
        mask |= (times >= start) & (times < stop)
    return mask


def controlled_decision(scores, idle, initial_threshold, gain, target_fp,
                        lower=-3.0, upper=3.0):
    scores = np.asarray(scores, float)
    idle = np.asarray(idle, bool)
    ma_steps = max(1, int(np.ceil(MA_SECONDS / STEP)))
    dwell_steps = max(1, int(np.ceil(DWELL_SECONDS / STEP)))
    refractory_steps = max(0, int(np.ceil(REFRACTORY_SECONDS / STEP)))
    alarms = np.zeros(len(scores), bool)
    thresholds = np.empty(len(scores), float)
    ma = np.full(len(scores), np.nan)
    history = []
    count, blocked_until, offset = 0, -1, 0.0
    expected_per_idle_sample = target_fp * STEP / 60.0
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
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
        if idle[index]:
            innovation = float(alarms[index]) - expected_per_idle_sample
            offset = float(np.clip(offset + gain * innovation, lower, upper))
    return alarms, thresholds, ma


def shifted_null(times, alarms, commands, offsets):
    indices = (np.arange(len(alarms))[None, :] - offsets[:, None]) % len(alarms)
    shifted = alarms[indices]
    hits = np.zeros(len(offsets), float)
    for start, _ in commands:
        left = int(np.searchsorted(times, start, side="left"))
        right = int(np.searchsorted(times, start + 1.0, side="right"))
        hits += np.any(shifted[:, left:right], axis=1)
    return float(np.median(hits / len(commands)))


def metrics(sequence, alarms, offsets):
    conventional = evaluate_alarms(
        sequence.times, alarms, sequence.idle_intervals, sequence.command_intervals
    )
    observed, delay = onset_rates(sequence.times, alarms, sequence.command_intervals)
    null = shifted_null(sequence.times, alarms, sequence.command_intervals, offsets)
    return {
        "fp_per_min": float(conventional.fp_per_minute),
        "observed_sen1": float(observed["1.0"]),
        "null_sen1": null,
        "excess_sen1": float(observed["1.0"] - null),
        "delay": None if not np.isfinite(delay) else float(delay),
    }


def analyze(path, subject, session, gains, shifts):
    sequence, blocks = load_cache(path)
    mean, variance, normalized = normalize_anchor(blocks)
    point = tune_ma_dwell([calibration(x) for x in normalized], 160)
    baseline = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=point.threshold, step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS, update_mode="all",
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=0.25,
    )
    idle = interval_mask(sequence.times, sequence.idle_intervals)
    rng = np.random.default_rng(940000 + session * 1000 + subject)
    offsets = rng.integers(1, len(sequence.scores), shifts)
    arms = {"ewma_only": metrics(sequence, baseline.alarms, offsets)}
    for gain in gains:
        alarms, thresholds, _ = controlled_decision(
            baseline.statistic, idle, point.threshold, gain, 2.0
        )
        result = metrics(sequence, alarms, offsets)
        result.update({
            "threshold_initial": float(point.threshold),
            "threshold_final": float(thresholds[-1]),
            "threshold_offset_final": float(thresholds[-1] - point.threshold),
            "threshold_offset_median": float(np.median(thresholds - point.threshold)),
            "oracle_idle": True,
        })
        arms[f"oracle_gain_{gain:g}"] = result
    return {"subject": subject, "session": session, "arms": arms}


def boot(values, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (20000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, session):
    chosen = [row for row in rows if row["session"] == session]
    output = {"session": session, "n": len(chosen), "arms": {}}
    names = chosen[0]["arms"]
    for index, name in enumerate(names):
        points = [row["arms"][name] for row in chosen]
        item = {}
        for key in ("fp_per_min", "observed_sen1", "null_sen1", "excess_sen1"):
            values = [point[key] for point in points]
            item[f"{key}_median"] = float(np.median(values))
            item[f"{key}_ci95"] = boot(values, 20261200 + session * 100 + index * 10 + len(key))
        item["under_count"] = int(sum(p["fp_per_min"] < 1.5 for p in points))
        item["admissible_count"] = int(sum(1.5 <= p["fp_per_min"] <= 2.5 for p in points))
        item["over_count"] = int(sum(p["fp_per_min"] > 2.5 for p in points))
        if name != "ewma_only":
            deltas = [
                row["arms"][name]["excess_sen1"]
                - row["arms"]["ewma_only"]["excess_sen1"] for row in chosen
            ]
            item["paired_delta_excess_median"] = float(np.median(deltas))
            item["paired_delta_excess_ci95"] = boot(
                deltas, 20261300 + session * 100 + index
            )
        output["arms"][name] = item
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path,
                        default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--gains", type=float, nargs="+", default=(0.05, 0.1, 0.2, 0.4))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_cnn_gru_oracle_alarm_controller.json"))
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session, args.gains, args.shifts,
            ))
            print(f"oracle controller session={session} subject={subject}", flush=True)
    payload = {
        "protocol": {
            "baseline": "strictly coordinate-matched all-score EWMA H30",
            "controller": "integral threshold offset from oracle-idle false-alarm innovations",
            "target_fp_per_min": 2.0,
            "gains": args.gains,
            "oracle_is_non_deployable": True,
            "selection_rule": "proxy arm proceeds only if a fixed oracle gain improves calibration without paired excess loss",
        },
        "summary": [summarize(rows, session) for session in args.sessions],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
