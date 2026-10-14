"""Separate onset-detection and false-alarm accounting for CNN-GRU EWMA."""

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
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import (
    controlled_decision, interval_mask, shifted_null,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration

HORIZONS = (0.5, 1.0, 2.0)


def onset_metrics(times, alarms, commands, offsets):
    output = {}
    for horizon in HORIZONS:
        hits = np.asarray([
            np.any(alarms & (times >= onset) & (times <= min(onset + horizon, stop)))
            for onset, stop in commands
        ])
        # Circular-shift null uses the same clipped within-command support.
        shifted = np.asarray([
            np.roll(alarms, int(offset)) for offset in offsets
        ])
        null_hits = np.zeros(len(offsets), float)
        for onset, stop in commands:
            mask = (times >= onset) & (times <= min(onset + horizon, stop))
            null_hits += np.any(shifted[:, mask], axis=1)
        null = float(np.median(null_hits / len(commands)))
        output[str(horizon)] = {
            "detected_events": int(hits.sum()),
            "total_events": int(len(hits)),
            "sensitivity": float(hits.mean()),
            "circular_shift_null": null,
            "excess": float(hits.mean() - null),
        }
    delays = []
    for onset, stop in commands:
        found = times[alarms & (times >= onset) & (times <= stop)]
        if len(found):
            delays.append(float(found[0] - onset))
    output["detected_within_command"] = len(delays)
    output["latency_median_seconds"] = float(np.median(delays)) if delays else None
    return output


def false_alarm_metrics(times, alarms, idle_intervals):
    idle = interval_mask(times, idle_intervals)
    indices = np.flatnonzero(alarms & idle)
    exposure = float(idle.sum() * STEP)
    cumulative = np.cumsum(idle) * STEP
    segment_counts = [int(np.sum(alarms & (times >= a) & (times < b)))
                      for a, b in idle_intervals]
    return {
        "false_alarm_count": int(len(indices)),
        "valid_idle_seconds": exposure,
        "fp_per_min": float(60 * len(indices) / exposure),
        "idle_segments": int(len(idle_intervals)),
        "idle_segments_with_false_alarm": int(np.sum(np.asarray(segment_counts) > 0)),
        "false_alarms_per_idle_segment": segment_counts,
        "first_fp_observed": bool(len(indices)),
        "first_fp_valid_idle_seconds": float(cumulative[indices[0]]) if len(indices) else exposure,
        "first_fp_wallclock_seconds": float(times[indices[0]] - times[0]) if len(indices) else float(times[-1] - times[0]),
    }


def analyze(path, subject, session, shifts):
    sequence, blocks = load_cache(path)
    mean, variance, normalized = normalize_anchor(blocks)
    point = tune_ma_dwell([calibration(x) for x in normalized], 160)
    adapted = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=point.threshold, step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS, update_mode="all",
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=.25,
    )
    idle = interval_mask(sequence.times, sequence.idle_intervals)
    oracle, _, _ = controlled_decision(
        adapted.statistic, idle, point.threshold, .8, 2.0
    )
    rng = np.random.default_rng(950000 + session * 1000 + subject)
    offsets = rng.integers(1, len(sequence.times), shifts)
    arms = {}
    for name, alarms in (("ewma_only", adapted.alarms),
                         ("oracle_idle_gain_0.8", oracle)):
        arms[name] = {
            "onset": onset_metrics(sequence.times, alarms, sequence.command_intervals, offsets),
            "false_alarm": false_alarm_metrics(sequence.times, alarms, sequence.idle_intervals),
            "total_alarm_count": int(alarms.sum()),
        }
    return {"subject": subject, "session": session, "arms": arms}


def boot(values, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (20000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def km_median(rows, arm):
    pairs = sorted((row["arms"][arm]["false_alarm"]["first_fp_valid_idle_seconds"],
                    row["arms"][arm]["false_alarm"]["first_fp_observed"])
                   for row in rows)
    survival = 1.0
    for time in sorted(set(x[0] for x in pairs)):
        at_risk = sum(duration >= time for duration, _ in pairs)
        events = sum(duration == time and observed for duration, observed in pairs)
        if at_risk:
            survival *= 1 - events / at_risk
        if survival <= .5:
            return float(time)
    return None


def summarize(rows, session):
    chosen = [row for row in rows if row["session"] == session]
    result = {"session": session, "n": len(chosen), "arms": {}}
    for arm_index, arm in enumerate(chosen[0]["arms"]):
        block = {"onset": {}}
        for horizon in HORIZONS:
            key = str(horizon)
            for metric in ("sensitivity", "circular_shift_null", "excess"):
                values = [row["arms"][arm]["onset"][key][metric] for row in chosen]
                block["onset"].setdefault(key, {})[f"{metric}_median"] = float(np.median(values))
                block["onset"][key][f"{metric}_ci95"] = boot(
                    values, 20261400 + session * 100 + arm_index * 20 + int(horizon * 10) + len(metric)
                )
            block["onset"][key]["pooled_detected"] = int(sum(
                row["arms"][arm]["onset"][key]["detected_events"] for row in chosen
            ))
            block["onset"][key]["pooled_total"] = int(sum(
                row["arms"][arm]["onset"][key]["total_events"] for row in chosen
            ))
        fa = [row["arms"][arm]["false_alarm"] for row in chosen]
        total_fp = sum(x["false_alarm_count"] for x in fa)
        total_idle = sum(x["valid_idle_seconds"] for x in fa)
        block["false_alarm"] = {
            "pooled_count": int(total_fp),
            "pooled_valid_idle_seconds": float(total_idle),
            "pooled_fp_per_min": float(60 * total_fp / total_idle),
            "subject_fp_per_min_median": float(np.median([x["fp_per_min"] for x in fa])),
            "subject_fp_per_min_ci95": boot([x["fp_per_min"] for x in fa], 20261500 + session * 10 + arm_index),
            "idle_segments_with_fp_pooled": int(sum(x["idle_segments_with_false_alarm"] for x in fa)),
            "idle_segments_pooled": int(sum(x["idle_segments"] for x in fa)),
            "km_median_first_fp_valid_idle_seconds": km_median(chosen, arm),
        }
        latencies = [row["arms"][arm]["onset"]["latency_median_seconds"] for row in chosen
                     if row["arms"][arm]["onset"]["latency_median_seconds"] is not None]
        block["latency_subject_median_seconds"] = float(np.median(latencies)) if latencies else None
        result["arms"][arm] = block
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_onset_false_alarm_audit.json"))
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz",
                                subject, session, args.shifts))
            print(f"onset/FP audit session={session} subject={subject}", flush=True)
    payload = {
        "protocol": {
            "onset": "cue-relative pseudo-onset at command interval start",
            "horizons_seconds": HORIZONS,
            "horizons_clipped_to_2s_command": True,
            "false_alarm_denominator": "valid injected-idle exposure",
            "random_idle_limitation": True,
            "oracle_arm_non_deployable": True,
        },
        "summary": [summarize(rows, session) for session in args.sessions],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
