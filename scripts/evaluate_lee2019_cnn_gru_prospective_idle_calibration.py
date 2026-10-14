"""Prospective known-idle threshold calibration for frozen CNN-GRU scores.

All arms are evaluated on the same final 40 command opportunities.  Only the
information used to set the decision threshold changes:

* anchor: unused final 30 s of session-start pre-rest only;
* anchor_plus_5: anchor plus known idle preceding the first five commands;
* anchor_plus_10: anchor plus known idle preceding the first ten commands;
* tail40_oracle: anchor plus future known idle in the evaluation tail.

The first three arms are causal.  The oracle arm is diagnostic only.  CNN-GRU
weights and all-score EWMA normalization are frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.decision import moving_average_dwell
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS,
    HALF_LIFE_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    STEP,
    normalize_anchor,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration
from scripts.run_lee2019_sequential_onset import onset_rates

TARGETS = (2.0, 4.0)
HORIZONS = (0.5, 1.0, 2.0)
SCREEN_TRIALS = 10


def idle_chunks(sequence, scores, indices):
    chunks = []
    for index in indices:
        start, stop = sequence.idle_intervals[index]
        mask = (sequence.times >= start) & (sequence.times < stop)
        values = np.asarray(scores[mask], float)
        if len(values):
            chunks.append(values)
    return chunks


def tune(sequences):
    return tune_source_operating_points(
        sequences,
        "ma_dwell",
        TARGETS,
        STEP,
        ma_seconds=MA_SECONDS,
        dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS,
        grid_size=240,
    )


def prefix_sequence(sequence, scores, trials):
    stop = float(sequence.command_intervals[trials - 1, 1])
    keep = sequence.times < stop
    return ScoredSequence(
        np.asarray(sequence.times[keep], float),
        np.asarray(scores[keep], float),
        np.asarray(sequence.idle_intervals[:trials], float),
        np.asarray(sequence.command_intervals[:trials], float),
    )


def tail_sequence(sequence, scores):
    start = float(sequence.idle_intervals[SCREEN_TRIALS, 0])
    keep = sequence.times >= start
    return ScoredSequence(
        np.asarray(sequence.times[keep], float),
        np.asarray(scores[keep], float),
        np.asarray(sequence.idle_intervals[SCREEN_TRIALS:], float),
        np.asarray(sequence.command_intervals[SCREEN_TRIALS:], float),
    )


def shifted_null(times, alarms, commands, offsets):
    output = {str(value): [] for value in HORIZONS}
    for offset in offsets:
        shifted = np.roll(alarms, int(offset))
        rates, _ = onset_rates(times, shifted, commands)
        for horizon in HORIZONS:
            output[str(horizon)].append(float(rates[str(horizon)]))
    return {key: float(np.median(values)) for key, values in output.items()}


def evaluate(sequence, threshold, offsets):
    decision = moving_average_dwell(
        sequence.scores,
        float(threshold),
        STEP,
        MA_SECONDS,
        DWELL_SECONDS,
        REFRACTORY_SECONDS,
    )
    conventional = evaluate_alarms(
        sequence.times,
        decision.alarms,
        sequence.idle_intervals,
        sequence.command_intervals,
    )
    observed, delay = onset_rates(
        sequence.times, decision.alarms, sequence.command_intervals
    )
    null = shifted_null(
        sequence.times, decision.alarms, sequence.command_intervals, offsets
    )
    onset = {}
    for horizon in HORIZONS:
        key = str(horizon)
        onset[key] = {
            "sensitivity": float(observed[key]),
            "circular_shift_null": float(null[key]),
            "excess": float(observed[key] - null[key]),
        }
    return {
        "threshold": float(threshold),
        "actual_fp_per_min": float(conventional.fp_per_minute),
        "false_alarm_count": int(conventional.false_alarms),
        "idle_seconds": float(conventional.idle_seconds),
        "any_overlap_tpr": float(conventional.true_positive_rate),
        "onset": onset,
        "latency_median_seconds": None if not np.isfinite(delay) else float(delay),
    }


def analyze(path, subject, session, shifts):
    sequence, _ = load_cache(path)
    with np.load(path, allow_pickle=False) as data:
        if "session_anchor_scores" not in data.files:
            raise ValueError(
                f"{path} lacks session_anchor_scores; regenerate with the "
                "prospective CNN-GRU score cache"
            )
        session_anchor = np.asarray(data["session_anchor_scores"], float)
    mean, variance, anchor_z = normalize_anchor([session_anchor])
    ewma = ewma_normalize_and_decide(
        sequence.scores,
        initial_mean=mean,
        initial_variance=variance,
        threshold=0.0,
        step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS,
        update_mode="all",
        ma_seconds=MA_SECONDS,
        dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS,
        sigma_floor_fraction=.25,
    )
    statistic = np.asarray(ewma.statistic, float)
    anchor = [np.asarray(values, float) for values in anchor_z]
    first5 = idle_chunks(sequence, statistic, range(5))
    first10 = idle_chunks(sequence, statistic, range(10))
    future = idle_chunks(sequence, statistic, range(SCREEN_TRIALS, len(sequence.idle_intervals)))

    anchor_sequences = [calibration(values) for values in anchor]
    tail = tail_sequence(sequence, statistic)
    calibration_sets = {
        "A_anchor": anchor_sequences,
        "B_anchor_plus_5": anchor_sequences + [prefix_sequence(sequence, statistic, 5)],
        "C_anchor_plus_10": anchor_sequences + [prefix_sequence(sequence, statistic, 10)],
        # Diagnostic upper bound: tune and evaluate the exact same continuous
        # tail, including command-induced MA/refractory state, while counting
        # false alarms only in its true idle intervals.
        "D_tail40_oracle": [tail],
    }
    points = {name: tune(sequences) for name, sequences in calibration_sets.items()}
    rng = np.random.default_rng(20260910 + 1000 * session + subject)
    offsets = rng.integers(1, len(tail.scores), shifts)
    arms = {}
    for name, point_by_target in points.items():
        arms[name] = {}
        for target in TARGETS:
            point = point_by_target[target]
            result = evaluate(tail, point.threshold, offsets)
            result.update({
                "target_fp_per_min": target,
                "calibration_achieved_fp_per_min": float(point.source_fp_per_minute),
                "operating_point_log_error": float(
                    abs(np.log((result["actual_fp_per_min"] + .05) / (target + .05)))
                ),
            })
            arms[name][str(target)] = result
    return {
        "subject": subject,
        "session": session,
        "anchor_seconds": float(sum(len(values) for values in anchor) * STEP),
        "first5_known_idle_seconds": float(sum(len(values) for values in first5) * STEP),
        "first10_known_idle_seconds": float(sum(len(values) for values in first10) * STEP),
        "tail40_idle_seconds": float(sum(len(values) for values in future) * STEP),
        "evaluation_trials": int(len(tail.command_intervals)),
        "arms": arms,
    }


def bootstrap_median(values, seed, repeats=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def summarize_group(rows, seed):
    output = {"n": len(rows), "arms": {}}
    names = tuple(rows[0]["arms"])
    for arm_index, name in enumerate(names):
        output["arms"][name] = {}
        for target in TARGETS:
            key = str(target)
            points = [row["arms"][name][key] for row in rows]
            result = {}
            scalar_paths = {
                "actual_fp_per_min": lambda p: p["actual_fp_per_min"],
                "operating_point_log_error": lambda p: p["operating_point_log_error"],
                "sensitivity_1s": lambda p: p["onset"]["1.0"]["sensitivity"],
                "null_1s": lambda p: p["onset"]["1.0"]["circular_shift_null"],
                "excess_1s": lambda p: p["onset"]["1.0"]["excess"],
                "sensitivity_2s": lambda p: p["onset"]["2.0"]["sensitivity"],
                "excess_2s": lambda p: p["onset"]["2.0"]["excess"],
            }
            for metric_index, (metric, getter) in enumerate(scalar_paths.items()):
                values = [getter(point) for point in points]
                result[f"{metric}_median"] = float(np.median(values))
                result[f"{metric}_ci95"] = bootstrap_median(
                    values, seed + arm_index * 100 + int(target) * 10 + metric_index
                )
            result["admissible_n"] = int(sum(
                .75 * target <= point["actual_fp_per_min"] <= 1.25 * target
                for point in points
            ))
            if name != "A_anchor":
                baseline = [row["arms"]["A_anchor"][key] for row in rows]
                for metric, getter in {
                    "log_error": lambda p: p["operating_point_log_error"],
                    "excess_1s": lambda p: p["onset"]["1.0"]["excess"],
                    "excess_2s": lambda p: p["onset"]["2.0"]["excess"],
                }.items():
                    delta = [getter(point) - getter(base) for point, base in zip(points, baseline)]
                    result[f"paired_delta_{metric}_median"] = float(np.median(delta))
                    result[f"paired_delta_{metric}_ci95"] = bootstrap_median(
                        delta, seed + 5000 + arm_index * 100 + int(target) * 10 + len(metric)
                    )
            output["arms"][name][key] = result
    for exposure in (
        "anchor_seconds",
        "first5_known_idle_seconds",
        "first10_known_idle_seconds",
        "tail40_idle_seconds",
    ):
        output[f"{exposure}_median"] = float(np.median([row[exposure] for row in rows]))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_prospective"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_prospective_idle_calibration_subset20.json"))
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject,
                session,
                args.shifts,
            ))
            print(f"prospective-idle S{session} sub-{subject:02d}", flush=True)
    summary = {
        f"session{session}": summarize_group(
            [row for row in rows if row["session"] == session], 202609100 + session * 10000
        )
        for session in args.sessions
    }
    summary["combined"] = summarize_group(rows, 202609199)
    payload = {
        "protocol": {
            "score": "frozen three-seed CNN-GRU; causal all-score EWMA H30",
            "decision": "MA 0.5 s + dwell 0.3 s + refractory 1.0 s",
            "threshold_targets_fp_per_min": TARGETS,
            "calibration_arms": {
                "A_anchor": "unused final 30 s of session-start pre_train rest only",
                "B_anchor_plus_5": "anchor plus known idle preceding first 5 commands",
                "C_anchor_plus_10": "anchor plus known idle preceding first 10 commands",
                "D_tail40_oracle": "anchor plus future known idle from evaluated tail; non-deployable",
            },
            "evaluation": "same untouched final 40 command opportunities for every arm",
            "onset_horizons_seconds": HORIZONS,
            "chance": "within-tail circular shifts",
            "claim_limit": "Lee2019 cue-relative pseudo-uncued replay",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
