"""Known-idle exposure curve for prospective CNN-GRU threshold calibration.

The evaluated stream is fixed to the final 20 Lee2019 trials.  Thresholds are
estimated from the true 30 s session-start anchor and an increasing, strictly
earlier prefix containing 0/5/10/20/30 command trials and their known idle.
The final arm tunes on the evaluation tail itself and is an oracle diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_cnn_gru_prospective_idle_calibration import (
    TARGETS,
    bootstrap_median,
    evaluate,
    prefix_sequence,
    tune,
)
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS,
    HALF_LIFE_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    STEP,
    normalize_anchor,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.operating_point import ScoredSequence
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration


CALIBRATION_TRIALS = (0, 5, 10, 20, 30)
EVALUATION_START_TRIAL = 30
HORIZONS = (0.5, 1.0, 2.0)


def fixed_tail(sequence, scores):
    start = float(sequence.idle_intervals[EVALUATION_START_TRIAL, 0])
    keep = sequence.times >= start
    return ScoredSequence(
        np.asarray(sequence.times[keep], float),
        np.asarray(scores[keep], float),
        np.asarray(sequence.idle_intervals[EVALUATION_START_TRIAL:], float),
        np.asarray(sequence.command_intervals[EVALUATION_START_TRIAL:], float),
    )


def known_idle_seconds(sequence, trials):
    if trials == 0:
        return 0.0
    intervals = np.asarray(sequence.idle_intervals[:trials], float)
    return float(np.sum(intervals[:, 1] - intervals[:, 0]))


def analyze(path, subject, session, shifts):
    sequence, _ = load_cache(path)
    with np.load(path, allow_pickle=False) as data:
        if "session_anchor_scores" not in data.files:
            raise ValueError(f"{path} lacks session_anchor_scores")
        raw_anchor = np.asarray(data["session_anchor_scores"], float)

    mean, variance, anchor_z = normalize_anchor([raw_anchor])
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
        sigma_floor_fraction=0.25,
    )
    scores = np.asarray(ewma.statistic, float)
    anchor_sequences = [calibration(np.asarray(values, float)) for values in anchor_z]
    tail = fixed_tail(sequence, scores)

    calibration_sets = {"prefix00": anchor_sequences}
    for trials in CALIBRATION_TRIALS[1:]:
        calibration_sets[f"prefix{trials:02d}"] = (
            anchor_sequences + [prefix_sequence(sequence, scores, trials)]
        )
    calibration_sets["tail20_oracle"] = [tail]
    operating_points = {name: tune(items) for name, items in calibration_sets.items()}

    rng = np.random.default_rng(20260910 + 1000 * session + subject)
    offsets = rng.integers(1, len(tail.scores), shifts)
    arms = {}
    for name, by_target in operating_points.items():
        arms[name] = {}
        for target in TARGETS:
            point = by_target[target]
            result = evaluate(tail, point.threshold, offsets)
            result.update(
                target_fp_per_min=float(target),
                calibration_achieved_fp_per_min=float(point.source_fp_per_minute),
                operating_point_log_error=float(
                    abs(np.log((result["actual_fp_per_min"] + 0.05) / (target + 0.05)))
                ),
            )
            arms[name][str(target)] = result

    anchor_seconds = float(sum(len(values) for values in anchor_z) * STEP)
    exposures = {
        f"prefix{trials:02d}": anchor_seconds + known_idle_seconds(sequence, trials)
        for trials in CALIBRATION_TRIALS
    }
    exposures["tail20_oracle"] = float(
        np.sum(tail.idle_intervals[:, 1] - tail.idle_intervals[:, 0])
    )
    return {
        "subject": int(subject),
        "session": int(session),
        "evaluation_trials": int(len(tail.command_intervals)),
        "known_idle_exposure_seconds": exposures,
        "arms": arms,
    }


def metric_values(rows, arm, target, getter):
    key = str(target)
    return [getter(row["arms"][arm][key]) for row in rows]


def summarize_group(rows, seed, baseline_arm="prefix00"):
    names = tuple(rows[0]["arms"])
    result = {"n": len(rows), "arms": {}}
    getters = {
        "actual_fp_per_min": lambda p: p["actual_fp_per_min"],
        "operating_point_log_error": lambda p: p["operating_point_log_error"],
        "sensitivity_1s": lambda p: p["onset"]["1.0"]["sensitivity"],
        "null_1s": lambda p: p["onset"]["1.0"]["circular_shift_null"],
        "excess_1s": lambda p: p["onset"]["1.0"]["excess"],
        "sensitivity_2s": lambda p: p["onset"]["2.0"]["sensitivity"],
        "excess_2s": lambda p: p["onset"]["2.0"]["excess"],
    }
    for arm_index, arm in enumerate(names):
        exposure = [row["known_idle_exposure_seconds"][arm] for row in rows]
        arm_result = {
            "known_idle_exposure_seconds_median": float(np.median(exposure)),
            "known_idle_exposure_seconds_iqr": np.quantile(exposure, [0.25, 0.75]).tolist(),
        }
        for target in TARGETS:
            target_result = {}
            for metric_index, (metric, getter) in enumerate(getters.items()):
                values = metric_values(rows, arm, target, getter)
                target_result[f"{metric}_median"] = float(np.median(values))
                target_result[f"{metric}_ci95"] = bootstrap_median(
                    values,
                    seed + arm_index * 1000 + int(target) * 100 + metric_index,
                )
            lower, upper = ((1.5, 2.5) if target == 2.0 else (3.0, 5.0))
            actual = metric_values(rows, arm, target, getters["actual_fp_per_min"])
            target_result["admissible_n"] = int(sum(lower <= value <= upper for value in actual))
            if arm != baseline_arm:
                for delta_index, (metric, getter) in enumerate(
                    {
                        "log_error": getters["operating_point_log_error"],
                        "excess_1s": getters["excess_1s"],
                        "excess_2s": getters["excess_2s"],
                    }.items()
                ):
                    current = metric_values(rows, arm, target, getter)
                    baseline = metric_values(rows, baseline_arm, target, getter)
                    delta = np.asarray(current) - np.asarray(baseline)
                    target_result[f"paired_delta_{metric}_median"] = float(np.median(delta))
                    target_result[f"paired_delta_{metric}_ci95"] = bootstrap_median(
                        delta,
                        seed + 50000 + arm_index * 1000 + int(target) * 100 + delta_index,
                    )
            arm_result[str(target)] = target_result
        result["arms"][arm] = arm_result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnn-root",
        type=Path,
        default=Path("data/interim/lee2019_cnn_gru_scores_prospective"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/lee2019_cnn_gru_idle_exposure_curve_subset20.json"),
    )
    args = parser.parse_args()

    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            rows.append(analyze(path, subject, session, args.shifts))
            print(f"idle-exposure S{session} sub-{subject:02d}", flush=True)

    summary = {
        f"session{session}": summarize_group(
            [row for row in rows if row["session"] == session],
            202609100 + session * 10000,
        )
        for session in args.sessions
    }
    summary["combined"] = summarize_group(rows, 202609199)
    payload = {
        "protocol": {
            "score": "frozen three-seed CNN-GRU; causal all-score EWMA H30",
            "decision": "MA 0.5 s + dwell 0.3 s + refractory 1.0 s",
            "threshold_targets_fp_per_min": TARGETS,
            "calibration_trials": CALIBRATION_TRIALS,
            "evaluation": "same untouched final 20 trials (trials 31-50) for every arm",
            "calibration": "true 30 s session-start anchor plus continuous earlier prefix",
            "oracle": "tail20 tunes and evaluates on the same stream; diagnostic only",
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
