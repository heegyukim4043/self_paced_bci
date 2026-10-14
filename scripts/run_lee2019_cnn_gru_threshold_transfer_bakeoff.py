"""P7 threshold-transfer bake-off on frozen CNN-GRU score streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import genpareto

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import instantaneous
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points
from scripts.run_lee2019_stateful_gru import calibration_sequence, onset_rates

STEP = 0.1
TARGET_FP = 2.0
ADMISSIBLE = (1.5, 2.5)
SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)


def robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(np.std(values, ddof=1)), 1e-6)
    return location, scale


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        calibrations = [
            np.asarray(data[key], dtype=float)
            for key in sorted(k for k in data.files if k.startswith("calibration_"))
        ]
        sequence = ScoredSequence(
            np.asarray(data["times"], dtype=float),
            np.asarray(data["scores"], dtype=float),
            np.asarray(data["idle_intervals"], dtype=float),
            np.asarray(data["command_intervals"], dtype=float),
        )
    return sequence, calibrations


def tune_empirical(calibrations):
    return tune_source_operating_points(
        [calibration_sequence(x) for x in calibrations], "threshold",
        [TARGET_FP], STEP, refractory_seconds=1.0, grid_size=200,
    )[TARGET_FP].threshold


def standardized(sequence, location, scale):
    return ScoredSequence(
        sequence.times, (sequence.scores - location) / scale,
        sequence.idle_intervals, sequence.command_intervals,
    )


def pooled_threshold(subject, session, loaded):
    target_cal = np.concatenate(loaded[(subject, session)][1])
    target_location, target_scale = robust_location_scale(target_cal)
    source_sequences = []
    for (other, other_session), (_, calibrations) in loaded.items():
        if other_session != session or other == subject:
            continue
        values = np.concatenate(calibrations)
        location, scale = robust_location_scale(values)
        z = (values - location) / scale
        source_sequences.append(calibration_sequence(z))
    threshold_z = tune_source_operating_points(
        source_sequences, "threshold", [TARGET_FP], STEP,
        refractory_seconds=1.0, grid_size=300,
    )[TARGET_FP].threshold
    return target_location + target_scale * threshold_z


def cluster_ratio(values, threshold):
    above = np.asarray(values) >= threshold
    exceedances = int(np.sum(above))
    clusters = int(np.sum(above & ~np.r_[False, above[:-1]]))
    return max(clusters / max(exceedances, 1), 1e-3)


def evt_pot_threshold(calibrations, tail_quantile=0.8):
    values = np.concatenate(calibrations).astype(float)
    u = float(np.quantile(values, tail_quantile))
    excess = values[values > u] - u
    if len(excess) < 30 or not np.all(np.isfinite(excess)):
        return tune_empirical(calibrations), "fallback_empirical"
    shape, _, scale = genpareto.fit(excess, floc=0.0)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return tune_empirical(calibrations), "fallback_empirical"
    theta = cluster_ratio(values, u)
    alarm_probability = TARGET_FP * STEP / 60.0
    exceedance_probability = min(max(alarm_probability / theta, 1e-6), 0.1)
    tail_mass = 1.0 - tail_quantile
    conditional_survival = min(max(exceedance_probability / tail_mass, 1e-8), 0.999999)
    threshold = u + float(genpareto.ppf(1.0 - conditional_survival, shape, loc=0, scale=scale))
    if not np.isfinite(threshold):
        return tune_empirical(calibrations), "fallback_empirical"
    return threshold, "evt_pot"


def cumulative_quantile_alarms(scores, initial_values, initial_threshold):
    scores = np.asarray(scores, dtype=float)
    history = list(np.asarray(initial_values, dtype=float))
    theta = cluster_ratio(history, initial_threshold)
    probability = min(max(TARGET_FP * STEP / 60.0 / theta, 1e-5), 0.1)
    threshold = float(initial_threshold)
    thresholds = np.empty(len(scores), dtype=float)
    alarms = np.zeros(len(scores), dtype=bool)
    previous_above = False
    refractory_steps = int(np.ceil(1.0 / STEP))
    blocked_until = -1
    for index, score in enumerate(scores):
        thresholds[index] = threshold
        above = bool(score >= threshold)
        if above and not previous_above and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
        # Deployable idle proxy: only sub-threshold, non-refractory samples enter.
        if index > blocked_until and not above:
            history.append(float(score))
            threshold = float(np.quantile(history, 1.0 - probability))
        previous_above = above
    return alarms, thresholds


def evaluate(sequence, alarms, offsets):
    conventional = evaluate_alarms(
        sequence.times, alarms, sequence.idle_intervals, sequence.command_intervals
    )
    observed, delay = onset_rates(sequence.times, alarms, sequence.command_intervals)
    null = []
    for offset in offsets:
        shifted, _ = onset_rates(
            sequence.times, np.roll(alarms, int(offset)), sequence.command_intervals
        )
        null.append(shifted["1.0"])
    null_value = float(np.median(null))
    return {
        "fp": float(conventional.fp_per_minute),
        "raw_tpr_1s": float(observed["1.0"]),
        "null_tpr_1s": null_value,
        "excess_1s": float(observed["1.0"] - null_value),
        "delay": None if not np.isfinite(delay) else float(delay),
        "admissible": bool(ADMISSIBLE[0] <= conventional.fp_per_minute <= ADMISSIBLE[1]),
    }


def bootstrap_median(values, seed, repeats=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, session, repeats):
    selected = [row for row in rows if row["session"] == session]
    result = {"session": session, "subjects": len(selected), "arms": {}}
    baseline_good = {
        row["subject"] for row in selected if row["arms"]["baseline"]["admissible"]
    }
    for arm in ("baseline", "pooled_anchor", "evt_pot", "cumulative_quantile"):
        metrics = [row["arms"][arm] for row in selected]
        baseline_metrics = [row["arms"]["baseline"] for row in selected]
        excess = [x["excess_1s"] for x in metrics]
        fp = [x["fp"] for x in metrics]
        delta = [x["excess_1s"] - b["excess_1s"] for x, b in zip(metrics, baseline_metrics)]
        op_error = [abs(np.log(max(x, 1e-9) / TARGET_FP)) for x in fp]
        worsened = sum(
            row["subject"] in baseline_good and not row["arms"][arm]["admissible"]
            for row in selected
        )
        result["arms"][arm] = {
            "fp_median": float(np.median(fp)),
            "fp_iqr": np.quantile(fp, [0.25, 0.75]).tolist(),
            "admissible_n": int(sum(x["admissible"] for x in metrics)),
            "bracket_failure_rate": float(np.mean([not x["admissible"] for x in metrics])),
            "excess_1s_median": float(np.median(excess)),
            "excess_1s_ci": bootstrap_median(excess, 20260907 + session, repeats),
            "paired_excess_vs_baseline_median": float(np.median(delta)),
            "paired_excess_vs_baseline_ci": bootstrap_median(
                delta, 20261907 + session, repeats
            ),
            "operating_point_abs_log_error_median": float(np.median(op_error)),
            "baseline_admissible_worsened_n": int(worsened),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_threshold_transfer_bakeoff.json"))
    args = parser.parse_args()
    loaded = {}
    for session in args.sessions:
        for subject in args.subjects:
            loaded[(subject, session)] = load_cache(
                args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            )
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            sequence, calibrations = loaded[(subject, session)]
            values = np.concatenate(calibrations)
            baseline_threshold = tune_empirical(calibrations)
            pooled = pooled_threshold(subject, session, loaded)
            evt, evt_status = evt_pot_threshold(calibrations)
            rng = np.random.default_rng(20260907 + 100 * session + subject)
            offsets = rng.integers(1, len(sequence.scores), size=args.shifts)
            arms = {}
            for name, threshold in (("baseline", baseline_threshold), ("pooled_anchor", pooled), ("evt_pot", evt)):
                alarms = instantaneous(sequence.scores, threshold, STEP, 1.0).alarms
                arms[name] = evaluate(sequence, alarms, offsets)
                arms[name]["threshold"] = float(threshold)
            cq_alarms, cq_thresholds = cumulative_quantile_alarms(
                sequence.scores, values, baseline_threshold
            )
            arms["cumulative_quantile"] = evaluate(sequence, cq_alarms, offsets)
            arms["cumulative_quantile"]["threshold_initial"] = float(baseline_threshold)
            arms["cumulative_quantile"]["threshold_final"] = float(cq_thresholds[-1])
            rows.append({"subject": subject, "session": session, "evt_status": evt_status, "arms": arms})
            print(f"P7 session={session} subject={subject}", flush=True)
    payload = {
        "protocol": {
            "target_fp_per_minute": TARGET_FP,
            "admissible_fp_range": ADMISSIBLE,
            "step_seconds": STEP,
            "arms": ["baseline", "pooled_anchor", "evt_pot", "cumulative_quantile"],
            "score": "frozen 3-seed mean CNN-GRU",
            "cumulative_update": "sub-threshold non-refractory proxy; no GT labels",
        },
        "summary": [summarize(rows, s, args.bootstrap) for s in args.sessions],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
