"""Locked comparison of causal smoothing and score-gradient decision arms."""

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
    HALF_LIFE_SECONDS, STEP, normalize_anchor,
)
from scripts.audit_lee2019_cnn_gru_onset_false_alarms import (
    false_alarm_metrics, onset_metrics,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache

DWELL = .3
REFRACTORY = 1.
TARGET_FP = 2.
ARMS = ("ma_0.5", "ema_0.5", "level_plus_gradient", "level_and_gradient")


def causal_ma(values, seconds=.5):
    steps = max(1, int(np.ceil(seconds / STEP)))
    result = np.convolve(values, np.ones(steps) / steps, mode="full")[:len(values)]
    result[:steps - 1] = np.nan
    return result


def causal_ema(values, half_life):
    values = np.asarray(values, float)
    alpha = 1 - 2 ** (-STEP / half_life)
    result = np.empty(len(values), float)
    state = float(values[0])
    for index, value in enumerate(values):
        state += alpha * (float(value) - state)
        result[index] = state
    return result


def components(z, gradient_scale):
    level = causal_ema(z, .5)
    fast = causal_ema(z, .2)
    slow = causal_ema(z, 1.)
    gradient = (fast - slow) / max(float(gradient_scale), 1e-8)
    return level, gradient


def alarm_from_statistic(statistic, threshold, gate=None):
    active = np.isfinite(statistic) & (statistic >= threshold)
    if gate is not None:
        active &= gate
    dwell_steps = max(1, int(np.ceil(DWELL / STEP)))
    refractory_steps = max(0, int(np.ceil(REFRACTORY / STEP)))
    alarms = np.zeros(len(active), bool)
    count, blocked_until = 0, -1
    for index, value in enumerate(active):
        count = count + 1 if value else 0
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
    return alarms


def arm_statistic(z, name, gradient_scale):
    if name == "ma_0.5":
        return causal_ma(z), None
    level, gradient = components(z, gradient_scale)
    if name == "ema_0.5":
        return level, None
    if name == "level_plus_gradient":
        return level + .5 * np.maximum(gradient, 0.), None
    if name == "level_and_gradient":
        return level, gradient > 0.
    raise ValueError(name)


def fp_rate(alarms, exposure_samples):
    return float(60 * np.sum(alarms) / (exposure_samples * STEP))


def tune_anchor(blocks, name, gradient_scale, grid=200):
    prepared = [arm_statistic(block, name, gradient_scale) for block in blocks]
    finite = np.concatenate([stat[np.isfinite(stat)] for stat, _ in prepared])
    candidates = np.unique(np.quantile(finite, np.linspace(.35, .9999, grid)))
    rows = []
    exposure = sum(len(block) for block in blocks)
    for threshold in candidates:
        alarms = [alarm_from_statistic(stat, threshold, gate) for stat, gate in prepared]
        rows.append((float(threshold), fp_rate(np.concatenate(alarms), exposure)))
    return min(rows, key=lambda item: abs(item[1] - TARGET_FP)), rows


def tune_common(statistic, gate, idle, grid=240):
    finite = statistic[np.isfinite(statistic)]
    candidates = np.unique(np.quantile(finite, np.linspace(.25, .9999, grid)))
    rows = []
    exposure = int(idle.sum())
    for threshold in candidates:
        alarms = alarm_from_statistic(statistic, threshold, gate)
        rows.append((float(threshold), fp_rate(alarms & idle, exposure)))
    return min(rows, key=lambda item: abs(item[1] - TARGET_FP))


def evaluate(sequence, alarms, offsets):
    onset = onset_metrics(sequence.times, alarms, sequence.command_intervals, offsets)
    false_alarm = false_alarm_metrics(sequence.times, alarms, sequence.idle_intervals)
    return {"onset": onset, "false_alarm": false_alarm,
            "total_alarm_count": int(alarms.sum())}


def analyze(path, subject, session, shifts):
    sequence, raw_blocks = load_cache(path)
    mean, variance, anchor_z = normalize_anchor(raw_blocks)
    # EWMA normalization is independent of the downstream arm threshold.
    normalized = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=np.inf, step_seconds=STEP, half_life_seconds=HALF_LIFE_SECONDS,
        update_mode="all", ma_seconds=.5, dwell_seconds=.3,
        refractory_seconds=1., sigma_floor_fraction=.25,
    ).statistic
    anchor_gradients = []
    for block in anchor_z:
        fast = causal_ema(block, .2); slow = causal_ema(block, 1.)
        anchor_gradients.append(fast - slow)
    gradient_scale = float(max(np.std(np.concatenate(anchor_gradients), ddof=1), 1e-8))
    idle = np.zeros(len(sequence.times), bool)
    for start, stop in sequence.idle_intervals:
        idle |= (sequence.times >= start) & (sequence.times < stop)
    rng = np.random.default_rng(970000 + session * 1000 + subject)
    offsets = rng.integers(1, len(sequence.times), shifts)
    output = {}
    for name in ARMS:
        transfer, _ = tune_anchor(anchor_z, name, gradient_scale)
        statistic, gate = arm_statistic(normalized, name, gradient_scale)
        common = tune_common(statistic, gate, idle)
        transferred_alarms = alarm_from_statistic(statistic, transfer[0], gate)
        common_alarms = alarm_from_statistic(statistic, common[0], gate)
        output[name] = {
            "gradient_scale_anchor": gradient_scale,
            "transferred_threshold": transfer[0],
            "anchor_fp_per_min": transfer[1],
            "transferred": evaluate(sequence, transferred_alarms, offsets),
            "common_fp_threshold": common[0],
            "common_achieved_fp_per_min": common[1],
            "common_fp": evaluate(sequence, common_alarms, offsets),
        }
    return {"subject": subject, "session": session, "arms": output}


def boot(values, seed):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (20000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def summarize(rows, session):
    chosen = [row for row in rows if row["session"] == session]
    summary = {"session": session, "n": len(chosen), "arms": {}}
    for arm_index, arm in enumerate(ARMS):
        block = {}
        for operating in ("transferred", "common_fp"):
            points = [row["arms"][arm][operating] for row in chosen]
            item = {}
            for horizon in (.5, 1., 2.):
                key = str(horizon)
                for metric in ("sensitivity", "circular_shift_null", "excess"):
                    values = [point["onset"][key][metric] for point in points]
                    item[f"sen{key}_{metric}_median"] = float(np.median(values))
                    item[f"sen{key}_{metric}_ci95"] = boot(
                        values, 20261700 + session * 100 + arm_index * 20
                        + (0 if operating == "transferred" else 10) + int(horizon * 2) + len(metric)
                    )
            fps = [point["false_alarm"]["fp_per_min"] for point in points]
            item["fp_per_min_median"] = float(np.median(fps))
            item["fp_per_min_ci95"] = boot(fps, 20261800 + session * 100 + arm_index)
            if arm != "ma_0.5":
                for horizon in (.5, 1., 2.):
                    key = str(horizon)
                    deltas = [
                        row["arms"][arm][operating]["onset"][key]["excess"]
                        - row["arms"]["ma_0.5"][operating]["onset"][key]["excess"]
                        for row in chosen
                    ]
                    label = str(horizon).replace(".", "p")
                    item[f"paired_delta_excess{label}_vs_ma_median"] = float(np.median(deltas))
                    item[f"paired_delta_excess{label}_vs_ma_ci95"] = boot(
                        deltas, 20261900 + session * 100 + arm_index * 20
                        + (0 if operating == "transferred" else 10) + int(horizon * 2)
                    )
            block[operating] = item
        summary["arms"][arm] = block
    return summary


def plot(summary, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["MA0.5", "EMA0.5", "Level+grad", "Level AND grad"]
    x = np.arange(len(ARMS))
    for row, session_summary in enumerate(summary):
        for column, operating in enumerate(("transferred", "common_fp")):
            ax = axes[row, column]
            excess = [session_summary["arms"][arm][operating]["sen1.0_excess_median"] for arm in ARMS]
            observed = [session_summary["arms"][arm][operating]["sen1.0_sensitivity_median"] for arm in ARMS]
            null = [session_summary["arms"][arm][operating]["sen1.0_circular_shift_null_median"] for arm in ARMS]
            width = .25
            ax.bar(x-width, observed, width, label="SEN@1s", color="#4c72b0")
            ax.bar(x, null, width, label="shift null", color="#999999")
            ax.bar(x+width, excess, width, label="excess", color="#dd4b39")
            fps = [session_summary["arms"][arm][operating]["fp_per_min_median"] for arm in ARMS]
            for i, fp in enumerate(fps):
                ax.text(i, max(observed[i], null[i], excess[i]) + .025, f"FP {fp:.2f}",
                        ha="center", fontsize=8)
            ax.set_xticks(x, labels, rotation=15, ha="right")
            ax.set_ylim(-.08, max(.7, max(observed)+.12))
            ax.set_title(f"S{session_summary['session']} {operating.replace('_',' ')}")
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle("CNN-GRU smoothing and causal-gradient comparison")
    fig.tight_layout(rect=(0, 0, 1, .96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_smoothing_gradient.json"))
    parser.add_argument("--figure", type=Path, default=Path("data/processed/lee2019_cnn_gru_smoothing_gradient.png"))
    parser.add_argument("--reuse-output", action="store_true",
                        help="Recompute summaries and figure from existing JSON rows.")
    args = parser.parse_args()
    if args.reuse_output:
        rows = json.loads(args.output.read_text(encoding="utf-8"))["rows"]
    else:
        rows = []
        for session in args.sessions:
            for subject in args.subjects:
                rows.append(analyze(args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz",
                                    subject, session, args.shifts))
                print(f"smoothing-gradient session={session} subject={subject}", flush=True)
    summary = [summarize(rows, session) for session in args.sessions]
    payload = {
        "protocol": {
            "score": "frozen CNN-GRU plus all-score EWMA H30",
            "arms": {
                "ma_0.5": "causal boxcar MA .5 s",
                "ema_0.5": "causal EMA half-life .5 s",
                "level_plus_gradient": "EMA.5 + .5*max((EMA.2-EMA1.0)/anchorSD,0)",
                "level_and_gradient": "EMA.5 threshold AND normalized gradient > 0",
            },
            "decision": "dwell .3 s; refractory 1 s",
            "threshold": "arm-specific full alarm-process anchor tuning; diagnostic common actual FP2",
            "primary_comparison": "paired excess SEN@1s versus MA0.5",
            "pseudo_uncued_warning": True,
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(summary, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
