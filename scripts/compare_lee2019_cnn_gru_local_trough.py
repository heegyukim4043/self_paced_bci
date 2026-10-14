"""Causal local-trough decision audit for frozen CNN-GRU scores.

The score representation and EWMA normalization are frozen.  The experiment
tests whether a rise relative to a recent robust low baseline can recover MI
peaks missed by an absolute threshold.  It reports both prospective prefix-30
calibration and a same-tail matched-FP diagnostic oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.decision import moving_average_dwell
from uncued_mi.operating_point import ScoredSequence
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS,
    HALF_LIFE_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    STEP,
    normalize_anchor,
)
from scripts.audit_lee2019_cnn_gru_onset_false_alarms import (
    false_alarm_metrics,
    onset_metrics,
)
from scripts.evaluate_lee2019_cnn_gru_idle_exposure_curve import (
    EVALUATION_START_TRIAL,
    fixed_tail,
)
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import interval_mask
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache


TARGETS = (2.0, 4.0)
PRIMARY_TARGET = 2.0
QUANTILE = 0.10
GAP_SECONDS = 0.5
ARMS = (
    "absolute_ma",
    "local_q10_w2",
    "local_q10_w4",
    "local_q10_w8",
    "local_q10_w4_floor0",
)


def causal_ma(values: np.ndarray, seconds: float = MA_SECONDS) -> np.ndarray:
    values = np.asarray(values, float)
    steps = max(1, int(np.ceil(seconds / STEP)))
    result = np.convolve(values, np.ones(steps) / steps, mode="full")[:len(values)]
    result[:steps - 1] = np.nan
    return result


def causal_local_trough(
    level: np.ndarray,
    history_seconds: float,
    gap_seconds: float = GAP_SECONDS,
    quantile: float = QUANTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return level minus a strictly-past rolling low quantile and baseline."""
    level = np.asarray(level, float)
    history_steps = max(1, int(np.ceil(history_seconds / STEP)))
    gap_steps = max(1, int(np.ceil(gap_seconds / STEP)))
    baseline = np.full(len(level), np.nan, float)
    minimum_points = max(5, int(np.ceil(1.0 / STEP)))
    for index in range(len(level)):
        stop = index - gap_steps + 1
        start = max(0, stop - history_steps)
        if stop <= start:
            continue
        history = level[start:stop]
        history = history[np.isfinite(history)]
        if len(history) >= minimum_points:
            baseline[index] = float(np.quantile(history, quantile))
    return level - baseline, baseline


def arm_components(z: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    level = causal_ma(z)
    if arm == "absolute_ma":
        return level, None, np.full(len(level), np.nan)
    if arm.startswith("local_q10_w2"):
        history = 2.0
    elif arm.startswith("local_q10_w8"):
        history = 8.0
    else:
        history = 4.0
    statistic, baseline = causal_local_trough(level, history)
    gate = level >= 0.0 if arm.endswith("floor0") else None
    return statistic, gate, baseline


def alarm_from_statistic(
    statistic: np.ndarray,
    threshold: float,
    gate: np.ndarray | None = None,
) -> np.ndarray:
    active = np.isfinite(statistic) & (statistic >= threshold)
    if gate is not None:
        active &= np.asarray(gate, bool)
    dwell_steps = max(1, int(np.ceil(DWELL_SECONDS / STEP)))
    refractory_steps = max(0, int(np.ceil(REFRACTORY_SECONDS / STEP)))
    alarms = np.zeros(len(active), bool)
    count = 0
    blocked_until = -1
    for index, value in enumerate(active):
        count = count + 1 if value else 0
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
    return alarms


def slice_sequence(
    sequence: ScoredSequence,
    statistic: np.ndarray,
    gate: np.ndarray | None,
    keep: np.ndarray,
    idle_intervals: np.ndarray,
    command_intervals: np.ndarray,
) -> tuple[ScoredSequence, np.ndarray | None]:
    return (
        ScoredSequence(
            np.asarray(sequence.times[keep], float),
            np.asarray(statistic[keep], float),
            np.asarray(idle_intervals, float),
            np.asarray(command_intervals, float),
        ),
        None if gate is None else np.asarray(gate[keep], bool),
    )


def prefix_arm_sequence(
    sequence: ScoredSequence,
    statistic: np.ndarray,
    gate: np.ndarray | None,
) -> tuple[ScoredSequence, np.ndarray | None]:
    stop = float(sequence.command_intervals[EVALUATION_START_TRIAL - 1, 1])
    keep = sequence.times < stop
    return slice_sequence(
        sequence, statistic, gate, keep,
        sequence.idle_intervals[:EVALUATION_START_TRIAL],
        sequence.command_intervals[:EVALUATION_START_TRIAL],
    )


def tail_arm_sequence(
    sequence: ScoredSequence,
    statistic: np.ndarray,
    gate: np.ndarray | None,
) -> tuple[ScoredSequence, np.ndarray | None]:
    start = float(sequence.idle_intervals[EVALUATION_START_TRIAL, 0])
    keep = sequence.times >= start
    return slice_sequence(
        sequence, statistic, gate, keep,
        sequence.idle_intervals[EVALUATION_START_TRIAL:],
        sequence.command_intervals[EVALUATION_START_TRIAL:],
    )


def fp_for_items(items: list[tuple[ScoredSequence, np.ndarray | None]], threshold: float) -> float:
    false_alarms = 0
    idle_seconds = 0.0
    for sequence, gate in items:
        alarms = alarm_from_statistic(sequence.scores, threshold, gate)
        metrics = false_alarm_metrics(sequence.times, alarms, sequence.idle_intervals)
        false_alarms += int(metrics["false_alarm_count"])
        idle_seconds += float(metrics["valid_idle_seconds"])
    return 60.0 * false_alarms / idle_seconds


def fp_curve_vectorized(
    items: list[tuple[ScoredSequence, np.ndarray | None]],
    candidates: np.ndarray,
) -> np.ndarray:
    """Evaluate all candidate thresholds together while preserving detector state."""
    candidates = np.asarray(candidates, float)
    false_alarm_counts = np.zeros(len(candidates), dtype=np.int64)
    idle_seconds = 0.0
    dwell_steps = max(1, int(np.ceil(DWELL_SECONDS / STEP)))
    refractory_steps = max(0, int(np.ceil(REFRACTORY_SECONDS / STEP)))
    for sequence, gate in items:
        statistic = np.asarray(sequence.scores, float)
        active = np.isfinite(statistic)[None, :] & (
            statistic[None, :] >= candidates[:, None]
        )
        if gate is not None:
            active &= np.asarray(gate, bool)[None, :]
        idle = interval_mask(sequence.times, sequence.idle_intervals)
        counts = np.zeros(len(candidates), dtype=np.int32)
        blocked_until = np.full(len(candidates), -1, dtype=np.int32)
        for index in range(len(statistic)):
            counts = np.where(active[:, index], counts + 1, 0)
            fire = (counts == dwell_steps) & (index > blocked_until)
            if idle[index]:
                false_alarm_counts += fire
            blocked_until[fire] = index + refractory_steps
        idle_seconds += float(np.sum(idle) * STEP)
    return 60.0 * false_alarm_counts.astype(float) / idle_seconds


def tune_thresholds(
    items: list[tuple[ScoredSequence, np.ndarray | None]],
    targets: tuple[float, ...] = TARGETS,
    grid_size: int = 240,
) -> dict[float, tuple[float, float]]:
    finite = []
    for sequence, gate in items:
        keep = np.isfinite(sequence.scores)
        if gate is not None:
            keep &= gate
        finite.append(sequence.scores[keep])
    pooled = np.concatenate(finite)
    candidates = np.unique(np.quantile(pooled, np.linspace(0.20, 0.9999, grid_size)))
    rates = fp_curve_vectorized(items, candidates)
    curve = list(zip(candidates.astype(float), rates.astype(float)))
    return {
        float(target): min(
            curve, key=lambda item: (abs(item[1] - target), -item[0])
        )
        for target in targets
    }


def evaluate(
    sequence: ScoredSequence,
    gate: np.ndarray | None,
    threshold: float,
    offsets: np.ndarray,
) -> dict:
    alarms = alarm_from_statistic(sequence.scores, threshold, gate)
    return {
        "threshold": float(threshold),
        "onset": onset_metrics(
            sequence.times, alarms, sequence.command_intervals, offsets
        ),
        "false_alarm": false_alarm_metrics(
            sequence.times, alarms, sequence.idle_intervals
        ),
        "total_alarm_count": int(np.sum(alarms)),
    }


def analyze_one(
    path: Path,
    subject: int,
    session: int,
    shifts: int,
) -> dict:
    sequence, _ = load_cache(path)
    with np.load(path, allow_pickle=False) as data:
        raw_anchor = np.asarray(data["session_anchor_scores"], float)
    mean, variance, anchor_blocks = normalize_anchor([raw_anchor])
    normalized = ewma_normalize_and_decide(
        sequence.scores,
        initial_mean=mean,
        initial_variance=variance,
        threshold=np.inf,
        step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS,
        update_mode="all",
        ma_seconds=MA_SECONDS,
        dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS,
        sigma_floor_fraction=0.25,
    ).statistic
    raw_tail = fixed_tail(sequence, normalized)
    rng = np.random.default_rng(20260910 + 1000 * session + subject)
    offsets = rng.integers(1, len(raw_tail.scores), shifts)

    arms = {}
    for arm in ARMS:
        full_statistic, full_gate, _ = arm_components(normalized, arm)
        prefix_item = prefix_arm_sequence(sequence, full_statistic, full_gate)
        tail_item = tail_arm_sequence(sequence, full_statistic, full_gate)

        anchor_items = []
        for block in anchor_blocks:
            anchor_statistic, anchor_gate, _ = arm_components(np.asarray(block, float), arm)
            anchor_times = np.arange(len(block), dtype=float) * STEP
            anchor_sequence = ScoredSequence(
                anchor_times,
                anchor_statistic,
                np.asarray([[0.0, len(block) * STEP]], float),
                np.empty((0, 2), float),
            )
            anchor_items.append((anchor_sequence, anchor_gate))

        prospective_items = anchor_items + [prefix_item]
        prospective_points = tune_thresholds(prospective_items)
        common_points = tune_thresholds([tail_item])
        arms[arm] = {}
        for target in TARGETS:
            prospective_threshold, prospective_source_fp = prospective_points[target]
            common_threshold, common_source_fp = common_points[target]
            arms[arm][str(target)] = {
                "prospective_source_fp_per_min": float(prospective_source_fp),
                "prospective": evaluate(
                    tail_item[0], tail_item[1], prospective_threshold, offsets
                ),
                "common_source_fp_per_min": float(common_source_fp),
                "common_fp": evaluate(
                    tail_item[0], tail_item[1], common_threshold, offsets
                ),
            }
    return {"subject": subject, "session": session, "arms": arms}


def cluster_bootstrap_median(
    values: np.ndarray,
    subjects: np.ndarray,
    seed: int,
    repeats: int = 10000,
) -> list[float]:
    unique = np.unique(subjects)
    lookup = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(seed)
    group_sizes = {len(indices) for indices in lookup.values()}
    if len(group_sizes) == 1:
        grouped = np.stack([values[lookup[subject]] for subject in unique])
        sampled = rng.integers(0, len(unique), size=(repeats, len(unique)))
        draws = np.median(grouped[sampled].reshape(repeats, -1), axis=1)
        return np.quantile(draws, [0.025, 0.975]).tolist()
    draws = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([lookup[subject] for subject in sampled])
        draws.append(float(np.median(values[indices])))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def summarize(rows: list[dict]) -> dict:
    subjects = np.asarray([row["subject"] for row in rows], int)
    output = {"n_subject_sessions": len(rows), "n_subjects": len(np.unique(subjects)), "arms": {}}
    for arm_index, arm in enumerate(ARMS):
        output["arms"][arm] = {}
        for target in TARGETS:
            key = str(target)
            target_output = {}
            for operating_index, operating in enumerate(("prospective", "common_fp")):
                points = [row["arms"][arm][key][operating] for row in rows]
                metrics = {
                    "actual_fp_per_min": np.asarray([
                        point["false_alarm"]["fp_per_min"] for point in points
                    ], float),
                    "sensitivity_1s": np.asarray([
                        point["onset"]["1.0"]["sensitivity"] for point in points
                    ], float),
                    "null_1s": np.asarray([
                        point["onset"]["1.0"]["circular_shift_null"] for point in points
                    ], float),
                    "excess_1s": np.asarray([
                        point["onset"]["1.0"]["excess"] for point in points
                    ], float),
                    "sensitivity_2s": np.asarray([
                        point["onset"]["2.0"]["sensitivity"] for point in points
                    ], float),
                    "excess_2s": np.asarray([
                        point["onset"]["2.0"]["excess"] for point in points
                    ], float),
                }
                operating_output = {}
                for metric_index, (name, values) in enumerate(metrics.items()):
                    operating_output[name + "_median"] = float(np.median(values))
                    operating_output[name + "_ci95"] = cluster_bootstrap_median(
                        values, subjects,
                        202609150 + arm_index * 1000 + int(target) * 100
                        + operating_index * 20 + metric_index,
                    )
                if arm != "absolute_ma":
                    baseline = [
                        row["arms"]["absolute_ma"][key][operating] for row in rows
                    ]
                    for horizon in ("1.0", "2.0"):
                        delta = np.asarray([
                            point["onset"][horizon]["excess"]
                            - base["onset"][horizon]["excess"]
                            for point, base in zip(points, baseline)
                        ], float)
                        label = horizon.replace(".", "p")
                        operating_output[f"paired_delta_excess_{label}_vs_absolute_median"] = float(np.median(delta))
                        operating_output[f"paired_delta_excess_{label}_vs_absolute_ci95"] = cluster_bootstrap_median(
                            delta, subjects,
                            202619150 + arm_index * 1000 + int(target) * 100
                            + operating_index * 20 + int(float(horizon) * 5),
                        )
                target_output[operating] = operating_output
            output["arms"][arm][key] = target_output
    return output


def plot(summary: dict, output: Path, target: float) -> None:
    labels = ["Absolute MA", "Q10 W2", "Q10 W4", "Q10 W8", "Q10 W4 + floor0"]
    x = np.arange(len(ARMS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, operating in zip(axes, ("prospective", "common_fp")):
        values = [summary["arms"][arm][str(float(target))][operating] for arm in ARMS]
        observed = [item["sensitivity_1s_median"] for item in values]
        null = [item["null_1s_median"] for item in values]
        excess = [item["excess_1s_median"] for item in values]
        fps = [item["actual_fp_per_min_median"] for item in values]
        width = 0.24
        ax.bar(x - width, observed, width, label="SEN@1s", color="#4c72b0")
        ax.bar(x, null, width, label="shift null", color="#999999")
        ax.bar(x + width, excess, width, label="excess", color="#dd4b39")
        for index, fp in enumerate(fps):
            ax.text(index, max(observed[index], null[index], excess[index]) + 0.025,
                    f"FP {fp:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title("Prospective prefix-30 threshold" if operating == "prospective"
                     else "Same-tail matched-FP diagnostic")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Median rate")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(
        f"CNN-GRU causal local-trough decision audit — subset20, "
        f"sessions pooled, target FP={target:g}",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnn-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_scores_prospective"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_local_trough_subset20.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_local_trough_subset20.png"),
    )
    args = parser.parse_args()

    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            rows.append(analyze_one(
                path, subject, session, args.shifts
            ))
            print(f"local-trough S{session} sub-{subject:02d}", flush=True)

    combined_summary = summarize(rows)
    summary = {
        "combined": combined_summary,
        **{
            f"session{session}": summarize([
                row for row in rows if row["session"] == session
            ])
            for session in args.sessions
        },
    }
    payload = {
        "protocol": {
            "score": "frozen three-seed source-pooled CNN-GRU; causal all-score EWMA H30",
            "evaluation": "fixed final 20 trials; Lee2019 cue-relative pseudo-uncued",
            "calibration": "30-s pre-rest plus known idle/commands in preceding 30 trials",
            "local_baseline": "causal rolling Q10; latest 0.5 s excluded",
            "primary_local_arm": "local_q10_w4_floor0",
            "floor": "causal 0.5-s MA(EWMA z) >= 0",
            "decision": "0.3-s dwell; 1-s refractory",
            "thresholds": "prospective prefix-30 and same-tail diagnostic matched-FP",
            "state_boundary": "all arms preserve causal history across the final-20 evaluation boundary",
            "targets_fp_per_min": TARGETS,
            "primary_metric": "paired excess SEN@1s versus absolute MA at FP=2",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(combined_summary, args.figure, 2.0)
    fp4_figure = args.figure.with_name(args.figure.stem + "_fp4" + args.figure.suffix)
    plot(combined_summary, fp4_figure, 4.0)
    print(json.dumps({
        key: {
            "n_subject_sessions": value["n_subject_sessions"],
            "n_subjects": value["n_subjects"],
        }
        for key, value in summary.items()
    }, indent=2))


if __name__ == "__main__":
    main()
