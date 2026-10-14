"""Fast local-trough / slow robust-scale prominence audit.

Frozen CNN-GRU and EWMA scores are compared using three decision coordinates:
absolute MA level, W2 local-trough subtraction, and W2 local prominence divided
by a causal 30-s rolling IQR.  The primary endpoint is prospective operating-
point admissibility; matched-FP excess must remain non-inferior.
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
from uncued_mi.operating_point import ScoredSequence
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS, HALF_LIFE_SECONDS, MA_SECONDS, REFRACTORY_SECONDS, STEP,
    normalize_anchor,
)
from scripts.compare_lee2019_cnn_gru_local_trough import (
    alarm_from_statistic,
    causal_local_trough,
    causal_ma,
    cluster_bootstrap_median,
    evaluate,
    prefix_arm_sequence,
    tail_arm_sequence,
    tune_thresholds,
)
from scripts.evaluate_lee2019_cnn_gru_idle_exposure_curve import (
    EVALUATION_START_TRIAL,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache


TARGETS = (2.0, 4.0)
ARMS = ("absolute_ma", "local_q10_w2", "prominence_w2_iqr30")
SLOW_SECONDS = 30.0
GAP_SECONDS = 0.5
EPSILON = 0.1


def rolling_iqr(
    level: np.ndarray,
    history_seconds: float = SLOW_SECONDS,
    gap_seconds: float = GAP_SECONDS,
) -> np.ndarray:
    level = np.asarray(level, float)
    history_steps = max(1, int(np.ceil(history_seconds / STEP)))
    gap_steps = max(1, int(np.ceil(gap_seconds / STEP)))
    result = np.full(len(level), np.nan, float)
    # The stream is prepended with the completed 30-s pre-rest, so a full
    # rolling window is available at deployment time. Calibration-anchor
    # handling is separate below.
    for index in range(len(level)):
        stop = index - gap_steps + 1
        start = stop - history_steps
        if start < 0 or stop <= start:
            continue
        history = level[start:stop]
        history = history[np.isfinite(history)]
        if len(history) >= int(0.9 * history_steps):
            q25, q75 = np.quantile(history, [0.25, 0.75])
            result[index] = float(q75 - q25)
    return result


def component_from_combined(
    combined_z: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    level = causal_ma(combined_z)
    if arm == "absolute_ma":
        return level, None, {"level": level}
    local, baseline = causal_local_trough(level, 2.0, GAP_SECONDS, 0.10)
    if arm == "local_q10_w2":
        return local, None, {"level": level, "baseline": baseline}
    scale = rolling_iqr(level)
    prominence = local / (scale + EPSILON)
    return prominence, None, {
        "level": level, "baseline": baseline, "scale": scale,
    }


def anchor_component(anchor_z: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray | None]:
    level = causal_ma(anchor_z)
    if arm == "absolute_ma":
        return level, None
    local, _ = causal_local_trough(level, 2.0, GAP_SECONDS, 0.10)
    if arm == "local_q10_w2":
        return local, None
    finite = level[np.isfinite(level)]
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    return local / (float(q75 - q25) + EPSILON), None


def anchor_item(statistic: np.ndarray, gate: np.ndarray | None) -> tuple[ScoredSequence, np.ndarray | None]:
    times = np.arange(len(statistic), dtype=float) * STEP
    return (
        ScoredSequence(
            times,
            np.asarray(statistic, float),
            np.asarray([[0.0, len(statistic) * STEP]], float),
            np.empty((0, 2), float),
        ),
        gate,
    )


def analyze_one(path: Path, subject: int, session: int, shifts: int) -> dict:
    sequence, _ = load_cache(path)
    with np.load(path, allow_pickle=False) as data:
        raw_anchor = np.asarray(data["session_anchor_scores"], float)
    mean, variance, anchor_blocks = normalize_anchor([raw_anchor])
    anchor_z = np.asarray(anchor_blocks[0], float)
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
    combined_z = np.r_[anchor_z, normalized]
    sequence_offset = len(anchor_z)
    null_seed = 20260910 + 1000 * session + subject

    output = {}
    for arm in ARMS:
        combined_statistic, combined_gate, _ = component_from_combined(combined_z, arm)
        statistic = combined_statistic[sequence_offset:]
        gate = None if combined_gate is None else combined_gate[sequence_offset:]
        prefix = prefix_arm_sequence(sequence, statistic, gate)
        tail = tail_arm_sequence(sequence, statistic, gate)
        anchor_statistic, anchor_gate = anchor_component(anchor_z, arm)
        prospective_items = [anchor_item(anchor_statistic, anchor_gate), prefix]
        prospective_points = tune_thresholds(prospective_items, TARGETS)
        common_points = tune_thresholds([tail], TARGETS)
        offsets = np.random.default_rng(null_seed).integers(
            1, len(tail[0].scores), shifts
        )
        output[arm] = {}
        for target in TARGETS:
            prospective_threshold, prospective_source_fp = prospective_points[target]
            common_threshold, common_source_fp = common_points[target]
            output[arm][str(target)] = {
                "prospective_source_fp_per_min": float(prospective_source_fp),
                "prospective": evaluate(
                    tail[0], tail[1], prospective_threshold, offsets
                ),
                "common_source_fp_per_min": float(common_source_fp),
                "common_fp": evaluate(
                    tail[0], tail[1], common_threshold, offsets
                ),
            }
    return {"subject": subject, "session": session, "arms": output}


def cluster_bootstrap_mean(
    values: np.ndarray, subjects: np.ndarray, seed: int, repeats: int = 10000
) -> list[float]:
    unique = np.unique(subjects)
    lookup = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    grouped = np.stack([values[lookup[subject]] for subject in unique])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(repeats, len(unique)))
    draws = np.mean(grouped[sampled].reshape(repeats, -1), axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def op_error(fp: np.ndarray, target: float) -> np.ndarray:
    return np.abs(np.log((np.asarray(fp, float) + 0.05) / (target + 0.05)))


def admissible(fp: np.ndarray, target: float) -> np.ndarray:
    lower, upper = ((1.5, 2.5) if target == 2.0 else (3.0, 5.0))
    return ((fp >= lower) & (fp <= upper)).astype(float)


def summarize(rows: list[dict]) -> dict:
    subjects = np.asarray([row["subject"] for row in rows], int)
    result = {
        "n_subject_sessions": len(rows),
        "n_subjects": len(np.unique(subjects)),
        "arms": {},
    }
    for arm_index, arm in enumerate(ARMS):
        result["arms"][arm] = {}
        for target in TARGETS:
            key = str(target)
            target_result = {}
            for operating_index, operating in enumerate(("prospective", "common_fp")):
                points = [row["arms"][arm][key][operating] for row in rows]
                fp = np.asarray([
                    point["false_alarm"]["fp_per_min"] for point in points
                ], float)
                excess1 = np.asarray([
                    point["onset"]["1.0"]["excess"] for point in points
                ], float)
                excess2 = np.asarray([
                    point["onset"]["2.0"]["excess"] for point in points
                ], float)
                sensitivity1 = np.asarray([
                    point["onset"]["1.0"]["sensitivity"] for point in points
                ], float)
                null1 = np.asarray([
                    point["onset"]["1.0"]["circular_shift_null"] for point in points
                ], float)
                error = op_error(fp, target)
                valid = admissible(fp, target)
                item = {
                    "actual_fp_per_min_median": float(np.median(fp)),
                    "op_log_error_median": float(np.median(error)),
                    "admissible_n": int(np.sum(valid)),
                    "admissible_rate": float(np.mean(valid)),
                    "sensitivity_1s_median": float(np.median(sensitivity1)),
                    "null_1s_median": float(np.median(null1)),
                    "excess_1s_median": float(np.median(excess1)),
                    "excess_2s_median": float(np.median(excess2)),
                }
                if arm != "absolute_ma":
                    baseline = [
                        row["arms"]["absolute_ma"][key][operating] for row in rows
                    ]
                    baseline_fp = np.asarray([
                        point["false_alarm"]["fp_per_min"] for point in baseline
                    ], float)
                    baseline_excess1 = np.asarray([
                        point["onset"]["1.0"]["excess"] for point in baseline
                    ], float)
                    baseline_excess2 = np.asarray([
                        point["onset"]["2.0"]["excess"] for point in baseline
                    ], float)
                    deltas = {
                        "op_log_error": error - op_error(baseline_fp, target),
                        "admissible_rate": valid - admissible(baseline_fp, target),
                        "excess_1s": excess1 - baseline_excess1,
                        "excess_2s": excess2 - baseline_excess2,
                    }
                    for delta_index, (name, values) in enumerate(deltas.items()):
                        if name == "admissible_rate":
                            item["paired_delta_admissible_rate_mean"] = float(np.mean(values))
                            item["paired_delta_admissible_rate_ci95"] = cluster_bootstrap_mean(
                                values, subjects,
                                202609150 + arm_index * 1000 + int(target) * 100
                                + operating_index * 20 + delta_index,
                            )
                        else:
                            item[f"paired_delta_{name}_median"] = float(np.median(values))
                            item[f"paired_delta_{name}_ci95"] = cluster_bootstrap_median(
                                values, subjects,
                                202619150 + arm_index * 1000 + int(target) * 100
                                + operating_index * 20 + delta_index,
                            )
                target_result[operating] = item
            result["arms"][arm][key] = target_result
    return result


def plot(summary: dict, output: Path) -> None:
    labels = ("Absolute MA", "W2 trough", "W2 / IQR30")
    x = np.arange(len(ARMS))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, target in enumerate(TARGETS):
        prospective = [summary["arms"][arm][str(target)]["prospective"] for arm in ARMS]
        common = [summary["arms"][arm][str(target)]["common_fp"] for arm in ARMS]
        ax = axes[0, column]
        fp = [item["actual_fp_per_min_median"] for item in prospective]
        rates = [100 * item["admissible_rate"] for item in prospective]
        ax.bar(x - 0.18, fp, 0.36, label="actual FP/min", color="#4c72b0")
        ax2 = ax.twinx()
        ax2.bar(x + 0.18, rates, 0.36, label="admissible %", color="#55a868")
        ax.axhline(target, color="#c44e52", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel("Actual FP/min")
        ax2.set_ylabel("Admissible sessions (%)")
        ax.set_title(f"Prospective operating point, target FP={target:g}")
        ax.grid(axis="y", alpha=0.2)
        if column == 0:
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)

        ax = axes[1, column]
        observed = [item["sensitivity_1s_median"] for item in common]
        null = [item["null_1s_median"] for item in common]
        excess = [item["excess_1s_median"] for item in common]
        width = 0.24
        ax.bar(x - width, observed, width, label="SEN@1s", color="#4c72b0")
        ax.bar(x, null, width, label="shift null", color="#999999")
        ax.bar(x + width, excess, width, label="excess", color="#dd4b39")
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_title(f"Same-tail matched-FP information, target FP={target:g}")
        ax.grid(axis="y", alpha=0.2)
        if column == 0:
            ax.legend(ncol=3, fontsize=8)
    fig.suptitle("CNN-GRU fast-trough / slow-scale normalized prominence audit")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
        default=Path("data/processed/lee2019_cnn_gru_normalized_prominence_subset20.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_normalized_prominence_subset20.png"),
    )
    args = parser.parse_args()

    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            rows.append(analyze_one(path, subject, session, args.shifts))
            print(f"normalized-prominence S{session} sub-{subject:02d}", flush=True)
    summaries = {
        "combined": summarize(rows),
        **{
            f"session{session}": summarize([
                row for row in rows if row["session"] == session
            ])
            for session in args.sessions
        },
    }
    payload = {
        "protocol": {
            "score": "frozen three-seed source-pooled CNN-GRU plus causal all-score EWMA H30",
            "arms": ARMS,
            "prominence": "(MA.5-Q10[past 2 s excluding latest .5 s])/(IQR[past 30 s excluding latest .5 s]+.1)",
            "prehistory": "completed 30-s target pre-rest initializes causal rolling context",
            "decision": "dwell .3 s; refractory 1 s",
            "calibration": "30-s pre-rest plus preceding 30 labeled opportunities; final 20 evaluated",
            "primary_endpoint": "prospective admissible rate; matched-FP excess@1s must be non-inferior",
            "admissible": {"2.0": [1.5, 2.5], "4.0": [3.0, 5.0]},
            "tie_break": "highest threshold; command hits never used",
            "claim_limit": "Lee2019 cue-relative pseudo-uncued subset20 exploratory audit",
        },
        "summary": summaries,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(summaries["combined"], args.figure)
    print(json.dumps({
        key: {"n": value["n_subject_sessions"]} for key, value in summaries.items()
    }, indent=2))


if __name__ == "__main__":
    main()
