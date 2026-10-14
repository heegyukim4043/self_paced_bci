"""Plot prospective-prefix versus same-tail oracle CNN-GRU thresholds.

This is a subject-level diagnostic visualization for the known-idle exposure
experiment.  Both arms use the same frozen score trace and evaluation stream;
only the threshold differs.  The same-tail arm uses evaluation idle labels and
is therefore an oracle diagnostic, not a deployable calibration method.
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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.decision import moving_average_dwell
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS,
    HALF_LIFE_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    STEP,
    normalize_anchor,
)
from scripts.evaluate_lee2019_cnn_gru_idle_exposure_curve import fixed_tail
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import interval_mask
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import load_cache


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values)
    n_positive = len(positive)
    return float(
        (ranks[:n_positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * len(negative))
    )


def rebase_intervals(intervals: np.ndarray, origin: float) -> np.ndarray:
    return np.asarray(intervals, float) - float(origin)


def shade_commands(ax, intervals: np.ndarray) -> None:
    for start, stop in intervals:
        ax.axvspan(start, stop, color="#e78ca0", alpha=0.28, linewidth=0)


def trace_panel(
    ax,
    times: np.ndarray,
    score_margin: np.ndarray,
    ma_margin: np.ndarray,
    alarms: np.ndarray,
    command_mask: np.ndarray,
    commands: np.ndarray,
    title: str,
) -> None:
    shade_commands(ax, commands)
    ax.plot(times, score_margin, color="#4c72ff", linewidth=0.9,
            alpha=0.72, label="EWMA z - threshold")
    ax.plot(times, ma_margin, color="#ff5733", linewidth=1.15,
            label="MA(z) - threshold")
    true_alarms = alarms & command_mask
    false_alarms = alarms & ~command_mask
    ax.scatter(times[true_alarms], np.zeros(np.sum(true_alarms)), marker="v",
               s=29, color="black", zorder=5, label="alarm in MI")
    ax.scatter(times[false_alarms], np.zeros(np.sum(false_alarms)), marker="x",
               s=42, linewidths=1.5, color="#b30000", zorder=6,
               label="false alarm")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Margin")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.16)


def find_row(payload: dict, subject: int, session: int) -> dict:
    for row in payload["rows"]:
        if row["subject"] == subject and row["session"] == session:
            return row
    raise ValueError(f"No result row for subject {subject}, session {session}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--target-fp", type=float, choices=(2.0, 4.0), default=2.0)
    parser.add_argument("--zoom-start", type=float, default=0.0)
    parser.add_argument("--zoom-stop", type=float, default=60.0)
    parser.add_argument(
        "--cnn-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_scores_prospective"),
    )
    parser.add_argument(
        "--results", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_idle_exposure_curve_subset20.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache_path = (
        args.cnn_root / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    )
    sequence, _ = load_cache(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        raw_anchor = np.asarray(data["session_anchor_scores"], float)

    mean, variance, _ = normalize_anchor([raw_anchor])
    normalized = ewma_normalize_and_decide(
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
    tail = fixed_tail(sequence, np.asarray(normalized.statistic, float))

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    row = find_row(payload, args.subject, args.session)
    target_key = str(float(args.target_fp))
    prospective_metrics = row["arms"]["prefix30"][target_key]
    oracle_metrics = row["arms"]["tail20_oracle"][target_key]
    prospective_threshold = float(prospective_metrics["threshold"])
    oracle_threshold = float(oracle_metrics["threshold"])

    prospective = moving_average_dwell(
        tail.scores, prospective_threshold, STEP, MA_SECONDS,
        DWELL_SECONDS, REFRACTORY_SECONDS,
    )
    oracle = moving_average_dwell(
        tail.scores, oracle_threshold, STEP, MA_SECONDS,
        DWELL_SECONDS, REFRACTORY_SECONDS,
    )

    origin = float(tail.times[0])
    times = np.asarray(tail.times, float) - origin
    commands = rebase_intervals(tail.command_intervals, origin)
    command_mask = interval_mask(tail.times, tail.command_intervals)
    idle_mask = interval_mask(tail.times, tail.idle_intervals)
    window_auc = auc(tail.scores[command_mask], tail.scores[idle_mask])

    prospective_score_margin = tail.scores - prospective_threshold
    oracle_score_margin = tail.scores - oracle_threshold
    prospective_ma_margin = prospective.statistic - prospective_threshold
    oracle_ma_margin = oracle.statistic - oracle_threshold

    exposure = float(row["known_idle_exposure_seconds"]["prefix30"])
    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 0.85),
                            hspace=0.48, wspace=0.25)
    ax1 = fig.add_subplot(grid[0, :])
    trace_panel(
        ax1, times, prospective_score_margin, prospective_ma_margin,
        prospective.alarms, command_mask, commands,
        f"(a) Prospective {exposure:.1f}-s idle calibration — "
        f"threshold {prospective_threshold:.3f}, "
        f"FP {prospective_metrics['actual_fp_per_min']:.2f}/min, "
        f"SEN@1s {prospective_metrics['onset']['1.0']['sensitivity']:.2f}, "
        f"excess {prospective_metrics['onset']['1.0']['excess']:+.2f}",
    )
    ax2 = fig.add_subplot(grid[1, :], sharex=ax1)
    trace_panel(
        ax2, times, oracle_score_margin, oracle_ma_margin,
        oracle.alarms, command_mask, commands,
        f"(b) Same-tail oracle (diagnostic only) — "
        f"threshold {oracle_threshold:.3f}, "
        f"FP {oracle_metrics['actual_fp_per_min']:.2f}/min, "
        f"SEN@1s {oracle_metrics['onset']['1.0']['sensitivity']:.2f}, "
        f"excess {oracle_metrics['onset']['1.0']['excess']:+.2f}",
    )
    ax2.set_xlabel("Time in fixed final-20-trial evaluation stream (s)")

    ax3 = fig.add_subplot(grid[2, 0])
    shade_commands(ax3, commands)
    zoom = (times >= args.zoom_start) & (times <= args.zoom_stop)
    ax3.plot(times[zoom], prospective.statistic[zoom], color="#505050",
             linewidth=1.25, label="MA(z), common trace")
    ax3.axhline(prospective_threshold, color="#1f77b4", linestyle="--",
                linewidth=1.4, label=f"prospective thr = {prospective_threshold:.3f}")
    ax3.axhline(oracle_threshold, color="#d62728", linestyle="--",
                linewidth=1.4, label=f"same-tail thr = {oracle_threshold:.3f}")
    ax3.set_xlim(args.zoom_start, args.zoom_stop)
    ax3.set_title(
        f"(c) Absolute MA score and thresholds, "
        f"{args.zoom_start:g}-{args.zoom_stop:g} s"
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("MA-normalized score")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.16)

    ax4 = fig.add_subplot(grid[2, 1])
    labels = ("SEN@0.5s", "SEN@1s", "SEN@2s", "Null@1s", "FP/min")
    prospective_values = [
        prospective_metrics["onset"][str(value)]["sensitivity"]
        for value in (0.5, 1.0, 2.0)
    ] + [
        prospective_metrics["onset"]["1.0"]["circular_shift_null"],
        prospective_metrics["actual_fp_per_min"],
    ]
    oracle_values = [
        oracle_metrics["onset"][str(value)]["sensitivity"]
        for value in (0.5, 1.0, 2.0)
    ] + [
        oracle_metrics["onset"]["1.0"]["circular_shift_null"],
        oracle_metrics["actual_fp_per_min"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    ax4.bar(x - width / 2, prospective_values, width,
            label="Prospective", color="#4c72b0")
    ax4.bar(x + width / 2, oracle_values, width,
            label="Same-tail oracle", color="#dd8452")
    ax4.set_xticks(x, labels, rotation=20, ha="right")
    ax4.set_title("(d) Onset detection and false alarms")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.2)

    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5,
               frameon=False)
    fig.suptitle(
        f"CNN-GRU prospective-vs-oracle threshold comparison — "
        f"Lee2019 subject {args.subject:02d}, session {args.session} "
        f"(target FP={args.target_fp:g}/min, window AUC {window_auc:.2f})",
        y=0.985, fontsize=15,
    )
    fig.subplots_adjust(bottom=0.09, top=0.93)

    output = args.output or Path(
        "data/processed/"
        f"lee2019_cnn_gru_prospective_vs_tail_oracle_s{args.subject:02d}_"
        f"session{args.session}_fp{args.target_fp:g}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
