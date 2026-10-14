"""Subject-level EWMA-only versus oracle alarm-controller comparison figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS, HALF_LIFE_SECONDS, MA_SECONDS, REFRACTORY_SECONDS, STEP,
    normalize_anchor, tune_ma_dwell,
)
from scripts.audit_lee2019_cnn_gru_onset_false_alarms import (
    false_alarm_metrics, onset_metrics,
)
from scripts.run_lee2019_cnn_gru_oracle_alarm_controller import (
    controlled_decision, interval_mask,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration


def moving_average(values):
    steps = max(1, int(np.ceil(MA_SECONDS / STEP)))
    result = np.convolve(values, np.ones(steps) / steps, mode="full")[:len(values)]
    result[:steps - 1] = np.nan
    return result


def auc(positive, negative):
    values = np.r_[positive, negative]
    ranks = rankdata(values)
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def shade_commands(ax, commands):
    for start, stop in commands:
        ax.axvspan(start, stop, color="#e78ca0", alpha=.28, linewidth=0)


def alarm_types(times, alarms, commands):
    command = interval_mask(times, commands)
    return alarms & command, alarms & ~command


def trace_panel(ax, times, score_margin, ma_margin, alarms, commands, title):
    shade_commands(ax, commands)
    ax.plot(times, score_margin, color="#4c72ff", linewidth=1.0,
            label="EWMA z - threshold")
    ax.plot(times, ma_margin, color="#ff5733", linewidth=1.1,
            label="MA(z) - threshold")
    tp, fp = alarm_types(times, alarms, commands)
    ax.scatter(times[tp], np.zeros(tp.sum()), marker="v", s=27, color="black",
               zorder=5, label="alarm in MI")
    ax.scatter(times[fp], np.zeros(fp.sum()), marker="x", s=35, color="#b30000",
               zorder=6, label="false alarm")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("Margin")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path,
                        default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--gain", type=float, default=.8)
    parser.add_argument("--zoom-start", type=float, default=0.)
    parser.add_argument("--zoom-stop", type=float, default=60.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.cnn_root / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    sequence, blocks = load_cache(path)
    mean, variance, normalized = normalize_anchor(blocks)
    point = tune_ma_dwell([calibration(x) for x in normalized], 160)
    baseline = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=point.threshold, step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS, update_mode="all",
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=.25,
    )
    idle = interval_mask(sequence.times, sequence.idle_intervals)
    oracle_alarms, oracle_threshold, oracle_ma = controlled_decision(
        baseline.statistic, idle, point.threshold, args.gain, 2.0
    )
    baseline_ma = moving_average(baseline.statistic)
    base_margin = baseline.statistic - point.threshold
    base_ma_margin = baseline_ma - point.threshold
    oracle_margin = baseline.statistic - oracle_threshold
    oracle_ma_margin = oracle_ma - oracle_threshold

    rng = np.random.default_rng(960000 + args.session * 1000 + args.subject)
    offsets = rng.integers(1, len(sequence.times), 500)
    base_on = onset_metrics(sequence.times, baseline.alarms, sequence.command_intervals, offsets)
    oracle_on = onset_metrics(sequence.times, oracle_alarms, sequence.command_intervals, offsets)
    base_fp = false_alarm_metrics(sequence.times, baseline.alarms, sequence.idle_intervals)
    oracle_fp = false_alarm_metrics(sequence.times, oracle_alarms, sequence.idle_intervals)
    mi = interval_mask(sequence.times, sequence.command_intervals)
    window_auc = auc(baseline.statistic[mi], baseline.statistic[idle])

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, .85), hspace=.48, wspace=.25)
    ax1 = fig.add_subplot(grid[0, :])
    trace_panel(
        ax1, sequence.times, base_margin, base_ma_margin, baseline.alarms,
        sequence.command_intervals,
        f"(a) EWMA only — FP {base_fp['fp_per_min']:.2f}/min, "
        f"SEN@1s {base_on['1.0']['sensitivity']:.2f}, "
        f"excess {base_on['1.0']['excess']:+.2f}",
    )
    ax2 = fig.add_subplot(grid[1, :], sharex=ax1)
    trace_panel(
        ax2, sequence.times, oracle_margin, oracle_ma_margin, oracle_alarms,
        sequence.command_intervals,
        f"(b) Oracle-idle controller (gain {args.gain:g}) — "
        f"FP {oracle_fp['fp_per_min']:.2f}/min, "
        f"SEN@1s {oracle_on['1.0']['sensitivity']:.2f}, "
        f"excess {oracle_on['1.0']['excess']:+.2f}",
    )
    ax2.set_xlabel("Time in evaluation stream (s)")

    ax3 = fig.add_subplot(grid[2, 0])
    shade_commands(ax3, sequence.command_intervals)
    mask = (sequence.times >= args.zoom_start) & (sequence.times <= args.zoom_stop)
    ax3.plot(sequence.times[mask], base_ma_margin[mask], color="#1f77b4",
             label="EWMA-only MA margin")
    ax3.plot(sequence.times[mask], oracle_ma_margin[mask], color="#d62728",
             label="oracle MA margin")
    ax3.axhline(0, color="black", linewidth=.8)
    ax3.set_xlim(args.zoom_start, args.zoom_stop)
    ax3.set_title(f"(c) MA-margin comparison, {args.zoom_start:g}-{args.zoom_stop:g} s")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("MA margin")
    ax3.legend(fontsize=8); ax3.grid(alpha=.16)

    ax4 = fig.add_subplot(grid[2, 1])
    labels = ("SEN@0.5s", "SEN@1s", "SEN@2s", "Null@1s", "FP/min")
    base_values = [base_on[str(x)]["sensitivity"] for x in (.5, 1., 2.)]
    base_values += [base_on["1.0"]["circular_shift_null"], base_fp["fp_per_min"]]
    oracle_values = [oracle_on[str(x)]["sensitivity"] for x in (.5, 1., 2.)]
    oracle_values += [oracle_on["1.0"]["circular_shift_null"], oracle_fp["fp_per_min"]]
    x = np.arange(len(labels)); width = .36
    ax4.bar(x - width / 2, base_values, width, label="EWMA only", color="#4c72b0")
    ax4.bar(x + width / 2, oracle_values, width, label="Oracle controller", color="#dd8452")
    ax4.set_xticks(x, labels, rotation=20, ha="right")
    ax4.set_title("(d) Onset detection and false alarms")
    ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=.2)

    handles, labels_legend = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"CNN-GRU threshold comparison — Lee2019 subject {args.subject:02d}, "
        f"session {args.session} (window AUC {window_auc:.2f})",
        y=.985, fontsize=15,
    )
    fig.subplots_adjust(bottom=.09, top=.93)
    output = args.output or Path(
        f"data/processed/lee2019_cnn_gru_ewma_oracle_s{args.subject:02d}_session{args.session}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
