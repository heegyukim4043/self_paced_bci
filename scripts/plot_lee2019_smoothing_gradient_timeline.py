"""Plot time-domain traces for the locked smoothing/gradient comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    HALF_LIFE_SECONDS, STEP, normalize_anchor,
)
from scripts.compare_lee2019_cnn_gru_smoothing_gradient import (
    ARMS, alarm_from_statistic, arm_statistic, tune_anchor, tune_common,
    causal_ema,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import load_cache


LABELS = {
    "ma_0.5": "MA 0.5 s (baseline)",
    "ema_0.5": "EMA half-life 0.5 s",
    "level_plus_gradient": "EMA level + positive gradient",
    "level_and_gradient": "EMA level AND gradient > 0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--stop", type=float, default=200.)
    parser.add_argument("--operating", choices=("transferred", "common_fp"),
                        default="common_fp")
    parser.add_argument("--cnn-root", type=Path,
                        default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    path = args.cnn_root / f"session{args.session}" / f"sub-{args.subject:02d}.npz"
    sequence, raw_blocks = load_cache(path)
    mean, variance, anchor_z = normalize_anchor(raw_blocks)
    z = ewma_normalize_and_decide(
        sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=np.inf, step_seconds=STEP, half_life_seconds=HALF_LIFE_SECONDS,
        update_mode="all", ma_seconds=.5, dwell_seconds=.3,
        refractory_seconds=1., sigma_floor_fraction=.25,
    ).statistic
    anchor_gradient = []
    for block in anchor_z:
        anchor_gradient.append(causal_ema(block, .2) - causal_ema(block, 1.))
    gradient_scale = max(float(np.std(np.concatenate(anchor_gradient), ddof=1)), 1e-8)

    idle = np.zeros(len(sequence.times), bool)
    for lo, hi in sequence.idle_intervals:
        idle |= (sequence.times >= lo) & (sequence.times < hi)

    traces = []
    for arm in ARMS:
        statistic, gate = arm_statistic(z, arm, gradient_scale)
        transfer, _ = tune_anchor(anchor_z, arm, gradient_scale)
        common = tune_common(statistic, gate, idle)
        threshold = transfer[0] if args.operating == "transferred" else common[0]
        alarms = alarm_from_statistic(statistic, threshold, gate)
        traces.append((arm, statistic, gate, threshold, alarms))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(traces) + 1, 1, figsize=(16, 11), sharex=True)
    mask = (sequence.times >= args.start) & (sequence.times <= args.stop)

    axes[0].plot(sequence.times[mask], z[mask], color="#777777", lw=1,
                 label="EWMA-normalized CNN-GRU score (before decision smoothing)")
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("z score")

    first_span = True
    for ax_index, (arm, statistic, gate, threshold, alarms) in enumerate(traces, start=1):
        ax = axes[ax_index]
        for lo, hi in sequence.command_intervals:
            if hi >= args.start and lo <= args.stop:
                ax.axvspan(lo, hi, color="#e98b9a", alpha=.22,
                           label="true MI interval" if first_span else None)
                first_span = False
        ax.plot(sequence.times[mask], statistic[mask], color="#2878b5", lw=1.2,
                label="decision statistic")
        ax.axhline(threshold, color="#c82433", ls="--", lw=1.2,
                   label=f"threshold {threshold:.2f}")
        alarm_mask = alarms & mask
        ax.scatter(sequence.times[alarm_mask], statistic[alarm_mask], marker="v",
                   s=32, color="black", zorder=5, label="alarm")
        if gate is not None:
            rejected = mask & np.isfinite(statistic) & (statistic >= threshold) & ~gate
            ax.scatter(sequence.times[rejected], statistic[rejected], marker="x",
                       s=18, color="#f28e2b", label="level pass, gradient reject")
        ax.set_ylabel("statistic")
        ax.set_title(LABELS[arm], loc="left", fontsize=10)
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="upper right", fontsize=7, ncol=4)

    axes[-1].set_xlabel("Time in evaluation stream (s)")
    axes[-1].set_xlim(args.start, args.stop)
    fig.suptitle(
        f"CNN-GRU smoothing/gradient time-domain comparison — Lee2019 "
        f"subject {args.subject:02d}, session {args.session} ({args.operating})"
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    output = args.output or Path(
        f"data/processed/lee2019_smoothing_gradient_timeline_sub{args.subject:02d}_"
        f"s{args.session}_{args.operating}_{args.start:g}-{args.stop:g}s.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
