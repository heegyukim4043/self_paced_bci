"""Plot the locked matched-achieved-FP CNN-GRU/Riemannian diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_vs_riemannian_matched_fp.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_vs_riemannian_matched_fp.png"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    targets = sorted(float(key) for key in payload["summary"])

    metrics = (
        ("excess_1.0", "Circular-shift excess sensitivity @ 1 s"),
        ("excess_2.0", "Circular-shift excess sensitivity @ 2 s"),
        ("coverage", "Interval coverage"),
    )
    colors = {"cnn_gru": "#d95f02", "riemannian": "#1b78b4"}
    labels = {"cnn_gru": "CNN-GRU", "riemannian": "Riemannian Arm A"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=False)

    for row_index, session in enumerate((1, 2)):
        for col_index, (metric, title) in enumerate(metrics):
            ax = axes[row_index, col_index]
            x = np.asarray([
                payload["summary"][str(target)]["session_medians"][str(session)][
                    "matched_actual_fp_per_min"
                ]
                for target in targets
            ])
            for model in ("cnn_gru", "riemannian"):
                y = np.asarray([
                    payload["summary"][str(target)]["session_medians"][str(session)][model][metric]
                    for target in targets
                ])
                ax.plot(x, y, "o-", lw=2, ms=5, color=colors[model], label=labels[model])
            ax.axhline(0, color="0.35", lw=.8)
            ax.grid(alpha=.25)
            ax.set_title(f"S{session}: {title}")
            ax.set_xlabel("Matched achieved false alarms/min")
            ax.set_ylabel(title)
            if col_index < 2:
                for target in (2., 4.):
                    delta = payload["summary"][str(target)][
                        "paired_delta_cnn_minus_riemannian"
                    ][metric][f"session{session}"]
                    lo, hi = delta["cluster_bootstrap_95ci"]
                    y_top = ax.get_ylim()[1]
                    ax.text(
                        target, y_top - .04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f"Δ@{target:g}={delta['median_delta']:+.3f}\n[{lo:+.3f}, {hi:+.3f}]",
                        ha="center", va="top", fontsize=8,
                        bbox={"facecolor": "white", "alpha": .78, "edgecolor": "0.8"},
                    )
            if row_index == 0 and col_index == 0:
                ax.legend(loc="best")

    fig.suptitle(
        "Lee2019 locked stream: CNN-GRU vs Riemannian at identical achieved FP\n"
        "Upper-tail threshold branch; evaluation-idle oracle (diagnostic, not deployable)",
        fontsize=14,
    )
    fig.text(
        .5, .012,
        "54 subjects × 2 sessions; cue+0.5 s pseudo-onset; EWMA H30, MA 0.5 s, "
        "dwell 0.3 s, refractory 1 s; 500 circular shifts",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, .035, 1, .92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
