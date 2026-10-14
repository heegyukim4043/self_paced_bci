"""Plot the prospective known-idle exposure curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREFIX_ARMS = ("prefix00", "prefix05", "prefix10", "prefix20", "prefix30")


def errorbar(ax, x, values, intervals, **kwargs):
    values = np.asarray(values, float)
    intervals = np.asarray(intervals, float)
    yerr = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))
    ax.errorbar(x, values, yerr=yerr, capsize=3, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/lee2019_cnn_gru_idle_exposure_curve_subset20.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/lee2019_cnn_gru_idle_exposure_curve_subset20.png"),
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    summary = data["summary"]["combined"]["arms"]
    x = [summary[arm]["known_idle_exposure_seconds_median"] for arm in PREFIX_ARMS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, target in enumerate(("2.0", "4.0")):
        target_value = float(target)
        fp = [summary[arm][target]["actual_fp_per_min_median"] for arm in PREFIX_ARMS]
        fp_ci = [summary[arm][target]["actual_fp_per_min_ci95"] for arm in PREFIX_ARMS]
        excess = [summary[arm][target]["excess_1s_median"] for arm in PREFIX_ARMS]
        excess_ci = [summary[arm][target]["excess_1s_ci95"] for arm in PREFIX_ARMS]
        oracle_fp = summary["tail20_oracle"][target]["actual_fp_per_min_median"]
        oracle_excess = summary["tail20_oracle"][target]["excess_1s_median"]

        errorbar(
            axes[0, column], x, fp, fp_ci,
            marker="o", linewidth=2, color="#276FBF", label="prospective prefix",
        )
        axes[0, column].axhline(target_value, color="#C5283D", linestyle=":", label="target FP")
        axes[0, column].axhline(oracle_fp, color="#555555", linestyle="--", label="same-tail oracle")
        axes[0, column].set_title(f"Nominal FP={target_value:g}/min")
        axes[0, column].set_ylabel("Achieved FP/min")
        axes[0, column].grid(alpha=0.25)

        errorbar(
            axes[1, column], x, excess, excess_ci,
            marker="o", linewidth=2, color="#008148", label="prospective prefix",
        )
        axes[1, column].axhline(oracle_excess, color="#555555", linestyle="--", label="same-tail oracle")
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].set_ylabel("1-s onset excess sensitivity")
        axes[1, column].set_xlabel("Known-idle calibration exposure (s; includes 30-s anchor)")
        axes[1, column].grid(alpha=0.25)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Frozen CNN-GRU prospective threshold calibration: idle-exposure curve\n"
        "Lee2019 subset20 x 2 sessions; fixed final-20-trial evaluation",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
