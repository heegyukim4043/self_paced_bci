"""Plot threshold displacement and local alarm-rate conditioning audit."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/lee2019_cnn_gru_threshold_conditioning.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_threshold_conditioning.png"))
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, session in enumerate((1, 2)):
        chosen = [row for row in rows if row["session"] == session]
        ax = axes[0, column]
        displacement = np.asarray([row["threshold_displacement"] for row in chosen])
        fp = np.asarray([row["deployed_fp_per_min"] for row in chosen])
        ax.scatter(displacement, fp, color="#1f77b4")
        ax.axhspan(1.5, 2.5, color="#2ca02c", alpha=.12, label="admissible FP")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Session {session}: displacement vs deployed FP")
        ax.set_xlabel(r"$\tau_{transfer}-\tau_{commonFP}$")
        ax.set_ylabel("Actual FP/min")
        ax.grid(alpha=.25)
        ax = axes[1, column]
        offsets = np.asarray([point["offset"] for point in chosen[0]["local_response"]])
        curves = np.asarray([[point["fp_per_min"] for point in row["local_response"]] for row in chosen])
        median = np.median(curves, axis=0)
        lo, hi = np.quantile(curves, [.25, .75], axis=0)
        ax.plot(offsets, median, marker="o", color="#d62728")
        ax.fill_between(offsets, lo, hi, color="#d62728", alpha=.15, label="IQR")
        ax.axhline(2, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"Session {session}: local alarm response")
        ax.set_xlabel(r"Threshold offset from $\tau_{transfer}$")
        ax.set_ylabel("Actual FP/min")
        ax.grid(alpha=.25)
    fig.suptitle("CNN-GRU EWMA threshold conditioning audit", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
