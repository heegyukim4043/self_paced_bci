"""Plot FP=2 artifact rate ratios for Lee2019 and Cho2017."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def panel(axis, labels, points, intervals, title):
    y = np.arange(len(labels))
    lower = np.asarray(points) - np.asarray([value[0] for value in intervals])
    upper = np.asarray([value[1] for value in intervals]) - np.asarray(points)
    colors = ["#DC2626" if low > 1.0 else "#2563EB" for low, _ in intervals]
    for index, color in enumerate(colors):
        axis.errorbar(
            points[index],
            y[index],
            xerr=np.asarray([[lower[index]], [upper[index]]]),
            fmt="none",
            ecolor=color,
            capsize=3,
            linewidth=1.5,
        )
    axis.scatter(points, y, c=colors, s=38, zorder=3)
    axis.axvline(1.0, color="#111827", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("FP rate ratio vs clean held-out rest")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.7)


def main() -> None:
    root = Path("data/processed")
    lee = json.loads((root / "lee2019_artifact_stress.json").read_text(encoding="utf-8"))
    cho = json.loads((root / "cho2017_transfer.json").read_text(encoding="utf-8"))

    lee_names = {
        "EMG_bothHandGrasping": "Both-hand grasp",
        "eyemovement_blinking": "Eye blink",
        "eyemovement_horizen": "Horizontal eye",
        "eyemovement_vertical": "Vertical eye",
        "teeth": "Teeth clench",
    }
    lee_rows = lee["summary"]
    cho_rows = [
        row for row in cho["noise_summary"] if row["target_fp_setting"] == 2.0
    ]
    cho_names = {
        "eye_blinking": "Eye blink",
        "eyeball_up_down": "Eye up/down",
        "eyeball_left_right": "Eye left/right",
        "jaw_clenching": "Jaw clench",
        "head_left_right": "Head left/right",
    }

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    panel(
        axes[0],
        [lee_names[row["artifact"]] for row in lee_rows],
        [row["rate_ratio_vs_clean_post"] for row in lee_rows],
        [row["rate_ratio_cluster_bootstrap_95_ci"] for row in lee_rows],
        "Lee2019 (54 subjects × 2 sessions)",
    )
    panel(
        axes[1],
        [cho_names[row["noise"]] for row in cho_rows],
        [row["rate_ratio_vs_clean_rest"] for row in cho_rows],
        [row["rate_ratio_subject_bootstrap_95_ci"] for row in cho_rows],
        "Cho2017 (50 subjects)",
    )
    axes[0].set_xlim(0, 4.4)
    figure.suptitle("Artifact stress at the aligned source FP=2 operating point")
    figure.tight_layout()
    output = root / "artifact_stress_comparison_fp2.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
