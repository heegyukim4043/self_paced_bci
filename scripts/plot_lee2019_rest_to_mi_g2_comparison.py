"""Forest comparison of Lee2019 G2 Method A and AR(2) Method B."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "A: causal BP": ROOT / "data/processed/lee2019_rest_to_mi_g2.json",
    "B: causal AR(2)": ROOT / "data/processed/lee2019_rest_to_mi_g2_state_space.json",
}
OUTPUT = ROOT / "data/processed/lee2019_rest_to_mi_g2_method_comparison.png"
PANELS = (
    ("unconstrained_contra_minus_medial_broad_tau_s", "Contra - broad medial breakpoint"),
    ("unconstrained_contra_minus_medial_strict_tau_s", "Contra - strict medial breakpoint"),
    ("unconstrained_laterality_minus_medial_broad_tau_s", "Laterality - broad medial breakpoint"),
    ("contra_minus_medial_broad_t50_s", "Contra - broad medial t50"),
)


def main() -> None:
    payloads = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in INPUTS.items()
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    rows = [("A: causal BP", "1"), ("A: causal BP", "2"),
            ("B: causal AR(2)", "1"), ("B: causal AR(2)", "2")]
    y = np.arange(len(rows))[::-1]
    labels = [f"{method}, S{session}" for method, session in rows]
    colors = ["#377eb8", "#7fb3d5", "#e6550d", "#fdae6b"]
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        for yi, (method, session), color in zip(y, rows, colors):
            result = payloads[method]["summary"][session]["contrasts"][key]
            center = result["median_s"]
            low, high = result["bootstrap_95ci"]
            ax.errorbar(
                center, yi,
                xerr=[[center - low], [high - center]],
                fmt="o", color=color, capsize=4, lw=1.8,
            )
        ax.axvline(0, color="0.25", lw=1)
        ax.axvspan(0, ax.get_xlim()[1], color="#d9f0d3", alpha=0.18)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel("Difference (s); positive = medial earlier")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Lee2019 G2 estimator-family comparison\n"
        "Median with subject-bootstrap 95% CI; 54 sessions per row",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
