"""Plot matched LOSO window AUC against fixed-threshold event excess."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "data/processed/lee2019_auc_event_relation.json").read_text(encoding="utf-8"))
rows = payload["rows"]
x = np.asarray([row["window_auc"] for row in rows])
y = np.asarray([row["excess_sensitivity_2s"] for row in rows])
s = np.asarray([row["session"] for row in rows])

fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
colors = {1: "#377eb8", 2: "#e41a1c"}
for session in (1, 2):
    mask = s == session
    ax.scatter(x[mask], y[mask], s=34, alpha=.72, color=colors[session],
               edgecolor="white", linewidth=.35, label=f"Session {session}")
coef = np.polyfit(x, y, 1)
grid = np.linspace(x.min(), x.max(), 100)
ax.plot(grid, np.polyval(coef, grid), color="#333333", linewidth=1.6,
        label="Linear visual guide")
ax.axvline(.8, color="#777777", linestyle="--", linewidth=1, label="AUC = 0.80")
ax.axhline(0, color="black", linewidth=.8)
summary = payload["summary"]
ax.text(.02, .98,
        f"Spearman rho = {summary['spearman_rho']:.3f}\n"
        f"subject-cluster 95% CI [{summary['subject_cluster_bootstrap_ci'][0]:.3f}, "
        f"{summary['subject_cluster_bootstrap_ci'][1]:.3f}]",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=.9, edgecolor="#bbbbbb"))
ax.set(
    xlabel="Strict 1-s window AUC (LOSO Arm A)",
    ylabel="2-s onset excess sensitivity\n(fixed nominal FP=4; circular-shift corrected)",
    title="Window discrimination predicts, but does not determine, event performance",
)
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=.18)
output = ROOT / "data/processed/lee2019_auc_event_relation.png"
fig.savefig(output, dpi=220)
print(output)
