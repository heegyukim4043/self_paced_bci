import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


path = Path("data/processed/lee2019_screening_fp_latency.json")
data = json.loads(path.read_text(encoding="utf-8"))
output = path.with_name("lee2019_screening_interval_metrics.png")
fps = [0.5, 1.0, 2.0, 4.0]
groups = [("all", "All"), ("selected", "Selected"), ("nonselected", "Non-selected")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
specs = [
    ("coverage", "Mean MI coverage"),
    ("temporal_iou", "Temporal IoU"),
    ("f1_dtc_gtc_0.7", "Strict interval F1 (0.7/0.7)"),
]
for axis, (metric, ylabel) in zip(axes, specs):
    for group, label in groups:
        values = [data["summary"][group]["surface"][str(fp)][metric] for fp in fps]
        axis.plot(fps, values, marker="o", linewidth=2, label=label)
    axis.set(xlabel="Actual FP/min target", ylabel=ylabel, xticks=fps)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
axes[0].legend()
fig.suptitle("Interval tracking after session screening", fontweight="bold")
fig.savefig(output, dpi=180)
print(output)
