import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


path = Path("data/processed/lee2019_screening_fp_latency.json")
data = json.loads(path.read_text(encoding="utf-8"))
output = path.with_suffix(".png")
fps = ["0.5", "1.0", "2.0", "4.0"]
latencies = ["0.5", "1.0", "2.0", "3.0", "4.0"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
groups = ["all", "selected", "nonselected"]
titles = ["All sessions", "Selected sessions", "Non-selected sessions"]
for axis, group, title in zip(axes[:3], groups, titles):
    values = np.array(
        [[data["summary"][group]["surface"][fp][lat] for lat in latencies] for fp in fps]
    )
    image = axis.imshow(values, vmin=0, vmax=0.8, cmap="viridis", aspect="auto")
    for row in range(len(fps)):
        for col in range(len(latencies)):
            color = "white" if values[row, col] < 0.6 else "black"
            axis.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color=color, fontsize=9)
    axis.set_xticks(range(len(latencies)), latencies)
    axis.set_yticks(range(len(fps)), fps)
    axis.set(
        xlabel="Allowed latency (s)",
        ylabel="Actual FP/min target",
        title=f'{title} (n={data["summary"][group]["n"]})',
    )

x = np.arange(3)
width = 0.22
for attempts in (1, 2, 3):
    values = [data["summary"][group][f"empirical_success_within_{attempts}_attempts"] for group in groups]
    axes[3].bar(x + (attempts - 2) * width, values, width, label=f"{attempts} attempt")
axes[3].set_xticks(x, ["All", "Selected", "Non-selected"])
axes[3].set_ylim(0, 1)
axes[3].set_ylabel("Empirical success within N attempts")
axes[3].set_title("FP near 2/min, 4-s tolerance")
axes[3].legend()
axes[3].grid(axis="y", alpha=0.2)
fig.colorbar(image, ax=axes[:3], label="Median onset sensitivity", shrink=0.8)
fig.suptitle("Locked screening x FP budget x onset latency", fontweight="bold")
fig.savefig(output, dpi=180)
print(output)
