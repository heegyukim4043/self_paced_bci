import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


path = Path("data/processed/lee2019_score_null_audit.json")
data = json.loads(path.read_text(encoding="utf-8"))
selected = [row for row in data["rows"] if row["selected"]]
nonselected = [row for row in data["rows"] if not row["selected"]]

labels = ["Observed", "Time reversed", "Circular shift", "Train-label permuted"]
keys = ["observed_sensitivity", "time_reversed_sensitivity", "circular_shift_mean", "training_label_permutation_mean"]
values = [np.array([row[key] for row in selected], float) for key in keys]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for index in range(len(selected)):
    axes[0].plot(range(4), [value[index] for value in values], color="0.75", alpha=.45, linewidth=.8)
axes[0].boxplot(values, tick_labels=labels, showfliers=False)
axes[0].set_ylabel("Sensitivity at actual FP=4/min, 2-s tolerance")
axes[0].tick_params(axis="x", rotation=18)
axes[0].grid(axis="y", alpha=.25)
axes[0].set_title("Selected sessions: leakage/null controls")

stats = ["idle_z_mean", "idle_z_std", "idle_z_skew", "idle_z_excess_kurtosis"]
names = ["Mean", "SD", "Skew", "Excess kurtosis"]
x = np.arange(len(stats)); width = .34
axes[1].bar(x-width/2, [np.median([r[k] for r in selected]) for k in stats], width, label="Selected")
axes[1].bar(x+width/2, [np.median([r[k] for r in nonselected]) for k in stats], width, label="Non-selected")
axes[1].axhline(0, color="black", linewidth=.7)
axes[1].set_xticks(x, names, rotation=18)
axes[1].set_title("EWMA idle-z diagnostics")
axes[1].legend()
axes[1].grid(axis="y", alpha=.25)
fig.savefig(path.with_suffix(".png"), dpi=180)
print(path.with_suffix(".png"))
