import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


loso = json.loads(Path("data/processed/lee2019_loso_deployment.json").read_text())
within = json.loads(Path("data/processed/lee2019_screening_fp_latency.json").read_text())
labels = ["C: supervised", "A: LOSO + target rest", "B: source only"]
interpolated = [
    within["summary"]["selected"]["surface"]["4.0"]["2.0"],
    loso["summary"]["combined"]["A_target_rest"]["surface"]["4.0"]["2.0"],
    loso["summary"]["combined"]["B_source_only"]["surface"]["4.0"]["2.0"],
]
deployable = [
    within["summary"]["selected"]["surface"]["4.0"]["deployable"]["2.0"],
    loso["summary"]["combined"]["A_target_rest"]["surface"]["4.0"]["deployable"]["2.0"],
    loso["summary"]["combined"]["B_source_only"]["surface"]["4.0"]["deployable"]["2.0"],
]
fixed = [
    within["summary"]["selected"]["surface"]["4.0"]["fixed_threshold"]["2.0"],
    loso["summary"]["combined"]["A_target_rest"]["surface"]["4.0"]["fixed_threshold"]["2.0"],
    loso["summary"]["combined"]["B_source_only"]["surface"]["4.0"]["fixed_threshold"]["2.0"],
]
coverage = [
    within["summary"]["selected"]["surface"]["4.0"]["fixed_threshold"]["coverage"],
    loso["summary"]["combined"]["A_target_rest"]["surface"]["4.0"]["fixed_threshold"]["coverage"],
    loso["summary"]["combined"]["B_source_only"]["surface"]["4.0"]["fixed_threshold"]["coverage"],
]
x = np.arange(3); width = .25
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].bar(x-width, interpolated, width, label="Linear common-FP interpolation")
axes[0].bar(x, deployable, width, label="Oracle single threshold")
axes[0].bar(x+width, fixed, width, label="Source/anchor-fixed threshold")
axes[0].set_xticks(x, labels, rotation=15)
axes[0].set_ylabel("2-s sensitivity at FP budget 4/min")
axes[0].set_ylim(0, .7); axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
axes[1].bar(x, coverage, color=["#4c78a8", "#f58518", "#54a24b"])
axes[1].set_xticks(x, labels, rotation=15)
axes[1].set_ylabel("Fixed-threshold interval coverage")
axes[1].set_ylim(0, .4); axes[1].grid(axis="y", alpha=.25)
fig.suptitle("Selected-session LOSO deployment comparison", fontweight="bold")
out = Path("data/processed/lee2019_loso_deployment.png")
fig.savefig(out, dpi=180)
print(out)
