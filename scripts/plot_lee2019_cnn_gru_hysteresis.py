"""Plot the cached CNN-GRU hysteresis sweep."""
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/"data/processed/lee2019_cnn_gru_hysteresis_subset20.json"
output = ROOT/"data/processed/lee2019_cnn_gru_hysteresis_subset20.png"
d = json.loads(source.read_text(encoding="utf-8")); drops = [0, .25, .5, 1, 1.5]
fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
specs = [("coverage_median", "MI interval coverage"), ("temporal_iou_median", "Temporal IoU"),
         ("idle_false_active_occupancy_median", "Idle false-active occupancy"), ("strict_f1_median", "Strict interval F1 (0.7/0.7)")]
for axis, (metric, title) in zip(ax.flat, specs):
    for session, color in (("1", "#2878b5"), ("2", "#d95f02")):
        values = [d["summary"][session]["arms"][str(float(x))][metric] for x in drops]
        axis.plot(drops, values, marker="o", lw=2.1, color=color, label=f"Session {session}")
    axis.set_title(title, fontweight="bold"); axis.grid(alpha=.22); axis.set_ylabel("Proportion")
for axis in ax[-1]: axis.set_xlabel("Release drop below trigger threshold (z)")
ax[0, 0].legend(frameon=False)
fig.suptitle("CNN-GRU + EWMA release-threshold hysteresis\nEntry alarms fixed: onset SEN and FP count do not change", fontweight="bold", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(output, dpi=190, bbox_inches="tight"); print(output)
