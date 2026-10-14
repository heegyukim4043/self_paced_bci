"""Cohort summary for the random evaluation idle/MI-ratio robustness study."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/lee2019_cnn_gru_eval_idle_mi_ratio_subset20.json"
OUTPUT = ROOT / "data/processed/lee2019_cnn_gru_eval_idle_mi_ratio_summary.png"
RATIOS = (0.75, 1.5, 3.0)
OPS = ("transferred", "common_fp")
COLORS = {"transferred": "#d95f02", "common_fp": "#1b9e77"}
LABELS = {"transferred": "Transferred threshold", "common_fp": "Common FP diagnostic"}

def values(rows, session, ratio, op, field):
    out = []
    for row in rows:
        if row["session"] != session:
            continue
        for condition in row["conditions"]:
            if condition["idle_mi_ratio"] != ratio:
                continue
            metric = condition["metrics"][op]
            if field == "false_alarms":
                out.append(metric["fp"] * condition["idle_seconds"] / 60.0)
            else:
                out.append(metric[field])
    return np.asarray(out, float)

def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8")); rows = payload["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharex=True)
    specs = [
        ("fp", "Achieved false alarms/min", "FP/min"),
        ("raw_tpr_1s", "Onset sensitivity within 1 s", "Sensitivity"),
        ("excess_1s", "Chance-corrected onset sensitivity", "Excess sensitivity"),
        ("false_alarms", "Absolute false alarms per 50 MI events", "False-alarm count"),
    ]
    x = np.arange(len(RATIOS))
    for ax, (field, title, ylabel) in zip(axes.flat, specs):
        for op in OPS:
            medians = []
            for ratio in RATIOS:
                pooled = np.concatenate([values(rows, s, ratio, op, field) for s in (1, 2)])
                medians.append(np.median(pooled))
            ax.plot(x, medians, marker="o", lw=2.2, ms=7, color=COLORS[op], label=LABELS[op])
            for session, ls in ((1, "--"), (2, ":")):
                session_medians = [np.median(values(rows, session, ratio, op, field)) for ratio in RATIOS]
                ax.plot(x, session_medians, ls=ls, lw=1.0, alpha=.52, color=COLORS[op])
        if field == "fp": ax.axhline(2, color="black", lw=1, ls="--", alpha=.65, label="Target FP=2/min")
        ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel); ax.grid(alpha=.22)
        if field in ("raw_tpr_1s", "excess_1s"): ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xticks(x, ["0.75:1\n(42.9% idle)", "1.5:1\n(60.0% idle)", "3:1\n(75.0% idle)"])
        ax.set_xlabel("Random evaluation idle/MI duration ratio")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .915))
    fig.suptitle("Frozen CNN-GRU robustness to random idle/MI spacing\n20 subjects × 2 sessions × 5 random compositions", fontsize=15, fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .84)); fig.savefig(OUTPUT, dpi=190, bbox_inches="tight")
    print(OUTPUT)

if __name__ == "__main__": main()
