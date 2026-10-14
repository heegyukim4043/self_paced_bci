"""Create the final project-level performance figures from locked result JSONs.

The main figure deliberately separates four questions:
  A. Did a more complex detector improve event-level performance?
  B. Did EWMA correct transferred operating points?
  C. Did prospective, within-session screening identify usable sessions?
  D. How long did screened sessions remain free of false alarms?

Run from the repository root:
    py scripts/plot_overall_performance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
OUT = IN / "figures"


def load(name: str):
    with (IN / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def err(value, ci):
    return np.array([[value - ci[0]], [ci[1] - value]])


def km_step(ax, block, label, color, linestyle="-"):
    curve = block["curve"]
    ax.step(
        curve["time"], curve["survival"], where="post", label=label,
        color=color, linestyle=linestyle, linewidth=2.0,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    three = load("three_model_validation_summary.json")
    mil = load("mil_locked.json")
    hsmm = load("hsmm_gate.json")
    lee_adapt = load("lee2019_sequential_adaptation.json")
    bcic = load("bciciv1_ewma_adaptation.json")
    stieger = load("stieger2021_expansion20_ewma.json")
    screen_s1 = load("lee2019_prospective_screening_n10_session1_relative.json")
    screen_s2 = load("lee2019_prospective_screening_n10_session2_relative.json")
    deployment = load("lee2019_screening_deployment.json")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), constrained_layout=True)

    # A — locked model ladder. Values are not presented as a common leaderboard:
    # HSMM changes only the decision layer and had paired delta exactly zero.
    ax = axes[0, 0]
    labels = ["Tier-0\nband-power", "Riemannian\nTS+LR", "EEGNet-SL",
              "SWPC-style\nSSL", "MIL", "HSMM", "MRCP"]
    s1 = [0.125, three["summary"][0]["riemannian_excess_median"],
          three["summary"][0]["eegnet_sl_excess_median"],
          three["summary"][0]["eegnet_ssl_excess_median"],
          mil["summary"][0]["median_mil_excess"],
          hsmm["locked"][0]["summary"]["hsmm_excess_tpr_median"], np.nan]
    s2 = [np.nan, three["summary"][1]["riemannian_excess_median"],
          three["summary"][1]["eegnet_sl_excess_median"],
          three["summary"][1]["eegnet_ssl_excess_median"],
          mil["summary"][1]["median_mil_excess"],
          hsmm["locked"][1]["summary"]["hsmm_excess_tpr_median"], np.nan]
    x = np.arange(len(labels))
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.scatter(x - 0.08, s1, color="#2878b5", label="Session 1", zorder=3)
    ax.scatter(x + 0.08, s2, color="#d95f02", marker="s", label="Session 2", zorder=3)
    for i in range(1, 6):
        if np.isfinite(s1[i]) and np.isfinite(s2[i]):
            ax.plot([x[i] - .08, x[i] + .08], [s1[i], s2[i]], color="0.78", zorder=1)
    ax.text(5, max(s1[5], s2[5]) + .035, "paired Δ = 0", ha="center", fontsize=8)
    ax.text(6, 0.12, "not identifiable\nfrom cue-timed data", ha="center",
            color="0.4", fontsize=8, bbox=dict(facecolor="0.93", edgecolor="0.7", hatch="///"))
    ax.set_xticks(x, labels)
    ax.set_ylabel("Excess event TPR at FP≈2/min")
    ax.set_ylim(-0.09, 0.39)
    ax.set_title("A  Model ladder: complexity did not yield a consistent gain", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.01, 0.01, "Locked diagnostic protocols; not a common-protocol leaderboard",
            transform=ax.transAxes, color="0.35", fontsize=7.5)

    # B — operating-point transfer correction.
    ax = axes[0, 1]
    lee_static = lee_adapt["summary"]["static"]["2.0"]["ratio_median"]
    lee_ewma = lee_adapt["summary"]["all_hl30"]["2.0"]["ratio_median"]
    bcic_static = bcic["summary"]["static"]["fp_target_ratio_median"]
    bcic_ewma = bcic["summary"]["ewma_h30"]["fp_target_ratio_median"]
    st_static = stieger["summary"]["raw"]["overall"]["static"]["target_ratio_median"]
    st_ewma = stieger["summary"]["raw"]["overall"]["ewma_h30"]["target_ratio_median"]
    datasets = ["Lee2019\n(n=54)", "BCI IV-1\n(n=4)", "Stieger S1\n(n=20)"]
    static = np.array([lee_static, bcic_static, st_static])
    adaptive = np.array([lee_ewma, bcic_ewma, st_ewma])
    for i in range(3):
        ax.plot([i - .13, i + .13], [static[i], adaptive[i]], color="0.65", linewidth=1.6)
    ax.scatter(np.arange(3) - .13, static, color="#b2182b", label="Static anchor", zorder=3)
    ax.scatter(np.arange(3) + .13, adaptive, color="#1b9e77", marker="s", label="EWMA H=30", zorder=3)
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="Target")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3), datasets)
    ax.set_ylabel("Actual / target FP rate (log scale)")
    ax.set_title("B  EWMA corrected operating-point collapse across datasets", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    for i, (a, b) in enumerate(zip(static, adaptive)):
        ax.text(i - .13, a * 1.08, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + .13, b / 1.18, f"{b:.2f}", ha="center", va="top", fontsize=8)

    # C — prospective 10-trial, same-session screening; use fixed dwell=0.2.
    ax = axes[1, 0]
    sessions = [screen_s1, screen_s2]
    selected = [d["summary"]["selected_dwell0.2"] for d in sessions]
    rejected = [d["summary"]["not_selected_dwell0.2"] for d in sessions]
    xpos = np.array([0, 1])
    width = .32
    sv = np.array([d["excess_median"] for d in selected])
    rv = np.array([d["excess_median"] for d in rejected])
    se = np.array([[v - d["excess_ci95"][0] for v, d in zip(sv, selected)],
                   [d["excess_ci95"][1] - v for v, d in zip(sv, selected)]])
    re = np.array([[v - d["excess_ci95"][0] for v, d in zip(rv, rejected)],
                   [d["excess_ci95"][1] - v for v, d in zip(rv, rejected)]])
    ax.bar(xpos - width/2, sv, width, color="#1b9e77", label="Selected session")
    ax.bar(xpos + width/2, rv, width, color="#bdbdbd", label="Not selected")
    ax.errorbar(xpos - width/2, sv, yerr=se, fmt="none", ecolor="black", capsize=3)
    ax.errorbar(xpos + width/2, rv, yerr=re, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(xpos, ["Session 1", "Session 2"])
    ax.set_ylabel("Prospective excess event TPR")
    ax.set_ylim(0, .61)
    ax.set_title("C  Ten-trial same-session screening recovered usable states", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    for i, d in enumerate(sessions):
        ax.text(i - width/2, sv[i] + se[1, i] + .025,
                f"coverage {100*d['selected_fraction']:.0f}%\nFP {selected[i]['actual_fp_median']:.2f}/min",
                ha="center", fontsize=7.5)

    # D — empirical first-FP survival. This existing analysis used 20 screening trials.
    ax = axes[1, 1]
    colors = {"S1": "#2878b5", "S2": "#d95f02"}
    for key, tag in [("session1", "S1"), ("session2", "S2")]:
        block = deployment[key]["km_wallclock"]
        km_step(ax, block["selected"], f"{tag} selected", colors[tag], "-")
        km_step(ax, block["not_selected"], f"{tag} not selected", colors[tag], "--")
    ax.set_xlabel("Wall-clock time to first false alarm (s)")
    ax.set_ylabel("Probability of remaining FP-free")
    ax.set_ylim(0, 1.03)
    ax.set_title("D  Screening prolonged false-alarm-free operation", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.text(.01, .02, "KM panel uses the 20-trial validation; medians: selected 142/81 s, non-selected 49/41 s",
            transform=ax.transAxes, fontsize=7.5, color="0.35")

    fig.suptitle("Pseudo-uncued MI gate: model diagnosis, adaptation, and deployable screening",
                 fontsize=14, fontweight="bold")
    png = OUT / "overall_performance_main.png"
    pdf = OUT / "overall_performance_main.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # Supplement E/F: evaluation audit trajectory and flat-response heterogeneity.
    screening_rows = load("lee2019_prospective_screening_n10_session1.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    ax = axes[0]
    stages = ["10-event\nexploration", "21-subject\nrepeats", "54-subject",
              "strict TP", "guard bug\nfixed", "end=4.0 s"]
    vals = [0.200, 0.089, 0.065, 0.041, 0.117, 0.127]
    ax.plot(range(len(vals)), vals, marker="o", linewidth=2, color="#6a3d9a")
    ax.fill_between([2.7, 3.3], 0, .22, color="#fddbc7", alpha=.45)
    ax.set_xticks(range(len(vals)), stages)
    ax.set_ylabel("Reported excess event TPR")
    ax.set_title("E  Evaluation-audit trajectory", loc="left", fontweight="bold")
    ax.set_ylim(0, .22)
    for i, v in enumerate(vals):
        ax.text(i, v + .009, f"{v:.3f}", ha="center", fontsize=8)

    ax = axes[1]
    flats = np.sort([r["screen_flat_fraction"] for r in screening_rows])
    ax.bar(np.arange(len(flats)), flats, color=np.where(flats <= .25, "#1b9e77", "#bdbdbd"), width=.85)
    ax.axhline(.25, color="#1b9e77", linestyle="--", linewidth=1.2, label="Descriptive 25% cutoff")
    ax.set_xlabel("Subject (sorted)")
    ax.set_ylabel("Flat-trial fraction")
    ax.set_ylim(0, 1)
    ax.set_title("F  Flat responses clustered strongly by subject", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(OUT / "overall_performance_supplement.png", bbox_inches="tight")
    fig.savefig(OUT / "overall_performance_supplement.pdf", bbox_inches="tight")
    plt.close(fig)

    snapshot = {
        "main_figure": str(png.relative_to(ROOT)),
        "panel_b": {
            "Lee2019": {"static": lee_static, "ewma_h30": lee_ewma},
            "BCI_IV_1": {"static": bcic_static, "ewma_h30": bcic_ewma},
            "Stieger_S1_n20": {"static": st_static, "ewma_h30": st_ewma},
        },
        "panel_c": {
            "session1": screen_s1["summary"],
            "session2": screen_s2["summary"],
        },
        "notes": [
            "Panel A combines locked diagnostic protocols and is not a common-protocol leaderboard.",
            "Panel C uses prospective 10-trial screening and dwell=0.2 s.",
            "Panel D uses the existing 20-trial KM validation because a 10-trial KM curve was not recomputed.",
            "All results are cue-paced pseudo-uncued simulation, not demonstrated self-initiated performance.",
        ],
    }
    with (OUT / "overall_performance_source_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(png)


if __name__ == "__main__":
    main()
