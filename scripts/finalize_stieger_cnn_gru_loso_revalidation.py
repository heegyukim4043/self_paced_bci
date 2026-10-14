"""Finalize and compare Stieger subject-LOSO CNN-GRU validation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
WITHIN = PROCESSED / "stieger2021_cnn_gru_early_usability.json"
LOSO = PROCESSED / "stieger2021_cnn_gru_loso_early_usability.json"
EXPOSURE = PROCESSED / "stieger2021_cnn_gru_loso_rest_exposure_curve.json"
FIGURE = PROCESSED / "stieger2021_cnn_gru_loso_revalidation.png"
REPORT = ROOT / "STIEGER_CNN_GRU_LOSO_REVALIDATION_20260921.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def med(payload, key):
    return payload["descriptive"][key]["median"]


def main() -> None:
    within, loso, exposure = load(WITHIN), load(LOSO), load(EXPOSURE)
    exposure["protocol"]["score"] = (
        "frozen three-seed CNN-GRU; subject-LOSO source pool; all target sessions excluded"
    )
    exposure["protocol"]["calibration_independence"] = (
        "target first-half intentional rest is independent of model fitting"
    )
    exposure["protocol"]["claim_limit"] = (
        "trial-wise partial validation; independent target calibration but not continuous natural idle"
    )
    EXPOSURE.write_text(json.dumps(exposure, indent=2), encoding="utf-8")

    transferred = np.asarray(
        [row["future_transferred_actual_fp2"] for row in loso["rows"]], float
    )
    admissible = int(np.sum((transferred >= 1.5) & (transferred <= 2.5)))
    exp = exposure["summary"]["exposures"]
    oracle = exposure["summary"]["oracle"]
    comparison = {
        "within_subject": {
            "window_auc": med(within, "future_window_auc"),
            "common_fp2_actual_fp": med(within, "common_fp2_actual_fp"),
            "excess_1s_fp2": med(within, "future_common_fp_excess_fp2"),
            "excess_2s_fp2": med(within, "future_common_fp_excess_2s_fp2"),
            "transferred_fp2": med(within, "future_transferred_actual_fp2"),
            "M2_rho": within["prediction"]["future_common_fp_excess_fp2"]["models"]["M2_combined"]["spearman_rho"],
        },
        "subject_loso": {
            "window_auc": med(loso, "future_window_auc"),
            "common_fp2_actual_fp": med(loso, "common_fp2_actual_fp"),
            "excess_1s_fp2": med(loso, "future_common_fp_excess_fp2"),
            "excess_2s_fp2": med(loso, "future_common_fp_excess_2s_fp2"),
            "transferred_fp2": med(loso, "future_transferred_actual_fp2"),
            "transferred_fp2_admissible": f"{admissible}/30",
            "transferred_fp2_range": [float(transferred.min()), float(transferred.max())],
            "M2_rho": loso["prediction"]["future_common_fp_excess_fp2"]["models"]["M2_combined"]["spearman_rho"],
            "M2_rho_ci95": loso["prediction"]["future_common_fp_excess_fp2"]["bootstrap"]["rho_ci95"]["M2_combined"],
        },
    }

    figure_payload = {
        "protocol": {
            "comparison": "Stieger within-subject chronological fit versus subject-LOSO source pool",
            "shared_decision": "cell-reset EWMA H30, MA .5 s, dwell .3 s, refractory 1 s",
            "claim_limit": "trial-cell feedback-onset validation; not continuous natural idle",
        },
        "comparison": comparison,
        "loso_exposure": exposure["summary"],
    }
    (PROCESSED / "stieger2021_cnn_gru_loso_revalidation.json").write_text(
        json.dumps(figure_payload, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    labels = ["within-subject", "subject-LOSO"]
    colors = ["#4c72b0", "#dd8452"]
    axes[0, 0].bar(labels, [comparison["within_subject"]["window_auc"], comparison["subject_loso"]["window_auc"]], color=colors)
    axes[0, 0].axhline(.5, color="black", ls="--", lw=1)
    axes[0, 0].set_ylim(.4, .58)
    axes[0, 0].set_title("Future feedback-window AUC")
    x = np.arange(2)
    width = .35
    axes[0, 1].bar(x - width / 2, [comparison["within_subject"]["excess_1s_fp2"], comparison["subject_loso"]["excess_1s_fp2"]], width, label="excess@1s", color="#55a868")
    axes[0, 1].bar(x + width / 2, [comparison["within_subject"]["excess_2s_fp2"], comparison["subject_loso"]["excess_2s_fp2"]], width, label="excess@2s", color="#c44e52")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].legend()
    axes[0, 1].set_title("Diagnostic common-FP2 event excess")
    exposures = np.asarray([30, 60, 90, 150])
    for target, color in (("2", "#4c72b0"), ("4", "#dd8452")):
        values = [exp[str(value)][target]["actual_fp_per_min"]["median"] for value in exposures]
        axes[1, 0].plot(exposures[:3], values[:3], "o-", color=color, label=f"target {target}/min")
        axes[1, 0].plot(exposures[2:], values[2:], "--", color=color, alpha=.65)
        axes[1, 0].scatter(exposures[-1], values[-1], facecolors="white", edgecolors=color, s=55, zorder=3)
        axes[1, 0].axhline(float(target), color=color, ls="--", alpha=.5)
    axes[1, 0].set_title("Independent target-rest threshold transfer")
    axes[1, 0].set_xlabel("Calibration rest exposure (s)")
    axes[1, 0].set_ylabel("Future FP/min (median)")
    axes[1, 0].text(150, min(values[-1] for values in ([exp[str(value)][target]["actual_fp_per_min"]["median"] for value in exposures] for target in ("2", "4"))) - .15, "n=25 risk set", ha="right", va="top", fontsize=9)
    axes[1, 0].legend()
    m2 = [comparison["within_subject"]["M2_rho"], comparison["subject_loso"]["M2_rho"]]
    axes[1, 1].bar(labels, m2, color=colors)
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set_title("Early-usability M2 grouped OOS rho")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Stieger2021 CNN-GRU: protocol-matched subject-LOSO revalidation", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    e90 = exp["90"]["2"]
    e150 = exp["150"]["2"]
    lines = [
        "# Stieger2021 CNN-GRU subject-LOSO revalidation",
        "",
        "## Protocol",
        "",
        "- One three-seed CNN-GRU ensemble per held-out subject; every target session is excluded from fitting.",
        "- Each of 19 source subjects contributes at most 64 first-half cells (48 active, 16 intentional rest); subjects with S2 are not overweighted.",
        "- Target first-half preparation is used for label-free input normalization. Target first-half intentional rest is used only for prospective threshold calibration.",
        "- Hidden state, EWMA, MA, dwell and refractory reset at every published cell seam.",
        "- Locked second-half first10/future split and circular within-run trial-label-shift null are unchanged.",
        "",
        "## Main comparison",
        "",
        "| Metric | Within-subject | Subject-LOSO |",
        "|---|---:|---:|",
        f"| Future window AUC | {comparison['within_subject']['window_auc']:.3f} | {comparison['subject_loso']['window_auc']:.3f} |",
        f"| Common FP2 achieved FP/min | {comparison['within_subject']['common_fp2_actual_fp']:.3f} | {comparison['subject_loso']['common_fp2_actual_fp']:.3f} |",
        f"| Common FP2 excess@1 s | {comparison['within_subject']['excess_1s_fp2']:.3f} | {comparison['subject_loso']['excess_1s_fp2']:.3f} |",
        f"| Common FP2 excess@2 s | {comparison['within_subject']['excess_2s_fp2']:.3f} | {comparison['subject_loso']['excess_2s_fp2']:.3f} |",
        f"| Transferred FP2 achieved FP/min | {comparison['within_subject']['transferred_fp2']:.3f} | {comparison['subject_loso']['transferred_fp2']:.3f} |",
        f"| Early-usability M2 OOS rho | {comparison['within_subject']['M2_rho']:.3f} | {comparison['subject_loso']['M2_rho']:.3f} |",
        "",
        f"LOSO common-FP2 sensitivity/null were {med(loso, 'common_fp2_onset_sensitivity_1s'):.3f}/{med(loso, 'common_fp2_circular_run_shift_null_median_1s'):.3f} at 1 s. Window AUC was {comparison['subject_loso']['window_auc']:.3f}; therefore the near-zero event excess is a representation/generalization floor, not merely a transferred-threshold error.",
        "",
        "## Prospective threshold transfer",
        "",
        f"- The nominal FP2 threshold yielded median {comparison['subject_loso']['transferred_fp2']:.3f}/min, IQR {loso['descriptive']['transferred_fp2_actual_fp']['iqr']}, range {comparison['subject_loso']['transferred_fp2_range']}; {admissible}/30 sessions were in 1.5--2.5/min.",
        f"- FP2 future medians at 30/60/90 s were {exp['30']['2']['actual_fp_per_min']['median']:.3f}/{exp['60']['2']['actual_fp_per_min']['median']:.3f}/{e90['actual_fp_per_min']['median']:.3f}/min. At 90 s, {e90['admissible_n']}/30 were admissible.",
        f"- The paired 30->90 s log-error delta was {e90['paired_delta_vs_30s_operating_point_log_error']['median']:+.3f}, subject-cluster CI {e90['paired_delta_vs_30s_operating_point_log_error']['subject_cluster_ci95']}; the upper bound touches zero.",
        f"- At 150 s, the changing risk set (n={e150['n_subject_sessions']}) reached {e150['actual_fp_per_min']['median']:.3f}/min with {e150['admissible_n']}/{e150['n_subject_sessions']} admissible; this is suggestive, not complete-cohort evidence.",
        f"- The future-rest diagnostic oracle achieved {oracle['2']['actual_fp_per_min']['median']:.3f}/min, but excess@2 s was still only {oracle['2']['excess_2s']['median']:.3f}.",
        "",
        "## Decision",
        "",
        "The protocol-matched Stieger LOSO rerun is negative. Removing within-subject fitting reduces window discrimination to chance and leaves onset excess near zero. Longer independent rest improves FP calibration for a subset but cannot recover event information. The Lee early-usability relationship does not transport under this stricter Stieger condition; its negative M2 rho is interpreted at a near-degenerate performance floor, not as a useful inverse predictor.",
        "",
        "This remains a trial-cell feedback-onset validation. It does not validate continuous natural idle or truly self-initiated onset detection.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
