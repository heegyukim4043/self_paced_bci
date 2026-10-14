"""Build an auditable Lee2019--Stieger2021 CNN-GRU validation summary.

This is deliberately not a leaderboard.  It extracts frozen results while
keeping training regime, stream semantics, threshold mode, and null definition
visible so numerically similar columns are not mistaken for identical tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("M0_detection_fraction", "M1_morphology", "M2_combined")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def median(rows: list[dict[str, Any]], getter) -> float:
    return float(np.median([getter(row) for row in rows]))


def lee_loso_event_rows(payload: dict[str, Any], target: str = "2.0") -> list[dict[str, Any]]:
    output = []
    target_rows = [row for row in payload["rows"] if float(row["target_fp_per_min"]) == float(target)]
    for session in (1, 2):
        rows = [row for row in target_rows if int(row["session"]) == session]
        output.append(
            {
                "dataset": "Lee2019",
                "session": session,
                "n": len(rows),
                "training": "source-pooled LOSO",
                "stream": "continuous pseudo-uncued; seam-guarded",
                "state": "continuous within replay stream",
                "threshold": "diagnostic matched achieved FP",
                "null": "500 circular time shifts",
                "actual_fp_per_min": median(rows, lambda r: r["cnn_gru"]["actual_fp_per_min"]),
                "sensitivity_1s": median(rows, lambda r: r["cnn_gru"]["onset_sensitivity"]["1.0"]),
                "null_1s": median(rows, lambda r: r["cnn_gru"]["circular_shift_null_median"]["1.0"]),
                "excess_1s": median(rows, lambda r: r["cnn_gru"]["circular_shift_excess"]["1.0"]),
                "sensitivity_2s": median(rows, lambda r: r["cnn_gru"]["onset_sensitivity"]["2.0"]),
                "null_2s": median(rows, lambda r: r["cnn_gru"]["circular_shift_null_median"]["2.0"]),
                "excess_2s": median(rows, lambda r: r["cnn_gru"]["circular_shift_excess"]["2.0"]),
                "coverage": median(rows, lambda r: r["cnn_gru"]["coverage"]),
                "claim_limit": "cue-relative pseudo-onset; diagnostic threshold",
            }
        )
    return output


def stieger_event_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for session in (1, 2):
        rows = [row for row in payload["rows"] if int(row["session"]) == session]
        output.append(
            {
                "dataset": "Stieger2021",
                "session": session,
                "n": len(rows),
                "training": "within-subject chronological first half",
                "stream": "independent preparation->feedback trial cells",
                "state": "reset at every cell seam",
                "threshold": "diagnostic future-rest common FP2",
                "null": "500 within-run circular trial-label shifts",
                "actual_fp_per_min": median(rows, lambda r: r["common_fp_diagnostic"]["2"]["actual_fp_per_min"]),
                "sensitivity_1s": median(rows, lambda r: r["common_fp_diagnostic"]["2"]["onset_sensitivity_1s"]),
                "null_1s": median(rows, lambda r: r["common_fp_diagnostic"]["2"]["circular_run_shift_null_median_1s"]),
                "excess_1s": median(rows, lambda r: r["common_fp_diagnostic"]["2"]["circular_run_shift_excess_1s"]),
                "sensitivity_2s": median(rows, lambda r: r["common_fp_diagnostic"]["2"]["onset_sensitivity_2s"]),
                "null_2s": median(rows, lambda r: r["common_fp_diagnostic"]["2"]["circular_run_shift_null_median_2s"]),
                "excess_2s": median(rows, lambda r: r["common_fp_diagnostic"]["2"]["circular_run_shift_excess_2s"]),
                "coverage": None,
                "window_auc": median(rows, lambda r: r["future_window_auc"]),
                "delay_seconds": median(
                    [row for row in rows if row["common_fp_diagnostic"]["2"]["median_detection_latency_s"] is not None],
                    lambda r: r["common_fp_diagnostic"]["2"]["median_detection_latency_s"],
                ),
                "claim_limit": "trial-wise feedback-onset partial validation; not continuous self-paced deployment",
            }
        )
    return output


def lee_within_subject_reference(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for block in payload["summary"]:
        session = int(block["session"])
        for mode in ("common_fp", "transferred"):
            item = block["modes"][mode]
            output.append(
                {
                    "dataset": "Lee2019",
                    "session": session,
                    "n": int(block["subjects"]),
                    "training": "within-subject first 50 trials",
                    "decision": "historical instantaneous crossing + refractory",
                    "threshold_mode": mode,
                    "actual_fp_per_min": float(item["fp_median"]),
                    "sensitivity_1s": float(item["raw_tpr_1s_median"]),
                    "excess_1s": float(item["excess_1s_median"]),
                    "delay_seconds": float(item["delay_median"]),
                    "comparable_to_harmonized_event_table": False,
                }
            )
    return output


def usability_rows(lee: dict[str, Any], stieger: dict[str, Any]) -> list[dict[str, Any]]:
    sources = (
        ("Lee2019", lee["summary"]["future_excess_fp2"], "source-pooled LOSO", "PASS"),
        (
            "Stieger2021",
            stieger["prediction"]["future_common_fp_excess_fp2"],
            "within-subject first-half",
            "FAIL",
        ),
    )
    output = []
    for dataset, block, training, gate in sources:
        for model in MODELS:
            item = block["models"][model]
            output.append(
                {
                    "dataset": dataset,
                    "training": training,
                    "target_median": float(block["target_median"]),
                    "model": model,
                    "oos_spearman_rho": float(item["spearman_rho"]),
                    "rho_ci95": [float(v) for v in block["bootstrap"]["rho_ci95"][model]],
                    "mae": float(item["mae"]),
                    "top20_auroc": float(item["usable_auroc"]),
                    "transport_gate": gate,
                }
            )
    return output


def exposure_rows(lee: dict[str, Any], stieger: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    lee_combined = lee["summary"]["combined"]["arms"]
    for arm in ("prefix00", "prefix30"):
        block = lee_combined[arm]
        for target in ("2.0", "4.0"):
            item = block[target]
            output.append(
                {
                    "dataset": "Lee2019",
                    "exposure_seconds": float(block["known_idle_exposure_seconds_median"]),
                    "n": 40,
                    "risk_set": "fixed subset20 x S1/S2",
                    "calibration_context": "session-start anchor + earlier stitched idle",
                    "target_fp_per_min": float(target),
                    "future_fp_per_min": float(item["actual_fp_per_min_median"]),
                    "admissible_n": int(item["admissible_n"]),
                    "log_error": float(item["operating_point_log_error_median"]),
                    "excess_2s": float(item["excess_2s_median"]),
                }
            )
    stieger_summary = stieger["summary"]["exposures"]
    for exposure in (30, 90, 150):
        block = stieger_summary[str(exposure)]
        for target in ("2", "4"):
            item = block[target]
            output.append(
                {
                    "dataset": "Stieger2021",
                    "exposure_seconds": float(exposure),
                    "n": int(item["n_subject_sessions"]),
                    "risk_set": "complete cohort" if exposure <= 90 else "supported risk set",
                    "calibration_context": "first-half intentional-rest feedback; in-sample with model fit",
                    "target_fp_per_min": float(target),
                    "future_fp_per_min": float(item["actual_fp_per_min"]["median"]),
                    "admissible_n": int(item["admissible_n"]),
                    "log_error": float(item["operating_point_log_error"]["median"]),
                    "excess_2s": float(item["excess_2s"]["median"]),
                }
            )
    return output


def protocol_rows() -> list[dict[str, Any]]:
    return [
        {
            "dataset": "Lee2019",
            "subjects_sessions": "54 / 108",
            "event_semantics": "cue-relative pseudo-onset",
            "evaluation_stream": "stitched continuous rest+MI replay with seam guard",
            "state": "continuous within replay",
            "early_usability_training": "source-pooled LOSO",
            "exposure_training": "within-subject CNN; subset20",
            "null": "continuous circular time shift",
        },
        {
            "dataset": "Stieger2021",
            "subjects_sessions": "20 / 30",
            "event_semantics": "feedback onset; active targets vs intentional rest",
            "evaluation_stream": "published trial cells; no natural continuous idle",
            "state": "reset at every cell",
            "early_usability_training": "within-subject chronological first half",
            "exposure_training": "same within-subject CNN; rest calibration in-sample",
            "null": "within-run circular trial-label shift",
        },
    ]


def build_payload(
    lee_matched: dict[str, Any],
    lee_reference: dict[str, Any],
    lee_usability: dict[str, Any],
    lee_exposure: dict[str, Any],
    stieger_usability: dict[str, Any],
    stieger_exposure: dict[str, Any],
) -> dict[str, Any]:
    event = lee_loso_event_rows(lee_matched) + stieger_event_rows(stieger_usability)
    usability = usability_rows(lee_usability, stieger_usability)
    exposure = exposure_rows(lee_exposure, stieger_exposure)
    return {
        "protocol_comparison": protocol_rows(),
        "harmonized_diagnostic_event_rows": event,
        "nonharmonized_lee_within_subject_reference": lee_within_subject_reference(lee_reference),
        "early_usability_external_validation": usability,
        "calibration_exposure_comparison": exposure,
        "conclusions": {
            "event": "Stieger did not reproduce Lee event evidence under a directly comparable claim; training and stream semantics remain different.",
            "usability": "Lee M2 passed internal grouped validation; Stieger M2 external transport failed.",
            "calibration": "More idle did not universally solve transfer: Lee remained under-fired while Stieger remained over-fired through about 90 s.",
            "claim": "Cross-dataset numbers diagnose transport limits and must not be ranked as if protocols were identical.",
        },
    }


def write_report(payload: dict[str, Any], path: Path) -> None:
    p = payload["protocol_comparison"]
    e = payload["harmonized_diagnostic_event_rows"]
    u = payload["early_usability_external_validation"]
    x = payload["calibration_exposure_comparison"]
    r = payload["nonharmonized_lee_within_subject_reference"]
    lines = [
        "# Lee2019--Stieger2021 CNN-GRU integrated validation",
        "",
        "## 1. Protocol boundary",
        "",
        "| Dataset | Subjects/sessions | Event semantics | Stream/state | Early-usability training | Null |",
        "|---|---:|---|---|---|---|",
    ]
    for row in p:
        lines.append(
            f"| {row['dataset']} | {row['subjects_sessions']} | {row['event_semantics']} | "
            f"{row['evaluation_stream']}; {row['state']} | {row['early_usability_training']} | {row['null']} |"
        )
    lines += [
        "",
        "## 2. Closest diagnostic event comparison (target FP=2/min)",
        "",
        "| Dataset | S | n | Training | Actual FP | SEN@1s | Null@1s | Excess@1s | SEN@2s | Excess@2s |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in e:
        lines.append(
            f"| {row['dataset']} | {row['session']} | {row['n']} | {row['training']} | "
            f"{row['actual_fp_per_min']:.3f} | {row['sensitivity_1s']:.3f} | {row['null_1s']:.3f} | "
            f"{row['excess_1s']:.3f} | {row['sensitivity_2s']:.3f} | {row['excess_2s']:.3f} |"
        )
    lines += [
        "",
        "The decision layer is EWMA H30 + MA .5 s + dwell .3 s + refractory 1 s in this table, but Lee is LOSO continuous replay and Stieger is within-subject trial-cell evaluation. The rows are diagnostic context, not a model leaderboard.",
        "Sensitivity, null, and matched-row excess are aggregated separately; median(sensitivity-null) need not equal median(sensitivity)-median(null).",
        "",
        "## 3. Lee within-subject historical reference (not harmonized)",
        "",
        "| S | Threshold mode | Actual FP | SEN@1s | Excess@1s | Delay |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in r:
        lines.append(
            f"| {row['session']} | {row['threshold_mode']} | {row['actual_fp_per_min']:.3f} | "
            f"{row['sensitivity_1s']:.3f} | {row['excess_1s']:.3f} | {row['delay_seconds']:.2f} |"
        )
    lines += [
        "",
        "These higher Lee values use within-subject first-50 training and a historical instantaneous-crossing decision. They must not be substituted into the harmonized table.",
        "",
        "## 4. Early-session usability transport (FP2 excess@1s)",
        "",
        "| Dataset | Training | Model | OOS rho [95% CI] | MAE | AUROC | Dataset transport gate |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in u:
        lines.append(
            f"| {row['dataset']} | {row['training']} | {row['model']} | {row['oos_spearman_rho']:.3f} "
            f"[{row['rho_ci95'][0]:.3f},{row['rho_ci95'][1]:.3f}] | {row['mae']:.3f} | "
            f"{row['top20_auroc']:.3f} | {row['transport_gate']} |"
        )
    lines += [
        "",
        "## 5. Calibration-exposure comparison",
        "",
        "| Dataset | Exposure | n | Target | Future FP | Admissible | Log error | Excess@2s | Context |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in x:
        lines.append(
            f"| {row['dataset']} | {row['exposure_seconds']:.1f}s | {row['n']} | "
            f"{row['target_fp_per_min']:.0f} | {row['future_fp_per_min']:.3f} | "
            f"{row['admissible_n']}/{row['n']} | {row['log_error']:.3f} | "
            f"{row['excess_2s']:.3f} | {row['calibration_context']} |"
        )
    lines += [
        "",
        "## Integrated conclusion",
        "",
        "1. The Lee early-usability signal did not externally transport to Stieger.",
        "2. The direction of calibration failure changed: Lee prospective thresholds under-fired, whereas Stieger thresholds over-fired through about 90 s.",
        "3. Calibration exposure improved resolution but did not yield stable individual operating-point transfer in either dataset.",
        "4. Stieger's trial-cell release cannot validate continuous self-paced hidden state or natural-idle false alarms.",
        "5. Therefore the defensible result is a negative external validation with an explicit protocol boundary, not a cross-dataset performance ranking.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(payload: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    event = payload["harmonized_diagnostic_event_rows"]
    labels = [f"{row['dataset']} S{row['session']}" for row in event]
    x = np.arange(len(event))
    excess_1s = [row["excess_1s"] for row in event]
    excess_2s = [row["excess_2s"] for row in event]
    axes[0, 0].bar(x - .18, excess_1s, .36, label="matched-row excess@1s")
    axes[0, 0].bar(x + .18, excess_2s, .36, label="matched-row excess@2s")
    axes[0, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 0].set_title("Closest diagnostic event comparison (FP=2)")
    axes[0, 0].legend()

    usability = payload["early_usability_external_validation"]
    for index, dataset in enumerate(("Lee2019", "Stieger2021")):
        rows = [row for row in usability if row["dataset"] == dataset]
        axes[0, 1].plot(
            np.arange(3), [row["oos_spearman_rho"] for row in rows], marker="o", label=dataset
        )
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].set_xticks(np.arange(3), ("M0", "M1", "M2"))
    axes[0, 1].set_title("First-10 prediction of future usability")
    axes[0, 1].set_ylabel("Grouped OOS Spearman rho")
    axes[0, 1].legend()

    exposure = payload["calibration_exposure_comparison"]
    for dataset, marker in (("Lee2019", "o"), ("Stieger2021", "s")):
        rows = [
            row for row in exposure
            if row["dataset"] == dataset and row["target_fp_per_min"] == 2.0
        ]
        primary_rows = [row for row in rows if dataset == "Lee2019" or row["exposure_seconds"] <= 90]
        axes[1, 0].plot(
            [row["exposure_seconds"] for row in primary_rows],
            [row["future_fp_per_min"] for row in primary_rows],
            marker=marker,
            label=dataset,
        )
        axes[1, 1].plot(
            [row["exposure_seconds"] for row in primary_rows],
            [row["admissible_n"] / row["n"] for row in primary_rows],
            marker=marker,
            label=dataset,
        )
        if dataset == "Stieger2021":
            risk_rows = [row for row in rows if row["exposure_seconds"] > 90]
            axes[1, 0].scatter(
                [row["exposure_seconds"] for row in risk_rows],
                [row["future_fp_per_min"] for row in risk_rows],
                marker=marker, facecolors="none", edgecolors="C1", s=70,
                label="Stieger supported risk set",
            )
            axes[1, 1].scatter(
                [row["exposure_seconds"] for row in risk_rows],
                [row["admissible_n"] / row["n"] for row in risk_rows],
                marker=marker, facecolors="none", edgecolors="C1", s=70,
                label="Stieger supported risk set",
            )
    axes[1, 0].axhline(2, color="black", ls="--", lw=1, label="target")
    axes[1, 0].set_title("Transferred FP at target 2/min")
    axes[1, 0].set_ylabel("Future FP/min")
    axes[1, 0].set_xlabel("Known-rest exposure (s)")
    axes[1, 1].set_title("Admissible operating-point fraction")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_xlabel("Known-rest exposure (s)")
    axes[1, 1].set_ylim(-.02, 1.02)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    axes[1, 0].legend()
    axes[1, 1].legend()
    fig.suptitle("CNN-GRU cross-dataset validation: protocol-aware comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lee-matched", type=Path, default=Path("data/processed/lee2019_cnn_gru_vs_riemannian_matched_fp.json"))
    parser.add_argument("--lee-reference", type=Path, default=Path("data/processed/lee2019_cnn_gru_full54_revalidation.json"))
    parser.add_argument("--lee-usability", type=Path, default=Path("data/processed/lee2019_cnn_gru_early_usability_full54.json"))
    parser.add_argument("--lee-exposure", type=Path, default=Path("data/processed/lee2019_cnn_gru_idle_exposure_curve_subset20.json"))
    parser.add_argument("--stieger-usability", type=Path, default=Path("data/processed/stieger2021_cnn_gru_early_usability.json"))
    parser.add_argument("--stieger-exposure", type=Path, default=Path("data/processed/stieger2021_cnn_gru_rest_exposure_curve.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee_stieger_cnn_gru_integrated_validation.json"))
    parser.add_argument("--figure", type=Path, default=Path("data/processed/lee_stieger_cnn_gru_integrated_validation.png"))
    parser.add_argument("--report", type=Path, default=Path("LEE_STIEGER_CNN_GRU_INTEGRATED_VALIDATION_20260921.md"))
    args = parser.parse_args()
    payload = build_payload(
        load_json(args.lee_matched),
        load_json(args.lee_reference),
        load_json(args.lee_usability),
        load_json(args.lee_exposure),
        load_json(args.stieger_usability),
        load_json(args.stieger_exposure),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload, args.figure)
    write_report(payload, args.report)
    print(json.dumps(payload["conclusions"], indent=2))


if __name__ == "__main__":
    main()
