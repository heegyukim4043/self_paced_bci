"""Evaluate external UI safety policies and the observed Lee2019 regime shift.

UI results are scenario analyses, not measurements of human behaviour.  The
regime arm uses the existing locked train/test window-AUC audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEED = 20260805
BOOTSTRAPS = 20_000


def median_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, float)
    samples = values[rng.integers(0, len(values), (BOOTSTRAPS, len(values)))]
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


def summarize(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, float)
    return {
        "median": float(np.median(values)),
        "iqr": np.quantile(values, [0.25, 0.75]).tolist(),
        "subject_bootstrap_95_ci": median_ci(values, rng),
    }


def ui_analysis(rng: np.random.Generator) -> dict:
    source = json.loads(
        (PROCESSED / "lee2019_two_stage_confirmation.json").read_text(encoding="utf-8")
    )
    rows = [row["operating_points"]["2.0"] for row in source["rows"]]
    candidate_fp = np.asarray([row["candidate_fp_per_minute"] for row in rows])
    candidate_tpr = np.asarray([row["candidate_tpr"] for row in rows])

    # External confirmation: sensitivity is P(confirm | intended command was
    # detected), and false_acceptance is P(confirm | idle false prompt).
    confirmation_grid = []
    for sensitivity in (0.80, 0.90, 0.95, 0.98):
        for false_acceptance in (0.01, 0.02, 0.05, 0.10):
            residual_fp = candidate_fp * false_acceptance
            retained_tpr = candidate_tpr * sensitivity
            confirmation_grid.append({
                "true_confirmation_sensitivity": sensitivity,
                "idle_prompt_false_acceptance": false_acceptance,
                "residual_committed_fp_per_minute": summarize(residual_fp, rng),
                "retained_raw_tpr": summarize(retained_tpr, rng),
                "idle_confirmation_prompts_per_hour": summarize(candidate_fp * 60.0, rng),
                "false_commands_prevented_fraction": 1.0 - false_acceptance,
                "detected_true_commands_lost_fraction": 1.0 - sensitivity,
            })

    # Undo cannot prevent transient execution. It only lowers the rate of
    # commands that remain uncorrected after an undo window.
    undo_grid = []
    for false_undo in (0.80, 0.90, 0.95, 0.98):
        for true_undo in (0.01, 0.02, 0.05):
            undo_grid.append({
                "false_command_undo_probability": false_undo,
                "true_command_accidental_undo_probability": true_undo,
                "transient_fp_per_minute": summarize(candidate_fp, rng),
                "uncorrected_fp_per_minute": summarize(candidate_fp * (1.0 - false_undo), rng),
                "retained_raw_tpr_after_undo": summarize(candidate_tpr * (1.0 - true_undo), rng),
            })

    return {
        "status": "scenario_analysis_not_human_validation",
        "base_gate_nominal_target": 2.0,
        "base_candidate_fp_per_minute": summarize(candidate_fp, rng),
        "base_candidate_raw_tpr": summarize(candidate_tpr, rng),
        "explicit_confirmation": confirmation_grid,
        "undo": undo_grid,
        "interpretation_limit": (
            "Confirmation/undo probabilities are assumed behavioural parameters. "
            "They quantify requirements for a usable interface but do not establish "
            "that users achieve those probabilities."
        ),
    }


def regime_analysis(rng: np.random.Generator) -> dict:
    source = json.loads(
        (PROCESSED / "integrated_train_test_auc_subset20.json").read_text(encoding="utf-8")
    )
    output = []
    for window in (0.5, 1.0, 2.0, 3.0):
        rows = [row for row in source["rows"] if row["window_seconds"] == window]
        train = np.asarray([row["train_auc"] for row in rows])
        test = np.asarray([row["test_auc"] for row in rows])
        delta = test - train
        statistic, pvalue = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        output.append({
            "window_seconds": window,
            "subjects": len(rows),
            "train_auc": summarize(train, rng),
            "feedback_test_auc": summarize(test, rng),
            "paired_delta_auc": summarize(delta, rng),
            "subjects_improved": int(np.sum(delta > 0)),
            "subjects_worsened": int(np.sum(delta < 0)),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p": float(pvalue),
        })
    return {
        "status": "observed_within_subject_regime_association",
        "windows": output,
        "binding_interpretation": (
            "Window separability is higher in the online-feedback test phase. "
            "Because cursor feedback, practice, engagement, and time are jointly changed, "
            "this is not an isolated causal estimate of feedback or deployable uncued performance."
        ),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    result = {
        "ui_ux_safety_path": ui_analysis(rng),
        "regime_change_path": regime_analysis(rng),
    }
    output = PROCESSED / "lee2019_ui_regime_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
