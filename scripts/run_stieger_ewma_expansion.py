"""Causal EWMA operating-point adaptation on the fixed Stieger S1--S20 cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_stieger_riemannian_smoke import (
    event_metrics, regularized_reference, tangent_features, tune_threshold,
)
from scripts.summarize_stieger_smoke import bootstrap_median


SUBJECTS = tuple(range(1, 21))
STEP_SECONDS = 0.1
HALF_LIFE_SECONDS = 30.0


def causal_ewma_z(
    scores: np.ndarray,
    initial_mean: float,
    initial_variance: float,
    half_life_seconds: float,
) -> np.ndarray:
    """Normalize before updating; state follows observed feedback exposure."""
    alpha = 1.0 - 2.0 ** (-STEP_SECONDS / half_life_seconds)
    mean = float(initial_mean)
    variance = max(float(initial_variance), 1e-8)
    floor = 0.25 * np.sqrt(variance)
    output = np.empty(len(scores), dtype=float)
    for index, value in enumerate(np.asarray(scores, float)):
        output[index] = (value - mean) / max(np.sqrt(max(variance, 0.0)), floor)
        delta = value - mean
        mean += alpha * delta
        variance = (1.0 - alpha) * (variance + alpha * delta * delta)
    return output


def analyze_subject(subject: int, session: int = 1) -> dict:
    cache = ROOT / "data" / "interim" / "stieger2021" / f"S{subject}_Session_{session}_covariances.npz"
    with np.load(cache, allow_pickle=False) as data:
        cov = data["covariances"].astype(np.float64)
        trial = data["trial_index"].astype(int)
        label = data["is_active"].astype(int)
        run = data["run_number"].astype(int)
        speed = data["cursor_speed"].astype(float)
        audit = json.loads(str(data["audit_json"]))

    train = trial < 225
    test = ~train
    rest_train = train & (label == 0)
    reference = regularized_reference(cov[rest_train])
    features = tangent_features(cov, reference)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear"
    ).fit(features[train], label[train])
    raw_score = classifier.decision_function(features)

    finite_train_speed = speed[train & np.isfinite(speed)]
    speed_fill = float(np.median(finite_train_speed))
    speed_missing = ~np.isfinite(speed)
    speed_feature = np.c_[
        np.log1p(np.where(speed_missing, speed_fill, speed)),
        speed_missing.astype(float),
    ]
    residualizer = LinearRegression().fit(speed_feature[train], raw_score[train])
    residual_score = raw_score - residualizer.predict(speed_feature)

    arms = {}
    for arm, score in (("raw", raw_score), ("speed_residualized", residual_score)):
        anchor_mean = float(np.mean(score[rest_train]))
        anchor_variance = float(np.var(score[rest_train], ddof=1))
        static_z = (score - anchor_mean) / np.sqrt(max(anchor_variance, 1e-8))
        threshold, training_fp = tune_threshold(
            trial[train], label[train], static_z[train]
        )
        adaptive_z = causal_ewma_z(
            score, anchor_mean, anchor_variance, HALF_LIFE_SECONDS
        )
        arms[arm] = {
            "training_static_fp_per_minute": training_fp,
            "threshold_on_anchor_z": threshold,
            "static": event_metrics(
                trial[test], label[test], run[test], static_z[test], threshold,
                shifts=500, seed=20260808 + subject,
            ),
            "ewma_h30": event_metrics(
                trial[test], label[test], run[test], adaptive_z[test], threshold,
                shifts=500, seed=20261808 + subject,
            ),
        }
    return {
        "subject": subject,
        "session": session,
        "group": "MBSR" if int(audit["mbsr_subject"]) else "control",
        "cursor_speed_missing_windows": int(speed_missing.sum()),
        "arms": arms,
    }


def summarize(rows: list[dict], arm: str, condition: str, seed: int) -> dict:
    fp = np.asarray([row["arms"][arm][condition]["fp_per_minute"] for row in rows])
    excess = np.asarray([row["arms"][arm][condition]["excess_tpr"] for row in rows])
    return {
        "n": len(rows),
        "fp_per_minute_median": float(np.median(fp)),
        "fp_per_minute_ci": bootstrap_median(fp, seed, 20_000),
        "target_ratio_median": float(np.median(fp / 2.0)),
        "target_ratio_ci": bootstrap_median(fp / 2.0, seed + 1, 20_000),
        "excess_tpr_median": float(np.median(excess)),
        "excess_tpr_ci": bootstrap_median(excess, seed + 2, 20_000),
        "subjects_with_lower_fp_than_static": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for subject in args.subjects:
        rows.append(analyze_subject(subject, args.session))
        print(f"done subject={subject}", flush=True)

    summary = {}
    for arm_index, arm in enumerate(("raw", "speed_residualized")):
        summary[arm] = {}
        for group_index, (group, selected) in enumerate(
            [("overall", rows),
             ("control", [row for row in rows if row["group"] == "control"]),
             ("MBSR", [row for row in rows if row["group"] == "MBSR"])]
        ):
            static = summarize(selected, arm, "static", 20260808 + arm_index * 100 + group_index * 10)
            adaptive = summarize(selected, arm, "ewma_h30", 20260809 + arm_index * 100 + group_index * 10)
            static_fp = np.asarray([row["arms"][arm]["static"]["fp_per_minute"] for row in selected])
            adaptive_fp = np.asarray([row["arms"][arm]["ewma_h30"]["fp_per_minute"] for row in selected])
            static_excess = np.asarray([row["arms"][arm]["static"]["excess_tpr"] for row in selected])
            adaptive_excess = np.asarray([row["arms"][arm]["ewma_h30"]["excess_tpr"] for row in selected])
            adaptive["subjects_with_lower_fp_than_static"] = int(np.sum(adaptive_fp < static_fp))
            inference = {
                "subjects_closer_to_target_fp2": int(
                    np.sum(np.abs(adaptive_fp - 2.0) < np.abs(static_fp - 2.0))
                ),
                "static_absolute_target_ratio_error_median": float(
                    np.median(np.abs(static_fp / 2.0 - 1.0))
                ),
                "ewma_absolute_target_ratio_error_median": float(
                    np.median(np.abs(adaptive_fp / 2.0 - 1.0))
                ),
                "fp_paired_wilcoxon_p": float(wilcoxon(adaptive_fp, static_fp).pvalue),
                "target_error_paired_wilcoxon_p": float(
                    wilcoxon(np.abs(adaptive_fp / 2.0 - 1.0),
                             np.abs(static_fp / 2.0 - 1.0)).pvalue
                ),
                "paired_excess_delta_median": float(
                    np.median(adaptive_excess - static_excess)
                ),
                "excess_paired_wilcoxon_p": float(
                    wilcoxon(adaptive_excess, static_excess).pvalue
                ),
                "subjects_fp_in_1p5_to_2p5": int(
                    np.sum((adaptive_fp >= 1.5) & (adaptive_fp <= 2.5))
                ),
                "subjects_fp_in_1_to_4": int(
                    np.sum((adaptive_fp >= 1.0) & (adaptive_fp <= 4.0))
                ),
            }
            summary[arm][group] = {
                "static": static, "ewma_h30": adaptive, "paired_inference": inference
            }

    payload = {
        "subjects": list(args.subjects),
        "session": args.session,
        "protocol": {
            "threshold": "first-half intentional-rest target 2/min on anchor z score",
            "ewma": "causal all-score H=30 over observed feedback windows; no labels",
            "decision": "MA 0.5 s + dwell 0.3 s + refractory 1 s, reset per trial cell",
            "limitation": (
                "Published cells omit EEG between feedback intervals. H=30 is therefore "
                "feedback-exposure time, not continuous wall-clock time."
            ),
        },
        "summary": summary,
        "rows": rows,
    }
    if args.output is not None:
        output = args.output
    elif args.session == 1 and tuple(args.subjects) == SUBJECTS:
        output = ROOT / "data" / "processed" / "stieger2021_expansion20_ewma.json"
    else:
        output = ROOT / "data" / "processed" / f"stieger2021_session{args.session}_ewma.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
