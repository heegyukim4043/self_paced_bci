from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.features import decision_subsample, sample_mask
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    tune_source_operating_points,
)


def half(intervals: np.ndarray, part: str) -> np.ndarray:
    intervals = np.asarray(intervals).reshape(-1, 2)
    midpoint = len(intervals) // 2
    return intervals[:midpoint] if part == "first" else intervals[midpoint:]


def invsqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, np.max(values) * 1e-8)
    return (vectors * values**-0.5) @ vectors.T


def tangent_features(covariances: np.ndarray, reference: np.ndarray) -> np.ndarray:
    whitening = invsqrt(reference)
    matrices = whitening @ covariances @ whitening.T
    values, vectors = np.linalg.eigh(matrices)
    values = np.maximum(values, np.max(values, axis=1, keepdims=True) * 1e-8)
    logs = (vectors * np.log(values)[:, None, :]) @ np.swapaxes(vectors, 1, 2)
    upper = np.triu_indices(logs.shape[1])
    features = logs[:, upper[0], upper[1]]
    features[:, upper[0] != upper[1]] *= np.sqrt(2.0)
    return features


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n = len(positive)
    return float((np.sum(ranks[:n]) - n * (n + 1) / 2) / (n * len(negative)))


def bootstrap_median_ci(values: np.ndarray, seed: int, n: int) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.median(
        values[rng.integers(0, len(values), size=(n, len(values)))], axis=1
    )
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_integrated/session1"))
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--windows", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--target-fp", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--decision-step", type=float, default=0.5)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=Path("data/processed/tier1_riemannian_subset20.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for subject in args.subjects:
        path = args.cache_root / f"sub-{subject:02d}_integrated.npz"
        with np.load(path, allow_pickle=False) as data:
            command_train = half(data["train_command_intervals"], "first")
            command_test = half(data["train_command_intervals"], "second")
            idle_train = half(data["train_idle_iti_intervals"], "first")
            idle_test = half(data["train_idle_iti_intervals"], "second")
            for window_seconds in args.windows:
                suffix = str(window_seconds).replace(".", "p")
                times = data[f"train_w{suffix}_times"]
                covariance = np.asarray(data[f"train_w{suffix}_covariance"], dtype=float)
                pre_covariance = np.asarray(data[f"train_w{suffix}_pre_covariance"], dtype=float)
                reference = np.mean(pre_covariance, axis=0)
                reference += np.eye(reference.shape[0]) * np.trace(reference) / reference.shape[0] * 1e-6
                features = tangent_features(covariance, reference)
                train_command_mask = sample_mask(times, command_train.tolist())
                train_idle_mask = sample_mask(times, idle_train.tolist())
                train_mask = train_command_mask | train_idle_mask
                y = train_command_mask[train_mask].astype(int)
                scaler = StandardScaler().fit(features[train_mask])
                model = LogisticRegression(
                    C=1.0, class_weight="balanced", solver="liblinear",
                    max_iter=2000, random_state=args.seed,
                ).fit(scaler.transform(features[train_mask]), y)
                scores = model.decision_function(scaler.transform(features))
                decision_times, decision_scores = decision_subsample(
                    times, scores, args.decision_step
                )
                sequence = ScoredSequence(
                    decision_times, decision_scores, idle_test, command_test
                )
                points = tune_source_operating_points(
                    [sequence], "ma_dwell", args.target_fp, args.decision_step,
                    ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
                    grid_size=160,
                )
                command_mask = sample_mask(decision_times, command_test.tolist())
                idle_mask = sample_mask(decision_times, idle_test.tolist())
                window_auc = auc(decision_scores[command_mask], decision_scores[idle_mask])
                for target_fp, point in points.items():
                    result = apply_decision(
                        sequence, "ma_dwell", point.threshold, args.decision_step,
                        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
                    )
                    observed = evaluate_alarms(
                        decision_times, result.alarms, idle_test, command_test
                    )
                    null = np.empty(args.shifts)
                    for index, offset in enumerate(
                        rng.integers(1, len(result.alarms), size=args.shifts)
                    ):
                        null[index] = evaluate_alarms(
                            decision_times, np.roll(result.alarms, int(offset)),
                            idle_test, command_test,
                        ).true_positive_rate
                    rows.append({
                        "subject": subject,
                        "window_seconds": window_seconds,
                        "target_fp_axis": target_fp,
                        "window_auc": window_auc,
                        "selected_threshold": point.threshold,
                        "fp_per_minute": observed.fp_per_minute,
                        "observed_tpr": observed.true_positive_rate,
                        "null_tpr_median": float(np.median(null)),
                        "excess_tpr": float(observed.true_positive_rate - np.median(null)),
                    })
        print(f"done subject {subject}")
    summary = []
    for window in args.windows:
        for target_fp in args.target_fp:
            selected = [
                x for x in rows
                if x["window_seconds"] == window and x["target_fp_axis"] == target_fp
            ]
            excess = np.asarray([x["excess_tpr"] for x in selected])
            summary.append({
                "window_seconds": window,
                "target_fp_axis": target_fp,
                "subjects": len(selected),
                "window_auc_median": float(np.median([x["window_auc"] for x in selected])),
                "fp_per_minute_median": float(np.median([x["fp_per_minute"] for x in selected])),
                "observed_tpr_median": float(np.median([x["observed_tpr"] for x in selected])),
                "null_tpr_median": float(np.median([x["null_tpr_median"] for x in selected])),
                "excess_tpr_median": float(np.median(excess)),
                "excess_tpr_subject_bootstrap_95_ci": bootstrap_median_ci(
                    excess, args.seed + int(window * 100) + int(target_fp), args.bootstrap
                ),
            })
    payload = {
        "protocol": {
            "training": "within-subject first 50 trials",
            "evaluation": "held-out last 50 trials",
            "reference": "current-session pre-rest arithmetic covariance (EA)",
            "fp_axis": "held-out diagnostic oracle; score-quality go/no-go only",
            "decision": "MA 0.5s + dwell 0.3s + refractory 1.0s",
            "binding": "FP=2 excess median subject-bootstrap CI lower > 0.10",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
