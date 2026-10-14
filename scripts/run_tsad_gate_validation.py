"""Locked comparison of three idle-trained TSAD scores and motor-score fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.riemannian import tangent_features, window_covariances
from uncued_mi.tsad import (
    IsolationForestDetector,
    RobustMahalanobis,
    fit_tcn_predictor,
    make_contexts,
    tcn_prediction_scores,
    zscore_from_reference,
)

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
SFREQ, WINDOW, STEP, CONTEXT = 250, 250, 0.5, 4


def positive_ends(data, stop: int = 50) -> np.ndarray:
    output = []
    for trial in range(stop):
        size = int(data["positive_bag_sizes"][trial])
        output.extend(data["positive_end_samples"][trial, :size])
    return np.asarray(output, dtype=np.int64)


def interval_mask(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in intervals:
        mask |= (times >= start) & (times < stop)
    return mask


def evaluate(times, scores, idle, command, seed, shifts):
    finite = np.isfinite(scores)
    fill = float(np.nanmedian(scores))
    scores = np.where(finite, scores, fill)
    sequence = ScoredSequence(times, scores, idle, command)
    point = tune_source_operating_points(
        [sequence], "ma_dwell", [2.0], STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0, grid_size=160,
    )[2.0]
    decision = apply_decision(
        sequence, "ma_dwell", point.threshold, STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
    )
    observed = evaluate_alarms(times, decision.alarms, idle, command)
    rng = np.random.default_rng(seed)
    null = np.asarray([
        evaluate_alarms(times, np.roll(decision.alarms, int(offset)), idle, command).true_positive_rate
        for offset in rng.integers(1, len(decision.alarms), shifts)
    ])
    return {
        "fp_per_minute": observed.fp_per_minute,
        "observed_tpr": observed.true_positive_rate,
        "null_tpr": float(np.median(null)),
        "excess_tpr": float(observed.true_positive_rate - np.median(null)),
        "latency": observed.median_delay_seconds,
        "threshold": point.threshold,
    }


def analyze(path: Path, session: int, subject: int, shifts: int, tcn_epochs: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        signal, pre = np.asarray(data["signal"]), np.asarray(data["pre_rest"])
        first_end, step = 3 * SFREQ, int(STEP * SFREQ)
        ends = np.arange(first_end, len(signal) + 1, step, dtype=np.int64)
        times = ends / SFREQ
        pre_ends = np.arange(WINDOW, len(pre) + 1, step, dtype=np.int64)
        reference = window_covariances(pre, pre_ends, WINDOW).mean(axis=0)
        reference += np.eye(len(reference)) * np.trace(reference) / len(reference) * 1e-6
        features = tangent_features(window_covariances(signal, ends, WINDOW), reference)
        pre_features = tangent_features(window_covariances(pre, pre_ends, WINDOW), reference)

        fit_idle_intervals = data["iti_intervals"][data["iti_trials"] < 50]
        idle_fit_mask = interval_mask(times, fit_idle_intervals)
        normal = np.r_[pre_features, features[idle_fit_mask]]
        normal_scaler = StandardScaler().fit(normal)
        normal_z = normal_scaler.transform(normal)
        features_z = normal_scaler.transform(features)

        train_positive = positive_ends(data)
        iti_instance_mask = data["iti_instance_trials"] < 50
        train_idle = data["iti_end_samples"][iti_instance_mask]
        train_ends = np.r_[train_positive, train_idle]
        train_y = np.r_[np.ones(len(train_positive)), np.zeros(len(train_idle))]
        train_x = tangent_features(window_covariances(signal, train_ends, WINDOW), reference)
        motor_scaler = StandardScaler().fit(train_x)
        motor = LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=SEEDS[0],
        ).fit(motor_scaler.transform(train_x), train_y)
        motor_scores = motor.decision_function(motor_scaler.transform(features))
        normal_motor = np.r_[
            motor.decision_function(motor_scaler.transform(pre_features)),
            motor_scores[idle_fit_mask],
        ]
        motor_z = zscore_from_reference(motor_scores, normal_motor)

        detectors = {
            "mahalanobis": RobustMahalanobis().fit(normal_z),
            "isolation_forest": IsolationForestDetector(SEEDS[0]).fit(normal_z),
        }
        anomaly_scores = {
            name: detector.score_samples(features_z) for name, detector in detectors.items()
        }
        anomaly_reference = {
            name: detector.score_samples(normal_z) for name, detector in detectors.items()
        }

        context_x, context_y = [], []
        pre_x, pre_y = make_contexts(normal_scaler.transform(pre_features), CONTEXT)
        context_x.append(pre_x)
        context_y.append(pre_y)
        for interval in fit_idle_intervals:
            sequence = features_z[(times >= interval[0]) & (times < interval[1])]
            x, y = make_contexts(sequence, CONTEXT)
            if len(x):
                context_x.append(x)
                context_y.append(y)
        context_x, context_y = np.concatenate(context_x), np.concatenate(context_y)
        tcn = fit_tcn_predictor(context_x, context_y, seed=SEEDS[0], epochs=tcn_epochs)
        tcn_score = tcn_prediction_scores(tcn, features_z, CONTEXT)
        # Training residual distribution, evaluated in batches through the same model.
        import torch
        with torch.no_grad():
            prediction = []
            for start in range(0, len(context_x), 256):
                prediction.append(tcn(torch.from_numpy(context_x[start:start + 256])).numpy())
        prediction = np.concatenate(prediction)
        tcn_reference = np.mean(np.square(prediction - context_y), axis=1)
        anomaly_scores["tcn_predictor"] = tcn_score
        anomaly_reference["tcn_predictor"] = tcn_reference

        locked_idle = data["iti_intervals"][(data["iti_trials"] >= 50) & (data["iti_trials"] < 100)]
        locked_command = data["command_intervals"][50:100]

    outputs = {}
    for index, name in enumerate(("mahalanobis", "isolation_forest", "tcn_predictor")):
        anomaly = anomaly_scores[name]
        finite_reference = anomaly_reference[name][np.isfinite(anomaly_reference[name])]
        anomaly_z = zscore_from_reference(anomaly, finite_reference)
        outputs[name] = {
            "anomaly_only": evaluate(
                times, anomaly_z, locked_idle, locked_command,
                SEEDS[0] + session * 1000 + subject + index * 10, shifts,
            ),
            "fusion": evaluate(
                times, anomaly_z + motor_z, locked_idle, locked_command,
                SEEDS[0] + session * 1000 + subject + index * 10, shifts,
            ),
        }
    return {"session": session, "subject": subject, "models": outputs}


def bootstrap(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--tcn-epochs", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/tsad_gate_validation.json"))
    args = parser.parse_args()
    if args.smoke:
        args.subjects, args.sessions, args.shifts, args.bootstrap, args.tcn_epochs = args.subjects[:1], [1], 20, 100, 2
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = args.cache_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz"
            rows.append(analyze(path, session, subject, args.shifts, args.tcn_epochs))
            print(f"done session={session} subject={subject}", flush=True)

    comparator = json.loads(Path("data/processed/mil_riemannian_comparator.json").read_text(encoding="utf-8"))
    summary = []
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        rmap = {row["subject"]: row["excess_tpr"] for row in comparator["rows"] if row["session"] == session}
        model_summary = {}
        for model in ("mahalanobis", "isolation_forest", "tcn_predictor"):
            anomaly = np.asarray([row["models"][model]["anomaly_only"]["excess_tpr"] for row in selected])
            fusion = np.asarray([row["models"][model]["fusion"]["excess_tpr"] for row in selected])
            baseline = np.asarray([rmap[row["subject"]] for row in selected])
            model_summary[model] = {
                "anomaly_excess_median": float(np.median(anomaly)),
                "anomaly_excess_ci": bootstrap(anomaly, 100 + session, args.bootstrap),
                "fusion_excess_median": float(np.median(fusion)),
                "fusion_excess_ci": bootstrap(fusion, 200 + session, args.bootstrap),
                "fusion_minus_riemannian_median": float(np.median(fusion - baseline)),
                "fusion_minus_riemannian_ci": bootstrap(fusion - baseline, 300 + session, args.bootstrap),
                "fusion_fp_median": float(np.median([row["models"][model]["fusion"]["fp_per_minute"] for row in selected])),
            }
        summary.append({"session": session, "subjects": len(selected), "models": model_summary})
    payload = {
        "protocol": {
            "preregistered": True, "context_seconds": CONTEXT * STEP,
            "tcn_epochs": args.tcn_epochs, "fp_axis": 2.0, "null_shifts": args.shifts,
            "fusion": "equal-weight robust-z anomaly + robust-z motor LR",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
