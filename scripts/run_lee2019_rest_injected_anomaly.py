"""Rest-only anomaly fusion with the fixed Riemannian motor gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence, apply_decision, tune_source_operating_points,
)
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features
from uncued_mi.tsad import (
    IsolationForestDetector, RobustMahalanobis, zscore_from_reference,
)

RESTS = ("pre_train", "post_train", "pre_test", "post_test")
SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
STEP = 0.1


def reference(values):
    matrix = np.mean(np.asarray(values, dtype=float), axis=0)
    return matrix + np.eye(len(matrix)) * max(
        float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12
    )


def calibration_sequence(scores):
    return ScoredSequence(
        np.arange(len(scores)) * STEP, scores,
        np.asarray([[0.0, len(scores) * STEP]]), np.empty((0, 2)),
    )


def evaluate(stream, calibration, offsets):
    transferred = tune_source_operating_points(
        [calibration_sequence(x) for x in calibration],
        "ma_dwell", [2.0], STEP, ma_seconds=0.5, dwell_seconds=0.3,
        refractory_seconds=1.0, grid_size=160,
    )[2.0]
    oracle = tune_source_operating_points(
        [stream], "ma_dwell", [2.0], STEP, ma_seconds=0.5,
        dwell_seconds=0.3, refractory_seconds=1.0, grid_size=160,
    )[2.0]

    def at(threshold):
        decision = apply_decision(
            stream, "ma_dwell", threshold, STEP, ma_seconds=0.5,
            dwell_seconds=0.3, refractory_seconds=1.0,
        )
        observed = evaluate_alarms(
            stream.times, decision.alarms, stream.idle_intervals,
            stream.command_intervals,
        )
        null = [
            evaluate_alarms(
                stream.times, np.roll(decision.alarms, int(offset)),
                stream.idle_intervals, stream.command_intervals,
            ).true_positive_rate
            for offset in offsets
        ]
        return {
            "fp": observed.fp_per_minute,
            "tpr": observed.true_positive_rate,
            "null": float(np.median(null)),
            "excess": float(observed.true_positive_rate - np.median(null)),
            "latency": observed.median_delay_seconds,
        }
    return {
        "transferred": at(transferred.threshold),
        "oracle": at(oracle.threshold),
    }


def analyze(path, subject, session, shifts):
    with np.load(path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        command = tangent_features(data["train_command_covariance"], ref)
        iti = tangent_features(data["train_idle_iti_covariance"], ref)
        command_trial = data["train_command_trial"]
        iti_trial = data["train_idle_iti_trial"]
        fit_command, fit_iti = command_trial < 50, iti_trial < 50
        motor_scaler = StandardScaler().fit(np.r_[command[fit_command], iti[fit_iti]])
        motor = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=2000,
            random_state=20260729,
        ).fit(
            motor_scaler.transform(np.r_[command[fit_command], iti[fit_iti]]),
            np.r_[np.ones(fit_command.sum()), np.zeros(fit_iti.sum())],
        )
        rests = {
            name: tangent_features(data[f"{name}_covariance"], ref) for name in RESTS
        }
        calibration_features = [x[:len(x)//2] for x in rests.values()]
        evaluation_features = [x[len(x)//2:] for x in rests.values()]
        normal = np.concatenate(calibration_features)
        normal_scaler = StandardScaler().fit(normal)
        normal_z = normal_scaler.transform(normal)
        detectors = {
            "mahalanobis": RobustMahalanobis().fit(normal_z),
            "isolation_forest": IsolationForestDetector(20260729).fit(normal_z),
        }
        motor_calibration = [
            motor.decision_function(motor_scaler.transform(x))
            for x in calibration_features
        ]
        motor_evaluation = [
            motor.decision_function(motor_scaler.transform(x))
            for x in evaluation_features
        ]
        motor_reference = np.concatenate(motor_calibration)
        motor_calibration = [
            zscore_from_reference(x, motor_reference) for x in motor_calibration
        ]
        motor_evaluation = [
            zscore_from_reference(x, motor_reference) for x in motor_evaluation
        ]
        eval_trials = []
        for trial in range(50, 100):
            x = command[command_trial == trial]
            eval_trials.append(motor.decision_function(motor_scaler.transform(x)))
        motor_trials = [
            zscore_from_reference(x, motor_reference) for x in eval_trials
        ]
        feature_trials = [
            command[command_trial == trial] for trial in range(50, 100)
        ]
    stream_seed = 20260728 + 100 * session + subject
    rng = np.random.default_rng(95000 + 100 * session + subject)
    offsets = rng.integers(1, 2000, size=shifts)
    outputs = {}
    for index, (name, detector) in enumerate(detectors.items()):
        anomaly_reference = detector.score_samples(normal_z)
        anomaly_calibration = [
            zscore_from_reference(
                detector.score_samples(normal_scaler.transform(x)),
                anomaly_reference,
            )
            for x in calibration_features
        ]
        anomaly_evaluation = [
            zscore_from_reference(
                detector.score_samples(normal_scaler.transform(x)),
                anomaly_reference,
            )
            for x in evaluation_features
        ]
        anomaly_trials = [
            zscore_from_reference(
                detector.score_samples(normal_scaler.transform(x)),
                anomaly_reference,
            )
            for x in feature_trials
        ]
        candidates = {
            "anomaly_only": (
                anomaly_calibration, anomaly_evaluation, anomaly_trials
            ),
            "fusion": (
                [a + m for a, m in zip(anomaly_calibration, motor_calibration)],
                [a + m for a, m in zip(anomaly_evaluation, motor_evaluation)],
                [a + m for a, m in zip(anomaly_trials, motor_trials)],
            ),
            "motor": (
                motor_calibration, motor_evaluation, motor_trials
            ),
        }
        outputs[name] = {}
        for arm, (calibration, evaluation_values, trials) in candidates.items():
            stream, _ = build_rest_injected_stream(
                evaluation_values, trials, step_seconds=STEP,
                minimum_rest_seconds=5, maximum_rest_seconds=11,
                guard_seconds=1, seed=stream_seed,
            )
            safe_offsets = (offsets % (len(stream.times) - 1)) + 1
            outputs[name][arm] = evaluate(
                stream, calibration, safe_offsets
            )
    return {"subject": subject, "session": session, "models": outputs}


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_four_rest"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_rest_injected_anomaly.json"))
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session, args.shifts,
            ))
            print(f"anomaly session={session} subject={subject}", flush=True)
    summary = []
    for session in args.sessions:
        selected = [x for x in rows if x["session"] == session]
        models = {}
        for name in ("mahalanobis", "isolation_forest"):
            motor = np.asarray([
                x["models"][name]["motor"]["oracle"]["excess"] for x in selected
            ])
            arms = {}
            for arm in ("anomaly_only", "fusion"):
                excess = np.asarray([
                    x["models"][name][arm]["oracle"]["excess"] for x in selected
                ])
                delta = excess - motor
                arms[arm] = {
                    "transferred_fp_median": float(np.median([
                        x["models"][name][arm]["transferred"]["fp"] for x in selected
                    ])),
                    "oracle_fp_median": float(np.median([
                        x["models"][name][arm]["oracle"]["fp"] for x in selected
                    ])),
                    "excess_median": float(np.median(excess)),
                    "excess_ci": bootstrap(excess, 100 + session, args.bootstrap),
                    "minus_motor_median": float(np.median(delta)),
                    "minus_motor_ci": bootstrap(delta, 200 + session, args.bootstrap),
                }
            arms["motor_excess_median"] = float(np.median(motor))
            models[name] = arms
        summary.append({"session": session, "subjects": len(selected), "models": models})
    payload = {
        "protocol": {
            "stream": "four-rest-injected",
            "normal_training": "first halves of all four rest blocks",
            "fusion": "robust-z anomaly + robust-z Riemannian motor score",
            "fp_axis": "transferred and oracle common FP=2",
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
