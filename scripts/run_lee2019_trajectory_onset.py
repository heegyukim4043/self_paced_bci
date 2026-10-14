"""Causal short-trajectory models for trial detection within one second."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import instantaneous
from uncued_mi.hsmm import GaussianEmission
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_sequential_onset import (
    RESTS, STEP, calibration_sequence, evaluate_arm, fit_score,
    onset_rates, reference, tune,
)

ARMS = ("trajectory_lr", "trajectory_hgb", "trajectory_rbf")


def causal_features(values):
    scores = np.asarray(values, dtype=float)
    columns = [scores]
    for lag in (1, 3):
        delayed = np.r_[np.repeat(scores[0], lag), scores[:-lag]]
        columns.append(scores - delayed)
    for length in (3, 5, 10):
        mean, maximum, standard = [], [], []
        for index in range(len(scores)):
            window = scores[max(0, index - length + 1):index + 1]
            mean.append(np.mean(window))
            maximum.append(np.max(window))
            standard.append(np.std(window))
        columns.extend([mean, maximum, standard])
    return np.column_stack(columns)


def split_rest(values, length=50):
    return [
        values[start:min(start + length, len(values))]
        for start in range(0, len(values), length)
        if len(values[start:min(start + length, len(values))]) >= 3
    ]


def fit_trajectory(positive_segments, negative_segments, idle_fill):
    positive_features, negative_features = [], []
    for index, positive in enumerate(positive_segments):
        negative = negative_segments[index % len(negative_segments)]
        transition = np.r_[
            negative, np.repeat(idle_fill, 10), positive
        ]
        values = causal_features(transition)
        negative_features.append(values[:len(negative)])
        positive_features.append(values[-len(positive):])
    for negative in negative_segments:
        negative_features.append(causal_features(negative))
    positive = np.concatenate(positive_features)
    negative = np.concatenate(negative_features)
    x = np.r_[positive, negative]
    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    lr = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=3000,
        random_state=20260729,
    ).fit(transformed, y)
    weights = np.where(
        y == 1, len(y) / (2 * np.sum(y == 1)),
        len(y) / (2 * np.sum(y == 0)),
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.0,
        random_state=20260729,
    ).fit(transformed, y, sample_weight=weights)
    rbf = SVC(
        C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
        random_state=20260729,
    ).fit(transformed, y)
    return scaler, {
        "trajectory_lr": lr,
        "trajectory_hgb": hgb,
        "trajectory_rbf": rbf,
    }


def predict_segments(scaler, model, segments):
    output = []
    for values in segments:
        features = scaler.transform(causal_features(values))
        if hasattr(model, "decision_function"):
            output.append(model.decision_function(features))
        else:
            output.append(model.predict_proba(features)[:, 1])
    return output


def predict_continuous(scaler, model, stream, idle_fill, valid_min):
    values = stream.scores.copy()
    values[values < valid_min - 1.0] = idle_fill
    features = scaler.transform(causal_features(values))
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
    else:
        scores = model.predict_proba(features)[:, 1]
    return ScoredSequence(
        stream.times, scores, stream.idle_intervals, stream.command_intervals
    )


def generic_evaluate(stream, calibration, offsets):
    point = tune([calibration_sequence(x) for x in calibration], "hsmm")
    common = tune([stream], "hsmm")

    def at(selected):
        alarms = instantaneous(
            stream.scores, selected.threshold, STEP, 1.0
        ).alarms
        conventional = evaluate_alarms(
            stream.times, alarms, stream.idle_intervals,
            stream.command_intervals,
        )
        observed, delay = onset_rates(
            stream.times, alarms, stream.command_intervals
        )
        null = []
        for offset in offsets:
            shifted, _ = onset_rates(
                stream.times, np.roll(alarms, int(offset)),
                stream.command_intervals,
            )
            null.append(shifted["1.0"])
        null_median = float(np.median(null))
        return {
            "fp_per_minute": conventional.fp_per_minute,
            "tpr_1s": observed["1.0"],
            "null_1s": null_median,
            "excess_1s": float(observed["1.0"] - null_median),
            "delay": delay,
        }
    return {"transferred": at(point), "common_fp": at(common)}


def analyze(path, subject, session, shifts):
    with np.load(path, allow_pickle=False) as data:
        pre = data["pre_train_covariance"]
        split = len(pre) // 2
        ref = reference(pre[:split])
        command = tangent_features(data["train_command_covariance"], ref)
        iti = tangent_features(data["train_idle_iti_covariance"], ref)
        rest = {
            name: tangent_features(data[f"{name}_covariance"], ref)
            for name in RESTS
        }
        command_trials = data["train_command_trial"]
        iti_trials = data["train_idle_iti_trial"]
        base_scaler, base_model = fit_score(
            command[command_trials < 50],
            iti[iti_trials < 50],
            rest["pre_train"][:split],
        )
        base_score = lambda values: base_model.decision_function(
            base_scaler.transform(values)
        )
        command_scores = base_score(command)
        iti_scores = base_score(iti)
        rest_scores = {name: base_score(value) for name, value in rest.items()}
        positive = [
            command_scores[command_trials == trial] for trial in range(50)
        ]
        negative = [
            iti_scores[iti_trials == trial] for trial in range(50)
        ] + split_rest(rest_scores["pre_train"][:split])
        positive = [values for values in positive if len(values)]
        negative = [values for values in negative if len(values)]
        idle_fill = float(np.median(np.concatenate(negative)))
        trajectory_scaler, models = fit_trajectory(
            positive, negative, idle_fill
        )
        calibration_raw = [
            rest_scores[name][:len(rest_scores[name])//2]
            for name in RESTS if name != "pre_train"
        ]
        evaluation_raw = [
            value[len(value)//2:] for value in rest_scores.values()
        ]
        trials_raw = [
            command_scores[command_trials == trial]
            for trial in range(50, 100)
        ]
        emission = GaussianEmission.fit(
            np.r_[iti_scores[iti_trials < 50], rest_scores["pre_train"][:split]],
            command_scores[command_trials < 50],
        )

    raw_stream, audit = build_rest_injected_stream(
        evaluation_raw, trials_raw, step_seconds=STEP,
        minimum_rest_seconds=5, maximum_rest_seconds=11,
        guard_seconds=1, seed=20260728 + 100 * session + subject,
    )
    rng = np.random.default_rng(99500 + 100 * session + subject)
    offsets = rng.integers(1, len(raw_stream.times), size=shifts)
    result = {
        "hsmm": evaluate_arm(
            raw_stream, calibration_raw, "hsmm", emission, offsets
        )
    }
    for arm, model in models.items():
        calibration = predict_segments(
            trajectory_scaler, model, calibration_raw
        )
        valid_min = float(np.min(np.concatenate(
            [*evaluation_raw, *trials_raw]
        )))
        stream = predict_continuous(
            trajectory_scaler, model, raw_stream, idle_fill, valid_min
        )
        result[arm] = generic_evaluate(stream, calibration, offsets)
    return {
        "subject": subject, "session": session,
        "audit": audit.__dict__, "arms": result,
    }


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, session, repeats):
    selected = [row for row in rows if row["session"] == session]
    result = {}
    for mode in ("transferred", "common_fp"):
        hsmm = np.asarray([
            row["arms"]["hsmm"][mode]["onset_excess"]["1.0"]
            for row in selected
        ])
        arms = {
            "hsmm": {
                "fp_median": float(np.median([
                    row["arms"]["hsmm"][mode]["fp_per_minute"]
                    for row in selected
                ])),
                "excess_median": float(np.median(hsmm)),
                "excess_ci": bootstrap(hsmm, 100 + session, repeats),
            }
        }
        for arm in ARMS:
            values = np.asarray([
                row["arms"][arm][mode]["excess_1s"] for row in selected
            ])
            delta = values - hsmm
            arms[arm] = {
                "fp_median": float(np.median([
                    row["arms"][arm][mode]["fp_per_minute"]
                    for row in selected
                ])),
                "raw_tpr_median": float(np.median([
                    row["arms"][arm][mode]["tpr_1s"] for row in selected
                ])),
                "excess_median": float(np.median(values)),
                "excess_ci": bootstrap(values, 200 + session, repeats),
                "minus_hsmm_median": float(np.median(delta)),
                "minus_hsmm_ci": bootstrap(delta, 300 + session, repeats),
            }
        result[mode] = arms
    return {"session": session, "subjects": len(selected), "modes": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_four_rest")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=range(1, 55))
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_trajectory_onset.json"),
    )
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.cache_root / f"session{session}"
                / f"sub-{subject:02d}.npz",
                subject, session, args.shifts,
            ))
        print(f"trajectory onset session={session}", flush=True)
    summary = [
        summarize(rows, session, args.bootstrap) for session in args.sessions
    ]
    payload = {
        "protocol": {
            "target": "trial detection within 1 second",
            "features": "causal score trajectory, reset per simulated segment",
            "arms": ("hsmm", *ARMS),
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
