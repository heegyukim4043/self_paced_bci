"""Compare MA/dwell, finite-horizon GLR, and causal HSMM for MI onset."""

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

from uncued_mi.decision import instantaneous
from uncued_mi.hsmm import GaussianEmission, _normal_logpdf, causal_two_state_hsmm
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence, apply_decision, tune_source_operating_points,
)
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features

RESTS = ("pre_train", "post_train", "pre_test", "post_test")
ARMS = ("ma_dwell", "glr_1s", "hsmm")
STEP = 0.1
HORIZONS = (0.5, 1.0, 2.0)


def reference(covariances):
    matrix = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12)
    return matrix + np.eye(len(matrix)) * ridge


def fit_score(command, iti, rest):
    negative = np.r_[iti, rest]
    scaler = StandardScaler().fit(np.r_[command, negative])
    model = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=3000,
        random_state=20260729,
    ).fit(
        scaler.transform(np.r_[command, negative]),
        np.r_[np.ones(len(command)), np.zeros(len(negative))],
    )
    return scaler, model


def finite_horizon_glr(scores, emission, horizon_steps=10):
    idle = _normal_logpdf(scores, emission.idle_mean, emission.idle_std)
    command = _normal_logpdf(scores, emission.mi_mean, emission.mi_std)
    llr = command - idle
    result = np.empty(len(llr), dtype=float)
    for index in range(len(llr)):
        values = llr[max(0, index - horizon_steps + 1):index + 1]
        result[index] = np.max(np.cumsum(values[::-1]))
    return result


def hsmm_statistic(scores, emission):
    return causal_two_state_hsmm(
        scores, emission,
        minimum_mi_steps=3,
        expected_mi_steps=10.0,
        maximum_mi_steps=15,
        entry_probability=0.01,
    )


def transformed(sequence, arm, emission):
    if arm == "glr_1s":
        values = finite_horizon_glr(sequence.scores, emission, 10)
    elif arm == "hsmm":
        values = hsmm_statistic(sequence.scores, emission)
    else:
        values = sequence.scores
    return ScoredSequence(
        sequence.times, values,
        sequence.idle_intervals, sequence.command_intervals,
    )


def calibration_sequence(scores):
    return ScoredSequence(
        np.arange(len(scores)) * STEP, scores,
        np.asarray([[0.0, len(scores) * STEP]]), np.empty((0, 2)),
    )


def tune(sequences, arm):
    method = "ma_dwell" if arm == "ma_dwell" else "threshold"
    kwargs = (
        {"ma_seconds": 0.5, "dwell_seconds": 0.3}
        if arm == "ma_dwell" else {}
    )
    return tune_source_operating_points(
        sequences, method, [2.0], STEP,
        refractory_seconds=1.0, grid_size=160, **kwargs,
    )[2.0]


def decisions(sequence, arm, threshold):
    if arm == "ma_dwell":
        return apply_decision(
            sequence, "ma_dwell", threshold, STEP,
            ma_seconds=0.5, dwell_seconds=0.3,
            refractory_seconds=1.0,
        ).alarms
    return instantaneous(sequence.scores, threshold, STEP, 1.0).alarms


def onset_rates(times, alarms, intervals):
    alarm_times = times[np.asarray(alarms, dtype=bool)]
    result = {}
    delays = []
    for horizon in HORIZONS:
        hits = []
        horizon_delays = []
        for start, _ in intervals:
            candidates = alarm_times[
                (alarm_times >= start) & (alarm_times <= start + horizon)
            ]
            hits.append(bool(len(candidates)))
            if len(candidates):
                horizon_delays.append(float(candidates[0] - start))
        result[str(horizon)] = float(np.mean(hits)) if hits else np.nan
        if horizon == max(HORIZONS):
            delays = horizon_delays
    return result, (float(np.median(delays)) if delays else np.nan)


def evaluate_arm(stream, calibration, arm, emission, offsets):
    calibration_transformed = [
        transformed(calibration_sequence(values), arm, emission)
        for values in calibration
    ]
    stream_transformed = transformed(stream, arm, emission)
    transferred = tune(calibration_transformed, arm)
    common = tune([stream_transformed], arm)

    def at(point):
        alarms = decisions(stream_transformed, arm, point.threshold)
        conventional = evaluate_alarms(
            stream.times, alarms, stream.idle_intervals,
            stream.command_intervals,
        )
        observed, delay = onset_rates(
            stream.times, alarms, stream.command_intervals
        )
        null = {str(value): [] for value in HORIZONS}
        for offset in offsets:
            shifted = np.roll(alarms, int(offset))
            rates, _ = onset_rates(
                stream.times, shifted, stream.command_intervals
            )
            for key, value in rates.items():
                null[key].append(value)
        null_median = {
            key: float(np.nanmedian(value)) for key, value in null.items()
        }
        return {
            "fp_per_minute": conventional.fp_per_minute,
            "onset_tpr": observed,
            "onset_null": null_median,
            "onset_excess": {
                key: float(observed[key] - null_median[key])
                for key in observed
            },
            "onset_delay": delay,
            "event_tpr": conventional.true_positive_rate,
        }
    return {
        "transferred": at(transferred),
        "common_fp": at(common),
    }


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
        scaler, model = fit_score(
            command[command_trials < 50],
            iti[iti_trials < 50],
            rest["pre_train"][:split],
        )
        score = lambda values: model.decision_function(
            scaler.transform(values)
        )
        fit_mi = score(command[command_trials < 50])
        fit_idle = np.r_[
            score(iti[iti_trials < 50]), score(rest["pre_train"][:split])
        ]
        emission = GaussianEmission.fit(fit_idle, fit_mi)
        rest_scores = {name: score(value) for name, value in rest.items()}
        calibration = [
            rest_scores[name][:len(rest_scores[name])//2]
            for name in RESTS if name != "pre_train"
        ]
        evaluation = [
            value[len(value)//2:] for value in rest_scores.values()
        ]
        command_scores = score(command)
        trials = [
            command_scores[command_trials == trial]
            for trial in range(50, 100)
        ]
        trials = [value for value in trials if len(value)]
    stream, audit = build_rest_injected_stream(
        evaluation, trials, step_seconds=STEP,
        minimum_rest_seconds=5.0, maximum_rest_seconds=11.0,
        guard_seconds=1.0, seed=20260728 + 100 * session + subject,
    )
    rng = np.random.default_rng(99000 + 100 * session + subject)
    offsets = rng.integers(1, len(stream.times), size=shifts)
    return {
        "subject": subject, "session": session, "audit": audit.__dict__,
        "arms": {
            arm: evaluate_arm(stream, calibration, arm, emission, offsets)
            for arm in ARMS
        },
    }


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, session, repeats):
    selected = [row for row in rows if row["session"] == session]
    result = {}
    for mode in ("transferred", "common_fp"):
        arms = {}
        baseline = {
            key: np.asarray([
                row["arms"]["ma_dwell"][mode]["onset_excess"][key]
                for row in selected
            ])
            for key in map(str, HORIZONS)
        }
        for arm in ARMS:
            values = {
                key: np.asarray([
                    row["arms"][arm][mode]["onset_excess"][key]
                    for row in selected
                ])
                for key in map(str, HORIZONS)
            }
            arms[arm] = {
                "fp_median": float(np.median([
                    row["arms"][arm][mode]["fp_per_minute"]
                    for row in selected
                ])),
                "delay_median": float(np.nanmedian([
                    row["arms"][arm][mode]["onset_delay"]
                    for row in selected
                ])),
                "horizons": {
                    key: {
                        "excess_median": float(np.median(value)),
                        "excess_ci": bootstrap(
                            value, 1000 + session + int(float(key) * 10),
                            repeats,
                        ),
                        "minus_ma_median": float(
                            np.median(value - baseline[key])
                        ),
                        "minus_ma_ci": bootstrap(
                            value - baseline[key],
                            2000 + session + int(float(key) * 10),
                            repeats,
                        ),
                    }
                    for key, value in values.items()
                },
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
        default=Path("data/processed/lee2019_sequential_onset.json"),
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
        print(f"sequential onset session={session}", flush=True)
    summary = [
        summarize(rows, session, args.bootstrap) for session in args.sessions
    ]
    payload = {
        "protocol": {
            "score": "pooled rest+ITI Riemannian LR",
            "arms": ARMS,
            "horizons_seconds": HORIZONS,
            "glr_horizon_seconds": 1.0,
            "hsmm_duration_seconds": [0.3, 1.0, 1.5],
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
