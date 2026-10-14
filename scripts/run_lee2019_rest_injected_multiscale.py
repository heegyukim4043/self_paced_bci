"""Multi-scale filter-bank Riemannian gate on rest-injected streams."""

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

from uncued_mi.features import _causal_band_signal
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence, apply_decision, tune_source_operating_points,
)
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features, window_covariances

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
RESTS = ("pre_train", "post_train", "pre_test", "post_test")
SFREQ, STEP = 250, 0.1
COMPONENTS = (
    ("full_0p5", "full", 0.5), ("full_1p0", "full", 1.0),
    ("mu_0p5", "mu", 0.5), ("mu_1p0", "mu", 1.0),
    ("beta_0p5", "beta", 0.5), ("beta_1p0", "beta", 1.0),
)
ARMS = {
    "base_full_1p0": ("full_1p0",),
    "multiscale_full": ("full_0p5", "full_1p0"),
    "filterbank_1p0": ("full_1p0", "mu_1p0", "beta_1p0"),
    "multiscale_filterbank": tuple(x[0] for x in COMPONENTS),
}


def band_signals(signal):
    return {
        "full": np.asarray(signal, dtype=np.float64),
        "mu": _causal_band_signal(signal, SFREQ, 8.0, 13.0, 4),
        "beta": _causal_band_signal(signal, SFREQ, 13.0, 30.0, 4),
    }


def common_ends(size):
    return np.arange(3 * SFREQ, size + 1, int(STEP * SFREQ), dtype=np.int64)


def component_covariances(signal, ends):
    bands = band_signals(signal)
    return {
        name: window_covariances(
            bands[band], ends, int(round(window * SFREQ))
        )
        for name, band, window in COMPONENTS
    }


def references(rest_cov, split):
    result = {}
    for name, values in rest_cov.items():
        matrix = np.mean(values[:split], axis=0)
        result[name] = matrix + np.eye(len(matrix)) * max(
            float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12
        )
    return result


def features(covariances, refs, arm):
    return np.concatenate([
        tangent_features(covariances[name], refs[name]) for name in ARMS[arm]
    ], axis=1)


def interval_mask(ends, intervals, trials, low, high):
    times, starts = ends / SFREQ, (ends - SFREQ) / SFREQ
    mask = np.zeros(len(ends), dtype=bool)
    owners = np.full(len(ends), -1, dtype=np.int16)
    for (start, stop), trial in zip(intervals, trials):
        if low <= trial < high:
            selected = (starts >= start - 1e-9) & (times <= stop + 1e-9)
            mask |= selected
            owners[selected] = trial
    return mask, owners


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
    return {"transferred": at(transferred.threshold), "oracle": at(oracle.threshold)}


def analyze(mil_path, rest_path, subject, session, shifts):
    with np.load(mil_path, allow_pickle=False) as data, np.load(rest_path, allow_pickle=False) as rest:
        signal, ends = data["signal"], common_ends(len(data["signal"]))
        signal_cov = component_covariances(signal, ends)
        rest_cov, rest_ends = {}, {}
        for rest_name in RESTS:
            e = common_ends(len(rest[rest_name]))
            rest_ends[rest_name] = e
            rest_cov[rest_name] = component_covariances(rest[rest_name], e)
        split = len(rest_ends["pre_train"]) // 2
        refs = references(rest_cov["pre_train"], split)
        command_fit, command_owner = interval_mask(
            ends, data["command_intervals"], data["command_trials"], 0, 50
        )
        iti_fit, _ = interval_mask(
            ends, data["iti_intervals"], data["iti_trials"], 0, 50
        )
        command_eval, command_eval_owner = interval_mask(
            ends, data["command_intervals"], data["command_trials"], 50, 100
        )
        raw = {}
        for arm in ARMS:
            x = features(signal_cov, refs, arm)
            rest_x = {
                name: features(rest_cov[name], refs, arm) for name in RESTS
            }
            scaler = StandardScaler().fit(np.r_[x[command_fit], x[iti_fit]])
            model = LogisticRegression(
                class_weight="balanced", solver="liblinear", max_iter=3000,
                random_state=20260729,
            ).fit(
                scaler.transform(np.r_[x[command_fit], x[iti_fit]]),
                np.r_[np.ones(command_fit.sum()), np.zeros(iti_fit.sum())],
            )
            rest_scores = {
                name: model.decision_function(scaler.transform(value))
                for name, value in rest_x.items()
            }
            calibration = [value[:len(value)//2] for value in rest_scores.values()]
            evaluation_values = [value[len(value)//2:] for value in rest_scores.values()]
            command_scores = model.decision_function(scaler.transform(x))
            trials = [
                command_scores[command_eval & (command_eval_owner == trial)]
                for trial in range(50, 100)
            ]
            trials = [value for value in trials if len(value)]
            stream, audit = build_rest_injected_stream(
                evaluation_values, trials, step_seconds=STEP,
                minimum_rest_seconds=5, maximum_rest_seconds=11,
                guard_seconds=1, seed=20260728 + 100 * session + subject,
            )
            rng = np.random.default_rng(97000 + 100 * session + subject)
            offsets = rng.integers(1, len(stream.times), size=shifts)
            raw[arm] = evaluate(stream, calibration, offsets)
        return {
            "subject": subject, "session": session, "audit": audit.__dict__,
            "arms": raw,
        }


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mil-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--rest-root", type=Path, default=Path("data/interim/lee2019_four_rest_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_rest_injected_multiscale.json"))
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.mil_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz",
                args.rest_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session, args.shifts,
            ))
            print(f"multiscale session={session} subject={subject}", flush=True)
    summary = []
    for session in args.sessions:
        selected = [x for x in rows if x["session"] == session]
        base = np.asarray([x["arms"]["base_full_1p0"]["oracle"]["excess"] for x in selected])
        arms = {}
        for arm in ARMS:
            excess = np.asarray([x["arms"][arm]["oracle"]["excess"] for x in selected])
            delta = excess - base
            arms[arm] = {
                "transferred_fp_median": float(np.median([
                    x["arms"][arm]["transferred"]["fp"] for x in selected
                ])),
                "oracle_fp_median": float(np.median([
                    x["arms"][arm]["oracle"]["fp"] for x in selected
                ])),
                "excess_median": float(np.median(excess)),
                "excess_ci": bootstrap(excess, 100 + session, args.bootstrap),
                "minus_base_median": float(np.median(delta)),
                "minus_base_ci": bootstrap(delta, 200 + session, args.bootstrap),
            }
        summary.append({"session": session, "subjects": len(selected), "arms": arms})
    payload = {
        "protocol": {
            "stream": "four-rest-injected",
            "bands": {"full": "cached 8-30", "mu": "causal 8-13", "beta": "causal 13-30"},
            "windows_seconds": [0.5, 1.0],
            "primary": "multiscale_filterbank",
            "comparator": "base_full_1p0",
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
