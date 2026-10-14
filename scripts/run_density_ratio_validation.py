"""Locked tangent-space QDA/GMM log-density-ratio validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.density_ratio import DiagonalGMMRatio, ShrinkageGaussianRatio
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.riemannian import tangent_features, window_covariances

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
SFREQ, WINDOW, STEP = 250, 250, 0.5


def positives(data):
    output = []
    for trial in range(50):
        size = int(data["positive_bag_sizes"][trial])
        output.extend(data["positive_end_samples"][trial, :size])
    return np.asarray(output, dtype=np.int64)


def evaluate(times, scores, idle, command, seed, shifts):
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
    null = [
        evaluate_alarms(times, np.roll(decision.alarms, int(offset)), idle, command).true_positive_rate
        for offset in rng.integers(1, len(decision.alarms), shifts)
    ]
    return {
        "fp_per_minute": observed.fp_per_minute,
        "observed_tpr": observed.true_positive_rate,
        "null_tpr": float(np.median(null)),
        "excess_tpr": float(observed.true_positive_rate - np.median(null)),
        "latency": observed.median_delay_seconds,
    }


def analyze(path, session, subject, shifts):
    with np.load(path, allow_pickle=False) as data:
        pre = data["pre_rest"]
        pre_ends = np.arange(WINDOW, len(pre) + 1, int(STEP * SFREQ))
        reference = window_covariances(pre, pre_ends, WINDOW).mean(axis=0)
        reference += np.eye(len(reference)) * np.trace(reference) / len(reference) * 1e-6
        pos = tangent_features(window_covariances(data["signal"], positives(data), WINDOW), reference)
        idle_mask = data["iti_instance_trials"] < 50
        neg_ends = data["iti_end_samples"][idle_mask]
        neg = tangent_features(window_covariances(data["signal"], neg_ends, WINDOW), reference)
        scaler = StandardScaler().fit(np.r_[pos, neg])
        pos, neg = scaler.transform(pos), scaler.transform(neg)
        ends = np.arange(3 * SFREQ, len(data["signal"]) + 1, int(STEP * SFREQ))
        times = ends / SFREQ
        stream = scaler.transform(
            tangent_features(window_covariances(data["signal"], ends, WINDOW), reference)
        )
        idle = data["iti_intervals"][(data["iti_trials"] >= 50) & (data["iti_trials"] < 100)]
        command = data["command_intervals"][50:100]
    models = {
        "shrinkage_qda": ShrinkageGaussianRatio().fit(pos, neg),
        "gmm2_diag": DiagonalGMMRatio(SEEDS[0]).fit(pos, neg),
    }
    return {
        "session": session, "subject": subject,
        "models": {
            name: evaluate(times, model.decision_function(stream), idle, command,
                           SEEDS[0] + session * 1000 + subject + index * 10, shifts)
            for index, (name, model) in enumerate(models.items())
        },
    }


def bootstrap(values, seed, repeats):
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/density_ratio_validation.json"))
    args = parser.parse_args()
    if args.smoke:
        args.subjects, args.sessions, args.shifts, args.bootstrap = args.subjects[:1], [1], 20, 100
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.cache_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz",
                session, subject, args.shifts,
            ))
            print(f"done session={session} subject={subject}", flush=True)
    comparator = json.loads(Path("data/processed/mil_riemannian_comparator.json").read_text(encoding="utf-8"))
    summary = []
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        rmap = {row["subject"]: row["excess_tpr"] for row in comparator["rows"] if row["session"] == session}
        models = {}
        for name in ("shrinkage_qda", "gmm2_diag"):
            values = np.asarray([row["models"][name]["excess_tpr"] for row in selected])
            baseline = np.asarray([rmap[row["subject"]] for row in selected])
            models[name] = {
                "excess_median": float(np.median(values)),
                "excess_ci": bootstrap(values, 100 + session, args.bootstrap),
                "minus_riemannian_median": float(np.median(values - baseline)),
                "minus_riemannian_ci": bootstrap(values - baseline, 200 + session, args.bootstrap),
                "fp_median": float(np.median([row["models"][name]["fp_per_minute"] for row in selected])),
            }
        summary.append({"session": session, "subjects": len(selected), "models": models})
    payload = {"protocol": {"preregistered": True, "fp_axis": 2.0, "null_shifts": args.shifts}, "summary": summary, "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
