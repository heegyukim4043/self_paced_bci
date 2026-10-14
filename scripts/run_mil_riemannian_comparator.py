"""Recompute the deterministic 1-s Riemannian comparator from MIL raw caches."""

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

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    tune_source_operating_points,
)
from uncued_mi.mil import SEEDS
from uncued_mi.riemannian import tangent_features, window_covariances

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
SFREQ = 250
WINDOW = 250
DECISION_STEP = 0.5


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(repeats, len(values)))
    ]
    return [float(value) for value in np.quantile(np.median(samples, axis=1), [0.025, 0.975])]


def unpad_positive(data, low: int, high: int) -> np.ndarray:
    values = []
    for trial in range(low, high):
        size = int(data["positive_bag_sizes"][trial])
        values.extend(data["positive_end_samples"][trial, :size])
    return np.asarray(values, dtype=np.int64)


def legacy_positive_ends(data, low: int, high: int) -> np.ndarray:
    """Old rule: select by window end-time only, even if support starts earlier."""
    step = int(0.1 * SFREQ)
    ends = np.arange(int(3.0 * SFREQ), len(data["signal"]) + 1, step)
    selected = []
    for start, stop in data["command_intervals"][low:high]:
        selected.extend(ends[(ends / SFREQ >= start) & (ends / SFREQ < stop)])
    return np.asarray(selected, dtype=np.int64)


def analyze(
    path: Path,
    session: int,
    subject: int,
    shifts: int,
    positive_support: str,
    train_stop: int,
    eval_start: int,
    eval_stop: int,
) -> dict:
    with np.load(path, allow_pickle=False) as data:
        signal = np.asarray(data["signal"])
        pre = np.asarray(data["pre_rest"])
        train_positive = (
            unpad_positive(data, 0, train_stop)
            if positive_support == "strict"
            else legacy_positive_ends(data, 0, train_stop)
        )
        iti_trials = data["iti_instance_trials"]
        train_idle = data["iti_end_samples"][iti_trials < train_stop]
        pre_ends = np.r_[
            data["pre_rest_fit_end_samples"],
            data["pre_rest_validation_end_samples"],
        ]
        reference = window_covariances(pre, pre_ends, WINDOW).mean(axis=0)
        reference += (
            np.eye(reference.shape[0])
            * np.trace(reference)
            / reference.shape[0]
            * 1e-6
        )

        train_ends = np.r_[train_positive, train_idle]
        train_y = np.r_[
            np.ones(len(train_positive), dtype=np.int8),
            np.zeros(len(train_idle), dtype=np.int8),
        ]
        train_x = tangent_features(
            window_covariances(signal, train_ends, WINDOW), reference
        )
        scaler = StandardScaler().fit(train_x)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=SEEDS[0],
        ).fit(scaler.transform(train_x), train_y)

        first_end = int(3.0 * SFREQ)
        step = int(DECISION_STEP * SFREQ)
        decision_ends = np.arange(first_end, len(signal) + 1, step, dtype=np.int64)
        decision_x = tangent_features(
            window_covariances(signal, decision_ends, WINDOW), reference
        )
        scores = model.decision_function(scaler.transform(decision_x))
        times = decision_ends / SFREQ
        idle = data["iti_intervals"][
            (data["iti_trials"] >= eval_start) & (data["iti_trials"] < eval_stop)
        ]
        command = data["command_intervals"][eval_start:eval_stop]

    sequence = ScoredSequence(times, scores, idle, command)
    point = tune_source_operating_points(
        [sequence],
        "ma_dwell",
        [2.0],
        DECISION_STEP,
        ma_seconds=0.5,
        dwell_seconds=0.3,
        refractory_seconds=1.0,
        grid_size=160,
    )[2.0]
    result = apply_decision(
        sequence,
        "ma_dwell",
        point.threshold,
        DECISION_STEP,
        ma_seconds=0.5,
        dwell_seconds=0.3,
        refractory_seconds=1.0,
    )
    observed = evaluate_alarms(times, result.alarms, idle, command)
    rng = np.random.default_rng(SEEDS[0] + session * 100 + subject)
    null = np.empty(shifts)
    for index, offset in enumerate(rng.integers(1, len(result.alarms), size=shifts)):
        null[index] = evaluate_alarms(
            times, np.roll(result.alarms, int(offset)), idle, command
        ).true_positive_rate
    command_mask = np.zeros(len(times), dtype=bool)
    idle_mask = np.zeros(len(times), dtype=bool)
    for start, stop in command:
        command_mask |= (times >= start) & (times < stop)
    for start, stop in idle:
        idle_mask |= (times >= start) & (times < stop)
    return {
        "session": session,
        "subject": subject,
        "window_auc": auc(scores[command_mask], scores[idle_mask]),
        "threshold": point.threshold,
        "fp_per_minute": observed.fp_per_minute,
        "observed_tpr": observed.true_positive_rate,
        "null_tpr_median": float(np.median(null)),
        "excess_tpr": float(observed.true_positive_rate - np.median(null)),
        "training_positive_instances": len(train_positive),
        "training_iti_instances": len(train_idle),
        "positive_support": positive_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--positive-support", choices=["strict", "legacy_end"], default="strict"
    )
    parser.add_argument("--train-stop", type=int, default=50)
    parser.add_argument("--eval-start", type=int, default=50)
    parser.add_argument("--eval-stop", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mil_riemannian_comparator.json"),
    )
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = (
                args.cache_root
                / f"session{session}"
                / f"sub-{subject:02d}_mil_raw.npz"
            )
            rows.append(
                analyze(
                    path,
                    session,
                    subject,
                    args.shifts,
                    args.positive_support,
                    args.train_stop,
                    args.eval_start,
                    args.eval_stop,
                )
            )
            print(f"done session={session} subject={subject}", flush=True)
    summary = []
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        excess = np.asarray([row["excess_tpr"] for row in selected])
        summary.append(
            {
                "session": session,
                "subjects": len(selected),
                "window_auc_median": float(np.median([row["window_auc"] for row in selected])),
                "fp_per_minute_median": float(np.median([row["fp_per_minute"] for row in selected])),
                "observed_tpr_median": float(np.median([row["observed_tpr"] for row in selected])),
                "null_tpr_median": float(np.median([row["null_tpr_median"] for row in selected])),
                "excess_tpr_median": float(np.median(excess)),
                "excess_tpr_bootstrap_95_ci": bootstrap_ci(
                    excess, SEEDS[0] + session, args.bootstrap
                ),
            }
        )
    payload = {
        "protocol": {
            "cache": "same causal raw cache as MIL",
            "training_trials": [0, args.train_stop],
            "evaluation_trials": [args.eval_start, args.eval_stop],
            "window": "1.0 s",
            "decision_step": DECISION_STEP,
            "fp_axis": "held-out diagnostic oracle at FP=2",
            "null_shifts": args.shifts,
            "positive_support": args.positive_support,
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
