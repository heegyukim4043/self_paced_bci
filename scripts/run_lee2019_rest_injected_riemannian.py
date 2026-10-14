"""Riemannian baseline on a four-rest injected pseudo-online stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    tune_source_operating_points,
)
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features

RESTS = ("pre_train", "post_train", "pre_test", "post_test")
STEP = 0.1


def reference(covariances: np.ndarray) -> np.ndarray:
    matrix = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12)
    return matrix + np.eye(len(matrix)) * ridge


def calibration_sequence(scores: np.ndarray) -> ScoredSequence:
    times = np.arange(len(scores), dtype=float) * STEP
    return ScoredSequence(
        times, scores, np.asarray([[0.0, len(scores) * STEP]]),
        np.empty((0, 2), dtype=float),
    )


def command_trials(
    scores: np.ndarray, trials: np.ndarray, relative_end: np.ndarray
) -> list[np.ndarray]:
    output = []
    for trial in np.unique(trials):
        mask = (trials == trial) & (relative_end >= 0.0) & (relative_end <= 3.0)
        values = scores[mask]
        if len(values):
            output.append(values)
    return output


def analyze(path: Path, subject: int, session: int, shifts: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        train_trial = data["train_command_trial"]
        train_mask = train_trial < 50
        iti_trial = data["train_idle_iti_trial"]
        iti_mask = iti_trial < 50
        command_x = tangent_features(
            data["train_command_covariance"][train_mask], ref
        )
        iti_x = tangent_features(data["train_idle_iti_covariance"][iti_mask], ref)
        scaler = StandardScaler().fit(np.r_[command_x, iti_x])
        model = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=2000,
            random_state=20260728,
        ).fit(
            scaler.transform(np.r_[command_x, iti_x]),
            np.r_[np.ones(len(command_x)), np.zeros(len(iti_x))],
        )
        rest_scores = {}
        calibration = []
        evaluation = []
        for name in RESTS:
            scores = model.decision_function(
                scaler.transform(tangent_features(data[f"{name}_covariance"], ref))
            )
            split = len(scores) // 2
            rest_scores[name] = scores
            calibration.append(scores[:split])
            evaluation.append(scores[split:])
        all_command_scores = model.decision_function(
            scaler.transform(
                tangent_features(data["train_command_covariance"], ref)
            )
        )
        eval_mask = train_trial >= 50
        trials = command_trials(
            all_command_scores[eval_mask],
            train_trial[eval_mask],
            data["train_command_relative_end"][eval_mask],
        )

    point = tune_source_operating_points(
        [calibration_sequence(values) for values in calibration],
        "ma_dwell", [1.0, 2.0, 4.0], STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        grid_size=160,
    )
    stream, audit = build_rest_injected_stream(
        evaluation, trials, step_seconds=STEP,
        minimum_rest_seconds=5.0, maximum_rest_seconds=11.0,
        guard_seconds=1.0, seed=20260728 + 100 * session + subject,
    )
    idle_mask = np.zeros(len(stream.times), dtype=bool)
    command_mask = np.zeros(len(stream.times), dtype=bool)
    for start, stop in stream.idle_intervals:
        idle_mask |= (stream.times >= start) & (stream.times < stop)
    for start, stop in stream.command_intervals:
        command_mask |= (stream.times >= start) & (stream.times < stop)
    row = {
        "subject": subject,
        "session": session,
        "window_auc": float(roc_auc_score(
            np.r_[np.zeros(idle_mask.sum()), np.ones(command_mask.sum())],
            np.r_[stream.scores[idle_mask], stream.scores[command_mask]],
        )),
        "audit": audit.__dict__,
        "operating_points": {},
    }
    rng = np.random.default_rng(91000 + 100 * session + subject)
    for target, operating_point in point.items():
        result = apply_decision(
            stream, "ma_dwell", operating_point.threshold, STEP,
            ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        )
        observed = evaluate_alarms(
            stream.times, result.alarms, stream.idle_intervals,
            stream.command_intervals,
        )
        null = np.empty(shifts)
        for index, offset in enumerate(
            rng.integers(1, len(result.alarms), size=shifts)
        ):
            null[index] = evaluate_alarms(
                stream.times, np.roll(result.alarms, int(offset)),
                stream.idle_intervals, stream.command_intervals,
            ).true_positive_rate
        row["operating_points"][str(target)] = {
            "calibration_fp_per_minute": operating_point.source_fp_per_minute,
            "evaluation_fp_per_minute": observed.fp_per_minute,
            "observed_tpr": observed.true_positive_rate,
            "null_tpr_median": float(np.median(null)),
            "excess_tpr": float(observed.true_positive_rate - np.median(null)),
            "median_delay_seconds": observed.median_delay_seconds,
        }
    return row


def bootstrap(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("data/interim/lee2019_four_rest"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_rest_injected_riemannian.json"),
    )
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session, args.shifts,
            ))
            print(f"done session={session} subject={subject}", flush=True)
    summary = []
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        points = {}
        for target in ("1.0", "2.0", "4.0"):
            excess = np.asarray([
                row["operating_points"][target]["excess_tpr"] for row in selected
            ])
            points[target] = {
                "evaluation_fp_per_minute_median": float(np.median([
                    row["operating_points"][target]["evaluation_fp_per_minute"]
                    for row in selected
                ])),
                "observed_tpr_median": float(np.median([
                    row["operating_points"][target]["observed_tpr"]
                    for row in selected
                ])),
                "null_tpr_median": float(np.median([
                    row["operating_points"][target]["null_tpr_median"]
                    for row in selected
                ])),
                "excess_tpr_median": float(np.median(excess)),
                "excess_tpr_ci": bootstrap(
                    excess, 20260728 + session + int(float(target) * 10),
                    args.bootstrap,
                ),
            }
        summary.append({
            "session": session,
            "subjects": len(selected),
            "window_auc_median": float(np.median([
                row["window_auc"] for row in selected
            ])),
            "command_events_median": float(np.median([
                row["audit"]["command_events"] for row in selected
            ])),
            "idle_seconds_median": float(np.median([
                row["audit"]["rest_windows"] * STEP for row in selected
            ])),
            "operating_points": points,
        })
    payload = {
        "protocol": {
            "name": "four-rest-injected pseudo-online simulation",
            "rest_split": "first half calibration, second half evaluation",
            "rest_chunks": "non-overlapping random 5-11 s chunks",
            "command": "last 50 train trials, strict cached command support",
            "guard": "1 s reset, excluded from FP and TPR",
            "decision": "1 s Riemannian TS+LR, MA0.5+dwell0.3+refractory1.0",
            "scope": "method-comparison simulation, not self-initiated deployment",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
