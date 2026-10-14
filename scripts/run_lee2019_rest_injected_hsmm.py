"""Select and evaluate a causal HSMM on rest-injected Lee2019 streams."""

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
from uncued_mi.hsmm import GaussianEmission, causal_two_state_hsmm
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    tune_source_operating_points,
)
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features

RESTS = ("pre_train", "post_train", "pre_test", "post_test")
SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
STEP = 0.1
ENTRY_PROBABILITY = 0.01
CANDIDATES = (
    (0.3, 0.5, 1.5),
    (0.3, 1.0, 1.5),
    (0.5, 0.5, 1.5),
    (0.5, 1.0, 1.5),
    (0.5, 1.5, 1.5),
    (1.0, 1.0, 2.0),
    (1.0, 1.5, 2.0),
)


def reference(covariances: np.ndarray) -> np.ndarray:
    matrix = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12)
    return matrix + np.eye(len(matrix)) * ridge


def fit_score_model(data, ref: np.ndarray, train_stop: int):
    command_trial = data["train_command_trial"]
    idle_trial = data["train_idle_iti_trial"]
    command_mask = command_trial < train_stop
    idle_mask = idle_trial < train_stop
    command_x = tangent_features(data["train_command_covariance"][command_mask], ref)
    idle_x = tangent_features(data["train_idle_iti_covariance"][idle_mask], ref)
    scaler = StandardScaler().fit(np.r_[command_x, idle_x])
    model = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260729,
    ).fit(
        scaler.transform(np.r_[command_x, idle_x]),
        np.r_[np.ones(len(command_x)), np.zeros(len(idle_x))],
    )
    return model, scaler, command_x, idle_x


def grouped_trials(scores, trials, relative, low: int, high: int) -> list[np.ndarray]:
    output = []
    for trial in range(low, high):
        mask = (trials == trial) & (relative >= 0.0) & (relative <= 3.0)
        if mask.any():
            output.append(scores[mask])
    return output


def statistic(scores: np.ndarray, emission: GaussianEmission, candidate) -> np.ndarray:
    minimum, expected, maximum = candidate
    return causal_two_state_hsmm(
        scores, emission,
        minimum_mi_steps=int(round(minimum / STEP)),
        expected_mi_steps=expected / STEP,
        maximum_mi_steps=int(round(maximum / STEP)),
        entry_probability=ENTRY_PROBABILITY,
    )


def transform_sequence(
    sequence: ScoredSequence, emission: GaussianEmission, candidate
) -> ScoredSequence:
    return ScoredSequence(
        sequence.times,
        statistic(sequence.scores, emission, candidate),
        sequence.idle_intervals,
        sequence.command_intervals,
    )


def evaluate_candidate(
    path: Path,
    subject: int,
    session: int,
    candidate,
    mode: str,
    shifts: int,
) -> dict:
    if mode == "validation":
        train_stop, eval_low, eval_high = 40, 40, 50
        calibration_slice = slice(0, None)
        evaluation_slice = slice(None, None)
    else:
        train_stop, eval_low, eval_high = 50, 50, 100
        calibration_slice = slice(0, None)
        evaluation_slice = slice(None, None)
    with np.load(path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        model, scaler, command_x, idle_x = fit_score_model(data, ref, train_stop)
        train_command_score = model.decision_function(scaler.transform(command_x))
        train_idle_score = model.decision_function(scaler.transform(idle_x))
        emission = GaussianEmission.fit(train_idle_score, train_command_score)
        calibration, evaluation = [], []
        for name in RESTS:
            values = model.decision_function(scaler.transform(
                tangent_features(data[f"{name}_covariance"], ref)
            ))
            if mode == "validation":
                quarter = len(values) // 4
                calibration.append(values[:quarter][calibration_slice])
                evaluation.append(values[quarter:2 * quarter][evaluation_slice])
            else:
                half = len(values) // 2
                calibration.append(values[:half][calibration_slice])
                evaluation.append(values[half:][evaluation_slice])
        all_command_score = model.decision_function(scaler.transform(
            tangent_features(data["train_command_covariance"], ref)
        ))
        trials = grouped_trials(
            all_command_score, data["train_command_trial"],
            data["train_command_relative_end"], eval_low, eval_high,
        )
    stream_seed = (
        20260728 + 100 * session + subject
        if mode == "locked"
        else 20260729 + 1000 * session + subject + 50000
    )
    stream, audit = build_rest_injected_stream(
        evaluation, trials, step_seconds=STEP,
        minimum_rest_seconds=5.0, maximum_rest_seconds=11.0,
        guard_seconds=1.0,
        seed=stream_seed,
    )
    calibration_raw = [
        ScoredSequence(
            np.arange(len(x)) * STEP, x,
            np.asarray([[0.0, len(x) * STEP]]), np.empty((0, 2)),
        )
        for x in calibration
    ]
    calibration_hsmm = [
        transform_sequence(x, emission, candidate) for x in calibration_raw
    ]
    hsmm_stream = transform_sequence(stream, emission, candidate)
    hsmm_point = tune_source_operating_points(
        calibration_hsmm, "threshold", [2.0], STEP,
        refractory_seconds=1.0, grid_size=160,
    )[2.0]
    ma_point = tune_source_operating_points(
        calibration_raw, "ma_dwell", [2.0], STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        grid_size=160,
    )[2.0]
    hsmm_result = instantaneous(
        hsmm_stream.scores, hsmm_point.threshold, STEP, 1.0
    )
    ma_result = apply_decision(
        stream, "ma_dwell", ma_point.threshold, STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
    )
    hsmm_observed = evaluate_alarms(
        stream.times, hsmm_result.alarms, stream.idle_intervals,
        stream.command_intervals,
    )
    ma_observed = evaluate_alarms(
        stream.times, ma_result.alarms, stream.idle_intervals,
        stream.command_intervals,
    )
    rng = np.random.default_rng(93000 + 1000 * session + subject)
    offsets = rng.integers(1, len(stream.times), size=shifts)
    hsmm_null, ma_null = [], []
    for offset in offsets:
        hsmm_null.append(evaluate_alarms(
            stream.times, np.roll(hsmm_result.alarms, int(offset)),
            stream.idle_intervals, stream.command_intervals,
        ).true_positive_rate)
        ma_null.append(evaluate_alarms(
            stream.times, np.roll(ma_result.alarms, int(offset)),
            stream.idle_intervals, stream.command_intervals,
        ).true_positive_rate)
    hsmm_excess = hsmm_observed.true_positive_rate - float(np.median(hsmm_null))
    ma_excess = ma_observed.true_positive_rate - float(np.median(ma_null))
    return {
        "subject": subject, "session": session, "mode": mode,
        "candidate": list(candidate), "audit": audit.__dict__,
        "hsmm": {
            "calibration_fp": hsmm_point.source_fp_per_minute,
            "fp_per_minute": hsmm_observed.fp_per_minute,
            "observed_tpr": hsmm_observed.true_positive_rate,
            "null_tpr": float(np.median(hsmm_null)),
            "excess_tpr": hsmm_excess,
            "latency": hsmm_observed.median_delay_seconds,
        },
        "ma_dwell": {
            "calibration_fp": ma_point.source_fp_per_minute,
            "fp_per_minute": ma_observed.fp_per_minute,
            "observed_tpr": ma_observed.true_positive_rate,
            "null_tpr": float(np.median(ma_null)),
            "excess_tpr": ma_excess,
            "latency": ma_observed.median_delay_seconds,
        },
        "paired_delta": hsmm_excess - ma_excess,
    }


def bootstrap(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows: list[dict], seed: int, repeats: int) -> dict:
    hsmm = np.asarray([x["hsmm"]["excess_tpr"] for x in rows])
    ma = np.asarray([x["ma_dwell"]["excess_tpr"] for x in rows])
    delta = hsmm - ma
    return {
        "subjects": len(rows),
        "hsmm_calibration_fp_median": float(np.median([x["hsmm"]["calibration_fp"] for x in rows])),
        "hsmm_fp_median": float(np.median([x["hsmm"]["fp_per_minute"] for x in rows])),
        "hsmm_observed_tpr_median": float(np.median([x["hsmm"]["observed_tpr"] for x in rows])),
        "hsmm_null_tpr_median": float(np.median([x["hsmm"]["null_tpr"] for x in rows])),
        "hsmm_excess_median": float(np.median(hsmm)),
        "hsmm_excess_ci": bootstrap(hsmm, seed, repeats),
        "ma_excess_median": float(np.median(ma)),
        "paired_delta_median": float(np.median(delta)),
        "paired_delta_ci": bootstrap(delta, seed + 10000, repeats),
        "hsmm_latency_median": float(np.nanmedian([x["hsmm"]["latency"] for x in rows])),
        "ma_latency_median": float(np.nanmedian([x["ma_dwell"]["latency"] for x in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_four_rest"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--selection-subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_rest_injected_hsmm.json"))
    args = parser.parse_args()
    validation = []
    for candidate in CANDIDATES:
        rows = [
            evaluate_candidate(
                args.cache_root / "session1" / f"sub-{subject:02d}.npz",
                subject, 1, candidate, "validation", args.shifts,
            )
            for subject in args.selection_subjects
        ]
        summary = summarize(rows, 1000 + len(validation), args.bootstrap)
        validation.append({"candidate": list(candidate), "summary": summary, "rows": rows})
        print(f"validated candidate={candidate}", flush=True)
    admissible = [
        x for x in validation
        if 1.5 <= x["summary"]["hsmm_fp_median"] <= 2.5
    ]
    if admissible:
        selected = sorted(
            admissible,
            key=lambda x: (
                -x["summary"]["hsmm_excess_median"],
                x["candidate"][0], x["candidate"][1], x["candidate"][2],
            ),
        )[0]
        calibration_failure = False
    else:
        selected = min(
            validation,
            key=lambda x: abs(x["summary"]["hsmm_fp_median"] - 2.0),
        )
        calibration_failure = True
    chosen = tuple(selected["candidate"])
    locked = []
    for session in (1, 2):
        rows = []
        for subject in args.subjects:
            rows.append(evaluate_candidate(
                args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session, chosen, "locked", args.shifts,
            ))
            print(f"locked session={session} subject={subject}", flush=True)
        locked.append({
            "session": session,
            "summary": summarize(rows, 2000 + session, args.bootstrap),
            "rows": rows,
        })
    payload = {
        "protocol": {
            "stream": "four-rest-injected pseudo-online",
            "selection": "session1 subset20, rest quarter2 + trials40-49",
            "locked": "54 subjects x sessions1-2, rest half2 + trials50-99",
            "entry_probability": ENTRY_PROBABILITY,
            "candidate_grid": [list(x) for x in CANDIDATES],
            "comparator": "same-score same-stream Riemannian MA+dwell",
            "paired_null": "identical circular offsets",
        },
        "validation": validation,
        "selection_audit": {
            "admissible_count": len(admissible),
            "calibration_failure": calibration_failure,
        },
        "selected_candidate": list(chosen),
        "locked": locked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection_audit": payload["selection_audit"],
        "selected_candidate": payload["selected_candidate"],
        "locked": [{"session": x["session"], **x["summary"]} for x in locked],
    }, indent=2))


if __name__ == "__main__":
    main()
