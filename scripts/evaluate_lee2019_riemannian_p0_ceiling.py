"""P0 diagnostic ceiling for the Lee2019 Riemannian representation.

This is deliberately non-deployable.  It cross-fits a target-specific
Riemannian logistic-regression model inside the locked last-40-trial stream:
five contiguous eight-trial folds are scored only by a model that did not see
those trials.  Detector state is reset at fold boundaries and the threshold is
chosen from evaluation idle to match achieved FP/min.  The result is therefore
an optimistic score-information ceiling, not prospective performance.

The comparator uses the same fold-reset/evaluation-idle oracle but trains the
model on the original first 50 labelled trials.  This keeps the evaluation and
decision layer matched while isolating the added value of within-evaluation
cross-fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import evaluate_lee2019_loso_deployment as base
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma,
    ewma_z,
    interval_metrics,
    intervals_from_alarm,
    ma_trace,
)
from uncued_mi.riemannian import tangent_features


SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20,
            23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
ALL_SUBJECTS = tuple(range(1, 55))
HOLDOUT34 = tuple(subject for subject in ALL_SUBJECTS if subject not in SUBSET20)
TARGET_FP = (2.0, 4.0)
DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)
FOLDS = 5
SCREEN = 10
MA_WINDOWS = 5
DWELL_WINDOWS = 3
REFRACTORY_WINDOWS = 10
MAX_TRAIN_PER_CLASS = 800


def evenly(indices: np.ndarray, limit: int = MAX_TRAIN_PER_CLASS) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return indices[positions]


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if not len(positive) or not len(negative):
        return float("nan")
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n_positive = len(positive)
    return float(
        (ranks[:n_positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * len(negative))
    )


def fit_model(command: np.ndarray, idle: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    x = np.r_[command, idle]
    y = np.r_[np.ones(len(command)), np.zeros(len(idle))]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=20260924,
    ).fit(scaler.transform(x), y)
    return scaler, model


def interval_mask(
    times: np.ndarray,
    starts: np.ndarray,
    events: np.ndarray,
    event_indices: np.ndarray,
    start_col: int,
    stop_col: int,
) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for index in event_indices:
        start = events[index, start_col] / base.SFREQ
        stop = events[index, stop_col] / base.SFREQ
        mask |= (starts >= start) & (times <= stop)
    return mask


def cell_mask(
    times: np.ndarray,
    events: np.ndarray,
    event_indices: np.ndarray,
    stream_stop: float,
) -> np.ndarray:
    first = int(event_indices[0])
    last = int(event_indices[-1])
    start = events[first, 2] / base.SFREQ
    stop = events[last + 1, 2] / base.SFREQ if last + 1 < len(events) else stream_stop
    return (times >= start) & (times < stop)


def make_fold_records(
    stream_features: np.ndarray,
    anchor_features: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    starts: np.ndarray,
    allowed: np.ndarray,
    arm: str,
    first50_model: tuple[StandardScaler, LogisticRegression] | None,
) -> list[dict[str, object]]:
    evaluation_events = np.arange(SCREEN, len(events), dtype=int)
    folds = [np.asarray(values, dtype=int) for values in np.array_split(evaluation_events, FOLDS)]
    if any(len(values) != 8 for values in folds):
        raise ValueError("P0 expects exactly five folds of eight evaluation trials")
    records: list[dict[str, object]] = []
    stream_stop = float(times[-1] + base.STEP)

    for fold_index, heldout in enumerate(folds):
        if arm == "eval_oof_oracle":
            training = np.concatenate([values for i, values in enumerate(folds) if i != fold_index])
            command_mask = interval_mask(times, starts, events, training, 6, 7) & allowed
            idle_mask = interval_mask(times, starts, events, training, 2, 3) & allowed
            command_indices = evenly(np.flatnonzero(command_mask))
            idle_indices = evenly(np.flatnonzero(idle_mask))
            scaler, model = fit_model(
                stream_features[command_indices], stream_features[idle_indices]
            )
        elif arm == "first50_supervised":
            if first50_model is None:
                raise ValueError("first50 model is required")
            scaler, model = first50_model
            training = np.arange(0, 50, dtype=int)
            command_indices = np.empty(0, dtype=int)
            idle_indices = np.empty(0, dtype=int)
        else:
            raise ValueError(arm)

        fold_mask = cell_mask(times, events, heldout, stream_stop)
        fold_indices = np.flatnonzero(fold_mask)
        fold_times = times[fold_indices]
        fold_starts = starts[fold_indices]
        fold_allowed = allowed[fold_indices]

        anchor_score = model.decision_function(scaler.transform(anchor_features))
        center = float(np.mean(anchor_score))
        variance = float(np.var(anchor_score, ddof=1))
        raw = model.decision_function(scaler.transform(stream_features[fold_indices]))
        z = ewma_z(raw, center, variance, fold_allowed)
        ma = ma_trace(z, n=MA_WINDOWS)

        cues = events[heldout, 5] / base.SFREQ
        onsets = cues + 0.5
        stops = cues + 4.0
        gt = np.c_[onsets, stops]
        state = np.zeros(len(fold_times), dtype=bool)
        for onset, stop in gt:
            state |= (fold_times >= onset) & (fold_times <= stop)
        idle = fold_allowed & ~state
        positive = interval_mask(
            fold_times, fold_starts, events, heldout, 6, 7
        ) & fold_allowed
        records.append(
            {
                "fold": fold_index,
                "heldout_events": heldout.tolist(),
                "training_events": training.tolist(),
                "training_command_windows": int(len(command_indices)) if arm == "eval_oof_oracle" else None,
                "training_idle_windows": int(len(idle_indices)) if arm == "eval_oof_oracle" else None,
                "times": fold_times,
                "starts": fold_starts,
                "allowed": fold_allowed,
                "idle": idle,
                "positive": positive,
                # Pooling raw LR margins across cross-fit folds is not valid
                # because each fitted head can have a different affine scale.
                # The detector consumes the causal EWMA-normalized score, so
                # window AUC must use that same comparable score axis.
                "score_for_auc": z,
                "z": z,
                "ma": ma,
                "gt": gt,
                "onsets": onsets,
            }
        )
    return records


def validate_records(records: list[dict[str, object]], arm: str) -> None:
    """Fail closed if the cross-fit partition or fold arrays are malformed."""
    heldout_all: list[int] = []
    for record in records:
        heldout = [int(value) for value in record["heldout_events"]]
        training = [int(value) for value in record["training_events"]]
        if len(heldout) != 8 or len(np.asarray(record["gt"])) != 8:
            raise AssertionError("each P0 fold must contain exactly eight held-out events")
        if arm == "eval_oof_oracle" and set(heldout) & set(training):
            raise AssertionError("held-out event leaked into its OOF training fold")
        lengths = {
            len(np.asarray(record[key]))
            for key in ("times", "starts", "allowed", "idle", "positive", "score_for_auc", "z", "ma")
        }
        if len(lengths) != 1:
            raise AssertionError("fold-aligned arrays have inconsistent lengths")
        heldout_all.extend(heldout)
    if sorted(heldout_all) != list(range(SCREEN, 50)) or len(set(heldout_all)) != 40:
        raise AssertionError("locked evaluation events must be held out exactly once")


def alarms_for(records: list[dict[str, object]], threshold: float) -> list[np.ndarray]:
    return [
        alarms_from_ma(
            np.asarray(record["ma"]),
            threshold,
            np.asarray(record["allowed"]),
            dwell=DWELL_WINDOWS,
            refractory=REFRACTORY_WINDOWS,
        )
        for record in records
    ]


def fp_rate(records: list[dict[str, object]], alarms: list[np.ndarray]) -> float:
    count = sum(int(np.sum(candidate & np.asarray(record["idle"]))) for record, candidate in zip(records, alarms))
    exposure = sum(float(np.sum(record["idle"]) * base.STEP) for record in records)
    return float(count / (exposure / 60.0))


def sensitivity(
    records: list[dict[str, object]], alarms: list[np.ndarray], delta: float
) -> float:
    detected: list[bool] = []
    for record, candidate in zip(records, alarms):
        times = np.asarray(record["times"])
        for onset in np.asarray(record["onsets"]):
            detected.append(bool(np.any(candidate & (times >= onset) & (times <= onset + delta))))
    return float(np.mean(detected))


def choose_threshold(records: list[dict[str, object]], target_fp: float) -> tuple[float, list[np.ndarray], float]:
    values = np.concatenate([
        np.asarray(record["ma"])[np.asarray(record["allowed"]) & np.isfinite(record["ma"])]
        for record in records
    ])
    candidates = np.unique(np.r_[
        np.quantile(values, np.linspace(0.0, 1.0, 321)),
        values.max() + max(1e-8, np.std(values) * 1e-6),
    ])
    best = None
    for threshold in candidates:
        alarms = alarms_for(records, float(threshold))
        achieved = fp_rate(records, alarms)
        key = (abs(achieved - target_fp), -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), alarms, achieved)
    assert best is not None
    return best[1], best[2], best[3]


def null_distribution(
    records: list[dict[str, object]],
    alarms: list[np.ndarray],
    delta: float,
    shifts: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(shifts, dtype=float)
    for draw in range(shifts):
        shifted = []
        for candidate in alarms:
            offset = int(rng.integers(1, max(2, len(candidate))))
            shifted.append(np.roll(candidate, offset))
        result[draw] = sensitivity(records, shifted, delta)
    return result


def interval_summary(records: list[dict[str, object]], alarms: list[np.ndarray], threshold: float) -> dict[str, float]:
    predicted = []
    ground_truth = []
    for record, candidate in zip(records, alarms):
        pred = intervals_from_alarm(
            np.asarray(record["times"]),
            candidate,
            np.asarray(record["ma"]),
            threshold,
            np.asarray(record["allowed"]),
        )
        if len(pred):
            predicted.append(pred)
        ground_truth.append(np.asarray(record["gt"]))
    pred_all = np.concatenate(predicted) if predicted else np.empty((0, 2), dtype=float)
    gt_all = np.concatenate(ground_truth)
    tolerant = interval_metrics(pred_all, gt_all, 0.1, 0.1)
    strict = interval_metrics(pred_all, gt_all, 0.7, 0.7)
    return {
        "coverage": float(tolerant["coverage"]),
        "temporal_iou": float(tolerant["temporal_iou"]),
        "strict_f1": float(strict["intersection_f1"]),
        "predicted_intervals": int(len(pred_all)),
    }


def evaluate_arm(
    records: list[dict[str, object]], shifts: int, seed: int
) -> dict[str, object]:
    positive = np.concatenate([
        np.asarray(r["score_for_auc"])[np.asarray(r["positive"])] for r in records
    ])
    negative = np.concatenate([
        np.asarray(r["score_for_auc"])[np.asarray(r["idle"])] for r in records
    ])
    output: dict[str, object] = {
        "window_auc": binary_auc(positive, negative),
        "positive_windows": int(len(positive)),
        "idle_windows": int(len(negative)),
        "folds": [
            {
                key: record[key]
                for key in (
                    "fold", "heldout_events", "training_events",
                    "training_command_windows", "training_idle_windows",
                )
            }
            for record in records
        ],
        "targets": {},
    }
    for target_fp in TARGET_FP:
        threshold, alarms, achieved = choose_threshold(records, target_fp)
        sensitivities = {str(delta): sensitivity(records, alarms, delta) for delta in DELTAS}
        nulls = {
            str(delta): null_distribution(
                records, alarms, delta, shifts,
                seed + int(target_fp * 1000) + int(delta * 100),
            )
            for delta in DELTAS
        }
        output["targets"][str(target_fp)] = {
            "threshold": threshold,
            "actual_fp_per_min": achieved,
            "false_alarm_count": int(sum(np.sum(a & np.asarray(r["idle"])) for r, a in zip(records, alarms))),
            "idle_exposure_seconds": float(sum(np.sum(r["idle"]) * base.STEP for r in records)),
            "onset_sensitivity": sensitivities,
            "circular_shift_null_median": {key: float(np.median(value)) for key, value in nulls.items()},
            "circular_shift_excess": {
                key: float(sensitivities[key] - np.median(nulls[key])) for key in sensitivities
            },
            **interval_summary(records, alarms, threshold),
        }
    return output


def analyze(subject: int, session: int, shifts: int) -> dict[str, object]:
    covariance_path = Path(f"data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz")
    with np.load(covariance_path, allow_pickle=False) as data:
        reference = base.reference(data["pre_train_covariance"])
        train_command = tangent_features(
            data["train_command_covariance"][data["train_command_trial"] < 50], reference
        )
        train_idle = tangent_features(
            data["train_idle_iti_covariance"][data["train_idle_iti_trial"] < 50], reference
        )
    first50_model = fit_model(train_command, train_idle)

    anchor_cov, stream_cov, events, times, starts, allowed = base.prepare_target(subject, session)
    anchor_features = tangent_features(anchor_cov, reference)
    stream_features = tangent_features(stream_cov, reference)
    arms = {}
    for arm in ("first50_supervised", "eval_oof_oracle"):
        records = make_fold_records(
            stream_features, anchor_features, events, times, starts, allowed,
            arm, first50_model,
        )
        validate_records(records, arm)
        arms[arm] = evaluate_arm(records, shifts, 930000 + 1000 * session + subject)
    return {"subject": subject, "session": session, "arms": arms}


def bootstrap_median(values: np.ndarray, rng: np.random.Generator, draws: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    boot = np.empty(draws, dtype=float)
    for index in range(draws):
        boot[index] = np.median(values[rng.integers(0, len(values), len(values))])
    return [float(value) for value in np.quantile(boot, [0.025, 0.975])]


def summarize(rows: list[dict[str, object]], bootstrap: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    output: dict[str, object] = {}
    for session in (1, 2):
        selected = [row for row in rows if row["session"] == session]
        if not selected:
            continue
        session_output: dict[str, object] = {}
        for arm in ("first50_supervised", "eval_oof_oracle"):
            arm_output: dict[str, object] = {
                "n": len(selected),
                "window_auc_median": float(np.median([row["arms"][arm]["window_auc"] for row in selected])),
                "targets": {},
            }
            for target_fp in TARGET_FP:
                key = str(target_fp)
                target_output = {}
                for delta in DELTAS:
                    delta_key = str(delta)
                    values = np.asarray([
                        row["arms"][arm]["targets"][key]["circular_shift_excess"][delta_key]
                        for row in selected
                    ])
                    target_output[f"excess_{delta_key}s_median"] = float(np.median(values))
                    target_output[f"excess_{delta_key}s_bootstrap_95_ci"] = bootstrap_median(values, rng, bootstrap)
                for metric in ("actual_fp_per_min", "coverage", "temporal_iou", "strict_f1"):
                    target_output[f"{metric}_median"] = float(np.median([
                        row["arms"][arm]["targets"][key][metric] for row in selected
                    ]))
                arm_output["targets"][key] = target_output
            session_output[arm] = arm_output

        for target_fp in TARGET_FP:
            key = str(target_fp)
            for delta in (1.0, 2.0):
                delta_key = str(delta)
                differences = np.asarray([
                    row["arms"]["eval_oof_oracle"]["targets"][key]["circular_shift_excess"][delta_key]
                    - row["arms"]["first50_supervised"]["targets"][key]["circular_shift_excess"][delta_key]
                    for row in selected
                ])
                session_output[f"paired_oof_minus_first50_fp{int(target_fp)}_excess{int(delta)}s"] = {
                    "median": float(np.median(differences)),
                    "bootstrap_95_ci": bootstrap_median(differences, rng, bootstrap),
                }
        output[str(session)] = session_output
    return output


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--cohort", choices=("subset20", "holdout34", "all54"), default="subset20")
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cohort_subjects = {
        "subset20": SUBSET20,
        "holdout34": HOLDOUT34,
        "all54": ALL_SUBJECTS,
    }[args.cohort]
    subjects = tuple(args.subjects) if args.subjects else cohort_subjects
    output = args.output or Path(
        f"data/processed/lee2019_riemannian_p0_ceiling_{args.cohort}.json"
    )
    rows = []
    for session in args.sessions:
        for index, subject in enumerate(subjects, 1):
            rows.append(analyze(subject, session, args.shifts))
            print(
                f"P0 {args.cohort} S{session} sub-{subject:02d} "
                f"[{index}/{len(subjects)}]",
                flush=True,
            )
    payload = finite(
        {
            "protocol": {
                "question": "target-specific Riemannian event-information ceiling",
                "development_cohort": list(SUBSET20),
                "locked_holdout_cohort": list(HOLDOUT34),
                "evaluation": "locked pseudo-uncued trials 10-49 (40 trials); cue-relative pseudo-onset",
                "oracle": "5-fold contiguous trial-level cross-fitting within evaluation; fold-reset detector; evaluation-idle achieved-FP threshold",
                "comparator": "first 50 labelled trials; identical fold-reset and evaluation-idle achieved-FP threshold",
                "features": "1-s causal covariance, 20 sensorimotor channels, target pre-rest arithmetic reference, tangent-space LR",
                "decision": "EWMA H30 + MA 0.5 s + dwell 0.3 s + refractory 1 s",
                "primary": "circular-shift excess onset sensitivity at 1 s and achieved FP=2/min",
                "classification": "diagnostic oracle only; not deployable; not actual self-initiated performance",
            },
            "cohort": args.cohort,
            "summary": summarize(rows, args.bootstrap, args.seed),
            "rows": rows,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {output} ({len(rows)} subject-sessions)", flush=True)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
