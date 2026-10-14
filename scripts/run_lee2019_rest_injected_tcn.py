"""Causal tangent-feature TCN on the Lee2019 rest-injected benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
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
from uncued_mi.swpc import balanced_gate_loss, seed_everything
from uncued_mi.tangent_tcn import TangentTCN

RESTS = ("pre_train", "post_train", "pre_test", "post_test")
SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
CONTEXTS = (3, 5, 10)
STEP = 0.1


def reference(covariances):
    matrix = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    return matrix + np.eye(len(matrix)) * max(
        float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12
    )


def contexts(values: np.ndarray, size: int) -> np.ndarray:
    if len(values) < size:
        return np.empty((0, size, values.shape[1]), dtype=np.float32)
    return np.asarray(
        [values[index - size + 1:index + 1] for index in range(size - 1, len(values))],
        dtype=np.float32,
    )


def grouped_contexts(values, trials, low, high, size):
    output = [contexts(values[trials == trial], size) for trial in range(low, high)]
    output = [x for x in output if len(x)]
    return np.concatenate(output)


def train_model(positive, rest, iti, features, seed, epochs):
    seed_everything(seed)
    model = TangentTCN(features, hidden=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    p, r, i = map(torch.from_numpy, (positive, rest, iti))
    generator = torch.Generator().manual_seed(seed)

    def draw(values):
        return values[torch.randint(len(values), (32,), generator=generator)]

    history = []
    for _ in range(epochs):
        losses = []
        model.train()
        for _ in range(max(8, int(np.ceil(len(positive) / 64)))):
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_gate_loss(model(draw(p)), model(draw(r)), model(draw(i)))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.no_grad()
def score_series(model, values, size):
    x = contexts(values, size)
    model.eval()
    predicted = []
    for start in range(0, len(x), 256):
        predicted.append(model(torch.from_numpy(x[start:start + 256])).numpy())
    output = np.full(len(values), np.nan, dtype=float)
    output[size - 1:] = np.concatenate(predicted)
    output[:size - 1] = float(np.nanmedian(output))
    return output


def make_calibration(scores):
    return ScoredSequence(
        np.arange(len(scores)) * STEP, scores,
        np.asarray([[0.0, len(scores) * STEP]]), np.empty((0, 2)),
    )


def evaluate_stream(stream, calibration, offsets):
    transferred_point = tune_source_operating_points(
        [make_calibration(x) for x in calibration],
        "ma_dwell", [2.0], STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        grid_size=160,
    )[2.0]
    oracle_point = tune_source_operating_points(
        [stream], "ma_dwell", [2.0], STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        grid_size=160,
    )[2.0]

    def at(threshold):
        result = apply_decision(
            stream, "ma_dwell", threshold, STEP,
            ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        )
        observed = evaluate_alarms(
            stream.times, result.alarms, stream.idle_intervals,
            stream.command_intervals,
        )
        null = [
            evaluate_alarms(
                stream.times, np.roll(result.alarms, int(offset)),
                stream.idle_intervals, stream.command_intervals,
            ).true_positive_rate
            for offset in offsets
        ]
        return {
            "fp_per_minute": observed.fp_per_minute,
            "observed_tpr": observed.true_positive_rate,
            "null_tpr": float(np.median(null)),
            "excess_tpr": float(observed.true_positive_rate - np.median(null)),
            "latency": observed.median_delay_seconds,
        }

    return {
        "calibration_fp": transferred_point.source_fp_per_minute,
        "transferred": at(transferred_point.threshold),
        "oracle_common_fp": at(oracle_point.threshold),
    }


def analyze(path, subject, session, context, seed, mode, epochs, shifts):
    train_stop, eval_low, eval_high = (
        (40, 40, 50) if mode == "validation" else (50, 50, 100)
    )
    with np.load(path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        command_feature = tangent_features(data["train_command_covariance"], ref)
        iti_feature = tangent_features(data["train_idle_iti_covariance"], ref)
        rest_feature = {
            name: tangent_features(data[f"{name}_covariance"], ref) for name in RESTS
        }
        command_trial = data["train_command_trial"]
        iti_trial = data["train_idle_iti_trial"]
        fit_mask = command_trial < train_stop
        idle_mask = iti_trial < train_stop
        scaler_parts = [
            command_feature[fit_mask],
            iti_feature[idle_mask],
        ] + [
            values[:len(values) // (4 if mode == "validation" else 2)]
            for values in rest_feature.values()
        ]
        scaler = StandardScaler().fit(np.concatenate(scaler_parts))
        command_feature = scaler.transform(command_feature)
        iti_feature = scaler.transform(iti_feature)
        rest_feature = {key: scaler.transform(value) for key, value in rest_feature.items()}
        positive = grouped_contexts(
            command_feature, command_trial, 0, train_stop, context
        )
        iti = grouped_contexts(iti_feature, iti_trial, 0, train_stop, context)
        rest_fit, calibration_feature, evaluation_feature = [], [], []
        for name in RESTS:
            values = rest_feature[name]
            if mode == "validation":
                quarter = len(values) // 4
                fit, calibration, evaluation = (
                    values[:quarter], values[:quarter], values[quarter:2 * quarter]
                )
            else:
                half = len(values) // 2
                fit, calibration, evaluation = (
                    values[:half], values[:half], values[half:]
                )
            rest_fit.append(contexts(fit, context))
            calibration_feature.append(calibration)
            evaluation_feature.append(evaluation)
        rest = np.concatenate(rest_fit)
        eval_trials = [
            command_feature[command_trial == trial]
            for trial in range(eval_low, eval_high)
        ]
    model, history = train_model(
        positive, rest, iti, command_feature.shape[1], seed, epochs
    )
    tcn_calibration = [score_series(model, x, context) for x in calibration_feature]
    tcn_evaluation = [score_series(model, x, context) for x in evaluation_feature]
    tcn_trials = [score_series(model, x, context) for x in eval_trials]

    # Deterministic Riemannian LR comparator from the identical features.
    lr = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260728,
    ).fit(
        np.r_[command_feature[fit_mask], iti_feature[idle_mask]],
        np.r_[np.ones(fit_mask.sum()), np.zeros(idle_mask.sum())],
    )
    lr_calibration = [lr.decision_function(x) for x in calibration_feature]
    lr_evaluation = [lr.decision_function(x) for x in evaluation_feature]
    lr_trials = [lr.decision_function(x) for x in eval_trials]

    stream_seed = (
        20260728 + 100 * session + subject
        if mode == "locked"
        else 20260729 + 1000 * session + subject + 60000
    )
    tcn_stream, audit = build_rest_injected_stream(
        tcn_evaluation, tcn_trials, step_seconds=STEP,
        minimum_rest_seconds=5, maximum_rest_seconds=11,
        guard_seconds=1, seed=stream_seed,
    )
    lr_stream, _ = build_rest_injected_stream(
        lr_evaluation, lr_trials, step_seconds=STEP,
        minimum_rest_seconds=5, maximum_rest_seconds=11,
        guard_seconds=1, seed=stream_seed,
    )
    rng = np.random.default_rng(seed + 1000 * session + subject)
    offsets = rng.integers(1, len(tcn_stream.times), size=shifts)
    tcn = evaluate_stream(tcn_stream, tcn_calibration, offsets)
    comparator = evaluate_stream(lr_stream, lr_calibration, offsets)
    return {
        "subject": subject, "session": session, "seed": seed,
        "context_steps": context, "audit": audit.__dict__,
        "training_counts": {
            "positive": len(positive), "rest": len(rest), "iti": len(iti)
        },
        "loss_start": history[0], "loss_end": history[-1],
        "tcn": tcn, "riemannian": comparator,
        "paired_delta": (
            tcn["oracle_common_fp"]["excess_tpr"]
            - comparator["oracle_common_fp"]["excess_tpr"]
        ),
    }


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(subject_rows, seed, repeats):
    tcn = np.asarray([x["tcn_excess"] for x in subject_rows])
    comparator = np.asarray([x["riemannian_excess"] for x in subject_rows])
    delta = tcn - comparator
    return {
        "subjects": len(subject_rows),
        "tcn_fp_median": float(np.median([x["tcn_fp"] for x in subject_rows])),
        "tcn_transferred_fp_median": float(np.median([
            x["tcn_transferred_fp"] for x in subject_rows
        ])),
        "tcn_excess_median": float(np.median(tcn)),
        "tcn_excess_ci": bootstrap(tcn, seed, repeats),
        "riemannian_excess_median": float(np.median(comparator)),
        "paired_delta_median": float(np.median(delta)),
        "paired_delta_ci": bootstrap(delta, seed + 10000, repeats),
        "tcn_latency_median": float(np.nanmedian([x["tcn_latency"] for x in subject_rows])),
        "riemannian_latency_median": float(np.nanmedian([x["riemannian_latency"] for x in subject_rows])),
    }


def collapse_seed_rows(rows):
    output = []
    for subject in sorted({x["subject"] for x in rows}):
        selected = [x for x in rows if x["subject"] == subject]
        output.append({
            "subject": subject,
        "tcn_excess": float(np.mean([
            x["tcn"]["oracle_common_fp"]["excess_tpr"] for x in selected
        ])),
        "riemannian_excess": float(np.mean([
            x["riemannian"]["oracle_common_fp"]["excess_tpr"] for x in selected
        ])),
        "tcn_fp": float(np.mean([
            x["tcn"]["oracle_common_fp"]["fp_per_minute"] for x in selected
        ])),
        "tcn_transferred_fp": float(np.mean([
            x["tcn"]["transferred"]["fp_per_minute"] for x in selected
        ])),
        "tcn_latency": float(np.nanmean([
            x["tcn"]["oracle_common_fp"]["latency"] for x in selected
        ])),
        "riemannian_latency": float(np.nanmean([
            x["riemannian"]["oracle_common_fp"]["latency"] for x in selected
        ])),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_four_rest"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--selection-subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_rest_injected_tcn.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.subjects = args.subjects[:1]
        args.selection_subjects = args.selection_subjects[:1]
        args.seeds = args.seeds[:1]
        args.epochs, args.shifts, args.bootstrap = 2, 20, 100
    validation = []
    for context in CONTEXTS:
        rows = [
            analyze(
                args.cache_root / "session1" / f"sub-{subject:02d}.npz",
                subject, 1, context, args.seeds[0], "validation",
                args.epochs, args.shifts,
            )
            for subject in args.selection_subjects
        ]
        collapsed = collapse_seed_rows(rows)
        summary = summarize(collapsed, 1000 + context, args.bootstrap)
        validation.append({"context_steps": context, "summary": summary, "rows": rows})
        print(f"validated context={context}", flush=True)
    admissible = [x for x in validation if 1.5 <= x["summary"]["tcn_fp_median"] <= 2.5]
    pool = admissible or validation
    selected = sorted(
        pool, key=lambda x: (-x["summary"]["tcn_excess_median"], x["context_steps"])
    )[0]
    context = selected["context_steps"]
    locked = []
    for session in (1, 2):
        rows = []
        for subject in args.subjects:
            for seed in args.seeds:
                rows.append(analyze(
                    args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz",
                    subject, session, context, seed, "locked",
                    args.epochs, args.shifts,
                ))
            print(f"locked session={session} subject={subject}", flush=True)
        collapsed = collapse_seed_rows(rows)
        locked.append({
            "session": session,
            "summary": summarize(collapsed, 2000 + session, args.bootstrap),
            "subject_rows": collapsed,
            "seed_rows": rows,
        })
    payload = {
        "protocol": {
            "model": "two-state causal tangent-feature TCN",
            "contexts": list(CONTEXTS),
            "selection": "session1 subset20 validation, one fixed seed",
            "locked": "54 subjects x two sessions x five seeds",
            "comparator": "same-feature same-stream Riemannian LR + MA/dwell",
            "transition_label": "not used",
        },
        "validation": validation,
        "selection_audit": {
            "admissible_count": len(admissible),
            "selected_context_steps": context,
        },
        "locked": locked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection_audit": payload["selection_audit"],
        "locked": [{"session": x["session"], **x["summary"]} for x in locked],
    }, indent=2))


if __name__ == "__main__":
    main()
