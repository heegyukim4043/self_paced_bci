"""Supervised EEGNet on the four-rest injected Lee2019 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_three_model_validation import (
    predict,
    split_arrays,
    train_supervised,
    windows,
)
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    tune_source_operating_points,
)
from uncued_mi.rest_injected import build_rest_injected_stream

RESTS = ("pre_train", "post_train", "pre_test", "post_test")
SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
SEEDS = (20260721, 20260722, 20260723)
SFREQ = 250
WINDOW = 250
STEP = 0.1


def rest_ends(size: int, transient: int) -> np.ndarray:
    return np.arange(transient + WINDOW, size + 1, int(STEP * SFREQ))


def calibration_sequence(scores):
    return ScoredSequence(
        np.arange(len(scores)) * STEP, scores,
        np.asarray([[0.0, len(scores) * STEP]]), np.empty((0, 2)),
    )


def evaluate(stream, calibration, offsets):
    transferred = tune_source_operating_points(
        [calibration_sequence(x) for x in calibration],
        "ma_dwell", [2.0], STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        grid_size=160,
    )[2.0]
    oracle = tune_source_operating_points(
        [stream], "ma_dwell", [2.0], STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        grid_size=160,
    )[2.0]

    def at(threshold):
        decision = apply_decision(
            stream, "ma_dwell", threshold, STEP,
            ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
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
            "fp_per_minute": observed.fp_per_minute,
            "observed_tpr": observed.true_positive_rate,
            "null_tpr": float(np.median(null)),
            "excess_tpr": float(observed.true_positive_rate - np.median(null)),
            "latency": observed.median_delay_seconds,
        }
    return {
        "calibration_fp": transferred.source_fp_per_minute,
        "transferred": at(transferred.threshold),
        "oracle_common_fp": at(oracle.threshold),
    }


def analyze(mil_path, rest_path, subject, session, seed, epochs, shifts, device):
    with np.load(mil_path, allow_pickle=False) as data, np.load(rest_path, allow_pickle=False) as rest:
        train = split_arrays(data, 0, 50, pre="all")
        started = time.perf_counter()
        model, history = train_supervised(*train, seed, epochs, device)
        training_seconds = time.perf_counter() - started
        calibration, evaluation = [], []
        for name in RESTS:
            signal = rest[name]
            scores = predict(model, windows(signal, rest_ends(len(signal), int(rest["transient_samples"]))), device)
            half = len(scores) // 2
            calibration.append(scores[:half])
            evaluation.append(scores[half:])
        trial_scores = []
        for trial in range(50, 100):
            size = int(data["positive_bag_sizes"][trial])
            ends = data["positive_end_samples"][trial, :size]
            trial_scores.append(predict(model, windows(data["signal"], ends), device))
    stream, audit = build_rest_injected_stream(
        evaluation, trial_scores, step_seconds=STEP,
        minimum_rest_seconds=5, maximum_rest_seconds=11,
        guard_seconds=1, seed=20260728 + 100 * session + subject,
    )
    rng = np.random.default_rng(seed + 1000 * session + subject)
    offsets = rng.integers(1, len(stream.times), size=shifts)
    return {
        "subject": subject, "session": session, "seed": seed,
        "device": str(device), "epochs": epochs,
        "training_seconds": training_seconds,
        "loss_start": history[0], "loss_end": history[-1],
        "audit": audit.__dict__,
        "metrics": evaluate(stream, calibration, offsets),
    }


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, seed, repeats):
    subjects = []
    for subject in sorted({x["subject"] for x in rows}):
        selected = [x for x in rows if x["subject"] == subject]
        subjects.append({
            "subject": subject,
            "transferred_fp": float(np.mean([
                x["metrics"]["transferred"]["fp_per_minute"] for x in selected
            ])),
            "oracle_fp": float(np.mean([
                x["metrics"]["oracle_common_fp"]["fp_per_minute"] for x in selected
            ])),
            "excess": float(np.mean([
                x["metrics"]["oracle_common_fp"]["excess_tpr"] for x in selected
            ])),
            "latency": float(np.nanmean([
                x["metrics"]["oracle_common_fp"]["latency"] for x in selected
            ])),
            "training_seconds": float(np.mean([
                x["training_seconds"] for x in selected
            ])),
        })
    excess = np.asarray([x["excess"] for x in subjects])
    return {
        "subjects": len(subjects),
        "transferred_fp_median": float(np.median([x["transferred_fp"] for x in subjects])),
        "oracle_fp_median": float(np.median([x["oracle_fp"] for x in subjects])),
        "excess_median": float(np.median(excess)),
        "excess_ci": bootstrap(excess, seed, repeats),
        "latency_median": float(np.nanmedian([x["latency"] for x in subjects])),
        "training_seconds_per_seed_median": float(np.median([
            x["training_seconds"] for x in subjects
        ])),
        "subject_rows": subjects,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mil-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--rest-root", type=Path, default=Path("data/interim/lee2019_four_rest_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_rest_injected_eegnet.json"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            for seed in args.seeds:
                rows.append(analyze(
                    args.mil_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz",
                    args.rest_root / f"session{session}" / f"sub-{subject:02d}.npz",
                    subject, session, seed, args.epochs, args.shifts, device,
                ))
            print(f"eegnet session={session} subject={subject}", flush=True)
    summary = [
        {
            "session": session,
            **summarize(
                [x for x in rows if x["session"] == session],
                20260729 + session, args.bootstrap,
            ),
        }
        for session in args.sessions
    ]
    payload = {
        "protocol": {
            "model": "supervised EEGNet",
            "device": str(device), "epochs": args.epochs,
            "seeds": list(args.seeds),
            "stream": "four-rest-injected pseudo-online",
            "metrics": "transferred operating point + oracle common-FP=2",
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
