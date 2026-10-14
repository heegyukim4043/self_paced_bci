"""Validate Riemannian+MA, supervised EEGNet, and SWPC-style SSL EEGNet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.swpc import (
    EEGNetGate,
    augment_windows,
    balanced_gate_loss,
    make_teacher,
    seed_everything,
    swpc_consistency_loss,
    update_ema,
)

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
SFREQ = 250
WINDOW = 250
DECISION_STEP = 0.5
SSL_EPOCH_CANDIDATES = (5, 15)


def windows(values: np.ndarray, ends: np.ndarray) -> np.ndarray:
    ends = np.asarray(ends, dtype=np.int64)
    if len(ends) == 0 or ends.min() < WINDOW or ends.max() > len(values):
        raise ValueError("invalid window ends")
    output = np.stack([values[end - WINDOW : end].T for end in ends])
    return np.asarray(output, dtype=np.float32)


def positive_ends(data, low: int, high: int) -> np.ndarray:
    output = []
    for trial in range(low, high):
        size = int(data["positive_bag_sizes"][trial])
        output.extend(data["positive_end_samples"][trial, :size])
    return np.asarray(output, dtype=np.int64)


def split_arrays(data, low: int, high: int, *, pre: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = windows(data["signal"], positive_ends(data, low, high))
    mask = (data["iti_instance_trials"] >= low) & (data["iti_instance_trials"] < high)
    iti = windows(data["signal"], data["iti_end_samples"][mask])
    if pre == "fit":
        rest_ends = data["pre_rest_fit_end_samples"]
    elif pre == "validation":
        rest_ends = data["pre_rest_validation_end_samples"]
    else:
        rest_ends = np.r_[data["pre_rest_fit_end_samples"], data["pre_rest_validation_end_samples"]]
    rest = windows(data["pre_rest"], rest_ends)
    return pos, rest, iti


def sample(values: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
    indices = torch.randint(len(values), (size,), generator=generator)
    return values[indices]


def train_supervised(
    pos: np.ndarray,
    rest: np.ndarray,
    iti: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[EEGNetGate, list[float]]:
    seed_everything(seed)
    model = EEGNetGate(pos.shape[1], pos.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    p = torch.from_numpy(pos)
    r = torch.from_numpy(rest)
    i = torch.from_numpy(iti)
    generator = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for _ in range(epochs):
        losses = []
        for _ in range(max(8, int(np.ceil(len(pos) / 64)))):
            optimizer.zero_grad(set_to_none=True)
            pp = sample(p, 32, generator).to(device)
            rr = sample(r, 32, generator).to(device)
            ii = sample(i, 32, generator).to(device)
            loss = balanced_gate_loss(model(pp), model(rr), model(ii))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


def refine_ssl(
    model: EEGNetGate,
    unlabeled: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[EEGNetGate, list[float]]:
    seed_everything(seed + 10_000)
    teacher = make_teacher(model).to(device)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4
    )
    dataset = TensorDataset(torch.from_numpy(unlabeled))
    generator = torch.Generator().manual_seed(seed + 10_000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    history = []
    model.train()
    for _ in range(epochs):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            view1, view2 = augment_windows(batch), augment_windows(batch)
            student = model.encode(view1)
            with torch.no_grad():
                teacher_feature = teacher.encode(view2)
            loss = swpc_consistency_loss(student, teacher_feature)
            loss.backward()
            optimizer.step()
            update_ema(model, teacher, 0.9995)
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    return model, history


@torch.no_grad()
def predict(model: EEGNetGate, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=256)
    return np.concatenate([model(batch.to(device)).cpu().numpy() for (batch,) in loader])


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.r_[positive, negative]
    ranks = rankdata(values, method="average")
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def evaluate_scores(
    times: np.ndarray,
    scores: np.ndarray,
    idle: np.ndarray,
    command: np.ndarray,
    seed: int,
    shifts: int,
) -> dict:
    sequence = ScoredSequence(times, scores, idle, command)
    point = tune_source_operating_points(
        [sequence], "ma_dwell", [2.0], DECISION_STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0, grid_size=160,
    )[2.0]
    decision = apply_decision(
        sequence, "ma_dwell", point.threshold, DECISION_STEP,
        ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
    )
    observed = evaluate_alarms(times, decision.alarms, idle, command)
    rng = np.random.default_rng(seed)
    null = np.asarray([
        evaluate_alarms(times, np.roll(decision.alarms, int(offset)), idle, command).true_positive_rate
        for offset in rng.integers(1, len(decision.alarms), size=shifts)
    ])
    command_mask = np.zeros(len(times), dtype=bool)
    idle_mask = np.zeros(len(times), dtype=bool)
    for start, stop in command:
        command_mask |= (times >= start) & (times < stop)
    for start, stop in idle:
        idle_mask |= (times >= start) & (times < stop)
    return {
        "fp_per_minute": observed.fp_per_minute,
        "observed_tpr": observed.true_positive_rate,
        "null_tpr": float(np.median(null)),
        "excess_tpr": float(observed.true_positive_rate - np.median(null)),
        "latency": observed.median_delay_seconds,
        "window_auc": auc(scores[command_mask], scores[idle_mask]),
        "threshold": point.threshold,
    }


def evaluate_model(model, data, low, high, device, seed, shifts):
    first_end = int(3 * SFREQ)
    step = int(DECISION_STEP * SFREQ)
    ends = np.arange(first_end, len(data["signal"]) + 1, step)
    scores = predict(model, windows(data["signal"], ends), device)
    times = ends / SFREQ
    idle = data["iti_intervals"][(data["iti_trials"] >= low) & (data["iti_trials"] < high)]
    command = data["command_intervals"][low:high]
    return evaluate_scores(times, scores, idle, command, seed, shifts)


def bootstrap(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--sl-epochs", type=int, default=30)
    parser.add_argument("--ssl-epochs", type=int, nargs="+", default=SSL_EPOCH_CANDIDATES)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/three_model_validation.json"))
    args = parser.parse_args()
    if args.smoke:
        args.subjects, args.sessions, args.seeds = args.subjects[:1], [1], args.seeds[:1]
        args.sl_epochs, args.ssl_epochs, args.shifts, args.bootstrap = 2, [1, 2], 20, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_rows = []
    for subject in args.subjects:
        path = args.cache_root / "session1" / f"sub-{subject:02d}_mil_raw.npz"
        with np.load(path, allow_pickle=False) as data:
            fit = split_arrays(data, 0, 40, pre="fit")
            unlabeled = np.concatenate(fit)
            for seed in args.seeds:
                sl, sl_history = train_supervised(*fit, seed, args.sl_epochs, device)
                sl_metric = evaluate_model(sl, data, 40, 50, device, seed + subject, args.shifts)
                for ssl_epochs in args.ssl_epochs:
                    refined, ssl_history = refine_ssl(
                        torch.jit._recursive.wrap_cpp_module(sl._c)
                        if False else __import__("copy").deepcopy(sl),
                        unlabeled, seed, ssl_epochs, device,
                    )
                    metric = evaluate_model(refined, data, 40, 50, device, seed + subject, args.shifts)
                    validation_rows.append({
                        "subject": subject, "seed": seed, "ssl_epochs": ssl_epochs,
                        "sl": sl_metric, "ssl": metric,
                        "sl_final_loss": sl_history[-1], "ssl_final_loss": ssl_history[-1],
                    })
        print(f"validation subject={subject}", flush=True)

    candidate_summary = []
    for candidate in args.ssl_epochs:
        rows = [r for r in validation_rows if r["ssl_epochs"] == candidate]
        candidate_summary.append({
            "ssl_epochs": candidate,
            "median_fp": float(np.median([r["ssl"]["fp_per_minute"] for r in rows])),
            "median_excess": float(np.median([r["ssl"]["excess_tpr"] for r in rows])),
            "median_delta_vs_sl": float(np.median([r["ssl"]["excess_tpr"] - r["sl"]["excess_tpr"] for r in rows])),
        })
    admissible = [r for r in candidate_summary if 1.5 <= r["median_fp"] <= 2.5]
    pool = admissible or candidate_summary
    selected = sorted(pool, key=lambda r: (-r["median_excess"], r["ssl_epochs"]))[0]["ssl_epochs"]
    calibration_failure = not bool(admissible)

    locked_rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = args.cache_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz"
            with np.load(path, allow_pickle=False) as data:
                train = split_arrays(data, 0, 50, pre="all")
                unlabeled = np.concatenate(train)
                for seed in args.seeds:
                    sl, sl_history = train_supervised(*train, seed, args.sl_epochs, device)
                    sl_metric = evaluate_model(sl, data, 50, 100, device, seed + session * 100 + subject, args.shifts)
                    ssl, ssl_history = refine_ssl(sl, unlabeled, seed, selected, device)
                    ssl_metric = evaluate_model(ssl, data, 50, 100, device, seed + session * 100 + subject, args.shifts)
                    locked_rows.append({
                        "session": session, "subject": subject, "seed": seed,
                        "sl": sl_metric, "ssl": ssl_metric,
                        "sl_final_loss": sl_history[-1], "ssl_final_loss": ssl_history[-1],
                    })
            print(f"locked session={session} subject={subject}", flush=True)

    summary = []
    for session in args.sessions:
        subjects = []
        for subject in args.subjects:
            rows = [r for r in locked_rows if r["session"] == session and r["subject"] == subject]
            subjects.append({
                "subject": subject,
                "sl_excess": float(np.mean([r["sl"]["excess_tpr"] for r in rows])),
                "ssl_excess": float(np.mean([r["ssl"]["excess_tpr"] for r in rows])),
                "sl_fp": float(np.mean([r["sl"]["fp_per_minute"] for r in rows])),
                "ssl_fp": float(np.mean([r["ssl"]["fp_per_minute"] for r in rows])),
                "sl_auc": float(np.mean([r["sl"]["window_auc"] for r in rows])),
                "ssl_auc": float(np.mean([r["ssl"]["window_auc"] for r in rows])),
            })
        sl = np.asarray([r["sl_excess"] for r in subjects])
        ssl = np.asarray([r["ssl_excess"] for r in subjects])
        summary.append({
            "session": session,
            "subjects": len(subjects),
            "sl_excess_median": float(np.median(sl)),
            "sl_excess_ci": bootstrap(sl, SEEDS[0] + session, args.bootstrap),
            "ssl_excess_median": float(np.median(ssl)),
            "ssl_excess_ci": bootstrap(ssl, SEEDS[1] + session, args.bootstrap),
            "ssl_minus_sl_median": float(np.median(ssl - sl)),
            "ssl_minus_sl_ci": bootstrap(ssl - sl, SEEDS[2] + session, args.bootstrap),
            "sl_auc_median": float(np.median([r["sl_auc"] for r in subjects])),
            "ssl_auc_median": float(np.median([r["ssl_auc"] for r in subjects])),
            "subject_rows": subjects,
        })
    payload = {
        "protocol": {
            "preregistered": True, "device": str(device), "sl_epochs": args.sl_epochs,
            "ssl_epoch_candidates": list(args.ssl_epochs), "selected_ssl_epochs": selected,
            "calibration_failure": calibration_failure, "seeds": list(args.seeds),
            "fp_axis": 2.0, "null_shifts": args.shifts,
        },
        "validation_candidates": candidate_summary,
        "validation_rows": validation_rows,
        "summary": summary,
        "locked_rows": locked_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_ssl_epochs": selected, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
