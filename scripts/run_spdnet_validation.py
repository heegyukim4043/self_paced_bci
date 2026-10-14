"""Preregistered compact multi-band SPDNet validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.spdnet import CompactMultiBandSPDNet
from uncued_mi.swpc import balanced_gate_loss, seed_everything

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
STEP = 0.5


def draw(values, size, generator):
    return values[torch.randint(len(values), (size,), generator=generator)]


def train_model(pos, rest, iti, seed, epochs):
    seed_everything(seed)
    model = CompactMultiBandSPDNet(3, 20, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    p, r, i = map(torch.from_numpy, (pos, rest, iti))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for _ in range(max(8, int(np.ceil(len(pos) / 64)))):
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_gate_loss(
                model(draw(p, 32, generator)),
                model(draw(r, 32, generator)),
                model(draw(i, 32, generator)),
            )
            loss.backward()
            optimizer.step()
            model.project_bimap()
    return model


@torch.no_grad()
def predict(model, values):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=256)
    return np.concatenate([model(batch).numpy() for (batch,) in loader])


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


def analyze(path, session, subject, seeds, epochs, shifts):
    with np.load(path) as data:
        pos, rest, iti, stream = data["positive"], data["rest"], data["iti"], data["stream"]
        times = data["stream_ends"] / 250.0
        idle = data["iti_intervals"][(data["iti_trials"] >= 50) & (data["iti_trials"] < 100)]
        command = data["command_intervals"][50:100]
    metrics = []
    for seed in seeds:
        model = train_model(pos, rest, iti, seed, epochs)
        metrics.append(evaluate(
            times, predict(model, stream), idle, command,
            seed + session * 1000 + subject, shifts,
        ))
    return {"session": session, "subject": subject, "seeds": metrics}


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/multiband_cov"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/spdnet_validation.json"))
    args = parser.parse_args()
    if args.smoke:
        args.subjects, args.sessions, args.seeds = args.subjects[:1], [1], args.seeds[:1]
        args.epochs, args.shifts, args.bootstrap = 2, 20, 100
    partial = args.output.with_suffix(".partial.json")
    rows = []
    if partial.exists() and not args.smoke:
        rows = json.loads(partial.read_text(encoding="utf-8"))
    for session in args.sessions:
        for subject in args.subjects:
            if any(row["session"] == session and row["subject"] == subject for row in rows):
                print(f"resume-skip session={session} subject={subject}", flush=True)
                continue
            rows.append(analyze(
                args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz",
                session, subject, args.seeds, args.epochs, args.shifts,
            ))
            if not args.smoke:
                partial.write_text(json.dumps(rows), encoding="utf-8")
            print(f"done session={session} subject={subject}", flush=True)
    comparator = json.loads(Path("data/processed/mil_riemannian_comparator.json").read_text(encoding="utf-8"))
    summary = []
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        rmap = {row["subject"]: row["excess_tpr"] for row in comparator["rows"] if row["session"] == session}
        values = np.asarray([np.mean([m["excess_tpr"] for m in row["seeds"]]) for row in selected])
        baseline = np.asarray([rmap[row["subject"]] for row in selected])
        summary.append({
            "session": session, "subjects": len(selected),
            "excess_median": float(np.median(values)),
            "excess_ci": bootstrap(values, 100 + session, args.bootstrap),
            "minus_riemannian_median": float(np.median(values - baseline)),
            "minus_riemannian_ci": bootstrap(values - baseline, 200 + session, args.bootstrap),
            "fp_median": float(np.median([np.mean([m["fp_per_minute"] for m in row["seeds"]]) for row in selected])),
        })
    payload = {"protocol": {"preregistered": True, "epochs": args.epochs, "seeds": list(args.seeds)}, "summary": summary, "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
