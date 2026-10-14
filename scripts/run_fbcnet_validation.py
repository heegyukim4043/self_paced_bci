"""Preregistered compact FBCNet locked validation."""

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

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import channel_indices, lee2019_path, load_lee2019_run
from uncued_mi.fbcnet import CompactFBCNet
from uncued_mi.features import _causal_band_signal, _causal_downsample, common_average_reference
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.swpc import balanced_gate_loss, seed_everything

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
BANDS = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
SFREQ, WINDOW, STEP = 250, 250, 0.5


def filter_banks(eeg, channels, native_sfreq, selected):
    config = ExperimentConfig(window_seconds=1.0)
    referenced = common_average_reference(eeg)
    selected_eeg = referenced[:, channel_indices(channels, selected)]
    output = []
    for low, high in BANDS:
        filtered = _causal_band_signal(selected_eeg, native_sfreq, low, high, config.filter_order)
        output.append(_causal_downsample(filtered, native_sfreq, SFREQ).T)
    return np.asarray(output, dtype=np.float32)


def normalize(signal, pre):
    calibration = pre[..., int(2 * SFREQ):]
    mean = calibration.mean(axis=-1, keepdims=True)
    std = calibration.std(axis=-1, keepdims=True).clip(1e-6)
    return (signal - mean) / std, (pre - mean) / std


def windows(values, ends):
    return np.asarray(
        [values[:, :, end - WINDOW:end] for end in np.asarray(ends, dtype=int)],
        dtype=np.float32,
    )


def positive_ends(cache):
    output = []
    for trial in range(50):
        size = int(cache["positive_bag_sizes"][trial])
        output.extend(cache["positive_end_samples"][trial, :size])
    return np.asarray(output)


def draw(values, size, generator):
    return values[torch.randint(len(values), (size,), generator=generator)]


def train_model(pos, rest, iti, seed, epochs):
    seed_everything(seed)
    model = CompactFBCNet(3, pos.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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


def analyze(data_root, cache_root, session, subject, seeds, epochs, shifts):
    run = load_lee2019_run(lee2019_path(data_root, subject, session), "train")
    selected = ExperimentConfig().sensorimotor_channels
    signal = filter_banks(run.eeg, run.channels, run.sfreq, selected)
    pre = filter_banks(run.pre_rest, run.channels, run.sfreq, selected)
    signal, pre = normalize(signal, pre)
    with np.load(cache_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz") as cache:
        pos = windows(signal, positive_ends(cache))
        iti_mask = cache["iti_instance_trials"] < 50
        iti = windows(signal, cache["iti_end_samples"][iti_mask])
        pre_ends = np.r_[cache["pre_rest_fit_end_samples"], cache["pre_rest_validation_end_samples"]]
        rest = windows(pre, pre_ends)
        ends = np.arange(3 * SFREQ, signal.shape[-1] + 1, int(STEP * SFREQ))
        stream = windows(signal, ends)
        times = ends / SFREQ
        idle = cache["iti_intervals"][(cache["iti_trials"] >= 50) & (cache["iti_trials"] < 100)]
        command = cache["command_intervals"][50:100]
    metrics = []
    for seed in seeds:
        model = train_model(pos, rest, iti, seed, epochs)
        metrics.append(evaluate(times, predict(model, stream), idle, command,
                                seed + session * 1000 + subject, shifts))
    return {"session": session, "subject": subject, "seeds": metrics}


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/fbcnet_validation.json"))
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
            rows.append(analyze(args.data_root, args.cache_root, session, subject,
                                args.seeds, args.epochs, args.shifts))
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
    payload = {"protocol": {"preregistered": True, "bands": BANDS, "epochs": args.epochs, "seeds": list(args.seeds)}, "summary": summary, "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
