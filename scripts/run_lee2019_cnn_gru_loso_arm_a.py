"""CNN-GRU source-pooled LOSO pilot under the locked Arm-A decision layer.

The target subject is excluded from CNN-GRU training.  Only target pre-rest is
used to initialize EWMA normalization.  This runner is deliberately separate
from ``run_lee2019_cnn_gru.py`` because that script is target-specific.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_lee2019_cnn_gru import (
    CausalCNNGRU, SFREQ, STEP, make_windows, normalize, predict, rest_windows,
    train_model,
)
from run_fbcnet_validation import filter_banks
from run_lee2019_stateful_gru import build_no_guard_stream, calibration_sequence, onset_rates
from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path, load_lee2019_run
from uncued_mi.decision import moving_average_dwell
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma, ewma_z, interval_metrics, intervals_from_alarm, ma_trace,
)


SUBJECTS = tuple(range(1, 55))
WINDOW = 250
MA_SECONDS = 0.5
DWELL_SECONDS = 0.3
REFRACTORY_SECONDS = 1.0
TARGETS = (2.0, 4.0)


def _sample(values, n):
    values = np.asarray(values)
    if n <= 0 or len(values) <= n:
        return values
    return values[np.linspace(0, len(values) - 1, n).round().astype(int)]


def prepare_subject(data_root, cache_root, subject, session, max_source_windows=200):
    run = load_lee2019_run(lee2019_path(data_root, subject, session), "train")
    selected = ExperimentConfig().sensorimotor_channels
    signal = filter_banks(run.eeg, run.channels, run.sfreq, selected)
    raw_rests = [
        filter_banks(values, run.channels, run.sfreq, selected)
        for values in (run.pre_rest, run.post_rest)
    ]
    test = load_lee2019_run(lee2019_path(data_root, subject, session), "test")
    raw_rests += [
        filter_banks(values, test.channels, test.sfreq, selected)
        for values in (test.pre_rest, test.post_rest)
    ]
    signal, pre = normalize(signal, raw_rests[0])
    mean = pre[..., int(2 * SFREQ):].mean(axis=-1, keepdims=True)
    std = pre[..., int(2 * SFREQ):].std(axis=-1, keepdims=True).clip(1e-6)
    rests = [pre] + [(values - mean) / std for values in raw_rests[1:]]

    with np.load(cache_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz") as cache:
        command_trials = []
        for start, stop in cache["command_intervals"]:
            ends = np.arange(int(round((start + 1.0) * SFREQ)), int(round(stop * SFREQ)), int(STEP * SFREQ))
            command_trials.append(make_windows(signal, ends))
        iti_trials = []
        for trial in range(50):
            mask = cache["iti_instance_trials"] == trial
            if np.any(mask):
                iti_trials.append(make_windows(signal, cache["iti_end_samples"][mask]))

    rest_split = [rest_windows(value) for value in rests]
    train_idle = iti_trials + [rest_split[0][0]]
    train_x, y, weights, _ = build_no_guard_stream(
        train_idle, command_trials[:50], 20260700 + 100 * session + subject,
    )
    if max_source_windows > 0 and len(train_x) > max_source_windows:
        # Keep a balanced, deterministic source subset to make pooled LOSO
        # training feasible on an 8 GB GPU.  The target stream is never capped.
        rng = np.random.default_rng(20260911 + 100 * session + subject)
        selected = []
        for label in (0.0, 1.0):
            indices = np.flatnonzero(np.asarray(y) == label)
            take = min(len(indices), max_source_windows // 2)
            selected.extend(rng.choice(indices, take, replace=False).tolist())
        selected = np.asarray(sorted(selected), dtype=int)
        train_x, y, weights = np.asarray(train_x)[selected], np.asarray(y)[selected], np.asarray(weights)[selected]
    eval_rest = [split[1] for split in rest_split]
    eval_x, _, _, template = build_no_guard_stream(
        eval_rest, command_trials[50:], 20260728 + 100 * session + subject,
    )
    # The second half of pre-train rest is the only target anchor used at test
    # time.  It precedes the evaluated stream and contains no command labels.
    anchor_x = rest_split[0][1]
    return {
        "train_x": np.asarray(train_x, dtype=np.float32),
        "train_y": np.asarray(y, dtype=np.float32),
        "train_w": np.asarray(weights, dtype=np.float32),
        "eval_x": np.asarray(eval_x, dtype=np.float32),
        "template": template,
        "anchor_x": np.asarray(anchor_x, dtype=np.float32),
    }


def train_blocks(blocks, seed, epochs, device):
    """Train with hidden state reset at each source subject boundary."""
    torch.manual_seed(seed)
    model = CausalCNNGRU(20, 32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for block in blocks:
            x, y, weights = block
            positive = max(float(np.sum(y == 0) / max(np.sum(y == 1), 1)), 1.0)
            pos_weight = torch.tensor(positive, device=device)
            hidden = None
            for start in range(0, len(x), 100):
                stop = min(start + 100, len(x))
                values = torch.from_numpy(x[start:stop]).to(device)
                target = torch.from_numpy(y[start:stop]).to(device)
                weight = torch.from_numpy(weights[start:stop]).to(device)
                opt.zero_grad(set_to_none=True)
                logits, hidden = model(values[None], hidden)
                if isinstance(hidden, torch.Tensor):
                    hidden = hidden.detach()
                else:
                    hidden = tuple(value.detach() for value in hidden)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[0], target, weight=weight, pos_weight=pos_weight,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
    return model


def tune(z_values):
    return tune_source_operating_points(
        [calibration_sequence(np.asarray(values, float)) for values in z_values],
        "ma_dwell", list(TARGETS), STEP, ma_seconds=MA_SECONDS,
        dwell_seconds=DWELL_SECONDS, refractory_seconds=REFRACTORY_SECONDS,
        grid_size=160,
    )


def evaluate_one(template, raw, anchor, transferred_thresholds, shifts):
    initial_mean = float(np.mean(anchor))
    initial_var = float(np.var(anchor, ddof=1))
    allowed = np.ones(len(raw), dtype=bool)
    z = ewma_z(raw, initial_mean, initial_var, allowed)
    ma = ma_trace(z, n=int(round(MA_SECONDS / STEP)))
    offsets = np.random.default_rng(88000 + len(raw)).integers(1, len(raw), shifts)
    state = np.zeros(len(raw), dtype=bool)
    for start, stop in template.command_intervals:
        state |= (template.times >= start) & (template.times <= stop)
    idle = ~state
    onsets = np.asarray(template.command_intervals)[:, 0]
    gt = np.asarray(template.command_intervals)

    def at(threshold):
        alarms = alarms_from_ma(
            ma, float(threshold), allowed,
            dwell=int(round(DWELL_SECONDS / STEP)),
            refractory=int(round(REFRACTORY_SECONDS / STEP)),
        )
        conventional = evaluate_alarms(template.times, alarms, template.idle_intervals, template.command_intervals)
        observed, delay = onset_rates(template.times, alarms, template.command_intervals)
        null = []
        for offset in offsets:
            shifted, _ = onset_rates(template.times, np.roll(alarms, int(offset)), template.command_intervals)
            null.append(shifted["1.0"])
        null_value = float(np.median(null))
        pred = intervals_from_alarm(template.times, alarms, ma, float(threshold), allowed)
        return {
            "actual_fp_per_min": float(conventional.fp_per_minute),
            "raw_tpr_1s": float(observed["1.0"]),
            "null_tpr_1s": null_value,
            "excess_1s": float(observed["1.0"] - null_value),
            "latency_median_seconds": None if not np.isfinite(delay) else float(delay),
            "coverage": interval_metrics(pred, gt, .1, .1)["coverage"],
            "temporal_iou": interval_metrics(pred, gt, .1, .1)["temporal_iou"],
            "strict_f1": interval_metrics(pred, gt, .7, .7)["intersection_f1"],
        }

    return {str(target): at(point.threshold) for target, point in transferred_thresholds.items()}


def run_target(subject, session, prepared, source_prepared, seeds, epochs, device, shifts):
    blocks = [(item["train_x"], item["train_y"], item["train_w"]) for item in source_prepared]
    target_predictions = []
    target_anchor_predictions = []
    source_anchor_predictions = [[] for _ in source_prepared]
    for seed in seeds:
        model = train_blocks(blocks, seed, epochs, device)
        target_predictions.append(predict(model, prepared["eval_x"], device))
        target_anchor_predictions.append(predict(model, prepared["anchor_x"], device))
        for index, item in enumerate(source_prepared):
            source_anchor_predictions[index].append(predict(model, item["anchor_x"], device))
    target_raw = np.mean(target_predictions, axis=0)
    target_anchor = np.mean(target_anchor_predictions, axis=0)
    source_anchors = [np.mean(values, axis=0) for values in source_anchor_predictions]
    source_z = [
        ewma_z(values, float(values.mean()), float(np.var(values, ddof=1)), np.ones(len(values), bool))
        for values in source_anchors
    ]
    transferred = tune(source_z)
    sequence = ScoredSequence(
        prepared["template"].times, target_raw,
        prepared["template"].idle_intervals, prepared["template"].command_intervals,
    )
    # Diagnostic common-FP thresholds are tuned on the target evaluation stream
    # after target-rest EWMA initialization; transferred thresholds use source
    # anchors only.
    target_z = ewma_z(
        target_raw, float(target_anchor.mean()), float(np.var(target_anchor, ddof=1)),
        np.ones(len(target_raw), bool),
    )
    common_sequence = ScoredSequence(
        prepared["template"].times, target_z,
        prepared["template"].idle_intervals, prepared["template"].command_intervals,
    )
    common = tune_source_operating_points(
        [common_sequence], "ma_dwell", list(TARGETS), STEP,
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, grid_size=160,
    )
    return {
        "subject": subject, "session": session,
        "transferred": evaluate_one(prepared["template"], target_raw, target_anchor, transferred, shifts),
        "common_fp": evaluate_one(prepared["template"], target_raw, target_anchor, common, shifts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--source-subjects", type=int, nargs="+", default=None,
                        help="Optional smoke-test source pool; default is all 54 except target")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260911])
    parser.add_argument("--max-source-windows", type=int, default=100,
                        help="Balanced windows per source subject; 0 uses all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shifts", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_loso_arm_a_pilot.json"))
    args = parser.parse_args()
    data_root = Path("data/raw/lee2019_mi")
    cache_root = Path("data/interim/lee2019_mil_raw")
    # Prepare each source subject once per session.  Keeping only the training
    # block and anchor avoids repeating expensive MAT/filter/window work for
    # every leave-one-subject-out target.
    source_pool = {}
    pool_ids = args.source_subjects or SUBJECTS
    for session in args.sessions:
        source_pool[session] = {}
        for source in pool_ids:
            item = prepare_subject(data_root, cache_root, source, session, args.max_source_windows)
            source_pool[session][source] = {
                "train_x": item["train_x"], "train_y": item["train_y"],
                "train_w": item["train_w"], "anchor_x": item["anchor_x"],
            }
            print(f"prepared source S{session} sub-{source:02d}", flush=True)
    partial = args.output.with_suffix(".partial.json")
    rows = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else []
    done = {(int(row["session"]), int(row["subject"])) for row in rows}
    for session in args.sessions:
        for target in args.subjects:
            if (session, target) in done:
                print(f"resume-skip S{session} target={target:02d}", flush=True)
                continue
            target_prepared = prepare_subject(data_root, cache_root, target, session, 0)
            source_ids = args.source_subjects or SUBJECTS
            sources = [source_pool[session][source] for source in source_ids if source != target]
            rows.append(run_target(target, session, target_prepared, sources, tuple(args.seeds), args.epochs, torch.device(args.device), args.shifts))
            partial.write_text(json.dumps(rows), encoding="utf-8")
            print(f"LOSO CNN-GRU S{session} target={target:02d}", flush=True)
    payload = {"protocol": {"model": "source-pooled CNN-GRU LOSO", "target_rest": "pre-train rest anchor only", "decision": "EWMA H30 + MA .5 s + dwell .3 s + refractory 1 s", "seeds": list(args.seeds), "source_sampling": f"balanced max {args.max_source_windows} windows per source subject" if args.max_source_windows else "all source windows", "claim_limit": "Lee2019 cue-relative pseudo-uncued replay"}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
