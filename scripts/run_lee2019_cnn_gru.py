"""Causal filter-bank CNN-GRU smoke/validation on rest-injected streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path, load_lee2019_run
from scripts.run_fbcnet_validation import filter_banks, normalize, windows
from scripts.run_lee2019_stateful_gru import (
    SEEDS,
    build_no_guard_stream,
    evaluate,
)

SFREQ = 250
WINDOW = 250
STEP = 0.1


class CausalCNNGRU(nn.Module):
    """EEGNet-like local encoder followed by a continuously stateful GRU."""

    def __init__(self, channels: int = 20, hidden: int = 32):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(3, 12, (1, 25), padding=(0, 0), groups=3, bias=False),
            nn.BatchNorm2d(12),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(12, 24, (channels, 1), groups=12, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.gru = nn.GRU(24, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.spatial(self.temporal(values)))

    def forward(self, values: torch.Tensor, hidden=None):
        # values: batch=1, sequence, band, channel, sample
        batch, steps = values.shape[:2]
        encoded = self.encode(values.reshape(-1, *values.shape[2:]))
        encoded = encoded.reshape(batch, steps, -1)
        state, hidden = self.gru(encoded, hidden)
        return self.output(state).squeeze(-1), hidden


def make_windows(signal: np.ndarray, ends: np.ndarray) -> np.ndarray:
    ends = np.asarray(ends, dtype=int)
    ends = ends[(ends >= WINDOW) & (ends <= signal.shape[-1])]
    return windows(signal, ends)


def rest_windows(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.arange(3 * SFREQ, signal.shape[-1] + 1, int(STEP * SFREQ))
    values = make_windows(signal, ends)
    midpoint = signal.shape[-1] / 2
    return values[ends <= midpoint], values[ends > midpoint]


def train_model(x, y, weights, seed, epochs, device, model_class=CausalCNNGRU):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = model_class(x.shape[2], 32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    positive = max(float(np.sum(y == 0) / max(np.sum(y == 1), 1)), 1.0)
    pos_weight = torch.tensor(positive, device=device)
    ty = torch.from_numpy(y).to(device)
    tw = torch.from_numpy(weights).to(device)
    chunk = 100
    for _ in range(epochs):
        model.train()
        hidden = None
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            values = torch.from_numpy(x[start:stop]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, hidden = model(values[None], hidden)
            if isinstance(hidden, torch.Tensor):
                hidden = hidden.detach()
            else:
                hidden = tuple(value.detach() for value in hidden)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[0], ty[start:stop], weight=tw[start:stop],
                pos_weight=pos_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    hidden = None
    output = []
    for start in range(0, len(x), 100):
        values = torch.from_numpy(x[start:start + 100]).to(device)
        logits, hidden = model(values[None], hidden)
        output.append(logits[0].cpu().numpy())
    return np.concatenate(output)


def analyze(
    data_root, cache_root, subject, session, epochs, seeds, device, shifts,
    score_cache_root=None, model_class=CausalCNNGRU,
):
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
    calibration = pre[..., int(2 * SFREQ):]
    mean = calibration.mean(axis=-1, keepdims=True)
    std = calibration.std(axis=-1, keepdims=True).clip(1e-6)
    rests = [pre] + [(values - mean) / std for values in raw_rests[1:]]

    with np.load(
        cache_root / f"session{session}" / f"sub-{subject:02d}_mil_raw.npz"
    ) as cache:
        command_trials = []
        for start, stop in cache["command_intervals"]:
            # A complete one-second support must lie after the registered
            # command onset; therefore the first score ends at onset + 1 s.
            ends = np.arange(
                int(round((start + 1.0) * SFREQ)),
                int(round(stop * SFREQ)),
                int(STEP * SFREQ),
            )
            command_trials.append(make_windows(signal, ends))
        iti_trials = []
        for trial in range(50):
            mask = cache["iti_instance_trials"] == trial
            if np.any(mask):
                iti_trials.append(
                    make_windows(signal, cache["iti_end_samples"][mask])
                )

    rest_split = [rest_windows(value) for value in rests]
    train_idle = iti_trials + [rest_split[0][0]]
    train_x, y, weights, _ = build_no_guard_stream(
        train_idle, command_trials[:50],
        20260700 + 100 * session + subject,
    )
    evaluation_rest = [split[1] for split in rest_split]
    eval_x, _, _, template = build_no_guard_stream(
        evaluation_rest, command_trials[50:],
        20260728 + 100 * session + subject,
    )
    calibration_x = [split[0] for split in rest_split[1:]]

    seed_eval, seed_cal, seed_session_anchor = [], [], []
    for seed in seeds:
        model = train_model(
            train_x, y, weights, seed, epochs, device, model_class=model_class
        )
        seed_eval.append(predict(model, eval_x, device))
        seed_cal.append([predict(model, values, device) for values in calibration_x])
        # Chronologically valid session-start anchor: the untouched second
        # half of pre-train rest.  It is scored with a fresh recurrent state
        # because it precedes the task replay in a deployment session.
        seed_session_anchor.append(predict(model, rest_split[0][1], device))
    scores = np.mean(seed_eval, axis=0)
    calibrations = [
        np.mean([prediction[index] for prediction in seed_cal], axis=0)
        for index in range(len(calibration_x))
    ]
    from uncued_mi.operating_point import ScoredSequence
    sequence = ScoredSequence(
        template.times, scores, template.idle_intervals,
        template.command_intervals,
    )
    if score_cache_root is not None:
        target = (
            Path(score_cache_root) / f"session{session}"
            / f"sub-{subject:02d}.npz"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "subject": np.asarray(subject, dtype=np.int16),
            "session": np.asarray(session, dtype=np.int8),
            "times": np.asarray(template.times, dtype=np.float64),
            "scores": np.asarray(scores, dtype=np.float32),
            "seed_scores": np.asarray(seed_eval, dtype=np.float32),
            "session_anchor_scores": np.asarray(
                np.mean(seed_session_anchor, axis=0), dtype=np.float32
            ),
            "seed_session_anchor_scores": np.asarray(
                seed_session_anchor, dtype=np.float32
            ),
            "idle_intervals": np.asarray(template.idle_intervals, dtype=np.float64),
            "command_intervals": np.asarray(
                template.command_intervals, dtype=np.float64
            ),
            "seeds": np.asarray(seeds, dtype=np.int64),
            "step_seconds": np.asarray(STEP, dtype=np.float64),
        }
        for index, values in enumerate(calibrations):
            payload[f"calibration_{index}"] = np.asarray(values, dtype=np.float32)
        np.savez_compressed(target, **payload)
    rng = np.random.default_rng(88000 + 100 * session + subject)
    offsets = rng.integers(1, len(scores), size=shifts)
    seed_metrics = []
    for index, seed in enumerate(seeds):
        seed_sequence = ScoredSequence(
            template.times, seed_eval[index], template.idle_intervals,
            template.command_intervals,
        )
        seed_metrics.append({
            "seed": seed,
            "metrics": evaluate(
                seed_sequence, seed_cal[index], offsets
            ),
        })
    return {
        "subject": subject,
        "session": session,
        "metrics": evaluate(sequence, calibrations, offsets),
        "seed_metrics": seed_metrics,
        "train_windows": len(train_x),
        "evaluation_windows": len(eval_x),
        "evaluation_trials": len(template.command_intervals),
    }


def bootstrap_median(values, seed, repeats=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, sessions, repeats):
    output = []
    for session in sessions:
        selected = [row for row in rows if row["session"] == session]
        item = {"session": session, "subjects": len(selected), "modes": {}}
        for mode in ("common_fp", "transferred"):
            excess = [row["metrics"][mode]["excess_1s"] for row in selected]
            fp = [row["metrics"][mode]["fp"] for row in selected]
            raw = [row["metrics"][mode]["raw_tpr_1s"] for row in selected]
            delay = [
                row["metrics"][mode]["delay"] for row in selected
                if np.isfinite(row["metrics"][mode]["delay"])
            ]
            item["modes"][mode] = {
                "fp_median": float(np.median(fp)),
                "raw_tpr_1s_median": float(np.median(raw)),
                "excess_1s_median": float(np.median(excess)),
                "excess_1s_ci": bootstrap_median(
                    excess, 81000 + session, repeats
                ),
                "delay_median": float(np.median(delay)) if delay else None,
            }
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--score-cache-root", type=Path, default=None,
        help="Optional directory for reusable evaluation/calibration score streams.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_subject03_s1.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    partial = args.output.with_suffix(".partial.json")
    rows = []
    if partial.exists():
        rows = json.loads(partial.read_text(encoding="utf-8"))
    for session in args.sessions:
        for subject in args.subjects:
            if any(
                row["session"] == session and row["subject"] == subject
                for row in rows
            ):
                print(
                    f"resume-skip session={session} subject={subject}",
                    flush=True,
                )
                continue
            rows.append(analyze(
                args.data_root, args.cache_root, subject, session,
                args.epochs, tuple(args.seeds), device, args.shifts,
                args.score_cache_root,
            ))
            partial.write_text(json.dumps(rows), encoding="utf-8")
            print(f"cnn-gru session={session} subject={subject}", flush=True)
    payload = {
        "protocol": {
            "model": "causal filter-bank CNN -> stateful GRU32",
            "window_seconds": 1.0,
            "step_seconds": STEP,
            "epochs": args.epochs,
            "seeds": list(args.seeds),
            "continuous_state": True,
            "trial_boundary_reset": False,
            "target": "trial detection within 1 second",
            "device": args.device,
        },
        "summary": summarize(rows, args.sessions, args.bootstrap),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
