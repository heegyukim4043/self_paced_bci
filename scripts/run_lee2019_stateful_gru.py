"""Stateful causal GRU on compressed Riemannian features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import instantaneous
from uncued_mi.hsmm import GaussianEmission, causal_two_state_hsmm
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points
from uncued_mi.rest_injected import random_partition
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_sequential_onset import (
    RESTS, STEP, calibration_sequence, fit_score, onset_rates, reference,
)

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
SEEDS = (20260729, 20260730, 20260731)


class StatefulGRU(nn.Module):
    def __init__(self, inputs=17, hidden=32):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, values, hidden=None):
        encoded, hidden = self.gru(values, hidden)
        return self.output(encoded).squeeze(-1), hidden


def build_no_guard_stream(rest_blocks, command_trials, seed):
    rng = np.random.default_rng(seed)
    partitions = []
    for block_index, values in enumerate(rest_blocks):
        # The registered pseudo-online stream is rest-part -> command trial.
        # One-to-three second idle contexts provide enough disjoint chunks to
        # evaluate every held-out command without duplicating rest samples.
        minimum = min(10, len(values))
        maximum = min(30, len(values))
        if minimum == 0:
            continue
        for start, stop in random_partition(
            len(values), minimum, maximum, rng
        ):
            partitions.append((block_index, start, stop))
    rng.shuffle(partitions)
    order = rng.permutation(len(command_trials))
    count = min(len(partitions), len(order))
    values, labels, weights = [], [], []
    idle, command = [], []

    def now():
        return len(values) * STEP

    for (block, start, stop), trial in zip(
        partitions[:count], order[:count]
    ):
        rest = rest_blocks[block][start:stop]
        idle_start = now()
        values.extend(rest.tolist())
        labels.extend([0.0] * len(rest))
        weights.extend([1.0] * len(rest))
        idle.append((idle_start, now()))

        event = command_trials[int(trial)]
        command_start = now()
        values.extend(event.tolist())
        labels.extend([1.0] * len(event))
        weights.extend([
            2.0 if index < 10 else 1.0 for index in range(len(event))
        ])
        command.append((command_start, now()))
    matrix = np.asarray(values, dtype=np.float32)
    return (
        matrix,
        np.asarray(labels, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        ScoredSequence(
            np.arange(len(matrix)) * STEP,
            np.zeros(len(matrix)),
            np.asarray(idle), np.asarray(command),
        ),
    )


def fit_representation(command, iti, rest, command_trials, iti_trials):
    fit = np.r_[
        command[command_trials < 50],
        iti[iti_trials < 50],
        rest,
    ]
    scaler = StandardScaler().fit(fit)
    pca = PCA(n_components=16, whiten=True, random_state=20260729).fit(
        scaler.transform(fit)
    )
    return scaler, pca


def transform(values, scaler, pca, gate_scaler, gate_model):
    compressed = pca.transform(scaler.transform(values))
    gate = gate_model.decision_function(
        gate_scaler.transform(values)
    )[:, None]
    return np.c_[compressed, gate].astype(np.float32)


def train_gru(x, y, weights, seed, epochs, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = StatefulGRU(x.shape[1], 32).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    positive = max(float(np.sum(y == 0) / max(np.sum(y == 1), 1)), 1.0)
    pos_weight = torch.tensor(positive, device=device)
    chunk = 200
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    tw = torch.from_numpy(weights).to(device)
    for _ in range(epochs):
        hidden = None
        model.train()
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            optimizer.zero_grad(set_to_none=True)
            logits, hidden = model(tx[None, start:stop], hidden)
            hidden = hidden.detach()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[0], ty[start:stop],
                weight=tw[start:stop], pos_weight=pos_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, values, device):
    model.eval()
    tensor = torch.from_numpy(values).to(device)
    output, _ = model(tensor[None])
    return output[0].cpu().numpy()


def tune_threshold(sequences):
    return tune_source_operating_points(
        sequences, "threshold", [2.0], STEP,
        refractory_seconds=1.0, grid_size=200,
    )[2.0]


def evaluate(sequence, calibration, offsets):
    transferred = tune_threshold([
        calibration_sequence(values) for values in calibration
    ])
    common = tune_threshold([sequence])

    def at(point):
        alarms = instantaneous(
            sequence.scores, point.threshold, STEP, 1.0
        ).alarms
        conventional = evaluate_alarms(
            sequence.times, alarms, sequence.idle_intervals,
            sequence.command_intervals,
        )
        observed, delay = onset_rates(
            sequence.times, alarms, sequence.command_intervals
        )
        null = []
        for offset in offsets:
            shifted, _ = onset_rates(
                sequence.times, np.roll(alarms, int(offset)),
                sequence.command_intervals,
            )
            null.append(shifted["1.0"])
        null_value = float(np.median(null))
        return {
            "fp": conventional.fp_per_minute,
            "raw_tpr_1s": observed["1.0"],
            "null_tpr_1s": null_value,
            "excess_1s": float(observed["1.0"] - null_value),
            "delay": delay,
        }
    return {"transferred": at(transferred), "common_fp": at(common)}


def hsmm_scores(raw_scores, emission):
    return causal_two_state_hsmm(
        raw_scores, emission,
        minimum_mi_steps=3, expected_mi_steps=10.0,
        maximum_mi_steps=15, entry_probability=0.01,
    )


def analyze(path, subject, session, epochs, device_name, shifts, seeds):
    device = torch.device(device_name)
    with np.load(path, allow_pickle=False) as data:
        pre = data["pre_train_covariance"]
        split = len(pre) // 2
        ref = reference(pre[:split])
        command = tangent_features(data["train_command_covariance"], ref)
        iti = tangent_features(data["train_idle_iti_covariance"], ref)
        rest = {
            name: tangent_features(data[f"{name}_covariance"], ref)
            for name in RESTS
        }
        command_trials = data["train_command_trial"]
        iti_trials = data["train_idle_iti_trial"]
        gate_scaler, gate_model = fit_score(
            command[command_trials < 50], iti[iti_trials < 50],
            rest["pre_train"][:split],
        )
        scaler, pca = fit_representation(
            command, iti, rest["pre_train"][:split],
            command_trials, iti_trials,
        )
        command_z = transform(
            command, scaler, pca, gate_scaler, gate_model
        )
        iti_z = transform(iti, scaler, pca, gate_scaler, gate_model)
        rest_z = {
            name: transform(value, scaler, pca, gate_scaler, gate_model)
            for name, value in rest.items()
        }
        train_commands = [
            command_z[command_trials == trial] for trial in range(50)
        ]
        train_idle = [
            iti_z[iti_trials == trial]
            for trial in range(50) if np.any(iti_trials == trial)
        ]
        train_idle.append(rest_z["pre_train"][:split])
        x, y, weights, _ = build_no_guard_stream(
            train_idle, train_commands,
            20260700 + 100 * session + subject,
        )
        evaluation_rest = [
            value[len(value)//2:] for value in rest_z.values()
        ]
        evaluation_commands = [
            command_z[command_trials == trial] for trial in range(50, 100)
        ]
        eval_x, _, _, template = build_no_guard_stream(
            evaluation_rest, evaluation_commands,
            20260728 + 100 * session + subject,
        )
        calibration_z = [
            rest_z[name][:len(rest_z[name])//2]
            for name in RESTS if name != "pre_train"
        ]

        raw_command_score = gate_model.decision_function(
            gate_scaler.transform(command)
        )
        raw_iti_score = gate_model.decision_function(
            gate_scaler.transform(iti)
        )
        raw_rest_score = {
            name: gate_model.decision_function(
                gate_scaler.transform(value)
            ) for name, value in rest.items()
        }
        raw_eval_rest = [
            value[len(value)//2:] for value in raw_rest_score.values()
        ]
        raw_eval_commands = [
            raw_command_score[command_trials == trial]
            for trial in range(50, 100)
        ]
        raw_eval, _, _, raw_template = build_no_guard_stream(
            raw_eval_rest, raw_eval_commands,
            20260728 + 100 * session + subject,
        )
        emission = GaussianEmission.fit(
            np.r_[raw_iti_score[iti_trials < 50],
                  raw_rest_score["pre_train"][:split]],
            raw_command_score[command_trials < 50],
        )
        raw_calibration = [
            raw_rest_score[name][:len(raw_rest_score[name])//2]
            for name in RESTS if name != "pre_train"
        ]

    rng = np.random.default_rng(99700 + 100 * session + subject)
    offsets = rng.integers(1, len(template.times), size=shifts)
    seed_results, calibration_predictions = [], []
    for seed in seeds:
        model = train_gru(x, y, weights, seed, epochs, device)
        seed_results.append(predict(model, eval_x, device))
        calibration_predictions.append([
            predict(model, values, device) for values in calibration_z
        ])
    gru_scores = np.mean(seed_results, axis=0)
    gru_calibration = [
        np.mean([values[index] for values in calibration_predictions], axis=0)
        for index in range(len(calibration_z))
    ]
    gru_sequence = ScoredSequence(
        template.times, gru_scores,
        template.idle_intervals, template.command_intervals,
    )

    hsmm_sequence = ScoredSequence(
        raw_template.times, hsmm_scores(raw_eval, emission),
        raw_template.idle_intervals, raw_template.command_intervals,
    )
    hsmm_calibration = [
        hsmm_scores(values, emission) for values in raw_calibration
    ]
    return {
        "subject": subject, "session": session,
        "gru": evaluate(gru_sequence, gru_calibration, offsets),
        "hsmm": evaluate(hsmm_sequence, hsmm_calibration, offsets),
        "train_windows": len(x), "eval_windows": len(eval_x),
        "evaluation_trials": len(template.command_intervals),
        "evaluation_idle_seconds": float(sum(
            stop - start for start, stop in template.idle_intervals
        )),
    }


def bootstrap(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, session, repeats):
    selected = [row for row in rows if row["session"] == session]
    result = {}
    for mode in ("transferred", "common_fp"):
        gru = np.asarray([row["gru"][mode]["excess_1s"] for row in selected])
        hsmm = np.asarray([row["hsmm"][mode]["excess_1s"] for row in selected])
        delta = gru - hsmm
        result[mode] = {
            "gru_fp_median": float(np.median([
                row["gru"][mode]["fp"] for row in selected
            ])),
            "gru_raw_tpr_median": float(np.median([
                row["gru"][mode]["raw_tpr_1s"] for row in selected
            ])),
            "gru_excess_median": float(np.median(gru)),
            "gru_excess_ci": bootstrap(gru, 100 + session, repeats),
            "hsmm_fp_median": float(np.median([
                row["hsmm"][mode]["fp"] for row in selected
            ])),
            "hsmm_excess_median": float(np.median(hsmm)),
            "gru_minus_hsmm_median": float(np.median(delta)),
            "gru_minus_hsmm_ci": bootstrap(delta, 200 + session, repeats),
        }
    return {"session": session, "subjects": len(selected), "modes": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_four_rest")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_stateful_gru_subset20.json"),
    )
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze(
                args.cache_root / f"session{session}"
                / f"sub-{subject:02d}.npz",
                subject, session, args.epochs, args.device, args.shifts,
                tuple(args.seeds),
            ))
            print(
                f"stateful-gru session={session} subject={subject}",
                flush=True,
            )
    summary = [
        summarize(rows, session, args.bootstrap) for session in args.sessions
    ]
    payload = {
        "protocol": {
            "model": "PCA16+tangent-gate score -> GRU32",
            "epochs": args.epochs, "seeds": tuple(args.seeds),
            "continuous_state": True, "trial_boundary_reset": False,
            "guard": False, "target": "trial detection within 1 second",
            "idle_context_seconds": [1.0, 3.0],
            "device": args.device,
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
