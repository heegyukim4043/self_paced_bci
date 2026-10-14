"""Run the preregistered session-1 MIL internal-validation grid."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import (
    SEEDS,
    CompactCausalEncoder,
    ValidationCandidate,
    mil_loss,
    select_validation_candidate,
)
from uncued_mi.operating_point import (
    ScoredSequence,
    apply_decision,
    tune_source_operating_points,
)

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
TAUS = (0.25, 2.0)
FALSE_ALARM_WEIGHTS = (0.5, 2.0)
CONTINUITY_WEIGHTS = (0.0, 0.05)
SFREQ = 250
WINDOW = 250
DECISION_STEP = 0.5


def extract_windows(signal: np.ndarray, ends: np.ndarray) -> torch.Tensor:
    ends = np.asarray(ends, dtype=np.int64)
    indices = ends[:, None] - WINDOW + np.arange(WINDOW)[None, :]
    values = signal[indices]  # windows, samples, channels
    return torch.from_numpy(np.transpose(values, (0, 2, 1)).copy()).float()


def positive_bags(data, low: int, high: int) -> list[torch.Tensor]:
    result = []
    signal = np.asarray(data["signal"])
    for trial in range(low, high):
        size = int(data["positive_bag_sizes"][trial])
        ends = data["positive_end_samples"][trial, :size]
        result.append(extract_windows(signal, ends))
    return result


def score_windows(
    model: torch.nn.Module,
    signal: np.ndarray,
    ends: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    windows = extract_windows(signal, ends)
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            values.append(model(windows[start : start + batch_size]).cpu().numpy())
    return np.concatenate(values)


def interval_subset(data, kind: str, low: int, high: int) -> np.ndarray:
    trials = data[f"{kind}_trials"]
    return data[f"{kind}_intervals"][(trials >= low) & (trials < high)]


def evaluate_range(
    model: torch.nn.Module,
    data,
    low: int,
    high: int,
    shifts: int,
    seed: int,
) -> dict:
    signal = np.asarray(data["signal"])
    decision_ends = np.arange(
        int(3.0 * SFREQ), len(signal) + 1, int(DECISION_STEP * SFREQ)
    )
    scores = score_windows(model, signal, decision_ends)
    times = decision_ends / SFREQ
    idle = interval_subset(data, "iti", low, high)
    command = data["command_intervals"][low:high]
    sequence = ScoredSequence(times, scores, idle, command)
    point = tune_source_operating_points(
        [sequence],
        "ma_dwell",
        [2.0],
        DECISION_STEP,
        ma_seconds=0.5,
        dwell_seconds=0.3,
        refractory_seconds=1.0,
        grid_size=160,
    )[2.0]
    alarms = apply_decision(
        sequence,
        "ma_dwell",
        point.threshold,
        DECISION_STEP,
        ma_seconds=0.5,
        dwell_seconds=0.3,
        refractory_seconds=1.0,
    ).alarms
    observed = evaluate_alarms(times, alarms, idle, command)
    rng = np.random.default_rng(seed)
    null = np.empty(shifts)
    for index, offset in enumerate(rng.integers(1, len(alarms), size=shifts)):
        null[index] = evaluate_alarms(
            times, np.roll(alarms, int(offset)), idle, command
        ).true_positive_rate
    null_median = float(np.median(null))
    return {
        "threshold": point.threshold,
        "fp_per_minute": observed.fp_per_minute,
        "observed_tpr": observed.true_positive_rate,
        "null_tpr_median": null_median,
        "excess_tpr": float(observed.true_positive_rate - null_median),
    }


def train_one(
    data,
    tau: float,
    false_alarm_weight: float,
    continuity_weight: float,
    seed: int,
    epochs: int,
    train_stop: int = 40,
) -> tuple[CompactCausalEncoder, dict]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    model = CompactCausalEncoder(channels=20)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    bags = positive_bags(data, 0, train_stop)
    pre_ends = data["pre_rest_fit_end_samples"]
    if train_stop == 50:
        pre_ends = np.r_[pre_ends, data["pre_rest_validation_end_samples"]]
    pre = extract_windows(
        np.asarray(data["pre_rest"]), pre_ends
    )
    iti_trials = data["iti_instance_trials"]
    iti = extract_windows(
        np.asarray(data["signal"]),
        data["iti_end_samples"][iti_trials < train_stop],
    )
    final_parts = None
    for _ in range(epochs):
        order = np_rng.permutation(len(bags))
        epoch_parts = []
        model.train()
        for start in range(0, len(order), 8):
            chosen = order[start : start + 8]
            positive_logits = [model(bags[index]) for index in chosen]
            rest_index = np_rng.integers(0, len(pre), size=64)
            iti_index = np_rng.integers(0, len(iti), size=64)
            rest_logits = model(pre[rest_index])
            iti_logits = model(iti[iti_index])
            loss, parts = mil_loss(
                positive_logits,
                rest_logits,
                iti_logits,
                tau=tau,
                false_alarm_weight=false_alarm_weight,
                continuity_weight=continuity_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_parts.append({key: float(value) for key, value in parts.items()})
        final_parts = {
            key: float(np.mean([value[key] for value in epoch_parts]))
            for key in epoch_parts[0]
        }
    assert final_parts is not None
    final_parts["subtype_equal_weight_assertion"] = bool(
        np.isclose(
            final_parts["false_alarm"],
            0.5 * (final_parts["rest"] + final_parts["iti"]),
            atol=1e-6,
        )
    )
    if not final_parts["subtype_equal_weight_assertion"]:
        raise AssertionError("False-alarm subtype weighting changed")
    return model, final_parts


def candidate_key(tau: float, false_alarm: float, continuity: float) -> str:
    return f"tau={tau}|fa={false_alarm}|continuity={continuity}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/interim/lee2019_mil_raw/session1"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--shifts", type=int, default=200)
    parser.add_argument(
        "--comparator",
        type=Path,
        default=Path("data/processed/mil_riemannian_validation_comparator.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mil_validation.json"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    grid = list(product(TAUS, FALSE_ALARM_WEIGHTS, CONTINUITY_WEIGHTS))
    if args.smoke:
        grid = grid[:1]
        args.seeds = args.seeds[:1]
        args.subjects = args.subjects[:1]
        args.epochs = 1
        args.shifts = 20
    checkpoint = args.output.with_suffix(".checkpoint.json")
    rows = (
        json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint.exists() and not args.smoke
        else []
    )
    completed = {
        (row["subject"], row["seed"], row["candidate"]) for row in rows
    }

    for subject in args.subjects:
        path = args.cache_root / f"sub-{subject:02d}_mil_raw.npz"
        with np.load(path, allow_pickle=False) as data:
            for tau, false_alarm, continuity in grid:
                key = candidate_key(tau, false_alarm, continuity)
                for seed in args.seeds:
                    if (subject, seed, key) in completed:
                        continue
                    model, loss_parts = train_one(
                        data,
                        tau,
                        false_alarm,
                        continuity,
                        seed,
                        args.epochs,
                    )
                    fit = evaluate_range(
                        model, data, 0, 40, args.shifts, seed + subject
                    )
                    validation = evaluate_range(
                        model, data, 40, 50, args.shifts, seed + subject + 1000
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "candidate": key,
                            "tau": tau,
                            "false_alarm_weight": false_alarm,
                            "continuity_weight": continuity,
                            "epochs": args.epochs,
                            "early_stopping": False,
                            "final_epoch_loss": loss_parts,
                            "fit": fit,
                            "validation": validation,
                            "fit_minus_validation_excess": (
                                fit["excess_tpr"] - validation["excess_tpr"]
                            ),
                        }
                    )
                    if not args.smoke:
                        checkpoint.write_text(
                            json.dumps(rows, indent=2), encoding="utf-8"
                        )
                    print(
                        f"subject={subject} seed={seed} candidate={key} "
                        f"fit_excess={fit['excess_tpr']:.3f} "
                        f"val_excess={validation['excess_tpr']:.3f}",
                        flush=True,
                    )

    candidates = []
    candidate_audit = []
    for tau, false_alarm, continuity in grid:
        key = candidate_key(tau, false_alarm, continuity)
        selected_rows = [row for row in rows if row["candidate"] == key]
        subject_values = []
        for subject in args.subjects:
            subject_rows = [
                row for row in selected_rows if row["subject"] == subject
            ]
            if not subject_rows:
                continue
            subject_values.append(
                {
                    "subject": subject,
                    "fp": float(np.mean([row["validation"]["fp_per_minute"] for row in subject_rows])),
                    "excess": float(np.mean([row["validation"]["excess_tpr"] for row in subject_rows])),
                    "gap": float(np.mean([row["fit_minus_validation_excess"] for row in subject_rows])),
                }
            )
        candidate = ValidationCandidate(
            tau=tau,
            false_alarm_weight=false_alarm,
            continuity_weight=continuity,
            median_fp_per_minute=float(np.median([row["fp"] for row in subject_values])),
            median_excess_tpr=float(np.median([row["excess"] for row in subject_values])),
        )
        candidates.append(candidate)
        candidate_audit.append(
            {
                **asdict(candidate),
                "candidate": key,
                "subjects": len(subject_values),
                "median_fit_minus_validation_excess": float(
                    np.median([row["gap"] for row in subject_values])
                ),
                "admissible": 1.5 <= candidate.median_fp_per_minute <= 2.5,
            }
        )
    selected, calibration_failure = select_validation_candidate(candidates)
    admissible = [item for item in candidates if 1.5 <= item.median_fp_per_minute <= 2.5]
    best = max((item.median_excess_tpr for item in admissible), default=float("nan"))
    tied = [
        item for item in admissible if best - item.median_excess_tpr < 0.01
    ]

    comparator = json.loads(args.comparator.read_text(encoding="utf-8"))
    comparator_s1 = next(
        row["excess_tpr_median"]
        for row in comparator["summary"]
        if row["session"] == 1
    )
    result = {
        "protocol": {
            "session": 1,
            "fit_trials": [0, 40],
            "validation_trials": [40, 50],
            "epochs": args.epochs,
            "early_stopping": False,
            "seeds": list(args.seeds),
            "shifts": args.shifts,
            "grid": {
                "tau": list(TAUS),
                "false_alarm_weight": list(FALSE_ALARM_WEIGHTS),
                "continuity_weight": list(CONTINUITY_WEIGHTS),
            },
            "penalty": "mean(rest BCE, ITI BCE), equal subtype weighting",
        },
        "selection": {
            "selected": asdict(selected),
            "calibration_failure": calibration_failure,
            "admissible_candidates": len(admissible),
            "tie_break_candidate_count": len(tied),
            "tie_break_invoked": len(tied) > 1,
            "validation_riemannian_excess_median": comparator_s1,
            "selected_minus_validation_riemannian": (
                selected.median_excess_tpr - comparator_s1
            ),
            "diagnostic_only_not_binding": True,
        },
        "candidate_audit": candidate_audit,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if checkpoint.exists() and not args.smoke:
        checkpoint.unlink()
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
