"""Locked MIL evaluation using the preregistered validation-selected setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_mil_validation import SUBSET20, evaluate_range, train_one
from uncued_mi.mil import SEEDS


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(repeats, len(values)))
    ]
    return [
        float(value)
        for value in np.quantile(np.median(samples, axis=1), [0.025, 0.975])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw")
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("data/processed/mil_validation.json")
    )
    parser.add_argument(
        "--comparator",
        type=Path,
        default=Path("data/processed/mil_riemannian_comparator.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/mil_locked.json")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--shifts", type=int, default=500)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if validation["selection"]["calibration_failure"]:
        raise RuntimeError("Preregistered calibration failure: locked success is forbidden")
    selected = validation["selection"]["selected"]
    comparator = json.loads(args.comparator.read_text(encoding="utf-8"))
    comparator_rows = {
        (row["session"], row["subject"]): row for row in comparator["rows"]
    }

    checkpoint = args.output.with_suffix(".checkpoint.json")
    rows = (
        json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint.exists()
        else []
    )
    completed = {(row["session"], row["subject"], row["seed"]) for row in rows}
    for session in (1, 2):
        for subject in args.subjects:
            path = (
                args.cache_root
                / f"session{session}"
                / f"sub-{subject:02d}_mil_raw.npz"
            )
            with np.load(path, allow_pickle=False) as data:
                for seed in args.seeds:
                    if (session, subject, seed) in completed:
                        continue
                    model, loss_parts = train_one(
                        data,
                        selected["tau"],
                        selected["false_alarm_weight"],
                        selected["continuity_weight"],
                        seed,
                        args.epochs,
                        train_stop=50,
                    )
                    locked = evaluate_range(
                        model,
                        data,
                        50,
                        100,
                        args.shifts,
                        seed + session * 1000 + subject,
                    )
                    rows.append(
                        {
                            "session": session,
                            "subject": subject,
                            "seed": seed,
                            "selected": selected,
                            "epochs": args.epochs,
                            "early_stopping": False,
                            "final_epoch_loss": loss_parts,
                            "locked": locked,
                            "riemannian_excess": comparator_rows[
                                (session, subject)
                            ]["excess_tpr"],
                            "paired_delta": (
                                locked["excess_tpr"]
                                - comparator_rows[(session, subject)]["excess_tpr"]
                            ),
                        }
                    )
                    checkpoint.write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )
                    print(
                        f"session={session} subject={subject} seed={seed} "
                        f"excess={locked['excess_tpr']:.3f}",
                        flush=True,
                    )

    summaries = []
    binding_pass = True
    for session in (1, 2):
        selected_rows = [row for row in rows if row["session"] == session]
        subject_rows = []
        for subject in args.subjects:
            subject_seed_rows = [
                row for row in selected_rows if row["subject"] == subject
            ]
            subject_rows.append(
                {
                    "subject": subject,
                    "mil_excess_seed_mean": float(
                        np.mean(
                            [row["locked"]["excess_tpr"] for row in subject_seed_rows]
                        )
                    ),
                    "paired_delta_seed_mean": float(
                        np.mean([row["paired_delta"] for row in subject_seed_rows])
                    ),
                    "fp_seed_mean": float(
                        np.mean(
                            [row["locked"]["fp_per_minute"] for row in subject_seed_rows]
                        )
                    ),
                }
            )
        excess = np.asarray([row["mil_excess_seed_mean"] for row in subject_rows])
        delta = np.asarray([row["paired_delta_seed_mean"] for row in subject_rows])
        excess_ci = bootstrap_ci(excess, 20260725 + session)
        delta_ci = bootstrap_ci(delta, 20260825 + session)
        session_pass = excess_ci[0] > 0.10 and delta_ci[0] > 0.0
        binding_pass &= session_pass
        seed_sensitivity = []
        for seed in args.seeds:
            seed_rows = [row for row in selected_rows if row["seed"] == seed]
            seed_sensitivity.append(
                {
                    "seed": seed,
                    "median_excess": float(
                        np.median([row["locked"]["excess_tpr"] for row in seed_rows])
                    ),
                    "median_paired_delta": float(
                        np.median([row["paired_delta"] for row in seed_rows])
                    ),
                }
            )
        summaries.append(
            {
                "session": session,
                "subjects": len(subject_rows),
                "median_fp_per_minute": float(
                    np.median([row["fp_seed_mean"] for row in subject_rows])
                ),
                "median_mil_excess": float(np.median(excess)),
                "mil_excess_bootstrap_95_ci": excess_ci,
                "median_paired_delta": float(np.median(delta)),
                "paired_delta_bootstrap_95_ci": delta_ci,
                "binding_session_pass": session_pass,
                "seed_sensitivity": seed_sensitivity,
                "subject_rows": subject_rows,
            }
        )

    result = {
        "protocol": {
            "selected_from_validation": selected,
            "training_trials": [0, 50],
            "locked_trials": [50, 100],
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "early_stopping": False,
            "shifts": args.shifts,
            "comparator": str(args.comparator),
        },
        "binding": {
            "rule": (
                "both sessions: MIL excess CI lower >0.10 and paired delta "
                "vs same-cache Riemannian CI lower >0"
            ),
            "pass": binding_pass,
            "model_search_action": (
                "expand to 54" if binding_pass else "terminate model search"
            ),
        },
        "summary": summaries,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if checkpoint.exists():
        checkpoint.unlink()
    print(json.dumps({"binding": result["binding"], "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
