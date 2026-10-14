"""Evaluate inference-time fusion of frozen CNN-GRU seed score paths.

This is a test-time aggregation audit, not a retraining experiment.  The
per-seed score streams and calibration streams are already present in the
CNN-GRU cache.  Each fusion rule is thresholded independently with the same
locked operating-point evaluator, so max fusion cannot hide its false-alarm
cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_lee2019_cnn_gru import bootstrap_median
from run_lee2019_stateful_gru import ScoredSequence, evaluate


FUSIONS = ("mean", "median", "max")


def fuse(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected seed x time array, got {values.shape}")
    if name == "mean":
        return np.mean(values, axis=0)
    if name == "median":
        return np.median(values, axis=0)
    if name == "max":
        return np.max(values, axis=0)
    raise ValueError(name)


def analyze(path: Path, shifts: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], float)
        seed_scores = np.asarray(data["seed_scores"], float)
        idle = np.asarray(data["idle_intervals"], float)
        command = np.asarray(data["command_intervals"], float)
        # The existing cache stores calibration streams after the three seeds
        # have already been averaged.  Keep that calibration fixed across
        # fusion rules; this avoids silently giving max fusion a different
        # calibration sample and makes the comparison explicitly diagnostic.
        calibration = [
            np.asarray(data[key], float)
            for key in sorted(key for key in data.files if key.startswith("calibration_"))
        ]
        subject = int(np.asarray(data["subject"]).item())
        session = int(np.asarray(data["session"]).item())

    rng = np.random.default_rng(88000 + 100 * session + subject)
    offsets = rng.integers(1, len(times), size=shifts)
    output = {"subject": subject, "session": session, "fusions": {}}
    for name in FUSIONS:
        scores = fuse(seed_scores, name)
        sequence = ScoredSequence(times, scores, idle, command)
        output["fusions"][name] = evaluate(sequence, calibration, offsets)
    return output


def summarize(rows: list[dict], bootstrap: int) -> dict:
    summary = {}
    for session in sorted({row["session"] for row in rows}):
        session_rows = [row for row in rows if row["session"] == session]
        summary[f"session{session}"] = {"n": len(session_rows), "fusions": {}}
        for fusion in FUSIONS:
            metrics = {}
            for mode in ("common_fp", "transferred"):
                values = [row["fusions"][fusion][mode] for row in session_rows]
                excess = [value["excess_1s"] for value in values]
                metrics[mode] = {
                    "fp_median": float(np.median([value["fp"] for value in values])),
                    "raw_tpr_1s_median": float(np.median([value["raw_tpr_1s"] for value in values])),
                    "null_tpr_1s_median": float(np.median([value["null_tpr_1s"] for value in values])),
                    "excess_1s_median": float(np.median(excess)),
                    "excess_1s_ci95": bootstrap_median(excess, 20260911 + session, bootstrap),
                    "delay_median": float(np.median([value["delay"] for value in values if np.isfinite(value["delay"])]))
                    if any(np.isfinite(value["delay"]) for value in values) else None,
                }
            summary[f"session{session}"]["fusions"][fusion] = metrics
        # Paired excess deltas versus the current mean fusion.  This is the
        # promotion-relevant comparison; cohort median differences alone can
        # hide subject-session reversals.
        paired = {}
        for fusion in ("median", "max"):
            paired[fusion] = {}
            for mode in ("common_fp", "transferred"):
                deltas = np.asarray([
                    row["fusions"][fusion][mode]["excess_1s"]
                    - row["fusions"]["mean"][mode]["excess_1s"]
                    for row in session_rows
                ], float)
                paired[fusion][mode] = {
                    "excess_1s_delta_median": float(np.median(deltas)),
                    "excess_1s_delta_ci95": bootstrap_median(
                        deltas, 20560911 + session * 100 + len(fusion), bootstrap
                    ),
                }
        summary[f"session{session}"]["paired_to_mean"] = paired
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/interim/lee2019_cnn_gru_scores_full54"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_test_time_ensemble_full54.json"))
    args = parser.parse_args()

    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = args.root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(analyze(path, args.shifts))
            print(f"test-time ensemble S{session} sub-{subject:02d}", flush=True)
    payload = {
        "protocol": {
            "score": "frozen CNN-GRU per-seed streams; no retraining",
            "fusions": FUSIONS,
            "threshold": "each fusion independently thresholded by the existing transferred/common-FP evaluator",
            "calibration": "the cache's already-averaged three-seed calibration stream is held fixed across fusions",
            "decision": "instantaneous score alarm + 1 s refractory (same as source cache baseline)",
            "null": "500 circular shifts per subject-session",
            "claim_limit": "Lee2019 cue-relative pseudo-uncued replay",
        },
        "summary": summarize(rows, args.bootstrap),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
