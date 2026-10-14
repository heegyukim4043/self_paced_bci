"""Exact-stream HSMM comparator for the full CNN-GRU validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.hsmm import GaussianEmission
from uncued_mi.operating_point import ScoredSequence
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_sequential_onset import RESTS, fit_score, reference
from scripts.run_lee2019_stateful_gru import (
    build_no_guard_stream,
    evaluate,
    hsmm_scores,
)


def analyze(path, subject, session, shifts):
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
    scaler, model = fit_score(
        command[command_trials < 50],
        iti[iti_trials < 50],
        rest["pre_train"][:split],
    )
    command_score = model.decision_function(scaler.transform(command))
    iti_score = model.decision_function(scaler.transform(iti))
    rest_score = {
        name: model.decision_function(scaler.transform(values))
        for name, values in rest.items()
    }
    emission = GaussianEmission.fit(
        np.r_[iti_score[iti_trials < 50], rest_score["pre_train"][:split]],
        command_score[command_trials < 50],
    )
    eval_rest = [values[len(values) // 2:] for values in rest_score.values()]
    eval_command = [
        command_score[command_trials == trial] for trial in range(50, 100)
    ]
    stream, _, _, template = build_no_guard_stream(
        eval_rest, eval_command, 20260728 + 100 * session + subject
    )
    sequence = ScoredSequence(
        template.times, hsmm_scores(stream, emission),
        template.idle_intervals, template.command_intervals,
    )
    calibration = [
        hsmm_scores(rest_score[name][:len(rest_score[name]) // 2], emission)
        for name in RESTS if name != "pre_train"
    ]
    rng = np.random.default_rng(88000 + 100 * session + subject)
    offsets = rng.integers(1, len(stream), size=shifts)
    return {
        "subject": subject,
        "session": session,
        "metrics": evaluate(sequence, calibration, offsets),
        "evaluation_trials": len(template.command_intervals),
    }


def bootstrap(values, seed, repeats):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_four_rest"))
    parser.add_argument("--cnn-result", type=Path, default=Path("data/processed/lee2019_cnn_gru_all54.json"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_gru_hsmm_comparator.json"))
    args = parser.parse_args()
    cnn = json.loads(args.cnn_result.read_text(encoding="utf-8"))
    subjects = sorted({row["subject"] for row in cnn["rows"]})
    sessions = sorted({row["session"] for row in cnn["rows"]})
    rows = []
    for session in sessions:
        for subject in subjects:
            rows.append(analyze(
                args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session, args.shifts,
            ))
            print(f"hsmm session={session} subject={subject}", flush=True)
    cnn_map = {(row["session"], row["subject"]): row for row in cnn["rows"]}
    summary = []
    for session in sessions:
        selected = [row for row in rows if row["session"] == session]
        modes = {}
        for mode in ("common_fp", "transferred"):
            hsmm = np.asarray([row["metrics"][mode]["excess_1s"] for row in selected])
            cnn_values = np.asarray([
                cnn_map[(session, row["subject"])]["metrics"][mode]["excess_1s"]
                for row in selected
            ])
            delta = cnn_values - hsmm
            modes[mode] = {
                "hsmm_excess_median": float(np.median(hsmm)),
                "cnn_excess_median": float(np.median(cnn_values)),
                "cnn_minus_hsmm_median": float(np.median(delta)),
                "cnn_minus_hsmm_ci": bootstrap(
                    delta, 91000 + session, args.bootstrap
                ),
                "cnn_win_fraction": float(np.mean(delta > 0)),
            }
        summary.append({"session": session, "subjects": len(selected), "modes": modes})
    payload = {"summary": summary, "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
