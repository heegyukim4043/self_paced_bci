"""Prospective recency audit for known-idle CNN-GRU threshold calibration.

All arms evaluate the same final 20 trials.  Equal-size calibration blocks are
taken from the beginning, middle, or immediately preceding part of the first
30 trials to separate idle quantity from temporal representativeness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS,
    HALF_LIFE_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    STEP,
    normalize_anchor,
)
from scripts.evaluate_lee2019_cnn_gru_idle_exposure_curve import (
    fixed_tail,
    summarize_group,
)
from scripts.evaluate_lee2019_cnn_gru_prospective_idle_calibration import (
    TARGETS,
    evaluate,
    tune,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration
from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.operating_point import ScoredSequence


BLOCKS = {
    "prefix00": None,
    "first10": (0, 10),
    "middle10": (10, 20),
    "recent10": (20, 30),
    "first20": (0, 20),
    "recent20": (10, 30),
    "all30": (0, 30),
}


def block_sequence(sequence, scores, start_trial, stop_trial):
    start = float(sequence.idle_intervals[start_trial, 0])
    stop = float(sequence.command_intervals[stop_trial - 1, 1])
    keep = (sequence.times >= start) & (sequence.times < stop)
    return ScoredSequence(
        np.asarray(sequence.times[keep], float),
        np.asarray(scores[keep], float),
        np.asarray(sequence.idle_intervals[start_trial:stop_trial], float),
        np.asarray(sequence.command_intervals[start_trial:stop_trial], float),
    )


def idle_exposure(sequence, block, anchor_seconds):
    if block is None:
        return anchor_seconds
    start, stop = block
    intervals = np.asarray(sequence.idle_intervals[start:stop], float)
    return anchor_seconds + float(np.sum(intervals[:, 1] - intervals[:, 0]))


def analyze(path, subject, session, shifts):
    sequence, _ = load_cache(path)
    with np.load(path, allow_pickle=False) as data:
        raw_anchor = np.asarray(data["session_anchor_scores"], float)
    mean, variance, anchor_z = normalize_anchor([raw_anchor])
    ewma = ewma_normalize_and_decide(
        sequence.scores,
        initial_mean=mean,
        initial_variance=variance,
        threshold=0.0,
        step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS,
        update_mode="all",
        ma_seconds=MA_SECONDS,
        dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS,
        sigma_floor_fraction=0.25,
    )
    scores = np.asarray(ewma.statistic, float)
    anchor_sequences = [calibration(np.asarray(values, float)) for values in anchor_z]
    tail = fixed_tail(sequence, scores)
    calibration_sets = {}
    for name, block in BLOCKS.items():
        calibration_sets[name] = list(anchor_sequences)
        if block is not None:
            calibration_sets[name].append(block_sequence(sequence, scores, *block))
    calibration_sets["tail20_oracle"] = [tail]
    points = {name: tune(items) for name, items in calibration_sets.items()}

    rng = np.random.default_rng(20260911 + 1000 * session + subject)
    offsets = rng.integers(1, len(tail.scores), shifts)
    arms = {}
    for name, by_target in points.items():
        arms[name] = {}
        for target in TARGETS:
            point = by_target[target]
            result = evaluate(tail, point.threshold, offsets)
            result.update(
                target_fp_per_min=float(target),
                calibration_achieved_fp_per_min=float(point.source_fp_per_minute),
                operating_point_log_error=float(
                    abs(np.log((result["actual_fp_per_min"] + 0.05) / (target + 0.05)))
                ),
            )
            arms[name][str(target)] = result

    anchor_seconds = float(sum(len(values) for values in anchor_z) * STEP)
    exposures = {
        name: idle_exposure(sequence, block, anchor_seconds)
        for name, block in BLOCKS.items()
    }
    exposures["tail20_oracle"] = float(
        np.sum(tail.idle_intervals[:, 1] - tail.idle_intervals[:, 0])
    )
    return {
        "subject": int(subject),
        "session": int(session),
        "evaluation_trials": int(len(tail.command_intervals)),
        "known_idle_exposure_seconds": exposures,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnn-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_scores_prospective"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_idle_recency_subset20.json"),
    )
    args = parser.parse_args()

    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            rows.append(analyze(path, subject, session, args.shifts))
            print(f"idle-recency S{session} sub-{subject:02d}", flush=True)
    summary = {
        f"session{session}": summarize_group(
            [row for row in rows if row["session"] == session],
            202609110 + session * 10000,
        )
        for session in args.sessions
    }
    summary["combined"] = summarize_group(rows, 202609119)
    payload = {
        "protocol": {
            "score": "frozen three-seed CNN-GRU; causal all-score EWMA H30",
            "threshold_targets_fp_per_min": TARGETS,
            "calibration_blocks": BLOCKS,
            "evaluation": "same untouched final 20 trials for every arm",
            "purpose": "separate calibration quantity from temporal recency",
            "chance": "within-tail circular shifts",
            "claim_limit": "Lee2019 cue-relative pseudo-uncued replay",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
