"""Matched-achieved-FP diagnostic for saved EEG-reinferred CNN-GRU sham scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_rest_sham_matched_fp import (
    compute_guard_local_ma,
    matched_curve,
    matched_pair,
)
from uncued_mi.decision import guard_rearmed_alarms


def summarize(path: Path, thresholds: int, target_fps: tuple[float, ...]) -> dict:
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["observed_raw"], dtype=np.float64)
        sham_raw = np.asarray(data["sham_raw_scores"], dtype=np.float64)
        anchor = np.asarray(data["anchor_scores"], dtype=np.float64)
        times = np.asarray(data["times"], dtype=np.float64)
        allowed = np.asarray(data["allowed"], dtype=bool)
        events = np.asarray(data["events"], dtype=np.int64)
    if sham_raw.shape[1] != len(raw):
        raise AssertionError("sham and observed score lengths differ")
    future = times >= events[20, 2] / 250.
    task = np.zeros(len(times), dtype=bool)
    hit_label = np.full(len(times), -1, dtype=np.int32)
    for trial, event in enumerate(events[20:]):
        cue = event[5] / 250.
        onset = cue + .5
        task |= (times >= onset) & (times <= cue + 4.)
        hit_label[(times >= onset) & (times <= onset + 2.)] = trial
    idle = future & allowed & ~task
    exposure_minutes = float(idle.sum() * .1 / 60.)
    mean, var = float(anchor.mean()), float(anchor.var(ddof=1))
    observed_ma = compute_guard_local_ma(raw, mean, var, allowed)
    observed_curve = matched_curve(observed_ma, allowed, future, idle,
                                   hit_label, thresholds)
    sham_mas = [compute_guard_local_ma(sham, mean, var, allowed)
                for sham in sham_raw]
    sham_curves = [matched_curve(ma, allowed, future, idle, hit_label,
                                 thresholds) for ma in sham_mas]
    result = {
        "scope": "single-session matched-achieved-FP EEG-level sham pilot; not cohort A1",
        "replicates": len(sham_raw),
        "future_trials": len(events) - 20,
        "idle_exposure_minutes": exposure_minutes,
        "thresholds_per_curve": thresholds,
        "target_fps": {},
    }
    for fp in target_fps:
        desired_count = fp * exposure_minutes
        pairs = [matched_pair(observed_curve, curve, desired_count)
                 for curve in sham_curves]
        valid = [pair for pair in pairs if pair is not None]
        for draw, pair in enumerate(pairs[:3]):
            if pair is None:
                continue
            count, observed_hits, sham_hits, observed_threshold, sham_threshold = pair
            for ma, threshold, hits in (
                (observed_ma, observed_threshold, observed_hits),
                (sham_mas[draw], sham_threshold, sham_hits),
            ):
                alarms = guard_rearmed_alarms(ma, threshold, allowed, 3, 10)
                if int(np.sum(alarms & idle)) != count:
                    raise AssertionError("scalar FP mismatch")
                if int(len(np.unique(hit_label[alarms & (hit_label >= 0)]))) != hits:
                    raise AssertionError("scalar hit mismatch")
        excess = np.asarray([(p[1] - p[2]) / 30. for p in valid])
        result["target_fps"][str(fp)] = {
            "desired_false_alarm_count": desired_count,
            "matched_draws": len(valid),
            "matched_false_alarm_count": [int(p[0]) for p in valid],
            "observed_sen2": [p[1] / 30. for p in valid],
            "sham_sen2": [p[2] / 30. for p in valid],
            "excess_sen2": excess.tolist(),
            "median_excess_sen2": float(np.median(excess)) if len(excess) else None,
            "median_achieved_fp_per_min": (
                float(np.median([p[0] for p in valid]) / exposure_minutes)
                if valid else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/processed/performance_revalidation_20261001/"
                        "a1_cnn_gru_eeg_rest_sham_pilot_sub03_s1_fp2.npz")
    parser.add_argument("--thresholds", type=int, default=301)
    parser.add_argument("--target-fps", type=float, nargs="+", default=[2., 4.])
    args = parser.parse_args()
    result = summarize(args.input, args.thresholds, tuple(args.target_fps))
    output = args.input.with_name(args.input.stem + "_matched_fp.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
