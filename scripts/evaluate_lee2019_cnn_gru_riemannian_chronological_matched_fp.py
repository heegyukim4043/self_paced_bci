"""Observed-only matched-FP CNN-GRU/Riemannian diagnostic on chronological replay.

This isolates achieved idle-alarm count but does NOT remove schedule/guard
chance effects. Do not treat a positive difference as validated EEG evidence.
"""

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


def _bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    if len(values) < 10:
        raise ValueError("fewer than 10 independent subjects: CI not reported")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[index], axis=1)
    return np.quantile(medians, [.025, .975]).tolist()


def high_at_first_valid_ma(ma: np.ndarray, threshold: float, times: np.ndarray,
                           allowed: np.ndarray, events: np.ndarray) -> tuple[int, int]:
    high = 0
    valid = 0
    for event in events[20:]:
        cue = event[5] / 250.
        candidates = np.flatnonzero((times >= cue) & (times < cue + 2.) &
                                    allowed & np.isfinite(ma))
        if len(candidates):
            valid += 1
            high += bool(ma[candidates[0]] >= threshold)
    return high, valid


def run(args: argparse.Namespace) -> dict:
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            riem_path = args.riemannian_root / f"session{session}" / f"sub-{subject:02d}.npz"
            cnn_path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not riem_path.exists() or not cnn_path.exists():
                if args.allow_missing:
                    continue
                raise FileNotFoundError(f"missing paired cache: {riem_path} or {cnn_path}")
            with np.load(riem_path, allow_pickle=False) as riem, np.load(
                cnn_path, allow_pickle=False,
            ) as cnn:
                for key in ("events", "times", "allowed"):
                    if not np.array_equal(riem[key], cnn[key]):
                        raise AssertionError(f"paired stream {key} mismatch S{session} sub-{subject:02d}")
                times = np.asarray(riem["times"], dtype=float)
                allowed = np.asarray(riem["allowed"], dtype=bool)
                events = np.asarray(riem["events"], dtype=np.int64)
                r_raw = np.asarray(riem["raw_scores"], dtype=float)
                c_raw = np.asarray(cnn["raw_scores"], dtype=float)
                r_anchor = np.asarray(riem["anchor_raw_scores"], dtype=float)
                c_anchor = np.asarray(cnn["anchor_scores"], dtype=float)
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
            def curve(raw: np.ndarray, anchor: np.ndarray):
                ma = compute_guard_local_ma(
                    raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed,
                )
                return ma, matched_curve(ma, allowed, future, idle, hit_label,
                                         args.thresholds)
            riem_ma, riem_curve = curve(r_raw, r_anchor)
            cnn_ma, cnn_curve = curve(c_raw, c_anchor)
            for fp in args.target_fps:
                point = matched_pair(riem_curve, cnn_curve, fp * exposure_minutes)
                if point is None:
                    rows.append({"subject": subject, "session": session,
                                 "target_fp": fp, "matched": False})
                    continue
                count, r_hits, c_hits, r_threshold, c_threshold = point
                r_high, r_valid = high_at_first_valid_ma(
                    riem_ma, r_threshold, times, allowed, events,
                )
                c_high, c_valid = high_at_first_valid_ma(
                    cnn_ma, c_threshold, times, allowed, events,
                )
                rows.append({
                    "subject": subject, "session": session, "target_fp": fp,
                    "matched": True, "matched_fp_count": count,
                    "achieved_fp_per_min": count / exposure_minutes,
                    "riemannian_hits2": r_hits, "cnn_gru_hits2": c_hits,
                    "riemannian_sen2": r_hits / 30., "cnn_gru_sen2": c_hits / 30.,
                    "delta_sen2": (c_hits - r_hits) / 30.,
                    "riemannian_threshold": r_threshold,
                    "cnn_gru_threshold": c_threshold,
                    "riemannian_high_at_first_valid_ma": r_high,
                    "cnn_gru_high_at_first_valid_ma": c_high,
                    "riemannian_first_valid_ma_trials": r_valid,
                    "cnn_gru_first_valid_ma_trials": c_valid,
                })
        print(f"finished chronological observed-only S{session}", flush=True)
    summary = {
        "status": "observed-only diagnostic; EEG-level sham/null remains required",
        "dataset": "Lee2019 chronological cue-relative pseudo-uncued replay",
        "models": "source-pooled LOSO CNN-GRU vs source-pooled Riemannian Arm A",
        "threshold": "model-specific evaluation-label oracle at exactly matched achieved idle alarm count",
        "dwell_score_samples": 3,
        "onset_collar_seconds": 2.,
        "thresholds_per_model_curve": args.thresholds,
        "rows": rows,
        "by_session": {},
    }
    for fp in args.target_fps:
        for session in args.sessions:
            matched = [row for row in rows if row["session"] == session and
                       row["target_fp"] == fp and row["matched"]]
            values = np.asarray([row["delta_sen2"] for row in matched], dtype=float)
            key = f"FP{fp:g}_S{session}"
            summary["by_session"][key] = {
                "matched_subjects": len(matched),
                "missing_subjects": len(args.subjects) - len(matched),
                "median_achieved_fp_per_min": float(np.median([
                    row["achieved_fp_per_min"] for row in matched
                ])) if matched else None,
                "median_riemannian_sen2": float(np.median([
                    row["riemannian_sen2"] for row in matched
                ])) if matched else None,
                "median_cnn_gru_sen2": float(np.median([
                    row["cnn_gru_sen2"] for row in matched
                ])) if matched else None,
                "median_riemannian_high_at_first_valid_ma": float(np.median([
                    row["riemannian_high_at_first_valid_ma"] for row in matched
                ])) if matched else None,
                "median_cnn_gru_high_at_first_valid_ma": float(np.median([
                    row["cnn_gru_high_at_first_valid_ma"] for row in matched
                ])) if matched else None,
                "paired_median_delta_sen2": float(np.median(values)) if len(values) else None,
                "paired_median_delta_ci95": _bootstrap_median_ci(
                    values, 20261001 + int(fp) * 100 + session,
                ) if len(values) >= 10 else None,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--riemannian-root", type=Path,
                        default=ROOT / "data/interim/lee2019_revalidation_riemannian_scores")
    parser.add_argument("--cnn-root", type=Path,
                        default=ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--thresholds", type=int, default=301)
    parser.add_argument("--target-fps", type=float, nargs="+", default=[2., 4.])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/performance_revalidation_20261001/"
                        "a2_cnn_gru_vs_riemannian_chronological_observed_matched_fp.json")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"],
                      "by_session": summary["by_session"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
