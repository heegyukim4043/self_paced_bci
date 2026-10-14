"""Compare Riemannian gate false alarms on rest-rest seams vs continuous rest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.pseudo_uncued import fixed_budget_exponential_chunks
from uncued_mi.riemannian import tangent_features

SFREQ = 250.0
STEP = 0.1
WINDOW = 1.0


def reference(covariances: np.ndarray) -> np.ndarray:
    matrix = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12)
    return matrix + np.eye(len(matrix)) * ridge


def rolling_covariance(signal: np.ndarray) -> np.ndarray:
    width = int(round(WINDOW * SFREQ))
    step = int(round(STEP * SFREQ))
    ends = np.arange(width, len(signal) + 1, step)
    return np.asarray([np.cov(signal[end-width:end], rowvar=False) for end in ends])


def score_signal(signal: np.ndarray, ref, scaler, model) -> np.ndarray:
    return model.decision_function(scaler.transform(tangent_features(rolling_covariance(signal), ref)))


def sequence(scores: np.ndarray) -> ScoredSequence:
    times = np.arange(len(scores)) * STEP
    return ScoredSequence(times, scores, np.asarray([[0.0, len(scores) * STEP]]),
                          np.empty((0, 2)))


def alarm_count(scores: np.ndarray, threshold: float) -> tuple[int, np.ndarray]:
    result = apply_decision(sequence(scores), "ma_dwell", threshold, STEP,
                            ma_seconds=0.5, dwell_seconds=0.3,
                            refractory_seconds=1.0)
    return int(result.alarms.sum()), result.alarms


def analyze(raw_path: Path, covariance_path: Path, subject: int, guard_seconds: float) -> dict:
    with np.load(covariance_path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        cmd_trial = data["train_command_trial"]
        iti_trial = data["train_idle_iti_trial"]
        cmd = tangent_features(data["train_command_covariance"][cmd_trial < 50], ref)
        iti = tangent_features(data["train_idle_iti_covariance"][iti_trial < 50], ref)
        scaler = StandardScaler().fit(np.r_[cmd, iti])
        model = LogisticRegression(class_weight="balanced", solver="liblinear",
                                   max_iter=2000, random_state=20260804).fit(
            scaler.transform(np.r_[cmd, iti]),
            np.r_[np.ones(len(cmd)), np.zeros(len(iti))])
    with np.load(raw_path, allow_pickle=False) as data:
        anchor = data["pre_train"]
        blocks = [data[name] for name in ("post_train", "pre_test", "post_test")]

    anchor_scores = score_signal(anchor, ref, scaler, model)
    point = tune_source_operating_points([sequence(anchor_scores)], "ma_dwell", [2.0], STEP,
                                         ma_seconds=0.5, dwell_seconds=0.3,
                                         refractory_seconds=1.0, grid_size=160)[2.0]
    chunks, _ = fixed_budget_exponential_chunks(
        blocks, chunks_per_block=(3, 3, 4), sfreq=SFREQ,
        minimum_seconds=5.0, seed=20260804 + subject)
    joined = np.concatenate(chunks)
    seam_samples = np.cumsum([len(chunk) for chunk in chunks[:-1]])
    joined_scores = score_signal(joined, ref, scaler, model)
    joined_n, joined_alarms = alarm_count(joined_scores, point.threshold)
    # Rolling scores are timestamped at their causal window end, not at zero.
    score_times = WINDOW + np.arange(len(joined_alarms)) * STEP
    alarm_times = score_times[joined_alarms]
    seam_times = seam_samples / SFREQ
    near = np.zeros(len(alarm_times), dtype=bool)
    near_grid = np.zeros(len(score_times), dtype=bool)
    for seam in seam_times:
        near |= np.abs(alarm_times - seam) <= guard_seconds
        near_grid |= np.abs(score_times - seam) <= guard_seconds
    guarded_n = int(np.sum(~near))
    guarded_exposure = float(np.sum(~near_grid) * STEP)
    near_exposure = float(np.sum(near_grid) * STEP)

    continuous_n = 0
    continuous_seconds = 0.0
    for block in blocks:
        scores = score_signal(block, ref, scaler, model)
        count, _ = alarm_count(scores, point.threshold)
        continuous_n += count
        continuous_seconds += len(scores) * STEP
    joined_seconds = len(joined_scores) * STEP
    return {
        "subject": subject,
        "threshold": float(point.threshold),
        "anchor_fp_per_minute": float(point.source_fp_per_minute),
        "joined_fp_per_minute": 60 * joined_n / joined_seconds,
        "joined_guarded_fp_per_minute": 60 * guarded_n / guarded_exposure,
        "joined_near_seam_fp_per_minute": 60 * int(near.sum()) / near_exposure,
        "continuous_fp_per_minute": 60 * continuous_n / continuous_seconds,
        "alarms_within_guard": int(near.sum()),
        "seams": len(seam_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest_raw/session1"))
    parser.add_argument("--cov-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest/session1"))
    parser.add_argument("--guard-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_false_seam_gate_audit.json"))
    args = parser.parse_args()
    subjects = sorted(set(int(p.stem.split("-")[1]) for p in args.raw_root.glob("sub-*.npz")))
    rows = [analyze(args.raw_root / f"sub-{s:02d}.npz",
                    args.cov_root / f"sub-{s:02d}.npz", s, args.guard_seconds)
            for s in subjects]
    joined = np.asarray([r["joined_fp_per_minute"] for r in rows])
    guarded = np.asarray([r["joined_guarded_fp_per_minute"] for r in rows])
    continuous = np.asarray([r["continuous_fp_per_minute"] for r in rows])
    near = np.asarray([r["joined_near_seam_fp_per_minute"] for r in rows])
    result = {
        "subjects": len(rows), "guard_seconds": args.guard_seconds,
        "joined_fp_per_minute_median": float(np.median(joined)),
        "joined_guarded_fp_per_minute_median": float(np.median(guarded)),
        "continuous_fp_per_minute_median": float(np.median(continuous)),
        "near_seam_fp_per_minute_median": float(np.median(near)),
        "joined_vs_continuous_wilcoxon_p": float(wilcoxon(joined, continuous).pvalue),
        "guarded_vs_continuous_wilcoxon_p": float(wilcoxon(guarded, continuous).pvalue),
        "near_seam_vs_continuous_wilcoxon_p": float(wilcoxon(near, continuous).pvalue),
        "alarms_within_2s_of_seam_total": int(sum(r["alarms_within_guard"] for r in rows)),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
