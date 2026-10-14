from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_matched_lda_scaling import (
    fit_lda,
    half,
    load_cache,
    rest_normalized_features,
    tune_ridge,
    xy_from_intervals,
)


def fit_subject(cache: dict, subject: int, session: int, seed: int) -> tuple[np.ndarray, float]:
    normalized = rest_normalized_features(cache)
    command = half(cache["command_intervals"], "first")
    idle = half(cache["iti_intervals"], "first")
    ridge = tune_ridge(
        [(subject, command, idle)],
        {subject: cache},
        {subject: normalized},
        [0.001, 0.01, 0.1, 1.0, 10.0],
        np.random.default_rng(seed + session * 1000 + subject),
    )
    x, y = xy_from_intervals(
        normalized, cache["continuous_times"], command, idle
    )
    _, scale, weight, _ = fit_lda(x, y, ridge)
    original_weight = weight / scale
    original_weight /= np.linalg.norm(original_weight)
    return original_weight, ridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/lee2019_tier0"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lda_weight_similarity.json"))
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    weights: dict[int, dict[int, np.ndarray]] = {1: {}, 2: {}}
    ridges = []
    for session in (1, 2):
        for subject in range(1, 55):
            cache = load_cache(
                args.cache_dir
                / f"session{session}"
                / f"sub-{subject:02d}_tier0_features.npz"
            )
            weights[session][subject], ridge = fit_subject(
                cache, subject, session, args.seed
            )
            ridges.append({"session": session, "subject": subject, "ridge": ridge})
    subject_means: dict[int, dict[int, float]] = {1: {}, 2: {}}
    for session in (1, 2):
        for subject in range(1, 55):
            subject_means[session][subject] = float(
                np.mean(
                    [
                        weights[session][subject] @ weights[session][other]
                        for other in range(1, 55)
                        if other != subject
                    ]
                )
            )
    delta = np.asarray(
        [subject_means[2][s] - subject_means[1][s] for s in range(1, 55)]
    )
    same_subject = np.asarray(
        [weights[1][s] @ weights[2][s] for s in range(1, 55)]
    )
    result = {
        "method": "within-subject first-half LDA; inner ridge tuning; weights mapped back from training-standardized to rest-normalized feature coordinates",
        "session1_leave_one_out_cosine_median": float(np.median(list(subject_means[1].values()))),
        "session2_leave_one_out_cosine_median": float(np.median(list(subject_means[2].values()))),
        "paired_s2_minus_s1_median": float(np.median(delta)),
        "paired_wilcoxon_p": float(wilcoxon(delta).pvalue),
        "same_subject_cross_session_cosine_median": float(np.median(same_subject)),
        "subject_rows": [
            {
                "subject": s,
                "session1_cohort_cosine": subject_means[1][s],
                "session2_cohort_cosine": subject_means[2][s],
                "same_subject_cross_session_cosine": float(same_subject[s - 1]),
            }
            for s in range(1, 55)
        ],
        "ridge_rows": ridges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("rows")}, indent=2))


if __name__ == "__main__":
    main()
