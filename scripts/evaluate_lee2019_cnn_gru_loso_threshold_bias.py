"""LOSO population correction of prospective CNN-GRU threshold bias.

For each held-out subject, the correction is the median difference between
the same-tail oracle threshold and the prospective prefix-30 threshold in all
other subjects.  No held-out evaluation information is used to form the
correction.  Both pooled-session and same-session source corrections are
tested against the raw prospective threshold and the diagnostic oracle.
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
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import load_cache
from uncued_mi.adaptation import ewma_normalize_and_decide


def prepared_tail(cache_path, subject, session, shifts):
    sequence, _ = load_cache(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        raw_anchor = np.asarray(data["session_anchor_scores"], float)
    mean, variance, _ = normalize_anchor([raw_anchor])
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
    tail = fixed_tail(sequence, np.asarray(ewma.statistic, float))
    rng = np.random.default_rng(20260910 + 1000 * session + subject)
    offsets = rng.integers(1, len(tail.scores), shifts)
    return tail, offsets


def add_metadata(result, target, correction):
    result.update(
        target_fp_per_min=float(target),
        threshold_correction=float(correction),
        operating_point_log_error=float(
            abs(np.log((result["actual_fp_per_min"] + 0.05) / (target + 0.05)))
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exposure-results", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_idle_exposure_curve_subset20.json"),
    )
    parser.add_argument(
        "--cnn-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_scores_prospective"),
    )
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_loso_threshold_bias_subset20.json"),
    )
    args = parser.parse_args()

    source = json.loads(args.exposure_results.read_text(encoding="utf-8"))
    original_rows = source["rows"]
    rows = []
    for original in original_rows:
        subject = int(original["subject"])
        session = int(original["session"])
        train = [row for row in original_rows if int(row["subject"]) != subject]
        cache_path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
        tail, offsets = prepared_tail(cache_path, subject, session, args.shifts)
        arms = {name: {} for name in (
            "prefix30_raw", "loso_pooled_bias", "loso_session_bias", "tail20_oracle"
        )}
        corrections = {}
        for target in TARGETS:
            key = str(target)
            raw_threshold = float(original["arms"]["prefix30"][key]["threshold"])
            oracle_threshold = float(original["arms"]["tail20_oracle"][key]["threshold"])
            pooled_residuals = [
                float(row["arms"]["tail20_oracle"][key]["threshold"])
                - float(row["arms"]["prefix30"][key]["threshold"])
                for row in train
            ]
            session_residuals = [
                float(row["arms"]["tail20_oracle"][key]["threshold"])
                - float(row["arms"]["prefix30"][key]["threshold"])
                for row in train if int(row["session"]) == session
            ]
            pooled_correction = float(np.median(pooled_residuals))
            session_correction = float(np.median(session_residuals))
            thresholds = {
                "prefix30_raw": (raw_threshold, 0.0),
                "loso_pooled_bias": (raw_threshold + pooled_correction, pooled_correction),
                "loso_session_bias": (raw_threshold + session_correction, session_correction),
                "tail20_oracle": (oracle_threshold, oracle_threshold - raw_threshold),
            }
            corrections[key] = {
                "pooled": pooled_correction,
                "same_session": session_correction,
            }
            for name, (threshold, correction) in thresholds.items():
                arms[name][key] = add_metadata(
                    evaluate(tail, threshold, offsets), target, correction
                )
        exposure = float(original["known_idle_exposure_seconds"]["prefix30"])
        oracle_exposure = float(original["known_idle_exposure_seconds"]["tail20_oracle"])
        rows.append({
            "subject": subject,
            "session": session,
            "evaluation_trials": int(original["evaluation_trials"]),
            "known_idle_exposure_seconds": {
                "prefix30_raw": exposure,
                "loso_pooled_bias": exposure,
                "loso_session_bias": exposure,
                "tail20_oracle": oracle_exposure,
            },
            "corrections": corrections,
            "arms": arms,
        })
        print(f"LOSO-bias S{session} sub-{subject:02d}", flush=True)

    summary = {
        f"session{session}": summarize_group(
            [row for row in rows if row["session"] == session],
            202609120 + session * 10000,
            baseline_arm="prefix30_raw",
        )
        for session in (1, 2)
    }
    summary["combined"] = summarize_group(
        rows, 202609129, baseline_arm="prefix30_raw"
    )
    payload = {
        "protocol": {
            "base_threshold": "prospective anchor + first-30-trial continuous prefix",
            "correction": "LOSO source median of tail-oracle minus prospective threshold",
            "arms": {
                "pooled": "other 19 subjects, both sessions",
                "same_session": "other 19 subjects, matching session number",
            },
            "evaluation": "same untouched final 20 trials",
            "target_leakage": "none; held-out subject excluded from correction",
            "claim_limit": "dataset-level calibration; Lee2019 cue-relative replay",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
