"""Audit EWMA threshold coordinates and quantify threshold conditioning.

This analysis reconstructs the canonical CNN-GRU -> all-score EWMA(H=30 s)
-> MA(0.5 s) -> dwell(0.3 s) -> refractory(1 s) path from frozen score
caches.  Thresholds are tuned by replaying that exact decision path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.decision import moving_average_dwell
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache
from scripts.run_lee2019_pseudo_uncued_riemannian import calibration

STEP = 0.1
TARGET_FP = 2.0
MA_SECONDS = 0.5
DWELL_SECONDS = 0.3
REFRACTORY_SECONDS = 1.0
HALF_LIFE_SECONDS = 30.0


def tune_ma_dwell(sequences: list[ScoredSequence], grid_size: int = 160):
    return tune_source_operating_points(
        sequences, "ma_dwell", [TARGET_FP], STEP,
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, grid_size=grid_size,
    )[TARGET_FP]


def normalize_anchor(blocks: list[np.ndarray]):
    joined = np.concatenate(blocks).astype(float)
    mean = float(np.mean(joined))
    variance = float(np.var(joined, ddof=1))
    scale = np.sqrt(max(variance, 1e-8))
    return mean, variance, [(np.asarray(block, float) - mean) / scale for block in blocks]


def moving_block_sample(values: np.ndarray, rng: np.random.Generator, block: int):
    values = np.asarray(values, float)
    starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
    indices = np.concatenate([
        (start + np.arange(block)) % len(values) for start in starts
    ])[: len(values)]
    return values[indices]


def bootstrap_thresholds(blocks, repeats, block_samples, grid_size, seed):
    joined = np.concatenate(blocks).astype(float)
    rng = np.random.default_rng(seed)
    thresholds = np.empty(repeats, float)
    for index in range(repeats):
        sample = moving_block_sample(joined, rng, block_samples)
        mean = float(sample.mean())
        scale = float(max(sample.std(ddof=1), 1e-8))
        thresholds[index] = tune_ma_dwell(
            [calibration((sample - mean) / scale)], grid_size
        ).threshold
    return thresholds


def local_response(sequence, center, offsets):
    rows = []
    for offset in offsets:
        threshold = center + offset
        decision = moving_average_dwell(
            sequence.scores, threshold, STEP, MA_SECONDS, DWELL_SECONDS,
            REFRACTORY_SECONDS,
        )
        metrics = evaluate_alarms(
            sequence.times, decision.alarms, sequence.idle_intervals,
            sequence.command_intervals,
        )
        rows.append({"offset": float(offset), "threshold": float(threshold),
                     "fp_per_min": float(metrics.fp_per_minute)})
    x = np.asarray([row["threshold"] for row in rows])
    y = np.asarray([row["fp_per_min"] for row in rows])
    slope = float(np.polyfit(x, y, 1)[0])
    return rows, slope


def analyze(path, subject, session, repeats, block_samples, grid_size, update_mode):
    raw_sequence, blocks = load_cache(path)
    mean, variance, normalized_blocks = normalize_anchor(blocks)
    transfer = tune_ma_dwell([calibration(x) for x in normalized_blocks], grid_size)

    adapted = ewma_normalize_and_decide(
        raw_sequence.scores, initial_mean=mean, initial_variance=variance,
        threshold=transfer.threshold, step_seconds=STEP,
        half_life_seconds=HALF_LIFE_SECONDS, update_mode=update_mode,
        ma_seconds=MA_SECONDS, dwell_seconds=DWELL_SECONDS,
        refractory_seconds=REFRACTORY_SECONDS, sigma_floor_fraction=0.25,
    )
    normalized_sequence = ScoredSequence(
        raw_sequence.times, adapted.statistic, raw_sequence.idle_intervals,
        raw_sequence.command_intervals,
    )
    common = tune_ma_dwell([normalized_sequence], grid_size)
    deployed_metrics = evaluate_alarms(
        normalized_sequence.times, adapted.alarms,
        normalized_sequence.idle_intervals, normalized_sequence.command_intervals,
    )
    bootstrap = bootstrap_thresholds(
        blocks, repeats, block_samples, grid_size,
        930000 + session * 1000 + subject,
    )
    response, slope = local_response(
        normalized_sequence, transfer.threshold,
        np.arange(-0.5, 0.5001, 0.1),
    )
    ci = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "subject": subject,
        "session": session,
        "anchor_samples": int(sum(len(x) for x in blocks)),
        "anchor_seconds": float(sum(len(x) for x in blocks) * STEP),
        "transfer_threshold": float(transfer.threshold),
        "anchor_achieved_fp_per_min": float(transfer.source_fp_per_minute),
        "common_fp_threshold": float(common.threshold),
        "common_achieved_fp_per_min": float(common.source_fp_per_minute),
        "threshold_displacement": float(transfer.threshold - common.threshold),
        "common_threshold_inside_anchor_ci": bool(ci[0] <= common.threshold <= ci[1]),
        "anchor_bootstrap_threshold_ci95": ci.tolist(),
        "anchor_bootstrap_threshold_sd": float(np.std(bootstrap, ddof=1)),
        "deployed_fp_per_min": float(deployed_metrics.fp_per_minute),
        "deployed_tpr": float(deployed_metrics.true_positive_rate),
        "local_fp_per_min_per_threshold_unit": slope,
        "local_response": response,
    }


def bootstrap_median(values, seed, repeats=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def summarize(rows, session):
    chosen = [row for row in rows if row["session"] == session]
    output = {"session": session, "n": len(chosen)}
    for key in ("deployed_fp_per_min", "threshold_displacement",
                "anchor_bootstrap_threshold_sd",
                "local_fp_per_min_per_threshold_unit"):
        values = [row[key] for row in chosen]
        output[f"{key}_median"] = float(np.median(values))
        output[f"{key}_ci95"] = bootstrap_median(
            values, 20261100 + session * 100 + len(key)
        )
    output["common_threshold_inside_anchor_ci_count"] = int(sum(
        row["common_threshold_inside_anchor_ci"] for row in chosen
    ))
    output["common_threshold_below_anchor_ci_count"] = int(sum(
        row["common_fp_threshold"] < row["anchor_bootstrap_threshold_ci95"][0]
        for row in chosen
    ))
    output["common_threshold_above_anchor_ci_count"] = int(sum(
        row["common_fp_threshold"] > row["anchor_bootstrap_threshold_ci95"][1]
        for row in chosen
    ))
    output["positive_displacement_count"] = int(sum(
        row["threshold_displacement"] > 0 for row in chosen
    ))
    standardized = [
        abs(row["threshold_displacement"])
        / max(row["anchor_bootstrap_threshold_sd"], 1e-12)
        for row in chosen
    ]
    output["absolute_displacement_over_bootstrap_sd_median"] = float(
        np.median(standardized)
    )
    rho, pvalue = spearmanr(
        [row["threshold_displacement"] for row in chosen],
        [row["deployed_fp_per_min"] for row in chosen],
    )
    output["displacement_vs_deployed_fp_spearman"] = {
        "rho": float(rho), "p": float(pvalue)
    }
    output["under_firing_count"] = int(sum(
        row["deployed_fp_per_min"] < 1.5 for row in chosen
    ))
    output["admissible_count"] = int(sum(
        1.5 <= row["deployed_fp_per_min"] <= 2.5 for row in chosen
    ))
    output["over_firing_count"] = int(sum(
        row["deployed_fp_per_min"] > 2.5 for row in chosen
    ))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path,
                        default=Path("data/interim/lee2019_cnn_gru_scores_subset20"))
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-seconds", type=float, default=5.0)
    parser.add_argument("--grid-size", type=int, default=160)
    parser.add_argument("--update-mode", choices=("all", "idle_only"), default="all")
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_cnn_gru_threshold_conditioning.json"))
    parser.add_argument("--reuse-output", action="store_true")
    args = parser.parse_args()
    if args.reuse_output:
        rows = json.loads(args.output.read_text(encoding="utf-8"))["rows"]
    else:
        rows = []
    for session in (() if args.reuse_output else args.sessions):
        for subject in args.subjects:
            rows.append(analyze(
                args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session, args.bootstrap,
                max(1, int(round(args.block_seconds / STEP))), args.grid_size,
                args.update_mode,
            ))
            print(f"threshold audit session={session} subject={subject}", flush=True)
    payload = {
        "protocol": {
            "coordinate": "anchor-standardized and causal EWMA-standardized z; threshold applied to MA(z)",
            "decision": "MA 0.5 s + dwell 0.3 s + refractory 1.0 s",
            "ewma": f"{args.update_mode} H=30 s; identical anchor initialization",
            "target_fp_per_min": TARGET_FP,
            "bootstrap_repeats": args.bootstrap,
            "bootstrap_block_seconds": args.block_seconds,
            "common_fp_uses_evaluation_idle": True,
        },
        "summary": [summarize(rows, session) for session in args.sessions],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
