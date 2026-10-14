"""Decompose the prospective-vs-oracle CNN-GRU threshold gap.

The analysis uses the exact subset-20 prospective score caches and final-20
evaluation tail from the locked idle-exposure experiment. Distributional
diagnostics are calculated on the statistic to which the threshold is applied:
causal all-score EWMA followed by the locked 0.5-s moving average.

This is a diagnostic decomposition, not a deployable threshold estimator.
Evaluation-idle labels are used only to explain the already-observed gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.decision import moving_average_dwell
from scripts.analyze_lee2019_cnn_gru_threshold_conditioning import (
    DWELL_SECONDS,
    HALF_LIFE_SECONDS,
    MA_SECONDS,
    REFRACTORY_SECONDS,
    STEP,
    normalize_anchor,
)
from scripts.evaluate_lee2019_cnn_gru_idle_exposure_curve import (
    EVALUATION_START_TRIAL,
    fixed_tail,
    prefix_sequence,
)
from scripts.run_lee2019_cnn_gru_threshold_transfer_bakeoff import SUBSET20, load_cache


QUANTILES = (0.90, 0.95, 0.975, 0.99)
TARGETS = (2.0, 4.0)
ACF_MAX_SECONDS = 5.0


def ma_statistic(values: np.ndarray) -> np.ndarray:
    return moving_average_dwell(
        np.asarray(values, float), np.inf, STEP, MA_SECONDS,
        DWELL_SECONDS, REFRACTORY_SECONDS,
    ).statistic


def interval_chunks(times: np.ndarray, values: np.ndarray, intervals: np.ndarray) -> list[np.ndarray]:
    output = []
    for start, stop in np.asarray(intervals, float):
        keep = (times >= start) & (times < stop) & np.isfinite(values)
        chunk = np.asarray(values[keep], float)
        if len(chunk):
            output.append(chunk)
    return output


def quantile_dict(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    return {str(q): float(np.quantile(values, q)) for q in QUANTILES}


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    return float(max(mad, 1e-8))


def chunked_acf(chunks: list[np.ndarray], lag: int) -> float | None:
    left, right = [], []
    for chunk in chunks:
        if len(chunk) > lag:
            left.append(chunk[:-lag])
            right.append(chunk[lag:])
    if not left:
        return None
    x, y = np.concatenate(left), np.concatenate(right)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def effective_sample_size(chunks: list[np.ndarray]) -> dict:
    n = int(sum(len(chunk) for chunk in chunks))
    correlations = []
    for lag in range(1, int(round(ACF_MAX_SECONDS / STEP)) + 1):
        value = chunked_acf(chunks, lag)
        if value is None or value <= 0:
            break
        correlations.append(value)
    tau = 1.0 + 2.0 * float(np.sum(correlations))
    n_eff = n / max(tau, 1.0)
    return {
        "n_windows": n,
        "acf1": chunked_acf(chunks, 1),
        "positive_acf_lags": len(correlations),
        "integrated_autocorrelation_time_windows": tau,
        "effective_n": float(n_eff),
        "expected_independent_q99_exceedances": float(0.01 * n_eff),
    }


def extreme_clustering(chunks: list[np.ndarray], threshold: float) -> dict:
    exceedances = 0
    runs = []
    for chunk in chunks:
        active = np.asarray(chunk >= threshold, bool)
        exceedances += int(np.sum(active))
        padded = np.r_[False, active, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        runs.extend((stops - starts).tolist())
    dwell_steps = max(1, int(np.ceil(DWELL_SECONDS / STEP)))
    eligible_runs = [length for length in runs if length >= dwell_steps]
    exposure_minutes = sum(len(chunk) for chunk in chunks) * STEP / 60.0
    return {
        "threshold": float(threshold),
        "exceedance_windows": exceedances,
        "run_count": len(runs),
        "run_per_exceedance": float(len(runs) / exceedances) if exceedances else None,
        "median_run_seconds": float(np.median(runs) * STEP) if runs else None,
        "max_run_seconds": float(np.max(runs) * STEP) if runs else None,
        "dwell_steps": dwell_steps,
        "dwell_eligible_run_count": len(eligible_runs),
        "dwell_eligible_runs_per_min": float(len(eligible_runs) / exposure_minutes) if exposure_minutes else None,
        "fraction_exceedance_windows_in_dwell_runs": float(
            sum(eligible_runs) / exceedances
        ) if exceedances else None,
    }


def analyze_one(cache_path: Path, result_row: dict) -> dict:
    sequence, _ = load_cache(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        raw_anchor = np.asarray(data["session_anchor_scores"], float)

    mean, variance, anchor_z_blocks = normalize_anchor([raw_anchor])
    anchor_z = np.asarray(anchor_z_blocks[0], float)
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
    prefix = prefix_sequence(sequence, scores, EVALUATION_START_TRIAL)
    tail = fixed_tail(sequence, scores)

    anchor_ma = ma_statistic(anchor_z)
    anchor_chunks = [anchor_ma[np.isfinite(anchor_ma)]]
    prefix_ma = ma_statistic(prefix.scores)
    prefix_chunks = interval_chunks(prefix.times, prefix_ma, prefix.idle_intervals)
    tail_ma = ma_statistic(tail.scores)
    tail_chunks = interval_chunks(tail.times, tail_ma, tail.idle_intervals)

    prospective_chunks = anchor_chunks + prefix_chunks
    prospective = np.concatenate(prospective_chunks)
    task_prefix = np.concatenate(prefix_chunks)
    future = np.concatenate(tail_chunks)
    q_prospective = quantile_dict(prospective)
    q_future = quantile_dict(future)
    q_anchor = quantile_dict(anchor_chunks[0])
    q_task_prefix = quantile_dict(task_prefix)

    # A chronological split is a low-cost within-calibration instability
    # comparator; it is not treated as an independent inferential sample.
    midpoint = len(prospective) // 2
    internal_q99_difference = float(
        np.quantile(prospective[midpoint:], 0.99)
        - np.quantile(prospective[:midpoint], 0.99)
    )
    prospective_q975 = float(np.quantile(prospective, 0.975))
    tail_q975 = float(np.quantile(future, 0.975))
    prospective_extremes = extreme_clustering(prospective_chunks, prospective_q975)
    tail_extremes = extreme_clustering(tail_chunks, tail_q975)

    distribution = {
        "decision_stat_quantiles": {
            "anchor": q_anchor,
            "task_prefix_idle": q_task_prefix,
            "prospective_combined": q_prospective,
            "evaluation_tail_idle": q_future,
        },
        "tail_minus_prospective_quantiles": {
            str(q): float(q_future[str(q)] - q_prospective[str(q)]) for q in QUANTILES
        },
        "internal_split_q99_difference": internal_q99_difference,
        "absolute_external_minus_internal_q99_shift": float(
            abs(q_future["0.99"] - q_prospective["0.99"])
            - abs(internal_q99_difference)
        ),
        "location_shift_tail_minus_prospective": float(np.median(future) - np.median(prospective)),
        "log_robust_scale_ratio_tail_over_prospective": float(
            np.log(robust_scale(future) / robust_scale(prospective))
        ),
        "ks_prospective_vs_tail": float(ks_2samp(prospective, future).statistic),
        "ks_anchor_vs_task_prefix": float(ks_2samp(anchor_chunks[0], task_prefix).statistic),
        "ks_anchor_vs_tail": float(ks_2samp(anchor_chunks[0], future).statistic),
        "wasserstein_prospective_vs_tail": float(wasserstein_distance(prospective, future)),
        "prospective_dependence": effective_sample_size(prospective_chunks),
        "tail_dependence": effective_sample_size(tail_chunks),
        "prospective_extreme_clustering_q975": prospective_extremes,
        "tail_extreme_clustering_q975": tail_extremes,
        "tail_minus_prospective_dwell_runs_per_min_q975": float(
            tail_extremes["dwell_eligible_runs_per_min"]
            - prospective_extremes["dwell_eligible_runs_per_min"]
        ),
        "tail_minus_prospective_fraction_in_dwell_runs_q975": float(
            tail_extremes["fraction_exceedance_windows_in_dwell_runs"]
            - prospective_extremes["fraction_exceedance_windows_in_dwell_runs"]
        ),
    }

    targets = {}
    for target in TARGETS:
        key = str(target)
        prospective_result = result_row["arms"]["prefix30"][key]
        oracle_result = result_row["arms"]["tail20_oracle"][key]
        prospective_threshold = float(prospective_result["threshold"])
        oracle_threshold = float(oracle_result["threshold"])
        prospective_at_prospective = extreme_clustering(
            prospective_chunks, prospective_threshold
        )
        tail_at_prospective = extreme_clustering(tail_chunks, prospective_threshold)
        targets[key] = {
            "prospective_threshold": prospective_threshold,
            "oracle_threshold": oracle_threshold,
            "threshold_gap_prospective_minus_oracle": prospective_threshold - oracle_threshold,
            "prospective_threshold_percentile_in_future_idle": float(
                np.mean(future <= prospective_threshold)
            ),
            "oracle_threshold_percentile_in_prospective_idle": float(
                np.mean(prospective <= oracle_threshold)
            ),
            "actual_fp_gap_prospective_minus_oracle": float(
                prospective_result["actual_fp_per_min"] - oracle_result["actual_fp_per_min"]
            ),
            "prospective_actual_fp_per_min": float(prospective_result["actual_fp_per_min"]),
            "oracle_actual_fp_per_min": float(oracle_result["actual_fp_per_min"]),
            "prospective_threshold_structure": {
                "prospective_idle": prospective_at_prospective,
                "future_idle": tail_at_prospective,
                "future_minus_prospective_dwell_runs_per_min": float(
                    tail_at_prospective["dwell_eligible_runs_per_min"]
                    - prospective_at_prospective["dwell_eligible_runs_per_min"]
                ),
                "future_minus_prospective_exceedance_fraction": float(
                    tail_at_prospective["exceedance_windows"]
                    / max(len(future), 1)
                    - prospective_at_prospective["exceedance_windows"]
                    / max(len(prospective), 1)
                ),
            },
        }
    return {
        "subject": int(result_row["subject"]),
        "session": int(result_row["session"]),
        "prospective_exposure_seconds": float(result_row["known_idle_exposure_seconds"]["prefix30"]),
        "tail_idle_seconds": float(result_row["known_idle_exposure_seconds"]["tail20_oracle"]),
        "distribution": distribution,
        "targets": targets,
    }


def cluster_bootstrap(values: np.ndarray, subjects: np.ndarray, seed: int, repeats: int = 10000) -> list[float]:
    unique = np.unique(subjects)
    by_subject = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_subject[subject] for subject in sampled])
        draws.append(float(np.median(values[indices])))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def cluster_spearman_ci(
    x: np.ndarray, y: np.ndarray, subjects: np.ndarray, seed: int, repeats: int = 5000
) -> list[float] | None:
    unique = np.unique(subjects)
    by_subject = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_subject[subject] for subject in sampled])
        result = spearmanr(x[indices], y[indices])
        if np.isfinite(result.statistic):
            draws.append(float(result.statistic))
    return np.quantile(draws, [0.025, 0.975]).tolist() if draws else None


def path_get(row: dict, path: str) -> float:
    value = row
    for key in path.split("/"):
        value = value[key]
    return float(value)


def scalar_summary(rows: list[dict], path: str, seed: int) -> dict:
    values = np.asarray([path_get(row, path) for row in rows], float)
    subjects = np.asarray([row["subject"] for row in rows], int)
    return {
        "median": float(np.median(values)),
        "iqr": np.quantile(values, [0.25, 0.75]).tolist(),
        "subject_cluster_bootstrap_95ci": cluster_bootstrap(values, subjects, seed),
    }


def correlation_summary(rows: list[dict], target: str, path: str, seed: int) -> dict:
    gap = np.asarray([
        row["targets"][target]["threshold_gap_prospective_minus_oracle"] for row in rows
    ], float)
    feature = np.asarray([path_get(row, path) for row in rows], float)
    subjects = np.asarray([row["subject"] for row in rows], int)
    result = spearmanr(feature, gap)
    return {
        "feature": path,
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "subject_cluster_bootstrap_95ci": cluster_spearman_ci(
            feature, gap, subjects, seed
        ),
    }


def summarize(rows: list[dict]) -> dict:
    scalar_paths = {
        "prospective_exposure_seconds": "prospective_exposure_seconds",
        "tail_idle_seconds": "tail_idle_seconds",
        "q975_tail_minus_prospective": "distribution/tail_minus_prospective_quantiles/0.975",
        "q99_tail_minus_prospective": "distribution/tail_minus_prospective_quantiles/0.99",
        "external_minus_internal_abs_q99": "distribution/absolute_external_minus_internal_q99_shift",
        "ks_prospective_vs_tail": "distribution/ks_prospective_vs_tail",
        "wasserstein_prospective_vs_tail": "distribution/wasserstein_prospective_vs_tail",
        "log_scale_ratio": "distribution/log_robust_scale_ratio_tail_over_prospective",
        "prospective_acf1": "distribution/prospective_dependence/acf1",
        "prospective_effective_n": "distribution/prospective_dependence/effective_n",
        "prospective_expected_independent_q99": "distribution/prospective_dependence/expected_independent_q99_exceedances",
        "tail_acf1": "distribution/tail_dependence/acf1",
        "tail_effective_n": "distribution/tail_dependence/effective_n",
        "tail_extreme_run_per_exceedance": "distribution/tail_extreme_clustering_q975/run_per_exceedance",
        "tail_extreme_max_run_seconds": "distribution/tail_extreme_clustering_q975/max_run_seconds",
        "prospective_dwell_runs_per_min_q975": "distribution/prospective_extreme_clustering_q975/dwell_eligible_runs_per_min",
        "tail_dwell_runs_per_min_q975": "distribution/tail_extreme_clustering_q975/dwell_eligible_runs_per_min",
        "tail_minus_prospective_dwell_runs_per_min_q975": "distribution/tail_minus_prospective_dwell_runs_per_min_q975",
        "tail_minus_prospective_fraction_in_dwell_runs_q975": "distribution/tail_minus_prospective_fraction_in_dwell_runs_q975",
    }
    output = {"n_subject_sessions": len(rows), "n_subjects": len({r['subject'] for r in rows})}
    output["distribution"] = {
        name: scalar_summary(rows, path, 202609140 + index)
        for index, (name, path) in enumerate(scalar_paths.items())
    }
    features = (
        "distribution/tail_minus_prospective_quantiles/0.975",
        "distribution/tail_minus_prospective_quantiles/0.99",
        "distribution/ks_prospective_vs_tail",
        "distribution/wasserstein_prospective_vs_tail",
        "distribution/log_robust_scale_ratio_tail_over_prospective",
        "distribution/prospective_dependence/effective_n",
        "distribution/prospective_dependence/expected_independent_q99_exceedances",
        "distribution/tail_dependence/effective_n",
        "distribution/tail_extreme_clustering_q975/run_per_exceedance",
        "distribution/tail_extreme_clustering_q975/max_run_seconds",
        "distribution/tail_minus_prospective_dwell_runs_per_min_q975",
        "distribution/tail_minus_prospective_fraction_in_dwell_runs_q975",
    )
    output["targets"] = {}
    for target_index, target in enumerate(TARGETS):
        key = str(target)
        target_output = {
            name: scalar_summary(rows, f"targets/{key}/{name}", 202609200 + target_index * 100 + i)
            for i, name in enumerate((
                "threshold_gap_prospective_minus_oracle",
                "prospective_threshold_percentile_in_future_idle",
                "oracle_threshold_percentile_in_prospective_idle",
                "actual_fp_gap_prospective_minus_oracle",
                "prospective_actual_fp_per_min",
                "oracle_actual_fp_per_min",
            ))
        }
        target_output["threshold_gap_correlations"] = [
            correlation_summary(rows, key, feature, 202610000 + target_index * 100 + i)
            for i, feature in enumerate(features)
        ]
        output["targets"][key] = target_output
    return output


def plot(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"2.0": "#1f77b4", "4.0": "#d95f02"}
    for key in ("2.0", "4.0"):
        gap = np.asarray([row["targets"][key]["threshold_gap_prospective_minus_oracle"] for row in rows])
        q99 = np.asarray([row["distribution"]["tail_minus_prospective_quantiles"]["0.99"] for row in rows])
        ks = np.asarray([row["distribution"]["ks_prospective_vs_tail"] for row in rows])
        axes[0, 0].scatter(q99, gap, alpha=0.7, s=30, color=colors[key], label=f"target FP={key}")
        axes[0, 1].scatter(ks, gap, alpha=0.7, s=30, color=colors[key], label=f"target FP={key}")
    axes[0, 0].axhline(0, color="0.3", lw=1)
    axes[0, 0].axvline(0, color="0.3", lw=1)
    axes[0, 0].set(xlabel="Future - prospective Q99 (MA-score)", ylabel="Prospective - oracle threshold", title="Tail displacement vs threshold gap")
    axes[0, 1].axhline(0, color="0.3", lw=1)
    axes[0, 1].set(xlabel="KS(prospective idle, future idle)", ylabel="Prospective - oracle threshold", title="Context shift vs threshold gap")
    axes[0, 0].legend(fontsize=8)

    data, labels = [], []
    for key in ("2.0", "4.0"):
        data.append([row["targets"][key]["prospective_threshold_percentile_in_future_idle"] for row in rows])
        labels.append(f"FP {key}")
    axes[1, 0].boxplot(data, tick_labels=labels, showfliers=True)
    axes[1, 0].set(ylabel="Percentile in future idle", title="Where prospective threshold lands in future tail")
    axes[1, 0].axhline(0.99, color="0.5", ls="--", lw=1)

    prospective_neff = np.asarray([row["distribution"]["prospective_dependence"]["effective_n"] for row in rows])
    tail_neff = np.asarray([row["distribution"]["tail_dependence"]["effective_n"] for row in rows])
    axes[1, 1].boxplot(
        [0.01 * prospective_neff, 0.01 * tail_neff],
        tick_labels=["prospective", "future tail"], showfliers=True,
    )
    axes[1, 1].axhline(1, color="#b2182b", ls="--", lw=1)
    axes[1, 1].set(ylabel="Expected independent observations above Q99", title="Rare-tail resolution after autocorrelation")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Lee2019 CNN-GRU calibration-gap decomposition\n"
        "Subset20 x 2 sessions; evaluation-idle diagnostic (not deployable)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnn-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_scores_prospective"),
    )
    parser.add_argument(
        "--exposure-results", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_idle_exposure_curve_subset20.json"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_calibration_gap_subset20.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_calibration_gap_subset20.png"),
    )
    args = parser.parse_args()

    source = json.loads(args.exposure_results.read_text(encoding="utf-8"))
    by_key = {(row["subject"], row["session"]): row for row in source["rows"]}
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            key = (subject, session)
            rows.append(analyze_one(
                args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz",
                by_key[key],
            ))
            print(f"gap-audit S{session} sub-{subject:02d}", flush=True)

    payload = {
        "protocol": {
            "score": "frozen three-seed CNN-GRU; causal all-score EWMA H30",
            "decision_coordinate": "causal 0.5-s moving average before dwell/refractory",
            "prospective": "30-s pre-rest anchor plus known idle in first 30 trials",
            "oracle": "same evaluation-tail idle; non-deployable diagnostic",
            "evaluation": "fixed final 20 trials from prior exposure-curve experiment",
            "quantiles": QUANTILES,
            "dependence": "within-idle-chunk ACF, positive-sequence integrated time, max 5 s",
            "inference": "subject-cluster bootstrap keeps S1/S2 paired",
            "claim_limit": "subset20 diagnostic; evaluation idle labels are used",
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(rows, args.figure)
    print(args.output)
    print(args.figure)


if __name__ == "__main__":
    main()
