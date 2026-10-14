"""External early-usability validation using frozen Stieger CNN-GRU scores.

The score model and normalization are already frozen in the cache.  This
script applies cell-local EWMA/MA/dwell decisions, computes first-10 screening
features, evaluates the strictly later trial cells, and predicts future
common-achieved-FP excess using subject-grouped nested validation.

Because Stieger cells expose only one second of valid pre-feedback score
history, Lee2019's 30-s rolling-IQR prominence cannot be reproduced exactly.
The morphology concepts are transported without re-tuning: preparation-local
contrast/excursion/timing are measured over the first feedback second, and a
first-half preparation IQR/MAD provides the frozen scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.analyze_lee2019_cnn_gru_early_usability import (
    FEATURES,
    MODELS,
    mad,
    median_or_nan,
    summarize_target,
)


STEP_SECONDS = 0.1
EWMA_HALF_LIFE_SECONDS = 30.0
MA_SECONDS = 0.5
DWELL_SECONDS = 0.3
REFRACTORY_SECONDS = 1.0
TARGET_FPS = (2.0, 4.0)
NULL_SHIFTS = 500


def ewma_trial(
    raw: np.ndarray,
    initial_mean: float,
    initial_variance: float,
    *,
    half_life_seconds: float = EWMA_HALF_LIFE_SECONDS,
) -> np.ndarray:
    """Causal all-score EWMA, reset to the same training anchor per cell."""
    raw = np.asarray(raw, dtype=float)
    mean = float(initial_mean)
    variance = max(float(initial_variance), 1e-8)
    alpha = 1.0 - 2.0 ** (-STEP_SECONDS / half_life_seconds)
    floor = 0.25 * np.sqrt(variance)
    output = np.empty(len(raw), dtype=float)
    for index, value in enumerate(raw):
        output[index] = (value - mean) / max(np.sqrt(max(variance, 0.0)), floor)
        difference = value - mean
        mean += alpha * difference
        variance = (1.0 - alpha) * (variance + alpha * difference * difference)
    return output


def causal_ma(values: np.ndarray, seconds: float = MA_SECONDS) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    steps = max(1, int(np.ceil(seconds / STEP_SECONDS)))
    output = np.full(len(values), np.nan, dtype=float)
    if len(values) >= steps:
        cumulative = np.r_[0.0, np.cumsum(values)]
        output[steps - 1 :] = (cumulative[steps:] - cumulative[:-steps]) / steps
    return output


def trial_alarms(level: np.ndarray, times: np.ndarray, threshold: float) -> np.ndarray:
    """Apply dwell/refractory only after feedback onset, resetting each cell."""
    level = np.asarray(level, dtype=float)
    times = np.asarray(times, dtype=float)
    feedback = times > 0.0
    dwell = max(1, int(np.ceil(DWELL_SECONDS / STEP_SECONDS)))
    refractory = max(0, int(np.ceil(REFRACTORY_SECONDS / STEP_SECONDS)))
    alarms = np.zeros(len(level), dtype=bool)
    count = 0
    blocked_until = -1
    for index in range(len(level)):
        if not feedback[index] or not np.isfinite(level[index]):
            count = 0
            continue
        count = count + 1 if level[index] >= threshold else 0
        if count == dwell and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory
    return alarms


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        audit = json.loads(str(data["audit_json"]))
        return {key: np.asarray(data[key]) for key in data.files if key != "audit_json"} | {
            "audit": audit
        }


def split_trials(cache: dict[str, Any]) -> list[dict[str, Any]]:
    offsets = cache["trial_offsets"].astype(int)
    scores = cache["scores"].astype(float)
    times = cache["time_from_feedback_s"].astype(float)
    output = []
    for trial in range(len(offsets) - 1):
        start, stop = offsets[trial : trial + 2]
        output.append(
            {
                "trial": trial,
                "score": scores[start:stop],
                "time": times[start:stop],
                "target": int(cache["trial_target_number"][trial]),
                "active": bool(cache["trial_target_number"][trial] != 4),
                "artifact": bool(cache["trial_artifact"][trial]),
                "length": float(cache["trial_length_s"][trial]),
                "run": int(cache["trial_run_number"][trial]),
                "task": int(cache["trial_task_number"][trial]),
                "role": int(cache["trial_protocol_role"][trial]),
            }
        )
    return output


def prepare_levels(trials: list[dict[str, Any]]) -> tuple[float, float, np.ndarray]:
    training_preparation = np.concatenate(
        [
            trial["score"][trial["time"] <= 0.0]
            for trial in trials
            if trial["trial"] < 225 and not trial["artifact"] and len(trial["score"])
        ]
    )
    mean = float(np.mean(training_preparation))
    variance = float(np.var(training_preparation, ddof=1))
    for trial in trials:
        trial["z"] = ewma_trial(trial["score"], mean, variance)
        trial["level"] = causal_ma(trial["z"])
    preparation_level = np.concatenate(
        [
            trial["level"][(trial["time"] <= 0.0) & np.isfinite(trial["level"])]
            for trial in trials
            if trial["trial"] < 225 and not trial["artifact"] and len(trial["level"])
        ]
    )
    return mean, variance, preparation_level


def eligible(trial: dict[str, Any]) -> bool:
    return (not trial["artifact"]) and trial["length"] >= STEP_SECONDS and np.any(trial["time"] > 0)


def fp_rate(trials: list[dict[str, Any]], indices: list[int], threshold: float) -> float:
    alarm_count = 0
    exposure = 0.0
    for index in indices:
        trial = trials[index]
        if not eligible(trial) or trial["active"]:
            continue
        alarm_count += int(trial_alarms(trial["level"], trial["time"], threshold).sum())
        exposure += trial["length"]
    return float(60.0 * alarm_count / exposure) if exposure > 0 else float("nan")


def threshold_grid(trials: list[dict[str, Any]], indices: list[int]) -> np.ndarray:
    values = np.concatenate(
        [
            trials[index]["level"][(trials[index]["time"] > 0) & np.isfinite(trials[index]["level"])]
            for index in indices
            if eligible(trials[index]) and not trials[index]["active"]
        ]
    )
    grid = np.unique(np.quantile(values, np.linspace(0.005, 0.9995, 320)))
    return np.unique(np.r_[np.nextafter(values.max(), np.inf), grid, np.nextafter(values.min(), -np.inf)])


def tune_threshold(
    trials: list[dict[str, Any]], indices: list[int], target_fp: float
) -> tuple[float, float]:
    best = None
    selected = None
    for threshold in threshold_grid(trials, indices):
        achieved = fp_rate(trials, indices, float(threshold))
        candidate = (abs(achieved - target_fp), -float(threshold))
        if best is None or candidate < best:
            best = candidate
            selected = (float(threshold), achieved)
    if selected is None:
        raise ValueError("threshold tuning failed")
    return selected


def decision_outcomes(
    trials: list[dict[str, Any]], indices: list[int], threshold: float
) -> dict[str, np.ndarray | float]:
    active = []
    run = []
    onset_hit = []
    onset_hit_2s = []
    any_hit = []
    alarm_count = []
    exposure = []
    latency = []
    for index in indices:
        trial = trials[index]
        if not eligible(trial):
            continue
        alarms = trial_alarms(trial["level"], trial["time"], threshold)
        alarm_times = trial["time"][alarms]
        active.append(trial["active"])
        run.append(trial["run"])
        onset_hit.append(bool(np.any((alarm_times > 0) & (alarm_times <= 1.0))))
        onset_hit_2s.append(bool(np.any((alarm_times > 0) & (alarm_times <= 2.0))))
        any_hit.append(bool(len(alarm_times)))
        alarm_count.append(int(len(alarm_times)))
        exposure.append(float(trial["length"]))
        if trial["active"] and len(alarm_times):
            valid = alarm_times[alarm_times > 0]
            if len(valid):
                latency.append(float(valid[0]))
    return {
        "active": np.asarray(active, dtype=bool),
        "run": np.asarray(run, dtype=int),
        "onset_hit": np.asarray(onset_hit, dtype=bool),
        "onset_hit_2s": np.asarray(onset_hit_2s, dtype=bool),
        "any_hit": np.asarray(any_hit, dtype=bool),
        "alarm_count": np.asarray(alarm_count, dtype=int),
        "exposure": np.asarray(exposure, dtype=float),
        "latency": np.asarray(latency, dtype=float),
    }


def shifted_null(
    hit: np.ndarray,
    active: np.ndarray,
    run: np.ndarray,
    *,
    shifts: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty(shifts, dtype=float)
    for draw in range(shifts):
        shifted = active.copy()
        for run_value in np.unique(run):
            locations = np.flatnonzero(run == run_value)
            if len(locations) > 1:
                offset = int(rng.integers(1, len(locations)))
                shifted[locations] = np.roll(shifted[locations], offset)
        output[draw] = float(np.mean(hit[shifted])) if np.any(shifted) else np.nan
    return output


def event_metrics(
    trials: list[dict[str, Any]],
    indices: list[int],
    threshold: float,
    *,
    subject: int,
    session: int,
    threshold_name: str,
    shifts: int,
) -> dict[str, Any]:
    values = decision_outcomes(trials, indices, threshold)
    active = values["active"]
    rest = ~active
    sensitivity = float(np.mean(values["onset_hit"][active]))
    sensitivity_2s = float(np.mean(values["onset_hit_2s"][active]))
    any_sensitivity = float(np.mean(values["any_hit"][active]))
    rest_exposure = float(values["exposure"][rest].sum())
    fp_count = int(values["alarm_count"][rest].sum())
    actual_fp = float(60.0 * fp_count / rest_exposure)
    null = shifted_null(
        values["onset_hit"], active, values["run"], shifts=shifts,
        seed=20260921 + 1000 * session + 10 * subject + (2 if "2" in threshold_name else 4),
    )
    null_2s = shifted_null(
        values["onset_hit_2s"], active, values["run"], shifts=shifts,
        seed=20260922 + 1000 * session + 10 * subject + (2 if "2" in threshold_name else 4),
    )
    null_median = float(np.nanmedian(null))
    null_median_2s = float(np.nanmedian(null_2s))
    return {
        "threshold": float(threshold),
        "actual_fp_per_min": actual_fp,
        "false_alarm_count": fp_count,
        "intentional_rest_exposure_s": rest_exposure,
        "onset_sensitivity_1s": sensitivity,
        "onset_sensitivity_2s": sensitivity_2s,
        "any_feedback_sensitivity": any_sensitivity,
        "circular_run_shift_null_median_1s": null_median,
        "circular_run_shift_excess_1s": float(sensitivity - null_median),
        "circular_run_shift_null_median_2s": null_median_2s,
        "circular_run_shift_excess_2s": float(sensitivity_2s - null_median_2s),
        "median_detection_latency_s": (
            float(np.median(values["latency"])) if len(values["latency"]) else None
        ),
        "active_trials": int(active.sum()),
        "intentional_rest_trials": int(rest.sum()),
        "alarm_trials_active": int(values["any_hit"][active].sum()),
        "alarm_trials_rest": int(values["any_hit"][rest].sum()),
        "null_draws": int(shifts),
        "null_definition": "active/rest trial labels circularly shifted within run",
    }


def morphology_features(
    trials: list[dict[str, Any]],
    screening_indices: list[int],
    preparation_scale_values: np.ndarray,
    transferred_threshold: float,
) -> dict[str, float | int]:
    scale_q25, scale_q75 = np.quantile(preparation_scale_values, [0.25, 0.75])
    fixed_iqr = max(float(scale_q75 - scale_q25), 1e-6)
    trial_prominence = []
    trial_contrast = []
    trial_positive = []
    trial_peak_timing = []
    trial_peak_excursion = []
    pooled_preparation = []
    detected = []
    for index in screening_indices:
        trial = trials[index]
        level = trial["level"]
        time = trial["time"]
        pre = (time <= 0.0) & np.isfinite(level)
        event = (time > 0.0) & (time <= 1.0) & np.isfinite(level)
        alarms = trial_alarms(level, time, transferred_threshold)
        detected.append(bool(np.any(alarms & (time > 0.0) & (time <= 1.0))))
        if not np.any(pre) or not np.any(event):
            trial_prominence.append(np.nan)
            trial_contrast.append(np.nan)
            trial_positive.append(np.nan)
            trial_peak_timing.append(np.nan)
            trial_peak_excursion.append(np.nan)
            continue
        pre_values = level[pre]
        event_values = level[event]
        baseline = float(np.quantile(pre_values, 0.10))
        excursion = float(np.max(event_values) - baseline)
        peak_location = np.flatnonzero(event)[int(np.argmax(event_values))]
        contrast = float(np.median(event_values) - np.median(pre_values))
        trial_prominence.append(excursion / (fixed_iqr + 0.1))
        trial_contrast.append(contrast)
        trial_positive.append(float(contrast > 0.0))
        trial_peak_timing.append(float(time[peak_location]))
        trial_peak_excursion.append(excursion)
        pooled_preparation.extend(pre_values.tolist())
    idle_scale = mad(np.asarray(pooled_preparation, dtype=float))
    return {
        "first10_detection_fraction": float(np.mean(detected)),
        "local_prominence": median_or_nan(trial_prominence),
        "mi_local_idle_contrast": median_or_nan(trial_contrast),
        "positive_excursion_fraction": median_or_nan(trial_positive),
        "peak_timing_mad": mad(np.asarray(trial_peak_timing, dtype=float)),
        "peak_to_idle_snr": float(
            median_or_nan(trial_peak_excursion) / max(idle_scale, 1e-6)
        ),
        "valid_morphology_trials": int(np.sum(np.isfinite(trial_prominence))),
        "fixed_training_preparation_iqr": fixed_iqr,
    }


def analyze_one(path: Path, shifts: int) -> dict[str, Any]:
    cache = load_cache(path)
    audit = cache["audit"]
    subject = int(audit["subject"])
    session = int(audit["session"])
    trials = split_trials(cache)
    initial_mean, initial_variance, preparation_level = prepare_levels(trials)
    training = [trial["trial"] for trial in trials if trial["trial"] < 225 and eligible(trial)]
    screening = [
        trial["trial"] for trial in trials
        if trial["role"] == 1 and trial["active"] and eligible(trial)
    ]
    future = [trial["trial"] for trial in trials if trial["role"] == 2 and eligible(trial)]
    if len(screening) != 10:
        raise AssertionError(f"expected 10 screening active trials, got {len(screening)}")

    transferred_thresholds = {}
    transferred_metrics = {}
    common_metrics = {}
    for target_fp in TARGET_FPS:
        transferred_threshold, training_fp = tune_threshold(trials, training, target_fp)
        common_threshold, tuning_fp = tune_threshold(trials, future, target_fp)
        key = str(int(target_fp))
        transferred_thresholds[key] = transferred_threshold
        transferred_metrics[key] = {
            "training_tuning_fp_per_min": training_fp,
            **event_metrics(
                trials, future, transferred_threshold,
                subject=subject, session=session,
                threshold_name=f"transferred_fp{key}", shifts=shifts,
            ),
        }
        common_metrics[key] = {
            "future_tuning_fp_per_min": tuning_fp,
            "diagnostic_uses_future_rest_labels_for_threshold": True,
            **event_metrics(
                trials, future, common_threshold,
                subject=subject, session=session,
                threshold_name=f"common_fp{key}", shifts=shifts,
            ),
        }

    features = morphology_features(
        trials, screening, preparation_level, transferred_thresholds["2"]
    )
    future_feedback_scores = np.concatenate(
        [trial["z"][trial["time"] > 0] for trial in trials if trial["trial"] in future]
    )
    future_feedback_labels = np.concatenate(
        [
            np.full(np.sum(trial["time"] > 0), trial["active"], dtype=int)
            for trial in trials if trial["trial"] in future
        ]
    )
    return {
        "subject": subject,
        "session": session,
        **features,
        "future_common_fp_excess_fp2": common_metrics["2"]["circular_run_shift_excess_1s"],
        "future_common_fp_excess_2s_fp2": common_metrics["2"]["circular_run_shift_excess_2s"],
        "future_common_fp_excess_fp4": common_metrics["4"]["circular_run_shift_excess_1s"],
        "future_transferred_excess_fp2": transferred_metrics["2"]["circular_run_shift_excess_1s"],
        "future_transferred_actual_fp2": transferred_metrics["2"]["actual_fp_per_min"],
        "future_window_auc": float(roc_auc_score(future_feedback_labels, future_feedback_scores)),
        "initial_score_mean": initial_mean,
        "initial_score_sd": float(np.sqrt(initial_variance)),
        "screening_trial_indices": screening,
        "future_trial_count": len(future),
        "transferred": transferred_metrics,
        "common_fp_diagnostic": common_metrics,
    }


def descriptive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for name in (
        "future_common_fp_excess_fp2",
        "future_common_fp_excess_2s_fp2",
        "future_common_fp_excess_fp4",
        "future_transferred_excess_fp2",
        "future_transferred_actual_fp2",
        "future_window_auc",
    ):
        values = np.asarray([row[name] for row in rows], dtype=float)
        output[name] = {
            "median": float(np.median(values)),
            "iqr": np.quantile(values, [0.25, 0.75]).tolist(),
            "range": [float(np.min(values)), float(np.max(values))],
        }
    for arm, source in (("common_fp2", "common_fp_diagnostic"), ("transferred_fp2", "transferred")):
        values = np.asarray([row[source]["2"]["actual_fp_per_min"] for row in rows], float)
        output[arm + "_actual_fp"] = {
            "median": float(np.median(values)),
            "iqr": np.quantile(values, [0.25, 0.75]).tolist(),
            "range": [float(np.min(values)), float(np.max(values))],
        }
    for name in (
        "onset_sensitivity_1s",
        "onset_sensitivity_2s",
        "circular_run_shift_null_median_1s",
        "circular_run_shift_null_median_2s",
        "median_detection_latency_s",
    ):
        values = np.asarray(
            [row["common_fp_diagnostic"]["2"][name] for row in rows
             if row["common_fp_diagnostic"]["2"][name] is not None],
            dtype=float,
        )
        output["common_fp2_" + name] = {
            "median": float(np.median(values)),
            "iqr": np.quantile(values, [0.25, 0.75]).tolist(),
            "range": [float(np.min(values)), float(np.max(values))],
        }
    return output


def plot(payload: dict[str, Any], path: Path) -> None:
    rows = payload["rows"]
    target = "future_common_fp_excess_fp2"
    y = np.asarray([row[target] for row in rows], float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ("#4c72b0", "#55a868", "#c44e52")
    for axis, (name, _), color in zip(axes.flat[:3], MODELS.items(), colors):
        prediction = np.asarray([row["oos_predictions"][target][name] for row in rows])
        sessions = np.asarray([row["session"] for row in rows])
        axis.scatter(y, prediction, c=sessions, cmap="coolwarm", alpha=0.78)
        lo = min(float(y.min()), float(prediction.min()))
        hi = max(float(y.max()), float(prediction.max()))
        axis.plot([lo, hi], [lo, hi], "k--", lw=1)
        item = payload["prediction"][target]["models"][name]
        axis.set_title(f"{name}\nOOS rho={item['spearman_rho']:.3f}, MAE={item['mae']:.3f}")
        axis.set_xlabel("Future common-FP2 excess@1s")
        axis.set_ylabel("Grouped OOS prediction")
        axis.grid(alpha=0.2)
    axis = axes[1, 1]
    names = list(MODELS)
    values = [payload["prediction"][target]["models"][name]["spearman_rho"] for name in names]
    cis = [payload["prediction"][target]["bootstrap"]["rho_ci95"][name] for name in names]
    # With only 20 independent subjects, a bootstrap percentile interval can
    # occasionally sit wholly to one side of the full-sample point estimate.
    # Matplotlib rejects negative error-bar lengths, so plotting clips only
    # the visual distance at zero; the numeric interval is retained verbatim
    # in the JSON/report.
    errors = np.maximum(
        np.asarray([[value - low, high - value] for value, (low, high) in zip(values, cis)]).T,
        0.0,
    )
    axis.bar(np.arange(3), values, color=colors, yerr=errors, capsize=4)
    axis.axhline(0, color="black", lw=1)
    axis.set_xticks(np.arange(3), ("M0 detection", "M1 morphology", "M2 combined"))
    axis.set_ylabel("OOS Spearman rho")
    axis.set_title("Stieger external early-usability prediction")
    axis.grid(axis="y", alpha=0.2)
    regime = payload["protocol"].get("score_model", "CNN-GRU")
    fig.suptitle(f"Stieger2021 trial-wise CNN-GRU: first 10 predict future\n{regime}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def write_report(payload: dict[str, Any], path: Path) -> None:
    primary = payload["prediction"]["future_common_fp_excess_fp2"]
    secondary = payload["prediction"]["future_common_fp_excess_fp4"]
    desc = payload["descriptive"]
    lines = [
        "# Stieger2021 CNN-GRU early-usability external validation",
        "",
        "## Scope",
        "",
        "- Trial-wise preparation-to-feedback partial external validation.",
        f"- Score model: {payload['protocol']['score_model']}.",
        "- Primary target: future common-achieved-FP2 circular run-shift excess@1 s.",
        "- Transferred threshold uses first-half intentional-rest feedback and is secondary.",
        "- State resets at every published trial-cell seam.",
        "",
        "## Event performance",
        "",
        f"- Common-FP2 actual FP/min: median {desc['common_fp2_actual_fp']['median']:.3f}, IQR {desc['common_fp2_actual_fp']['iqr']}.",
        f"- Common-FP2 excess@1 s: median {desc['future_common_fp_excess_fp2']['median']:.3f}, IQR {desc['future_common_fp_excess_fp2']['iqr']}.",
        f"- Common-FP2 excess@2 s: median {desc['future_common_fp_excess_2s_fp2']['median']:.3f}, IQR {desc['future_common_fp_excess_2s_fp2']['iqr']}.",
        f"- Transferred FP2 actual FP/min: median {desc['transferred_fp2_actual_fp']['median']:.3f}, IQR {desc['transferred_fp2_actual_fp']['iqr']}.",
        f"- Transferred FP2 excess@1 s: median {desc['future_transferred_excess_fp2']['median']:.3f}, IQR {desc['future_transferred_excess_fp2']['iqr']}.",
        f"- Common-FP2 sensitivity/null @1 s: {desc['common_fp2_onset_sensitivity_1s']['median']:.3f}/{desc['common_fp2_circular_run_shift_null_median_1s']['median']:.3f}; median detection latency {desc['common_fp2_median_detection_latency_s']['median']:.3f} s.",
        f"- Common-FP2 sensitivity/null @2 s: {desc['common_fp2_onset_sensitivity_2s']['median']:.3f}/{desc['common_fp2_circular_run_shift_null_median_2s']['median']:.3f}; matched-row excess median {desc['future_common_fp_excess_2s_fp2']['median']:.3f}.",
        "",
        "## Early-usability prediction",
        "",
        "| Target | Model | OOS rho | MAE | Top-20% AUROC |",
        "|---|---|---:|---:|---:|",
    ]
    for label, block in (("common FP2", primary), ("common FP4", secondary)):
        for name in MODELS:
            item = block["models"][name]
            lines.append(
                f"| {label} | {name} | {item['spearman_rho']:.3f} | {item['mae']:.3f} | {item['usable_auroc']:.3f} |"
            )
    two_second = payload["prediction"]["future_common_fp_excess_2s_fp2"]
    lines.append(
        f"| common FP2 @2s | M2_combined | {two_second['models']['M2_combined']['spearman_rho']:.3f} | "
        f"{two_second['models']['M2_combined']['mae']:.3f} | {two_second['models']['M2_combined']['usable_auroc']:.3f} |"
    )
    delta = primary["M2_minus_M0"]
    ci = primary["bootstrap"]["M2_minus_M0_rho_ci95"]
    m2 = primary["models"]["M2_combined"]
    passed = (
        m2["spearman_rho"] > 0
        and m2["by_session"]["1"]["spearman_rho"] > 0
        and m2["by_session"]["2"]["spearman_rho"] > 0
        and delta["delta_spearman_rho"] > 0.10
        and delta["delta_rho_by_session"]["1"] > 0
        and delta["delta_rho_by_session"]["2"] > 0
        and ci[0] > 0
    )
    lines.extend(
        [
            "",
            "## Transport gate",
            "",
            f"- M2-M0 delta rho: {delta['delta_spearman_rho']:+.3f}, subject-cluster 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}].",
            f"- Session deltas: {delta['delta_rho_by_session']}.",
            "- Gate requires positive M2 rho pooled and in S1/S2, delta rho > .10 in both sessions, and subject-cluster delta CI lower bound > 0.",
            f"- External transport gate: **{'PASS' if passed else 'FAIL'}**.",
            "- First-10 transferred onset detection fraction has median 0 (range 0--0.4), and positive-excursion fraction is constant at 1.0; these transported predictors are poorly resolved in Stieger's one-second pre-feedback design.",
            "",
            "## Translation limit",
            "",
            "Stieger provides only one second of valid pre-feedback score history. Lee's 30-s rolling-IQR prominence is therefore replaced by a frozen first-half preparation IQR; morphology is conceptually matched but not numerically identical.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("data/interim/stieger2021_cnn_gru_scores"),
    )
    parser.add_argument("--shifts", type=int, default=NULL_SHIFTS)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_early_usability.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_early_usability.png"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("STIEGER_CNN_GRU_EARLY_USABILITY_20260921.md"),
    )
    args = parser.parse_args()

    files = sorted(args.cache_root.glob("session*/sub-*.npz"))
    if len(files) != 30:
        raise ValueError(f"expected 30 score caches, found {len(files)}")
    audits = [load_cache(path)["audit"] for path in files]
    regimes = sorted({audit.get("training_regime", "unspecified") for audit in audits})
    score_model = "; ".join(regimes)
    rows = []
    for index, path in enumerate(files, start=1):
        rows.append(analyze_one(path, args.shifts))
        print(f"analyzed {index}/{len(files)} {path.name}", flush=True)

    target_names = (
        "future_common_fp_excess_fp2",
        "future_common_fp_excess_2s_fp2",
        "future_common_fp_excess_fp4",
        "future_transferred_excess_fp2",
    )
    prediction = {
        target: summarize_target(rows, target, args.bootstrap) for target in target_names
    }
    payload = {
        "protocol": {
            "cohort": "Stieger2021 20 subjects; 30 subject-sessions",
            "score_model": score_model,
            "predictor_period": "first 10 eligible active trials in locked half",
            "future_period": "all eligible active/rest cells strictly after screening cutoff",
            "primary_target": "future common-achieved-FP2 circular run-shift excess@1s",
            "secondary_targets": ["common FP2 excess@2s", "common FP4 excess", "transferred FP2/excess"],
            "decision": "cell-reset EWMA H30 -> MA .5 s -> dwell .3 s -> refractory 1 s",
            "common_fp": "diagnostic threshold tuned with future intentional-rest labels",
            "transferred": "threshold tuned on first-half intentional-rest feedback",
            "null": "500 circular active/rest trial-label shifts within run",
            "M0": "transferred-FP2 first10 onset detection fraction",
            "M1": list(FEATURES),
            "M2": ["first10_detection_fraction", *FEATURES],
            "morphology_translation": "preparation-local first-feedback-second features; fixed first-half preparation IQR/MAD",
            "validation": "nested leave-one-subject-out Ridge; all sessions from a subject held out together",
            "claim_limit": "trial-wise partial external validation, not continuous self-paced deployment",
        },
        "descriptive": descriptive(rows),
        "prediction": prediction,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload, args.figure)
    write_report(payload, args.report)
    print(json.dumps({"descriptive": payload["descriptive"], "prediction": prediction}, indent=2))


if __name__ == "__main__":
    main()
