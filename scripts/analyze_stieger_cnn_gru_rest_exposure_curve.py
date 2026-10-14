"""Cumulative intentional-rest calibration exposure for Stieger CNN-GRU.

The frozen CNN-GRU cache is not recomputed.  For each subject-session, the
threshold is calibrated on increasing chronological amounts of intentional-
rest feedback from the first half, while the locked post-screening future set
is held fixed.  Exposures through 90 s form the complete-case primary curve;
longer exposures use explicitly reported supported risk sets.

Important limitation: first-half scores are in-sample with respect to the
within-subject CNN-GRU fit.  The curve therefore diagnoses threshold exposure
and transfer, not a deployable independent calibration protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.analyze_stieger_cnn_gru_early_usability import (
    eligible,
    event_metrics,
    load_cache,
    prepare_levels,
    split_trials,
    threshold_grid,
    trial_alarms,
    tune_threshold,
)


EXPOSURES_SECONDS = (15, 30, 60, 90, 120, 150, 180, 210, 240)
PRIMARY_EXPOSURES_SECONDS = (15, 30, 60, 90)
TARGET_FPS = (2.0, 4.0)
BOOTSTRAP_DRAWS = 5000


def calibration_alarm_table(
    trials: list[dict[str, Any]],
    indices: list[int],
    thresholds: np.ndarray,
) -> tuple[float, list[np.ndarray]]:
    """Return total rest exposure and cumulative-rest alarm positions.

    Each trial keeps its own preparation history and state reset.  Its
    post-onset alarm times are placed on an artificial cumulative-rest axis;
    this axis is used only to take exact exposure prefixes.
    """
    rest = [trials[index] for index in indices if eligible(trials[index]) and not trials[index]["active"]]
    positions: list[list[float]] = [[] for _ in thresholds]
    offset = 0.0
    for trial in rest:
        for threshold_index, threshold in enumerate(thresholds):
            mask = trial_alarms(trial["level"], trial["time"], float(threshold))
            alarm_times = trial["time"][mask]
            positions[threshold_index].extend((offset + alarm_times).tolist())
        offset += float(trial["length"])
    return offset, [np.sort(np.asarray(values, dtype=float)) for values in positions]


def select_threshold_for_exposure(
    thresholds: np.ndarray,
    alarm_positions: list[np.ndarray],
    exposure_seconds: float,
    target_fp: float,
) -> tuple[float, float, int]:
    """Choose closest alarm rate, resolving ties conservatively."""
    best: tuple[float, float] | None = None
    selected: tuple[float, float, int] | None = None
    for threshold, positions in zip(thresholds, alarm_positions):
        count = int(np.searchsorted(positions, exposure_seconds, side="right"))
        achieved = float(60.0 * count / exposure_seconds)
        candidate = (abs(achieved - target_fp), -float(threshold))
        if best is None or candidate < best:
            best = candidate
            selected = (float(threshold), achieved, count)
    if selected is None:
        raise ValueError("no calibration threshold candidates")
    return selected


def analyze_one(path: Path, shifts: int) -> dict[str, Any]:
    cache = load_cache(path)
    audit = cache["audit"]
    subject = int(audit["subject"])
    session = int(audit["session"])
    trials = split_trials(cache)
    prepare_levels(trials)

    training = [trial["trial"] for trial in trials if trial["trial"] < 225 and eligible(trial)]
    future = [trial["trial"] for trial in trials if trial["role"] == 2 and eligible(trial)]
    training_rest = [index for index in training if not trials[index]["active"]]
    thresholds = threshold_grid(trials, training_rest)
    available_seconds, alarm_positions = calibration_alarm_table(
        trials, training_rest, thresholds
    )

    metric_cache: dict[float, dict[str, Any]] = {}
    arms: dict[str, Any] = {}
    for exposure in EXPOSURES_SECONDS:
        if available_seconds + 1e-8 < exposure:
            continue
        arm: dict[str, Any] = {
            "requested_exposure_seconds": float(exposure),
            "available_training_rest_seconds": float(available_seconds),
        }
        for target_fp in TARGET_FPS:
            threshold, calibration_fp, calibration_alarms = select_threshold_for_exposure(
                thresholds, alarm_positions, float(exposure), target_fp
            )
            if threshold not in metric_cache:
                metric_cache[threshold] = event_metrics(
                    trials,
                    future,
                    threshold,
                    subject=subject,
                    session=session,
                    threshold_name=f"exposure_{exposure}_{target_fp:g}",
                    shifts=shifts,
                )
            metrics = dict(metric_cache[threshold])
            metrics.update(
                target_fp_per_min=float(target_fp),
                calibration_achieved_fp_per_min=float(calibration_fp),
                calibration_alarm_count=int(calibration_alarms),
                operating_point_log_error=float(
                    abs(np.log((metrics["actual_fp_per_min"] + 0.05) / (target_fp + 0.05)))
                ),
            )
            arm[str(int(target_fp))] = metrics
        arms[str(exposure)] = arm

    oracle: dict[str, Any] = {}
    for target_fp in TARGET_FPS:
        threshold, tuning_fp = tune_threshold(trials, future, target_fp)
        metrics = event_metrics(
            trials,
            future,
            threshold,
            subject=subject,
            session=session,
            threshold_name=f"future_oracle_{target_fp:g}",
            shifts=shifts,
        )
        metrics.update(
            target_fp_per_min=float(target_fp),
            future_tuning_fp_per_min=float(tuning_fp),
            operating_point_log_error=float(
                abs(np.log((metrics["actual_fp_per_min"] + 0.05) / (target_fp + 0.05)))
            ),
            diagnostic_uses_future_rest_labels=True,
        )
        oracle[str(int(target_fp))] = metrics

    return {
        "subject": subject,
        "session": session,
        "training_rest_trials": len(training_rest),
        "available_training_rest_seconds": float(available_seconds),
        "future_trials": len(future),
        "arms": arms,
        "future_common_fp_oracle": oracle,
    }


def metric_value(row: dict[str, Any], exposure: int, target: int, metric: str) -> float:
    return float(row["arms"][str(exposure)][str(target)][metric])


def cluster_bootstrap_median(
    rows: list[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[float]:
    subjects = np.asarray(sorted({int(row["subject"]) for row in rows}), dtype=int)
    grouped = {subject: [row for row in rows if int(row["subject"]) == subject] for subject in subjects}
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        draw_values = [getter(row) for subject in sampled for row in grouped[int(subject)]]
        values[draw] = np.median(draw_values)
    return np.quantile(values, [0.025, 0.975]).tolist()


def summarize_metric(
    rows: list[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
    *,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray([getter(row) for row in rows], dtype=float)
    return {
        "median": float(np.median(values)),
        "iqr": np.quantile(values, [0.25, 0.75]).tolist(),
        "range": [float(np.min(values)), float(np.max(values))],
        "subject_cluster_ci95": cluster_bootstrap_median(rows, getter, seed=seed),
    }


def summarize_exposure(
    all_rows: list[dict[str, Any]], exposure: int, target: int, *, seed: int
) -> dict[str, Any]:
    rows = [row for row in all_rows if str(exposure) in row["arms"]]
    getters = {
        "threshold": lambda row: metric_value(row, exposure, target, "threshold"),
        "calibration_achieved_fp_per_min": lambda row: metric_value(
            row, exposure, target, "calibration_achieved_fp_per_min"
        ),
        "actual_fp_per_min": lambda row: metric_value(row, exposure, target, "actual_fp_per_min"),
        "operating_point_log_error": lambda row: metric_value(
            row, exposure, target, "operating_point_log_error"
        ),
        "sensitivity_1s": lambda row: metric_value(row, exposure, target, "onset_sensitivity_1s"),
        "excess_1s": lambda row: metric_value(row, exposure, target, "circular_run_shift_excess_1s"),
        "sensitivity_2s": lambda row: metric_value(row, exposure, target, "onset_sensitivity_2s"),
        "excess_2s": lambda row: metric_value(row, exposure, target, "circular_run_shift_excess_2s"),
    }
    summary: dict[str, Any] = {
        "n_subject_sessions": len(rows),
        "n_subjects": len({row["subject"] for row in rows}),
        "complete_case_primary": exposure in PRIMARY_EXPOSURES_SECONDS,
    }
    for index, (name, getter) in enumerate(getters.items()):
        summary[name] = summarize_metric(rows, getter, seed=seed + index * 100)
    lower, upper = ((1.5, 2.5) if target == 2 else (3.0, 5.0))
    actual = np.asarray([getters["actual_fp_per_min"](row) for row in rows])
    summary["admissible_n"] = int(np.sum((actual >= lower) & (actual <= upper)))
    summary["admissible_fraction"] = float(np.mean((actual >= lower) & (actual <= upper)))

    # The 15-s point exposes the alarm-count quantization floor; 30 s is the
    # first point at which FP=2/min can be represented by exactly one alarm.
    # Preserve both paired baselines so apparent cohort-median improvements at
    # long exposure can be distinguished from within-session changes.
    for baseline_index, baseline_exposure in enumerate((15, 30)):
        if exposure == baseline_exposure or exposure < baseline_exposure:
            continue
        for index, metric in enumerate(
            ("actual_fp_per_min", "operating_point_log_error", "excess_1s", "excess_2s")
        ):
            current = getters[metric]
            baseline = lambda row, name=metric, base_exp=baseline_exposure: metric_value(
                row, base_exp, target, name if not name.startswith("excess") else (
                    "circular_run_shift_excess_1s" if name == "excess_1s" else "circular_run_shift_excess_2s"
                )
            )
            delta_getter = lambda row, cur=current, base=baseline: cur(row) - base(row)
            summary[f"paired_delta_vs_{baseline_exposure}s_{metric}"] = summarize_metric(
                rows,
                delta_getter,
                seed=seed + 5000 + baseline_index * 1000 + index * 100,
            )
    return summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"exposures": {}, "oracle": {}}
    for exposure in EXPOSURES_SECONDS:
        supported = [row for row in rows if str(exposure) in row["arms"]]
        if not supported:
            continue
        output["exposures"][str(exposure)] = {
            str(int(target)): summarize_exposure(
                rows, exposure, int(target), seed=202609210 + exposure * 100 + int(target)
            )
            for target in TARGET_FPS
        }
    for target in TARGET_FPS:
        key = str(int(target))
        oracle_getters = {
            "actual_fp_per_min": lambda row, k=key: float(
                row["future_common_fp_oracle"][k]["actual_fp_per_min"]
            ),
            "excess_1s": lambda row, k=key: float(
                row["future_common_fp_oracle"][k]["circular_run_shift_excess_1s"]
            ),
            "excess_2s": lambda row, k=key: float(
                row["future_common_fp_oracle"][k]["circular_run_shift_excess_2s"]
            ),
        }
        output["oracle"][key] = {
            name: summarize_metric(rows, getter, seed=202609900 + int(target) * 100 + index)
            for index, (name, getter) in enumerate(oracle_getters.items())
        }
    return output


def plot_summary(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]["exposures"]
    exposures = np.asarray([int(value) for value in summary], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    colors = {"2": "#4c72b0", "4": "#dd8452"}
    for target in ("2", "4"):
        color = colors[target]
        fp = np.asarray([summary[str(int(e))][target]["actual_fp_per_min"]["median"] for e in exposures])
        fp_iqr = np.asarray([summary[str(int(e))][target]["actual_fp_per_min"]["iqr"] for e in exposures])
        axes[0, 0].plot(exposures, fp, marker="o", color=color, label=f"target {target}/min")
        axes[0, 0].fill_between(exposures, fp_iqr[:, 0], fp_iqr[:, 1], color=color, alpha=.15)
        axes[0, 0].axhline(float(target), color=color, ls="--", alpha=.55)

        admissible = np.asarray([summary[str(int(e))][target]["admissible_fraction"] for e in exposures])
        axes[0, 1].plot(exposures, admissible, marker="o", color=color, label=f"target {target}/min")

        error = np.asarray([summary[str(int(e))][target]["operating_point_log_error"]["median"] for e in exposures])
        axes[1, 0].plot(exposures, error, marker="o", color=color, label=f"target {target}/min")

        excess = np.asarray([summary[str(int(e))][target]["excess_2s"]["median"] for e in exposures])
        axes[1, 1].plot(exposures, excess, marker="o", color=color, label=f"target {target}/min")

    n = [summary[str(int(e))]["2"]["n_subject_sessions"] for e in exposures]
    for axis in axes.flat:
        axis.axvline(90, color="0.35", ls=":", lw=1)
        axis.grid(alpha=.2)
        axis.legend()
        axis.set_xticks(exposures)
    axes[0, 0].set_ylabel("Future achieved FP/min (median, IQR)")
    axes[0, 1].set_ylabel("Admissible fraction")
    axes[0, 1].set_ylim(-.02, 1.02)
    axes[1, 0].set_ylabel("Absolute log FP error (median)")
    axes[1, 1].set_ylabel("Circular-shift excess@2s (median)")
    axes[1, 0].set_xlabel("Cumulative training-rest exposure (s)")
    axes[1, 1].set_xlabel("Cumulative training-rest exposure (s)")
    axes[0, 0].set_title("Operating-point transfer")
    axes[0, 1].set_title("Sessions within admissible FP band")
    axes[1, 0].set_title("Transfer error")
    axes[1, 1].set_title("Event information at transferred threshold")
    fig.suptitle(
        "Stieger2021 frozen CNN-GRU: cumulative intentional-rest calibration\n"
        "15--90 s complete cohort; >90 s supported risk sets; n=" + "/".join(map(str, n)),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]["exposures"]
    oracle = payload["summary"]["oracle"]
    lines = [
        "# Stieger2021 CNN-GRU cumulative intentional-rest exposure curve",
        "",
        "## Scope",
        "",
        "- Frozen within-subject CNN-GRU scores; fixed post-screening future evaluation set.",
        "- Chronological first-half intentional-rest feedback is accumulated to set the threshold.",
        "- 15/30/60/90 s are complete-case primary points; longer points use supported risk sets.",
        "- First-half calibration scores are in-sample with model fitting, so this is an optimistic threshold-transfer diagnostic.",
        "",
        "## Combined results",
        "",
        "| Exposure | n | Target | Calibration FP/min | Future FP/min | Admissible | Log error | Excess@2s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exposure_text, block in summary.items():
        for target in ("2", "4"):
            item = block[target]
            lines.append(
                f"| {exposure_text} | {item['n_subject_sessions']} | {target} | "
                f"{item['calibration_achieved_fp_per_min']['median']:.3f} | "
                f"{item['actual_fp_per_min']['median']:.3f} | "
                f"{item['admissible_n']}/{item['n_subject_sessions']} | "
                f"{item['operating_point_log_error']['median']:.3f} | "
                f"{item['excess_2s']['median']:.3f} |"
            )
    fp2_90 = summary["90"]["2"]["paired_delta_vs_30s_operating_point_log_error"]
    fp4_90 = summary["90"]["4"]["paired_delta_vs_30s_operating_point_log_error"]
    fp2_150 = summary.get("150", {}).get("2", {}).get(
        "paired_delta_vs_30s_operating_point_log_error"
    )
    lines.extend(
        [
            "",
            "## Paired interpretation",
            "",
            f"- Complete-case 30->90 s FP2 log-error delta: median {fp2_90['median']:+.3f}, cluster CI {fp2_90['subject_cluster_ci95']}.",
            f"- Complete-case 30->90 s FP4 log-error delta: median {fp4_90['median']:+.3f}, cluster CI {fp4_90['subject_cluster_ci95']}.",
            *(
                [f"- At 150 s (supported n={summary['150']['2']['n_subject_sessions']}), FP2 delta versus the same sessions at 30 s was {fp2_150['median']:+.3f}, cluster CI {fp2_150['subject_cluster_ci95']}."]
                if fp2_150 is not None else []
            ),
            "- A negative log-error delta is improvement. Confidence intervals touching zero are not treated as confirmed gain.",
            f"- Future-rest oracle achieved {oracle['2']['actual_fp_per_min']['median']:.3f}/min at target 2 and {oracle['4']['actual_fp_per_min']['median']:.3f}/min at target 4; its excess@2s was {oracle['2']['excess_2s']['median']:.3f}/{oracle['4']['excess_2s']['median']:.3f}.",
            "",
            "## Interpretation boundary",
            "",
            "The curve tests whether more known intentional-rest exposure improves threshold transfer while the future stream is fixed. It does not test continuous natural idle, independent calibration, or cross-subject deployment.",
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
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_rest_exposure_curve.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_rest_exposure_curve.png"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("STIEGER_CNN_GRU_REST_EXPOSURE_CURVE_20260921.md"),
    )
    args = parser.parse_args()

    files = sorted(args.cache_root.glob("session*/sub-*.npz"))
    if len(files) != 30:
        raise ValueError(f"expected 30 score caches, found {len(files)}")
    rows = []
    for index, path in enumerate(files, start=1):
        rows.append(analyze_one(path, args.shifts))
        print(f"rest-exposure {index}/{len(files)} {path.parent.name}/{path.name}", flush=True)
    payload = {
        "protocol": {
            "cohort": "Stieger2021: 20 S1 plus 10 S2 subject-sessions",
            "score": "frozen three-seed CNN-GRU; chronological within-subject first-half fit",
            "normalization": "cell-reset causal EWMA H30 initialized from first-half preparation",
            "decision": "MA .5 s, dwell .3 s, refractory 1 s; alarms disabled before feedback",
            "calibration": "chronological cumulative first-half intentional-rest feedback",
            "exposures_seconds": list(EXPOSURES_SECONDS),
            "primary_complete_case_exposures_seconds": list(PRIMARY_EXPOSURES_SECONDS),
            "evaluation": "fixed locked post-first10 future cells",
            "targets_fp_per_min": list(TARGET_FPS),
            "chance": "500 circular active/rest label shifts within run",
            "oracle": "future rest sets threshold and is diagnostic only",
            "claim_limit": "trial-wise partial validation; calibration scores are in-sample with CNN fit",
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_summary(payload, args.figure)
    write_report(payload, args.report)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
