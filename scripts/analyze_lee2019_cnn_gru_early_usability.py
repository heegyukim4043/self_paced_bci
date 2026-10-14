"""Predict future CNN-GRU event usability from the first ten opportunities.

Predictors are computed from the first ten known MI opportunities.  The target
is the locked diagnostic matched-achieved-FP circular-shift excess on the
remaining 40 opportunities.  Both sessions from a subject are always held out
together.  This is a screening/usability analysis, not a deployable threshold
calibration method.
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
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.compare_lee2019_cnn_gru_local_trough import (
    causal_local_trough,
    causal_ma,
)
from scripts.compare_lee2019_cnn_gru_normalized_prominence import rolling_iqr
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma,
    ewma_z,
    ma_trace,
)
from scripts.run_lee2019_cnn_gru_loso_arm_a_stream_v2 import (
    SFREQ,
    STEP,
    _calibration_threshold,
    _zscore,
)


FEATURES = (
    "local_prominence",
    "mi_local_idle_contrast",
    "positive_excursion_fraction",
    "peak_timing_mad",
    "peak_to_idle_snr",
)
MODELS = {
    "M0_detection_fraction": ("first10_detection_fraction",),
    "M1_morphology": FEATURES,
    "M2_combined": ("first10_detection_fraction", *FEATURES),
}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def median_or_nan(values: list[float]) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def load_future_targets(path: Path) -> dict[tuple[int, int, float], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload["rows"]:
        target = float(row["target_fp_per_min"])
        if target not in (2.0, 4.0):
            continue
        key = (int(row["subject"]), int(row["session"]), target)
        result[key] = float(row["cnn_gru"]["circular_shift_excess"]["1.0"])
    return result


def first10_features(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as data:
        subject = int(data["subject"])
        session = int(data["session"])
        times = np.asarray(data["times"], float)
        allowed = np.asarray(data["allowed"], bool)
        events = np.asarray(data["events"], float)
        raw = np.asarray(data["raw_scores"], float)
        anchor = np.asarray(data["anchor_scores"], float)

    # All predictors stop at the end of opportunity 10.  Causal EWMA and
    # rolling morphology before this boundary are exactly invariant to
    # discarding the remaining 40-opportunity suffix.
    cues = events[:, 5] / SFREQ
    keep = times <= cues[9] + 4.0
    times = times[keep]
    allowed = allowed[keep]
    raw = raw[keep]

    mean = float(np.mean(anchor))
    variance = float(np.var(anchor, ddof=1))
    z = ewma_z(raw, mean, variance, allowed)
    anchor_z = _zscore(anchor)

    # Historical screening comparator: prospective pre-rest FP=2 threshold,
    # final alarm logic, and cue+1.5--4.0 response window.
    threshold, _ = _calibration_threshold([anchor_z], 2.0)
    screen_alarms = alarms_from_ma(
        ma_trace(z, n=int(round(0.5 / STEP))), threshold, allowed,
        dwell=int(round(0.3 / STEP)),
        refractory=int(round(1.0 / STEP)),
    )
    detected = [
        np.any(
            screen_alarms
            & allowed
            & (times >= cue + 1.5)
            & (times < cue + 4.0)
        )
        for cue in cues[:10]
    ]

    # Prepend completed target pre-rest so all rolling quantities are causal
    # and already defined at the first stream opportunity.
    combined = np.r_[anchor_z, z]
    level_all = causal_ma(combined, 0.5)
    local_all, _ = causal_local_trough(level_all, 2.0, 0.5, 0.10)
    scale_all = rolling_iqr(level_all, 30.0, 0.5)
    prominence_all = local_all / (scale_all + 0.1)
    offset = len(anchor_z)
    level = level_all[offset:]
    prominence = prominence_all[offset:]

    trial_prominence: list[float] = []
    trial_contrast: list[float] = []
    trial_positive: list[float] = []
    trial_peak_timing: list[float] = []
    trial_peak_excursion: list[float] = []
    pooled_idle: list[float] = []

    for event, cue in zip(events[:10], cues[:10]):
        onset = cue + 0.5
        mi = allowed & (times >= onset) & (times <= cue + 4.0)
        # Last two valid idle seconds before the cue/fixation seam.  The
        # intervening three-second guard is deliberately not used as idle.
        idle_stop = event[4] / SFREQ
        pre = allowed & (times >= idle_stop - 2.0) & (times < idle_stop)
        mi_values = level[mi & np.isfinite(level)]
        idle_values = level[pre & np.isfinite(level)]
        prom_indices = np.flatnonzero(mi & np.isfinite(prominence))
        if not len(mi_values) or not len(idle_values) or not len(prom_indices):
            trial_prominence.append(float("nan"))
            trial_contrast.append(float("nan"))
            trial_positive.append(float("nan"))
            trial_peak_timing.append(float("nan"))
            trial_peak_excursion.append(float("nan"))
            continue
        prom_values = prominence[prom_indices]
        peak_index = int(prom_indices[int(np.argmax(prom_values))])
        contrast = float(np.median(mi_values) - np.median(idle_values))
        idle_floor = float(np.quantile(idle_values, 0.10))
        trial_prominence.append(float(np.max(prom_values)))
        trial_contrast.append(contrast)
        trial_positive.append(float(contrast > 0.0))
        trial_peak_timing.append(float(times[peak_index] - onset))
        trial_peak_excursion.append(float(np.max(mi_values) - idle_floor))
        pooled_idle.extend(idle_values.tolist())

    idle_scale = mad(np.asarray(pooled_idle, float))
    snr = median_or_nan(trial_peak_excursion) / max(idle_scale, 1e-6)
    return {
        "subject": subject,
        "session": session,
        "first10_detection_fraction": float(np.mean(detected)),
        "local_prominence": median_or_nan(trial_prominence),
        "mi_local_idle_contrast": median_or_nan(trial_contrast),
        "positive_excursion_fraction": median_or_nan(trial_positive),
        "peak_timing_mad": mad(np.asarray(trial_peak_timing, float)),
        "peak_to_idle_snr": float(snr),
        "prospective_screen_threshold_fp2": float(threshold),
        "valid_morphology_trials": int(np.sum(np.isfinite(trial_prominence))),
    }


def pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def select_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    folds = GroupKFold(n_splits=min(5, len(unique)))
    losses = []
    for alpha in ALPHAS:
        fold_loss = []
        for train, valid in folds.split(x, y, groups):
            model = pipeline(alpha).fit(x[train], y[train])
            fold_loss.append(mean_absolute_error(y[valid], model.predict(x[valid])))
        losses.append(float(np.mean(fold_loss)))
    return float(ALPHAS[int(np.argmin(losses))])


def nested_group_loso(
    rows: list[dict], feature_names: tuple[str, ...], target_name: str
) -> tuple[np.ndarray, list[float]]:
    x = np.asarray([[row[name] for name in feature_names] for row in rows], float)
    y = np.asarray([row[target_name] for row in rows], float)
    groups = np.asarray([row["subject"] for row in rows], int)
    predictions = np.full(len(rows), np.nan, float)
    selected = []
    for subject in np.unique(groups):
        test = groups == subject
        train = ~test
        alpha = select_alpha(x[train], y[train], groups[train])
        predictions[test] = pipeline(alpha).fit(x[train], y[train]).predict(x[test])
        selected.append(alpha)
    return predictions, selected


def rho(x: np.ndarray, y: np.ndarray) -> float:
    value = spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic
    return float(value) if np.isfinite(value) else float("nan")


def metric_summary(y: np.ndarray, prediction: np.ndarray) -> dict:
    cutoff = float(np.quantile(y, 0.80))
    usable = y >= cutoff
    return {
        "spearman_rho": rho(prediction, y),
        "mae": float(mean_absolute_error(y, prediction)),
        "usable_definition": "cohort-relative top 20% future excess",
        "usable_cutoff": cutoff,
        "usable_n": int(np.sum(usable)),
        "usable_auroc": float(roc_auc_score(usable, prediction)),
    }


def cluster_bootstrap(
    rows: list[dict], y: np.ndarray, predictions: dict[str, np.ndarray],
    repeats: int, seed: int,
) -> dict:
    groups = np.asarray([row["subject"] for row in rows], int)
    unique = np.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    model_rho = {name: [] for name in predictions}
    delta = []
    for _ in range(repeats):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([lookup[group] for group in sampled])
        values = {name: rho(pred[indices], y[indices]) for name, pred in predictions.items()}
        for name, value in values.items():
            model_rho[name].append(value)
        delta.append(values["M2_combined"] - values["M0_detection_fraction"])
    return {
        "rho_ci95": {
            name: np.nanquantile(values, [0.025, 0.975]).tolist()
            for name, values in model_rho.items()
        },
        "M2_minus_M0_rho_ci95": np.nanquantile(delta, [0.025, 0.975]).tolist(),
    }


def summarize_target(rows: list[dict], target_name: str, repeats: int) -> dict:
    y = np.asarray([row[target_name] for row in rows], float)
    predictions = {}
    alphas = {}
    for name, feature_names in MODELS.items():
        predictions[name], selected = nested_group_loso(rows, feature_names, target_name)
        alphas[name] = {
            str(alpha): int(np.sum(np.isclose(selected, alpha))) for alpha in ALPHAS
        }

    result = {
        "n_subject_sessions": len(rows),
        "n_subjects": len({row["subject"] for row in rows}),
        "target_median": float(np.median(y)),
        "target_iqr": np.quantile(y, [0.25, 0.75]).tolist(),
        "models": {},
        "feature_univariate_spearman": {},
    }
    for name, prediction in predictions.items():
        item = metric_summary(y, prediction)
        item["selected_alpha_counts"] = alphas[name]
        item["by_session"] = {}
        for session in (1, 2):
            keep = np.asarray([row["session"] == session for row in rows])
            item["by_session"][str(session)] = metric_summary(y[keep], prediction[keep])
        result["models"][name] = item
    result["M2_minus_M0"] = {
        "delta_spearman_rho": (
            result["models"]["M2_combined"]["spearman_rho"]
            - result["models"]["M0_detection_fraction"]["spearman_rho"]
        ),
        "delta_mae": (
            result["models"]["M2_combined"]["mae"]
            - result["models"]["M0_detection_fraction"]["mae"]
        ),
        "delta_rho_by_session": {
            str(session): (
                result["models"]["M2_combined"]["by_session"][str(session)]["spearman_rho"]
                - result["models"]["M0_detection_fraction"]["by_session"][str(session)]["spearman_rho"]
            ) for session in (1, 2)
        },
    }
    result["bootstrap"] = cluster_bootstrap(
        rows, y, predictions, repeats, 202609150 + (2 if target_name.endswith("fp2") else 4)
    )
    for feature in ("first10_detection_fraction", *FEATURES):
        values = np.asarray([row[feature] for row in rows], float)
        valid = np.isfinite(values)
        result["feature_univariate_spearman"][feature] = rho(values[valid], y[valid])
    for index, row in enumerate(rows):
        row.setdefault("oos_predictions", {})[target_name] = {
            name: float(prediction[index]) for name, prediction in predictions.items()
        }
    return result


def plot(payload: dict, output: Path) -> None:
    rows = payload["rows"]
    target = "future_excess_fp2"
    y = np.asarray([row[target] for row in rows], float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ("#4c72b0", "#55a868", "#c44e52")
    for ax, (name, _), color in zip(axes.flat[:3], MODELS.items(), colors):
        pred = np.asarray([row["oos_predictions"][target][name] for row in rows])
        ax.scatter(y, pred, c=[row["session"] for row in rows], cmap="coolwarm",
                   alpha=0.72, edgecolor="none")
        lo = min(float(np.min(y)), float(np.min(pred)))
        hi = max(float(np.max(y)), float(np.max(pred)))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        item = payload["summary"][target]["models"][name]
        ax.set_title(f"{name}\nOOS rho={item['spearman_rho']:.3f}, MAE={item['mae']:.3f}")
        ax.set_xlabel("Future excess@FP2 (locked diagnostic)")
        ax.set_ylabel("Subject-grouped OOS prediction")
        ax.grid(alpha=0.2)
    ax = axes[1, 1]
    names = list(MODELS)
    values = [payload["summary"][target]["models"][name]["spearman_rho"] for name in names]
    ci = [payload["summary"][target]["bootstrap"]["rho_ci95"][name] for name in names]
    errors = np.asarray([[value-low, high-value] for value, (low, high) in zip(values, ci)]).T
    ax.bar(np.arange(3), values, color=colors, yerr=errors, capsize=4)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(np.arange(3), ("M0 detection", "M1 morphology", "M2 combined"))
    ax.set_ylabel("OOS Spearman rho")
    ax.set_title("Primary future usability prediction")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("CNN-GRU early-session usability — first 10 predict future 40", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def write_report(payload: dict, path: Path) -> None:
    p2 = payload["summary"]["future_excess_fp2"]
    p4 = payload["summary"]["future_excess_fp4"]
    lines = [
        "# CNN-GRU early-session usability prediction (Lee2019)", "",
        "## Locked design", "",
        "- Predictors: first 10 known MI opportunities only.",
        "- Outcome: subsequent 40-trial circular-shift excess at diagnostic matched achieved FP.",
        "- Primary: excess@1 s at FP=2/min; secondary: FP=4/min.",
        "- Validation: nested leave-one-subject-out; both sessions held out together.",
        "- M0: historical prospective-threshold first-10 detection fraction.",
        "- M1: five threshold-independent morphology features.",
        "- M2: M0 plus morphology.", "",
        "## Results", "",
        "| Target | Model | OOS rho | MAE | Top-20% AUROC |", "|---|---|---:|---:|---:|",
    ]
    for label, block in (("FP2", p2), ("FP4", p4)):
        for name in MODELS:
            item = block["models"][name]
            lines.append(f"| {label} | {name} | {item['spearman_rho']:.3f} | {item['mae']:.3f} | {item['usable_auroc']:.3f} |")
    lines += ["", "## Incremental value of morphology", ""]
    for label, block in (("FP2", p2), ("FP4", p4)):
        delta = block["M2_minus_M0"]
        ci = block["bootstrap"]["M2_minus_M0_rho_ci95"]
        lines.append(
            f"- {label}: delta rho={delta['delta_spearman_rho']:+.3f} "
            f"(subject-cluster bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]); "
            f"delta MAE={delta['delta_mae']:+.3f}; "
            f"session deltas={delta['delta_rho_by_session']}"
        )
    d2 = p2["M2_minus_M0"]["delta_spearman_rho"]
    s2 = p2["M2_minus_M0"]["delta_rho_by_session"]
    passed = d2 > 0.10 and s2["1"] > 0 and s2["2"] > 0
    lines += ["", "## Pre-specified interpretation", "",
              f"- Practical incremental gate (delta rho > 0.10 and same positive direction in S1/S2): **{'PASS' if passed else 'FAIL'}**.",
              "- This analysis tests session-usability prediction. It does not repair prospective FP calibration and does not estimate true self-initiated performance."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("data/interim/lee2019_cnn_gru_loso_arm_a_stream_v2_score_cache"),
    )
    parser.add_argument(
        "--matched-results", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_vs_riemannian_matched_fp.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_early_usability_full54.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_early_usability_full54.png"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("CNN_GRU_EARLY_SESSION_USABILITY_20260915.md"),
    )
    args = parser.parse_args()

    targets = load_future_targets(args.matched_results)
    files = sorted(args.cache_root.glob("session*/sub-*.npz"))
    rows = []
    for index, file in enumerate(files, 1):
        row = first10_features(file)
        row["future_excess_fp2"] = targets[(row["subject"], row["session"], 2.0)]
        row["future_excess_fp4"] = targets[(row["subject"], row["session"], 4.0)]
        rows.append(row)
        if index % 12 == 0 or index == len(files):
            print(f"features {index}/{len(files)}", flush=True)

    summary = {
        target: summarize_target(rows, target, args.bootstrap)
        for target in ("future_excess_fp2", "future_excess_fp4")
    }
    payload = {
        "protocol": {
            "cohort": "Lee2019 full 54 subjects x 2 sessions",
            "predictor_period": "first 10 known MI opportunities",
            "future_period": "subsequent 40 opportunities",
            "future_target": "CNN-GRU circular-shift excess@1s at diagnostic matched achieved FP; existing locked 500-shift result",
            "M0": "prospective target-pre-rest FP2 alarm hit fraction, cue+1.5--4.0 s",
            "M1": list(FEATURES),
            "M2": ["first10_detection_fraction", *FEATURES],
            "morphology": "causal EWMA H30; MA.5; local Q10 past2 excluding .5 s; IQR30 normalized prominence",
            "local_idle": "last 2 valid idle seconds before each trial cue/fixation seam; guard excluded",
            "validation": "nested leave-one-subject-out Ridge; S1/S2 held out together; inner grouped alpha selection",
            "binary_secondary": "cohort-relative top 20% future excess",
            "claim_limit": "offline session usability diagnostic on cue-relative pseudo-uncued replay; not threshold calibration",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(payload, args.figure)
    write_report(payload, args.report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
