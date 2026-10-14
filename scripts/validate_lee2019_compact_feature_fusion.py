"""Strict sequential validation of compact mu/beta feature fusion.

The locked questions are:
1. Does fusion recover sustained-MI detection with a 3-s seam guard?
2. Does removing MA recover it when the usual MA+dwell decision does not?
3. Are the candidate features too correlated to provide complementary evidence?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.features import _causal_band_signal
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import tune_source_operating_points
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, contiguous_intervals,
)

FEATURES = ("mu_high", "individual_mu", "laplacian_mu", "beta_mid", "beta_high")
TARGETS = (.25, .5, 1., 2., 4., 8., 16.)
GUARDS = (2., 3.)
DECISIONS = {
    "instant": {"ma": 0., "dwell": .1},
    "ma_dwell": {"ma": .5, "dwell": .3},
}


def ends_for(signal: np.ndarray) -> np.ndarray:
    width, step = int(WINDOW * SFREQ), int(STEP * SFREQ)
    return np.arange(width, len(signal) + 1, step)


def log_variance(signal: np.ndarray, ends: np.ndarray, indices: list[int]) -> np.ndarray:
    width = int(WINDOW * SFREQ)
    return np.asarray([
        np.log(max(np.var(signal[end-width:end, indices]), 1e-12)) for end in ends
    ])


def peak_mu(signal: np.ndarray, channels: tuple[str, ...]) -> float:
    lookup = {name.casefold(): i for i, name in enumerate(channels)}
    indices = [lookup[name.casefold()] for name in ("C3", "Cz", "C4")]
    tapered = signal[:, indices] * np.hanning(len(signal))[:, None]
    spectrum = np.abs(np.fft.rfft(tapered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), 1 / SFREQ)
    keep = (frequencies >= 7.) & (frequencies <= 14.)
    return float(frequencies[keep][np.argmax(spectrum[keep].mean(axis=1))])


def feature_matrix(signal: np.ndarray, channels: tuple[str, ...], mu_peak: float):
    ends = ends_for(signal)
    lookup = {name.casefold(): i for i, name in enumerate(channels)}
    central = [lookup[name.casefold()] for name in ("C3", "Cz", "C4")]
    c3, c4 = lookup["c3"], lookup["c4"]
    left = [lookup[name.casefold()] for name in ("FC3", "C5", "C1", "CP3")]
    right = [lookup[name.casefold()] for name in ("FC4", "C2", "C6", "CP4")]

    mu_high = _causal_band_signal(signal, SFREQ, 10., 13., 4)
    individual = _causal_band_signal(
        signal, SFREQ, max(6., mu_peak - 1.5), min(16., mu_peak + 1.5), 4
    )
    beta_mid = _causal_band_signal(signal, SFREQ, 18., 24., 4)
    beta_high = _causal_band_signal(signal, SFREQ, 24., 30., 4)
    laplacian = np.c_[
        signal[:, c3] - signal[:, left].mean(axis=1),
        signal[:, c4] - signal[:, right].mean(axis=1),
    ]
    laplacian = _causal_band_signal(laplacian, SFREQ, 8., 13., 4)
    columns = {
        "mu_high": log_variance(mu_high, ends, central),
        "individual_mu": log_variance(individual, ends, central),
        "laplacian_mu": log_variance(laplacian, ends, [0, 1]),
        "beta_mid": log_variance(beta_mid, ends, central),
        "beta_high": log_variance(beta_high, ends, central),
    }
    return np.column_stack([columns[name] for name in FEATURES]), ends


def interval_mask(ends, intervals, trials, low, high):
    times = ends / SFREQ
    starts = (ends - int(WINDOW * SFREQ)) / SFREQ
    mask = np.zeros(len(ends), dtype=bool)
    for (begin, end), trial in zip(intervals, trials):
        if low <= trial < high:
            mask |= (starts >= begin) & (times <= end)
    return mask


def interpolate(points, key):
    pairs = sorted((value["fp"], value[key]) for value in points.values())
    x = np.asarray([a for a, _ in pairs])
    y = np.asarray([b for _, b in pairs])
    unique = np.unique(x)
    y = np.asarray([np.median(y[x == value]) for value in unique])
    x = unique
    if len(x) < 2 or 2. < x.min() or 2. > x.max():
        return None
    return float(np.interp(2., x, y))


def analyze(subject, mil_root, stream_root, shifts):
    with np.load(mil_root / f"sub-{subject:02d}_mil_raw.npz", allow_pickle=False) as data:
        train_signal = data["signal"]
        anchor_signal = data["pre_rest"]
        channels = tuple(data["channels"].tolist())
        command_intervals = data["command_intervals"]
        command_trials = data["command_trials"]
        iti_intervals = data["iti_intervals"]
        iti_trials = data["iti_trials"]

    mu_peak = peak_mu(anchor_signal, channels)
    train_x, train_ends = feature_matrix(train_signal, channels, mu_peak)
    anchor_x, _ = feature_matrix(anchor_signal, channels, mu_peak)
    command_fit = interval_mask(train_ends, command_intervals, command_trials, 0, 50)
    iti_fit = interval_mask(train_ends, iti_intervals, iti_trials, 0, 50)
    fit_mask = command_fit | iti_fit
    y = command_fit[fit_mask].astype(int)
    scaler = StandardScaler().fit(train_x[fit_mask])
    model = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260811,
    ).fit(scaler.transform(train_x[fit_mask]), y)
    anchor_score = model.decision_function(scaler.transform(anchor_x))
    anchor_mean = float(anchor_score.mean())
    anchor_variance = float(anchor_score.var(ddof=1))
    anchor_z = (anchor_score - anchor_mean) / np.sqrt(anchor_variance)

    standardized_fit = scaler.transform(train_x[fit_mask])
    correlation = spearmanr(standardized_fit, axis=0).statistic

    operating = {}
    for name, decision in DECISIONS.items():
        operating[name] = tune_source_operating_points(
            [calibration(anchor_z)], "ma_dwell", TARGETS, STEP,
            ma_seconds=decision["ma"], dwell_seconds=decision["dwell"],
            refractory_seconds=1., grid_size=160,
        )

    with np.load(stream_root / f"sub-{subject:02d}.npz", allow_pickle=False) as data:
        stream_signal = data["signal"]
        events = data["events"]
    stream_x, ends = feature_matrix(stream_signal, channels, mu_peak)
    raw_score = model.decision_function(scaler.transform(stream_x))
    times = ends / SFREQ
    starts = (ends - int(WINDOW * SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    commands = np.asarray([(row[6] / SFREQ + WINDOW, row[7] / SFREQ) for row in events])
    rng = np.random.default_rng(420000 + subject)
    offsets = rng.integers(1, len(times), shifts)

    conditions = {}
    for guard_width in GUARDS:
        guard = np.zeros(len(times), dtype=bool)
        for seam in seams:
            guard |= (starts < seam + guard_width) & (times > seam)
        strict = np.zeros(len(times), dtype=bool)
        for row in events:
            strict |= (starts >= row[6] / SFREQ) & (times <= row[7] / SFREQ)
        idle = ~strict & ~guard
        idle_intervals = contiguous_intervals(idle, times)
        for decision_name, decision_config in DECISIONS.items():
            points = {}
            for target, point in operating[decision_name].items():
                decision = ewma_normalize_and_decide(
                    raw_score, initial_mean=anchor_mean,
                    initial_variance=anchor_variance, threshold=point.threshold,
                    step_seconds=STEP, half_life_seconds=30., update_mode="all",
                    update_mask=~guard, ma_seconds=decision_config["ma"],
                    dwell_seconds=decision_config["dwell"], refractory_seconds=1.,
                    sigma_floor_fraction=.25,
                )
                observed = evaluate_alarms(times, decision.alarms, idle_intervals, commands)
                null = [
                    evaluate_alarms(
                        times, np.roll(decision.alarms, int(offset)) & ~guard,
                        idle_intervals, commands,
                    ).true_positive_rate
                    for offset in offsets
                ]
                points[str(target)] = {
                    "threshold": float(point.threshold),
                    "fp": observed.fp_per_minute,
                    "tpr": observed.true_positive_rate,
                    "null": float(np.median(null)),
                    "excess": float(observed.true_positive_rate - np.median(null)),
                }
            key = f"guard{int(guard_width)}_{decision_name}"
            conditions[key] = {
                "points": points,
                **{f"{metric}_at_fp2": interpolate(points, metric)
                   for metric in ("threshold", "tpr", "null", "excess")},
            }
    return {
        "subject": subject,
        "individual_mu_hz": mu_peak,
        "feature_correlation": np.asarray(correlation).tolist(),
        "model_coefficients": model.coef_[0].tolist(),
        "conditions": conditions,
    }


def bootstrap(values, seed):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (10000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mil-root", type=Path,
                        default=Path("data/interim/lee2019_mil_raw/session1"))
    parser.add_argument("--stream-root", type=Path,
                        default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_compact_feature_fusion.json"))
    args = parser.parse_args()
    subjects = args.subjects or list(range(1, 55))
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"feature-fusion {index}/{len(subjects)} subject={subject}", flush=True)
        rows.append(analyze(subject, args.mil_root, args.stream_root, args.shifts))

    condition_names = [
        f"guard{int(guard)}_{decision}" for guard in GUARDS for decision in DECISIONS
    ]
    summary = {}
    for index, name in enumerate(condition_names):
        values = [row["conditions"][name]["excess_at_fp2"] for row in rows]
        valid = np.asarray([value for value in values if value is not None])
        tpr = [row["conditions"][name]["tpr_at_fp2"] for row in rows
               if row["conditions"][name]["tpr_at_fp2"] is not None]
        null = [row["conditions"][name]["null_at_fp2"] for row in rows
                if row["conditions"][name]["null_at_fp2"] is not None]
        summary[name] = {
            "bracketed": len(valid),
            "excess_median": float(np.median(valid)) if len(valid) else None,
            "excess_ci95": bootstrap(valid, 20261000 + index) if len(valid) else None,
            "tpr_median": float(np.median(tpr)) if tpr else None,
            "null_median": float(np.median(null)) if null else None,
        }
    correlations = np.asarray([row["feature_correlation"] for row in rows])
    median_correlation = np.nanmedian(correlations, axis=0)
    off_diagonal = np.abs(median_correlation[np.triu_indices(len(FEATURES), 1)])
    payload = {
        "binding_rule": (
            "sustained-MI recovery requires guard3 excess CI lower bound > 0; "
            "guard2-only gain is transition-assisted"
        ),
        "features": FEATURES,
        "summary": summary,
        "feature_correlation_median": median_correlation.tolist(),
        "feature_correlation_abs_offdiagonal_median": float(np.median(off_diagonal)),
        "feature_correlation_abs_offdiagonal_max": float(np.max(off_diagonal)),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": summary,
        "feature_correlation_median": median_correlation.tolist(),
        "abs_offdiagonal_median": float(np.median(off_diagonal)),
        "abs_offdiagonal_max": float(np.max(off_diagonal)),
    }, indent=2))


if __name__ == "__main__":
    main()
