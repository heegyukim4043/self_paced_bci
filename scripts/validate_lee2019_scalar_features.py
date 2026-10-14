"""Causal FP=2 validation of scalar features selected by trajectory screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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

TARGETS = (.25, .5, 1., 2., 4., 8., 16., 32.)
FEATURES = ("individual_mu", "laplacian_mu", "beta_mid", "beta_high",
            "cov_logdet", "cov_effective_rank")


def ends_for(signal):
    width, step = int(WINDOW * SFREQ), int(STEP * SFREQ)
    return np.arange(width, len(signal) + 1, step)


def log_variance(signal, ends, indices):
    width = int(WINDOW * SFREQ)
    return np.asarray([np.log(max(np.var(signal[e-width:e, indices]), 1e-12)) for e in ends])


def peak_mu(signal, channels):
    lookup = {n.casefold(): i for i, n in enumerate(channels)}
    indices = [lookup[n.casefold()] for n in ("C3", "Cz", "C4")]
    spectrum = np.abs(np.fft.rfft(signal[:, indices] * np.hanning(len(signal))[:, None], axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), 1 / SFREQ)
    keep = (frequencies >= 7) & (frequencies <= 14)
    power = spectrum[keep].mean(axis=1)
    return float(frequencies[keep][np.argmax(power)])


def feature_sequences(signal, channels, individual_mu):
    ends = ends_for(signal)
    lookup = {n.casefold(): i for i, n in enumerate(channels)}
    central = [lookup[n.casefold()] for n in ("C3", "Cz", "C4")]
    c3, c4 = lookup["c3"], lookup["c4"]
    left = [lookup[n.casefold()] for n in ("FC3", "C5", "C1", "CP3")]
    right = [lookup[n.casefold()] for n in ("FC4", "C2", "C6", "CP4")]
    mu = _causal_band_signal(signal, SFREQ, max(6., individual_mu-1.5),
                             min(16., individual_mu+1.5), 4)
    lap = np.c_[signal[:, c3] - signal[:, left].mean(1),
                signal[:, c4] - signal[:, right].mean(1)]
    lap = _causal_band_signal(lap, SFREQ, 8., 13., 4)
    beta_mid = _causal_band_signal(signal, SFREQ, 18., 24., 4)
    beta_high = _causal_band_signal(signal, SFREQ, 24., 30., 4)
    output = {
        "individual_mu": log_variance(mu, ends, central),
        "laplacian_mu": log_variance(lap, ends, [0, 1]),
        "beta_mid": log_variance(beta_mid, ends, central),
        "beta_high": log_variance(beta_high, ends, central),
    }
    width = int(WINDOW * SFREQ)
    logdet, rank = [], []
    for end in ends:
        covariance = np.cov(signal[end-width:end], rowvar=False)
        eigen = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
        proportions = eigen / eigen.sum()
        logdet.append(np.log(eigen).sum())
        rank.append(np.exp(-np.sum(proportions * np.log(proportions))))
    output["cov_logdet"] = np.asarray(logdet)
    output["cov_effective_rank"] = np.asarray(rank)
    return output, ends


def interval_mask(ends, intervals, trials, low, high):
    times, starts = ends / SFREQ, (ends - int(WINDOW * SFREQ)) / SFREQ
    mask = np.zeros(len(ends), bool)
    for (a, b), trial in zip(intervals, trials):
        if low <= trial < high:
            mask |= (starts >= a) & (times <= b)
    return mask


def interpolate_at_two(points):
    xy = sorted((v["fp"], v["excess"]) for v in points.values())
    x = np.asarray([v[0] for v in xy]); y = np.asarray([v[1] for v in xy])
    unique = np.unique(x)
    y = np.asarray([np.median(y[x == value]) for value in unique]); x = unique
    if len(x) < 2 or 2. < x.min() or 2. > x.max():
        return None
    return float(np.interp(2., x, y))


def analyze(subject, mil_root, rest_root, stream_root, shifts):
    with np.load(mil_root / f"sub-{subject:02d}_mil_raw.npz", allow_pickle=False) as d:
        train_signal = d["signal"]
        pre_rest = d["pre_rest"]
        channels = tuple(d["channels"].tolist())
        command_intervals, command_trials = d["command_intervals"], d["command_trials"]
        iti_intervals, iti_trials = d["iti_intervals"], d["iti_trials"]
    mu = peak_mu(pre_rest, channels)
    train_features, train_ends = feature_sequences(train_signal, channels, mu)
    anchor_features, _ = feature_sequences(pre_rest, channels, mu)
    command_fit = interval_mask(train_ends, command_intervals, command_trials, 0, 50)
    iti_fit = interval_mask(train_ends, iti_intervals, iti_trials, 0, 50)
    with np.load(stream_root / f"sub-{subject:02d}.npz", allow_pickle=False) as d:
        stream_signal, events = d["signal"], d["events"]
    stream_features, ends = feature_sequences(stream_signal, channels, mu)
    times, starts = ends / SFREQ, (ends-int(WINDOW*SFREQ))/SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), bool)
    for seam in seams:
        guard |= (starts < seam+2.) & (times > seam-2.)
    strict = np.zeros(len(times), bool)
    for row in events:
        strict |= (starts >= row[6]/SFREQ) & (times <= row[7]/SFREQ)
    idle = ~strict & ~guard
    idle_intervals_out = contiguous_intervals(idle, times)
    command_intervals_out = np.asarray([(r[6]/SFREQ, r[7]/SFREQ) for r in events])
    rng = np.random.default_rng(20260811 + subject)
    offsets = rng.integers(1, len(times), shifts)
    output = {}
    for name in FEATURES:
        train = train_features[name]
        direction = 1. if np.median(train[command_fit]) >= np.median(train[iti_fit]) else -1.
        anchor = direction * anchor_features[name]
        mean, variance = float(anchor.mean()), float(anchor.var(ddof=1))
        anchor_z = (anchor-mean)/np.sqrt(variance)
        operating = tune_source_operating_points(
            [calibration(anchor_z)], "ma_dwell", TARGETS, STEP,
            ma_seconds=.5, dwell_seconds=.3, refractory_seconds=1., grid_size=160)
        scores = direction * stream_features[name]
        feature_points = {}
        for target, point in operating.items():
            decision = ewma_normalize_and_decide(
                scores, initial_mean=mean, initial_variance=variance,
                threshold=point.threshold, step_seconds=STEP, half_life_seconds=30.,
                update_mode="all", update_mask=~guard, ma_seconds=.5,
                dwell_seconds=.3, refractory_seconds=1., sigma_floor_fraction=.25)
            observed = evaluate_alarms(times, decision.alarms, idle_intervals_out,
                                       command_intervals_out)
            null = [evaluate_alarms(times, np.roll(decision.alarms, int(offset)),
                                    idle_intervals_out, command_intervals_out).true_positive_rate
                    for offset in offsets]
            feature_points[str(target)] = {
                "fp": observed.fp_per_minute,
                "tpr": observed.true_positive_rate,
                "null": float(np.median(null)),
                "excess": float(observed.true_positive_rate-np.median(null)),
            }
        output[name] = {"direction": int(direction), "points": feature_points,
                        "excess_at_actual_fp2": interpolate_at_two(feature_points)}
    return {"subject": subject, "individual_mu_hz": mu, "features": output}


def bootstrap(values, seed):
    rng = np.random.default_rng(seed); values = np.asarray(values)
    draws = values[rng.integers(0, len(values), (10000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mil-root", type=Path,
                        default=Path("data/interim/lee2019_mil_raw/session1"))
    parser.add_argument("--rest-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest_raw/session1"))
    parser.add_argument("--stream-root", type=Path,
                        default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_scalar_feature_validation.json"))
    args = parser.parse_args()
    subjects = args.subjects or sorted(int(p.stem.split("-")[1].split("_")[0])
                                      for p in args.mil_root.glob("sub-*_mil_raw.npz"))
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"scalar validation {index}/{len(subjects)} subject={subject}", flush=True)
        rows.append(analyze(subject, args.mil_root, args.rest_root, args.stream_root, args.shifts))
    summary = {}
    for index, name in enumerate(FEATURES):
        values = [r["features"][name]["excess_at_actual_fp2"] for r in rows]
        valid = np.asarray([v for v in values if v is not None])
        summary[name] = {
            "subjects_bracketing_fp2": len(valid),
            "bracket_failure_fraction": 1-len(valid)/len(rows),
            "excess_at_actual_fp2_median": float(np.median(valid)) if len(valid) else None,
            "excess_ci95": bootstrap(valid, 20260811+index) if len(valid) else None,
        }
    payload = {"summary": summary, "subjects": rows,
               "comparator": "Riemannian+EWMA sequential excess at actual FP2 ~=0.075",
               "claim_limit": "candidate screen follow-up; no left/right classification"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
