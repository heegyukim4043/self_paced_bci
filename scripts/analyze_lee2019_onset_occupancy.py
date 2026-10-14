"""Diagnose cue-label timing and state occupancy in Lee2019 pseudo streams.

This is a diagnostic analysis, not a self-initiated-onset claim.  It reuses the
fixed 1 s Riemannian TS+LR score, all-score EWMA (30 s), and the source-derived
FP=2/min operating point used by the sequential deployment analysis.
"""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, reference, rolling_covariance,
)


def rolling_mean(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < steps:
        return out
    cumulative = np.r_[0.0, np.cumsum(values)]
    out[steps - 1:] = (cumulative[steps:] - cumulative[:-steps]) / steps
    return out


def dwell_state(active: np.ndarray, steps: int, valid: np.ndarray) -> np.ndarray:
    """Return all samples belonging to a threshold run lasting >= steps."""
    output = np.zeros(len(active), bool)
    run_start = None
    for index, flag in enumerate(active & valid):
        if flag and run_start is None:
            run_start = index
        if (not flag or index == len(active) - 1) and run_start is not None:
            stop = index if not flag else index + 1
            if stop - run_start >= steps:
                output[run_start:stop] = True
            run_start = None
    return output


def bootstrap_median(values: np.ndarray, seed: int, repeats: int = 10000) -> list[float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def analyze(subject: int, stream_root: Path, raw_root: Path, cov_root: Path) -> dict:
    cov_path = cov_root / f"sub-{subject:02d}.npz"
    raw_path = raw_root / f"sub-{subject:02d}.npz"
    stream_path = stream_root / f"sub-{subject:02d}.npz"
    with np.load(cov_path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        ct, it = data["train_command_trial"], data["train_idle_iti_trial"]
        command = tangent_features(data["train_command_covariance"][ct < 50], ref)
        iti = tangent_features(data["train_idle_iti_covariance"][it < 50], ref)
    scaler = StandardScaler().fit(np.r_[command, iti])
    model = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260809,
    ).fit(scaler.transform(np.r_[command, iti]),
          np.r_[np.ones(len(command)), np.zeros(len(iti))])

    with np.load(raw_path, allow_pickle=False) as data:
        anchor_cov, _ = rolling_covariance(data["pre_train"])
    anchor_scores = model.decision_function(
        scaler.transform(tangent_features(anchor_cov, ref)))
    anchor_mean = float(anchor_scores.mean())
    anchor_variance = float(anchor_scores.var(ddof=1))
    anchor_z = (anchor_scores - anchor_mean) / np.sqrt(anchor_variance)
    point = tune_source_operating_points(
        [calibration(anchor_z)], "ma_dwell", [2.0], STEP,
        ma_seconds=.5, dwell_seconds=.3, refractory_seconds=1., grid_size=160,
    )[2.0]

    with np.load(stream_path, allow_pickle=False) as data:
        signal, events = data["signal"], data["events"]
    covariance, ends = rolling_covariance(signal)
    raw_scores = model.decision_function(
        scaler.transform(tangent_features(covariance, ref)))
    times = ends / SFREQ
    starts = (ends - int(WINDOW * SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), bool)
    for seam in seams:
        guard |= (starts < seam + 2.) & (times > seam - 2.)

    normalized = ewma_normalize_and_decide(
        raw_scores, initial_mean=anchor_mean, initial_variance=anchor_variance,
        threshold=point.threshold, step_seconds=STEP, half_life_seconds=30.,
        update_mode="all", update_mask=~guard, ma_seconds=.5,
        dwell_seconds=.3, refractory_seconds=1., sigma_floor_fraction=.25,
    ).statistic
    ma = rolling_mean(normalized, int(np.ceil(.5 / STEP)))
    valid = ~guard & np.isfinite(ma)
    active = valid & (ma >= point.threshold)
    occupied = dwell_state(active, int(np.ceil(.3 / STEP)), valid)

    mi_mask = np.zeros(len(times), bool)
    rest_mask = np.zeros(len(times), bool)
    trial_non_mi_mask = np.zeros(len(times), bool)
    event_records = []
    for row in events:
        rest_start, rest_stop = row[2] / SFREQ, row[3] / SFREQ
        trial_start = row[4] / SFREQ
        cue = row[5] / SFREQ
        mi_start, mi_stop = row[6] / SFREQ, row[7] / SFREQ
        trial_stop = cue + 4.
        rest_mask |= (starts >= rest_start) & (times <= rest_stop)
        strict = (starts >= mi_start) & (times <= mi_stop)
        mi_mask |= strict
        trial_non_mi_mask |= (
            (starts >= trial_start) & (times <= trial_stop) & ~strict)

        broad = valid & (times >= cue) & (times <= cue + 4.)
        strict_broad = valid & (starts >= cue + .5) & (times <= cue + 3.5)
        if not np.any(broad):
            continue
        indices = np.flatnonzero(broad)
        peak_index = indices[int(np.nanargmax(ma[indices]))]
        crossing = indices[active[indices]]
        event_records.append({
            "source_trial": int(row[1]),
            "peak_latency_seconds": float(times[peak_index] - cue),
            "peak_inside_nominal_support": bool(strict_broad[peak_index]),
            "peak_end_inside_cue_0.5_to_3.5": bool(
                cue + .5 <= times[peak_index] <= cue + 3.5),
            "first_crossing_latency_seconds": (
                float(times[crossing[0]] - cue) if len(crossing) else None),
            "threshold_response": bool(len(crossing)),
            "strict_mi_occupancy": float(np.mean(occupied[strict_broad]))
            if np.any(strict_broad) else None,
        })

    def fraction(mask: np.ndarray) -> float:
        member = mask & valid
        return float(np.mean(occupied[member])) if np.any(member) else float("nan")

    peaks = np.asarray([r["peak_latency_seconds"] for r in event_records])
    crossings = np.asarray([r["first_crossing_latency_seconds"] for r in event_records
                            if r["first_crossing_latency_seconds"] is not None])
    return {
        "subject": subject,
        "threshold": float(point.threshold),
        "events": len(event_records),
        "peak_latency_median_seconds": float(np.median(peaks)),
        "peak_latency_iqr_seconds": np.quantile(peaks, [.25, .75]).tolist(),
        "peak_inside_nominal_fraction": float(np.mean(
            [r["peak_inside_nominal_support"] for r in event_records])),
        "threshold_no_response_fraction": float(np.mean(
            [not r["threshold_response"] for r in event_records])),
        "first_crossing_latency_median_seconds": (
            float(np.median(crossings)) if len(crossings) else None),
        "occupancy": {
            "strict_mi": fraction(mi_mask),
            "injected_rest": fraction(rest_mask),
            "trial_non_mi": fraction(trial_non_mi_mask),
        },
        "event_records": event_records,
    }


def summarize(rows: list[dict]) -> dict:
    def values(path):
        group, key = path
        return np.asarray([r[group][key] for r in rows], float)
    mi = values(("occupancy", "strict_mi"))
    rest = values(("occupancy", "injected_rest"))
    non_mi = values(("occupancy", "trial_non_mi"))
    peaks = np.concatenate([[e["peak_latency_seconds"] for e in r["event_records"]]
                            for r in rows])
    crossings = np.asarray([
        e["first_crossing_latency_seconds"] for r in rows for e in r["event_records"]
        if e["first_crossing_latency_seconds"] is not None], float)
    diff = mi - rest
    test = wilcoxon(diff, alternative="greater", zero_method="wilcox")
    return {
        "subjects": len(rows),
        "events": int(sum(r["events"] for r in rows)),
        "trial_peak_latency_seconds": {
            "median": float(np.median(peaks)),
            "iqr": np.quantile(peaks, [.25, .75]).tolist(),
            "inside_nominal_support_fraction": float(np.mean([
                e["peak_inside_nominal_support"] for r in rows for e in r["event_records"]])),
            "window_end_inside_cue_0.5_to_3.5_fraction": float(np.mean([
                e["peak_end_inside_cue_0.5_to_3.5"]
                for r in rows for e in r["event_records"]])),
            "support_definition": (
                "inside_nominal_support requires the complete 1 s window to lie "
                "inside cue+0.5..3.5; its end therefore lies at cue+1.5..3.5"
            ),
        },
        "threshold_response": {
            "no_response_fraction": float(np.mean([
                not e["threshold_response"] for r in rows for e in r["event_records"]])),
            "first_crossing_latency_median_seconds": (
                float(np.median(crossings)) if len(crossings) else None),
        },
        "state_occupancy_subject_medians": {
            "strict_mi": float(np.median(mi)),
            "injected_rest": float(np.median(rest)),
            "trial_non_mi": float(np.median(non_mi)),
            "mi_minus_rest": float(np.median(diff)),
            "mi_minus_rest_ci95": bootstrap_median(diff, 20260809),
            "mi_greater_rest_wilcoxon_p": float(test.pvalue),
        },
        "claim_limit": (
            "Cue-aligned diagnostic on synthetic rest+trial streams; peak timing is "
            "not an independently observed intention onset."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-root", type=Path,
                        default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest_raw/session1"))
    parser.add_argument("--cov-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest/session1"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_onset_occupancy.json"))
    args = parser.parse_args()
    subjects = args.subjects or sorted(
        int(p.stem.split("-")[1]) for p in args.stream_root.glob("sub-*.npz"))
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"onset/occupancy {index}/{len(subjects)} subject={subject}", flush=True)
        rows.append(analyze(subject, args.stream_root, args.raw_root, args.cov_root))
    output = {"summary": summarize(rows), "subjects": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
