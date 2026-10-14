"""Evaluate the fixed Riemannian gate on seam-guarded pseudo-uncued streams."""

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
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, apply_decision, tune_source_operating_points
from uncued_mi.riemannian import tangent_features

SFREQ, STEP, WINDOW = 250.0, 0.1, 1.0


def reference(covariances):
    matrix = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12)
    return matrix + np.eye(len(matrix)) * ridge


def rolling_covariance(signal):
    width, step = int(WINDOW * SFREQ), int(STEP * SFREQ)
    ends = np.arange(width, len(signal) + 1, step)
    return np.asarray([np.cov(signal[e-width:e], rowvar=False) for e in ends]), ends


def contiguous_intervals(mask: np.ndarray, times: np.ndarray) -> np.ndarray:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return np.asarray([(times[s], times[e-1] + STEP) for s, e in zip(starts, stops)],
                      dtype=float).reshape(-1, 2)


def calibration(scores):
    times = np.arange(len(scores)) * STEP
    return ScoredSequence(times, scores, np.asarray([[0, len(scores) * STEP]]),
                          np.empty((0, 2)))


def analyze(stream_path: Path, raw_path: Path, cov_path: Path, subject: int, shifts: int):
    with np.load(cov_path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        cmd_trial, iti_trial = data["train_command_trial"], data["train_idle_iti_trial"]
        cmd = tangent_features(data["train_command_covariance"][cmd_trial < 50], ref)
        iti = tangent_features(data["train_idle_iti_covariance"][iti_trial < 50], ref)
        scaler = StandardScaler().fit(np.r_[cmd, iti])
        model = LogisticRegression(class_weight="balanced", solver="liblinear",
                                   max_iter=2000, random_state=20260804).fit(
            scaler.transform(np.r_[cmd, iti]),
            np.r_[np.ones(len(cmd)), np.zeros(len(iti))])
    with np.load(raw_path, allow_pickle=False) as data:
        anchor_cov, _ = rolling_covariance(data["pre_train"])
    anchor_scores = model.decision_function(scaler.transform(tangent_features(anchor_cov, ref)))
    target_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    points = tune_source_operating_points([calibration(anchor_scores)], "ma_dwell",
                                          target_grid, STEP,
                                          ma_seconds=0.5, dwell_seconds=0.3,
                                          refractory_seconds=1.0, grid_size=160)
    with np.load(stream_path, allow_pickle=False) as data:
        signal, state, events = data["signal"], data["state"], data["events"]
        rest_source_ids = data["rest_source_ids"]
    covariance, ends = rolling_covariance(signal)
    scores = model.decision_function(scaler.transform(tangent_features(covariance, ref)))
    times = ends / SFREQ
    start_times = (ends - int(WINDOW * SFREQ)) / SFREQ
    # Both cross-recording directions are synthetic: rest->trial and the
    # previous trial->next rest. Guard both; time zero is not a seam.
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), dtype=bool)
    for seam in seams:
        # Exclude any window whose support touches seam +/- 2 s.
        guard |= (start_times < seam + 2.0) & (times > seam - 2.0)
    strict_mi = np.zeros(len(times), dtype=bool)
    for row in events:
        start, stop = row[6] / SFREQ, row[7] / SFREQ
        strict_mi |= (start_times >= start) & (times <= stop)
    valid_idle = ~strict_mi & ~guard
    idle_intervals = contiguous_intervals(valid_idle, times)
    command_intervals = np.asarray([(r[6] / SFREQ, r[7] / SFREQ) for r in events])
    reset = float(scores.min() - 10 * max(scores.std(), 1.0))
    guarded_scores = scores.copy()
    guarded_scores[guard] = reset
    scored = ScoredSequence(times, guarded_scores, idle_intervals, command_intervals)
    rng = np.random.default_rng(20260804 + subject)
    output = {}
    for target, point in points.items():
        decision = apply_decision(scored, "ma_dwell", point.threshold, STEP,
                                  ma_seconds=0.5, dwell_seconds=0.3,
                                  refractory_seconds=1.0)
        observed = evaluate_alarms(times, decision.alarms, idle_intervals, command_intervals)
        null = np.empty(shifts)
        for i, offset in enumerate(rng.integers(1, len(decision.alarms), shifts)):
            null[i] = evaluate_alarms(times, np.roll(decision.alarms, int(offset)),
                                      idle_intervals, command_intervals).true_positive_rate
        alarm_times = times[decision.alarms]
        idle_alarm_mask = np.asarray([
            any(a <= t < b for a, b in idle_intervals) for t in alarm_times
        ], dtype=bool)
        idle_alarm_times = alarm_times[idle_alarm_mask]
        alarm_indices = np.flatnonzero(decision.alarms & valid_idle)
        cumulative_idle = np.cumsum(valid_idle) * STEP
        first_fp_idle_exposure = (
            float(cumulative_idle[alarm_indices[0]]) if len(alarm_indices) else None
        )
        # Rest-only FP hazard by elapsed time from each injected-rest onset.
        hazard_edges = np.asarray([2.0, 5.0, 10.0, 20.0, np.inf])
        hazard_exposure = np.zeros(4, dtype=float)
        hazard_alarms = np.zeros(4, dtype=int)
        hazard_source_exposure = np.zeros((3, 4), dtype=float)
        hazard_source_alarms = np.zeros((3, 4), dtype=int)
        for row, source_id in zip(events, rest_source_ids):
            rest_start, rest_stop = row[2] / SFREQ, row[3] / SFREQ
            elapsed = times - rest_start
            in_rest = (times >= rest_start) & (times < rest_stop) & ~guard
            alarm_in_rest = in_rest & decision.alarms
            for bin_index, (low, high) in enumerate(zip(hazard_edges[:-1], hazard_edges[1:])):
                member = in_rest & (elapsed >= low) & (elapsed < high)
                hazard_exposure[bin_index] += member.sum() * STEP
                hazard_alarms[bin_index] += int(np.sum(alarm_in_rest & (elapsed >= low) & (elapsed < high)))
                hazard_source_exposure[int(source_id), bin_index] += member.sum() * STEP
                hazard_source_alarms[int(source_id), bin_index] += int(
                    np.sum(alarm_in_rest & (elapsed >= low) & (elapsed < high)))
        output[str(target)] = {
            "nominal_anchor_target_fp_per_minute": target,
            "threshold": float(point.threshold),
            "anchor_fp_per_minute": point.source_fp_per_minute,
            "stream_fp_per_minute": observed.fp_per_minute,
            "stream_to_nominal_target_fp_ratio": observed.fp_per_minute / target,
            "stream_to_anchor_fp_ratio": (
                observed.fp_per_minute / point.source_fp_per_minute
                if point.source_fp_per_minute > 0 else None
            ),
            "observed_tpr": observed.true_positive_rate,
            "null_tpr_median": float(np.median(null)),
            "excess_tpr": float(observed.true_positive_rate - np.median(null)),
            "median_delay_seconds": observed.median_delay_seconds,
            "idle_seconds_to_first_false_positive": first_fp_idle_exposure,
            "first_false_positive_observed": first_fp_idle_exposure is not None,
            "first_false_positive_censor_idle_seconds": float(cumulative_idle[-1]),
            "stream_duration_seconds": float(times[-1]),
            "rest_hazard_bins_seconds": ["2-5", "5-10", "10-20", "20+"],
            "rest_hazard_exposure_seconds": hazard_exposure.tolist(),
            "rest_hazard_alarm_counts": hazard_alarms.tolist(),
            "rest_hazard_source_names": ["post_train", "pre_test", "post_test"],
            "rest_hazard_source_exposure_seconds": hazard_source_exposure.tolist(),
            "rest_hazard_source_alarm_counts": hazard_source_alarms.tolist(),
            "rest_hazard_fp_per_minute": np.divide(
                60 * hazard_alarms, hazard_exposure,
                out=np.full(4, np.nan), where=hazard_exposure > 0,
            ).tolist(),
        }
    return {"subject": subject, "events": len(events),
            "valid_idle_seconds": float(sum(b-a for a, b in idle_intervals)),
            "guarded_score_windows": int(guard.sum()), "operating_points": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-root", type=Path,
                        default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest_raw/session1"))
    parser.add_argument("--cov-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest/session1"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_pseudo_uncued_riemannian.json"))
    args = parser.parse_args()
    subjects = sorted(int(p.stem.split("-")[1]) for p in args.stream_root.glob("sub-*.npz"))
    rows = [analyze(args.stream_root / f"sub-{s:02d}.npz",
                    args.raw_root / f"sub-{s:02d}.npz",
                    args.cov_root / f"sub-{s:02d}.npz", s, args.shifts) for s in subjects]
    def boot_median(values, seed):
        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), (10000, len(values)))]
        return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()

    summary = {}
    target_names = ("0.25", "0.5", "1.0", "2.0", "4.0", "8.0")
    for target in target_names:
        fields = ("stream_fp_per_minute", "observed_tpr", "null_tpr_median", "excess_tpr")
        summary[target] = {f + "_median": float(np.median([
            r["operating_points"][target][f] for r in rows])) for f in fields}
        summary[target]["stream_to_anchor_fp_ratio_median"] = float(np.median([
            r["operating_points"][target]["stream_to_anchor_fp_ratio"]
            for r in rows
            if r["operating_points"][target]["stream_to_anchor_fp_ratio"] is not None
        ])) if any(r["operating_points"][target]["stream_to_anchor_fp_ratio"] is not None
                  for r in rows) else None
        nominal_ratios = [r["operating_points"][target]["stream_to_nominal_target_fp_ratio"]
                          for r in rows]
        summary[target]["stream_to_nominal_target_fp_ratio_median"] = float(
            np.median(nominal_ratios))
        summary[target]["stream_to_nominal_target_fp_ratio_ci"] = boot_median(
            nominal_ratios, 20260804 + int(float(target) * 100))
    # Diagnostic common-FP interpolation. This is an evaluation curve, not a
    # deployable target-specific threshold selection procedure.
    common_fp2 = []
    for row in rows:
        pairs = sorted((row["operating_points"][name]["stream_fp_per_minute"],
                        row["operating_points"][name]["excess_tpr"])
                       for name in target_names)
        fp = np.asarray([p[0] for p in pairs]); excess = np.asarray([p[1] for p in pairs])
        if fp[0] <= 2.0 <= fp[-1]:
            common_fp2.append(float(np.interp(2.0, fp, excess)))
    summary["common_actual_fp2"] = {
        "subjects_bracketing_fp2": len(common_fp2),
        "interpolated_excess_tpr_median": float(np.median(common_fp2)) if common_fp2 else None,
        "interpolated_excess_tpr_ci": boot_median(common_fp2, 20261004)
        if common_fp2 else None,
        "interpretation": "diagnostic common-FP comparison; not deployable threshold tuning",
    }
    result = {"subjects": len(rows), "summary": summary, "rows": rows}
    for target in target_names:
        counts = np.sum([r["operating_points"][target]["rest_hazard_alarm_counts"]
                         for r in rows], axis=0)
        exposure = np.sum([r["operating_points"][target]["rest_hazard_exposure_seconds"]
                           for r in rows], axis=0)
        summary[target]["pooled_rest_hazard_fp_per_minute"] = (
            60 * counts / exposure).tolist()
        summary[target]["rest_hazard_bins_seconds"] = ["2-5", "5-10", "10-20", "20+"]
        early, late = [], []
        for row in rows:
            point = row["operating_points"][target]
            counts = point["rest_hazard_alarm_counts"]
            exposures = point["rest_hazard_exposure_seconds"]
            early.append(60 * sum(counts[:2]) / sum(exposures[:2]))
            late.append(60 * counts[3] / exposures[3])
        summary[target]["early_2_10s_fp_per_minute_median"] = float(np.median(early))
        summary[target]["late_20plus_fp_per_minute_median"] = float(np.median(late))
        summary[target]["late_vs_early_wilcoxon_p"] = float(wilcoxon(late, early).pvalue)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"subjects": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
