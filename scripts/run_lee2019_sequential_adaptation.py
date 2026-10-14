"""EWMA score adaptation on the fixed Lee2019 sequential stream."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, contiguous_intervals, reference, rolling_covariance,
)
from scripts.analyze_lee2019_pseudo_uncued_survival_hazard import poisson_cluster_model

TARGETS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
CONDITIONS = (("static", None, 0.0), ("all", 10.0, 0.0),
              ("all", 30.0, 0.0), ("all", 60.0, 0.0),
              ("freeze_only", 10.0, 0.0), ("freeze_only", 30.0, 0.0),
              ("freeze_only", 60.0, 0.0), ("idle_only", 10.0, 0.0),
              ("idle_only", 30.0, 0.0), ("idle_only", 60.0, 0.0))


def analyze(stream_path, raw_path, cov_path, subject, shifts=100, condition_specs=CONDITIONS):
    with np.load(cov_path, allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        ct, it = data["train_command_trial"], data["train_idle_iti_trial"]
        cmd = tangent_features(data["train_command_covariance"][ct < 50], ref)
        iti = tangent_features(data["train_idle_iti_covariance"][it < 50], ref)
        scaler = StandardScaler().fit(np.r_[cmd, iti])
        model = LogisticRegression(class_weight="balanced", solver="liblinear",
                                   max_iter=2000, random_state=20260805).fit(
            scaler.transform(np.r_[cmd, iti]), np.r_[np.ones(len(cmd)), np.zeros(len(iti))])
    with np.load(raw_path, allow_pickle=False) as data:
        anchor_cov, _ = rolling_covariance(data["pre_train"])
    anchor_raw = model.decision_function(scaler.transform(tangent_features(anchor_cov, ref)))
    anchor_mean, anchor_var = float(anchor_raw.mean()), float(anchor_raw.var(ddof=1))
    anchor_z = (anchor_raw - anchor_mean) / np.sqrt(anchor_var)
    points = tune_source_operating_points([calibration(anchor_z)], "ma_dwell", TARGETS, STEP,
                                          ma_seconds=.5, dwell_seconds=.3,
                                          refractory_seconds=1., grid_size=160)
    with np.load(stream_path, allow_pickle=False) as data:
        signal, events = data["signal"], data["events"]
        rest_source_ids = data["rest_source_ids"]
    covariance, ends = rolling_covariance(signal)
    raw_scores = model.decision_function(scaler.transform(tangent_features(covariance, ref)))
    times, starts = ends / SFREQ, (ends - int(WINDOW * SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), bool)
    for seam in seams:
        guard |= (starts < seam + 2.) & (times > seam - 2.)
    strict_mi = np.zeros(len(times), bool)
    for row in events:
        strict_mi |= (starts >= row[6] / SFREQ) & (times <= row[7] / SFREQ)
    idle = ~strict_mi & ~guard
    idle_intervals = contiguous_intervals(idle, times)
    command_intervals = np.asarray([(r[6] / SFREQ, r[7] / SFREQ) for r in events])
    rng = np.random.default_rng(30260805 + subject)
    condition_results = {}
    for mode, half_life, prior_n0 in condition_specs:
        name = "static" if mode == "static" else f"{mode}_hl{int(half_life)}"
        if prior_n0 > 0:
            name += f"_n0{int(prior_n0)}"
        output = {}
        for target in TARGETS:
            point = points[target]
            decision = ewma_normalize_and_decide(
                raw_scores, initial_mean=anchor_mean, initial_variance=anchor_var,
                threshold=point.threshold, step_seconds=STEP,
                half_life_seconds=half_life, update_mode=mode, update_mask=~guard,
                ma_seconds=.5, dwell_seconds=.3, refractory_seconds=1.,
                anchor_prior_n0=prior_n0, between_variance=False,
                sigma_floor_fraction=.25)
            observed = evaluate_alarms(times, decision.alarms, idle_intervals, command_intervals)
            null = np.empty(shifts)
            for i, offset in enumerate(rng.integers(1, len(decision.alarms), shifts)):
                null[i] = evaluate_alarms(times, np.roll(decision.alarms, int(offset)),
                                          idle_intervals, command_intervals).true_positive_rate
            idle_alarm_indices = np.flatnonzero(decision.alarms & idle)
            cumulative_idle = np.cumsum(idle) * STEP
            edges = np.asarray([2., 5., 10., 20., np.inf])
            source_exposure = np.zeros((3, 4), float)
            source_counts = np.zeros((3, 4), int)
            for event, source_id in zip(events, rest_source_ids):
                rest_start, rest_stop = event[2] / SFREQ, event[3] / SFREQ
                elapsed = times - rest_start
                in_rest = (times >= rest_start) & (times < rest_stop) & ~guard
                for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                    member = in_rest & (elapsed >= low) & (elapsed < high)
                    source_exposure[int(source_id), bin_index] += member.sum() * STEP
                    source_counts[int(source_id), bin_index] += int(np.sum(member & decision.alarms))
            output[str(target)] = {
                "threshold": float(point.threshold),
                "actual_fp_per_minute": observed.fp_per_minute,
                "actual_to_nominal_ratio": observed.fp_per_minute / target,
                "observed_tpr": observed.true_positive_rate,
                "null_tpr_median": float(np.median(null)),
                "excess_tpr": float(observed.true_positive_rate - np.median(null)),
                "idle_seconds_to_first_fp": float(cumulative_idle[idle_alarm_indices[0]])
                if len(idle_alarm_indices) else None,
                "first_fp_censor_idle_seconds": float(cumulative_idle[-1]),
                "rest_hazard_source_exposure_seconds": source_exposure.tolist(),
                "rest_hazard_source_alarm_counts": source_counts.tolist(),
            }
        condition_results[name] = output
    return {"subject": subject, "conditions": condition_results}


def bootstrap_median(values, seed):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (10000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-prior-grid", action="store_true")
    parser.add_argument("--half-life-sweep", action="store_true")
    args = parser.parse_args()
    conditions = CONDITIONS
    if args.anchor_prior_grid:
        conditions = (("static", None, 0.0), ("all", 30.0, 0.0),
                      ("all", 30.0, 140.0), ("all", 30.0, 280.0),
                      ("all", 30.0, 560.0), ("all", 30.0, 1120.0))
    if args.half_life_sweep:
        conditions = (("static", None, 0.0), ("all", 30.0, 0.0),
                      ("all", 10.0, 280.0), ("all", 20.0, 280.0),
                      ("all", 30.0, 280.0), ("all", 60.0, 280.0))
    stream_root = Path("data/interim/lee2019_pseudo_uncued_raw/session1")
    raw_root = Path("data/interim/lee2019_four_rest_raw/session1")
    cov_root = Path("data/interim/lee2019_four_rest/session1")
    subjects = list(range(1, 55))
    rows = [analyze(stream_root / f"sub-{s:02d}.npz", raw_root / f"sub-{s:02d}.npz",
                    cov_root / f"sub-{s:02d}.npz", s, condition_specs=conditions) for s in subjects]
    summary = {}
    for condition in rows[0]["conditions"]:
        csum = {}
        for target in map(str, TARGETS):
            ratios = [r["conditions"][condition][target]["actual_to_nominal_ratio"] for r in rows]
            fps = [r["conditions"][condition][target]["actual_fp_per_minute"] for r in rows]
            csum[target] = {"actual_fp_median": float(np.median(fps)),
                            "ratio_median": float(np.median(ratios)),
                            "ratio_ci": bootstrap_median(ratios, 20260805 + len(condition) + int(float(target)*100))}
        common, failures = [], 0
        for row in rows:
            pts = row["conditions"][condition]
            pairs = sorted((pts[str(t)]["actual_fp_per_minute"], pts[str(t)]["excess_tpr"])
                           for t in TARGETS)
            fp, ex = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
            if fp[0] <= 2 <= fp[-1]: common.append(float(np.interp(2, fp, ex)))
            else: failures += 1
        first = []
        for row in rows:
            p = row["conditions"][condition]["2.0"]
            first.append(p["idle_seconds_to_first_fp"] if p["idle_seconds_to_first_fp"] is not None
                         else p["first_fp_censor_idle_seconds"])
        csum["common_actual_fp2"] = {
            "bracket_failures": failures,
            "excess_median": float(np.median(common)) if common else None,
            "excess_ci": bootstrap_median(common, 20260905 + len(condition)) if common else None,
            "target2_time_or_censor_to_first_fp_median": float(np.median(first)),
        }
        # Proper KM median on cumulative valid-idle exposure at nominal target 2.
        durations, observed = [], []
        for row in rows:
            p2 = row["conditions"][condition]["2.0"]
            event = p2["idle_seconds_to_first_fp"] is not None
            durations.append(p2["idle_seconds_to_first_fp"] if event else
                             p2["first_fp_censor_idle_seconds"])
            observed.append(event)
        durations, observed = np.asarray(durations), np.asarray(observed)
        survival, km_median = 1.0, None
        for event_time in np.sort(np.unique(durations[observed])):
            survival *= 1 - np.sum((durations == event_time) & observed) / np.sum(durations >= event_time)
            if survival <= .5:
                km_median = float(event_time); break
        actual_fp_median = csum["2.0"]["actual_fp_median"]
        poisson_median = 60 * np.log(2) / actual_fp_median if actual_fp_median > 0 else None
        csum["common_actual_fp2"]["km_median_idle_seconds_to_first_fp"] = km_median
        csum["common_actual_fp2"]["poisson_expected_median_seconds"] = poisson_median
        csum["common_actual_fp2"]["km_to_poisson_median_ratio"] = (
            km_median / poisson_median if km_median is not None and poisson_median else None)
        fake_rows = [{"operating_points": {"2.0": row["conditions"][condition]["2.0"]}}
                     for row in rows]
        csum["target2_block_adjusted_hazard"] = poisson_cluster_model(fake_rows, "2.0")
        summary[condition] = csum
    result = {"subjects": 54, "summary": summary, "rows": rows}
    output = Path(
        "data/processed/lee2019_sequential_anchor_prior_half_life.json"
        if args.half_life_sweep else
        "data/processed/lee2019_sequential_anchor_prior.json"
        if args.anchor_prior_grid else
        "data/processed/lee2019_sequential_adaptation.json"
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: {"target2": v["2.0"], "common": v["common_actual_fp2"],
                           "hazard": v["target2_block_adjusted_hazard"]}
                      for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
