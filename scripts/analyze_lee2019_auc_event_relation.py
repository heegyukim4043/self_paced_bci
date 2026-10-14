"""Relate matched LOSO window AUC to fixed-threshold event excess in Lee2019."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.riemannian import tangent_features
from scripts.evaluate_lee2019_loso_deployment import (
    SUBJECTS, fit_model, load_covariances, prepare_target, source_reference,
    source_training, tune_anchor,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma, ewma_z, ma_trace,
)
from scripts.run_lee2019_pseudo_uncued_riemannian import reference


def sensitivity(times: np.ndarray, alarms: np.ndarray, onsets: np.ndarray) -> float:
    return float(np.mean([
        np.any(alarms & (times >= onset) & (times <= onset + 2.0))
        for onset in onsets
    ]))


def analyze_subject(subject, session, domains, own_cache, auc_lookup, shifts, rng):
    anchor_cov, stream_cov, events, times, starts, allowed = prepare_target(subject, session)
    target_ref = reference(domains[subject]["pre"])
    x_command, x_idle = source_training(
        domains, subject, "A_target_rest", source_reference(domains, subject), own_cache
    )
    scaler, model = fit_model(x_command, x_idle)
    anchor = model.decision_function(scaler.transform(tangent_features(anchor_cov, target_ref)))
    raw = model.decision_function(scaler.transform(tangent_features(stream_cov, target_ref)))
    threshold = tune_anchor(anchor)[4.0]
    z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
    evalmask = times >= events[10, 2] / 250.0
    onsets = events[10:, 5] / 250.0 + 0.5
    observed_alarms = alarms_from_ma(ma_trace(z), threshold, allowed) & evalmask
    observed = sensitivity(times, observed_alarms, onsets)
    minimum_shift = int(20 / 0.1)
    candidates = np.arange(minimum_shift, len(z) - minimum_shift)
    offsets = rng.choice(candidates, min(shifts, len(candidates)), replace=False)
    null = []
    for offset in offsets:
        shifted = alarms_from_ma(ma_trace(np.roll(z, int(offset))), threshold, allowed) & evalmask
        null.append(sensitivity(times, shifted, onsets))
    null_median = float(np.median(null))
    return {
        "subject": subject, "session": session,
        "window_auc": float(auc_lookup[(session, subject)]),
        "observed_sensitivity_2s": observed,
        "circular_shift_null_2s": null_median,
        "excess_sensitivity_2s": observed - null_median,
        "nominal_fp_per_min": 4.0,
    }


def cluster_bootstrap_spearman(rows, repeats, seed):
    rng = np.random.default_rng(seed)
    subjects = np.asarray(SUBJECTS)
    by_subject = {subject: [row for row in rows if row["subject"] == subject] for subject in SUBJECTS}
    values = []
    for _ in range(repeats):
        draw = rng.choice(subjects, len(subjects), replace=True)
        sampled = [row for subject in draw for row in by_subject[int(subject)]]
        rho = spearmanr(
            [row["window_auc"] for row in sampled],
            [row["excess_sensitivity_2s"] for row in sampled],
        ).statistic
        if np.isfinite(rho):
            values.append(rho)
    return np.quantile(values, [0.025, 0.975]).tolist()


def group_summary(rows, predicate):
    selected = [row for row in rows if predicate(row)]
    if not selected:
        return {"n": 0}
    return {
        "n": len(selected),
        "auc_range": [min(row["window_auc"] for row in selected), max(row["window_auc"] for row in selected)],
        "auc_median": float(np.median([row["window_auc"] for row in selected])),
        "observed_sensitivity_2s_median": float(np.median([row["observed_sensitivity_2s"] for row in selected])),
        "circular_null_2s_median": float(np.median([row["circular_shift_null_2s"] for row in selected])),
        "excess_sensitivity_2s_median": float(np.median([row["excess_sensitivity_2s"] for row in selected])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_auc_event_relation.json"))
    args = parser.parse_args()
    deployment = json.loads(Path("data/processed/lee2019_loso_deployment.json").read_text(encoding="utf-8"))
    auc_lookup = {
        (int(row["session"]), int(row["subject"])): row["arms"]["A_target_rest"]["window_auc"]
        for row in deployment["rows"]
    }
    rng = np.random.default_rng(20260813)
    rows = []
    for session in (1, 2):
        domains = load_covariances(session)
        own_cache = {}
        for subject in SUBJECTS:
            own_ref = reference(domains[subject]["pre"])
            own_cache[subject] = {
                "command": tangent_features(domains[subject]["command"], own_ref),
                "idle": tangent_features(domains[subject]["idle"], own_ref),
            }
        for subject in SUBJECTS:
            rows.append(analyze_subject(subject, session, domains, own_cache, auc_lookup, args.shifts, rng))
            print(f"session{session} sub-{subject:02d}", flush=True)
    auc = np.asarray([row["window_auc"] for row in rows])
    excess = np.asarray([row["excess_sensitivity_2s"] for row in rows])
    test = spearmanr(auc, excess)
    summary = {
        "n_subject_sessions": len(rows),
        "spearman_rho": float(test.statistic),
        "spearman_p": float(test.pvalue),
        "subject_cluster_bootstrap_ci": cluster_bootstrap_spearman(rows, args.bootstrap, 20260813),
        "all": group_summary(rows, lambda row: True),
        "auc_ge_0.70": group_summary(rows, lambda row: row["window_auc"] >= 0.70),
        "auc_ge_0.80": group_summary(rows, lambda row: row["window_auc"] >= 0.80),
        "auc_ge_0.85": group_summary(rows, lambda row: row["window_auc"] >= 0.85),
    }
    payload = {
        "protocol": {
            "model": "Riemannian LOSO Arm A with target pre-rest reference",
            "window_auc": "locked last 40 trials; strict full-window support",
            "event": "fixed nominal FP=4 threshold chosen from target pre-rest; 2-s cue-relative pseudo-onset tolerance",
            "null": f"median of {args.shifts} circular shifts per subject-session",
            "bootstrap": "subject-clustered across the two sessions",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
