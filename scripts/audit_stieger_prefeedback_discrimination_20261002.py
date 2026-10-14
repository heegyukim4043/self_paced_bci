"""Audit whether frozen Stieger LOSO scores separate trial types before feedback.

The released cells are not continuous. Pre-feedback discrimination can arise
from target preparation, cue/context, or anticipatory MI; it is not a proof of
any single mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.analyze_stieger_cnn_gru_early_usability import (
    eligible, load_cache, prepare_levels, split_trials,
)

BINS = {
    "pre_feedback": (-1.0, 0.0),
    "post_0_1": (0.0, 1.0),
    "post_1_2": (1.0, 2.0),
}


def cell_value(trial: dict, lower: float, upper: float):
    times, score = trial["time"], trial["z"]
    selected = (times > lower + 1e-6) & (times <= upper + 1e-6) & np.isfinite(score)
    return float(np.median(score[selected])) if selected.sum() >= 8 else float("nan")


def within_run_null(values: np.ndarray, labels: np.ndarray, runs: np.ndarray,
                    *, shifts: int, seed: int) -> dict:
    valid = np.isfinite(values)
    observed = float(roc_auc_score(labels[valid], values[valid]))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(shifts):
        shifted = labels.copy()
        for run in np.unique(runs):
            loc = np.flatnonzero(runs == run)
            if len(loc) > 1:
                offset = int(rng.integers(1, len(loc)))
                shifted[loc] = np.roll(shifted[loc], offset)
        if len(np.unique(shifted[valid])) == 2:
            null.append(float(roc_auc_score(shifted[valid], values[valid])))
    array = np.asarray(null)
    return {
        "auc": observed,
        "null_median": float(np.median(array)),
        "auc_minus_null": observed - float(np.median(array)),
        "one_sided_permutation_p": float((1 + np.sum(array >= observed)) / (1 + len(array))),
        "n_active": int(np.sum(valid & labels)),
        "n_rest": int(np.sum(valid & ~labels)),
    }


def audit_one(path: Path, shifts: int) -> dict:
    cache = load_cache(path)
    trials = split_trials(cache)
    prepare_levels(trials)
    future = [t for t in trials if t["role"] == 2 and eligible(t)]
    if not future:
        raise ValueError("no future cells")
    labels = np.asarray([t["active"] for t in future], bool)
    runs = np.asarray([t["run"] for t in future], int)
    scores = {name: np.asarray([cell_value(t, *bounds) for t in future])
              for name, bounds in BINS.items()}
    scores["post0_minus_pre"] = scores["post_0_1"] - scores["pre_feedback"]
    rows = {}
    for index, (name, values) in enumerate(scores.items()):
        valid = np.isfinite(values)
        if len(np.unique(labels[valid])) != 2:
            rows[name] = {"status": "not_estimable", "n_valid": int(valid.sum())}
        else:
            rows[name] = within_run_null(
                values, labels, runs, shifts=shifts,
                seed=20261002 + 1000 * int(cache["audit"]["subject"])
                     + 10 * int(cache["audit"]["session"]) + index,
            )
    matched = np.isfinite(scores["pre_feedback"]) & np.isfinite(scores["post_0_1"])
    matched_result = {
        "n_active": int(np.sum(matched & labels)),
        "n_rest": int(np.sum(matched & ~labels)),
    }
    if len(np.unique(labels[matched])) == 2:
        matched_result["pre_auc"] = float(roc_auc_score(
            labels[matched], scores["pre_feedback"][matched]))
        matched_result["post_auc"] = float(roc_auc_score(
            labels[matched], scores["post_0_1"][matched]))
        matched_result["paired_auc_delta"] = (
            matched_result["post_auc"] - matched_result["pre_auc"])
    return {
        "subject": int(cache["audit"]["subject"]),
        "session": int(cache["audit"]["session"]),
        "future_cells": len(future), "bins": rows,
        "matched_pre_post": matched_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path(
        "data/interim/stieger2021_cnn_gru_loso_scores"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/stieger2021_prefeedback_discrimination_20261002.json"))
    args = parser.parse_args()
    files = sorted(args.cache_root.glob("session*/sub-*.npz"))
    if len(files) != 30:
        raise ValueError(f"expected 30 caches, found {len(files)}")
    rows = [audit_one(path, args.shifts) for path in files]
    summary = {}
    for name in (*BINS, "post0_minus_pre"):
        estimable = [r["bins"][name] for r in rows if "auc" in r["bins"][name]]
        summary[name] = {
            "estimable_sessions": len(estimable),
            "auc_median": float(np.median([v["auc"] for v in estimable])) if estimable else None,
            "auc_minus_null_median": float(np.median([
                v["auc_minus_null"] for v in estimable])) if estimable else None,
            "p_below_0_05_sessions": int(np.sum([
                v["one_sided_permutation_p"] < 0.05 for v in estimable])),
        }
    matched_rows = [r for r in rows if "paired_auc_delta" in r["matched_pre_post"]]
    subjects = sorted({r["subject"] for r in matched_rows})
    by_subject = {
        subject: [r["matched_pre_post"]["paired_auc_delta"]
                  for r in matched_rows if r["subject"] == subject]
        for subject in subjects
    }
    rng = np.random.default_rng(20261002)
    bootstrap = []
    for _ in range(2000):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        bootstrap.append(float(np.median([
            value for subject in draw for value in by_subject[int(subject)]
        ])))
    summary["matched_pre_post"] = {
        "estimable_sessions": len(matched_rows),
        "paired_auc_delta_median": float(np.median([
            r["matched_pre_post"]["paired_auc_delta"] for r in matched_rows])),
        "subject_cluster_bootstrap_ci95": np.quantile(
            bootstrap, [0.025, 0.975]).tolist(),
        "median_active_cells": float(np.median([
            r["matched_pre_post"]["n_active"] for r in matched_rows])),
        "median_rest_cells": float(np.median([
            r["matched_pre_post"]["n_rest"] for r in matched_rows])),
    }
    payload = {
        "protocol": {
            "score": "frozen source-pooled subject-LOSO CNN-GRU z, future trial cells only",
            "bins_seconds_from_feedback": BINS,
            "null": f"{args.shifts} within-run active/rest label circular shifts",
            "meaning": "pre-feedback condition discrimination, not isolated cue or MI mechanism",
            "claim_limit": "cells, no natural continuous idle; exploratory multi-bin audit",
        },
        "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
