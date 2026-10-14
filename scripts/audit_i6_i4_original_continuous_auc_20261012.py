"""Exploratory future MI-vs-task-idle MA-score AUC on original Lee timelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_e4_original_continuous_fp_20261009 import STEP, _trial_strata, _union
from scripts.run_e4b_task_idle_20261010 import _prepare as prepare_tier0
from scripts.run_e4c_original_riemannian_fp_20261010 import prepare as prepare_riemannian
from scripts.run_i6_cnn_gru_original_continuous_20261012 import prepare_cache as prepare_i6


def ci_median(values, rng, draws=10000):
    values = np.asarray(values, dtype=float)
    sampled = rng.integers(len(values), size=(draws, len(values)))
    return [float(x) for x in np.quantile(np.median(values[sampled], axis=1), [.025, .975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i6-root", type=Path, default=Path("data/interim/lee2019_i6_cnn_gru_original_cuda"))
    ap.add_argument("--riem-root", type=Path, default=Path("data/interim/lee2019_e4c_riemannian_original"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--geometry", type=Path, default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/i6_i4_original_continuous_auc_20261012.json"))
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()
    if args.output.exists() and not args.limit_cases:
        raise FileExistsError(args.output)
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))["rows"]
    if len(geometry) != 108:
        raise AssertionError("incomplete original-continuous geometry")
    if args.limit_cases:
        geometry = geometry[:args.limit_cases]
    out = []
    for g in geometry:
        s, sess = g["subject"], g["session"]
        tier = args.tier0_root / f"session{sess}" / f"sub-{s:02d}_tier0_features.npz"
        riem = args.riem_root / f"session{sess}" / f"sub-{s:02d}.npz"
        i6 = args.i6_root / f"session{sess}" / f"sub-{s:02d}.npz"
        preps = {
            "tier0": prepare_tier0(tier, s, sess),
            "riemannian": prepare_riemannian(riem, tier, s, sess),
            "cnn_gru_stateful": prepare_i6(i6, tier, s, sess, "stateful"),
            "cnn_gru_window_reset": prepare_i6(i6, tier, s, sess, "window_reset"),
        }
        reference_positive_times = None
        reference_idle_times = None
        for name, (times, cues, fixation, iti, ma) in preps.items():
            idle = _union(_trial_strata(times, cues, fixation, iti, list(range(70, 100))), False)
            positive = np.zeros(len(times), dtype=bool)
            for cue in cues[70:100]:
                positive |= (times >= cue + .5 - .0001) & (times <= cue + 3.5 + .0001)
            if np.any(idle & positive) or idle.sum() < 100 or positive.sum() < 100:
                raise AssertionError(f"invalid MI/idle masks: {name} S{sess} sub{s}")
            if not np.isclose(idle.sum() * STEP, g["seconds"]["E_full"], atol=.001):
                raise AssertionError(f"idle exposure mismatch: {name} S{sess} sub{s}")
            if reference_positive_times is None:
                reference_positive_times = times[positive]
                reference_idle_times = times[idle]
            elif (not np.allclose(times[positive], reference_positive_times, rtol=0, atol=.001)
                  or not np.allclose(times[idle], reference_idle_times, rtol=0, atol=.001)):
                raise AssertionError(f"future decision timestamps mismatch: {name} S{sess} sub{s}")
            if not np.all(np.isfinite(ma[idle | positive])):
                raise AssertionError(f"nonfinite MA score: {name} S{sess} sub{s}")
            labels = np.r_[np.ones(positive.sum(), dtype=int), np.zeros(idle.sum(), dtype=int)]
            values = np.r_[ma[positive], ma[idle]]
            out.append({"subject": s, "session": sess, "score": name,
                        "auc": float(roc_auc_score(labels, values)),
                        "n_positive_decisions": int(positive.sum()),
                        "n_idle_decisions": int(idle.sum())})
    if args.limit_cases:
        print(f"AUC mask preflight {len(geometry)} case(s), no cohort output")
        return
    rng = np.random.default_rng(20261012)
    indexed = {(r["subject"], r["session"], r["score"]): r for r in out}
    scores = ("tier0", "riemannian", "cnn_gru_stateful", "cnn_gru_window_reset")
    summary = {}
    for sess in (1, 2):
        summary[str(sess)] = {}
        for score in scores:
            values = [indexed[(s, sess, score)]["auc"] for s in range(1, 55)]
            summary[str(sess)][score] = {"median_auc": float(np.median(values)),
                                         "median_ci95": ci_median(values, rng)}
        for other in ("tier0", "riemannian", "cnn_gru_window_reset"):
            delta = [indexed[(s, sess, "cnn_gru_stateful")]["auc"] - indexed[(s, sess, other)]["auc"]
                     for s in range(1, 55)]
            summary[str(sess)][f"cnn_gru_stateful_minus_{other}"] = {
                "median_paired_delta_auc": float(np.median(delta)),
                "median_paired_ci95": ci_median(delta, rng)}
    result = {
        "status": "Lee original-continuous cue-relative window AUC; exploratory; correlated windows",
        "positive": "future trial70-99 cue+0.5..3.5 s MA decision points",
        "negative": "same future trials' E4/I4 eligible fixation and ITI MA decision points",
        "unit": "subject, sessions separately", "rows": out, "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
