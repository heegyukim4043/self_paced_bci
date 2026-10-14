"""Consolidate exploratory trajectories and causal scalar validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def interpolate(points):
    pairs = sorted((float(v["actual_fp_per_minute"]), float(v["excess_tpr"]))
                   for v in points.values())
    x = np.asarray([a for a, _ in pairs]); y = np.asarray([b for _, b in pairs])
    unique = np.unique(x)
    y = np.asarray([np.median(y[x == value]) for value in unique]); x = unique
    if len(x) < 2 or 2 < x.min() or 2 > x.max():
        return None
    return float(np.interp(2, x, y))


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues)+1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty(len(pvalues)); output[order] = np.minimum(ranked, 1.)
    return output


def bootstrap(values, seed):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (10000, len(values)))]
    return np.quantile(np.median(draws, axis=1), [.025, .975]).tolist()


def main():
    processed = Path("data/processed")
    screen = json.loads((processed / "lee2019_sequential_feature_screen.json").read_text())
    scalar = json.loads((processed / "lee2019_scalar_feature_validation.json").read_text())
    adaptation = json.loads((processed / "lee2019_sequential_adaptation.json").read_text())
    names = list(screen["summary"])
    qvalues = bh_adjust([
        screen["summary"][name]["mi_greater_precue_wilcoxon_p_uncorrected"]
        for name in names])
    exploratory = {}
    for name, qvalue in zip(names, qvalues):
        item = screen["summary"][name]
        exploratory[name] = {
            "net_rescue_median": item["net_rescue_median"],
            "mi_minus_precue_median": item["mi_minus_precue_median"],
            "within_trial_bh_q": float(qvalue),
        }

    baseline = {int(row["subject"]): interpolate(row["conditions"]["all_hl30"])
                for row in adaptation["rows"]}
    causal = {}
    for index, name in enumerate(scalar["summary"]):
        feature_values, baseline_values = [], []
        for row in scalar["subjects"]:
            subject = int(row["subject"])
            feature = row["features"][name]["excess_at_actual_fp2"]
            comparator = baseline[subject]
            if feature is not None and comparator is not None:
                feature_values.append(feature); baseline_values.append(comparator)
        feature_values = np.asarray(feature_values); baseline_values = np.asarray(baseline_values)
        delta = feature_values - baseline_values
        causal[name] = {
            **scalar["summary"][name],
            "paired_subjects": len(delta),
            "riemannian_excess_median_same_subjects": float(np.median(baseline_values)),
            "minus_riemannian_median": float(np.median(delta)),
            "minus_riemannian_ci95": bootstrap(delta, 20260820+index),
            "paired_wilcoxon_p_two_sided": float(wilcoxon(delta).pvalue),
        }
    output = {
        "exploratory": exploratory,
        "causal_validation": causal,
        "decision": (
            "Several features change sequentially in Riemannian non-response trials, "
            "but no scalar candidate exceeds the causal Riemannian+EWMA comparator "
            "at actual FP=2. Retain trajectories as physiology diagnostics; do not "
            "replace the deployed score."
        ),
    }
    target = processed / "lee2019_feature_screen_decision.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["causal_validation"], indent=2))


if __name__ == "__main__":
    main()
