"""Block-controlled rest hazard and idle-exposure KM analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln
from scipy.stats import norm


def poisson_cluster_model(rows: list[dict], target: str) -> dict:
    design, counts, exposure, groups = [], [], [], []
    for subject_index, row in enumerate(rows):
        point = row["operating_points"][target]
        source_counts = np.asarray(point["rest_hazard_source_alarm_counts"])
        source_exposure = np.asarray(point["rest_hazard_source_exposure_seconds"])
        for source in range(3):
            # Early combines 2--5 and 5--10 s; late is 20+ s. The 10--20 s
            # transition bin is omitted from this prespecified contrast.
            for late in (0, 1):
                bins = [0, 1] if late == 0 else [3]
                design.append([1.0, late, source == 1, source == 2])
                counts.append(source_counts[source, bins].sum())
                exposure.append(source_exposure[source, bins].sum())
                groups.append(subject_index)
    x = np.asarray(design, float)
    y = np.asarray(counts, float)
    exposure = np.asarray(exposure, float)
    groups = np.asarray(groups)
    offset = np.log(np.maximum(exposure, 1e-12))
    beta = np.zeros(x.shape[1])
    beta[0] = np.log((y.sum() + 0.5) / exposure.sum())
    for _ in range(100):
        mu = np.exp(np.clip(offset + x @ beta, -30, 30))
        information = x.T @ (mu[:, None] * x)
        score = x.T @ (y - mu)
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(np.clip(offset + x @ beta, -30, 30))
    bread = np.linalg.inv(x.T @ (mu[:, None] * x))
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in np.unique(groups):
        score_g = x[groups == group].T @ (y[groups == group] - mu[groups == group])
        meat += np.outer(score_g, score_g)
    n_groups, n_obs, p = len(np.unique(groups)), len(y), x.shape[1]
    correction = (n_groups / (n_groups - 1)) * ((n_obs - 1) / (n_obs - p))
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(covariance))
    z = beta[1] / se[1]
    return {
        "model": "Poisson log-rate, offset=log(exposure), subject-cluster robust SE",
        "contrast": "late 20+s vs early 2-10s, adjusted for rest source",
        "late_log_rate_ratio": float(beta[1]),
        "late_rate_ratio": float(np.exp(beta[1])),
        "late_rate_ratio_ci": np.exp(beta[1] + np.asarray([-1, 1]) * 1.96 * se[1]).tolist(),
        "late_z": float(z),
        "late_p_two_sided": float(2 * norm.sf(abs(z))),
        "source_pre_test_rate_ratio_vs_post_train": float(np.exp(beta[2])),
        "source_post_test_rate_ratio_vs_post_train": float(np.exp(beta[3])),
    }


def kaplan_meier(rows: list[dict], target: str):
    durations, observed = [], []
    for row in rows:
        point = row["operating_points"][target]
        event = bool(point["first_false_positive_observed"])
        durations.append(point["idle_seconds_to_first_false_positive"] if event else
                         point["first_false_positive_censor_idle_seconds"])
        observed.append(event)
    durations, observed = np.asarray(durations), np.asarray(observed)
    times, survival = [0.0], [1.0]
    current = 1.0
    for time in np.sort(np.unique(durations[observed])):
        at_risk = np.sum(durations >= time)
        events = np.sum((durations == time) & observed)
        current *= 1 - events / at_risk
        times.append(float(time)); survival.append(float(current))
    return durations, observed, np.asarray(times), np.asarray(survival)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/lee2019_pseudo_uncued_riemannian.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_pseudo_uncued_survival_hazard.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("data/processed/lee2019_pseudo_uncued_km_fp2.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    target = "2.0"
    model = poisson_cluster_model(data["rows"], target)
    durations, observed, times, survival = kaplan_meier(data["rows"], target)
    model["subjects"] = len(data["rows"])
    model["subjects_with_false_positive"] = int(observed.sum())
    model["median_idle_seconds_to_first_fp"] = (
        float(times[np.flatnonzero(survival <= 0.5)[0]]) if np.any(survival <= 0.5) else None)
    model["km_times_seconds"] = times.tolist()
    model["km_survival"] = survival.tolist()
    args.output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(times, survival, where="post", color="#1769aa", linewidth=2)
    ax.set(xlabel="Cumulative valid idle exposure (s)",
           ylabel="Probability of remaining FP-free",
           title="Lee2019 pseudo-uncued stream: time to first FP\n(anchor target 2 FP/min)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.figure, dpi=180); plt.close(fig)
    print(json.dumps({k: v for k, v in model.items() if not k.startswith("km_")}, indent=2))


if __name__ == "__main__":
    main()
