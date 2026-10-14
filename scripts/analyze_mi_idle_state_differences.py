"""Subject-level statistical comparison of MI task and idle/rest states."""

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

from uncued_mi.riemannian import tangent_features

RESTS = ("pre_train", "post_train", "pre_test", "post_test")
CENTRAL = ("C3", "Cz", "C4")


def complete_interval_mask(times, intervals, window_seconds):
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in intervals:
        mask |= (
            (times - window_seconds >= start - 1e-9)
            & (times <= stop + 1e-9)
        )
    return mask


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, values[index] * (len(values) - rank))
        adjusted[index] = min(running, 1.0)
    return adjusted


def bootstrap_median(values, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def comparisons(rows, metric, conditions, seed, repeats):
    task = np.asarray([row[metric]["mi"] for row in rows])
    result = []
    for condition in conditions:
        idle = np.asarray([row[metric][condition] for row in rows])
        difference = task - idle
        test = wilcoxon(
            difference, zero_method="pratt", alternative="two-sided"
        )
        result.append({
            "condition": condition,
            "mi_median": float(np.median(task)),
            "idle_median": float(np.median(idle)),
            "mi_minus_idle_median": float(np.median(difference)),
            "difference_ci": bootstrap_median(
                difference, seed + len(result), repeats
            ),
            "wilcoxon_statistic": float(test.statistic),
            "p": float(test.pvalue),
            "wins_ties_losses": [
                int(np.sum(difference > 0)),
                int(np.sum(difference == 0)),
                int(np.sum(difference < 0)),
            ],
        })
    adjusted = holm_adjust([item["p"] for item in result])
    for item, value in zip(result, adjusted):
        item["holm_p"] = float(value)
    return result


def fit_gate(command, iti, rest):
    negative = np.r_[iti, rest]
    scaler = StandardScaler().fit(np.r_[command, negative])
    model = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=3000,
        random_state=20260729,
    ).fit(
        scaler.transform(np.r_[command, negative]),
        np.r_[np.ones(len(command)), np.zeros(len(negative))],
    )
    return scaler, model


def analyze_subject(tier_path, covariance_path, subject, session):
    with np.load(tier_path, allow_pickle=False) as tier:
        channel_lookup = {
            name.casefold(): index
            for index, name in enumerate(tier["channels"].tolist())
        }
        central = [channel_lookup[name.casefold()] for name in CENTRAL]
        times = tier["continuous_times"]
        command_mask = complete_interval_mask(
            times, tier["command_intervals"], 2.0
        )
        iti_mask = complete_interval_mask(times, tier["iti_intervals"], 2.0)
        bandpower = {
            "mi": tier["continuous_features"][command_mask][:, central],
            "iti": tier["continuous_features"][iti_mask][:, central],
            "pre_train": tier["pre_features"][:, central],
            "post_train": tier["post_features"][:, central],
        }
        metrics = {}
        for band_index, name in enumerate(("mu_log_power", "beta_log_power")):
            metrics[name] = {
                state: float(np.median(values[:, :, band_index]))
                for state, values in bandpower.items()
            }

    with np.load(covariance_path, allow_pickle=False) as data:
        covariance_channels = data["channels"].tolist()
        lookup = {
            name.casefold(): index
            for index, name in enumerate(covariance_channels)
        }
        central_cov = [lookup[name.casefold()] for name in CENTRAL]
        pre = data["pre_train_covariance"]
        split = len(pre) // 2
        reference = np.mean(pre[:split], axis=0)
        ridge = max(float(np.trace(reference) / len(reference) * 1e-6), 1e-12)
        reference = reference + np.eye(len(reference)) * ridge

        covariance_states = {
            "mi": data["train_command_covariance"],
            "iti": data["train_idle_iti_covariance"],
            **{
                name: data[f"{name}_covariance"]
                for name in RESTS
            },
        }
        metrics["log_variance_8_30"] = {
            state: float(np.median(np.log(np.maximum(
                np.diagonal(values, axis1=1, axis2=2)[:, central_cov],
                1e-12,
            ))))
            for state, values in covariance_states.items()
        }

        tangent = {
            state: tangent_features(values, reference)
            for state, values in covariance_states.items()
        }
        command_trials = data["train_command_trial"]
        iti_trials = data["train_idle_iti_trial"]
        scaler, model = fit_gate(
            tangent["mi"][command_trials < 50],
            tangent["iti"][iti_trials < 50],
            tangent["pre_train"][:split],
        )
        metrics["gate_score"] = {
            "mi": float(np.median(model.decision_function(
                scaler.transform(tangent["mi"][command_trials >= 50])
            ))),
            "iti": float(np.median(model.decision_function(
                scaler.transform(tangent["iti"][iti_trials >= 50])
            ))),
            **{
                name: float(np.median(model.decision_function(
                    scaler.transform(values[len(values)//2:])
                )))
                for name, values in tangent.items()
                if name in RESTS
            },
        }
    return {"subject": subject, "session": session, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument(
        "--cov-root", type=Path, default=Path("data/interim/lee2019_four_rest")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=range(1, 55))
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/mi_idle_state_differences.json"),
    )
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(analyze_subject(
                args.tier_root / f"session{session}"
                / f"sub-{subject:02d}_tier0_features.npz",
                args.cov_root / f"session{session}" / f"sub-{subject:02d}.npz",
                subject, session,
            ))
        print(f"state comparison session={session}", flush=True)

    summary = []
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        metrics = {
            "mu_log_power": comparisons(
                selected, "mu_log_power",
                ("pre_train", "post_train", "iti"),
                1000 + session, args.bootstrap,
            ),
            "beta_log_power": comparisons(
                selected, "beta_log_power",
                ("pre_train", "post_train", "iti"),
                2000 + session, args.bootstrap,
            ),
            "log_variance_8_30": comparisons(
                selected, "log_variance_8_30",
                (*RESTS, "iti"), 3000 + session, args.bootstrap,
            ),
            "gate_score": comparisons(
                selected, "gate_score",
                (*RESTS, "iti"), 4000 + session, args.bootstrap,
            ),
        }
        summary.append({
            "session": session, "subjects": len(selected), "metrics": metrics
        })
    payload = {
        "protocol": {
            "unit": "subject median",
            "central_channels": CENTRAL,
            "test": "two-sided paired Wilcoxon with within-metric Holm correction",
            "difference": "MI minus idle/rest",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
