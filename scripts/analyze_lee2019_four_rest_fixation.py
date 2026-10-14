"""Compare all four Lee2019 rest blocks with train/test fixation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import moving_average_dwell
from uncued_mi.riemannian import tangent_features

STEP_SECONDS = 0.1
RESTS = ("pre_train", "post_train", "pre_test", "post_test")
FIXATIONS = (
    "train_fixation_full", "train_fixation_early", "train_fixation_late",
    "test_fixation_full", "test_fixation_early", "test_fixation_late",
)


def reference(covariances: np.ndarray) -> np.ndarray:
    matrix = np.mean(np.asarray(covariances, dtype=np.float64), axis=0)
    ridge = max(float(np.trace(matrix) / len(matrix) * 1e-6), 1e-12)
    return matrix + np.eye(len(matrix)) * ridge


def alarms_grouped(
    scores: np.ndarray, groups: np.ndarray, threshold: float
) -> tuple[int, float]:
    alarms = 0
    exposure = 0.0
    for group in np.unique(groups):
        values = scores[groups == group]
        decision = moving_average_dwell(
            values, threshold, STEP_SECONDS,
            ma_seconds=0.5, dwell_seconds=0.3, refractory_seconds=1.0,
        )
        alarms += int(decision.alarms.sum())
        exposure += len(values) * STEP_SECONDS
    return alarms, exposure


def tune(scores: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.999, 250)))
    best = None
    for threshold in grid:
        count, seconds = alarms_grouped(scores, groups, float(threshold))
        fp = 60.0 * count / seconds
        candidate = (abs(fp - 2.0), -float(threshold), float(threshold), fp)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[2], best[3]


def score_rate(
    scores: np.ndarray, threshold: float, groups: np.ndarray | None = None
) -> float:
    if groups is None:
        groups = np.zeros(len(scores), dtype=int)
    count, seconds = alarms_grouped(scores, groups, threshold)
    return 60.0 * count / seconds


def fixation_masks(relative_end: np.ndarray) -> dict[str, np.ndarray]:
    relative_end = np.asarray(relative_end)
    return {
        "full": (relative_end >= -2.0 - 1e-6) & (relative_end <= 1e-6),
        "early": (relative_end >= -2.0 - 1e-6) & (relative_end < -1.0 - 1e-6),
        "late": (relative_end >= -1.0 - 1e-6) & (relative_end <= 1e-6),
    }


def analyze(path: Path, subject: int, session: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        cov = {key: data[f"{key}_covariance"].astype(np.float64) for key in RESTS}
        ref = reference(cov["pre_train"])
        rest_scores = {
            key: tangent_features(value, ref) for key, value in cov.items()
        }
        train_command_trial = data["train_command_trial"]
        train_iti_trial = data["train_idle_iti_trial"]
        command_mask = train_command_trial < 50
        iti_mask = train_iti_trial < 50
        command_x = tangent_features(
            data["train_command_covariance"][command_mask], ref
        )
        iti_x = tangent_features(data["train_idle_iti_covariance"][iti_mask], ref)
        model = LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="liblinear"
        ).fit(
            np.r_[command_x, iti_x],
            np.r_[np.ones(len(command_x)), np.zeros(len(iti_x))],
        )
        rest_score = {
            key: model.decision_function(value) for key, value in rest_scores.items()
        }
        iti_score = model.decision_function(iti_x)
        threshold, tuning_fp = tune(iti_score, train_iti_trial[iti_mask])

        fixation_score: dict[str, np.ndarray] = {}
        fixation_group: dict[str, np.ndarray] = {}
        for phase in ("train", "test"):
            features = tangent_features(data[f"{phase}_fixation_covariance"], ref)
            scores = model.decision_function(features)
            trials = data[f"{phase}_fixation_trial"]
            masks = fixation_masks(data[f"{phase}_fixation_relative_end"])
            for part, mask in masks.items():
                name = f"{phase}_fixation_{part}"
                fixation_score[name] = scores[mask]
                fixation_group[name] = trials[mask]

    rest_rate = {
        key: score_rate(value, threshold) for key, value in rest_score.items()
    }
    fixation_rate = {
        key: score_rate(value, threshold, fixation_group[key])
        for key, value in fixation_score.items()
    }
    comparisons = {}
    for rest in RESTS:
        for fixation in FIXATIONS:
            r = rest_score[rest]
            f = fixation_score[fixation]
            auc = roc_auc_score(
                np.r_[np.zeros(len(r)), np.ones(len(f))],
                np.r_[r, f],
            )
            comparisons[f"{fixation}_vs_{rest}"] = {
                "signed_auc_fixation_higher": float(auc),
                "separability_auc": float(max(auc, 1.0 - auc)),
                "median_score_difference_fixation_minus_rest": float(
                    np.median(f) - np.median(r)
                ),
                "fixation_alarm_per_minute": fixation_rate[fixation],
                "rest_alarm_per_minute": rest_rate[rest],
            }
    return {
        "subject": subject,
        "session": session,
        "threshold": threshold,
        "training_iti_fp_per_minute": tuning_fp,
        "rest": {
            key: {
                "windows": len(rest_score[key]),
                "score_median": float(np.median(rest_score[key])),
                "alarm_per_minute": rest_rate[key],
            }
            for key in RESTS
        },
        "fixation": {
            key: {
                "windows": len(fixation_score[key]),
                "score_median": float(np.median(fixation_score[key])),
                "alarm_per_minute": fixation_rate[key],
            }
            for key in FIXATIONS
        },
        "comparisons": comparisons,
    }


def bootstrap_median(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("data/interim/lee2019_four_rest"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_four_rest_fixation.json"),
    )
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            rows.append(
                analyze(
                    args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz",
                    subject, session,
                )
            )
            print(f"done session={session} subject={subject}", flush=True)
    summary = []
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        comparisons = {}
        for rest in RESTS:
            for fixation in FIXATIONS:
                name = f"{fixation}_vs_{rest}"
                auc = np.asarray([
                    row["comparisons"][name]["signed_auc_fixation_higher"]
                    for row in selected
                ])
                difference = np.asarray([
                    row["comparisons"][name][
                        "median_score_difference_fixation_minus_rest"
                    ]
                    for row in selected
                ])
                comparisons[name] = {
                    "signed_auc_median": float(np.median(auc)),
                    "signed_auc_ci": bootstrap_median(
                        auc, 1000 + session, args.bootstrap
                    ),
                    "score_difference_median": float(np.median(difference)),
                    "score_difference_ci": bootstrap_median(
                        difference, 2000 + session, args.bootstrap
                    ),
                    "fixation_alarm_per_minute_median": float(np.median([
                        row["comparisons"][name]["fixation_alarm_per_minute"]
                        for row in selected
                    ])),
                    "rest_alarm_per_minute_median": float(np.median([
                        row["comparisons"][name]["rest_alarm_per_minute"]
                        for row in selected
                    ])),
                }
        summary.append({
            "session": session,
            "subjects": len(selected),
            "comparisons": comparisons,
        })
    payload = {
        "protocol": {
            "reference": "current-session preTrainRest mean covariance",
            "model": "within-subject first-50-trial Riemannian TS+LR gate",
            "threshold": "first-50-trial train ITI FP=2/min",
            "rest_blocks": list(RESTS),
            "fixation_conditions": list(FIXATIONS),
            "signed_auc": "P(fixation gate score > rest gate score)",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
