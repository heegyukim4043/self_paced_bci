"""Source-only, causal EEG splice-detector diagnostic on Lee v2 streams.

The classifier is trained on rest-rest decoys versus ordinary rest points.
Trial-start labels are NEVER used in training or threshold calibration.
The detector is intentionally simple and does not certify G1c' safety.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import rankdata

STEP = 25  # 0.1 s at 250 Hz
FS = 250
SEED = 20261008


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.r_[pos, neg])
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def nearest_distance(points: np.ndarray, seams: np.ndarray) -> np.ndarray:
    index = np.searchsorted(seams, points)
    left = np.abs(points - seams[np.clip(index - 1, 0, len(seams) - 1)])
    right = np.abs(points - seams[np.clip(index, 0, len(seams) - 1)])
    return np.minimum(left, right)


def load_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        x = np.asarray(saved["signal"], dtype=np.float64)
        state = saved["state"]
        seams = saved["seams"]
        events = saved["events"]
    centers = np.arange(FS, len(x) - 50, STEP, dtype=np.int64)
    cs = np.concatenate((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
    cs2 = np.concatenate((np.zeros((1, x.shape[1])), np.cumsum(x * x, axis=0)))
    p = (cs[centers] - cs[centers - 50]) / 50
    q = (cs[centers + 50] - cs[centers]) / 50
    e_p = (cs2[centers] - cs2[centers - 50]) / 50
    e_q = (cs2[centers + 50] - cs2[centers]) / 50
    delta = q - p
    features = np.column_stack((
        np.log1p(np.sqrt(np.mean(delta * delta, axis=1))),
        np.abs(np.log((np.mean(e_q, axis=1) + 1e-9) /
                      (np.mean(e_p, axis=1) + 1e-9))),
        np.log1p(np.sqrt(np.mean((x[centers] - x[centers - 1]) ** 2, axis=1))),
        np.log1p(np.sqrt(np.mean((np.sqrt(np.maximum(e_q, 0)) -
                                    np.sqrt(np.maximum(e_p, 0))) ** 2, axis=1))),
    ))
    decoy = np.sort(seams[seams[:, 1] == 0, 0])
    trial = np.sort(seams[seams[:, 1] == 1, 0])
    all_seams = np.sort(seams[:, 0])
    distance = nearest_distance(centers, all_seams)
    positive = nearest_distance(centers, decoy) <= 2 * STEP
    quiet_rest = ((state[centers - 50] == 0) & (state[centers + 50] == 0) &
                  (distance >= 5 * STEP))
    assert np.sum(positive) > 0 and np.sum(quiet_rest) > 0
    return {"features": features, "centers": centers, "decoy": decoy,
            "trial": trial, "all_seams": all_seams,
            "positive": positive, "quiet_rest": quiet_rest,
            "events": events, "stream_samples": len(x)}


def fit_source(items: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.concatenate([item["features"][item["positive"]] for item in items])
    negative = np.concatenate([item["features"][item["quiet_rest"]] for item in items])
    source_mean = np.mean(negative, axis=0)
    source_sd = np.maximum(np.std(negative, axis=0), 1e-8)
    pos_z = (positive - source_mean) / source_sd
    neg_z = (negative - source_mean) / source_sd
    mu_p, mu_n = np.mean(pos_z, axis=0), np.mean(neg_z, axis=0)
    cov = (np.cov(pos_z, rowvar=False) + np.cov(neg_z, rowvar=False)) / 2
    cov += np.eye(cov.shape[0]) * (.1 * np.trace(cov) / cov.shape[0] + 1e-6)
    weights = np.linalg.solve(cov, mu_p - mu_n)
    return source_mean, source_sd, weights


def seam_scores(scores: np.ndarray, centers: np.ndarray,
                seams: np.ndarray) -> np.ndarray:
    result = np.empty(len(seams))
    for i, seam in enumerate(seams):
        near = np.abs(centers - seam) <= 2 * STEP
        if not np.any(near):
            raise AssertionError("seam lacks a candidate")
        result[i] = float(np.max(scores[near]))
    return result


def evaluate(item: dict, mean: np.ndarray, sd: np.ndarray,
             weights: np.ndarray, threshold: float) -> dict:
    scores = ((item["features"] - mean) / sd) @ weights
    centers = item["centers"]
    decoy_scores = seam_scores(scores, centers, item["decoy"])
    trial_scores = seam_scores(scores, centers, item["trial"])
    rng = np.random.default_rng(np.random.SeedSequence([
        SEED, int(item["subject"]), int(item["session"]), 4]))
    eligible = np.flatnonzero(item["quiet_rest"])
    eligible = eligible[(eligible >= 2) & (eligible < len(scores) - 2)]
    chosen = rng.choice(eligible, size=len(item["decoy"]), replace=False)
    # Identical five-candidate search for seams and no-seam anchors; comparing
    # seam maxima with single negative samples would inflate both AUCs.
    neg_match = np.asarray([np.max(scores[i - 2:i + 3]) for i in chosen])
    detections = centers[find_peaks(scores, height=threshold, distance=10)[0]]
    # Score at t is only available at t+0.2s. Report center and availability
    # separately; the composed attack must use availability, not t.
    sensitivity = lambda seams: float(np.mean(
        nearest_distance(seams, detections) <= 2 * STEP)) if len(detections) else 0.0
    rest_det = detections[(item["state_at"](detections) == 0) &
                          (nearest_distance(detections, item["all_seams"]) >= 5 * STEP)]
    return {"subject": item["subject"], "session": item["session"],
            "auc_decoy_vs_nonsplice_rest": auc(decoy_scores, neg_match),
            "auc_trial_vs_nonsplice_rest": auc(trial_scores, neg_match),
            "auc_trial_vs_decoy": auc(trial_scores, decoy_scores),
            "decoy_sensitivity_within_0p2s_center": sensitivity(item["decoy"]),
            "trial_sensitivity_within_0p2s_center": sensitivity(item["trial"]),
            "false_detections_per_min_replay_rest": float(
                60 * len(rest_det) / (30 * 16.8)),
            "threshold_source_only": threshold,
            "detection_samples_available_at": (detections + 50).astype(np.int64).tolist(),
            "n_detections": len(detections)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("data/interim/lee2019_v2_random_precue_balanced_20261008"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/simulation_redesign_20261008/e3_g1b_splice_detector.json"))
    args = parser.parse_args()
    paths = sorted(args.root.glob("session*/sub-*.npz"))
    if len(paths) != 108:
        raise AssertionError(f"expected 108, got {len(paths)}")
    data = {}
    for path in paths:
        subject = int(path.stem.split("-")[1])
        session = int(path.parent.name.replace("session", ""))
        item = load_features(path)
        item["subject"], item["session"] = subject, session
        with np.load(path, allow_pickle=False) as saved:
            states = saved["state"]
        item["state_at"] = lambda points, states=states: states[points]
        data[(subject, session)] = item
    rows = []
    for subject in range(1, 55):
        donors = [item for (who, _), item in data.items() if who != subject]
        mean, sd, weights = fit_source(donors)
        donor_scores = np.concatenate([
            (((item["features"][item["quiet_rest"]] - mean) / sd) @ weights)
            for item in donors])
        threshold = float(np.quantile(donor_scores, .995))
        for session in (1, 2):
            rows.append(evaluate(data[(subject, session)], mean, sd, weights,
                                 threshold))
    summary = {}
    for session in (1, 2):
        group = [row for row in rows if row["session"] == session]
        summary[str(session)] = {key: float(np.median([row[key] for row in group]))
                                 for key in ("auc_decoy_vs_nonsplice_rest",
                                             "auc_trial_vs_nonsplice_rest",
                                             "auc_trial_vs_decoy",
                                             "decoy_sensitivity_within_0p2s_center",
                                             "trial_sensitivity_within_0p2s_center",
                                             "false_detections_per_min_replay_rest")}
    result = {"scope": "E3 G1b source-subject-only rest-rest splice detector",
              "n_subject_sessions": len(rows),
              "classifier": "4-feature shrinkage LDA, decoy rest-rest vs quiet rest",
              "source_threshold": "99.5th percentile of source quiet-rest candidate scores",
              "candidate_step_s": .1, "feature_post_context_s": .2,
              "false_rate_denominator": "504 s replay rest, only 168 s unique EEG per session",
              "g1c_composed_attack_tested": False,
              "session_summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
