"""LOSO deployment audit with target-rest and source-only tangent references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.riemannian import tangent_features
from uncued_mi.operating_point import tune_source_operating_points
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, reference, rolling_covariance,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma, ewma_z, interval_metrics, intervals_from_alarm, ma_trace,
)

SUBJECTS = tuple(range(1, 55))
SCREEN = 10
PER_CLASS_SOURCE = 200
TARGETS = (2., 4.)
DELTAS = (1., 2., 3., 4.)
OFFSETS = np.linspace(-5, 5, 61)


def evenly(values, n=PER_CLASS_SOURCE):
    values = np.asarray(values)
    if len(values) <= n:
        return values
    return values[np.linspace(0, len(values) - 1, n).round().astype(int)]


def load_covariances(session):
    domains = {}
    for subject in SUBJECTS:
        path = Path(f"data/interim/lee2019_four_rest/session{session}/sub-{subject:02d}.npz")
        with np.load(path, allow_pickle=False) as data:
            ct, it = data["train_command_trial"], data["train_idle_iti_trial"]
            domains[subject] = {
                "command": evenly(data["train_command_covariance"][ct < 50]),
                "idle": evenly(data["train_idle_iti_covariance"][it < 50]),
                "pre": np.asarray(data["pre_train_covariance"]),
            }
    return domains


def fit_model(x_command, x_idle):
    x = np.r_[x_command, x_idle]
    y = np.r_[np.ones(len(x_command)), np.zeros(len(x_idle))]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1., class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260812,
    ).fit(scaler.transform(x), y)
    return scaler, model


def source_reference(domains, target):
    matrices = [np.mean(domains[s]["pre"], axis=0) for s in SUBJECTS if s != target]
    return reference(np.asarray(matrices))


def source_training(domains, target, arm, src_ref, own_cache):
    command, idle = [], []
    for source in SUBJECTS:
        if source == target:
            continue
        if arm == "A_target_rest":
            command.append(own_cache[source]["command"])
            idle.append(own_cache[source]["idle"])
        else:
            command.append(tangent_features(domains[source]["command"], src_ref))
            idle.append(tangent_features(domains[source]["idle"], src_ref))
    return np.concatenate(command), np.concatenate(idle)


def tune_anchor(anchor):
    z = (anchor - anchor.mean()) / max(anchor.std(ddof=1), 1e-8)
    points = tune_source_operating_points(
        [calibration(z)], "ma_dwell", list(TARGETS), STEP, ma_seconds=.5,
        dwell_seconds=.2, refractory_seconds=1., grid_size=160,
    )
    return {target: float(points[target].threshold) for target in TARGETS}


def prepare_target(subject, session):
    raw_path = Path(f"data/interim/lee2019_four_rest_raw/session{session}/sub-{subject:02d}.npz")
    stream_path = Path(f"data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz")
    with np.load(raw_path, allow_pickle=False) as data:
        anchor_cov, _ = rolling_covariance(data["pre_train"])
    with np.load(stream_path, allow_pickle=False) as data:
        signal, events = data["signal"], data["events"]
    stream_cov, ends = rolling_covariance(signal)
    times = ends / SFREQ
    starts = (ends - int(WINDOW * SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), bool)
    for seam in seams:
        guard |= (starts < seam + 3.) & (times > seam)
    return anchor_cov, stream_cov, events, times, starts, ~guard


def threshold_metrics(ma, threshold, allowed, evalmask, idle, exposure, gt, times, onsets):
    alarms = alarms_from_ma(ma, threshold, allowed) & evalmask
    fp = float(np.sum(alarms & idle) / (exposure / 60))
    sensitivities = [float(np.mean([
        np.any(alarms & (times >= onset) & (times <= onset + delta)) for onset in onsets
    ])) for delta in DELTAS]
    pred = intervals_from_alarm(times, alarms, ma, threshold, allowed & evalmask)
    tolerant = interval_metrics(pred, gt, .1, .1); strict = interval_metrics(pred, gt, .7, .7)
    return [fp, *sensitivities, tolerant["coverage"], tolerant["temporal_iou"], strict["intersection_f1"]]


def binary_auc(positive, negative):
    values = np.r_[positive, negative]
    ranks = rankdata(values)
    n_positive = len(positive)
    return float(
        (ranks[:n_positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * len(negative))
    )


def evaluate_score(raw, ma, thresholds, allowed, events, times, starts):
    base_threshold = thresholds[2.]
    cues = events[:, 5] / SFREQ
    screen_flat = []; screen_margin = []
    for cue in cues[:SCREEN]:
        valid = (times >= cue + 1.5) & (times < cue + 4.) & allowed
        screen_flat.append(not np.any(ma[valid] >= base_threshold))
        finite = ma[valid & np.isfinite(ma)]
        screen_margin.append(float(np.max(finite) - base_threshold) if len(finite) else -np.inf)
    onsets = cues[SCREEN:] + .5
    stops = cues[SCREEN:] + 4.
    evalmask = times >= events[SCREEN, 2] / SFREQ
    state = np.zeros(len(times), bool)
    for onset, stop in zip(onsets, stops):
        state |= (times >= onset) & (times <= stop)
    idle = evalmask & allowed & ~state
    exposure = idle.sum() * STEP
    gt = np.c_[onsets, stops]
    positive_scores = np.concatenate([
        raw[(starts >= onset) & (times <= stop) & allowed & evalmask]
        for onset, stop in gt
    ])
    window_auc = binary_auc(positive_scores, raw[idle])
    points = []
    for offset in OFFSETS:
        threshold = base_threshold + offset
        points.append(threshold_metrics(ma,threshold,allowed,evalmask,idle,exposure,gt,times,onsets))
    points = np.asarray(points)
    order = np.argsort(points[:, 0]); x = points[order, 0]
    surface = {}
    for target in TARGETS:
        row = {"bracketed": bool(x[0] <= target <= x[-1])}
        for index, delta in enumerate(DELTAS, 1):
            y = np.maximum.accumulate(points[order, index])
            row[str(delta)] = float(np.interp(target, x, y)) if row["bracketed"] else None
        for index, metric in enumerate(("coverage", "temporal_iou", "strict_f1"), 1 + len(DELTAS)):
            row[metric] = float(np.interp(target, x, points[order, index])) if row["bracketed"] else None
        admissible = np.flatnonzero(points[:, 0] <= target)
        if len(admissible):
            # A real deployment must choose one available threshold. Select the
            # admissible threshold with maximum 2-s sensitivity; ties prefer
            # fewer false alarms.
            best_sen = np.max(points[admissible, 2])
            tied = admissible[np.isclose(points[admissible, 2], best_sen)]
            best = int(tied[np.argmin(points[tied, 0])])
            row["deployable"] = {
                "actual_fp": float(points[best, 0]),
                **{str(delta): float(points[best, index]) for index, delta in enumerate(DELTAS, 1)},
                "coverage": float(points[best, 1 + len(DELTAS)]),
                "temporal_iou": float(points[best, 2 + len(DELTAS)]),
                "strict_f1": float(points[best, 3 + len(DELTAS)]),
            }
        else:
            row["deployable"] = None
        fixed = threshold_metrics(ma,thresholds[target],allowed,evalmask,idle,exposure,gt,times,onsets)
        row["fixed_threshold"] = {"actual_fp":float(fixed[0]),**{str(delta):float(fixed[index]) for index,delta in enumerate(DELTAS,1)},"coverage":float(fixed[1+len(DELTAS)]),"temporal_iou":float(fixed[2+len(DELTAS)]),"strict_f1":float(fixed[3+len(DELTAS)])}
        surface[str(target)] = row
    return float(np.mean(screen_flat)), float(np.mean(screen_margin)), window_auc, surface


def analyze_target(subject, session, domains, own_cache):
    anchor_cov, stream_cov, events, times, starts, allowed = prepare_target(subject, session)
    rows = {}
    for arm in ("A_target_rest", "B_source_only"):
        src_ref = source_reference(domains, subject)
        x_command, x_idle = source_training(domains, subject, arm, src_ref, own_cache)
        scaler, model = fit_model(x_command, x_idle)
        target_ref = reference(domains[subject]["pre"]) if arm == "A_target_rest" else src_ref
        anchor_features = tangent_features(anchor_cov, target_ref)
        stream_features = tangent_features(stream_cov, target_ref)
        anchor = model.decision_function(scaler.transform(anchor_features))
        raw = model.decision_function(scaler.transform(stream_features))
        if arm == "A_target_rest":
            initial_mean, initial_var = float(anchor.mean()), float(anchor.var(ddof=1))
            thresholds = tune_anchor(anchor)
        else:
            source_anchor = []
            for source in SUBJECTS:
                if source != subject:
                    features = tangent_features(evenly(domains[source]["pre"], 60), src_ref)
                    source_anchor.append(model.decision_function(scaler.transform(features)))
            source_anchor = np.concatenate(source_anchor)
            initial_mean, initial_var = float(source_anchor.mean()), float(source_anchor.var(ddof=1))
            thresholds = tune_anchor(source_anchor)
        z = ewma_z(raw, initial_mean, initial_var, allowed)
        ma = ma_trace(z)
        flat, margin, window_auc, surface = evaluate_score(
            raw, ma, thresholds, allowed, events, times, starts
        )
        rows[arm] = {
            "screen_flat_fraction": flat,
            "screen_margin_mean": margin,
            "window_auc": window_auc,
            "source_thresholds": {str(k): v for k, v in thresholds.items()},
            "surface": surface,
        }
    return {"subject": subject, "session": session, "arms": rows}


def summarize(rows):
    summary = {}
    for session in (1, 2):
        session_rows = [row for row in rows if row["session"] == session]
        summary[str(session)] = {}
        for arm in ("A_target_rest", "B_source_only"):
            ranked = sorted(session_rows, key=lambda row: (row["arms"][arm]["screen_flat_fraction"], -row["arms"][arm]["screen_margin_mean"]))
            selected = ranked[:int(np.ceil(.2 * len(ranked)))]
            cutoff = float(selected[-1]["arms"][arm]["screen_flat_fraction"])
            arm_summary = {"cutoff_flat_fraction": cutoff, "tie_break": "higher mean peak-minus-threshold margin", "selected_n": len(selected), "selected_subjects": [row["subject"] for row in selected], "window_auc_median": float(np.nanmedian([row["arms"][arm]["window_auc"] for row in selected])), "surface": {}}
            for fp in TARGETS:
                values = {}
                for key in (*[str(delta) for delta in DELTAS], "coverage", "temporal_iou", "strict_f1"):
                    v = [row["arms"][arm]["surface"][str(fp)][key] for row in selected]
                    values[key] = float(np.nanmedian(v))
                arm_summary["surface"][str(fp)] = values
                deployable = [row["arms"][arm]["surface"][str(fp)]["deployable"] for row in selected]
                arm_summary["surface"][str(fp)]["deployable"] = {
                    key: float(np.nanmedian([item[key] for item in deployable if item is not None]))
                    for key in ("actual_fp", *[str(delta) for delta in DELTAS], "coverage", "temporal_iou", "strict_f1")
                }
                fixed = [row["arms"][arm]["surface"][str(fp)]["fixed_threshold"] for row in selected]
                arm_summary["surface"][str(fp)]["fixed_threshold"] = {key:float(np.nanmedian([item[key] for item in fixed])) for key in ("actual_fp",*[str(delta) for delta in DELTAS],"coverage","temporal_iou","strict_f1")}
            summary[str(session)][arm] = arm_summary
    combined = {}
    for arm in ("A_target_rest", "B_source_only"):
        selected = []
        for session in (1, 2):
            ids = set(summary[str(session)][arm]["selected_subjects"])
            selected.extend([row for row in rows if row["session"] == session and row["subject"] in ids])
        combined[arm] = {"n": len(selected), "window_auc_median": float(np.nanmedian([row["arms"][arm]["window_auc"] for row in selected])), "surface": {}}
        for fp in TARGETS:
            combined[arm]["surface"][str(fp)] = {
                key: float(np.nanmedian([row["arms"][arm]["surface"][str(fp)][key] for row in selected]))
                for key in (*[str(delta) for delta in DELTAS], "coverage", "temporal_iou", "strict_f1")
            }
            deployable = [row["arms"][arm]["surface"][str(fp)]["deployable"] for row in selected]
            combined[arm]["surface"][str(fp)]["deployable"] = {
                key: float(np.nanmedian([item[key] for item in deployable if item is not None]))
                for key in ("actual_fp", *[str(delta) for delta in DELTAS], "coverage", "temporal_iou", "strict_f1")
            }
            fixed = [row["arms"][arm]["surface"][str(fp)]["fixed_threshold"] for row in selected]
            combined[arm]["surface"][str(fp)]["fixed_threshold"] = {key:float(np.nanmedian([item[key] for item in fixed])) for key in ("actual_fp",*[str(delta) for delta in DELTAS],"coverage","temporal_iou","strict_f1")}
    summary["combined"] = combined
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_loso_deployment.json"))
    args = parser.parse_args()
    targets = args.subjects or list(SUBJECTS)
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
        for subject in targets:
            rows.append(analyze_target(subject, session, domains, own_cache))
            print(f"session{session} sub-{subject:02d}", flush=True)
    payload = {
        "protocol": {
            "LOSO": "same-session source subjects; target excluded from LR training",
            "source_sampling": f"{PER_CLASS_SOURCE} evenly spaced windows per class per source",
            "A": "per-domain target/source pre-rest tangent references; target pre-rest threshold/EWMA initialization",
            "B": "target-excluded source pre-rest common reference, threshold, and EWMA initialization",
            "screening": "recomputed per arm/session; relative lowest-flat 20%",
            "claim_limit": "Lee2019 cue-relative pseudo-onset",
        },
        "summary": summarize(rows) if len(targets) == len(SUBJECTS) else None,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
