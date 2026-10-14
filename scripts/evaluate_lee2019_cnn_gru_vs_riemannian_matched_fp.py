"""Diagnostic matched-achieved-FP curves for LOSO CNN-GRU vs Riemannian.

Thresholds are selected using the evaluation stream's true idle intervals, so
this is an oracle diagnostic of score/event information, not deployment
calibration. For each subject/session/target FP, both systems are compared at
the same achievable idle false-alarm count.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import evaluate_lee2019_loso_deployment as base
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma,
    ewma_z,
    interval_metrics,
    intervals_from_alarm,
    ma_trace,
)
from uncued_mi.riemannian import tangent_features


TARGET_FPS = (1., 2., 3., 4., 5.)
DELTAS = (1., 2.)
MA_SECONDS = .5
DWELL_SECONDS = .3
REFRACTORY_SECONDS = 1.
SCREEN = base.SCREEN
STEP = base.STEP
SFREQ = base.SFREQ


def onset_sensitivity(times: np.ndarray, alarms: np.ndarray,
                      onsets: np.ndarray, delta: float) -> float:
    alarm_times = times[np.asarray(alarms, bool)]
    if not len(alarm_times):
        return 0.
    first = np.searchsorted(alarm_times, onsets, side="left")
    hit = first < len(alarm_times)
    indices = np.minimum(first, len(alarm_times) - 1)
    hit &= alarm_times[indices] <= onsets + delta
    return float(np.mean(hit))


def threshold_curve(ma: np.ndarray, times: np.ndarray, allowed: np.ndarray,
                    evalmask: np.ndarray, idle: np.ndarray, exposure: float,
                    onsets: np.ndarray, n_thresholds: int) -> dict[int, dict]:
    valid = np.asarray(ma)[allowed & evalmask & np.isfinite(ma)]
    if not len(valid):
        raise ValueError("No valid evaluation scores for threshold sweep")
    quantiles = np.quantile(valid, np.linspace(0., 1., n_thresholds))
    thresholds = np.unique(np.r_[valid.min() - 1e-8, quantiles, valid.max() + 1e-8])
    candidates: list[dict] = []
    for threshold in thresholds:
        alarms = alarms_from_ma(
            ma, float(threshold), allowed,
            dwell=int(round(DWELL_SECONDS / STEP)),
            refractory=int(round(REFRACTORY_SECONDS / STEP)),
        ) & evalmask
        false_count = int(np.sum(alarms & idle))
        row = {
            "threshold": float(threshold),
            "false_alarm_count": false_count,
            "actual_fp_per_min": float(false_count / (exposure / 60.)),
            "sensitivity_1.0": onset_sensitivity(times, alarms, onsets, 1.),
            "sensitivity_2.0": onset_sensitivity(times, alarms, onsets, 2.),
        }
        candidates.append(row)

    # Follow the operating curve from the conservative end (high threshold)
    # and stop at the first maximum false-alarm count.  Beyond that point,
    # lowering the threshold can *reduce* the number of emitted alarms because
    # the dwell counter emits only once during a continuous high run and is
    # reset by guarded islands.  That descending branch is a detector/reset
    # artifact, not an upper-tail operating point.
    candidates.sort(key=lambda item: item["threshold"], reverse=True)
    peak_index = int(np.argmax([item["false_alarm_count"] for item in candidates]))
    candidates = candidates[:peak_index + 1]

    by_false_count: dict[int, dict] = {}
    for row in candidates:
        false_count = int(row["false_alarm_count"])
        prior = by_false_count.get(false_count)
        # The dwell detector emits only once during an uninterrupted
        # above-threshold run.  At extremely low thresholds this creates a
        # non-monotone, guard-reset branch: one alarm can appear at the start
        # of every allowed island.  It is not a valid high-tail operating
        # point and can look spuriously good because islands are trial-linked.
        # For an identical idle false-alarm count, retain the *highest*
        # (most conservative) threshold, i.e. the normal upper-tail branch.
        # Command labels are deliberately not used to break the tie.
        if prior is None or row["threshold"] > prior["threshold"]:
            by_false_count[false_count] = row
    return by_false_count


def circular_shift_null(times: np.ndarray, alarms: np.ndarray,
                        onsets: np.ndarray, offsets: np.ndarray) -> dict[str, float]:
    alarm_indices = np.flatnonzero(alarms)
    samples: dict[str, list[float]] = {str(delta): [] for delta in DELTAS}
    n = len(times)
    for offset in offsets:
        shifted_times = np.sort(times[(alarm_indices + int(offset)) % n])
        for delta in DELTAS:
            if not len(shifted_times):
                samples[str(delta)].append(0.)
                continue
            indices = np.searchsorted(shifted_times, onsets, side="left")
            valid = indices < len(shifted_times)
            safe_indices = np.minimum(indices, len(shifted_times) - 1)
            hit = valid & (shifted_times[safe_indices] <= onsets + delta)
            samples[str(delta)].append(float(np.mean(hit)))
    return {key: float(np.median(values)) for key, values in samples.items()}


def selected_metrics(ma: np.ndarray, threshold: float, times: np.ndarray,
                     allowed: np.ndarray, evalmask: np.ndarray,
                     idle: np.ndarray, exposure: float, onsets: np.ndarray,
                     gt: np.ndarray, offsets: np.ndarray) -> dict:
    alarms = alarms_from_ma(
        ma, threshold, allowed,
        dwell=int(round(DWELL_SECONDS / STEP)),
        refractory=int(round(REFRACTORY_SECONDS / STEP)),
    ) & evalmask
    observed = {
        str(delta): onset_sensitivity(times, alarms, onsets, delta)
        for delta in DELTAS
    }
    null = circular_shift_null(times, alarms, onsets, offsets)
    pred = intervals_from_alarm(
        times, alarms, ma, threshold, allowed & evalmask,
    )
    tolerant = interval_metrics(pred, gt, .1, .1)
    strict = interval_metrics(pred, gt, .7, .7)
    false_count = int(np.sum(alarms & idle))
    return {
        "threshold": float(threshold),
        "actual_fp_per_min": float(false_count / (exposure / 60.)),
        "false_alarm_count": false_count,
        "onset_sensitivity": observed,
        "circular_shift_null_median": null,
        "circular_shift_excess": {
            key: float(observed[key] - null[key]) for key in observed
        },
        "coverage": tolerant["coverage"],
        "temporal_iou": tolerant["temporal_iou"],
        "strict_f1": strict["intersection_f1"],
        "predicted_intervals": int(len(pred)),
    }


def fit_riemannian_scores(subject: int, domains: dict, own_cache: dict,
                          anchor_cov: np.ndarray,
                          stream_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_ref = base.source_reference(domains, subject)
    x_command, x_idle = base.source_training(
        domains, subject, "A_target_rest", src_ref, own_cache,
    )
    x = np.r_[x_command, x_idle]
    y = np.r_[np.ones(len(x_command)), np.zeros(len(x_idle))]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1., class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260812,
    ).fit(scaler.transform(x), y)
    target_ref = base.reference(domains[subject]["pre"])
    anchor = model.decision_function(scaler.transform(
        tangent_features(anchor_cov, target_ref)))
    raw = model.decision_function(scaler.transform(
        tangent_features(stream_cov, target_ref)))
    return raw, anchor


def bootstrap_delta(values: np.ndarray, seed: int,
                    repeats: int) -> dict:
    """Subject-cluster bootstrap; values shape [subject, session]."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    n_subjects = values.shape[0]
    draws = np.empty(repeats)
    for i in range(repeats):
        chosen = rng.integers(0, n_subjects, size=n_subjects)
        draws[i] = np.nanmedian(values[chosen])
    return {
        "median_delta": float(np.nanmedian(values)),
        "cluster_bootstrap_95ci": np.nanquantile(draws, [.025, .975]).tolist(),
        "n_subjects": int(n_subjects),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-root", type=Path, default=Path(
        "data/interim/lee2019_cnn_gru_loso_arm_a_stream_v2_score_cache"))
    parser.add_argument("--thresholds", type=int, default=301)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--allow-subset-for-smoke", action="store_true",
                        help="Permit partial subjects/sessions for technical validation only")
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/lee2019_cnn_gru_vs_riemannian_matched_fp.json"))
    args = parser.parse_args()
    if (len(args.subjects) != 54 or set(args.subjects) != set(range(1, 55))) and not args.allow_subset_for_smoke:
        raise ValueError("Primary comparison requires the full 54-subject cohort")

    rows: list[dict] = []
    curve_rows: list[dict] = []
    for session in args.sessions:
        domains = base.load_covariances(session)
        own_cache = {}
        for subject in base.SUBJECTS:
            own_ref = base.reference(domains[subject]["pre"])
            own_cache[subject] = {
                "command": tangent_features(domains[subject]["command"], own_ref),
                "idle": tangent_features(domains[subject]["idle"], own_ref),
            }
        for index, subject in enumerate(args.subjects, start=1):
            cache_path = args.cnn_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing exact-stream CNN score cache: {cache_path}")
            with np.load(cache_path, allow_pickle=False) as cache:
                cache_subject = int(cache["subject"])
                cache_session = int(cache["session"])
                cnn_raw = np.asarray(cache["raw_scores"], float)
                cnn_anchor = np.asarray(cache["anchor_scores"], float)
                times = np.asarray(cache["times"], float)
                starts = np.asarray(cache["starts"], float)
                allowed = np.asarray(cache["allowed"], bool)
                events = np.asarray(cache["events"], float)
            if (cache_subject, cache_session) != (subject, session):
                raise ValueError(f"Cache identity mismatch: {cache_path}")
            # Event and timing integrity must still hold against the locked baseline.
            anchor_cov, stream_cov, locked_events, locked_times, locked_starts, locked_allowed = base.prepare_target(subject, session)
            locked_events = np.load(
                Path(f"data/interim/lee2019_pseudo_uncued_raw/session{session}/sub-{subject:02d}.npz"),
                allow_pickle=False,
            )["events"]
            if not (np.array_equal(events, locked_events)
                    and np.array_equal(times, locked_times)
                    and np.array_equal(starts, locked_starts)
                    and np.array_equal(allowed, locked_allowed)):
                raise AssertionError(f"CNN score cache no longer matches locked stream: {cache_path}")

            evalmask = times >= events[SCREEN, 2] / SFREQ
            cues = events[:, 5] / SFREQ
            onsets = cues[SCREEN:] + .5
            gt = np.c_[onsets, cues[SCREEN:] + 4.]
            state = np.zeros(len(times), bool)
            for start, stop in gt:
                state |= (times >= start) & (times <= stop)
            idle = evalmask & allowed & ~state
            exposure = float(idle.sum() * STEP)

            ria_raw, ria_anchor = fit_riemannian_scores(
                subject, domains, own_cache, anchor_cov, stream_cov,
            )
            cnn_z = ewma_z(cnn_raw, float(cnn_anchor.mean()),
                           float(cnn_anchor.var(ddof=1)), allowed)
            ria_z = ewma_z(ria_raw, float(ria_anchor.mean()),
                           float(ria_anchor.var(ddof=1)), allowed)
            cnn_ma = ma_trace(cnn_z, n=int(round(MA_SECONDS / STEP)))
            ria_ma = ma_trace(ria_z, n=int(round(MA_SECONDS / STEP)))
            cnn_curve = threshold_curve(
                cnn_ma, times, allowed, evalmask, idle, exposure, onsets, args.thresholds,
            )
            ria_curve = threshold_curve(
                ria_ma, times, allowed, evalmask, idle, exposure, onsets, args.thresholds,
            )
            shared_counts = sorted(set(cnn_curve) & set(ria_curve))
            if not shared_counts:
                raise RuntimeError(f"No common achieved idle FP count for S{session} sub-{subject:02d}")

            for model_name, curve in (("cnn_gru", cnn_curve), ("riemannian", ria_curve)):
                for point in curve.values():
                    curve_rows.append({
                        "subject": subject, "session": session, "model": model_name,
                        **point,
                    })
            for target_fp in TARGET_FPS:
                desired_count = target_fp * exposure / 60.
                matched_count = min(shared_counts, key=lambda value: (abs(value - desired_count), value))
                rng = np.random.default_rng(
                    72000 + 1000 * session + 10 * subject + int(target_fp)
                )
                offsets = rng.integers(1, max(2, len(times)), size=args.shifts)
                selected = {}
                for model_name, curve, ma in (
                    ("cnn_gru", cnn_curve, cnn_ma),
                    ("riemannian", ria_curve, ria_ma),
                ):
                    point = curve[matched_count]
                    metric = selected_metrics(
                        ma, point["threshold"], times, allowed, evalmask,
                        idle, exposure, onsets, gt, offsets,
                    )
                    selected[model_name] = {
                        "threshold": point["threshold"],
                        "matched_false_alarm_count": matched_count,
                        "matched_actual_fp_per_min": matched_count / (exposure / 60.),
                        **metric,
                    }
                rows.append({
                    "subject": subject,
                    "session": session,
                    "target_fp_per_min": target_fp,
                    "idle_exposure_seconds": exposure,
                    "target_false_alarm_count": float(desired_count),
                    "matched_false_alarm_count": matched_count,
                    "matched_actual_fp_per_min": float(matched_count / (exposure / 60.)),
                    "cnn_gru": selected["cnn_gru"],
                    "riemannian": selected["riemannian"],
                })
            print(f"matched-FP S{session} sub-{subject:02d} [{index}/54] "
                  f"common_counts={len(shared_counts)}", flush=True)

    # Subject-level paired deltas, stratified by session and pooled with the
    # two sessions resampled together.
    summary: dict[str, dict] = {}
    for target_fp in TARGET_FPS:
        group = [r for r in rows if r["target_fp_per_min"] == target_fp]
        subject_ids = sorted({int(r["subject"]) for r in group})
        metric_names = (
            "sensitivity_1.0", "sensitivity_2.0", "excess_1.0", "excess_2.0",
            "coverage", "temporal_iou", "strict_f1",
        )
        summary[str(target_fp)] = {"session_medians": {}, "paired_delta_cnn_minus_riemannian": {}}
        def summarize_value(item: dict, metric: str) -> float:
            if metric == "actual_fp_per_min":
                return float(item["actual_fp_per_min"])
            if metric.startswith("sensitivity_"):
                return float(item["onset_sensitivity"][metric.removeprefix("sensitivity_")])
            if metric.startswith("excess_"):
                return float(item["circular_shift_excess"][metric.removeprefix("excess_")])
            return float(item[metric])

        for session in args.sessions:
            session_rows = [r for r in group if r["session"] == session]
            summary[str(target_fp)]["session_medians"][str(session)] = {
                "matched_actual_fp_per_min": float(np.median([r["matched_actual_fp_per_min"] for r in session_rows])),
                "cnn_gru": {
                    key: float(np.median([summarize_value(r["cnn_gru"], key) for r in session_rows]))
                    for key in ("actual_fp_per_min", "sensitivity_1.0", "sensitivity_2.0", "excess_1.0", "excess_2.0", "coverage", "temporal_iou", "strict_f1")
                },
                "riemannian": {
                    key: float(np.median([summarize_value(r["riemannian"], key) for r in session_rows]))
                    for key in ("actual_fp_per_min", "sensitivity_1.0", "sensitivity_2.0", "excess_1.0", "excess_2.0", "coverage", "temporal_iou", "strict_f1")
                },
            }
        summary[str(target_fp)]["paired_delta_cnn_minus_riemannian"] = {}
        for metric in metric_names:
            if metric.startswith("sensitivity_"):
                value_key = "onset_sensitivity"
                subkey = metric.removeprefix("sensitivity_")
            elif metric.startswith("excess_"):
                value_key = "circular_shift_excess"
                subkey = metric.removeprefix("excess_")
            else:
                value_key, subkey = metric, None

            def get_value(row: dict, model: str) -> float:
                value = row[model][value_key]
                return float(value[subkey] if subkey is not None else value)

            delta = np.asarray([
                [
                    get_value(next(r for r in group if r["subject"] == subject and r["session"] == session), "cnn_gru")
                    - get_value(next(r for r in group if r["subject"] == subject and r["session"] == session), "riemannian")
                    for session in args.sessions
                ]
                for subject in subject_ids
            ], float)
            summary[str(target_fp)]["paired_delta_cnn_minus_riemannian"][metric] = {
                "all_sessions": bootstrap_delta(delta, 202609140 + int(target_fp * 10) + len(metric), args.bootstrap),
                **{
                    f"session{session}": bootstrap_delta(
                        delta[:, offset],
                        203000000 + int(target_fp * 10) + len(metric) + session,
                        args.bootstrap,
                    )
                    for offset, session in enumerate(args.sessions)
                },
            }

    payload = {
        "protocol": {
            "models": "source-pooled CNN-GRU LOSO vs Riemannian Arm A source-pooled LR; target pre-rest initializes normalization/EWMA",
            "stream": "same locked seam-guarded stream; CNN score cache event/timing/guard arrays asserted exactly against Riemannian cache",
            "decision": "EWMA H30 + MA 0.5 s + dwell 0.3 s + refractory 1 s",
            "threshold_sweep": f"{args.thresholds} evaluation-score quantiles per model; exact common idle false-alarm count selected nearest target FP={TARGET_FPS}; same count per paired model",
            "primary_targets": "FP=2/min primary; FP=4/min secondary; FP=1/3/5/min additional curve points",
            "threshold_selection_caveat": "Uses evaluation idle labels to match false-alarm counts; at an identical count the highest threshold is retained to exclude the non-monotone low-threshold guard-reset branch. Diagnostic oracle only, not a deployable calibration method",
            "null": f"{args.shifts} same circular-shift offsets per subject/session/target-FP for both models",
            "bootstrap": f"{args.bootstrap} subject-cluster draws; both sessions resampled together for pooled estimate",
            "claim_limit": "Lee2019 cue-relative pseudo-onset, not actual self-initiated performance",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    curve_path = args.output.with_name(args.output.stem + "_threshold_curves.csv")
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(f"wrote {args.output} and {curve_path}", flush=True)


if __name__ == "__main__":
    main()
