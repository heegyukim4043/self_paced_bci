"""A0/A1 preparation: exact-LOSO Arm A score with guard-rearmed decision.

This produces descriptive raw alarm counts only. It does not compute an excess,
null p-value, or confirmed model comparison. Target pre-rest is the sole
target-side calibration input; source training uses trial IDs 0--49 only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from uncued_mi.guard_rearm_calibration import tune_guard_rearmed_anchor
from uncued_mi.riemannian import tangent_features
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma, ewma_z, ma_trace,
)
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, reference, rolling_covariance,
)

SUBJECTS = tuple(range(1, 55))
PER_SOURCE_CLASS = 200


def evenly(values: np.ndarray, n: int = PER_SOURCE_CLASS) -> np.ndarray:
    if len(values) <= n:
        return values
    indices = np.linspace(0, len(values) - 1, n).round().astype(int)
    return values[indices]


def source_domains(session: int, cov_root: Path) -> dict[int, dict]:
    domains = {}
    for subject in SUBJECTS:
        path = cov_root / f"session{session}" / f"sub-{subject:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            command_trials = data["train_command_trial"]
            idle_trials = data["train_idle_iti_trial"]
            if np.any(command_trials < 0) or np.any(idle_trials < 0):
                raise AssertionError(f"negative source trial ID: {path}")
            domains[subject] = {
                "command": evenly(data["train_command_covariance"][command_trials < 50]),
                "idle": evenly(data["train_idle_iti_covariance"][idle_trials < 50]),
                "pre": np.asarray(data["pre_train_covariance"]),
            }
    return domains


def fit_loso(domains: dict[int, dict], target: int, target_reference: np.ndarray):
    positive = np.concatenate([
        tangent_features(domains[subject]["command"], target_reference)
        for subject in SUBJECTS if subject != target
    ])
    negative = np.concatenate([
        tangent_features(domains[subject]["idle"], target_reference)
        for subject in SUBJECTS if subject != target
    ])
    x = np.concatenate([positive, negative])
    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1., class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260812,
    ).fit(scaler.transform(x), y)
    return scaler, model


def evaluate_target(subject: int, session: int, domains: dict[int, dict],
                    stream_root: Path, score_cache_root: Path | None) -> dict:
    path = stream_root / f"session{session}" / f"sub-{subject:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        signal = np.asarray(data["signal"])
        events = np.asarray(data["events"])
        trial_ids = np.asarray(data["trial_ids"])
    if not np.array_equal(trial_ids, np.arange(50, 100)):
        raise AssertionError(f"target trial order is not chronological: {path}")
    if len(events) != 50:
        raise AssertionError(f"expected 50 events: {path}")

    target_reference = reference(domains[subject]["pre"])
    scaler, model = fit_loso(domains, subject, target_reference)
    anchor_raw = model.decision_function(scaler.transform(
        tangent_features(domains[subject]["pre"], target_reference)
    ))
    stream_cov, ends = rolling_covariance(signal)
    raw = model.decision_function(scaler.transform(
        tangent_features(stream_cov, target_reference)
    ))
    times = ends / SFREQ
    starts = (ends - int(WINDOW * SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), dtype=bool)
    for seam in seams:
        guard |= (starts < seam + 3.) & (times > seam)
    allowed = ~guard

    initial_mean = float(anchor_raw.mean())
    initial_var = float(anchor_raw.var(ddof=1))
    anchor_z = ewma_z(
        anchor_raw, initial_mean, initial_var,
        np.ones(len(anchor_raw), dtype=bool),
    )
    stream_z = ewma_z(raw, initial_mean, initial_var, allowed)
    stream_ma = guard_local_moving_average(stream_z, allowed, ma_steps=5)
    legacy_ma = ma_trace(stream_z, n=5)
    if score_cache_root is not None:
        score_path = score_cache_root / f"session{session}" / f"sub-{subject:02d}.npz"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            subject=np.asarray(subject), session=np.asarray(session),
            raw_scores=raw.astype(np.float32),
            anchor_raw_scores=anchor_raw.astype(np.float32),
            times=times, starts=starts, allowed=allowed, events=events,
        )
    future = times >= events[20, 2] / SFREQ
    cues = events[20:, 5] / SFREQ
    onsets = cues + .5
    task = np.zeros(len(times), dtype=bool)
    for cue in cues:
        task |= (times >= cue + .5) & (times <= cue + 4.)
    idle = future & allowed & ~task
    exposure = float(idle.sum() * STEP)

    rows = {}
    for target in (2., 4.):
        point = tune_guard_rearmed_anchor(
            anchor_z, target, STEP, ma_steps=5,
            dwell_steps=3, refractory_steps=10,
            admissible_half_width=.5 if target == 2. else 1.,
        )
        alarms = guard_rearmed_alarms(
            stream_ma, point.threshold, allowed,
            dwell_steps=3, refractory_steps=10,
        ) & future
        legacy_same_threshold = alarms_from_ma(
            legacy_ma, point.threshold, allowed, dwell=3, refractory=10,
        ) & future
        edge_only_same_threshold = guard_rearmed_alarms(
            legacy_ma, point.threshold, allowed,
            dwell_steps=3, refractory_steps=10,
        ) & future
        hits = []
        delays = []
        high_at_first_valid_ma = []
        for cue, onset in zip(cues, onsets):
            candidates = np.flatnonzero(
                (times >= cue) & (times < cue + 2.)
                & allowed & np.isfinite(stream_ma)
            )
            high_at_first_valid_ma.append(
                bool(stream_ma[candidates[0]] >= point.threshold)
                if len(candidates) else None
            )
            alarm_times = times[alarms & (times >= onset) & (times <= onset + 2.)]
            hits.append(bool(len(alarm_times)))
            if len(alarm_times):
                delays.append(float(alarm_times[0] - onset))
        fp_count = int(np.sum(alarms & idle))
        rows[str(target)] = {
            "anchor": asdict(point),
            "future_fp_count": fp_count,
            "future_idle_seconds": exposure,
            "future_fp_per_min": fp_count * 60. / exposure if exposure else None,
            "future_sen_at_2s": float(np.mean(hits)),
            "future_onset_hits_2s": int(sum(hits)),
            "future_onset_count": len(onsets),
            "detected_median_delay_seconds": (
                float(np.median(delays)) if delays else None
            ),
            "high_at_first_valid_post_guard_ma": int(sum(
                value is True for value in high_at_first_valid_ma
            )),
            "first_valid_post_guard_ma_missing": int(sum(
                value is None for value in high_at_first_valid_ma
            )),
            "total_future_alarms": int(alarms.sum()),
            "legacy_same_threshold": {
                "future_fp_count": int(np.sum(legacy_same_threshold & idle)),
                "future_onset_hits_2s": int(sum(
                    np.any(legacy_same_threshold & (times >= onset)
                           & (times <= onset + 2.))
                    for onset in onsets
                )),
                "total_future_alarms": int(legacy_same_threshold.sum()),
            },
            "edge_only_same_threshold": {
                "future_fp_count": int(np.sum(edge_only_same_threshold & idle)),
                "future_onset_hits_2s": int(sum(
                    np.any(edge_only_same_threshold & (times >= onset)
                           & (times <= onset + 2.))
                    for onset in onsets
                )),
                "total_future_alarms": int(edge_only_same_threshold.sum()),
            },
        }
    return {"subject": subject, "session": session, "targets": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[3])
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--score-cache-root", type=Path, default=None)
    parser.add_argument(
        "--cov-root", type=Path,
        default=ROOT / "data/interim/lee2019_revalidation_norm_cov",
    )
    parser.add_argument(
        "--stream-root", type=Path,
        default=ROOT / "data/interim/lee2019_revalidation_chronological",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/processed/performance_revalidation_20261001/"
        "guard_rearm_riemannian_smoke.json",
    )
    args = parser.parse_args()
    rows = []
    for session in args.sessions:
        domains = source_domains(session, args.cov_root)
        for subject in args.subjects:
            print(f"A0/A1 preparation S{session} sub-{subject:02d}", flush=True)
            rows.append(evaluate_target(
                subject, session, domains, args.stream_root,
                args.score_cache_root,
            ))
            if not args.quiet:
                print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for session in args.sessions:
        selected = [row for row in rows if row["session"] == session]
        summary[str(session)] = {}
        for target in (2., 4.):
            metrics = [row["targets"][str(target)] for row in selected]
            summary[str(session)][str(target)] = {
                "n": len(metrics),
                "anchor_target_band_n": sum(
                    item["anchor"]["within_target_band"] for item in metrics
                ),
                "future_fp_median": float(np.median([
                    item["future_fp_per_min"] for item in metrics
                ])),
                "future_fp_range": [
                    float(min(item["future_fp_per_min"] for item in metrics)),
                    float(max(item["future_fp_per_min"] for item in metrics)),
                ],
                "future_sen2_median": float(np.median([
                    item["future_sen_at_2s"] for item in metrics
                ])),
                "future_sen2_pooled_descriptive": float(
                    sum(item["future_onset_hits_2s"] for item in metrics)
                    / sum(item["future_onset_count"] for item in metrics)
                ),
                "high_at_first_valid_ma_total": sum(
                    item["high_at_first_valid_post_guard_ma"] for item in metrics
                ),
                "legacy_same_threshold_sen2_pooled_descriptive": float(
                    sum(item["legacy_same_threshold"]["future_onset_hits_2s"]
                        for item in metrics)
                    / sum(item["future_onset_count"] for item in metrics)
                ),
            }
    payload = {
        "protocol": {
            "scope": "descriptive A0/A1 preparation, not validated EEG performance",
            "input": "normalized covariance source trial0-49; chronological target future trial70-99",
            "target_data": "target pre-train rest reference and anchor only; no target command labels",
            "decision": "causal EWMA H30 + guard-local MA5 + below-threshold re-arm + dwell3 + refractory10",
            "threshold": "target pre-rest idle-only exact count sweep; tie highest threshold",
            "primary_claim_limit": "cue-relative pseudo-onset in pseudo-uncued simulation",
            "null_status": "A1 rest-only sham-MI Type-I and CI coverage not run",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
