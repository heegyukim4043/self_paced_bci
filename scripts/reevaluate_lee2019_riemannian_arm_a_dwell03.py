"""Re-evaluate Riemannian LOSO Arm A with the CNN-GRU 0.3-s dwell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import evaluate_lee2019_loso_deployment as base
from scripts.evaluate_lee2019_locked_selfpaced_metrics import (
    alarms_from_ma, ewma_z, interval_metrics, intervals_from_alarm, ma_trace,
)
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points
from uncued_mi.riemannian import tangent_features

DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)
STEP = base.STEP
SFREQ = base.SFREQ
DWELL = 0.3
MA = 0.5
REFRACTORY = 1.0


def tune_anchor(anchor: np.ndarray) -> dict[float, tuple[float, float]]:
    z = (anchor - anchor.mean()) / max(anchor.std(ddof=1), 1e-8)
    sequence = ScoredSequence(
        np.arange(len(z)) * STEP, z, np.asarray([[0., len(z) * STEP]]),
        np.empty((0, 2)),
    )
    points = tune_source_operating_points(
        [sequence], "ma_dwell", list(base.TARGETS), STEP,
        ma_seconds=MA, dwell_seconds=DWELL, refractory_seconds=REFRACTORY,
        grid_size=160,
    )
    return {fp: (float(points[fp].threshold),
                 float(points[fp].source_fp_per_minute)) for fp in base.TARGETS}


def metric_at(ma, threshold, allowed, evalmask, idle, exposure, gt, times,
              onsets, shifts: int, seed: int) -> dict:
    alarms = alarms_from_ma(
        ma, threshold, allowed, dwell=round(DWELL / STEP),
        refractory=round(REFRACTORY / STEP),
    ) & evalmask

    def sensitivity(candidate, delta):
        return float(np.mean([
            np.any(candidate & (times >= onset) & (times <= onset + delta))
            for onset in onsets
        ]))

    observed = {str(d): sensitivity(alarms, d) for d in DELTAS}
    null = {str(d): [] for d in DELTAS}
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, max(2, len(alarms)), size=shifts)
    for offset in offsets:
        shifted = np.roll(alarms, int(offset))
        for d in DELTAS:
            null[str(d)].append(sensitivity(shifted, d))
    null = {k: float(np.median(v)) for k, v in null.items()}
    pred = intervals_from_alarm(times, alarms, ma, threshold, allowed & evalmask)
    tolerant = interval_metrics(pred, gt, 0.1, 0.1)
    strict = interval_metrics(pred, gt, 0.7, 0.7)
    fp_count = int(np.sum(alarms & idle))
    return {
        "actual_fp": float(fp_count / (exposure / 60.)) if exposure else None,
        "false_alarm_count": fp_count,
        "idle_exposure_seconds": float(exposure),
        "onset_sensitivity": observed,
        "circular_shift_null_median": null,
        "circular_shift_excess": {k: float(observed[k] - null[k]) for k in observed},
        "coverage": tolerant["coverage"],
        "temporal_iou": tolerant["temporal_iou"],
        "strict_f1": strict["intersection_f1"],
        "predicted_intervals": int(len(pred)),
        "alarm_count": int(alarms.sum()),
        "threshold": float(threshold),
    }


def analyze(subject, session, domains, own_cache, shifts):
    anchor_cov, stream_cov, events, times, starts, allowed = base.prepare_target(subject, session)
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
    z = ewma_z(raw, float(anchor.mean()), float(anchor.var(ddof=1)), allowed)
    ma = ma_trace(z, n=round(MA / STEP))
    cues = events[:, 5] / SFREQ
    onsets = cues[base.SCREEN:] + 0.5
    gt = np.c_[onsets, cues[base.SCREEN:] + 4.0]
    evalmask = times >= events[base.SCREEN, 2] / SFREQ
    state = np.zeros(len(times), dtype=bool)
    for a, b in gt:
        state |= (times >= a) & (times <= b)
    idle = evalmask & allowed & ~state
    exposure = float(idle.sum() * STEP)
    thresholds = tune_anchor(anchor)
    surface = {}
    for fp, (threshold, anchor_fp) in thresholds.items():
        # Match the CNN-GRU evaluator's deterministic null-shift stream exactly.
        null_seed = 88000 + 100 * session + subject + int(fp * 10)
        surface[str(fp)] = {
            "anchor_fp_per_min": anchor_fp,
            "fixed_threshold": metric_at(
                ma, threshold, allowed, evalmask, idle, exposure, gt,
                times, onsets, shifts, null_seed,
            ),
        }
    return {"subject": subject, "session": session,
            "arms": {"A_target_rest": {"surface": surface}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/lee2019_loso_deployment_dwell03_null500.json"))
    args = parser.parse_args()
    targets = args.subjects or list(base.SUBJECTS)
    rows = []
    for session in args.sessions:
        domains = base.load_covariances(session)
        own_cache = {}
        for subject in base.SUBJECTS:
            own_ref = base.reference(domains[subject]["pre"])
            own_cache[subject] = {
                "command": tangent_features(domains[subject]["command"], own_ref),
                "idle": tangent_features(domains[subject]["idle"], own_ref),
            }
        for i, subject in enumerate(targets, 1):
            rows.append(analyze(subject, session, domains, own_cache, args.shifts))
            print(f"dwell=.3 S{session} sub-{subject:02d} [{i}/{len(targets)}]", flush=True)
    payload = {
        "protocol": {
            "model": "Riemannian LOSO Arm A; target pre-rest tangent reference and threshold",
            "stream": "same locked seam-guarded pseudo-uncued stream",
            "decision": "EWMA H30 + MA 0.5 s + dwell 0.3 s + refractory 1 s",
            "null": f"{args.shifts} circular shifts of alarm series; same target/FP seeds and shift draws as CNN-GRU",
            "onset": "cue-relative pseudo-onset cue+0.5 s; first 10 trials excluded",
            "claim_limit": "not actual self-initiated onset performance",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
