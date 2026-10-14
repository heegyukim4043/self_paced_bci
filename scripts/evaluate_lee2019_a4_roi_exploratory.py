"""Exploratory fixed-ROI versus 20-channel LOSO Riemannian comparison.

This bypasses the planned A3-success gate at the user's request to execute
the full A0--A5 sequence. It must not be reported as confirmatory A4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_revalidation_guard_rearm_smoke import (
    fit_loso, source_domains,
)
from scripts.evaluate_lee2019_rest_sham_matched_fp import (
    compute_guard_local_ma, matched_curve, matched_pair,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.run_lee2019_pseudo_uncued_riemannian import reference, rolling_covariance
from uncued_mi.guard_rearm_calibration import tune_guard_rearmed_anchor
from uncued_mi.decision import guard_rearmed_alarms
from uncued_mi.riemannian import tangent_features

SUBSET20 = (2, 3, 9, 10, 12, 13, 14, 17, 19, 20, 23, 25, 26, 29, 38, 40, 42, 51, 53, 54)
ROI = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")


def roi_domains(full: dict[int, dict], indices: np.ndarray) -> dict[int, dict]:
    return {subject: {key: value[:, indices][:, :, indices] for key, value in data.items()}
            for subject, data in full.items()}


def evaluate_one(subject: int, session: int, domains: dict[int, dict],
                 indices: np.ndarray, stream_root: Path, baseline_root: Path) -> list[dict]:
    with np.load(stream_root / f"session{session}/sub-{subject:02d}.npz",
                 allow_pickle=False) as replay, np.load(
                     baseline_root / f"session{session}/sub-{subject:02d}.npz",
                     allow_pickle=False) as baseline:
        signal = np.asarray(replay["signal"], np.float64)[:, indices]
        events = np.asarray(replay["events"], np.int64)
        times = np.asarray(baseline["times"], float)
        allowed = np.asarray(baseline["allowed"], bool)
        base_raw = np.asarray(baseline["raw_scores"], float)
        base_anchor = np.asarray(baseline["anchor_raw_scores"], float)
        assert np.array_equal(events, baseline["events"])
    ref = reference(domains[subject]["pre"])
    scaler, model = fit_loso(domains, subject, ref)
    roi_anchor = model.decision_function(scaler.transform(
        tangent_features(domains[subject]["pre"], ref)))
    stream_cov, ends = rolling_covariance(signal)
    if not np.array_equal(ends / 250., times):
        raise AssertionError("ROI/baseline score grids differ")
    roi_raw = model.decision_function(scaler.transform(tangent_features(stream_cov, ref)))
    future = times >= events[20, 2] / 250.
    idle = future & allowed
    hit_label = np.full(len(times), -1, np.int32)
    for trial, event in enumerate(events[20:]):
        cue = event[5] / 250.
        idle &= ~((times >= cue + .5) & (times <= cue + 4.))
        hit_label[(times >= cue + .5) & (times <= cue + 2.5)] = trial
    exposure_min = idle.sum() * .1 / 60.
    curves = {}
    for name, raw, anchor in (("baseline", base_raw, base_anchor),
                              ("roi", roi_raw, roi_anchor)):
        ma = compute_guard_local_ma(raw, float(anchor.mean()),
                                    float(anchor.var(ddof=1)), allowed)
        curves[name] = matched_curve(ma, allowed, future, idle, hit_label, 301)
    rows = []
    for target in (2., 4.):
        pair = matched_pair(curves["baseline"], curves["roi"], target * exposure_min)
        if pair is None:
            rows.append({"subject": subject, "session": session, "target_fp": target,
                         "matched": False})
            continue
        count, base_hits, roi_hits, base_thr, roi_thr = pair
        rows.append({"subject": subject, "session": session, "target_fp": target,
                     "matched": True, "matched_fp_count": count,
                     "achieved_fp_per_min": count / exposure_min,
                     "baseline_sen2": base_hits / 30., "roi_sen2": roi_hits / 30.,
                     "paired_delta_roi_minus_baseline_sen2": (roi_hits - base_hits) / 30.,
                     "baseline_threshold": base_thr, "roi_threshold": roi_thr})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBSET20))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/"
                        "performance_revalidation_20261001/a4_roi7_vs20_exploratory_subset20.json")
    args = parser.parse_args()
    rows = []
    cov_root = ROOT / "data/interim/lee2019_revalidation_norm_cov"
    stream_root = ROOT / "data/interim/lee2019_revalidation_chronological"
    baseline_root = ROOT / "data/interim/lee2019_revalidation_riemannian_scores"
    for session in args.sessions:
        full = source_domains(session, cov_root)
        with np.load(cov_root / f"session{session}/sub-03.npz", allow_pickle=False) as data:
            channels = data["channels"].tolist()
        indices = np.asarray([channels.index(channel) for channel in ROI], int)
        domains = roi_domains(full, indices)
        for subject in args.subjects:
            rows.extend(evaluate_one(subject, session, domains, indices,
                                     stream_root, baseline_root))
            print(f"A4X fixed ROI S{session} sub-{subject:02d} complete", flush=True)
    result = {"scope": "exploratory fixed-ROI source-pooled LOSO diagnostic only",
              "warning": "A1 and the planned A3 prerequisite are unmet; matched-FP threshold uses evaluation idle; no null correction",
              "roi_channels": list(ROI), "baseline_channels": 20,
              "band_hz": [8, 30], "rows": rows, "by_session": {}}
    for session in args.sessions:
        for fp in (2., 4.):
            subset = [row for row in rows if row["session"] == session and
                      row["target_fp"] == fp and row["matched"]]
            delta = np.asarray([row["paired_delta_roi_minus_baseline_sen2"]
                                for row in subset], float)
            entry = {"matched_subjects": len(subset),
                     "baseline_median_sen2": float(np.median([row["baseline_sen2"]
                                                               for row in subset])) if subset else None,
                     "roi_median_sen2": float(np.median([row["roi_sen2"]
                                                          for row in subset])) if subset else None,
                     "paired_median_delta": float(np.median(delta)) if len(delta) else None}
            if len(delta) >= 10:
                rng = np.random.default_rng(20261001 + session + int(fp) * 100)
                draws = np.median(delta[rng.integers(0, len(delta), (5000, len(delta)))], axis=1)
                entry["descriptive_bootstrap_ci95"] = np.quantile(draws, [.025, .975]).tolist()
            result["by_session"][f"FP{fp:g}_S{session}"] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["by_session"], indent=2))


if __name__ == "__main__":
    main()
