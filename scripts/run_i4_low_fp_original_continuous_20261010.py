"""I4 exploratory low-FP curves on Lee2019 original continuous tier-0 scores.

The future-idle-tuned arm is a diagnostic oracle, not a deployable threshold.
No result is a self-paced MI performance estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_e4_original_continuous_fp_20261009 import (
    DWELL, REFRACTORY, STEP, _trial_strata, _union,
)
from scripts.run_e4b_task_idle_20261010 import _prepare, curve, pick
from scripts.run_e6_tall_fixed_onset_20261010 import (
    BOOT, MIN_SHIFT, PERIOD_EXCLUSION, SEED, collar_indices, hit_vector,
    null_by_all_shifts,
)
from uncued_mi.decision import guard_rearmed_alarms

TARGETS = (1., 2., 4., 8.)
ARMS = ("T_all", "future_idle_oracle")


def fast_shift_null(alarm: np.ndarray, starts: np.ndarray,
                    ends: np.ndarray, period: float) -> dict:
    """Exactly reproduce E6's all-offset alarm-shift null without Python rolls."""
    length = len(alarm)
    min_steps = int(round(MIN_SHIFT / STEP))
    offsets = np.arange(min_steps + 1, length - min_steps, dtype=np.int64)
    if len(offsets) < 1000:
        raise AssertionError("too few circular offsets")
    if np.any(ends <= starts) or np.any(ends > length):
        raise AssertionError("invalid collar support")
    # roll(alarm, k)[s:e] = alarm[(s-k):(e-k)] on the circular timeline.
    doubled = np.r_[alarm.astype(np.int32), alarm.astype(np.int32)]
    prefix = np.r_[0, np.cumsum(doubled, dtype=np.int32)]
    hits = np.zeros(len(offsets), dtype=np.int16)
    for start, end in zip(starts, ends):
        source_start = (int(start) - offsets) % length
        counts = prefix[source_start + int(end - start)] - prefix[source_start]
        hits += (counts > 0).astype(np.int16)
    values = hits / len(starts)
    seconds = offsets * STEP
    nearest = np.rint(seconds / period)
    near_period = (nearest >= 1) & (
        np.abs(seconds - nearest * period) <= PERIOD_EXCLUSION + 1e-12)
    kept = values[~near_period]
    if not len(kept):
        raise AssertionError("period sensitivity has zero offsets")
    return {
        "mean": float(np.mean(values)),
        "period_excluded_mean": float(np.mean(kept)),
        "n_offsets": int(len(values)),
        "n_period_excluded": int(len(kept)),
        "q05_q50_q95": [float(x) for x in np.quantile(values, (.05, .5, .95))],
    }


def test_fast_shift_null() -> None:
    rng = np.random.default_rng(20261010)
    alarm = rng.random(1500) < .025
    starts = np.asarray([150, 350, 550, 750, 950, 1150], dtype=int)
    ends = starts + 20
    old = null_by_all_shifts(alarm, starts, ends, starts, ends, 11.05)
    new = fast_shift_null(alarm, starts, ends, 11.05)
    if not np.isclose(new["mean"], old["mi_primary_mean"], atol=1e-12):
        raise AssertionError("fast shift null differs from E6 primary")
    if not np.isclose(new["period_excluded_mean"],
                      old["mi_period_excluded_mean"], atol=1e-12):
        raise AssertionError("fast shift null differs from E6 period sensitivity")
    if new["n_offsets"] != old["n_primary_offsets"]:
        raise AssertionError("fast shift offset count differs")


def one(cache_path: Path, geometry: dict, e4b: dict | None, e6: dict | None,
        *, prepare_fn=None, tier0_path: Path | None = None) -> list[dict]:
    subject, session = geometry["subject"], geometry["session"]
    if prepare_fn is None:
        times, cues, fixation, iti, ma = _prepare(cache_path, subject, session)
    else:
        if tier0_path is None:
            raise ValueError("tier0_path is required for external score preparation")
        times, cues, fixation, iti, ma = prepare_fn(
            cache_path, tier0_path, subject, session)
    cal_ids = list(range(10, 70))
    eval_ids = list(range(70, 100))
    if max(cal_ids) >= min(eval_ids) or len(cues) != 100:
        raise AssertionError("trial-order split invalid")
    cal_idle = _union(_trial_strata(times, cues, fixation, iti, cal_ids), False)
    future_idle = _union(_trial_strata(times, cues, fixation, iti, eval_ids), False)
    if np.any(cal_idle & future_idle):
        raise AssertionError("calibration/future support overlap")
    if not np.isclose(cal_idle.sum() * STEP, geometry["seconds"]["T_all"], atol=.001):
        raise AssertionError("T-all exposure differs from E4b preflight")
    if not np.isclose(future_idle.sum() * STEP, geometry["seconds"]["E_full"], atol=.001):
        raise AssertionError("future exposure differs from E4b preflight")
    if times[np.flatnonzero(cal_idle)[-1]] >= times[np.flatnonzero(future_idle)[0]]:
        raise AssertionError("calibration is not earlier than future")
    cal_curve = curve(ma, cal_idle)
    future_curve = curve(ma, future_idle)
    i0 = int(np.flatnonzero(future_idle)[0])
    i1 = int(np.searchsorted(times, cues[99] + 4., side="right"))
    if np.any(future_idle[:i0]) or np.any(future_idle[i1:]):
        raise AssertionError("future idle outside circular evaluation segment")
    mi_starts, mi_ends = collar_indices(times[i0:i1], cues[70:100] + .5)
    fix_starts, fix_ends = collar_indices(times[i0:i1], fixation[70:100, 0] + .5)
    if np.any(fix_ends > mi_starts):
        raise AssertionError("fixation and MI collars overlap")
    period = float(np.mean(np.diff(cues[70:100])))
    rows = []
    for target in TARGETS:
        for arm, source_curve in (("T_all", cal_curve),
                                  ("future_idle_oracle", future_curve)):
            point = pick(source_curve, target)
            threshold = point["threshold"]
            alarm = guard_rearmed_alarms(ma, threshold,
                                         np.ones(len(ma), dtype=bool),
                                         dwell_steps=DWELL,
                                         refractory_steps=REFRACTORY)
            count = int(np.sum(alarm & future_idle))
            seconds = float(future_idle.sum() * STEP)
            segment = alarm[i0:i1]
            mi_hits = hit_vector(segment, mi_starts, mi_ends)
            fix_hits = hit_vector(segment, fix_starts, fix_ends)
            mi_null = fast_shift_null(segment, mi_starts, mi_ends, period)
            fix_null = fast_shift_null(segment, fix_starts, fix_ends, period)
            delays = []
            for start, end, onset in zip(mi_starts, mi_ends, cues[70:100] + .5):
                hit = np.flatnonzero(segment[start:end])
                if len(hit):
                    delays.append(float(times[i0 + start + hit[0]] - onset))
            if len(delays) != int(np.sum(mi_hits)):
                raise AssertionError("latency/hit mismatch")
            mi_mask = np.zeros(len(segment), dtype=bool)
            for start, end in zip(mi_starts, mi_ends):
                mi_mask[start:end] = True
            if np.any(mi_mask & future_idle[i0:i1]):
                raise AssertionError("MI/idle mask overlap")
            row = {
                "subject": subject, "session": session, "arm": arm,
                "target_fp": target, "calibration": point,
                "future_idle": {"count": count, "seconds": seconds,
                                "fp_per_min": count * 60. / seconds},
                "mi": {"hits": int(np.sum(mi_hits)), "trials": 30,
                       "sen_2s": float(np.mean(mi_hits)),
                       "null_2s": mi_null["mean"],
                       "excess_2s": float(np.mean(mi_hits) - mi_null["mean"]),
                       "period_excluded_excess": float(
                           np.mean(mi_hits) - mi_null["period_excluded_mean"]),
                       "detected_latency_s": delays},
                "fixation": {"hits": int(np.sum(fix_hits)), "trials": 30,
                             "sen_2s": float(np.mean(fix_hits)),
                             "null_2s": fix_null["mean"],
                             "excess_2s": float(np.mean(fix_hits) - fix_null["mean"])},
                "n_future_alarms": int(np.sum(segment)),
                "n_future_unclassified_alarms": int(np.sum(
                    segment & ~mi_mask & ~future_idle[i0:i1])),
                "shift_null_n": mi_null["n_offsets"],
            }
            if target in (2., 4.) and arm == "T_all" and e4b is not None:
                old = e4b[(subject, session, target)]
                if not np.isclose(threshold, old["calibration"]["threshold"], atol=1e-10):
                    raise AssertionError("T-all threshold differs from E4b")
                old_eval = old["evaluation"]["E_full"]
                if count != old_eval["count"] or not np.isclose(seconds, old_eval["seconds"]):
                    raise AssertionError("T-all future FP differs from E4b")
            if target == 4. and arm == "T_all" and e6 is not None:
                old = e6[(subject, session)]
                if count != old["future_idle"]["count"]:
                    raise AssertionError("FP4 future count differs from E6")
                if row["mi"]["hits"] != old["mi"]["hits"]:
                    raise AssertionError("FP4 MI hits differ from E6")
                if not np.isclose(row["mi"]["null_2s"], old["mi"]["null_sen_2s"], atol=1e-6):
                    raise AssertionError("FP4 MI shift null differs from E6")
            rows.append(row)
    return rows


def median_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(values), size=(BOOT, len(values)))
    return [float(v) for v in np.quantile(np.median(values[drawn], axis=1), (.025, .975))]


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for session in (1, 2):
        summary[str(session)] = {}
        for arm in ARMS:
            summary[str(session)][arm] = {}
            for target in TARGETS:
                group = sorted((r for r in rows if r["session"] == session and
                                r["arm"] == arm and r["target_fp"] == target),
                               key=lambda r: r["subject"])
                if len(group) != 54 or len({r["subject"] for r in group}) != 54:
                    raise AssertionError("incomplete subject group")
                arr = lambda key: np.asarray([key(r) for r in group], dtype=float)
                fp = arr(lambda r: r["future_idle"]["fp_per_min"])
                sen = arr(lambda r: r["mi"]["sen_2s"])
                null = arr(lambda r: r["mi"]["null_2s"])
                excess = arr(lambda r: r["mi"]["excess_2s"])
                fixation_excess = arr(lambda r: r["fixation"]["excess_2s"])
                latency = [x for r in group for x in r["mi"]["detected_latency_s"]]
                cell = {
                    "n_subjects": len(group),
                    "median_calibration_seconds": float(np.median(arr(
                        lambda r: r["calibration"]["seconds"]))),
                    "median_calibration_alarms": float(np.median(arr(
                        lambda r: r["calibration"]["count"]))),
                    "median_future_idle_seconds": float(np.median(arr(
                        lambda r: r["future_idle"]["seconds"]))),
                    "median_future_alarm_count": float(np.median(arr(
                        lambda r: r["future_idle"]["count"]))),
                    "median_actual_fp_per_min": float(np.median(fp)),
                    "fp_q10_q90_per_min": [float(v) for v in np.quantile(fp, (.1, .9))],
                    "n_fp_le_1": int(np.sum(fp <= 1.)),
                    "n_fp_le_2": int(np.sum(fp <= 2.)),
                    "median_sen_2s": float(np.median(sen)),
                    "median_null_2s": float(np.median(null)),
                    "median_subject_excess_2s": float(np.median(excess)),
                    "excess_median_bootstrap_ci95": median_ci(
                        excess, SEED + 1000 * session + 100 * ARMS.index(arm) + int(target)),
                    "median_fixation_excess_2s": float(np.median(fixation_excess)),
                    "median_detected_latency_s": float(np.median(latency)) if latency else None,
                    "detected_trials": len(latency),
                    "missed_trials": 54 * 30 - len(latency),
                }
                summary[str(session)][arm][str(target)] = cell
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    parser.add_argument("--preflight", type=Path,
                        default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    parser.add_argument("--e4b", type=Path,
                        default=Path("data/processed/e4b_task_idle_fp_20261010.json"))
    parser.add_argument("--e6", type=Path,
                        default=Path("data/processed/e6_tall_fixed_onset_20261010_v2.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/i4_low_fp_original_continuous_20261010.json"))
    parser.add_argument("--limit-cases", type=int, default=0,
                        help="preflight only; do not write results")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    test_fast_shift_null()
    geometry = json.loads(args.preflight.read_text(encoding="utf-8"))["rows"]
    old_e4b = json.loads(args.e4b.read_text(encoding="utf-8"))["rows"]
    old_e6 = json.loads(args.e6.read_text(encoding="utf-8"))["rows"]
    e4b = {(r["subject"], r["session"], r["target_fp"]): r
           for r in old_e4b if r["arm"] == "T_all"}
    e6 = {(r["subject"], r["session"]): r for r in old_e6}
    if len(geometry) != 108 or len(e4b) != 216 or len(e6) != 108:
        raise AssertionError("missing frozen cohort reference")
    rows = []
    for j, geo in enumerate(geometry):
        path = args.cache_root / f"session{geo['session']}" / f"sub-{geo['subject']:02d}_tier0_features.npz"
        rows.extend(one(path, geo, e4b, e6))
        if (j + 1) % 10 == 0:
            print(f"I4 {j + 1}/108", flush=True)
        if args.limit_cases and j + 1 >= args.limit_cases:
            print(f"preflight only: {j + 1} case(s); no output written")
            return
    output = {
        "status": "exploratory Lee2019 cue-paced low-FP diagnostic; not self-paced MI validation",
        "design": "I4_LOW_FP_ORIGINAL_CONTINUOUS_PRERUN_20261010.md",
        "score": "frozen tier-0, original continuous recording",
        "threshold_arms": {"T_all": "trial10-69 cue-labelled task idle",
                           "future_idle_oracle": "future trial70-99 idle; diagnostic only"},
        "unit": "subject; sessions separate; 2000 subject bootstrap draws",
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
