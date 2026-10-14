"""E6: frozen T-all FP4, original-continuous cue-relative event diagnostic.

The fixation start is protocol-derived (cue-3 s), not an independent marker.
No threshold is tuned on trial 70-99. This is not self-paced MI validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_e4_original_continuous_fp_20261009 import STEP, DWELL, REFRACTORY, _trial_strata, _union
from scripts.run_e4b_task_idle_20261010 import _prepare
from uncued_mi.config import ExperimentConfig
from uncued_mi.decision import guard_rearmed_alarms

SEED = 20261010
BOOT = 2000
TARGET_FP = 4.0
COLLAR = 2.0
MIN_SHIFT = 10.0
PERIOD_EXCLUSION = 2.0
WINDOW = ExperimentConfig().window_seconds


def collar_indices(times: np.ndarray, onsets: np.ndarray,
                   width: float = COLLAR) -> tuple[np.ndarray, np.ndarray]:
    starts = np.searchsorted(times, onsets, side="left")
    ends = np.searchsorted(times, onsets + width, side="right")
    if np.any(starts < 0) or np.any(ends > len(times)) or np.any(ends <= starts):
        raise AssertionError("invalid event collar")
    return starts, ends


def hit_vector(alarms: np.ndarray, starts: np.ndarray,
               ends: np.ndarray) -> np.ndarray:
    counts = np.r_[0, np.cumsum(alarms.astype(np.int32))]
    return (counts[ends] - counts[starts]) > 0


def null_by_all_shifts(segment: np.ndarray, mi_starts: np.ndarray,
                       mi_ends: np.ndarray, fix_starts: np.ndarray,
                       fix_ends: np.ndarray, period: float) -> dict:
    length = len(segment)
    min_steps = int(round(MIN_SHIFT / STEP))
    offsets = np.arange(min_steps + 1, length - min_steps, dtype=np.int32)
    if len(offsets) < 1000:
        raise AssertionError("too few admissible shifts")
    mi_null = np.empty(len(offsets), dtype=np.float32)
    fix_null = np.empty(len(offsets), dtype=np.float32)
    for j, offset in enumerate(offsets):
        rolled = np.roll(segment, int(offset))
        mi_null[j] = hit_vector(rolled, mi_starts, mi_ends).mean()
        fix_null[j] = hit_vector(rolled, fix_starts, fix_ends).mean()
    seconds = offsets * STEP
    nearest_multiple = np.rint(seconds / period)
    near_period = (nearest_multiple >= 1) & (
        np.abs(seconds - nearest_multiple * period) <= PERIOD_EXCLUSION + 1e-12)
    keep = ~near_period
    if not np.any(keep):
        raise AssertionError("period-excluded sensitivity has no offsets")
    return {
        "offset_seconds": np.round(seconds, 3).tolist(),
        "mi_sen_by_offset": np.round(mi_null, 6).tolist(),
        "fixation_sen_by_offset": np.round(fix_null, 6).tolist(),
        "period_excluded_offsets": int(np.sum(near_period)),
        "n_primary_offsets": int(len(offsets)),
        "n_period_sensitivity_offsets": int(np.sum(keep)),
        "mi_primary_mean": float(np.mean(mi_null)),
        "fixation_primary_mean": float(np.mean(fix_null)),
        "mi_period_excluded_mean": float(np.mean(mi_null[keep])),
        "fixation_period_excluded_mean": float(np.mean(fix_null[keep])),
        "mi_primary_q05_q50_q95": [float(x) for x in np.quantile(mi_null, (.05, .5, .95))],
        "fixation_primary_q05_q50_q95": [float(x) for x in np.quantile(fix_null, (.05, .5, .95))],
    }


def case(cache_path: Path, frozen_row: dict) -> dict:
    subject, session = frozen_row["subject"], frozen_row["session"]
    times, cues, fixation, iti, ma = _prepare(cache_path, subject, session)
    if len(cues) != 100 or fixation.shape != (100, 2):
        raise AssertionError("unexpected cue/fixation geometry")
    if not np.allclose(fixation[:, 0], cues - 3., atol=.005, rtol=0):
        raise AssertionError("fixation start differs from protocol cue-3 s")
    if not np.allclose(fixation[:, 1], cues, atol=.005, rtol=0):
        raise AssertionError("fixation stop differs from arrow cue")
    future_idle = _union(_trial_strata(times, cues, fixation, iti, list(range(70, 100))), False)
    threshold = float(frozen_row["calibration"]["threshold"])
    alarms = guard_rearmed_alarms(ma, threshold, np.ones(len(ma), dtype=bool),
                                   dwell_steps=DWELL, refractory_steps=REFRACTORY)
    n_idle = int(np.sum(alarms & future_idle))
    seconds_idle = float(np.sum(future_idle) * STEP)
    old = frozen_row["evaluation"]["E_full"]
    if n_idle != old["count"] or not np.isclose(seconds_idle, old["seconds"], atol=1e-9):
        raise AssertionError("E6 failed to reproduce E4b T-all FP4 alarms/exposure")
    # The frozen E4b mask admits a 1-ms boundary tolerance; a float32 grid
    # may include the immediately preceding 0.1-s point. Preserve its exact
    # future support rather than silently dropping one eligible idle point.
    i0 = int(np.flatnonzero(future_idle)[0])
    if abs(float(times[i0]) - float(fixation[70, 0])) > STEP + .002:
        raise AssertionError("future start differs from first fixation")
    # The final trial has no following labelled ITI. End after its 4-s task.
    i1 = int(np.searchsorted(times, cues[99] + 4., side="right"))
    if i1 <= i0 or np.any(future_idle[:i0]) or np.any(future_idle[i1:]):
        raise AssertionError("future idle outside future circular interval")
    segment = alarms[i0:i1]
    segment_times = times[i0:i1]
    mi_onsets = cues[70:100] + .5
    fix_onsets = fixation[70:100, 0] + .5
    mi_starts, mi_ends = collar_indices(segment_times, mi_onsets)
    fix_starts, fix_ends = collar_indices(segment_times, fix_onsets)
    if np.any(fix_ends > mi_starts) or np.any(mi_ends[:-1] > fix_starts[1:]):
        raise AssertionError("event collars overlap")
    mi_hits = hit_vector(segment, mi_starts, mi_ends)
    fix_hits = hit_vector(segment, fix_starts, fix_ends)
    period = float(np.mean(np.diff(cues[70:100])))
    null = null_by_all_shifts(segment, mi_starts, mi_ends,
                              fix_starts, fix_ends, period)
    latency = []
    mi_support_fraction = []
    for start, end, onset in zip(mi_starts, mi_ends, mi_onsets):
        loc = np.flatnonzero(segment[start:end])
        if len(loc):
            alarm_time = float(segment_times[start + loc[0]])
            latency.append(alarm_time - float(onset))
            # Nominal finite-window overlap; causal IIR filter memory may
            # reach farther back, so this is not a cue-free support claim.
            overlap = max(0., min(alarm_time, float(onset) + 3.) -
                          max(alarm_time - WINDOW, float(onset)))
            mi_support_fraction.append(float(overlap / WINDOW))
    if len(latency) != int(mi_hits.sum()):
        raise AssertionError("latency and hit counts differ")
    future_mask = np.zeros(len(times), dtype=bool)
    future_mask[i0:i1] = True
    mi_collar_mask = np.zeros(len(times), dtype=bool)
    for start, end in zip(mi_starts + i0, mi_ends + i0):
        mi_collar_mask[start:end] = True
    if np.any(mi_collar_mask & future_idle):
        raise AssertionError("MI and idle accounting overlap")
    unclassified = int(np.sum(alarms & future_mask & ~mi_collar_mask & ~future_idle))
    return {
        "subject": subject, "session": session, "target_fp": TARGET_FP,
        "threshold": threshold,
        "calibration": frozen_row["calibration"],
        "future_idle": {"count": n_idle, "seconds": seconds_idle,
                         "fp_per_min": n_idle * 60. / seconds_idle},
        "future_time_seconds": float((i1 - i0) * STEP),
        "future_alarm_count": int(np.sum(segment)),
        "future_mi_collar_alarm_count": int(np.sum(alarms & mi_collar_mask)),
        "future_unclassified_alarm_count": unclassified,
        "mean_cue_period_s": period,
        "mi": {"n_events": 30, "hits": int(mi_hits.sum()),
               "sen_2s": float(np.mean(mi_hits)),
               "detected_latency_s": latency,
               "first_alarm_mi_window_support_fraction": mi_support_fraction,
               "first_alarm_strict_support_n": int(np.sum(np.asarray(mi_support_fraction) >= 1. - 1e-4)),
               "null_sen_2s": null["mi_primary_mean"],
               "excess_2s": float(np.mean(mi_hits) - null["mi_primary_mean"]),
               "null_period_excluded": null["mi_period_excluded_mean"],
               "excess_period_excluded": float(np.mean(mi_hits) - null["mi_period_excluded_mean"])},
        "fixation": {"n_events": 30, "hits": int(fix_hits.sum()),
                     "sen_2s": float(np.mean(fix_hits)),
                     "null_sen_2s": null["fixation_primary_mean"],
                     "excess_2s": float(np.mean(fix_hits) - null["fixation_primary_mean"]),
                     "null_period_excluded": null["fixation_period_excluded_mean"],
                     "excess_period_excluded": float(np.mean(fix_hits) - null["fixation_period_excluded_mean"])},
        "mi_minus_fixation_excess": float(
            np.mean(mi_hits) - null["mi_primary_mean"] -
            np.mean(fix_hits) + null["fixation_primary_mean"]),
        "null": null,
    }


def ci_median(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOT, len(values)))
    return [float(x) for x in np.quantile(np.median(values[draws], axis=1), (.025, .975))]


def summarize(rows: list[dict]) -> dict:
    result = {}
    for session in (1, 2):
        sub = [r for r in rows if r["session"] == session]
        if len(sub) != 54 or len({r["subject"] for r in sub}) != 54:
            raise AssertionError("missing subject/session in E6")
        def arr(path):
            return np.asarray([path(r) for r in sub], dtype=float)
        metrics = {
            "mi_sen_2s": arr(lambda r: r["mi"]["sen_2s"]),
            "mi_null_2s": arr(lambda r: r["mi"]["null_sen_2s"]),
            "mi_excess_2s": arr(lambda r: r["mi"]["excess_2s"]),
            "mi_excess_period_excluded": arr(lambda r: r["mi"]["excess_period_excluded"]),
            "fixation_excess_2s": arr(lambda r: r["fixation"]["excess_2s"]),
            "mi_minus_fixation_excess": arr(lambda r: r["mi_minus_fixation_excess"]),
            "fp_per_min": arr(lambda r: r["future_idle"]["fp_per_min"]),
        }
        result[str(session)] = {
            name: {"median": float(np.median(v)),
                   "mean": float(np.mean(v)),
                   "subject_bootstrap_median_ci95": ci_median(v, SEED + 100 * session + j)}
            for j, (name, v) in enumerate(metrics.items())
        }
        result[str(session)]["fp_admissible_3_to_5"] = int(
            np.sum((metrics["fp_per_min"] >= 3.) & (metrics["fp_per_min"] <= 5.)))
        result[str(session)]["fp_q90"] = float(np.quantile(metrics["fp_per_min"], .9))
        delays = np.asarray([x for r in sub for x in r["mi"]["detected_latency_s"]], dtype=float)
        support = np.asarray([x for r in sub for x in r["mi"]["first_alarm_mi_window_support_fraction"]], dtype=float)
        result[str(session)]["detected_trial_latency_median_s"] = (
            float(np.median(delays)) if len(delays) else None)
        result[str(session)]["detected_trial_latency_n"] = len(delays)
        result[str(session)]["missed_trial_n"] = 30 * len(sub) - len(delays)
        result[str(session)]["first_alarm_mi_window_support_median_fraction"] = (
            float(np.median(support)) if len(support) else None)
        result[str(session)]["first_alarm_strict_support_n"] = int(
            sum(r["mi"]["first_alarm_strict_support_n"] for r in sub))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--e4b", type=Path, default=Path("data/processed/e4b_task_idle_fp_20261010.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/e6_tall_fixed_onset_20261010.json"))
    ap.add_argument("--limit-cases", type=int, default=0, help="preflight only; do not write output")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.e4b.read_text(encoding="utf-8"))
    frozen = [r for r in data["rows"] if r["target_fp"] == TARGET_FP and r["arm"] == "T_all"]
    if len(frozen) != 108 or len({(r["subject"], r["session"]) for r in frozen}) != 108:
        raise AssertionError("E4b T-all FP4 is not a full frozen cohort")
    rows = []
    for j, r in enumerate(sorted(frozen, key=lambda x: (x["session"], x["subject"]))):
        cache = args.cache_root / f"session{r['session']}" / f"sub-{r['subject']:02d}_tier0_features.npz"
        rows.append(case(cache, r))
        if (j + 1) % 10 == 0:
            print(f"E6 {j + 1}/108", flush=True)
        if args.limit_cases and len(rows) >= args.limit_cases:
            print(f"preflight only: {len(rows)} case(s); no output written")
            return
    result = {
        "status": "exploratory cue-relative E6, original continuous Lee2019; not self-paced MI-specific detection",
        "source": str(args.e4b),
        "threshold_rule": "frozen T-all FP4 from trial10-69; no future retuning",
        "fixation_onset": "protocol-derived cue-3s, not independently timestamped",
        "nominal_causal_feature_window_s": WINDOW,
        "support_audit": "post-run audit correction: tier-0 uses 2-s, not 1-s windows; causal filter memory is longer",
        "null": "all 0.1s circular alarm shifts excluding signed +/-10s; secondary excludes +/-2s of positive multiples of mean future cue interval",
        "unit": "subject, session reported separately; 2000 subject bootstrap draws",
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
