"""E4 native-continuous task-idle threshold-transfer diagnostic.

No splicing, guard, target labels in calibration, or onset claim. The full
continuous detector is run at every threshold; masks affect counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.precheck_e4_original_continuous_20261009 import following_iti, in_intervals
from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms

STEP = .1
DWELL = 3
REFRACTORY = 10
MA_STEPS = 5
TARGETS = (4., 2.)
BOOT = 2000
SEED = 20261009


@njit(cache=True)
def _count_masked(ma: np.ndarray, mask: np.ndarray, threshold: float,
                  stop: int) -> int:
    armed = False
    count = 0
    blocked_until = -1
    total = 0
    for index in range(stop):
        value = ma[index]
        if not np.isfinite(value):
            armed = False
            count = 0
            continue
        if not armed:
            if value < threshold:
                armed = True
            continue
        count = count + 1 if value >= threshold else 0
        if count == DWELL and index > blocked_until:
            if mask[index]:
                total += 1
            blocked_until = index + REFRACTORY
    return total


@njit(cache=True)
def _candidate_counts(ma: np.ndarray, mask: np.ndarray,
                      thresholds: np.ndarray, stop: int) -> np.ndarray:
    result = np.empty(len(thresholds), dtype=np.int64)
    for i in range(len(thresholds)):
        result[i] = _count_masked(ma, mask, thresholds[i], stop)
    return result


def threshold_grid(ma: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = ma[mask]
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        raise AssertionError("calibration has no finite MA values")
    stop = np.flatnonzero(mask)[-1] + 1
    past = ma[:stop]
    past = past[np.isfinite(past)]
    # A pre-fixed discrete grid, not an exact partition of every intervening
    # task-period MA value. Endpoints allow a zero-alarm solution.
    return np.unique(np.r_[np.nextafter(np.min(past), -np.inf), selected,
                           np.nextafter(np.max(past), np.inf)])


def tune(ma: np.ndarray, mask: np.ndarray, target: float) -> dict:
    mask = np.asarray(mask, dtype=np.bool_)
    exposure = float(mask.sum() * STEP)
    if exposure < 56. - 1e-6:
        raise AssertionError(f"calibration exposure {exposure} < 56 s")
    thresholds = threshold_grid(ma, mask)
    stop = int(np.flatnonzero(mask)[-1] + 1)
    counts = _candidate_counts(np.asarray(ma, np.float64), mask,
                               np.asarray(thresholds, np.float64), stop)
    rates = counts * 60. / exposure
    error = np.abs(rates - target)
    best = np.flatnonzero(np.isclose(error, np.min(error), rtol=0, atol=1e-12))[-1]
    return {"threshold": float(thresholds[best]),
            "calibration_count": int(counts[best]),
            "calibration_seconds": exposure,
            "calibration_rate": float(rates[best]),
            "candidate_count": int(len(thresholds)),
            "attainable_alarm_counts": sorted(set(int(x) for x in counts))}


def _trial_strata(times: np.ndarray, cues: np.ndarray, fixation: np.ndarray,
                  iti: np.ndarray, ids: list[int]) -> dict[str, np.ndarray]:
    masks = {k: np.zeros(len(times), dtype=bool) for k in
             ("fixation", "iti_early", "iti_late")}
    for i in ids:
        masks["fixation"] |= in_intervals(times, fixation[i:i + 1])
        trial_iti = following_iti(iti, float(cues[i]))
        if len(trial_iti):
            start, stop = trial_iti[0]
            split = min(start + 2., stop)
            masks["iti_early"] |= in_intervals(times, np.array([[start, split]]))
            masks["iti_late"] |= in_intervals(times, np.array([[split, stop]]))
    if np.any(masks["fixation"] & masks["iti_early"]) or np.any(
            masks["iti_early"] & masks["iti_late"]):
        raise AssertionError("overlapping strata")
    return masks


def _union(strata: dict[str, np.ndarray], late_only: bool) -> np.ndarray:
    out = strata["fixation"] | strata["iti_late"]
    if not late_only:
        out = out | strata["iti_early"]
    return out


def _alarm_result(ma: np.ndarray, threshold: float, strata: dict[str, np.ndarray],
                  late_only: bool) -> dict:
    alarm = guard_rearmed_alarms(ma, threshold, np.ones(len(ma), dtype=bool),
                                 dwell_steps=DWELL, refractory_steps=REFRACTORY)
    mask = _union(strata, late_only)
    result = {"count": int(np.sum(alarm & mask)),
              "seconds": float(mask.sum() * STEP)}
    result["fp_per_min"] = result["count"] * 60. / result["seconds"]
    result["strata"] = {}
    for name, part in strata.items():
        exposure = float(part.sum() * STEP)
        result["strata"][name] = {
            "count": int(np.sum(alarm & part)), "seconds": exposure,
            "fp_per_min": float(np.sum(alarm & part) * 60. / exposure) if exposure else None}
    return result


def one(cache_path: Path, geometry: dict) -> dict:
    with np.load(cache_path, allow_pickle=False) as data:
        subject, session = int(data["subject"]), int(data["session"])
        pre_times = np.asarray(data["pre_times"], dtype=float)
        pre_features = np.asarray(data["pre_features"], dtype=float)
        times = np.asarray(data["continuous_times"], dtype=float)
        features = np.asarray(data["continuous_features"], dtype=float)
        channels = tuple(str(x) for x in data["channels"])
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
        fixation = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
    if (subject, session) != (geometry["subject"], geometry["session"]):
        raise AssertionError("geometry/cache identity mismatch")
    if len(cues) != 100 or not np.all(np.diff(times) > 0):
        raise AssertionError("invalid original continuous timeline")
    # Historical S1 caches for 15 subjects contain 2-4 s extra feature ends.
    # Harmonize the *continuous EWMA start* as well as the pre-rest anchor.
    common_stream = times >= 4. - .001
    times, features = times[common_stream], features[common_stream]
    if not np.isclose(times[0], 4., atol=.001):
        raise AssertionError("common continuous start must be 4.0 s")
    pre_mask = (pre_times >= 4. - .001) & (pre_times < 60. - .001)
    if pre_mask.sum() != 560:
        raise AssertionError("anchor must have exactly 560 common points")
    normalizer = LocationScale.fit(pre_features[pre_mask], axis=0)
    pre_raw = tier0_from_features(pre_features[pre_mask], channels, normalizer)
    raw = tier0_from_features(features, channels, normalizer)
    mean, var = float(pre_raw.mean()), float(pre_raw.var(ddof=1))
    anchor_z = ewma_z(pre_raw, mean, var, np.ones(len(pre_raw), dtype=bool))
    stream_z = ewma_z(raw, mean, var, np.ones(len(raw), dtype=bool))
    anchor_ma = guard_local_moving_average(anchor_z, np.ones(len(anchor_z), dtype=bool),
                                           ma_steps=MA_STEPS)
    stream_ma = guard_local_moving_average(stream_z, np.ones(len(stream_z), dtype=bool),
                                           ma_steps=MA_STEPS)
    future = _trial_strata(times, cues, fixation, iti, list(range(70, 100)))
    rows = []
    for late_only in (False, True):
        mode = "late_only" if late_only else "all_idle"
        future_mask = _union(future, late_only)
        expected_future = (geometry["future_late_only_seconds"] if late_only
                           else geometry["future_total_seconds"])
        if not np.isclose(future_mask.sum() * STEP, expected_future, atol=.001):
            raise AssertionError("future mask differs from preflight")
        for target in TARGETS:
            for arm in ("A", "T_far", "T_near"):
                if arm == "A":
                    calibration_ma = anchor_ma
                    calibration_mask = np.ones(len(anchor_ma), dtype=bool)
                else:
                    key = arm + ("_late_only" if late_only else "")
                    arm_geom = geometry["arms"][key]
                    trial_ids = arm_geom["trial_ids"]
                    if trial_ids != list(range(trial_ids[0], trial_ids[-1] + 1)):
                        raise AssertionError("calibration trial IDs are not consecutive")
                    if max(trial_ids) >= 70:
                        raise AssertionError("future trial in calibration")
                    cal_strata = _trial_strata(times, cues, fixation, iti, trial_ids)
                    calibration_mask = _union(cal_strata, late_only)
                    if not np.isclose(calibration_mask.sum() * STEP,
                                      arm_geom["endpoint_seconds"], atol=.001):
                        raise AssertionError("calibration mask differs from preflight")
                    if times[np.flatnonzero(calibration_mask)[-1]] >= fixation[70, 0]:
                        raise AssertionError("future temporal leakage")
                    calibration_ma = stream_ma
                point = tune(calibration_ma, calibration_mask, target)
                future_result = _alarm_result(stream_ma, point["threshold"],
                                              future, late_only)
                rows.append({"subject": subject, "session": session,
                             "mode": mode, "target_fp": target, "arm": arm,
                             "calibration": point, "future": future_result})
    return {"subject": subject, "session": session, "rows": rows}


def _paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(values), size=(BOOT, len(values)))
    sample = values[draw]
    return [float(x) for x in np.quantile(sample.mean(axis=1), (.025, .975))]


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for session in (1, 2):
        sess = {}
        for mode in ("all_idle", "late_only"):
            modes = {}
            for target in TARGETS:
                group = [r for r in rows if r["session"] == session and
                         r["mode"] == mode and r["target_fp"] == target]
                by_arm = {arm: sorted((r for r in group if r["arm"] == arm),
                                      key=lambda r: r["subject"])
                          for arm in ("A", "T_far", "T_near")}
                if any(len(v) != 54 for v in by_arm.values()):
                    raise AssertionError("54 paired subjects per arm required")
                if len({tuple(r["subject"] for r in group) for group in by_arm.values()}) != 1:
                    raise AssertionError("subject pairing differs")
                stats = {}
                for arm, case in by_arm.items():
                    fp = np.asarray([r["future"]["fp_per_min"] for r in case])
                    expected = np.asarray([r["future"]["seconds"] * target / 60.
                                           for r in case])
                    count = np.asarray([r["future"]["count"] for r in case])
                    band = (3., 5.) if target == 4. else (1.5, 2.5)
                    stats[arm] = {
                        "median_fp_per_min": float(np.median(fp)),
                        "q90_fp_per_min": float(np.quantile(fp, .9)),
                        "admissible_n": int(np.sum((fp >= band[0]) & (fp <= band[1]))),
                        "admissible_fraction": float(np.mean((fp >= band[0]) & (fp <= band[1]))),
                        "median_signed_log_error": float(np.median(np.log((count + .5) / (expected + .5)))),
                        "median_calibration_seconds": float(np.median([
                            r["calibration"]["calibration_seconds"] for r in case])),
                        "median_calibration_count": float(np.median([
                            r["calibration"]["calibration_count"] for r in case])),
                        "median_future_seconds": float(np.median([
                            r["future"]["seconds"] for r in case])),
                        "n_zero_future_alarm": int(np.sum(count == 0)),
                    }
                comparisons = {}
                for left, right in (("T_far", "A"), ("T_near", "A"),
                                    ("T_near", "T_far")):
                    la, ra = by_arm[left], by_arm[right]
                    band = (3., 5.) if target == 4. else (1.5, 2.5)
                    delta_adm = np.asarray([
                        float(band[0] <= l["future"]["fp_per_min"] <= band[1]) -
                        float(band[0] <= r["future"]["fp_per_min"] <= band[1])
                        for l, r in zip(la, ra)])
                    delta_fp = np.asarray([l["future"]["fp_per_min"] - r["future"]["fp_per_min"]
                                           for l, r in zip(la, ra)])
                    comparisons[f"{left}_minus_{right}"] = {
                        "admissible_fraction_difference": float(delta_adm.mean()),
                        "admissible_fraction_difference_ci95": _paired_ci(delta_adm, SEED + session + int(target) * 10),
                        "median_paired_fp_difference": float(np.median(delta_fp)),
                        "mean_paired_fp_difference_ci95": _paired_ci(delta_fp, SEED + session + int(target) * 100),
                    }
                modes[str(target)] = {"arms": stats, "paired": comparisons}
            sess[mode] = modes
        summary[str(session)] = sess
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    parser.add_argument("--preflight", type=Path,
                        default=Path("data/processed/e4_full_trial_blocks_preflight_20261009.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/e4_original_continuous_fp_20261009_v2.json"))
    parser.add_argument("--limit-cases", type=int, default=0,
                        help="test-only partial run; never writes a cohort summary")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    geo_rows = preflight["rows"]
    if len(geo_rows) != 108:
        raise AssertionError("preflight requires 108 cases")
    case_rows = []
    for index, geometry in enumerate(geo_rows):
        path = args.cache_root / f"session{geometry['session']}" / f"sub-{geometry['subject']:02d}_tier0_features.npz"
        case_rows.append(one(path, geometry))
        if (index + 1) % 10 == 0:
            print(f"E4 {index + 1}/{len(geo_rows)}", flush=True)
        if args.limit_cases and len(case_rows) >= args.limit_cases:
            print(f"test-only {len(case_rows)} case(s); no output written")
            return
    rows = [r for case in case_rows for r in case["rows"]]
    result = {
        "status": "exploratory E4 original continuous task-idle FP, not uncued/self-paced performance",
        "score": "tier-0 max(C3,C4) mu/beta ERD; common pre-rest 4.0-59.9 s normalizer",
        "continuous_start": "all cases trimmed to first feature end 4.0 s before EWMA and decision",
        "decision": "causal EWMA H30, MA5, rearmed dwell3, refractory10; no reset/splice/guard",
        "calibration_grid": "selected calibration MA values plus past global min/max sentinels; discrete approximate task-arm grid",
        "unit": "subject; sessions separate; 2000 subject-paired bootstrap draws",
        "preflight": str(args.preflight), "n_cases": len(case_rows),
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
