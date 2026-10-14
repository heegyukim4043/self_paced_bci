"""E4b frozen tier-0 task-idle calibration on Lee original continuous EEG.

Internal exploratory comparison only. The cumulative arm knows the cued idle
schedule and is not a deployable unlabelled self-paced method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.run_e4_original_continuous_fp_20261009 import (
    STEP, DWELL, REFRACTORY, MA_STEPS, _candidate_counts,
    _trial_strata, _union, threshold_grid,
)
from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms

TARGETS = (4., 2.)
ARMS = ("T56", "T_all", "T_cum", "T_sham_update")
WINDOWS = ("E_full", "E_updated", "E_70_79", "E_80_89", "E_90_99")
BOOT = 2000
SEED = 20261010


@njit(cache=True)
def rearmed_with_updates(ma: np.ndarray, thresholds: np.ndarray,
                         update_indices: np.ndarray) -> np.ndarray:
    """Continuous decision state, with rearm required after threshold changes.

    Refractory state persists over an update. The sham arm applies the identical
    resets but keeps threshold fixed, isolating update-boundary effects.
    """
    out = np.zeros(len(ma), dtype=np.bool_)
    armed = False
    count = 0
    blocked_until = -1
    phase = 0
    for index in range(len(ma)):
        if phase < len(update_indices) and index == update_indices[phase]:
            armed = False
            count = 0
            phase += 1
        threshold = thresholds[phase]
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
            out[index] = True
            blocked_until = index + REFRACTORY
    return out


def curve(ma: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=np.bool_)
    exposure = float(mask.sum() * STEP)
    if exposure < 56. - 1e-6:
        raise AssertionError("calibration exposure shorter than 56 seconds")
    thresholds = np.asarray(threshold_grid(ma, mask), dtype=np.float64)
    stop = int(np.flatnonzero(mask)[-1] + 1)
    counts = _candidate_counts(np.asarray(ma, np.float64), mask, thresholds, stop)
    return {"thresholds": thresholds, "counts": counts,
            "seconds": exposure, "candidate_count": len(thresholds)}


def pick(point_curve: dict, target: float) -> dict:
    rates = point_curve["counts"] * 60. / point_curve["seconds"]
    error = np.abs(rates - target)
    # Preserve the original E4 tie-break: highest threshold among closest rates.
    best = int(np.flatnonzero(np.isclose(error, np.min(error), rtol=0, atol=1e-12))[-1])
    return {"threshold": float(point_curve["thresholds"][best]),
            "count": int(point_curve["counts"][best]),
            "seconds": point_curve["seconds"],
            "rate_per_min": float(rates[best]),
            "candidate_count": point_curve["candidate_count"]}


def _prepare(cache_path: Path, subject: int, session: int):
    with np.load(cache_path, allow_pickle=False) as data:
        if (int(data["subject"]), int(data["session"])) != (subject, session):
            raise AssertionError("cache identity mismatch")
        pre_times = np.asarray(data["pre_times"], dtype=float)
        pre_features = np.asarray(data["pre_features"], dtype=float)
        times = np.asarray(data["continuous_times"], dtype=float)
        features = np.asarray(data["continuous_features"], dtype=float)
        channels = tuple(str(x) for x in data["channels"])
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
        fixation = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
    common_stream = times >= 4. - .001
    times, features = times[common_stream], features[common_stream]
    if not np.isclose(times[0], 4., atol=.001):
        raise AssertionError("continuous start not 4 seconds")
    pre_mask = (pre_times >= 4. - .001) & (pre_times < 60. - .001)
    if pre_mask.sum() != 560:
        raise AssertionError("not 560 common anchor points")
    normalizer = LocationScale.fit(pre_features[pre_mask], axis=0)
    pre_raw = tier0_from_features(pre_features[pre_mask], channels, normalizer)
    raw = tier0_from_features(features, channels, normalizer)
    mean, var = float(pre_raw.mean()), float(pre_raw.var(ddof=1))
    z = ewma_z(raw, mean, var, np.ones(len(raw), dtype=bool))
    ma = guard_local_moving_average(z, np.ones(len(z), dtype=bool), ma_steps=MA_STEPS)
    return times, cues, fixation, iti, ma


def one(cache_path: Path, preflight: dict, e4_reference: dict | None = None) -> list[dict]:
    subject, session = preflight["subject"], preflight["session"]
    times, cues, fixation, iti, ma = _prepare(cache_path, subject, session)
    ids = {"T56": preflight["T56_trial_ids"],
           "T_all": list(range(10, 70)),
           "T_79": list(range(10, 80)),
           "T_89": list(range(10, 90)),
           "E_full": list(range(70, 100)),
           "E_updated": list(range(80, 100)),
           "E_70_79": list(range(70, 80)),
           "E_80_89": list(range(80, 90)),
           "E_90_99": list(range(90, 100))}
    masks = {name: _union(_trial_strata(times, cues, fixation, iti, trial_ids), False)
             for name, trial_ids in ids.items()}
    for name, mask in masks.items():
        if not np.isclose(mask.sum() * STEP, preflight["seconds"][name], atol=.001):
            raise AssertionError(f"mask/exposure changed after metadata preflight: {name}")
    updates = np.asarray([b["index"] for b in preflight["boundaries"]], dtype=np.int64)
    for b in preflight["boundaries"]:
        if not np.isclose(times[b["index"]], b["time_s"], atol=STEP + .001):
            raise AssertionError("update index and time disagree")
    curves = {name: curve(ma, masks[name]) for name in ("T56", "T_all", "T_79", "T_89")}
    rows = []
    for target in TARGETS:
        points = {name: pick(c, target) for name, c in curves.items()}
        if e4_reference is not None:
            old = e4_reference[(subject, session, target)]
            if not np.isclose(points["T56"]["threshold"], old["threshold"], atol=1e-10):
                raise AssertionError("T56 threshold does not reproduce frozen E4 T-near")
        t56 = guard_rearmed_alarms(ma, points["T56"]["threshold"],
                                   np.ones(len(ma), dtype=bool), DWELL, REFRACTORY)
        tall = guard_rearmed_alarms(ma, points["T_all"]["threshold"],
                                    np.ones(len(ma), dtype=bool), DWELL, REFRACTORY)
        tcum = rearmed_with_updates(ma, np.asarray([points[k]["threshold"] for k in
                                                    ("T_all", "T_79", "T_89")], dtype=np.float64), updates)
        sham = rearmed_with_updates(ma, np.asarray([points["T_all"]["threshold"]] * 3,
                                                   dtype=np.float64), updates)
        if not np.array_equal(tall[:updates[0]], tcum[:updates[0]]):
            raise AssertionError("T-cum differs from T-all before first update")
        if not np.array_equal(tall[:updates[0]], sham[:updates[0]]):
            raise AssertionError("sham differs from T-all before first reset")
        for arm, alarm in (("T56", t56), ("T_all", tall),
                           ("T_cum", tcum), ("T_sham_update", sham)):
            counts = {name: {"count": int(np.sum(alarm & masks[name])),
                             "seconds": float(masks[name].sum() * STEP),
                             "fp_per_min": float(np.sum(alarm & masks[name]) * 60. /
                                                 (masks[name].sum() * STEP))}
                      for name in WINDOWS}
            if counts["E_full"]["count"] != (counts["E_70_79"]["count"] +
                                             counts["E_updated"]["count"]):
                raise AssertionError("full count does not partition at update")
            if counts["E_updated"]["count"] != (counts["E_80_89"]["count"] +
                                                counts["E_90_99"]["count"]):
                raise AssertionError("updated count does not partition")
            rows.append({"subject": subject, "session": session, "target_fp": target,
                         "arm": arm, "calibration": (
                             points["T56"] if arm == "T56" else points["T_all"] if
                             arm == "T_sham_update" else
                             {"initial": points["T_all"], "after_79": points["T_79"],
                              "after_89": points["T_89"]} if arm == "T_cum" else points["T_all"]),
                         "evaluation": counts})
    return rows


def _bootstrap_stat(a: np.ndarray, b: np.ndarray, stat: str, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(a), size=(BOOT, len(a)))
    x, y = a[draw], b[draw]
    if stat == "median":
        d = np.median(x, axis=1) - np.median(y, axis=1)
    elif stat == "sd":
        d = np.std(x, axis=1, ddof=1) - np.std(y, axis=1, ddof=1)
    elif stat == "mean":
        d = np.mean(x, axis=1) - np.mean(y, axis=1)
    else:
        raise ValueError(stat)
    return [float(v) for v in np.quantile(d, (.025, .975))]


def _poisson_reference(seconds: np.ndarray, target: float) -> float:
    lo, hi = ((3., 5.) if target == 4. else (1.5, 2.5))
    expected = seconds / 60. * target
    lower_count = np.ceil(lo * seconds / 60. - 1e-12).astype(int)
    upper_count = np.floor(hi * seconds / 60. + 1e-12).astype(int)
    return float(np.mean(poisson.cdf(upper_count, expected) -
                         poisson.cdf(lower_count - 1, expected)))


def _stats(case: list[dict], target: float, window: str) -> tuple[dict, np.ndarray]:
    observed = [r["evaluation"][window] for r in case]
    n = np.asarray([x["count"] for x in observed], dtype=float)
    sec = np.asarray([x["seconds"] for x in observed], dtype=float)
    fp = n * 60. / sec
    log_error = np.log((n + .5) / (sec / 60. * target))
    band = (3., 5.) if target == 4. else (1.5, 2.5)
    result = {"n_subjects": len(case), "window": window,
              "median_seconds": float(np.median(sec)),
              "min_seconds": float(np.min(sec)),
              "median_count": float(np.median(n)),
              "median_fp_per_min": float(np.median(fp)),
              "q90_fp_per_min": float(np.quantile(fp, .9)),
              "median_log_error": float(np.median(log_error)),
              "sd_log_error": float(np.std(log_error, ddof=1)),
              "iqr_log_error": float(np.diff(np.quantile(log_error, (.25, .75)))[0]),
              "under_half_n": int(np.sum(fp < target / 2.)),
              "over_twice_n": int(np.sum(fp > target * 2.)),
              "admissible_n": int(np.sum((fp >= band[0]) & (fp <= band[1]))),
              "poisson_admissible_reference": _poisson_reference(sec, target)}
    return result, log_error


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for session in (1, 2):
        summary[str(session)] = {}
        for target in TARGETS:
            cases = {arm: sorted((r for r in rows if r["session"] == session and
                                  r["target_fp"] == target and r["arm"] == arm),
                                 key=lambda r: r["subject"]) for arm in ARMS}
            if any(len(group) != 54 for group in cases.values()) or len({
                    tuple(r["subject"] for r in group) for group in cases.values()}) != 1:
                raise AssertionError("cohort subject pairing incomplete")
            per_target = {"arms": {}, "comparisons": {}}
            for window in ("E_full", "E_updated"):
                per_target["arms"][window] = {}
                for arm in ARMS:
                    per_target["arms"][window][arm], _ = _stats(cases[arm], target, window)
            for name, left, right, window in (
                    ("T_all_minus_T56_primary", "T_all", "T56", "E_full"),
                    ("T_cum_minus_T_all_primary", "T_cum", "T_all", "E_updated"),
                    ("T_cum_minus_T_sham_primary", "T_cum", "T_sham_update", "E_updated"),
                    ("T_sham_minus_T_all_primary", "T_sham_update", "T_all", "E_updated"),
                    ("T_cum_minus_T_all_secondary", "T_cum", "T_all", "E_full")):
                a, alog = _stats(cases[left], target, window)
                b, blog = _stats(cases[right], target, window)
                afp = np.asarray([r["evaluation"][window]["fp_per_min"] for r in cases[left]])
                bfp = np.asarray([r["evaluation"][window]["fp_per_min"] for r in cases[right]])
                band = (3., 5.) if target == 4. else (1.5, 2.5)
                indicator = ((afp >= band[0]) & (afp <= band[1])).astype(float)
                reference = ((bfp >= band[0]) & (bfp <= band[1])).astype(float)
                under = (afp < target / 2.).astype(float)
                under_ref = (bfp < target / 2.).astype(float)
                over = (afp > target * 2.).astype(float)
                over_ref = (bfp > target * 2.).astype(float)
                seed = SEED + session * 100 + int(target)
                per_target["comparisons"][name] = {
                    "window": window,
                    "delta_median_log_error": a["median_log_error"] - b["median_log_error"],
                    "delta_median_log_error_ci95": _bootstrap_stat(alog, blog, "median", seed),
                    "delta_sd_log_error": a["sd_log_error"] - b["sd_log_error"],
                    "delta_sd_log_error_ci95": _bootstrap_stat(alog, blog, "sd", seed + 1),
                    "delta_admissible_fraction": float(np.mean(indicator - reference)),
                    "delta_admissible_fraction_ci95": _bootstrap_stat(indicator, reference, "mean", seed + 2),
                    "delta_under_half_fraction": float(np.mean(under - under_ref)),
                    "delta_under_half_fraction_ci95": _bootstrap_stat(under, under_ref, "mean", seed + 3),
                    "delta_over_twice_fraction": float(np.mean(over - over_ref)),
                    "delta_over_twice_fraction_ci95": _bootstrap_stat(over, over_ref, "mean", seed + 4)}
            summary[str(session)][str(target)] = per_target
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    parser.add_argument("--preflight", type=Path,
                        default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    parser.add_argument("--e4-reference", type=Path,
                        default=Path("data/processed/e4_original_continuous_fp_20261009_v2.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/e4b_task_idle_fp_20261010.json"))
    parser.add_argument("--limit-cases", type=int, default=0,
                        help="test-only partial run; does not write output")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    old = json.loads(args.e4_reference.read_text(encoding="utf-8"))
    old_thresholds = {(r["subject"], r["session"], r["target_fp"]):
                      r["calibration"] for r in old["rows"] if
                      r["arm"] == "T_near" and r["mode"] == "all_idle"}
    if len(preflight["rows"]) != 108 or len(old_thresholds) != 216:
        raise AssertionError("missing frozen cohort or E4 threshold reference")
    rows = []
    for index, geo in enumerate(preflight["rows"]):
        cache = args.cache_root / f"session{geo['session']}" / f"sub-{geo['subject']:02d}_tier0_features.npz"
        rows.extend(one(cache, geo, old_thresholds))
        if (index + 1) % 10 == 0:
            print(f"E4b {index + 1}/108", flush=True)
        if args.limit_cases and index + 1 >= args.limit_cases:
            print(f"test-only {index + 1} case(s); no cohort output written")
            return
    result = {"status": "exploratory E4b original continuous cued task-idle FP; not self-paced performance",
              "score": "frozen tier-0; same 4.0-59.9 s pre-rest normalization as E4",
              "decision": "causal EWMA H30, MA5, rearmed dwell3, refractory10, no splice or guard",
              "T_cum_label_access": "cue-labelled idle; conditional oracle, not deployable without labels",
              "calibration": "full continuous alarm chain, masked count; E4 grid and tie-break",
              "unit": "subject, session separate; 2000 paired subject bootstrap samples",
              "preflight": str(args.preflight), "n_cases": len(preflight["rows"]),
              "summary": summarize(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
