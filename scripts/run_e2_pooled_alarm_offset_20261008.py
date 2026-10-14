"""E2: source-pooled FP16-to-FP2 alarm offset on frozen Lee2019 scores.

Exploratory cue-paced pseudo-uncued FP diagnostic. Lee replay failed G1.
No future rest is used to choose thresholds. Each target subject/session is
one primary evaluation unit; the nine rest partitions are technical cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.decompose_calibration_error_20261008 import (  # noqa: E402
    STEP, alarms_exact, load_case, source_rest_mask,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z  # noqa: E402
from uncued_mi.decision import guard_local_moving_average  # noqa: E402


DELTA_GRID = np.round(np.arange(0.0, 10.0001, 0.05), 10)
SOURCE_RATE_GRID = (8.0, 12.0, 16.0, 24.0)
OUTPUT = ROOT / "data/processed/rate_curve_transfer_20261008"


def ma_for_anchor(raw: np.ndarray, mean: float, var: float, seconds: int = 56) -> np.ndarray:
    steps = int(round(seconds / STEP))
    if len(raw) < steps:
        raise ValueError(f"need {steps} score steps; found {len(raw)}")
    z = ewma_z(np.asarray(raw[:steps], dtype=float), mean, var,
               np.ones(steps, dtype=bool))
    return guard_local_moving_average(z, np.ones(steps, dtype=bool), ma_steps=5)


def exact_sweep(ma: np.ndarray, seconds: int = 56) -> dict:
    """Replicate P1's exact candidate partition and highest-threshold tie rule."""
    finite = np.unique(ma[np.isfinite(ma)])
    if not len(finite):
        raise ValueError("no finite calibration MA")
    candidates = np.unique(np.r_[np.nextafter(float(finite[0]), -np.inf), finite,
                                  np.nextafter(float(finite[-1]), np.inf)])
    allowed = np.ones(len(ma), dtype=bool)
    counts = np.array([int(alarms_exact(ma, threshold, allowed).sum())
                       for threshold in candidates], dtype=int)
    return {"thresholds": candidates, "counts": counts,
            "rates": counts * 60.0 / seconds}


def choose_point(sweep: dict, target_rate: float) -> dict:
    error = np.abs(sweep["rates"] - target_rate)
    selected = np.flatnonzero(np.isclose(error, error.min(), rtol=0, atol=1e-12))[-1]
    return {"threshold": float(sweep["thresholds"][selected]),
            "alarm_count": int(sweep["counts"][selected]),
            "achieved_rate": float(sweep["rates"][selected])}


def choose_offset(count_matrix: np.ndarray, donor_seconds: int = 56) -> dict:
    """Choose source-only pooled FP2 point on the frozen delta grid."""
    if count_matrix.ndim != 2 or count_matrix.shape[1] != len(DELTA_GRID):
        raise ValueError("donor x fixed-delta count matrix required")
    rates = count_matrix.sum(axis=0) * 60.0 / (count_matrix.shape[0] * donor_seconds)
    if np.min(rates) > 2.0:
        return {"valid": False, "reason": "no pooled grid point at or below FP2",
                "rates": rates.tolist()}
    error = np.abs(rates - 2.0)
    selected = np.flatnonzero(np.isclose(error, error.min(), rtol=0, atol=1e-12))[-1]
    return {"valid": True, "delta": float(DELTA_GRID[selected]),
            "pooled_rate": float(rates[selected]),
            "pooled_alarm_count": int(count_matrix[:, selected].sum()),
            "nonmonotone_steps": int(np.count_nonzero(np.diff(rates) > 0)),
            "rates": rates.tolist()}


def beta_from_points(points: dict[float, dict]) -> float:
    thresholds = np.array([points[r]["threshold"] for r in SOURCE_RATE_GRID])
    rates = np.array([points[r]["achieved_rate"] for r in SOURCE_RATE_GRID])
    if len(np.unique(thresholds)) < 3 or np.any(rates <= 0):
        return float("nan")
    slopes = []
    for a in range(len(rates)):
        for b in range(a + 1, len(rates)):
            denominator = thresholds[b] - thresholds[a]
            if denominator != 0:
                slopes.append(-(np.log(rates[b]) - np.log(rates[a])) / denominator)
    value = float(np.median(slopes)) if slopes else float("nan")
    return value if np.isfinite(value) and value > 0 else float("nan")


def episode_diagnostics(ma: np.ndarray, threshold: float) -> dict:
    """Raw above-threshold runs and runs merged across sub-1-second low gaps."""
    above = np.isfinite(ma) & (ma >= threshold)
    changes = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    merged_intervals = []
    for start, end in zip(starts, ends):
        if not merged_intervals or start - merged_intervals[-1][1] >= 10:
            merged_intervals.append([int(start), int(end)])
        else:
            merged_intervals[-1][1] = int(end)
    alarms = np.flatnonzero(alarms_exact(ma, threshold, np.ones(len(ma), dtype=bool)))
    alarm_counts = np.array([np.count_nonzero((alarms >= start) & (alarms < end))
                             for start, end in merged_intervals], dtype=int)
    merged = len(merged_intervals)
    return {"alarm_count": int(len(alarms)), "raw_runs": int(len(starts)),
            "merged_1s_episodes": merged,
            "alarms_per_merged_episode": float(len(alarms) / merged) if merged else None,
            "multi_alarm_episodes": int(np.count_nonzero(alarm_counts >= 2)),
            "multi_alarm_episode_fraction": float(np.mean(alarm_counts >= 2)) if merged else None}


def preflight_rearm() -> None:
    """One uninterrupted high run may create no more than one alarm."""
    ma = np.r_[np.full(5, -1.0), np.full(50, 1.0), np.full(5, -1.0)]
    allowed = np.ones(len(ma), dtype=bool)
    if int(alarms_exact(ma, 0.0, allowed).sum()) != 1:
        raise AssertionError("continuous suprathreshold run re-alarmed")


def evaluation_masks(data: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    technical = []
    for fold in (2, 3, 4):
        masks = [source_rest_mask(data["events"], data["source_ids"], data["starts"],
                                  data["times"], data["allowed"], fold, source)
                 for source in (0, 1, 2)]
        for source in (0, 1, 2):
            technical.append(masks[(source + 1) % 3] | masks[(source + 2) % 3])
    full = np.zeros(len(data["times"]), dtype=bool)
    for fold in (2, 3, 4):
        for source in (0, 1, 2):
            full |= source_rest_mask(data["events"], data["source_ids"], data["starts"],
                                     data["times"], data["allowed"], fold, source)
    if not np.any(full) or any(not np.any(mask) for mask in technical):
        raise AssertionError("empty future-rest evaluation support")
    return full, technical


def load_args() -> SimpleNamespace:
    return SimpleNamespace(
        replay_root=ROOT / "data/interim/lee2019_revalidation_chronological",
        riem_root=ROOT / "data/interim/lee2019_revalidation_riemannian_scores",
        cnn_root=ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache",
        rest_score_root=ROOT / "data/interim/p1_fixed_target_rest_scores_20261008",
    )


def run_case(session: int, subject: int) -> dict:
    data = load_case("riemannian", session, subject, load_args())
    anchor = data["anchor"]
    mean, var = float(anchor.mean()), float(anchor.var(ddof=1))
    target_ma = ma_for_anchor(anchor, mean, var)
    target_sweep = exact_sweep(target_ma)
    a = choose_point(target_sweep, 2.0)
    target16 = choose_point(target_sweep, 16.0)
    if not data["donor_mask"].all():
        raise AssertionError("Riemannian source donor rest should be entirely eligible")

    donor_points = []
    donor_episodes = []
    beta_values = []
    count_matrix = []
    allowed_anchor = np.ones(len(target_ma), dtype=bool)
    for donor_id, raw in zip(data["donor_subjects"], data["donor_scores"]):
        if int(donor_id) == subject:
            raise AssertionError("target leaked into source donor list")
        donor_ma = ma_for_anchor(raw, mean, var)
        donor_sweep = exact_sweep(donor_ma)
        points = {rate: choose_point(donor_sweep, rate) for rate in SOURCE_RATE_GRID}
        donor16 = points[16.0]
        donor2 = choose_point(donor_sweep, 2.0)
        beta_values.append(beta_from_points(points))
        donor_points.append({"subject": int(donor_id), "fp2_threshold": donor2["threshold"],
                             "fp16_threshold": donor16["threshold"],
                             "fp16_alarm_count": donor16["alarm_count"]})
        donor_episodes.append(episode_diagnostics(donor_ma, donor16["threshold"]))
        count_matrix.append([int(alarms_exact(donor_ma, donor16["threshold"] + delta,
                                              allowed_anchor).sum()) for delta in DELTA_GRID])
    counts = np.asarray(count_matrix, dtype=int)
    if counts.shape != (53, len(DELTA_GRID)):
        raise AssertionError("expected 53 source donor curves")
    pooled = choose_offset(counts)
    donor_jackknife = []
    if pooled["valid"]:
        for donor_index in range(53):
            leave_one_out = choose_offset(np.delete(counts, donor_index, axis=0))
            donor_jackknife.append(leave_one_out.get("delta"))
    betas = np.asarray(beta_values)
    finite_betas = betas[np.isfinite(betas)]
    source_beta = float(np.median(finite_betas)) if len(finite_betas) >= 40 else None
    arm_thresholds = {"K_A": a["threshold"],
                      "K_0": float(np.median([p["fp2_threshold"] for p in donor_points]))}
    if pooled["valid"]:
        arm_thresholds["pooled_offset"] = target16["threshold"] + pooled["delta"]
    if source_beta is not None and target16["achieved_rate"] > 0:
        arm_thresholds["beta_slope"] = (target16["threshold"]
                                        + np.log(target16["achieved_rate"] / 2.0) / source_beta)

    z = ewma_z(data["raw"], mean, var, data["allowed"])
    future_ma = guard_local_moving_average(z, data["allowed"], ma_steps=5)
    full_mask, technical_masks = evaluation_masks(data)
    exposure_min = float(full_mask.sum() * STEP / 60.0)
    evaluation = {}
    for arm, threshold in arm_thresholds.items():
        alarms = alarms_exact(future_ma, threshold, data["allowed"])
        count = int(np.count_nonzero(alarms & full_mask))
        technical_fp = [float(np.count_nonzero(alarms & mask) /
                              (mask.sum() * STEP / 60.0)) for mask in technical_masks]
        fp = count / exposure_min
        evaluation[arm] = {
            "threshold": float(threshold), "future_alarm_count": count,
            "future_fp_per_min": float(fp), "future_exposure_s": exposure_min * 60.0,
            "signed_fp_bias": float(fp - 2.0),
            "signed_log_error": float(np.log((count + .5) / (exposure_min * 2.0))),
            "admissible_fp2": bool(1.5 <= fp <= 2.5),
            "above_4_fp_per_min": bool(fp > 4.0),
            "technical_fp_per_min": technical_fp,
            "threshold_above_anchor_ma_max": bool(threshold > np.nanmax(target_ma)),
        }
    return {
        "session": session, "subject": subject, "status": "E2 exploratory; G1 failed",
        "target_fp16": target16, "target_fp2": a,
        "target_fp16_episodes": episode_diagnostics(target_ma, target16["threshold"]),
        "source_donor_count": len(donor_points), "source_donor_points": donor_points,
        "source_donor_fp16_episodes": donor_episodes,
        "source_pooled_offset": {key: value for key, value in pooled.items() if key != "rates"},
        "source_pooled_rate_curve": pooled["rates"],
        "source_offset_leave_one_donor_out": donor_jackknife,
        "source_beta_eligible_donors": int(len(finite_betas)),
        "source_beta": source_beta,
        "evaluation": evaluation,
    }


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = np.mean(values[indices], axis=1)
    return [float(x) for x in np.quantile(estimates, (0.025, 0.975))]


def summarize(cases: list[dict]) -> dict:
    result = {"status": "exploratory Lee replay FP diagnostic; not self-paced BCI performance",
              "inference_unit": "54 subjects; S1/S2 separately; nine rest masks technical only",
              "arms": {}, "paired": {}}
    for session in (1, 2):
        current = sorted([case for case in cases if case["session"] == session],
                         key=lambda case: case["subject"])
        if len(current) != 54:
            raise AssertionError("full cohort of 54 required for summary")
        for arm in ("K_A", "pooled_offset", "beta_slope", "K_0"):
            available = [case for case in current if arm in case["evaluation"]]
            fp = np.array([case["evaluation"][arm]["future_fp_per_min"] for case in available])
            bias = np.array([case["evaluation"][arm]["signed_fp_bias"] for case in available])
            within = np.array([np.std(case["evaluation"][arm]["technical_fp_per_min"], ddof=0)
                               for case in available])
            result["arms"][f"S{session}_{arm}"] = {
                "n_thresholds": len(available), "n_total": 54,
                "admissible_fraction_total": float(np.mean([
                    case["evaluation"].get(arm, {}).get("admissible_fp2", False)
                    for case in current])),
                "fp_median": float(np.median(fp)) if len(fp) else None,
                "fp_q90": float(np.quantile(fp, .9)) if len(fp) else None,
                "fp_above_4_fraction_total": float(np.mean([
                    case["evaluation"].get(arm, {}).get("above_4_fp_per_min", False)
                    for case in current])),
                "signed_fp_bias_mean": float(np.mean(bias)) if len(bias) else None,
                "signed_fp_bias_median": float(np.median(bias)) if len(bias) else None,
                "between_subject_fp_sd": float(np.std(fp, ddof=0)) if len(fp) else None,
                "within_subject_technical_fp_sd_rms": float(np.sqrt(np.mean(within ** 2)))
                if len(within) else None,
            }
        a = np.array([case["evaluation"]["K_A"]["admissible_fp2"] for case in current], dtype=float)
        b = np.array([case["evaluation"].get("pooled_offset", {}).get("admissible_fp2", False)
                      for case in current], dtype=float)
        diff = b - a
        result["paired"][f"S{session}_pooled_minus_K_A_admissible"] = {
            "mean_difference": float(np.mean(diff)),
            "subject_bootstrap_ci95": bootstrap_ci(diff, 20261008 + session),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subjects = parser.add_mutually_exclusive_group(required=True)
    subjects.add_argument("--subjects", nargs="+", type=int)
    subjects.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--sessions", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    chosen = list(range(1, 55)) if args.all_subjects else args.subjects
    if any(subject < 1 or subject > 54 for subject in chosen):
        parser.error("subjects must be 1..54")
    preflight_rearm()
    for session in args.sessions:
        for subject in chosen:
            destination = OUTPUT / f"session{session}" / f"sub-{subject:02d}.json"
            if destination.exists() and not args.overwrite:
                continue
            result = run_case(session, subject)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"S{session} sub-{subject:02d} pooled δ={result['source_pooled_offset'].get('delta')} "
                  f"FP={result['evaluation'].get('pooled_offset', {}).get('future_fp_per_min')}", flush=True)
    if args.all_subjects and args.sessions == [1, 2]:
        cases = [json.loads((OUTPUT / f"session{session}" / f"sub-{subject:02d}.json")
                                 .read_text(encoding="utf-8"))
                 for session in (1, 2) for subject in range(1, 55)]
        summary = summarize(cases)
        (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        print(f"saved {OUTPUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
