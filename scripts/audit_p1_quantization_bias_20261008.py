"""Correct P1 FP diagnostics for the attained *calibration* alarm rate.

This is an exploratory diagnostic on Lee2019 pseudo-uncued replay. The
evaluated stream failed the G1 seam gate; outputs are not BCI performance.
Technical cells are repeated within subject, not independent subjects.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from scripts.decompose_calibration_error_20261008 import (
    ROOT, STEP, alarms_exact, load_case, source_rest_mask,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from uncued_mi.decision import guard_local_moving_average

MODELS = ("riemannian", "cnn-gru")
KINDS = ("K_A", "K_F", "K_S")
LENGTHS = (28, 36)
TARGETS = (2., 4.)


def effective_error(row: dict, reference_rate: float | None = None) -> float:
    """Signed log FP error against the rate the calibrator actually set."""
    rate = (float(row["calibration_fp_per_min"]) if reference_rate is None
            else float(reference_rate))
    if rate <= 0:
        raise ValueError("attained calibration rate must be positive")
    exposure_min = float(row["evaluation_rest_seconds"]) / 60.
    return float(np.log((int(row["future_fp_count"]) + .5) / (exposure_min * rate)))


def bootstrap_median(values: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(values), (draws, len(values)))
    return np.quantile(np.median(values[ix], axis=1), [.025, .975]).tolist()


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, draws: int, seed: int) -> dict:
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(draws):
        ix = rng.integers(0, len(x), len(x))
        result = float(spearmanr(x[ix], y[ix]).statistic)
        if np.isfinite(result):
            sampled.append(result)
    return {"rho": rho, "ci95": np.quantile(sampled, [.025, .975]).tolist(),
            "bootstrap_valid_draws": len(sampled)}


def load_rows(root: Path) -> dict:
    grouped = defaultdict(list)
    for model in MODELS:
        for session in (1, 2):
            for subject in range(1, 55):
                path = root / model / f"session{session}" / f"sub-{subject:02d}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                for row in payload["rows"]:
                    if int(row["length_seconds"]) not in LENGTHS:
                        continue
                    key = (model, session, subject, int(row["length_seconds"]),
                           float(row["target_fp_per_min"]), row["calibration_kind"])
                    grouped[key].append(row)
    return grouped


def cell_matrix(grouped: dict, model: str, session: int, length: int,
                target: float, kind: str) -> tuple[np.ndarray, np.ndarray]:
    values, rates = [], []
    for subject in range(1, 55):
        members = grouped[(model, session, subject, length, target, kind)]
        if len(members) != 9:
            raise AssertionError(f"expected 9 cells: {model=} {session=} {subject=} {kind=}")
        members = sorted(members, key=lambda r: (r["future_fold"], r["heldout_rest_source"]))
        if [(r["future_fold"], r["heldout_rest_source"]) for r in members] != [
            (fold, source) for fold in (2, 3, 4) for source in (0, 1, 2)
        ]:
            raise AssertionError("non-identical technical-cell grid")
        values.append([effective_error(r) for r in members])
        rates.append([r["calibration_fp_per_min"] for r in members])
    return np.asarray(values), np.asarray(rates)


def summarize_matrix(matrix: np.ndarray, draws: int, seed: int) -> dict:
    subject_mean = matrix.mean(axis=1)
    cell_sd = float(matrix.std(ddof=0))
    between_sd = float(subject_mean.std(ddof=0))
    within_sd = float(np.sqrt(matrix.var(axis=1, ddof=0).mean()))
    if not np.isclose(cell_sd ** 2, between_sd ** 2 + within_sd ** 2,
                      rtol=1e-10, atol=1e-10):
        raise AssertionError("law of total variance failed")
    return {
        "median_subject_mean_signed_log_error": float(np.median(subject_mean)),
        "median_subject_mean_signed_log_error_ci95": bootstrap_median(subject_mean, draws, seed),
        "cell_sd": cell_sd, "between_subject_sd": between_sd,
        "within_subject_cell_sd": within_sd,
    }


def make_load_args() -> SimpleNamespace:
    return SimpleNamespace(
        replay_root=ROOT / "data/interim/lee2019_revalidation_chronological",
        riem_root=ROOT / "data/interim/lee2019_revalidation_riemannian_scores",
        cnn_root=ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache",
        rest_score_root=ROOT / "data/interim/p1_fixed_target_rest_scores_20261008",
    )


def constant_threshold_control(grouped: dict, draws: int) -> list[dict]:
    """Median source-donor threshold under each frozen target LOSO model.

    The other targets' K_A thresholds are not safe: those targets' LOSO models
    were trained with the current evaluation subject included in their source.
    """
    output = []
    args = make_load_args()
    for model in MODELS:
        for session in (1, 2):
            for target in TARGETS:
                length = 28
                baseline_errors, baseline_fp = [], []
                anchor_errors = []
                rate = round(length * target / 60.) * 60. / length
                for subject in range(1, 55):
                    donor_rows = grouped[(model, session, subject, length, target, "K_X")]
                    donor_thresholds = defaultdict(set)
                    for row in donor_rows:
                        donor_thresholds[int(row["donor_subject"])].add(float(row["threshold"]))
                    if len(donor_thresholds) != 53 or subject in donor_thresholds \
                            or any(len(values) != 1 for values in donor_thresholds.values()):
                        raise AssertionError("K0 donor thresholds are not target-LOSO/source-only")
                    threshold = float(np.median([next(iter(values))
                                                 for values in donor_thresholds.values()]))
                    data = load_case(model, session, subject, args)
                    anchor = data["anchor"]
                    z = ewma_z(data["raw"], float(anchor.mean()),
                               float(anchor.var(ddof=1)), data["allowed"])
                    ma = guard_local_moving_average(z, data["allowed"], ma_steps=5)
                    alarm = alarms_exact(ma, threshold, data["allowed"])
                    errors, fps = [], []
                    for fold in (2, 3, 4):
                        masks = [source_rest_mask(
                            data["events"], data["source_ids"], data["starts"],
                            data["times"], data["allowed"], fold, source,
                        ) for source in (0, 1, 2)]
                        for source in (0, 1, 2):
                            mask = masks[(source + 1) % 3] | masks[(source + 2) % 3]
                            exposure = mask.sum() * STEP / 60.
                            count = int(np.count_nonzero(alarm & mask))
                            errors.append(np.log((count + .5) / (exposure * rate)))
                            fps.append(count / exposure)
                    baseline_errors.append(np.asarray(errors))
                    baseline_fp.append(np.asarray(fps))
                    members = grouped[(model, session, subject, length, target, "K_A")]
                    anchor_errors.append(np.mean([effective_error(r) for r in members]))
                baseline = np.vstack(baseline_errors)
                fp = np.vstack(baseline_fp)
                anchor = np.asarray(anchor_errors)
                # Pair on subject, not on the 9 repeated evaluation cells.
                delta_abs = np.abs(baseline).mean(axis=1) - np.asarray([
                    np.mean([abs(effective_error(r)) for r in grouped[
                        (model, session, subject, length, target, "K_A")
                    ]]) for subject in range(1, 55)
                ])
                output.append({
                    "model": model, "session": session, "length_seconds": length,
                    "target_fp_per_min": target, "reference_rate_fp_per_min": rate,
                    "median_k0_subject_mean_fp_per_min": float(np.median(fp.mean(axis=1))),
                    "median_k0_subject_mean_signed_log_error": float(np.median(baseline.mean(axis=1))),
                    "median_ka_subject_mean_signed_log_error": float(np.median(anchor)),
                    "median_paired_k0_minus_ka_mean_abs_log_error": float(np.median(delta_abs)),
                    "median_paired_k0_minus_ka_ci95": bootstrap_median(
                        delta_abs, draws, 20261008 + session * 100 + int(target * 10)
                        + MODELS.index(model)),
                    "k0_subject_mean_fp_q90": float(np.quantile(fp.mean(axis=1), .9)),
                })
    return output


def run(input_root: Path, draws: int, include_k0: bool) -> dict:
    grouped = load_rows(input_root)
    summaries = []
    matrices = {}
    for model in MODELS:
        for session in (1, 2):
            for length in LENGTHS:
                for target in TARGETS:
                    for kind in KINDS:
                        matrix, rates = cell_matrix(grouped, model, session, length, target, kind)
                        count = round(length * target / 60.)
                        expected_rate = count * 60. / length
                        if not np.allclose(rates, expected_rate, atol=1e-12, rtol=0):
                            raise AssertionError(f"non-fixed calibration count/rate: {model=} {session=} {length=} {target=} {kind=}")
                        matrices[(model, session, length, target, kind)] = matrix
                        summaries.append({
                            "model": model, "session": session, "length_seconds": length,
                            "target_fp_per_min": target, "kind": kind,
                            "technical_cells": int(matrix.size),
                            "attained_calibration_alarm_count": count,
                            "attained_calibration_rate_fp_per_min": expected_rate,
                            **summarize_matrix(matrix, draws, 20261008 + session * 100
                                               + length + int(target * 10)
                                               + KINDS.index(kind) + 1000 * MODELS.index(model)),
                        })
    paired = []
    for model in MODELS:
        for session in (1, 2):
            for length in LENGTHS:
                for target in TARGETS:
                    a = matrices[(model, session, length, target, "K_A")].mean(axis=1)
                    f = matrices[(model, session, length, target, "K_F")].mean(axis=1)
                    difference = a - f
                    paired.append({
                        "model": model, "session": session, "length_seconds": length,
                        "target_fp_per_min": target,
                        "median_paired_signed_ka_minus_kf": float(np.median(difference)),
                        "median_paired_signed_ka_minus_kf_ci95": bootstrap_median(
                            difference, draws, 20262008 + session * 100 + length
                            + int(target * 10) + 1000 * MODELS.index(model)),
                    })
    correlations = []
    for model in MODELS:
        for length in LENGTHS:
            for target in TARGETS:
                first = matrices[(model, 1, length, target, "K_A")].mean(axis=1)
                second = matrices[(model, 2, length, target, "K_A")].mean(axis=1)
                correlations.append({"model": model, "length_seconds": length,
                                     "target_fp_per_min": target, "comparison": "K_A_S1_vs_K_A_S2",
                                     **bootstrap_spearman(first, second, draws,
                                                          20261008 + length + int(target * 10)
                                                          + 1000 * MODELS.index(model))})
                for session in (1, 2):
                    a = matrices[(model, session, length, target, "K_A")].mean(axis=1)
                    s = matrices[(model, session, length, target, "K_S")].mean(axis=1)
                    correlations.append({"model": model, "session": session,
                                         "length_seconds": length, "target_fp_per_min": target,
                                         "comparison": "K_A_vs_K_S",
                                         **bootstrap_spearman(a, s, draws,
                                                              20261108 + session * 100 + length
                                                              + int(target * 10)
                                                              + 1000 * MODELS.index(model))})
    return {
        "status": "exploratory pseudo-uncued FP audit; G1 failed",
        "error_reference": "calibration_alarm_count * 60 / calibration_seconds",
        "offset": "evaluation count +0.5 in log error",
        "variance": "descriptive law-of-total-variance on 54 subject x 9 technical cells",
        "summaries": summaries, "paired_signed_ka_kf": paired,
        "subject_correlations": correlations,
        "k0": constant_threshold_control(grouped, draws) if include_k0 else [],
        "k0_scope": "median of 53 source-donor K_X z thresholds under frozen target LOSO model; target's own anchor mean/variance retained; donor source-training exposure caveat",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "data/processed/calibration_error_decomposition_20261008/rows")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/calibration_error_decomposition_20261008/quantization_bias_audit.json")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--skip-k0", action="store_true")
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("at least 100 bootstrap draws required")
    result = run(args.input_root, args.draws, not args.skip_k0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}: {len(result['summaries'])} source cells, "
          f"{len(result['paired_signed_ka_kf'])} paired contrasts, "
          f"{len(result['k0'])} K0 controls")


if __name__ == "__main__":
    main()
