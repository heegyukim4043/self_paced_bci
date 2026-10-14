"""P1 diagnostic calibration-source comparison on locked Lee2019 scores.

This is a cue-paced pseudo-uncued simulation analysis, not a deployment claim.
K_F uses future rest labels and is an oracle *diagnostic*. K_X donors belong to
the target LOSO model's source cohort, so K_X is source-to-unseen-target
transfer, not an identified pure between-person effect. Full episode and
cluster-bootstrap analyses are separate later stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms

try:
    from numba import njit
except ImportError:  # The GPU training environment does not include Numba.
    njit = None

STEP = .1
SFREQ = 250.
LENGTHS = (14, 28, 36)
TARGETS = (2., 4.)
MODELS = ("riemannian", "cnn-gru")


def _alarm_loop(statistic, threshold, allowed, dwell_steps, refractory_steps):
    """Byte-for-byte equivalent state machine to guard_rearmed_alarms."""
    alarms = np.zeros(len(statistic), dtype=np.bool_)
    armed = False
    count = 0
    blocked_until = -1
    for index in range(len(statistic)):
        value = statistic[index]
        if not allowed[index] or not np.isfinite(value):
            armed = False
            count = 0
            continue
        if not armed:
            if value < threshold:
                armed = True
            continue
        count = count + 1 if value >= threshold else 0
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
    return alarms


_fast_alarm_loop = njit(cache=True)(_alarm_loop) if njit is not None else _alarm_loop


def alarms_exact(statistic, threshold, allowed) -> np.ndarray:
    """Use optional JIT only for the unchanged 3-sample/10-sample rule."""
    return _fast_alarm_loop(
        np.asarray(statistic, dtype=np.float64), float(threshold),
        np.asarray(allowed, dtype=np.bool_), 3, 10,
    )


def source_rest_mask(
    events: np.ndarray, source_ids: np.ndarray, starts: np.ndarray,
    times: np.ndarray, allowed: np.ndarray, fold: int, source: int,
) -> np.ndarray:
    result = np.zeros(len(times), dtype=bool)
    for index in range(fold * 10, (fold + 1) * 10):
        if int(source_ids[index]) == source:
            result |= ((starts >= events[index, 2] / SFREQ)
                       & (times <= events[index, 3] / SFREQ))
    return result & allowed


def compress_selected_ma(ma: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain time-order and reset state across omitted stream gaps.

    Ten invalid samples between fragments exceed the 1 s refractory. MA is
    computed on the original full stream *before* this compression.
    """
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("empty calibration segment")
    values = []
    allowed = []
    previous = indices[0] - 1
    for index in indices:
        if index != previous + 1:
            values.extend([np.nan] * 11)
            allowed.extend([False] * 11)
        values.append(float(ma[index]))
        allowed.append(True)
        previous = index
    return np.asarray(values), np.asarray(allowed, dtype=bool)


def tune_pre_rest(raw: np.ndarray, mean: float, var: float, length: int, target: float) -> dict:
    steps = int(round(length / STEP))
    if len(raw) < steps:
        raise ValueError(f"calibration source has {len(raw)} < {steps} steps")
    z = ewma_z(raw[:steps], mean, var, np.ones(steps, dtype=bool))
    ma = guard_local_moving_average(z, np.ones(steps, dtype=bool), ma_steps=5)
    return _tune_ma(ma, np.ones(steps, dtype=bool), length, target)


def _tune_ma(ma: np.ndarray, allowed: np.ndarray, length: int, target: float) -> dict:
    """Same exact partition/tie rule as tune_guard_rearmed_anchor."""
    finite = np.unique(ma[np.isfinite(ma)])
    if not len(finite):
        raise ValueError("calibration contains no valid MA statistic")
    candidates = np.unique(np.r_[
        np.nextafter(float(finite[0]), -np.inf), finite,
        np.nextafter(float(finite[-1]), np.inf),
    ])
    counts = np.asarray([
        int(alarms_exact(ma, float(thr), allowed).sum())
        for thr in candidates
    ], dtype=int)
    rates = counts * 60. / length
    errors = np.abs(rates - target)
    best = np.flatnonzero(np.isclose(errors, errors.min(), rtol=0, atol=1e-12))[-1]
    return {
        "threshold": float(candidates[best]),
        "calibration_alarm_count": int(counts[best]),
        "calibration_fp_per_min": float(rates[best]),
        "calibration_attainable_counts": sorted(set(counts.tolist())),
    }


def tune_future_rest(
    stream_ma: np.ndarray, eligible: np.ndarray, length: int, target: float,
) -> dict:
    """Exact count sweep on selected future-rest windows, with gap resets."""
    steps = int(round(length / STEP))
    indices = np.flatnonzero(eligible)
    if len(indices) < steps:
        raise ValueError(f"future rest has {len(indices)} < {steps} allowed steps")
    selected = np.zeros(len(eligible), dtype=bool)
    selected[indices[:steps]] = True
    ma, allowed = compress_selected_ma(stream_ma, selected)
    return _tune_ma(ma, allowed, length, target)


def load_case(model: str, session: int, subject: int, args: argparse.Namespace) -> dict:
    stem = Path(f"session{session}") / f"sub-{subject:02d}.npz"
    original_root = args.riem_root if model == "riemannian" else args.cnn_root
    with np.load(original_root / stem, allow_pickle=False) as original, \
         np.load(args.rest_score_root / model / stem, allow_pickle=False) as rest, \
         np.load(args.replay_root / stem, allow_pickle=False) as replay:
        anchor_key = "anchor_raw_scores" if model == "riemannian" else "anchor_scores"
        data = {
            "raw": np.asarray(original["raw_scores"], dtype=float),
            "anchor": np.asarray(original[anchor_key], dtype=float),
            "times": np.asarray(original["times"], dtype=float),
            "starts": np.asarray(original["starts"], dtype=float),
            "allowed": np.asarray(original["allowed"], dtype=bool),
            "events": np.asarray(original["events"], dtype=np.int64),
            "source_ids": np.asarray(replay["rest_source_ids"], dtype=np.int64),
            "other_anchor": np.asarray(rest["other_session_anchor_scores"], dtype=float),
            "donor_scores": np.asarray(rest["donor_anchor_scores"], dtype=float),
            "donor_subjects": np.asarray(rest["donor_subjects"], dtype=int),
            "donor_mask": np.asarray(rest["donor_out_of_training_mask"], dtype=bool),
        }
    if not np.array_equal(data["events"], replay_events(args.replay_root / stem)):
        raise AssertionError(f"events differ: {stem}")
    if (len(data["anchor"]) != 571 or data["donor_scores"].shape != (53, 571)
            or data["source_ids"].shape != (50,)):
        raise AssertionError(f"unexpected score geometry: {stem}")
    return data


def replay_events(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as replay:
        return np.asarray(replay["events"], dtype=np.int64)


def case_rows(model: str, session: int, subject: int, args: argparse.Namespace) -> list[dict]:
    data = load_case(model, session, subject, args)
    anchor = data["anchor"]
    mean, var = float(anchor.mean()), float(anchor.var(ddof=1))
    z = ewma_z(data["raw"], mean, var, data["allowed"])
    ma = guard_local_moving_average(z, data["allowed"], ma_steps=5)
    donor_mask = data["donor_mask"]
    if model == "cnn-gru" and int(donor_mask.sum()) != 291:
        raise AssertionError("CNN donor clean-rest support changed")
    if model == "riemannian" and not donor_mask.all():
        raise AssertionError("Riemannian donor rest unexpectedly masked")

    # These three sources do not depend on future fold/source. K_F does.
    calibrations = {}
    for length in args.lengths:
        for target in args.targets:
            calibrations[("K_A", length, target, 0)] = tune_pre_rest(
                anchor, mean, var, length, target,
            )
            calibrations[("K_S", length, target, 0)] = tune_pre_rest(
                data["other_anchor"], mean, var, length, target,
            )
            if length * 10 <= int(donor_mask.sum()):
                for donor_index, donor in enumerate(data["donor_subjects"]):
                    calibrations[("K_X", length, target, int(donor))] = tune_pre_rest(
                        data["donor_scores"][donor_index, donor_mask],
                        mean, var, length, target,
                    )

    alarm_cache = {}

    def alarms_for(threshold: float) -> np.ndarray:
        if threshold not in alarm_cache:
            alarm_cache[threshold] = alarms_exact(ma, threshold, data["allowed"])
        return alarm_cache[threshold]

    rows = []
    for fold in (2, 3, 4):
        source_masks = [
            source_rest_mask(data["events"], data["source_ids"], data["starts"],
                             data["times"], data["allowed"], fold, source)
            for source in (0, 1, 2)
        ]
        for source in (0, 1, 2):
            cal_mask = source_masks[source]
            eval_mask = source_masks[(source + 1) % 3] | source_masks[(source + 2) % 3]
            exposure_min = float(eval_mask.sum() * STEP / 60.)
            if exposure_min <= 0 or np.any(cal_mask & eval_mask):
                raise AssertionError("calibration/evaluation rest overlap or empty exposure")
            for length in args.lengths:
                for target in args.targets:
                    local = dict(calibrations)
                    local[("K_F", length, target, 0)] = tune_future_rest(
                        ma, cal_mask, length, target,
                    )
                    for (kind, source_length, source_target, donor), point in local.items():
                        if source_length != length or source_target != target:
                            continue
                        alarms = alarms_for(point["threshold"])
                        count = int(np.count_nonzero(alarms & eval_mask))
                        fp = count / exposure_min
                        log_error = float(np.log((count + .5) / (exposure_min * target)))
                        rows.append({
                            "model": model, "session": session, "subject": subject,
                            "future_fold": fold, "heldout_rest_source": source,
                            "calibration_kind": kind, "donor_subject": donor or None,
                            "length_seconds": length, "target_fp_per_min": target,
                            "calibration_resolution_valid": not (length == 14 and target == 2.),
                            **point,
                            "evaluation_rest_seconds": exposure_min * 60.,
                            "future_fp_count": count,
                            "future_fp_per_min": fp,
                            "signed_log_fp_error": log_error,
                            "abs_log_fp_error": abs(log_error),
                            "within_future_target_band": bool(
                                abs(fp - target) <= (.5 if target == 2. else 1.)
                            ),
                        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    subjects = parser.add_mutually_exclusive_group(required=True)
    subjects.add_argument("--subjects", nargs="+", type=int)
    subjects.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--sessions", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--lengths", nargs="+", type=int, choices=LENGTHS, default=list(LENGTHS))
    parser.add_argument("--targets", nargs="+", type=float, choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay-root", type=Path, default=ROOT / "data/interim/lee2019_revalidation_chronological")
    parser.add_argument("--riem-root", type=Path, default=ROOT / "data/interim/lee2019_revalidation_riemannian_scores")
    parser.add_argument("--cnn-root", type=Path, default=ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache")
    parser.add_argument("--rest-score-root", type=Path, default=ROOT / "data/interim/p1_fixed_target_rest_scores_20261008")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/calibration_error_decomposition_20261008/rows")
    args = parser.parse_args()
    args.subjects = list(range(1, 55)) if args.all_subjects else args.subjects
    if any(s < 1 or s > 54 for s in args.subjects):
        parser.error("subjects must be 1..54")
    return args


def main() -> None:
    args = parse_args()
    for session in args.sessions:
        for subject in args.subjects:
            for model in args.models:
                destination = args.output_root / model / f"session{session}" / f"sub-{subject:02d}.json"
                if destination.exists() and not args.overwrite:
                    print(f"skip existing {destination}", flush=True)
                    continue
                rows = case_rows(model, session, subject, args)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({
                    "status": "P1 exploratory FP-error rows; no G1 pass or confirmatory claim",
                    "model": model, "session": session, "subject": subject,
                    "rows": rows,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"saved {destination}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
