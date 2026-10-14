"""Strict-support tier-0 task-window audit on native-continuous Lee EEG.

This is an exploratory window-ranking diagnostic, not an onset detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.calibration import LocationScale, tier0_from_features  # noqa: E402

WINDOW_S = 2.0
TOL_S = 1e-3
FIRST_TRIAL = 70
LAST_TRIAL = 99
SEED = 20261009
N_BOOT = 2000


def full_support_mask(ends: np.ndarray, intervals: np.ndarray,
                      window_s: float = WINDOW_S) -> np.ndarray:
    """Select windows whose whole nominal power support lies in an interval."""
    ends = np.asarray(ends, dtype=float)
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    keep = np.zeros(len(ends), dtype=bool)
    for start, stop in intervals:
        keep |= ((ends - window_s >= start - TOL_S) &
                 (ends <= stop + TOL_S))
    return keep


def support_union_seconds(ends: np.ndarray,
                          window_s: float = WINDOW_S) -> float:
    """Unique EEG time covered by selected overlapping power windows."""
    ends = np.sort(np.asarray(ends, dtype=float))
    if not len(ends):
        return 0.0
    left, right = float(ends[0] - window_s), float(ends[0])
    total = 0.0
    for end in ends[1:]:
        start = float(end - window_s)
        if start > right:
            total += right - left
            left, right = start, float(end)
        else:
            right = max(right, float(end))
    return float(total + right - left)


def _following_iti(iti: np.ndarray, cue_s: float) -> np.ndarray:
    matches = iti[np.isclose(iti[:, 0], cue_s + 4.5,
                             atol=0.005, rtol=0.0)]
    if len(matches) > 1:
        raise AssertionError("multiple following task ITIs")
    return matches


def auc_or_none(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if not len(positive) or not len(negative):
        return None
    return float(roc_auc_score(
        np.r_[np.ones(len(positive)), np.zeros(len(negative))],
        np.r_[positive, negative],
    ))


def audit_one(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        subject, session = int(data["subject"]), int(data["session"])
        ends = np.asarray(data["continuous_times"], dtype=float)
        features = np.asarray(data["continuous_features"], dtype=float)
        pre = np.asarray(data["pre_features"], dtype=float)
        channels = tuple(str(x) for x in data["channels"])
        command = np.asarray(data["command_intervals"], dtype=float)
        fixation = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
    if len(cues) != 100 or len(command) != 100 or len(fixation) != 100:
        raise AssertionError(f"unexpected trial count: {subject}/{session}")
    if not (np.all(np.diff(ends) > 0) and np.all(np.diff(cues) > 0)):
        raise AssertionError("timestamps are not chronological")
    if features.shape[0] != len(ends) or pre.shape[1:] != features.shape[1:]:
        raise AssertionError("feature shape mismatch")
    if not all(name in channels for name in ("C3", "C4")):
        raise AssertionError("tier-0 C3/C4 channels unavailable")
    if not np.allclose(command[:, 0], cues + .5, atol=.005, rtol=0):
        raise AssertionError("command onset not cue + 0.5 s")
    model = LocationScale.fit(pre, axis=0)
    scores = tier0_from_features(features, channels, model)
    future_command = command[FIRST_TRIAL:LAST_TRIAL + 1]
    future_fixation = fixation[FIRST_TRIAL:LAST_TRIAL + 1]
    future_iti = np.concatenate([
        _following_iti(iti, float(cues[trial]))
        for trial in range(FIRST_TRIAL, LAST_TRIAL)
    ]).reshape(-1, 2)
    masks = {
        "mi": full_support_mask(ends, future_command),
        "iti": full_support_mask(ends, future_iti),
        "fixation": full_support_mask(ends, future_fixation),
    }
    if np.any(masks["mi"] & masks["iti"]):
        raise AssertionError("MI and ITI support overlap")
    values = {kind: scores[mask] for kind, mask in masks.items()}
    local = []
    for trial in range(FIRST_TRIAL, LAST_TRIAL):
        mi = scores[full_support_mask(ends, command[trial:trial + 1])]
        matched_iti = _following_iti(iti, float(cues[trial]))
        if not len(matched_iti):
            continue
        idle = scores[full_support_mask(ends, matched_iti)]
        if len(mi) and len(idle):
            local.append(float(np.median(mi) - np.median(idle)))
    return {
        "subject": subject, "session": session,
        "future_trial_ids": [FIRST_TRIAL, LAST_TRIAL],
        "future_task_iti_intervals": len(future_iti),
        "n_windows": {kind: int(mask.sum()) for kind, mask in masks.items()},
        "unique_support_seconds": {
            kind: support_union_seconds(ends[mask]) for kind, mask in masks.items()
        },
        "auc_mi_vs_iti": auc_or_none(values["mi"], values["iti"]),
        "auc_mi_vs_fixation": auc_or_none(values["mi"], values["fixation"]),
        "median_mi_minus_following_iti": (
            float(np.median(local)) if local else None),
        "n_local_pairs": len(local),
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for session in (1, 2):
        group = sorted((r for r in rows if r["session"] == session),
                       key=lambda r: r["subject"])
        if len(group) != 54 or len({r["subject"] for r in group}) != 54:
            raise AssertionError("expected 54 independent subjects per session")
        metrics = {}
        for key in ("auc_mi_vs_iti", "auc_mi_vs_fixation",
                    "median_mi_minus_following_iti"):
            vals = np.asarray([r[key] for r in group if r[key] is not None],
                              dtype=float)
            if not len(vals):
                metrics[key] = {"n_valid": 0, "median": None, "iqr": None,
                                "bootstrap_ci95": None}
                continue
            rng = np.random.default_rng(np.random.SeedSequence([SEED, session,
                                                                   len(metrics)]))
            draws = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
            ci = np.quantile(np.median(vals[draws], axis=1), (.025, .975))
            metrics[key] = {
                "n_valid": int(len(vals)),
                "median": float(np.median(vals)),
                "iqr": [float(x) for x in np.quantile(vals, (.25, .75))],
                "bootstrap_ci95": [float(x) for x in ci],
                "n_above_half": int(np.sum(vals > .5)) if key.startswith("auc_") else None,
            }
        result[str(session)] = {
            "n_subjects": 54,
            "median_n_windows": {kind: float(np.median([
                r["n_windows"][kind] for r in group]))
                for kind in ("mi", "iti", "fixation")},
            "median_unique_support_seconds": {kind: float(np.median([
                r["unique_support_seconds"][kind] for r in group]))
                for kind in ("mi", "iti", "fixation")},
            "median_n_local_pairs": float(np.median([
                r["n_local_pairs"] for r in group])),
            "metrics": metrics,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path(
        "data/interim/lee2019_tier0"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/original_continuous_lee_strict_windows_20261009.json"))
    args = parser.parse_args()
    paths = sorted(args.cache_root.glob("session*/sub-*_tier0_features.npz"))
    if len(paths) != 108:
        raise AssertionError(f"expected 108 original continuous caches, got {len(paths)}")
    rows = [audit_one(path) for path in paths]
    result = {
        "status": "exploratory native-continuous window-ranking diagnostic; not onset detection",
        "date": "2026-10-09",
        "score": "fixed tier-0 max C3/C4 mu-beta ERD, pre-train-rest normalization",
        "selection": "full nominal 2-s window support inside future trial 70-99 task interval; task ITI 70-98; fixation secondary",
        "filter_memory_limit": "causal IIR state reaches before nominal 2-s window; cue physiology not ruled out",
        "seed": SEED, "n_boot_subject": N_BOOT,
        "python_numpy_sklearn": [sys.version.split()[0], np.__version__, sklearn.__version__],
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
