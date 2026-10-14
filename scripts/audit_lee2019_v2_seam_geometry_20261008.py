"""Metadata-only geometry check for E3 decoys; does not inspect EEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def auc(values_true: np.ndarray, values_decoy: np.ndarray) -> float:
    """Probability a true-start gap exceeds a decoy gap (ties = 0.5)."""
    combined = np.r_[values_true, values_decoy]
    ranks = rankdata(combined)
    n_true, n_decoy = len(values_true), len(values_decoy)
    return float((ranks[:n_true].sum() - n_true * (n_true + 1) / 2) /
                 (n_true * n_decoy))


def audit(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        seams = saved["seams"]
        events = saved["events"]
        piece_lengths = saved["rest_piece_lengths"] / 250.0
    # The preceding gap is available at the candidate seam. The following gap
    # is a noncausal construction diagnostic only, not an onset-time attacker.
    preceding = np.diff(seams[:, 0]) / 250.0
    labels = seams[1:, 1]
    before_trial = preceding[labels == 1]
    before_decoy = preceding[labels == 0]
    following = np.diff(seams[:, 0]) / 250.0
    labels_before = seams[:-1, 1]
    after_trial = following[labels_before == 1]
    after_decoy = following[labels_before == 0]
    assert len(before_trial) in (len(events) - 1, len(events))
    assert len(after_trial) == len(events) - 1
    return {"file": str(path), "session": int(path.parent.name.replace("session", "")),
            "subject": int(path.stem.split("-")[1]),
            "decoy_count": int(np.sum(seams[:, 1] == 0)),
            "preceding_gap_auc_longer_means_trial": auc(before_trial, before_decoy),
            "following_gap_auc_noncausal": auc(after_trial, after_decoy),
            "preceding_trial_gap_median_s": float(np.median(before_trial)),
            "preceding_decoy_gap_median_s": float(np.median(before_decoy)),
            "rest_piece_median_s": float(np.median(piece_lengths)),
            "trial_duration_median_s": float(np.median(
                (events[:, 7] - events[:, 4]) / 250.0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("data/interim/lee2019_v2_random_precue_balanced_20261008"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/simulation_redesign_20261008/e3_metadata_geometry.json"))
    args = parser.parse_args()
    paths = sorted(args.root.glob("session*/sub-*.npz"))
    if len(paths) != 108:
        raise AssertionError(f"expected 108, found {len(paths)}")
    rows = [audit(path) for path in paths]
    result = {"scope": "metadata-only paired geometry, not EEG seam detection",
              "preceding_gap_causal": True,
              "following_gap_causal_at_onset": False,
              "n_subject_sessions": len(rows),
              "session_summary": {}, "rows": rows}
    for session in (1, 2):
        group = [row for row in rows if row["session"] == session]
        result["session_summary"][str(session)] = {
            "median_preceding_gap_auc": float(np.median([
                row["preceding_gap_auc_longer_means_trial"] for row in group])),
            "median_noncausal_following_gap_auc": float(np.median([
                row["following_gap_auc_noncausal"] for row in group])),
            "median_rest_piece_s": float(np.median([
                row["rest_piece_median_s"] for row in group])),
            "median_trial_duration_s": float(np.median([
                row["trial_duration_median_s"] for row in group])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
