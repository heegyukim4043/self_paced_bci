"""Count-matched random deletion control for E3 G1 output-censoring audit.

Uses actual attack hit only; it is not a replacement for the shift-null excess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_e3_g1_failure_attribution_20261009 import attack_family  # noqa: E402

SEED = 20261009
N_RANDOM = 200
ARMS = ("drop_seam", "drop_cue", "drop_union")


def one(original: dict, detector: dict, stream_root: Path) -> dict:
    subject, session = int(original["subject"]), int(original["session"])
    path = stream_root / f"session{session}/sub-{subject:02d}.npz"
    with np.load(path, allow_pickle=False) as stream:
        events = stream["events"].copy()
        horizon = len(stream["signal"]) / 250.0
    observed = np.sort(np.asarray(detector["detection_samples_available_at"],
                                  dtype=float) / 250.0)
    idle = events[:, 2:4] / 250.0
    idle[:, 0] = np.maximum(idle[:, 0], 10.0)
    idle[:, 1] = np.minimum(idle[:, 1], horizon - 10.0)
    idle = idle[idle[:, 1] > idle[:, 0]]
    onsets = events[:, 6] / 250.0
    onsets = onsets[(onsets >= 10.0) & (onsets + 2.0 <= horizon - 10.0)]
    full_hit = attack_family(observed, onsets, idle, horizon)
    if not np.isclose(full_hit, original["arms"]["full"]["actual_max_hit"],
                      atol=1e-12, rtol=0):
        raise AssertionError("full attack not reproduced")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, subject, session]))
    result = {"subject": subject, "session": session,
              "total_outputs": len(observed), "full_actual_max_hit": full_hit,
              "arms": {}}
    for arm in ARMS:
        selected = original["arms"][arm]
        removed = len(observed) - selected["retained_outputs"]
        selective_reduction = full_hit - selected["actual_max_hit"]
        random_reductions = np.zeros(N_RANDOM)
        if removed:
            for draw in range(N_RANDOM):
                keep = np.ones(len(observed), dtype=bool)
                keep[rng.choice(len(observed), removed, replace=False)] = False
                random_reductions[draw] = full_hit - attack_family(
                    observed[keep], onsets, idle, horizon)
        result["arms"][arm] = {
            "removed_outputs": removed,
            "selective_actual_hit_reduction": selective_reduction,
            "random_deletion_reduction_median": float(np.median(random_reductions)),
            "random_deletion_reduction_q95": float(np.quantile(random_reductions, .95)),
            "selective_reduction_percentile_among_random": float(
                np.mean(random_reductions <= selective_reduction)) if removed else None,
            "selective_exceeds_random_q95": bool(
                selective_reduction > np.quantile(random_reductions, .95)) if removed else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1_failure_attribution_20261009.json"))
    parser.add_argument("--detector", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1b_splice_detector_matched_window.json"))
    parser.add_argument("--stream-root", type=Path, default=Path(
        "data/interim/lee2019_v2_random_precue_balanced_20261008"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1_random_deletion_20261009.json"))
    args = parser.parse_args()
    origin = json.loads(args.attribution.read_text(encoding="utf-8"))["rows"]
    detector = json.loads(args.detector.read_text(encoding="utf-8"))["rows"]
    by_key = {(r["subject"], r["session"]): r for r in detector}
    if len(origin) != 108 or len(by_key) != 108:
        raise AssertionError("expected 108 paired rows")
    rows = [one(r, by_key[(r["subject"], r["session"])], args.stream_root)
            for r in origin]
    summary = {}
    for session in (1, 2):
        group = [r for r in rows if r["session"] == session]
        summary[str(session)] = {}
        for arm in ARMS:
            used = [r["arms"][arm] for r in group if r["arms"][arm]["removed_outputs"]]
            summary[str(session)][arm] = {
                "subjects_with_removed_outputs": len(used),
                "median_selective_actual_reduction": float(np.median([
                    r["selective_actual_hit_reduction"] for r in used])) if used else 0.0,
                "median_random_deletion_reduction": float(np.median([
                    r["random_deletion_reduction_median"] for r in used])) if used else 0.0,
                "median_selective_percentile": float(np.median([
                    r["selective_reduction_percentile_among_random"] for r in used])) if used else None,
                "subjects_selective_above_random_q95": sum(
                    r["selective_exceeds_random_q95"] for r in used),
            }
    result = {"scope": "post-hoc count-matched random deletion; actual hit only",
              "n_random_draws_per_subject_arm": N_RANDOM, "seed": SEED,
              "warning": "No shift-null excess is recomputed here; percentiles are descriptive, not p-values",
              "session_summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
