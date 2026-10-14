"""Post-G1 exploratory censoring audit of EEG splice-detector outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from simulate_composed_splice_attack_20261008 import composed_attack  # noqa: E402

WAITS = (1.0, 2.0, 3.5)
ARMS = ("full", "drop_seam", "drop_cue", "drop_union")
SEED = 20261008
N_SHIFTS = 1000
N_BOOT = 2000


def in_post_interval(times: np.ndarray, references: np.ndarray,
                     width: float) -> np.ndarray:
    """True if output is in [reference, reference+width] for any reference."""
    if not len(times) or not len(references):
        return np.zeros(len(times), dtype=bool)
    lag = times[:, None] - references[None, :]
    return np.any((lag >= 0.0) & (lag <= width + 1e-12), axis=1)


def attack_family(observed: np.ndarray, onsets: np.ndarray,
                  idle: np.ndarray, horizon: float) -> float:
    return max(composed_attack(observed, onsets, idle, horizon,
                               wait_s=wait)[0] for wait in WAITS)


def one(detector: dict, stream_root: Path, original: dict) -> dict:
    subject, session = int(detector["subject"]), int(detector["session"])
    path = stream_root / f"session{session}/sub-{subject:02d}.npz"
    with np.load(path, allow_pickle=False) as stream:
        events = stream["events"].copy()
        seams = stream["seams"][:, 0] / 250.0
        horizon = len(stream["signal"]) / 250.0
    cue = events[:, 5] / 250.0
    outputs = np.sort(np.asarray(detector["detection_samples_available_at"],
                                 dtype=float) / 250.0)
    seam_local = in_post_interval(outputs, seams, .4)
    cue_local = in_post_interval(outputs, cue, .7)
    idle = events[:, 2:4] / 250.0
    idle[:, 0] = np.maximum(idle[:, 0], 10.0)
    idle[:, 1] = np.minimum(idle[:, 1], horizon - 10.0)
    idle = idle[idle[:, 1] > idle[:, 0]]
    onsets = events[:, 6] / 250.0
    onsets = onsets[(onsets >= 10.0) & (onsets + 2.0 <= horizon - 10.0)]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, subject, session, 6]))
    offsets = rng.uniform(10.0, horizon - 10.0, N_SHIFTS)
    keep = {
        "full": np.ones(len(outputs), dtype=bool),
        "drop_seam": ~seam_local,
        "drop_cue": ~cue_local,
        "drop_union": ~(seam_local | cue_local),
    }
    arms = {}
    for arm in ARMS:
        observed = outputs[keep[arm]]
        actual = attack_family(observed, onsets, idle, horizon)
        null = np.asarray([
            attack_family(np.sort((observed + offset) % horizon),
                          onsets, idle, horizon)
            for offset in offsets], dtype=float)
        null_median = float(np.median(null))
        arms[arm] = {"retained_outputs": len(observed),
                     "actual_max_hit": float(actual),
                     "shift_null_median_max_hit": null_median,
                     "excess": float(actual - null_median)}
    if not np.isclose(arms["full"]["excess"],
                      original["paired_max_hit_excess"], atol=1e-12, rtol=0):
        raise AssertionError(f"full G1c replay mismatch: {subject}/{session}")
    return {"subject": subject, "session": session,
            "outputs_total": len(outputs),
            "outputs_seam_local": int(seam_local.sum()),
            "outputs_cue_local": int(cue_local.sum()),
            "outputs_overlap": int(np.sum(seam_local & cue_local)),
            "arms": arms}


def session_summary(rows: list[dict]) -> dict:
    summary = {}
    for session in (1, 2):
        group = sorted((r for r in rows if r["session"] == session),
                       key=lambda r: r["subject"])
        if len(group) != 54:
            raise AssertionError("subject-level full cohort required")
        rng = np.random.default_rng(np.random.SeedSequence([SEED, session, 9]))
        draws = rng.integers(0, 54, size=(N_BOOT, 54))
        full = np.asarray([r["arms"]["full"]["excess"] for r in group])
        values = {}
        for arm in ARMS:
            measure = np.asarray([r["arms"][arm]["excess"] for r in group])
            contrast = full - measure
            interval = np.quantile(np.median(contrast[draws], axis=1),
                                   (.025, .975))
            values[arm] = {
                "median_excess": float(np.median(measure)),
                "median_paired_full_minus_arm": float(np.median(contrast)),
                "paired_median_bootstrap_ci95": [float(x) for x in interval],
                "median_retained_outputs": float(np.median([
                    r["arms"][arm]["retained_outputs"] for r in group])),
            }
        summary[str(session)] = {
            "n_subjects": 54,
            "median_outputs_total": float(np.median([r["outputs_total"] for r in group])),
            "median_outputs_seam_local": float(np.median([
                r["outputs_seam_local"] for r in group])),
            "median_outputs_cue_local": float(np.median([
                r["outputs_cue_local"] for r in group])),
            "median_outputs_overlap": float(np.median([
                r["outputs_overlap"] for r in group])),
            "arms": values,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1b_splice_detector_matched_window.json"))
    parser.add_argument("--original", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1c_composed_attack.json"))
    parser.add_argument("--stream-root", type=Path, default=Path(
        "data/interim/lee2019_v2_random_precue_balanced_20261008"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/simulation_redesign_20261008/e3_g1_failure_attribution_20261009.json"))
    args = parser.parse_args()
    detector = json.loads(args.detector.read_text(encoding="utf-8"))["rows"]
    original = json.loads(args.original.read_text(encoding="utf-8"))["rows"]
    by_key = {(r["subject"], r["session"]): r for r in original}
    if len(detector) != 108 or len(by_key) != 108:
        raise AssertionError("expected paired 108 files")
    rows = [one(r, args.stream_root, by_key[(r["subject"], r["session"])])
            for r in detector]
    result = {"status": "post-G1 exploratory source-attribution; not a new gate pass",
              "date": "2026-10-09", "n_subject_sessions": len(rows),
              "seam_window_after_s": [0.0, .4],
              "cue_window_after_s": [0.0, .7],
              "attack_waits_s": WAITS,
              "null": "surviving outputs circular-shifted with the same 1000 offsets per file",
              "bootstrap": "2000 subject resamples, paired two-sided 95% intervals by session",
              "full_original_row_reproduced": True,
              "interpretation_limit": "Removing detections may also remove gap-attack interference; no causal attribution or MI-specificity claim",
              "session_summary": session_summary(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
