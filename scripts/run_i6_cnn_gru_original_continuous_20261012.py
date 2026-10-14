"""Evaluate complete I6 stateful/window-reset caches with frozen I4 rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.run_i4_low_fp_original_continuous_20261010 import one, summarize, test_fast_shift_null
from uncued_mi.decision import guard_local_moving_average


def prepare_cache(path, tier0_path, subject, session, arm):
    with np.load(path, allow_pickle=False) as d:
        if (int(d["subject"]), int(d["session"])) != (subject, session):
            raise AssertionError("CNN-GRU cache identity mismatch")
        if tuple(d["seeds"].tolist()) != (20260729, 20260730, 20260731) or int(d["epochs"]) != 10:
            raise AssertionError("CNN-GRU training protocol mismatch")
        provenance = json.loads(str(d["provenance_json"]))
        if provenance["target_trials_used_for_fit"] or subject in provenance["source_subjects"] or len(provenance["source_subjects"]) != 53:
            raise AssertionError("CNN-GRU target training exposure")
        times = np.asarray(d["times"], float)
        pre_times = np.asarray(d["pre_times"], float)
        cues_model = np.asarray(d["cue_times"], float)
        raw = np.asarray(d["raw_scores" if arm == "stateful" else "reset_raw_scores"], float)
        pre_raw = np.asarray(d["pre_raw_scores" if arm == "stateful" else "pre_reset_raw_scores"], float)
    with np.load(tier0_path, allow_pickle=False) as d:
        tier_times = np.asarray(d["continuous_times"], float)
        cues = np.asarray(d["cue_samples"], float) / float(d["native_sfreq"])
        fixation = np.asarray(d["fixation_intervals"], float)
        iti = np.asarray(d["iti_intervals"], float).reshape(-1, 2)
    if len(times) != len(tier_times) or not np.allclose(times, tier_times, atol=.001, rtol=0):
        raise AssertionError("CNN-GRU/tier-0 time grid mismatch")
    # Scores are aligned index-for-index to the frozen tier-0 grid. Use the
    # latter's exact timestamps for trial masks; rounded 250-Hz score ends
    # can otherwise move one 0.1-s decision across a boundary.
    times = tier_times
    if not (np.all(np.diff(times) > 0) and np.all(np.diff(pre_times) > 0)):
        raise AssertionError("CNN-GRU time grid not strictly increasing")
    if not np.allclose(cues_model, cues, atol=.005, rtol=0):
        raise AssertionError("CNN-GRU/tier-0 cue mismatch")
    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(pre_raw))):
        raise AssertionError("CNN-GRU scores contain non-finite values")
    anchor = (pre_times >= 4.-.001) & (pre_times < 60.-.001)
    if anchor.sum() != 560 or len(raw) != len(times):
        raise AssertionError("CNN-GRU anchor/grid length mismatch")
    mean = float(np.mean(pre_raw[anchor]))
    var = float(np.var(pre_raw[anchor], ddof=1))
    z = ewma_z(raw, mean, var, np.ones(len(raw), dtype=bool))
    ma = guard_local_moving_average(z, np.ones(len(z), dtype=bool), ma_steps=5)
    return times, cues, fixation, iti, ma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_i6_cnn_gru_original"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--preflight", type=Path, default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/i6_cnn_gru_original_continuous_20261012.json"))
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()
    if args.output.exists() and not args.limit_cases:
        raise FileExistsError(args.output)
    test_fast_shift_null()
    geometry = json.loads(args.preflight.read_text(encoding="utf-8"))["rows"]
    if len(geometry) != 108:
        raise AssertionError("incomplete I4 geometry")
    check_geometry = geometry[:args.limit_cases] if args.limit_cases else geometry
    for g in check_geometry:
        path = args.cache_root / f"session{g['session']}" / f"sub-{g['subject']:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"I6 requires complete score cohort; missing {path}")
    result = {}
    for arm in ("stateful", "window_reset"):
        rows = []
        for j, g in enumerate(geometry):
            subject, session = g["subject"], g["session"]
            score = args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz"
            tier0 = args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
            fn = lambda p, t, s, sess: prepare_cache(p, t, s, sess, arm)
            rows.extend(one(score, g, None, None, prepare_fn=fn, tier0_path=tier0))
            if args.limit_cases and j + 1 >= args.limit_cases:
                print(f"preflight {arm} {j+1} case(s); no result")
                break
            if (j + 1) % 10 == 0:
                print(f"I6 {arm} {j+1}/108", flush=True)
        if args.limit_cases:
            continue
        result[arm] = {"summary": summarize(rows), "rows": rows}
    if args.limit_cases:
        return
    output = {"status": "exploratory Lee cue-paced I6; not uncued MI validation",
              "design": "I6_CNN_GRU_ORIGINAL_CONTINUOUS_DESIGN_20261012.md",
              "score_window_seconds": 1.0, "unit": "subject; S1/S2 separate",
              "arms": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({a: r["summary"] for a, r in result.items()}, indent=2))


if __name__ == "__main__":
    main()
