"""Apply the frozen I4 FP1/2/4/8 rules to CBraMod LOSO score caches.

Lee remains cue-paced. The future-idle threshold arm is diagnostic only; the
T-all arm uses earlier cued task-idle for calibration. No self-paced claim.
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
from scripts.run_i4_low_fp_original_continuous_20261010 import (
    one, summarize, test_fast_shift_null,
)
from uncued_mi.cbramod_transfer import (
    FINETUNE_PROTOCOL, FINETUNED_MODEL_NAME, MODEL_NAME, PROTOCOL_VERSION,
)
from uncued_mi.decision import guard_local_moving_average


def prepare_cache(path: Path, tier0_path: Path, subject: int, session: int,
                  *, expected_model: str = MODEL_NAME):
    expected_protocol = (PROTOCOL_VERSION if expected_model == MODEL_NAME
                         else FINETUNE_PROTOCOL)
    with np.load(path, allow_pickle=False) as data:
        if ((int(data["subject"]), int(data["session"])) != (subject, session)
                or str(data["model_name"]) != expected_model
                or str(data["protocol_version"]) != expected_protocol):
            raise AssertionError("CBraMod score cache identity/model mismatch")
        provenance = json.loads(str(data["provenance_json"]))
        donors = provenance["source_subjects"]
        if (provenance["target_trials_used_for_fit"] or subject in donors
                or len(donors) != 53 or len(set(donors)) != 53):
            raise AssertionError("target leakage or incomplete LOSO donors")
        times = np.asarray(data["times"], dtype=float)
        pre_times = np.asarray(data["pre_times"], dtype=float)
        cues_model = np.asarray(data["cue_times"], dtype=float)
        raw = np.asarray(data["raw_scores"], dtype=float)
        pre_raw = np.asarray(data["pre_raw_scores"], dtype=float)
    with np.load(tier0_path, allow_pickle=False) as data:
        tier_times = np.asarray(data["continuous_times"], dtype=float)
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
        fixation = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
    if (len(times) != len(tier_times)
            or not np.allclose(times, tier_times, atol=0.00251, rtol=0)
            or len(raw) != len(times)
            or not np.allclose(cues_model, cues, atol=0.005, rtol=0)):
        raise AssertionError("CBraMod/tier-0 decision grid differs")
    if (not np.all(np.diff(times) > 0) or not np.all(np.diff(pre_times) > 0)
            or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(pre_raw))):
        raise AssertionError("invalid CBraMod score trace")
    anchor = (pre_times >= 4.0 - 0.001) & (pre_times < 60.0 - 0.001)
    if anchor.sum() != 560:
        raise AssertionError("expected exactly 56 s of 0.1-s anchor decisions")
    mean = float(np.mean(pre_raw[anchor]))
    variance = float(np.var(pre_raw[anchor], ddof=1))
    if variance <= 0:
        raise AssertionError("zero-variance pre-rest score")
    z = ewma_z(raw, mean, variance, np.ones(len(raw), dtype=bool))
    ma = guard_local_moving_average(z, np.ones(len(z), dtype=bool), ma_steps=5)
    # Use the exact tier-0 timestamps for frozen trial/idle masks.
    return tier_times, cues, fixation, iti, ma


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path,
                    default=Path("data/interim/lee2019_cbramod_frozen_loso_scores"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--preflight", type=Path,
                    default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/processed/cbramod_original_continuous_20261014.json"))
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--model-name", choices=(MODEL_NAME, FINETUNED_MODEL_NAME),
                    default=MODEL_NAME)
    args = ap.parse_args()
    if args.limit_cases < 0:
        raise ValueError("limit-cases must be non-negative")
    if args.output.exists() and not args.limit_cases:
        raise FileExistsError(args.output)
    test_fast_shift_null()
    geometry = json.loads(args.preflight.read_text(encoding="utf-8"))["rows"]
    if len(geometry) != 108:
        raise AssertionError("evaluation geometry must include 54 subjects x 2 sessions")
    cases = geometry[:args.limit_cases] if args.limit_cases else geometry
    rows = []
    for case in cases:
        subject, session = case["subject"], case["session"]
        path = args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz"
        tier = args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
        if not path.is_file() or not tier.is_file():
            raise FileNotFoundError(f"missing CBraMod score or tier-0 geometry: {path}, {tier}")
        prepare = lambda p, t, s, sess: prepare_cache(
            p, t, s, sess, expected_model=args.model_name)
        rows.extend(one(path, case, None, None, prepare_fn=prepare,
                        tier0_path=tier))
        print(f"evaluated {session}/{subject:02d}", flush=True)
    if args.limit_cases:
        print(f"smoke only: {len(cases)} cases, no result file", flush=True)
        return
    report = {
        "status": "exploratory Lee cue-paced; not uncued MI validation",
        "model": args.model_name,
        "source_training": "53 other subjects, original continuous trial 0..49 windows",
        "backbone": ("published pretrained CBraMod, frozen"
                     if args.model_name == MODEL_NAME else
                     "published pretrained CBraMod, source-only full fine-tuning"),
        "head": ("source-only balanced logistic regression"
                 if args.model_name == MODEL_NAME else "source-only linear neural head"),
        "score_window_seconds": 2.0,
        "comparator_note": "CNN-GRU uses a 1-s window; this is a model+input protocol comparison",
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
