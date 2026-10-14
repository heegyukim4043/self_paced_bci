"""Exploratory large-CNN-GRU evaluation with unchanged I6/I4 decision rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_i4_low_fp_original_continuous_20261010 import (
    one, summarize, test_fast_shift_null,
)
from scripts.run_i6_cnn_gru_original_continuous_20261012 import prepare_cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path,
                    default=Path("data/interim/lee2019_large_cnn_gru_original"))
    ap.add_argument("--tier0-root", type=Path,
                    default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--preflight", type=Path,
                    default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/processed/large_cnn_gru_original_continuous_20261014.json"))
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()
    if args.limit_cases < 0:
        raise ValueError("limit-cases must be >= 0")
    if args.output.exists() and not args.limit_cases:
        raise FileExistsError(args.output)
    test_fast_shift_null()
    geometry = json.loads(args.preflight.read_text(encoding="utf-8"))["rows"]
    if len(geometry) != 108:
        raise AssertionError("expected 54 subjects x 2 sessions")
    selected = geometry[:args.limit_cases] if args.limit_cases else geometry
    for case in selected:
        path = args.cache_root / f"session{case['session']}" / f"sub-{case['subject']:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["model_name"]) != "LargeCausalCNNGRU":
                raise AssertionError(f"unexpected model in {path}")
    result = {}
    for arm in ("stateful", "window_reset"):
        rows = []
        for case in selected:
            subject, session = case["subject"], case["session"]
            path = args.cache_root / f"session{session}" / f"sub-{subject:02d}.npz"
            tier0 = args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
            prepare = lambda p, t, s, sess: prepare_cache(p, t, s, sess, arm)
            rows.extend(one(path, case, None, None,
                            prepare_fn=prepare, tier0_path=tier0))
        if args.limit_cases:
            print(f"smoke {arm}: {len(selected)} cases; no result file", flush=True)
        else:
            result[arm] = {"summary": summarize(rows), "rows": rows}
    if args.limit_cases:
        return
    output = {
        "status": "exploratory Lee cue-paced capacity+architecture arm; not uncued MI validation",
        "model": "LargeCausalCNNGRU",
        "comparator": "I6_CNN_GRU_ORIGINAL_CONTINUOUS_RESULT_20261012.md",
        "training_context": "simulated source sequence; original continuous target",
        "score_window_seconds": 1.0,
        "unit": "subject; sessions reported separately",
        "arms": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
