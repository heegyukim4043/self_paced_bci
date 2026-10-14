"""Riemannian extension of the frozen Lee I4 low-FP curve (exploratory)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_e4c_original_riemannian_fp_20261010 import prepare
from scripts.run_i4_low_fp_original_continuous_20261010 import (
    one, summarize, test_fast_shift_null,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-root", type=Path, default=Path("data/interim/lee2019_e4c_riemannian_original"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--preflight", type=Path, default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    ap.add_argument("--e4c", type=Path, default=Path("data/processed/e4c_original_riemannian_fp_20261010.json"))
    ap.add_argument("--e6r", type=Path, default=Path("data/processed/e6_strict_e6r_20261010.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/i4_riemannian_original_continuous_20261012.json"))
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()
    if args.output.exists() and not args.limit_cases:
        raise FileExistsError(args.output)
    test_fast_shift_null()
    geometry = json.loads(args.preflight.read_text(encoding="utf-8"))["rows"]
    old_fp = json.loads(args.e4c.read_text(encoding="utf-8"))["rows"]
    old_events = json.loads(args.e6r.read_text(encoding="utf-8"))["rows"]
    fp = {(r["subject"], r["session"], r["target_fp"]): r for r in old_fp if r["arm"] == "T_all"}
    events = {(r["subject"], r["session"]): r for r in old_events if r["model"] == "riemannian"}
    if len(geometry) != 108 or len(fp) != 216 or len(events) != 108:
        raise AssertionError("Frozen Riemannian cohort incomplete")
    rows = []
    for j, geo in enumerate(geometry):
        subject, session = geo["subject"], geo["session"]
        score = args.score_root / f"session{session}" / f"sub-{subject:02d}.npz"
        tier0 = args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
        new = one(score, geo, fp, None, prepare_fn=prepare, tier0_path=tier0)
        fixed = next(r for r in new if r["arm"] == "T_all" and r["target_fp"] == 4.)
        old = events[(subject, session)]
        if fixed["future_idle"]["count"] != old["future_idle"]["count"]:
            raise AssertionError("E6-R FP4 idle-count parity failed")
        if not np.isclose(fixed["mi"]["sen_2s"], old["alarm"]["primary"]["observed_any"], atol=1e-12):
            raise AssertionError("E6-R FP4 SEN parity failed")
        if not np.isclose(fixed["mi"]["null_2s"], old["alarm"]["primary"]["null_any_mean"], atol=2e-6):
            raise AssertionError("E6-R FP4 null parity failed")
        rows.extend(new)
        if (j + 1) % 10 == 0:
            print(f"I4 Riemannian {j+1}/108", flush=True)
        if args.limit_cases and j + 1 >= args.limit_cases:
            print(f"preflight only: {j+1} cases; no output")
            return
    result = {
        "status": "exploratory Lee2019 cue-paced Riemannian low-FP diagnostic; not self-paced MI validation",
        "score": "frozen LOSO Riemannian, original continuous, 1-s feature window",
        "design": "I4_LOW_FP_ORIGINAL_CONTINUOUS_PRERUN_20261010.md; post-hoc cross-model extension",
        "arms": {"T_all": "trial10-69 cue-labelled task idle", "future_idle_oracle": "future idle; diagnostic only"},
        "unit": "subject; sessions separate; 2000 subject-bootstrap draws",
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
