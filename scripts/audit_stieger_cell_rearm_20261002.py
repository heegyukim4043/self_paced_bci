"""Sensitivity audit of cell-boundary arming on frozen Stieger LOSO scores.

Two prespecified structural variants are reported without selecting the better
one: pre-feedback below-threshold observations may arm, or only feedback-time
observations may arm. Neither yields a continuous-stream FP estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import analyze_stieger_cnn_gru_early_usability as baseline


def cell_alarms(level: np.ndarray, times: np.ndarray, threshold: float,
                *, arm_before_feedback: bool) -> np.ndarray:
    level = np.asarray(level, float)
    times = np.asarray(times, float)
    alarms = np.zeros(len(level), bool)
    armed, count, blocked_until = False, 0, -1
    dwell = max(1, int(np.ceil(baseline.DWELL_SECONDS / baseline.STEP_SECONDS)))
    refractory = max(0, int(np.ceil(baseline.REFRACTORY_SECONDS / baseline.STEP_SECONDS)))
    for index, value in enumerate(level):
        if not np.isfinite(value):
            armed, count = False, 0
            continue
        feedback = times[index] > 0.0
        if not feedback:
            count = 0
            if arm_before_feedback and value < threshold:
                armed = True
            continue
        if not armed:
            if value < threshold:
                armed = True
            continue
        count = count + 1 if value >= threshold else 0
        if count == dwell and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory
    return alarms


def run(cache_root: Path, shifts: int) -> dict:
    files = sorted(cache_root.glob("session*/sub-*.npz"))
    if len(files) != 30:
        raise ValueError(f"expected 30 frozen LOSO caches; found {len(files)}")
    arms = {}
    original = baseline.trial_alarms
    try:
        for arm_before_feedback in (True, False):
            name = "pre_feedback_can_arm" if arm_before_feedback else "feedback_only_can_arm"
            baseline.trial_alarms = (
                lambda level, times, threshold, flag=arm_before_feedback:
                cell_alarms(level, times, threshold, arm_before_feedback=flag)
            )
            rows = []
            for index, path in enumerate(files, 1):
                rows.append(baseline.analyze_one(path, shifts))
                print(f"{name} {index}/30 {path.name}", flush=True)
            arms[name] = {
                "rows": [{
                    "subject": r["subject"], "session": r["session"],
                    "window_auc": r["future_window_auc"],
                    "common_fp2": r["common_fp_diagnostic"]["2"],
                    "transferred_fp2": r["transferred"]["2"],
                } for r in rows],
                "summary": {
                    "common_excess_2s_median": float(np.median([
                        r["common_fp_diagnostic"]["2"]["circular_run_shift_excess_2s"]
                        for r in rows])),
                    "transferred_actual_fp2_median": float(np.median([
                        r["transferred"]["2"]["actual_fp_per_min"]
                        for r in rows])),
                    "transferred_excess_2s_median": float(np.median([
                        r["transferred"]["2"]["circular_run_shift_excess_2s"]
                        for r in rows])),
                },
            }
    finally:
        baseline.trial_alarms = original
    return {
        "protocol": {
            "scope": "30 frozen subject-LOSO Stieger trial-cell caches; descriptive structural sensitivity",
            "decision": "cell-local MA/dwell/refractory; require observed below-threshold before first alarm",
            "arms": ["pre_feedback_can_arm", "feedback_only_can_arm"],
            "claim_limit": "feedback-onset trial-cell simulation, not native-continuous or free-onset MI",
        },
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/interim/stieger2021_cnn_gru_loso_scores"))
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/stieger2021_cell_rearm_audit_20261002.json"))
    args = parser.parse_args()
    payload = run(args.cache_root, args.shifts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in payload["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
