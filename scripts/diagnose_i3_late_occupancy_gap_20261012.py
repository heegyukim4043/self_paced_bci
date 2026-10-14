"""Partition tier-0 late occupancy without late alarm by earlier trial alarm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_e4_original_continuous_fp_20261009 import DWELL, REFRACTORY
from scripts.run_e4b_task_idle_20261010 import _prepare
from uncued_mi.decision import guard_rearmed_alarms


def boot_median(values, rng, draws=10000):
    x = np.asarray(values, float)
    sample = rng.integers(len(x), size=(draws, len(x)))
    return [float(v) for v in np.quantile(np.median(x[sample], axis=1), [.025, .975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--i4", type=Path, default=Path("data/processed/i4_low_fp_original_continuous_20261010.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/i3_late_occupancy_gap_20261012.json"))
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = json.loads(args.i4.read_text(encoding="utf-8"))["rows"]
    thresholds = {(r["subject"], r["session"], float(r["target_fp"])): r["calibration"]["threshold"]
                  for r in old if r["arm"] == "T_all" and float(r["target_fp"]) in (2., 4.)}
    if len(thresholds) != 216:
        raise AssertionError("missing frozen T-all thresholds")
    rows = []
    for sess in (1, 2):
        for subject in range(1, 55):
            path = args.tier0_root / f"session{sess}" / f"sub-{subject:02d}_tier0_features.npz"
            times, cues, _, _, ma = _prepare(path, subject, sess)
            for fp in (2., 4.):
                threshold = thresholds[(subject, sess, fp)]
                alarm = guard_rearmed_alarms(ma, threshold, np.ones(len(ma), bool),
                                               dwell_steps=DWELL, refractory_steps=REFRACTORY)
                categories = {key: 0 for key in (
                    "late_occupancy", "late_alarm", "early_alarm", "any_alarm",
                    "gap_late_occupancy_no_late_alarm",
                    "gap_already_early_alarm", "gap_no_early_or_late_alarm")}
                for cue in cues[70:100]:
                    early = (times >= cue + .5 - .0001) & (times < cue + 2.5 - .0001)
                    late = (times >= cue + 2.5 - .0001) & (times <= cue + 3.5 + .0001)
                    if not early.any() or not late.any() or np.any(early & late):
                        raise AssertionError("invalid nonoverlapping early/late collars")
                    has_occ = bool(np.any(ma[late] >= threshold))
                    has_late_alarm = bool(np.any(alarm[late]))
                    has_early_alarm = bool(np.any(alarm[early]))
                    gap = has_occ and not has_late_alarm
                    categories["late_occupancy"] += has_occ
                    categories["late_alarm"] += has_late_alarm
                    categories["early_alarm"] += has_early_alarm
                    categories["any_alarm"] += has_early_alarm or has_late_alarm
                    categories["gap_late_occupancy_no_late_alarm"] += gap
                    categories["gap_already_early_alarm"] += gap and has_early_alarm
                    categories["gap_no_early_or_late_alarm"] += gap and not has_early_alarm
                if (categories["gap_already_early_alarm"] + categories["gap_no_early_or_late_alarm"]
                        != categories["gap_late_occupancy_no_late_alarm"]):
                    raise AssertionError("gap categories do not partition")
                rows.append({"subject": subject, "session": sess, "target_fp": fp,
                             "n_trials": 30, **categories})
    rng = np.random.default_rng(20261012)
    summary = {}
    keys = ("late_occupancy", "late_alarm", "early_alarm", "any_alarm",
            "gap_late_occupancy_no_late_alarm", "gap_already_early_alarm",
            "gap_no_early_or_late_alarm")
    for sess in (1, 2):
        summary[str(sess)] = {}
        for fp in (2., 4.):
            selection = [r for r in rows if r["session"] == sess and r["target_fp"] == fp]
            summary[str(sess)][str(fp)] = {}
            for key in keys:
                values = [r[key] / 30 for r in selection]
                summary[str(sess)][str(fp)][key] = {
                    "total_trials": int(sum(r[key] for r in selection)),
                    "median_subject_fraction": float(np.median(values)),
                    "median_ci95": boot_median(values, rng)}
            gaps = sum(r["gap_late_occupancy_no_late_alarm"] for r in selection)
            summary[str(sess)][str(fp)]["fraction_of_gap_already_early_alarm"] = (
                sum(r["gap_already_early_alarm"] for r in selection) / gaps if gaps else None)
    result = {"status": "exploratory frozen-tier0 Lee cue-paced decision-layer mechanism diagnostic",
              "design": "I3_LATE_OCCUPANCY_GAP_DIAGNOSTIC_DESIGN_20261012.md",
              "rows": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for sess in ("1", "2"):
        for fp in ("2.0", "4.0"):
            x = summary[sess][fp]
            print("S", sess, "FP", fp, "gap", x["gap_late_occupancy_no_late_alarm"]["total_trials"],
                  "already early", x["gap_already_early_alarm"]["total_trials"],
                  "undetected", x["gap_no_early_or_late_alarm"]["total_trials"],
                  "early share", x["fraction_of_gap_already_early_alarm"])


if __name__ == "__main__":
    main()
