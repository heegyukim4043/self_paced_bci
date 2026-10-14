"""E4c frozen LOSO Riemannian: E4b task-idle calibration on original time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.run_e4_original_continuous_fp_20261009 import (
    STEP, DWELL, REFRACTORY, _trial_strata, _union,
)
from scripts.run_e4b_task_idle_20261010 import (
    TARGETS, WINDOWS, curve, pick, rearmed_with_updates, summarize,
)
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms


def prepare(score_path: Path, tier0_path: Path, subject: int, session: int):
    with np.load(score_path, allow_pickle=False) as data:
        if (int(data["subject"]), int(data["session"])) != (subject, session):
            raise AssertionError("Riemannian score identity mismatch")
        times = np.asarray(data["times"], dtype=float)
        raw = np.asarray(data["raw_scores"], dtype=float)
        pre_times = np.asarray(data["pre_times"], dtype=float)
        pre_raw = np.asarray(data["pre_raw_scores"], dtype=float)
        cue_model = np.asarray(data["cue_times"], dtype=float)
        source_subjects = np.asarray(data["source_subjects"])
        provenance = json.loads(str(data["provenance_json"]))
        parity = json.loads(str(data["parity_json"]))
    if len(source_subjects) != 53 or subject in source_subjects or provenance["target_future_used_for_fit"]:
        raise AssertionError("target training leakage")
    if parity["max_abs_raw_score_error"] > parity["atol"]:
        raise AssertionError("replay score parity not passed")
    with np.load(tier0_path, allow_pickle=False) as data:
        original_times = np.asarray(data["continuous_times"], dtype=float)
        cues = np.asarray(data["cue_samples"], dtype=float) / float(data["native_sfreq"])
        fixation = np.asarray(data["fixation_intervals"], dtype=float)
        iti = np.asarray(data["iti_intervals"], dtype=float).reshape(-1, 2)
    if not np.allclose(cue_model, cues, atol=.005, rtol=0):
        raise AssertionError("raw-score and tier-0 cue times differ")
    keep = times >= 4. - .001
    times, raw = times[keep], raw[keep]
    original_times = original_times[original_times >= 4. - .001]
    if len(times) != len(original_times) or not np.allclose(times, original_times, atol=.001, rtol=0):
        raise AssertionError("score grid differs from E4b original time grid")
    anchor = (pre_times >= 4. - .001) & (pre_times < 60. - .001)
    if int(np.sum(anchor)) != 560:
        raise AssertionError("not 560 common pre-rest score points")
    mean = float(np.mean(pre_raw[anchor]))
    var = float(np.var(pre_raw[anchor], ddof=1))
    z = ewma_z(raw, mean, var, np.ones(len(raw), dtype=bool))
    ma = guard_local_moving_average(z, np.ones(len(z), dtype=bool), ma_steps=5)
    return times, cues, fixation, iti, ma


def one(score_path: Path, tier0_path: Path, geometry: dict) -> list[dict]:
    subject, session = geometry["subject"], geometry["session"]
    times, cues, fixation, iti, ma = prepare(score_path, tier0_path, subject, session)
    ids = {"T56": geometry["T56_trial_ids"],
           "T_all": list(range(10, 70)),
           "T_79": list(range(10, 80)),
           "T_89": list(range(10, 90)),
           "E_full": list(range(70, 100)),
           "E_updated": list(range(80, 100)),
           "E_70_79": list(range(70, 80)),
           "E_80_89": list(range(80, 90)),
           "E_90_99": list(range(90, 100))}
    masks = {name: _union(_trial_strata(times, cues, fixation, iti, values), False)
             for name, values in ids.items()}
    for name, mask in masks.items():
        if not np.isclose(mask.sum() * STEP, geometry["seconds"][name], atol=.001):
            raise AssertionError(f"E4c exposure differs from E4b: {name}")
    updates = np.asarray([x["index"] for x in geometry["boundaries"]], dtype=np.int64)
    curves = {name: curve(ma, masks[name]) for name in ("T56", "T_all", "T_79", "T_89")}
    out = []
    for target in TARGETS:
        points = {name: pick(c, target) for name, c in curves.items()}
        fixed_56 = guard_rearmed_alarms(ma, points["T56"]["threshold"],
                                         np.ones(len(ma), dtype=bool), DWELL, REFRACTORY)
        fixed_all = guard_rearmed_alarms(ma, points["T_all"]["threshold"],
                                          np.ones(len(ma), dtype=bool), DWELL, REFRACTORY)
        cum = rearmed_with_updates(ma, np.asarray([points[k]["threshold"] for k in
                                                    ("T_all", "T_79", "T_89")]), updates)
        sham = rearmed_with_updates(ma, np.repeat(points["T_all"]["threshold"], 3), updates)
        if not np.array_equal(fixed_all[:updates[0]], cum[:updates[0]]):
            raise AssertionError("cumulative arm differs before first update")
        # A sham reset can change alarm timing even at a fixed threshold.
        # Preserve that difference as the pre-specified negative control;
        # do not assume it is identically zero for every score family.
        for arm, alarms in (("T56", fixed_56), ("T_all", fixed_all),
                            ("T_cum", cum), ("T_sham_update", sham)):
            counts = {name: {"count": int(np.sum(alarms & masks[name])),
                             "seconds": float(np.sum(masks[name]) * STEP),
                             "fp_per_min": float(np.sum(alarms & masks[name]) * 60. /
                                                 (np.sum(masks[name]) * STEP))}
                      for name in WINDOWS}
            if counts["E_full"]["count"] != (counts["E_70_79"]["count"] + counts["E_updated"]["count"]):
                raise AssertionError("future count blocks do not partition")
            calibration = (points["T56"] if arm == "T56" else
                           {"initial": points["T_all"], "after_79": points["T_79"],
                            "after_89": points["T_89"]} if arm == "T_cum" else
                           points["T_all"])
            out.append({"subject": subject, "session": session, "target_fp": target,
                        "arm": arm, "calibration": calibration, "evaluation": counts})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-root", type=Path,
                    default=Path("data/interim/lee2019_e4c_riemannian_original"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--preflight", type=Path,
                    default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/processed/e4c_original_riemannian_fp_20261010.json"))
    ap.add_argument("--limit-cases", type=int, default=0, help="preflight only; do not write result")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    geometry = json.loads(args.preflight.read_text(encoding="utf-8"))["rows"]
    if len(geometry) != 108:
        raise AssertionError("E4b geometry incomplete")
    rows = []
    for j, g in enumerate(geometry):
        session, subject = g["session"], g["subject"]
        score = args.score_root / f"session{session}" / f"sub-{subject:02d}.npz"
        tier0 = args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
        rows.extend(one(score, tier0, g))
        if (j + 1) % 10 == 0:
            print(f"E4c FP {j + 1}/108", flush=True)
        if args.limit_cases and j + 1 >= args.limit_cases:
            print(f"preflight only: {j+1} case(s), no result written")
            return
    result = {"status": "exploratory E4c LOSO Riemannian original continuous cued task-idle FP",
              "source": "E4b masks/decision with new parity-audited original Riemannian scores",
              "unit": "subject, session separate; 2000 paired subject bootstrap draws",
              "summary": summarize(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
