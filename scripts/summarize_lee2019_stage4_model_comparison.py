"""Join frozen chronological operating and matched-FP model comparisons.

No thresholds are refit here. The matched arm is a future-idle-label oracle,
while the target-pre-rest arm is prospective in the simulation. Neither arm
has passed a cue-matched, MI-specific EEG-level null.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SOURCE = ROOT / "data/processed/performance_revalidation_20261001"

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z


def sign_counts(delta: np.ndarray) -> dict[str, int]:
    return {"cnn_more": int(np.sum(delta > 0)),
            "tie": int(np.sum(delta == 0)),
            "riemannian_more": int(np.sum(delta < 0))}


def summarize(operating: dict, matched: dict) -> dict:
    op_rows = {(int(r["subject"]), int(r["session"]), float(r["target_fp"])): r
               for r in operating["rows"]}
    match_rows = {(int(r["subject"]), int(r["session"]), float(r["target_fp"])): r
                  for r in matched["rows"]}
    if len(op_rows) != 216 or op_rows.keys() != match_rows.keys():
        raise AssertionError("expected complete 54 subjects x 2 sessions x 2 targets")
    result = {}
    for session in (1, 2):
        for target in (2., 4.):
            keys = [(subject, session, target) for subject in range(1, 55)]
            rows = [(op_rows[key], match_rows[key]) for key in keys]
            if any(not m["matched"] for _, m in rows):
                raise AssertionError("a diagnostic matched-FP pair is missing")
            op_fp = np.asarray([o["cnn_gru"]["future_fp_per_min"] -
                                o["riemannian"]["future_fp_per_min"]
                                for o, _ in rows])
            op_hits = np.asarray([o["cnn_gru"]["future_onset_hits2"] -
                                  o["riemannian"]["future_onset_hits2"]
                                  for o, _ in rows])
            match_hits = np.asarray([m["cnn_gru_hits2"] - m["riemannian_hits2"]
                                     for _, m in rows])
            exposure_difference = np.asarray([
                o["cnn_gru"]["future_idle_exposure_min"] -
                o["riemannian"]["future_idle_exposure_min"] for o, _ in rows])
            if not np.allclose(exposure_difference, 0., rtol=0., atol=1e-12):
                raise AssertionError("model idle exposure differs")
            for o, m in rows:
                if abs(m["achieved_fp_per_min"] -
                       m["matched_fp_count"] /
                       o["riemannian"]["future_idle_exposure_min"]) > 1e-9:
                    raise AssertionError("matched FP rate/count mismatch")
            half_width = .5 if target == 2. else 1.
            r_ok = np.asarray([abs(o["riemannian"]["future_fp_per_min"] - target)
                               <= half_width for o, _ in rows])
            c_ok = np.asarray([abs(o["cnn_gru"]["future_fp_per_min"] - target)
                               <= half_width for o, _ in rows])
            result[f"FP{int(target)}_S{session}"] = {
                "n_subjects": 54,
                "operating_median_paired_fp_delta_cnn_minus_riemannian": float(np.median(op_fp)),
                "operating_fp_delta_sign_counts": sign_counts(op_fp),
                "operating_hit_delta_sign_counts": sign_counts(op_hits),
                "operating_pooled_hit_delta_per_onset": float(op_hits.sum() / 1620.),
                "matched_hit_delta_sign_counts": sign_counts(match_hits),
                "matched_pooled_hit_delta_per_onset": float(match_hits.sum() / 1620.),
                "operating_joint_admissibility": {
                    "both": int(np.sum(r_ok & c_ok)),
                    "riemannian_only": int(np.sum(r_ok & ~c_ok)),
                    "cnn_only": int(np.sum(~r_ok & c_ok)),
                    "neither": int(np.sum(~r_ok & ~c_ok)),
                },
                "cnn_hit_gain_with_more_fp_n": int(np.sum((op_hits > 0) & (op_fp > 0))),
                "cnn_hit_gain_without_more_fp_n": int(np.sum((op_hits > 0) & (op_fp <= 0))),
                "matched_exact_fp_count_all_pairs": True,
            }
    return result


def scalar_redecode(operating: dict, matched: dict) -> dict:
    """Independently replay every saved threshold with the scalar reference."""
    op = {(int(r["subject"]), int(r["session"]), float(r["target_fp"])): r
          for r in operating["rows"]}
    ma = {(int(r["subject"]), int(r["session"]), float(r["target_fp"])): r
          for r in matched["rows"]}
    checked = 0
    for session in (1, 2):
        for subject in range(1, 55):
            cached = {}
            for model, root, anchor_key in (
                ("riemannian", ROOT / "data/interim/lee2019_revalidation_riemannian_scores",
                 "anchor_raw_scores"),
                ("cnn_gru", ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache",
                 "anchor_scores"),
            ):
                with np.load(root / f"session{session}" / f"sub-{subject:02d}.npz",
                             allow_pickle=False) as data:
                    raw = data["raw_scores"].astype(float)
                    anchor = data[anchor_key].astype(float)
                    times = data["times"].astype(float)
                    allowed = data["allowed"].astype(bool)
                    events = data["events"].astype(np.int64)
                z = ewma_z(raw, float(anchor.mean()),
                           float(anchor.var(ddof=1)), allowed)
                statistic = guard_local_moving_average(z, allowed, ma_steps=5)
                cached[model] = (times, allowed, events, statistic)
            for key in range(3):
                if not np.array_equal(cached["riemannian"][key],
                                      cached["cnn_gru"][key]):
                    raise AssertionError("paired future stream geometry differs")
            times, allowed, events, _ = cached["riemannian"]
            future = times >= events[20, 2] / 250.
            cues = events[20:, 5] / 250.
            onsets = cues + .5
            task = np.zeros(len(times), dtype=bool)
            for cue, onset in zip(cues, onsets):
                task |= (times >= onset) & (times <= cue + 4.)
            idle = future & allowed & ~task
            for target in (2., 4.):
                key = (subject, session, target)
                for model in ("riemannian", "cnn_gru"):
                    statistic = cached[model][3]
                    for condition, threshold in (
                        ("operating", op[key][model]["threshold"]),
                        ("matched", ma[key][f"{model}_threshold"]),
                    ):
                        alarms = guard_rearmed_alarms(
                            statistic, float(threshold), allowed,
                            dwell_steps=3, refractory_steps=10,
                        ) & future
                        fp = int(np.sum(alarms & idle))
                        hits = int(sum(np.any(
                            alarms & (times >= onset) & (times <= onset + 2.)
                        ) for onset in onsets))
                        if condition == "operating":
                            expected = (op[key][model]["future_fp_count"],
                                        op[key][model]["future_onset_hits2"])
                        else:
                            expected = (ma[key]["matched_fp_count"],
                                        ma[key][f"{model}_hits2"])
                        if (fp, hits) != expected:
                            raise AssertionError(
                                f"scalar mismatch {key}/{model}/{condition}: "
                                f"{(fp, hits)} != {expected}")
                        checked += 1
    return {"decoded_model_condition_target_rows": checked,
            "all_saved_fp_and_hits_reproduced": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operating", type=Path,
                        default=SOURCE / "a2_chronological_anchor_operating_points_stage4_20261002.json")
    parser.add_argument("--matched", type=Path,
                        default=SOURCE / "a2_cnn_gru_vs_riemannian_chronological_matched_fp_stage4_20261002.json")
    parser.add_argument("--output", type=Path,
                        default=SOURCE / "a2_stage4_integrated_operating_vs_matched_20261002.json")
    args = parser.parse_args()
    operating = json.loads(args.operating.read_text(encoding="utf-8"))
    matched = json.loads(args.matched.read_text(encoding="utf-8"))
    payload = {
        "scope": "descriptive observed-only matched-FP and prospective anchor operating-point join",
        "warning": "future-label diagnostic oracle is not deployable; no MI-specific null validation",
        "summary": summarize(operating, matched),
        "scalar_validation": scalar_redecode(operating, matched),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
