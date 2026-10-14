"""Descriptive, subject-level audit of frozen I6 full-future score/alarm patterns.

No thresholds, model parameters, or evaluation supports are changed. The visual
states use the same cue-defined MI and task-idle masks as the frozen I6 result.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import plot_i6_original_continuous_four_panel_20261014 as plotter
from scripts.run_e6_tall_fixed_onset_20261010 import collar_indices, hit_vector

OUTDIR = ROOT / "data/processed/i6_cnn_gru_original_continuous_fp2_full_future_20261014"
STATES = ("cue_mi", "fixation", "iti_early", "iti_late", "task_idle")
STEP = .1


def values_for(mask: np.ndarray, t: dict) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n_points": 0, "seconds": 0.0, "alarm_count": 0,
                "alarm_per_min": None, "ma_margin_median": None,
                "ma_margin_q90": None, "ma_margin_q99": None,
                "above_threshold_fraction": None}
    score = t["ma_margin"][mask]
    alarms = int(np.sum(t["alarm"] & mask))
    seconds = n * STEP
    return {"n_points": n, "seconds": seconds, "alarm_count": alarms,
            "alarm_per_min": alarms * 60.0 / seconds,
            "ma_margin_median": float(np.median(score)),
            "ma_margin_q90": float(np.quantile(score, .9)),
            "ma_margin_q99": float(np.quantile(score, .99)),
            "above_threshold_fraction": float(np.mean(score >= 0))}


def main() -> None:
    out_csv = OUTDIR / "full_future_pattern_rows.csv"
    out_json = OUTDIR / "full_future_pattern_summary.json"
    if out_csv.exists() or out_json.exists():
        raise FileExistsError("audit refuses to overwrite existing result")
    original = json.loads(plotter.RESULT.read_text(encoding="utf-8"))
    rows_by_id = {
        (int(r["subject"]), int(r["session"])): r
        for r in original["arms"]["stateful"]["rows"]
        if r["arm"] == "T_all" and r["target_fp"] == 2.0
    }
    if len(rows_by_id) != 108:
        raise AssertionError("incomplete frozen I6 FP2 result")
    rows = []
    for subject in range(1, 55):
        for session in (1, 2):
            old = rows_by_id[(subject, session)]
            t = plotter.one_case(subject, session, old, view_seconds=None)
            masks = {"cue_mi": t["command"], "fixation": t["fixation"],
                     "iti_early": t["iti_early"], "iti_late": t["iti_late"],
                     "task_idle": t["idle"]}
            if np.any(t["command"] & t["idle"]):
                raise AssertionError("cue-MI/task-idle masks overlap")
            if sum(int(np.sum(t["alarm"] & masks[k])) for k in
                   ("fixation", "iti_early", "iti_late")) != old["future_idle"]["count"]:
                raise AssertionError("future-idle alarm stratum sum differs from I6")
            starts, ends = collar_indices(t["times"], t["cues"] + .5)
            hits = hit_vector(t["alarm"], starts, ends)
            if int(np.sum(hits)) != int(old["mi"]["hits"]):
                raise AssertionError("SEN@2s hits differ from I6")
            occupancy = np.asarray([
                bool(np.any(t["ma_margin"][i:j] >= 0))
                for i, j in zip(starts, ends)
            ], dtype=bool)
            state_values = {k: values_for(mask, t) for k, mask in masks.items()}
            trial_gap = int(np.sum(occupancy & ~hits))
            no_evidence = int(np.sum(~occupancy))
            row = {
                "subject": subject, "session": session,
                "future_duration_s": float(t["future_duration_s"]),
                "threshold": float(t["threshold"]),
                "future_fp_per_min": float(old["future_idle"]["fp_per_min"]),
                "sen_2s": float(old["mi"]["sen_2s"]),
                "excess_2s": float(old["mi"]["excess_2s"]),
                "mi_minus_idle_median_margin": (
                    state_values["cue_mi"]["ma_margin_median"] -
                    state_values["task_idle"]["ma_margin_median"]),
                "mi_minus_idle_above_threshold_fraction": (
                    state_values["cue_mi"]["above_threshold_fraction"] -
                    state_values["task_idle"]["above_threshold_fraction"]),
                "primary_2s_trials_above_threshold": int(np.sum(occupancy)),
                "primary_2s_trials_alarm": int(np.sum(hits)),
                "primary_2s_trials_above_no_alarm": trial_gap,
                "primary_2s_trials_no_above": no_evidence,
                "states": state_values,
            }
            if row["primary_2s_trials_alarm"] > row["primary_2s_trials_above_threshold"]:
                raise AssertionError("alarm without threshold occupancy")
            rows.append(row)
    if len(rows) != 108:
        raise AssertionError("incomplete audit")
    flat = []
    for row in rows:
        base = {k: v for k, v in row.items() if k != "states"}
        for state in STATES:
            flat.append({**base, "state": state, **row["states"][state]})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary = {"status": "exploratory cue-paced descriptive audit; no new model or threshold",
               "source": str(plotter.RESULT.relative_to(ROOT)),
               "n_subjects": 54, "n_subject_sessions": 108,
               "figure_scope": "full original continuous future trials 70-99",
               "sessions": {}}
    fields = ("future_fp_per_min", "sen_2s", "excess_2s",
              "mi_minus_idle_median_margin", "mi_minus_idle_above_threshold_fraction",
              "primary_2s_trials_above_threshold", "primary_2s_trials_alarm",
              "primary_2s_trials_above_no_alarm", "primary_2s_trials_no_above")
    for session in (1, 2):
        part = [r for r in rows if r["session"] == session]
        session_summary = {}
        for field in fields:
            data = np.asarray([r[field] for r in part], float)
            session_summary[field] = {
                "median": float(np.median(data)),
                "q10": float(np.quantile(data, .1)),
                "q90": float(np.quantile(data, .9)),
            }
        session_summary["subjects_zero_primary_hits"] = sum(r["primary_2s_trials_alarm"] == 0 for r in part)
        session_summary["subjects_positive_mi_idle_median_contrast"] = sum(r["mi_minus_idle_median_margin"] > 0 for r in part)
        session_summary["subjects_future_fp_above_2_5"] = sum(r["future_fp_per_min"] > 2.5 for r in part)
        session_summary["states"] = {}
        for state in STATES:
            session_summary["states"][state] = {
                field: float(np.median([r["states"][state][field] for r in part]))
                for field in ("seconds", "alarm_count", "alarm_per_min",
                              "ma_margin_median", "ma_margin_q90",
                              "above_threshold_fraction")
            }
        summary["sessions"][str(session)] = session_summary
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary["sessions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
