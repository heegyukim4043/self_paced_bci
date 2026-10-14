"""Independent row-level E6 result and post-run support-only revision audit."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path,
                    default=Path("data/processed/e6_tall_fixed_onset_20261010_v2.json"))
    ap.add_argument("--v1", type=Path,
                    default=Path("data/processed/e6_tall_fixed_onset_20261010.json"))
    ap.add_argument("--e4b", type=Path,
                    default=Path("data/processed/e4b_task_idle_fp_20261010.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/processed/e6_tall_fixed_onset_audit_20261010.json"))
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    v1 = json.loads(args.v1.read_text(encoding="utf-8"))
    e4b = json.loads(args.e4b.read_text(encoding="utf-8"))
    frozen = {(r["subject"], r["session"]): r for r in e4b["rows"]
              if r["target_fp"] == 4. and r["arm"] == "T_all"}
    old = {(r["subject"], r["session"]): r for r in v1["rows"]}
    rows = {(r["subject"], r["session"]): r for r in result["rows"]}
    if not (len(frozen) == len(old) == len(rows) == 108):
        raise AssertionError("not 108 paired E6 rows")
    max_mean_error = 0.
    max_primary_revision_error = 0.
    min_offsets = 10**9
    max_offsets = 0
    for key, row in rows.items():
        original = frozen[key]
        earlier = old[key]
        if row["threshold"] != original["calibration"]["threshold"]:
            raise AssertionError("frozen threshold mismatch")
        future = row["future_idle"]
        if (future["count"] != original["evaluation"]["E_full"]["count"] or
                not np.isclose(future["seconds"], original["evaluation"]["E_full"]["seconds"])):
            raise AssertionError("E4b future FP mismatch")
        if row["future_alarm_count"] != (row["future_mi_collar_alarm_count"] +
                                          future["count"] + row["future_unclassified_alarm_count"]):
            raise AssertionError("future alarm accounting mismatch")
        for event in ("mi", "fixation"):
            a = row[event]
            b = earlier[event]
            for metric in ("hits", "sen_2s", "null_sen_2s", "excess_2s",
                           "null_period_excluded", "excess_period_excluded"):
                if a[metric] != b[metric]:
                    raise AssertionError("post-run support audit changed primary/secondary event metric")
            if not np.isclose(a["sen_2s"], a["hits"] / a["n_events"]):
                raise AssertionError("SEN denominator mismatch")
            if not np.isclose(a["excess_2s"], a["sen_2s"] - a["null_sen_2s"]):
                raise AssertionError("excess arithmetic mismatch")
            old_metric = earlier[event]["excess_2s"]
            max_primary_revision_error = max(max_primary_revision_error,
                                             abs(a["excess_2s"] - old_metric))
        if len(row["mi"]["detected_latency_s"]) != row["mi"]["hits"]:
            raise AssertionError("detected-trial latency count mismatch")
        support = row["mi"]["first_alarm_mi_window_support_fraction"]
        if len(support) != row["mi"]["hits"] or any(x < 0 or x > 1 + 1e-9 for x in support):
            raise AssertionError("MI support fraction invalid")
        if row["mi"]["first_alarm_strict_support_n"] != sum(x >= 1 - 1e-4 for x in support):
            raise AssertionError("strict support count mismatch")
        null = row["null"]
        offsets = np.asarray(null["offset_seconds"])
        mi = np.asarray(null["mi_sen_by_offset"])
        fix = np.asarray(null["fixation_sen_by_offset"])
        if len(offsets) != len(mi) or len(mi) != len(fix):
            raise AssertionError("per-offset null array mismatch")
        if len(offsets) != null["n_primary_offsets"] or np.any(np.diff(offsets) <= 0):
            raise AssertionError("shift enumeration invalid")
        if np.min(offsets) <= 10. or np.max(offsets) >= row["future_time_seconds"] - 10.:
            raise AssertionError("near-zero circular shift not excluded")
        period = row["mean_cue_period_s"]
        m = np.rint(offsets / period)
        near = (m >= 1) & (np.abs(offsets - m * period) <= 2. + 1e-12)
        if int(np.sum(near)) != null["period_excluded_offsets"]:
            raise AssertionError("period exclusion count mismatch")
        if int(np.sum(~near)) != null["n_period_sensitivity_offsets"]:
            raise AssertionError("period sensitivity count mismatch")
        for values, name in ((mi, "mi"), (fix, "fixation")):
            max_mean_error = max(max_mean_error,
                                 abs(float(np.mean(values)) - null[f"{name}_primary_mean"]),
                                 abs(float(np.mean(values[~near])) - null[f"{name}_period_excluded_mean"]))
        min_offsets = min(min_offsets, len(offsets))
        max_offsets = max(max_offsets, len(offsets))
    if max_mean_error > 1e-6:
        raise AssertionError("shift-null mean not reproduced within stored rounding")
    audit = {"status": "pass", "n_subject_sessions": 108,
             "frozen_E4b_threshold_FP_matches": 108,
             "post_run_support_revision_changed_event_metric_max": max_primary_revision_error,
             "max_stored_null_mean_rounding_error": max_mean_error,
             "primary_offsets_min_max": [min_offsets, max_offsets],
             "strict_mi_support_first_alarm": {
                 str(session): sum(r["mi"]["first_alarm_strict_support_n"] for r in rows.values()
                                   if r["session"] == session)
                 for session in (1, 2)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
