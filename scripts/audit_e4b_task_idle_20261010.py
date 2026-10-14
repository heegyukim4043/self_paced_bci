"""Independent row-level invariants for the frozen E4b output JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path,
                        default=Path("data/processed/e4b_task_idle_fp_20261010.json"))
    parser.add_argument("--e4", type=Path,
                        default=Path("data/processed/e4_original_continuous_fp_20261009_v2.json"))
    parser.add_argument("--preflight", type=Path,
                        default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    args = parser.parse_args()
    current = json.loads(args.result.read_text(encoding="utf-8"))
    e4 = json.loads(args.e4.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    old = {(r["subject"], r["session"], r["target_fp"]): r for r in e4["rows"]
           if r["arm"] == "T_near" and r["mode"] == "all_idle"}
    geometry = {(r["subject"], r["session"]): r for r in preflight["rows"]}
    groups = defaultdict(dict)
    for row in current["rows"]:
        key = (row["subject"], row["session"], row["target_fp"])
        if row["arm"] in groups[key]:
            raise AssertionError("duplicate arm")
        groups[key][row["arm"]] = row
    if len(groups) != 216 or len(current["rows"]) != 864:
        raise AssertionError("not 54 subjects x 2 sessions x 2 targets x 4 arms")
    n_changed = defaultdict(int)
    for (subject, session, target), arms in groups.items():
        if set(arms) != {"T56", "T_all", "T_cum", "T_sham_update"}:
            raise AssertionError("arm set incomplete")
        geo = geometry[(subject, session)]
        old_row = old[(subject, session, target)]
        t56 = arms["T56"]
        if t56["evaluation"]["E_full"]["count"] != old_row["future"]["count"]:
            raise AssertionError("T56 does not reproduce E4 alarm count")
        if not np.isclose(t56["calibration"]["threshold"],
                          old_row["calibration"]["threshold"], atol=1e-10):
            raise AssertionError("T56 does not reproduce E4 threshold")
        for arm in arms.values():
            ev = arm["evaluation"]
            if ev["E_full"]["count"] != (ev["E_70_79"]["count"] +
                                         ev["E_updated"]["count"]):
                raise AssertionError("future count partition broken")
            if ev["E_updated"]["count"] != (ev["E_80_89"]["count"] +
                                            ev["E_90_99"]["count"]):
                raise AssertionError("updated count partition broken")
            for window, value in ev.items():
                if not np.isclose(value["seconds"], geo["seconds"][window], atol=.001):
                    raise AssertionError("evaluation exposure differs from preflight")
                if not np.isclose(value["fp_per_min"], value["count"] * 60. /
                                  value["seconds"], atol=1e-12):
                    raise AssertionError("FP denominator mismatch")
        for window in ("E_full", "E_updated", "E_70_79", "E_80_89", "E_90_99"):
            if arms["T_all"]["evaluation"][window]["count"] != arms[
                    "T_sham_update"]["evaluation"][window]["count"]:
                raise AssertionError("sham reset changes alarm count")
        if arms["T_all"]["evaluation"]["E_70_79"]["count"] != arms[
                "T_cum"]["evaluation"]["E_70_79"]["count"]:
            raise AssertionError("cumulative update affected pre-update block")
        cp = arms["T_cum"]["calibration"]
        if not (cp["initial"]["seconds"] < cp["after_79"]["seconds"] <
                cp["after_89"]["seconds"]):
            raise AssertionError("cumulative exposure did not grow")
        if not np.isclose(cp["initial"]["threshold"],
                          arms["T_all"]["calibration"]["threshold"]):
            raise AssertionError("T-cum initial threshold differs from T-all")
        if cp["initial"]["threshold"] != cp["after_79"]["threshold"] or cp[
                "after_79"]["threshold"] != cp["after_89"]["threshold"]:
            n_changed[(session, target)] += 1
    print(json.dumps({"status": "pass", "n_subject_sessions": 108,
                      "n_subject_session_target_groups": len(groups),
                      "n_rows": len(current["rows"]),
                      "E4_T56_count_and_threshold_matches": len(groups),
                      "sham_update_equals_static": len(groups),
                      "preupdate_Tcum_equals_Tall": len(groups),
                      "subjects_with_threshold_change": {
                          f"S{s}_FP{t:g}": n_changed[(s, t)]
                          for s in (1, 2) for t in (4., 2.)}}, indent=2))


if __name__ == "__main__":
    main()
