"""Independent metadata, count, and training-exposure audit for E4c rows."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path,
                    default=Path("data/processed/e4c_original_riemannian_fp_20261010.json"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/e4c_riemannian_original_score_manifest_20261010.json"))
    ap.add_argument("--preflight", type=Path,
                    default=Path("data/processed/e4b_task_idle_preflight_20261010.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/processed/e4c_original_riemannian_fp_audit_20261010.json"))
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    geometry = {(r["subject"], r["session"]): r for r in preflight["rows"]}
    caches = {(r["subject"], r["session"]): r for r in manifest["rows"]}
    if len(geometry) != 108 or len(caches) != 108:
        raise AssertionError("not 108 independent subject-session caches")
    parities = [r["parity"]["max_abs_raw_score_error"] for r in caches.values()]
    if max(parities) > 5e-5 or any(r["n_source_subjects"] != 53 for r in caches.values()):
        raise AssertionError("score parity or LOSO source count failure")
    groups = defaultdict(dict)
    for row in result["rows"]:
        key = (row["subject"], row["session"], row["target_fp"])
        if row["arm"] in groups[key]:
            raise AssertionError("duplicate arm")
        groups[key][row["arm"]] = row
    if len(groups) != 216 or len(result["rows"]) != 864:
        raise AssertionError("not 54*2*2*4 rows")
    sham_diff = defaultdict(int)
    updates = defaultdict(int)
    for (subject, session, target), arms in groups.items():
        if set(arms) != {"T56", "T_all", "T_cum", "T_sham_update"}:
            raise AssertionError("missing arm")
        if (subject, session) not in caches:
            raise AssertionError("missing score cache")
        geo = geometry[(subject, session)]
        for arm in arms.values():
            ev = arm["evaluation"]
            if ev["E_full"]["count"] != ev["E_70_79"]["count"] + ev["E_updated"]["count"]:
                raise AssertionError("future count split failure")
            if ev["E_updated"]["count"] != ev["E_80_89"]["count"] + ev["E_90_99"]["count"]:
                raise AssertionError("updated count split failure")
            for name, item in ev.items():
                if not np.isclose(item["seconds"], geo["seconds"][name], atol=.001):
                    raise AssertionError("exposure differs from E4b geometry")
                if not np.isclose(item["fp_per_min"], item["count"] * 60. / item["seconds"], atol=1e-12):
                    raise AssertionError("FP denominator mismatch")
        if arms["T_all"]["evaluation"]["E_70_79"]["count"] != arms["T_cum"]["evaluation"]["E_70_79"]["count"]:
            raise AssertionError("T-cum changed pre-update count")
        c = arms["T_cum"]["calibration"]
        if not (c["initial"]["seconds"] < c["after_79"]["seconds"] < c["after_89"]["seconds"]):
            raise AssertionError("cumulative calibration exposure failed")
        if not np.isclose(c["initial"]["threshold"], arms["T_all"]["calibration"]["threshold"]):
            raise AssertionError("T-cum initial threshold mismatch")
        if c["initial"]["threshold"] != c["after_79"]["threshold"] or c["after_79"]["threshold"] != c["after_89"]["threshold"]:
            updates[(session, target)] += 1
        if any(arms["T_all"]["evaluation"][w]["count"] != arms["T_sham_update"]["evaluation"][w]["count"]
               for w in ("E_full", "E_updated", "E_70_79", "E_80_89", "E_90_99")):
            sham_diff[(session, target)] += 1
    audit = {
        "status": "pass",
        "n_subject_sessions": 108,
        "n_groups": len(groups),
        "n_rows": len(result["rows"]),
        "max_replay_parity_abs_raw_score_error": max(parities),
        "all_loso_source_count": 53,
        "sham_count_differs_from_static_subjects": {
            f"S{s}_FP{target:g}": sham_diff[(s, target)]
            for s in (1, 2) for target in (4., 2.)},
        "subjects_with_cumulative_threshold_change": {
            f"S{s}_FP{target:g}": updates[(s, target)]
            for s in (1, 2) for target in (4., 2.)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
