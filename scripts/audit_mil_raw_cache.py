"""Aggregate and validate all preregistered MIL raw-cache audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path("data/interim/lee2019_mil_raw")
    rows = []
    for session in (1, 2):
        paths = sorted((root / f"session{session}").glob("sub-*_mil_raw.npz"))
        if len(paths) != 20:
            raise AssertionError(f"session {session}: expected 20 caches, got {len(paths)}")
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                audit = json.loads(str(data["audit_json"]))
                sizes = audit["positive_bags"]["instances_per_bag"]
                if not set(sizes).issubset({20, 21}):
                    raise AssertionError(f"{path}: invalid bag sizes {sizes}")
                if audit["positive_bags"]["fit_trials_0_39"] != 40:
                    raise AssertionError(f"{path}: fitting bag count")
                if audit["positive_bags"]["validation_trials_40_49"] != 10:
                    raise AssertionError(f"{path}: validation bag count")
                if audit["positive_bags"]["locked_trials_50_99"] != 50:
                    raise AssertionError(f"{path}: locked bag count")
                rows.append(
                    {
                        "session": session,
                        "subject": int(path.stem.split("-")[1].split("_")[0]),
                        **audit,
                    }
                )

    summary = {}
    for session in (1, 2):
        selected = [row for row in rows if row["session"] == session]
        for key in (
            "pre_rest_fit",
            "pre_rest_validation",
            "iti_fit_trials_0_39",
            "iti_validation_trials_40_49",
            "iti_locked_trials_50_99",
        ):
            values = [row["negative_instances"][key] for row in selected]
            summary[f"session{session}_{key}"] = {
                "min": int(np.min(values)),
                "median": float(np.median(values)),
                "max": int(np.max(values)),
            }
    result = {
        "status": "passed",
        "caches": len(rows),
        "summary": summary,
        "rows": rows,
    }
    output = Path("data/processed/mil_raw_cache_audit.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "caches": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
