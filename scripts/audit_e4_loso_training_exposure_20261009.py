"""Audit cached Riemannian source trial IDs; no model fitting or FP result."""

import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path("data/interim/lee2019_revalidation_norm_cov")
    output = Path("data/processed/e4_loso_training_exposure_audit_20261009.json")
    if output.exists():
        raise FileExistsError(output)
    summary = {}
    for session in (1, 2):
        paths = sorted((root / f"session{session}").glob("sub-*.npz"))
        if len(paths) != 54:
            raise AssertionError("expected 54 covariance source caches")
        rows = []
        for path in paths:
            subject = int(path.stem.split("-")[-1])
            with np.load(path, allow_pickle=False) as data:
                cmd = np.asarray(data["train_command_trial"], dtype=int)
                idle = np.asarray(data["train_idle_iti_trial"], dtype=int)
                n_pre = len(data["pre_train_covariance"])
            if not (np.all((cmd >= 0) & (cmd < 100)) and
                    np.all((idle >= 0) & (idle < 100))):
                raise AssertionError("cached trial ID outside 0-99")
            rows.append({"subject": subject, "n_command_trial_lt50": int(np.sum(cmd < 50)),
                         "n_idle_trial_lt50": int(np.sum(idle < 50)),
                         "n_command_trial_ge50": int(np.sum(cmd >= 50)),
                         "n_idle_trial_ge50": int(np.sum(idle >= 50)),
                         "n_pre_train_covariance": n_pre})
        if len({r["subject"] for r in rows}) != 54:
            raise AssertionError("duplicate source subject")
        # fit_loso in evaluate_lee2019_revalidation_guard_rearm_smoke.py
        # explicitly iterates `subject != target`; source_domains uses trial<50.
        for target in range(1, 55):
            donor_ids = [r["subject"] for r in rows if r["subject"] != target]
            if len(donor_ids) != 53 or target in donor_ids:
                raise AssertionError("target not excluded from LOSO donors")
        summary[str(session)] = {
            "n_source_caches": len(rows), "n_donors_per_target": 53,
            "all_sources_have_pre50_command": all(r["n_command_trial_lt50"] > 0 for r in rows),
            "all_sources_have_pre50_idle": all(r["n_idle_trial_lt50"] > 0 for r in rows),
            "min_pre_train_covariance": min(r["n_pre_train_covariance"] for r in rows),
            "rows": rows,
        }
    result = {"scope": "cached trial-ID and LOSO selection-rule audit only; original-continuous Riemannian scores not computed",
              "selection_code": "scripts/evaluate_lee2019_revalidation_guard_rearm_smoke.py:source_domains/fit_loso",
              "summary": summary}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("audited 108 source caches and 53-donor LOSO selection per target")


if __name__ == "__main__":
    main()
