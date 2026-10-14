"""Summarize a bounded five-placement-seed A5 pilot (two subjects only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/performance_revalidation_20261001"
SUBJECTS = (3, 54)
SESSIONS = (1, 2)
SEEDS = range(20261001, 20261006)


def main() -> None:
    baseline = json.loads((OUT / "a2_chronological_anchor_operating_points.json").read_text(
        encoding="utf-8"))
    original = {(row["session"], row["subject"], str(row["target_fp"])):
                row["riemannian"] for row in baseline["rows"]}
    rows = []
    for seed in SEEDS:
        if seed == 20261001:
            continue
        payload = json.loads((OUT / f"a5_placement_probe_sub03_sub54_seed{seed}.json").read_text(
            encoding="utf-8"))
        for row in payload["rows"]:
            for fp in ("2.0", "4.0"):
                target = row["targets"][fp]
                rows.append({"seed": seed, "subject": row["subject"],
                             "session": row["session"], "target_fp": fp,
                             "threshold": target["anchor"]["threshold"],
                             "future_fp_per_min": target["future_fp_per_min"],
                             "sen2": target["future_sen_at_2s"]})
    for subject in SUBJECTS:
        for session in SESSIONS:
            for fp in ("2.0", "4.0"):
                point = original[(session, subject, fp)]
                rows.append({"seed": 20261001, "subject": subject,
                             "session": session, "target_fp": fp,
                             "threshold": point["threshold"],
                             "future_fp_per_min": point["future_fp_per_min"],
                             "sen2": point["future_sen2"]})
            base_path = ROOT / f"data/interim/lee2019_revalidation_chronological/session{session}/sub-{subject:02d}.npz"
            with np.load(base_path, allow_pickle=False) as baseline_replay:
                base_ids = baseline_replay["trial_ids"].copy()
                base_spans = baseline_replay["trial_source_spans_250hz"].copy()
                base_count = len(baseline_replay["signal"])
            for seed in list(SEEDS)[1:]:
                path = ROOT / f"data/interim/lee2019_revalidation_placement_{seed}/session{session}/sub-{subject:02d}.npz"
                with np.load(path, allow_pickle=False) as replay:
                    assert np.array_equal(replay["trial_ids"], base_ids)
                    assert np.array_equal(replay["trial_source_spans_250hz"], base_spans)
                    assert len(replay["signal"]) == base_count
    by_cell = []
    for subject in SUBJECTS:
        for session in SESSIONS:
            for fp in ("2.0", "4.0"):
                selected = sorted((row for row in rows if row["subject"] == subject and
                                   row["session"] == session and row["target_fp"] == fp),
                                  key=lambda row: row["seed"])
                thresholds = [row["threshold"] for row in selected]
                if max(thresholds) - min(thresholds) > 1e-5:
                    raise AssertionError("placement changed supposedly frozen anchor threshold")
                fps = [row["future_fp_per_min"] for row in selected]
                sens = [row["sen2"] for row in selected]
                by_cell.append({"subject": subject, "session": session,
                                "target_fp": fp, "threshold": thresholds[0],
                                "fp_min_max_per_min": [min(fps), max(fps)],
                                "fp_spread_per_min": max(fps) - min(fps),
                                "sen2_min_max": [min(sens), max(sens)],
                                "sen2_spread": max(sens) - min(sens)})
    result = {"scope": "exploratory placement-only sensitivity, not locked A5 validation",
              "dataset": "Lee2019 cue-relative pseudo-uncued replay",
              "subjects": list(SUBJECTS), "sessions": list(SESSIONS),
              "placement_seeds": list(SEEDS),
              "constant": "same source-pooled Riemannian model recipe, target pre-rest anchor threshold, original MI trial order and identity, 840 s replay rest budget",
              "not_tested": "idle:MI ratio variation, full 54-subject cohort, null-corrected excess",
              "rows": rows, "by_cell": by_cell}
    output = OUT / "a5_placement_pilot_2subjects_5seeds_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(by_cell, indent=2))


if __name__ == "__main__":
    main()
