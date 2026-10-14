"""Descriptive audit of the fixed 3-subject A1 EEG-sham engineering pilot.

The 20 draws per subject-session are repeated simulations of the same EEG,
not 20 independent people. This script deliberately makes no CI or p-value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/processed/performance_revalidation_20261001"
SUBJECTS = (3, 28, 54)
SESSIONS = (1, 2)
MODES = ("rest_only", "cue_preserving")
FPS = ("2.0", "4.0")


def run() -> dict:
    rows = []
    for subject in SUBJECTS:
        for session in SESSIONS:
            data = {}
            for mode in MODES:
                qualifier = "aligned_" if mode == "cue_preserving" else ""
                stem = f"a1_structural_20261002_{qualifier}{mode}_sub{subject:02d}_s{session}"
                with np.load(RESULTS / f"{stem}.npz", allow_pickle=False) as saved:
                    data[mode] = {key: np.asarray(saved[key]) for key in saved.files}
                metadata = json.loads((RESULTS / f"{stem}.json").read_text(encoding="utf-8"))
                summary = json.loads(
                    (RESULTS / f"{stem}_matched_fp.json").read_text(encoding="utf-8")
                )
                assert metadata["subject"] == subject and metadata["session"] == session
                assert metadata["sham_mode"] == mode and metadata["replicates"] == 20
                assert summary["replicates"] == 20 and summary["future_trials"] == 30
                assert metadata["retrained_observed_score_max_difference"] <= 1e-5
                assert metadata["retrained_anchor_score_max_difference"] <= 1e-5
                assert metadata["repeated_original_rest_eeg_max_difference"] <= 1e-5
                data[mode]["summary"] = summary
            rest, cue = data["rest_only"], data["cue_preserving"]
            for key in ("source_draws", "observed_raw", "anchor_scores", "times", "allowed", "events"):
                if not np.array_equal(rest[key], cue[key]):
                    raise AssertionError(f"unpaired {key}: sub{subject:02d} s{session}")
            for fp in FPS:
                rest_result = rest["summary"]["target_fps"][fp]
                cue_result = cue["summary"]["target_fps"][fp]
                if rest_result["matched_draws"] != 20 or cue_result["matched_draws"] != 20:
                    raise AssertionError(f"unmatched draw at FP{fp}: sub{subject:02d} s{session}")
                rest_excess = np.asarray(rest_result["excess_sen2"], dtype=float)
                cue_excess = np.asarray(cue_result["excess_sen2"], dtype=float)
                paired_delta = cue_excess - rest_excess
                rows.append({
                    "subject": subject,
                    "session": session,
                    "diagnostic_fp_target_per_min": float(fp),
                    "draws_per_mode": 20,
                    "source_rest_draws_identical": True,
                    "rest_samples_after_crossfade_aligned_by_construction": True,
                    "observed_and_anchor_scores_identical": True,
                    "rest_only_median_excess_sen2": float(np.median(rest_excess)),
                    "cue_preserving_median_excess_sen2": float(np.median(cue_excess)),
                    "paired_draw_delta_cue_minus_rest_median": float(np.median(paired_delta)),
                    "paired_draw_delta_min_max": [float(np.min(paired_delta)), float(np.max(paired_delta))],
                    "paired_draw_delta_positive_zero_negative": [
                        int(np.sum(paired_delta > 0)),
                        int(np.sum(paired_delta == 0)),
                        int(np.sum(paired_delta < 0)),
                    ],
                    "rest_only_median_achieved_fp_per_min": rest_result["median_achieved_fp_per_min"],
                    "cue_preserving_median_achieved_fp_per_min": cue_result["median_achieved_fp_per_min"],
                })
    result = {
        "scope": "fixed 3 subjects x 2 sessions x 2 EEG-level shams x 20 draws; engineering pilot only",
        "data": "Lee2019 cue-relative chronological pseudo-uncued simulation",
        "estimand": "within-cell descriptive sensitivity to preserving EEG through cue+0.5 s",
        "not_inferred": [
            "MI-specificity", "cohort Type-I error or CI coverage", "deployment performance",
            "independent effect of a splice transient",
        ],
        "unit_of_analysis": "six subject-session cells; draws are paired simulation realizations",
        "rows": rows,
    }
    output = RESULTS / "a1_structural_20261002_pilot_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    for row in result["rows"]:
        print(
            f"sub{row['subject']:02d} s{row['session']} FP{row['diagnostic_fp_target_per_min']:.0f}: "
            f"rest {row['rest_only_median_excess_sen2']:+.3f}, "
            f"cue {row['cue_preserving_median_excess_sen2']:+.3f}, "
            f"paired-draw delta {row['paired_draw_delta_cue_minus_rest_median']:+.3f}"
        )
