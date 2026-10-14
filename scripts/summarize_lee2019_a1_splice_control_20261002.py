"""Three-way common-FP audit of A1 rest/cue/splice engineering controls.

All three sham curves and the observed curve must share the same *integer*
achieved FP count per draw. No p-value/CI: draws reuse the same six EEG cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.evaluate_lee2019_rest_sham_matched_fp import (
    compute_guard_local_ma, matched_curve,
)

RESULTS = ROOT / "data/processed/performance_revalidation_20261001"
MODES = ("rest_only", "cue_preserving", "rest_rest_splice")


def stem(mode: str, subject: int, session: int) -> str:
    qualifier = "aligned_" if mode == "cue_preserving" else ""
    return f"a1_structural_20261002_{qualifier}{mode}_sub{subject:02d}_s{session}"


def load_cell(subject: int, session: int) -> dict:
    loaded = {}
    for mode in MODES:
        name = stem(mode, subject, session)
        with np.load(RESULTS / f"{name}.npz", allow_pickle=False) as saved:
            loaded[mode] = {key: np.asarray(saved[key]) for key in saved.files}
        metadata = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
        if (metadata["subject"], metadata["session"], metadata["sham_mode"],
            metadata["replicates"]) != (subject, session, mode, 20):
            raise AssertionError(f"metadata mismatch: {name}")
        for key in ("retrained_observed_score_max_difference",
                    "retrained_anchor_score_max_difference",
                    "repeated_original_rest_eeg_max_difference"):
            if metadata[key] > 1e-5:
                raise AssertionError(f"cache/source mismatch {name}: {key}")
    rest = loaded["rest_only"]
    for mode in MODES[1:]:
        for key in ("source_draws", "observed_raw", "anchor_scores", "times",
                    "allowed", "events"):
            if not np.array_equal(rest[key], loaded[mode][key]):
                raise AssertionError(f"unpaired {key}: {subject} S{session} {mode}")
    splice = loaded["rest_rest_splice"]
    if np.any(np.all(splice["source_draws"] == splice["pre_source_draws"], axis=-1)):
        raise AssertionError("rest-rest splice reuses identical pre/post source")
    return loaded


def run() -> dict:
    rows = []
    for subject in (3, 28, 54):
        for session in (1, 2):
            data = load_cell(subject, session)
            rest = data["rest_only"]
            times = np.asarray(rest["times"], dtype=float)
            allowed = np.asarray(rest["allowed"], dtype=bool)
            events = np.asarray(rest["events"], dtype=np.int64)
            if len(events) != 50:
                raise AssertionError("expected 50 replay events")
            future = times >= events[20, 2] / 250.
            task = np.zeros(len(times), dtype=bool)
            hit_label = np.full(len(times), -1, dtype=np.int32)
            for trial, event in enumerate(events[20:]):
                cue = event[5] / 250.
                onset = cue + .5
                task |= (times >= onset) & (times <= cue + 4.)
                hit_label[(times >= onset) & (times <= onset + 2.)] = trial
            idle = future & allowed & ~task
            exposure_minutes = float(np.sum(idle) * .1 / 60.)
            anchor = np.asarray(rest["anchor_scores"], dtype=float)
            mean, var = float(anchor.mean()), float(anchor.var(ddof=1))
            common_args = (allowed, future, idle, hit_label, 301)
            observed = matched_curve(compute_guard_local_ma(
                np.asarray(rest["observed_raw"], dtype=float), mean, var, allowed,
            ), *common_args)
            curves = {mode: [matched_curve(compute_guard_local_ma(
                np.asarray(raw, dtype=float), mean, var, allowed,
            ), *common_args) for raw in data[mode]["sham_raw_scores"]]
                      for mode in MODES}
            if any(len(curves[mode]) != 20 for mode in MODES):
                raise AssertionError("expected 20 draws per mode")
            for target_fp in (2., 4.):
                target_count = target_fp * exposure_minutes
                counts = []
                excess = {mode: [] for mode in MODES}
                sham_hits = {mode: [] for mode in MODES}
                observed_hits = []
                for draw in range(20):
                    shared = set(observed)
                    for mode in MODES:
                        shared &= set(curves[mode][draw])
                    if not shared:
                        continue
                    count = min(shared, key=lambda x: (abs(x - target_count), x))
                    counts.append(count)
                    hits = observed[count][1]
                    observed_hits.append(hits)
                    for mode in MODES:
                        null_hits = curves[mode][draw][count][1]
                        sham_hits[mode].append(null_hits)
                        excess[mode].append((hits - null_hits) / 30.)
                if len(counts) != 20:
                    raise AssertionError(
                        f"no three-way FP match: sub{subject:02d} S{session} FP{target_fp}"
                    )
                rows.append({
                    "subject": subject, "session": session,
                    "target_fp_per_min": target_fp,
                    "matched_draws": len(counts),
                    "achieved_fp_min_median_max_per_min": [
                        float(np.min(counts) / exposure_minutes),
                        float(np.median(counts) / exposure_minutes),
                        float(np.max(counts) / exposure_minutes),
                    ],
                    "observed_sen2_median": float(np.median(observed_hits) / 30.),
                    "median_excess_sen2": {
                        mode: float(np.median(excess[mode])) for mode in MODES
                    },
                    "median_sham_sen2": {
                        mode: float(np.median(sham_hits[mode]) / 30.) for mode in MODES
                    },
                    "draws_with_splice_more_hits_than_rest_only": int(np.sum(
                        np.asarray(sham_hits["rest_rest_splice"]) >
                        np.asarray(sham_hits["rest_only"]))),
                    "draws_with_splice_more_hits_than_cue_preserving": int(np.sum(
                        np.asarray(sham_hits["rest_rest_splice"]) >
                        np.asarray(sham_hits["cue_preserving"]))),
                    "paired_splice_minus_rest_sham_hits_min_median_max": [
                        int(np.min(np.asarray(sham_hits["rest_rest_splice"]) -
                                   np.asarray(sham_hits["rest_only"]))),
                        float(np.median(np.asarray(sham_hits["rest_rest_splice"]) -
                                        np.asarray(sham_hits["rest_only"]))),
                        int(np.max(np.asarray(sham_hits["rest_rest_splice"]) -
                                   np.asarray(sham_hits["rest_only"]))),
                    ],
                })
    result = {
        "scope": "post-hoc engineering stress control, six fixed Lee2019 cue-relative simulated cells",
        "fp_matching": "same integer achieved idle FP count shared by observed and all three shams per draw",
        "unit": "subject-session; 20 repeated sham draws are not independent participants",
        "claim_limit": "Rest-rest splice response can implicate splice/schedule artifact; absence does not validate cue-preserving sham or MI specificity",
        "rows": rows,
    }
    output = RESULTS / "a1_rest_rest_splice_common_fp_summary_20261002.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    for row in result["rows"]:
        values = row["median_excess_sen2"]
        print(f"sub{row['subject']:02d} S{row['session']} FP{row['target_fp_per_min']:.0f}: "
              f"rest {values['rest_only']:+.3f}, cue {values['cue_preserving']:+.3f}, "
              f"rest-splice {values['rest_rest_splice']:+.3f}; "
              f"splice>rest {row['draws_with_splice_more_hits_than_rest_only']}/20")
