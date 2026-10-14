"""Audit frozen chronological replay geometry and FP2 decision boundaries.

This is a structural simulation audit, not a null-corrected EEG result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_locked_selfpaced_metrics import intervals_from_alarm
from scripts.evaluate_lee2019_rest_sham_matched_fp import compute_guard_local_ma
from uncued_mi.decision import guard_rearmed_alarms


def main() -> None:
    replay_root = ROOT / "data/interim/lee2019_revalidation_chronological"
    score_roots = {
        "riemannian": ROOT / "data/interim/lee2019_revalidation_riemannian_scores",
        "cnn_gru": ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache",
    }
    operating = json.loads((ROOT / "data/processed/performance_revalidation_20261001/"
                            "a2_chronological_anchor_operating_points.json").read_text(encoding="utf-8"))
    threshold_by_cell = {(r["session"], r["subject"]): r for r in operating["rows"]
                         if r["target_fp"] == 2.}
    rows = []
    for session in (1, 2):
        for subject in range(1, 55):
            cell = (session, subject)
            with np.load(replay_root / f"session{session}/sub-{subject:02d}.npz",
                         allow_pickle=False) as replay:
                events = np.asarray(replay["events"], dtype=np.int64)
                trial_ids = np.asarray(replay["trial_ids"])
                source_trials = np.asarray(replay["all_train_trial_source_spans_250hz"])
                meta = json.loads(str(replay["metadata_json"].item()))
                assert np.array_equal(trial_ids, np.arange(50, 100))
                assert np.array_equal(events[:, 1], trial_ids)
                assert np.array_equal(replay["trial_source_spans_250hz"], source_trials[50:100])
                assert source_trials[:50, 1].max() <= source_trials[70:100, 0].min()
                assert source_trials[50:70, 1].max() <= source_trials[70:100, 0].min()
                assert np.all(events[:, 6] - events[:, 5] == 125)
                assert np.all(events[:, 7] - events[:, 5] == 875)
                assert meta["task_ground_truth_seconds_from_cue"] == [.5, 4.]
                assert meta["training_positive_support_seconds_from_cue"] == [.5, 3.5]
                assert meta["idle_seconds_per_fold"] == 168.
                assert meta["rest_repartition_folds"] == 5
            paths = {name: root / f"session{session}/sub-{subject:02d}.npz"
                     for name, root in score_roots.items()}
            cell_row = {"subject": subject, "session": session}
            baseline = None
            for model, path in paths.items():
                with np.load(path, allow_pickle=False) as data:
                    times = np.asarray(data["times"], dtype=float)
                    allowed = np.asarray(data["allowed"], dtype=bool)
                    assert np.array_equal(data["events"], events)
                    if baseline is not None:
                        assert np.array_equal(times, baseline[0])
                        assert np.array_equal(allowed, baseline[1])
                    baseline = times, allowed
                    raw = np.asarray(data["raw_scores"], dtype=float)
                    anchor = np.asarray(data["anchor_raw_scores" if model == "riemannian"
                                             else "anchor_scores"], dtype=float)
                threshold = threshold_by_cell[cell][model]["threshold"]
                ma = compute_guard_local_ma(raw, float(anchor.mean()),
                                            float(anchor.var(ddof=1)), allowed)
                future = times >= events[20, 2] / 250.
                alarms = guard_rearmed_alarms(ma, threshold, allowed, 3, 10) & future
                assert not np.any(alarms & ~allowed)
                onsets = events[20:, 5] / 250. + .5
                hit_counts = np.array([np.sum(alarms & (times >= o) & (times <= o + 2.))
                                       for o in onsets])
                # Collars do not overlap: a single alarm cannot credit two trials.
                collar_starts = onsets
                assert np.all(collar_starts[1:] > collar_starts[:-1] + 2.)
                assert int(np.sum(hit_counts > 0)) == threshold_by_cell[cell][model]["future_onset_hits2"]
                pred = intervals_from_alarm(times, alarms, ma, threshold, allowed)
                for start, stop in pred:
                    left = int(np.searchsorted(times, start, side="left"))
                    right = int(np.searchsorted(times, stop, side="right"))
                    assert right > left and np.all(allowed[left:right]), (cell, model, start, stop)
                cell_row[model] = {"future_alarms": int(alarms.sum()),
                                   "onsets_with_duplicate_alarms": int(np.sum(hit_counts > 1)),
                                   "predicted_intervals": len(pred),
                                   "guard_crossing_intervals": 0}
            times, allowed = baseline
            # Optimistic reachability: first valid MA sample may be low, followed
            # by three high samples. Actual score constraints can only make this later.
            valid_ma = np.isfinite(compute_guard_local_ma(
                np.zeros(len(times)), 0., 1., allowed))
            possible = np.zeros(len(times), dtype=bool)
            for index in np.flatnonzero(valid_ma & ~np.r_[False, valid_ma[:-1]]):
                end = index
                while end + 1 < len(times) and valid_ma[end + 1]:
                    end += 1
                if index + 3 <= end:
                    possible[index + 3:end + 1] = True
            onsets = events[20:, 5] / 250. + .5
            for collar in (1., 2.):
                masks = [(times >= onset) & (times <= onset + collar) for onset in onsets]
                cell_row[f"optimistic_blind_fraction_{collar:g}s"] = float(np.mean([
                    1. - np.sum(possible & mask) / np.sum(mask) for mask in masks
                ]))
            rows.append(cell_row)
        print(f"A0 boundary audit S{session} complete", flush=True)
    summary = {
        "scope": "108 Lee2019 cue-relative pseudo-uncued simulation cells; FP2 pre-rest thresholds",
        "status": "structural audit only; A1 full null and A2 excess still gated",
        "checks": {"chronological_and_disjoint_trial_support": "108/108",
                   "task_vs_training_label_endpoint": "108/108",
                   "paired_stream_grid": "108/108",
                   "alarm_guard_and_interval_guard": "216/216 model-cell paths"},
        "median_optimistic_blind_fraction_1s": float(np.median([
            row["optimistic_blind_fraction_1s"] for row in rows])),
        "median_optimistic_blind_fraction_2s": float(np.median([
            row["optimistic_blind_fraction_2s"] for row in rows])),
        "duplicate_alarm_onsets_by_model": {model: int(sum(
            row[model]["onsets_with_duplicate_alarms"] for row in rows))
            for model in score_roots},
        "rows": rows,
    }
    output = ROOT / "data/processed/performance_revalidation_20261001/" \
             "a0_final_boundary_audit.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
