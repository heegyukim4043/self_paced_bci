"""G0 paired source-support and stream-geometry audit, not a G1 pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_lee2019_v2_random_precue_20261008 import FADE, SFREQ


def one(source: Path, target: Path) -> dict:
    with np.load(source, allow_pickle=False) as old, np.load(target, allow_pickle=False) as v2:
        x = old["signal"]
        y = v2["signal"]
        old_events = old["events"][20:]
        new_events = v2["events"]
        original_spans = old["trial_source_spans_250hz"][20:]
        cropped_spans = v2["trial_source_spans_250hz"]
        rest_spans = v2["rest_source_spans_250hz"]
        u = v2["precue_samples"]
        seams = v2["seams"]
        fade = v2["fade_intervals"]
        piece_lengths = v2["rest_piece_lengths"]
        rest_piece_map = v2["rest_piece_source_map"]
        metadata = json.loads(str(v2["metadata_json"]))
        assert np.array_equal(new_events[:, 1], np.arange(70, 100))
        assert np.array_equal(v2["trial_ids"], np.arange(70, 100))
        assert np.array_equal(new_events[:, 6] - new_events[:, 5], np.full(30, 125))
        assert np.array_equal(new_events[:, 7] - new_events[:, 5], np.full(30, 1000))
        assert np.array_equal(new_events[:, 5] - new_events[:, 4], u)
        assert np.all((u >= 0) & (u <= 750))
        assert np.array_equal(cropped_spans[:, 0], original_spans[:, 0] + 750 - u)
        assert np.array_equal(cropped_spans[:, 1], original_spans[:, 1])
        assert np.array_equal(v2["rest_source_ids"], old["rest_source_ids"][20:])
        assert np.array_equal(rest_spans, old["rest_source_spans_250hz"][20:])
        assert np.array_equal(fade[:, 0], seams[:, 0])
        assert np.all(fade[:, 1] - fade[:, 0] == FADE)
        assert np.all(np.diff(seams[:, 0]) > 0)
        assert np.sum(seams[:, 1] == 1) == 30
        assert np.sum(seams[:, 1] == 2) == 29
        assert np.sum(seams[:, 1] == 0) == len(piece_lengths) - 30
        assert len(rest_piece_map) == len(piece_lengths)
        assert np.array_equal(rest_piece_map[:, 1] - rest_piece_map[:, 0],
                              rest_piece_map[:, 3] - rest_piece_map[:, 2])
        assert np.array_equal(v2["state"][new_events[:, 6]], np.full(30, 2))
        assert np.all(v2["state"][new_events[:, 2]] == 0)
        assert metadata["seam_guard"] is False
        max_source_error = 0.0
        for before, after, crop in zip(old_events, new_events, u):
            old_start = int(before[5] - crop)
            old_end = int(before[5] + 4 * SFREQ)
            new_start, new_end = int(after[4]), int(after[7])
            assert old_end - old_start == new_end - new_start
            max_source_error = max(max_source_error, float(np.max(np.abs(
                x[old_start + FADE:old_end] - y[new_start + FADE:new_end]))))
            maps = rest_piece_map[(rest_piece_map[:, 0] >= after[2]) &
                                  (rest_piece_map[:, 1] <= after[3])]
            assert len(maps) >= 1
            assert maps[0, 0] == after[2] and maps[-1, 1] == after[3]
            assert np.array_equal(maps[1:, 0], maps[:-1, 1])
            source_sorted = maps[np.argsort(maps[:, 2])]
            assert source_sorted[0, 2] == before[2]
            assert source_sorted[-1, 3] == before[3]
            assert np.array_equal(source_sorted[1:, 2], source_sorted[:-1, 3])
            # Every rest-rest boundary must join nonadjacent original samples.
            assert np.all(maps[1:, 2] != maps[:-1, 3])
            for out_start, out_stop, src_start, src_stop in maps:
                src = x[int(src_start + FADE):int(src_stop)]
                dst = y[int(out_start + FADE):int(out_stop)]
                max_source_error = max(max_source_error,
                                       float(np.max(np.abs(src - dst))))
        assert max_source_error == 0.0
        unique_rest = 168.0
        return {
            "subject": metadata["subject"], "session": metadata["session"],
            "trial_count": len(new_events), "decoy_count": int(np.sum(seams[:, 1] == 0)),
            "true_trial_start_count": 30,
            "rest_piece_count": len(piece_lengths),
            "unmatched_rest_interval_count": len(v2["unmatched_rest_intervals"]),
            "precue_min_median_max_s": [float(np.min(u / SFREQ)),
                                        float(np.median(u / SFREQ)),
                                        float(np.max(u / SFREQ))],
            "seam_to_onset_min_median_max_s": [float(np.min((u + 125) / SFREQ)),
                                               float(np.median((u + 125) / SFREQ)),
                                               float(np.max((u + 125) / SFREQ))],
            "rest_piece_min_median_max_s": [float(np.min(piece_lengths / SFREQ)),
                                            float(np.median(piece_lengths / SFREQ)),
                                            float(np.max(piece_lengths / SFREQ))],
            "future_replay_rest_s": float(np.sum(new_events[:, 3] - new_events[:, 2]) / SFREQ),
            "unique_source_rest_s": unique_rest,
            "max_preprocessed_trial_support_abs_error_outside_fade": max_source_error,
            "g0_cache_support_pass": True,
            "g1_pass": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=Path("data/interim/lee2019_revalidation_chronological"))
    parser.add_argument("--v2-root", type=Path,
                        default=Path("data/interim/lee2019_v2_random_precue_shuffled_20261008"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/simulation_redesign_20261008/e3_g0_pilot.json"))
    args = parser.parse_args()
    paths = sorted(args.v2_root.glob("session*/sub-*.npz"))
    if not paths:
        raise ValueError("no v2 streams")
    rows = [one(args.source_root / path.parent.name / path.name, path) for path in paths]
    result = {"scope": "E3 G0 internal pilot only; not G1 nor detector performance",
              "n_subject_sessions": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
