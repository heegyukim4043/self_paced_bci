"""Metadata-only G1 attack against a fixed number of decoy seams per rest.

This is a structural falsification, not an EEG or actual re-splice analysis.
The attacker sees only the ordered seam times and knows the simulator's fixed
number K of decoys per rest interval. No cue, task label, or EEG is supplied
to the score constructor. The original event table is used only to generate
the test schedule and to score the resulting alarms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluate_lee2019_rest_sham_a1 import evaluate_raw

SFREQ = 250.


def seam_schedule(events: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Make an ordered, unlabeled seam list with exactly k decoys per rest."""
    rng = np.random.default_rng(seed)
    seams = []
    for i, event in enumerate(events):
        rest_start, rest_end, trial_start = map(int, event[[2, 3, 4]])
        assert rest_end == trial_start and rest_end > rest_start
        if i:
            seams.append(rest_start)  # trial-end / next-rest-start seam
        width = rest_end - rest_start
        # Random interior positions, order preserved; all are feasible splits.
        fractions = (np.arange(1, k + 1) + rng.uniform(-.15, .15, k)) / (k + 1)
        decoys = rest_start + np.rint(fractions * width).astype(int)
        assert np.all(np.diff(decoys) > 0)
        assert np.all((decoys > rest_start) & (decoys < rest_end))
        seams.extend(decoys.tolist())
        seams.append(trial_start)
    return np.asarray(seams, dtype=np.int64)


def decode_trial_starts(seams: np.ndarray, k: int) -> np.ndarray:
    """EEG-blind ordinal decoder; first seam is decoy 1 of the first rest."""
    return seams[np.arange(k, len(seams), k + 2)]


def allowed_after_guard(times: np.ndarray, starts: np.ndarray,
                        seams: np.ndarray, guard_seconds: float) -> np.ndarray:
    allowed = np.ones(len(times), dtype=np.bool_)
    for seam in seams / SFREQ:
        left = np.searchsorted(times, seam, side="right")
        right = np.searchsorted(starts, seam + guard_seconds, side="left")
        if right > left:
            allowed[left:right] = False
    return allowed


def evaluate_case(times: np.ndarray, starts: np.ndarray, events: np.ndarray,
                  seams: np.ndarray, k: int, guard_seconds: float) -> dict:
    inferred = decode_trial_starts(seams, k)
    actual = events[:, 4].astype(np.int64)
    assert np.array_equal(inferred, actual), "seam count did not identify trial starts"
    allowed = allowed_after_guard(times, starts, seams, guard_seconds)
    raw = np.full(len(times), -1., dtype=np.float64)
    for trial_start in inferred / SFREQ:
        raw[(times >= trial_start + 4.6) &
            (times <= trial_start + 5.5)] = 1.
    cues = events[20:, 5] / SFREQ
    onsets = cues + .5
    future = times >= events[20, 2] / SFREQ
    task = np.zeros(len(times), dtype=np.bool_)
    for cue in cues:
        task |= (times >= cue + .5) & (times <= cue + 4.)
    idle = future & allowed & ~task
    hits, fp = evaluate_raw(raw, -1., .01, allowed, idle, times, onsets, .5)
    exposure = float(idle.sum() * .1)
    return {
        "seam_count": int(len(seams)),
        "inferred_trial_starts": int(len(inferred)),
        "future_onsets": int(len(onsets)),
        "hit_count_2s": int(hits),
        "idle_fp_count": int(fp),
        "valid_idle_seconds": exposure,
        "fp_per_min": float(fp * 60. / exposure) if exposure else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "data/processed/simulation_redesign_20261008/"
                        "g1_fixed_decoy_seam_count_attack.json")
    args = parser.parse_args()
    score_root = ROOT / "data/interim/lee2019_revalidation_riemannian_scores"
    configs = [(3, 3.), (3, 1.), (6, 1.)]
    rows = []
    for session in (1, 2):
        for subject in range(1, 55):
            with np.load(score_root / f"session{session}/sub-{subject:02d}.npz",
                         allow_pickle=False) as score:
                times = score["times"].astype(np.float64)
                starts = score["starts"].astype(np.float64)
                events = score["events"].astype(np.int64)
                cached_allowed = score["allowed"].astype(bool)
            original_seams = np.unique(np.r_[events[:, 4], events[1:, 2]])
            assert np.array_equal(
                allowed_after_guard(times, starts, original_seams, 3.),
                cached_allowed,
            ), "guard reconstruction differs from locked cache"
            for k, guard in configs:
                for placement_seed in range(args.seeds):
                    seed = 100000 * session + 1000 * subject + placement_seed
                    seams = seam_schedule(events, k, seed)
                    result = evaluate_case(times, starts, events, seams, k, guard)
                    rows.append({"subject": subject, "session": session,
                                 "k": k, "guard_seconds": guard,
                                 "placement_seed": placement_seed, **result})
    by_config = {}
    for k, guard in configs:
        subset = [row for row in rows if row["k"] == k and
                  row["guard_seconds"] == guard]
        key = f"k{k}_guard{guard:g}"
        exposure = [row["valid_idle_seconds"] for row in subset]
        by_config[key] = {
            "rows": len(subset),
            "onsets": sum(row["future_onsets"] for row in subset),
            "hits_2s": sum(row["hit_count_2s"] for row in subset),
            "idle_fp": sum(row["idle_fp_count"] for row in subset),
            "idle_exposure_median_seconds": float(np.median(exposure)),
            "idle_exposure_min_seconds": float(np.min(exposure)),
            "zero_idle_exposure_rows": sum(value == 0 for value in exposure),
        }
    output = {
        "status": "metadata-only structural attack; not an EEG re-splice or model performance estimate",
        "score": "-1 except +1 at decoded trial-start seam +4.6 to +5.5 s; no EEG",
        "decision": "existing EWMA H30, MA5, rearm, dwell3, refractory10; threshold 0.5",
        "warning": "fixed K and stream-start phase reveal true trial starts by seam ordinal",
        "by_config": by_config,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(by_config, indent=2))


if __name__ == "__main__":
    main()
