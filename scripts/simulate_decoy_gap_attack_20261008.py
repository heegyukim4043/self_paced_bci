"""Metadata-only bound: can random decoy splices hide trial seams from a causal gap attacker?

No EEG is used. For each Lee2019 chronological replay (future trials 70-99), rest gaps are taken from the
cached events. Decoy splices are a Poisson process inside each rest gap; the MI onset sits U s after its
trial-start seam. The attacker fires w s after any seam that is not followed by another seam within w s,
skips alarms inside the seam guard, and thins alarms at random to meet FP=2/min over guard-valid idle.
Output: median onset hit fraction (2-s collar) versus the Poisson chance level for the same collar.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data/interim/lee2019_revalidation_chronological"
OUT = ROOT / "data/processed/simulation_redesign_20261008/decoy_gap_attack_bound.json"
FP_PER_MIN, COLLAR, GUARD, REPS = 2.0, 2.0, 1.0, 3


def gaps():
    rows = []
    for session in (1, 2):
        for subject in range(1, 55):
            ev = np.load(REPLAY / f"session{session}/sub-{subject:02d}.npz")["events"][20:]
            rows.append((ev[:, 3] - ev[:, 2]) / 250.0)
    return rows


def one(gap, rate, offset, rng):
    t, seams, onsets, idle = 0.0, [], [], []
    for g in gap:
        seams.append(t)
        seams += sorted(rng.uniform(t, t + g, rng.poisson(rate * g)))
        idle.append((t, t + g))
        t += g
        u = rng.uniform(*offset) if isinstance(offset, tuple) else offset
        seams.append(t)
        onsets.append(t + u)
        t += u + 3.5
    seams, onsets = np.array(seams), np.array(onsets)
    w = float(np.mean(offset)) if isinstance(offset, tuple) else offset
    nxt = np.r_[seams[1:], np.inf]
    fire = seams[(nxt - seams) > w] + w
    last = seams[np.searchsorted(seams, fire, side="right") - 1]
    fire = fire[(fire - last) >= GUARD]
    in_idle = np.zeros(len(fire), bool)
    for a, b in idle:
        in_idle |= (fire >= a) & (fire < b)
    n_idle_seams = sum(1 for x in seams if any(a <= x < b for a, b in idle))
    exposure = sum(b - a for a, b in idle) - GUARD * n_idle_seams
    budget = FP_PER_MIN / 60 * max(exposure, 1.0)
    thin = min(1.0, budget / max(in_idle.sum(), 1e-9))
    hit = np.mean([np.any((fire >= o) & (fire <= o + COLLAR)) for o in onsets]) * thin
    return hit, exposure


def main():
    rng = np.random.default_rng(20261008)
    data = gaps()
    chance = 1 - np.exp(-FP_PER_MIN / 60 * COLLAR)
    results = []
    for offset in (3.5, (3.0, 6.0)):
        for rate in (0.0, 0.1, 0.2, 0.4):
            vals = np.array([one(g, rate, offset, rng) for g in data for _ in range(REPS)])
            results.append({"onset_offset_s": offset, "decoy_rate_per_s": rate,
                            "attack_hit_fraction_median": float(np.median(vals[:, 0])),
                            "idle_exposure_median_s": float(np.median(vals[:, 1]))})
            print(results[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "metadata-only bound; no EEG; simple gap attacker (not the strongest possible)",
                               "chance_2s_collar_fp2": chance, "guard_s": GUARD, "reps_per_session": REPS,
                               "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
