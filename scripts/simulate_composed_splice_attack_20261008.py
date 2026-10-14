"""E1: hypothetical EEG splice detector followed by the causal gap attack.

This extends simulate_decoy_gap_attack_20261008.py without changing its legacy
metadata-only result. No EEG is read and no actual splice-detection AUC is measured.
The Gaussian ROC assumption merely maps an assumed AUC and false-seam rate to
splice sensitivity; AUC alone never fixes a detector operating point.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from simulate_decoy_gap_attack_20261008 import COLLAR, FP_PER_MIN, REPLAY, ROOT


OUT_DIR = ROOT / "data/processed/simulation_redesign_20261008"
NORM = NormalDist()
AUC_GRID = (0.50, 0.55, 0.60, 0.70)
FALSE_RATE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DECOY_GRID = (0.1, 0.2, 0.4)
OFFSET_GRID = (("fixed_3p5", 3.5), ("uniform_3_to_6", (3.0, 6.0)))


def splice_sensitivity(auc, false_seams_per_min, candidate_period_s=1.0):
    """TPR under an equal-variance Gaussian ROC with explicit negative opportunities."""
    fpr = false_seams_per_min * candidate_period_s / 60.0
    if not (0.0 < auc < 1.0 and 0.0 < fpr < 1.0):
        raise ValueError("AUC and candidate-level FPR must be inside (0, 1)")
    d_prime = np.sqrt(2.0) * NORM.inv_cdf(auc)
    return 1.0 - NORM.cdf(NORM.inv_cdf(1.0 - fpr) - d_prime)


def make_timeline(gaps, decoy_rate, onset_offset, rng):
    """Return all real splices, onsets, unique idle intervals, and stream end."""
    t, seams, onsets, idle = 0.0, [], [], []
    for gap in gaps:
        seams.append(t)  # rest-start splice
        seams.extend(rng.uniform(t, t + gap, rng.poisson(decoy_rate * gap)))
        idle.append((t, t + gap))
        t += gap
        seams.append(t)  # trial-start splice; detector cannot know its type
        offset = rng.uniform(*onset_offset) if isinstance(onset_offset, tuple) else onset_offset
        onsets.append(t + offset)
        t += offset + 3.5
    return np.sort(np.asarray(seams)), np.asarray(onsets), np.asarray(idle), float(t)


def detect_splice_bins(seams, horizon, marks, sensitivity, false_seams_per_min,
                       candidate_period_s=1.0):
    """Apply the ROC model on every candidate bin, including non-splice bins.

    At AUC=0.5 sensitivity equals background FPR, so the output is independent
    of seam locations. A real splice merely marks its containing bin as positive.
    Detection is emitted at the *end* of the bin to preserve causality.
    """
    n_bins = int(np.ceil(horizon / candidate_period_s))
    if len(marks) != n_bins:
        raise ValueError("one fixed uniform mark is needed for each candidate bin")
    fpr = false_seams_per_min * candidate_period_s / 60.0
    probabilities = np.full(n_bins, fpr)
    seam_bins = np.unique(np.minimum((seams / candidate_period_s).astype(int), n_bins - 1))
    probabilities[seam_bins] = sensitivity
    detected = (np.flatnonzero(marks < probabilities) + 1) * candidate_period_s
    return detected[detected <= horizon]


def composed_attack(observed_seams, onsets, idle, horizon, wait_s=3.5):
    """Expected hit and FP with diagnostic, target-idle-oracle FP2 thinning.

    A seam is eligible only if no other *observed* seam appeared over wait_s.
    Random thinning probability is chosen from the resulting true idle alarms;
    this is an attacker-favorable diagnostic, not a deployable threshold.
    """
    if len(observed_seams) == 0:
        return 0.0, 0.0, 0, 0
    observed = np.sort(np.asarray(observed_seams))
    following_gap = np.diff(np.r_[observed, np.inf])
    fire = observed[(following_gap > wait_s) & (observed + wait_s <= horizon)] + wait_s
    if len(fire) == 0:
        return 0.0, 0.0, 0, 0
    in_idle = ((fire[:, None] >= idle[None, :, 0]) &
               (fire[:, None] < idle[None, :, 1])).any(axis=1)
    n_idle = int(in_idle.sum())
    exposure_s = float(np.sum(idle[:, 1] - idle[:, 0]))
    budget = FP_PER_MIN * exposure_s / 60.0
    keep = min(1.0, budget / n_idle) if n_idle else 1.0
    alarms_in_collar = ((fire[None, :] >= onsets[:, None]) &
                        (fire[None, :] <= onsets[:, None] + COLLAR)).sum(axis=1)
    expected_hit = np.mean(1.0 - (1.0 - keep) ** alarms_in_collar)
    return float(expected_hit), float(n_idle * keep * 60.0 / exposure_s), n_idle, len(fire)


def chance_reference(reps, rng, n_subjects=54, n_onsets=30, draws=10000):
    """Cohort-median reference after averaging technical repeats within subject."""
    p = 1.0 - np.exp(-FP_PER_MIN * COLLAR / 60.0)
    subject_hit = rng.binomial(n_onsets, p, size=(draws, n_subjects, reps)).mean(axis=2) / n_onsets
    return p, float(np.quantile(np.median(subject_hit, axis=1), 0.95))


def run(subjects=54, reps=3):
    if not (1 <= subjects <= 54 and reps >= 1):
        raise ValueError("subjects must be 1..54 and reps must be positive")
    rng = np.random.default_rng(20261008)
    rows = []
    for session in (1, 2):
        for subject in range(1, subjects + 1):
            with np.load(REPLAY / f"session{session}/sub-{subject:02d}.npz") as cache:
                events = cache["events"][20:]
            gaps = (events[:, 3] - events[:, 2]) / 250.0
            for offset_name, offset in OFFSET_GRID:
                for decoy_rate in DECOY_GRID:
                    for rep in range(reps):
                        seams, onsets, idle, horizon = make_timeline(gaps, decoy_rate, offset, rng)
                        # Common random numbers preserve paired scenario comparisons.
                        n_bins = int(np.ceil(horizon))
                        seam_bins = np.unique(np.minimum(seams.astype(int), n_bins - 1))
                        detector_marks = rng.uniform(size=n_bins)
                        for auc in AUC_GRID:
                            for false_rate in FALSE_RATE_GRID:
                                sensitivity = splice_sensitivity(auc, false_rate)
                                observed = detect_splice_bins(
                                    seams, horizon, detector_marks, sensitivity, false_rate)
                                detected_bins = observed.astype(int) - 1
                                detected_true = np.intersect1d(detected_bins, seam_bins).size
                                detected_false = len(detected_bins) - detected_true
                                hit, fp, n_idle, n_total = composed_attack(observed, onsets, idle, horizon)
                                rows.append({
                                    "session": session, "subject": subject, "rep": rep,
                                    "onset_offset": offset_name, "decoy_rate_per_s": decoy_rate,
                                    "auc_assumed": auc, "false_seams_per_min_assumed": false_rate,
                                    "splice_sensitivity_implied": sensitivity,
                                    "splice_sensitivity_realized": detected_true / len(seam_bins),
                                    "false_seams_per_min_realized": detected_false * 60.0 / (n_bins - len(seam_bins)),
                                    "attack_expected_hit": hit, "attack_expected_fp_per_min": fp,
                                    "candidate_idle_alarms": n_idle, "candidate_all_alarms": n_total,
                                    "unique_idle_s": float(np.sum(idle[:, 1] - idle[:, 0])),
                                })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row_path = OUT_DIR / "composed_splice_attack_rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p_chance, chance_upper = chance_reference(reps, np.random.default_rng(20261009), subjects)
    summary = []
    bootstrap_rng = np.random.default_rng(20261010)
    for session in (1, 2):
        for offset_name, _ in OFFSET_GRID:
            for decoy_rate in DECOY_GRID:
                for auc in AUC_GRID:
                    for false_rate in FALSE_RATE_GRID:
                        selected = [r for r in rows if r["session"] == session and
                                    r["onset_offset"] == offset_name and
                                    r["decoy_rate_per_s"] == decoy_rate and
                                    r["auc_assumed"] == auc and
                                    r["false_seams_per_min_assumed"] == false_rate]
                        subject_hits = np.array([
                            np.mean([r["attack_expected_hit"] for r in selected if r["subject"] == subject])
                            for subject in range(1, subjects + 1)])
                        subject_fps = np.array([
                            np.mean([r["attack_expected_fp_per_min"] for r in selected if r["subject"] == subject])
                            for subject in range(1, subjects + 1)])
                        baseline = [r for r in rows if r["session"] == session and
                                    r["onset_offset"] == offset_name and
                                    r["decoy_rate_per_s"] == decoy_rate and
                                    r["auc_assumed"] == 0.5 and
                                    r["false_seams_per_min_assumed"] == false_rate]
                        baseline_hits = np.array([
                            np.mean([r["attack_expected_hit"] for r in baseline if r["subject"] == subject])
                            for subject in range(1, subjects + 1)])
                        paired_delta = subject_hits - baseline_hits
                        resampled = bootstrap_rng.integers(0, subjects, size=(2000, subjects))
                        interval = np.quantile(np.median(paired_delta[resampled], axis=1), (0.025, 0.975))
                        summary.append({
                            "session": session, "onset_offset": offset_name,
                            "decoy_rate_per_s": decoy_rate, "auc_assumed": auc,
                            "false_seams_per_min_assumed": false_rate,
                            "splice_sensitivity_implied": splice_sensitivity(auc, false_rate),
                            "splice_sensitivity_realized_median": float(np.median([
                                r["splice_sensitivity_realized"] for r in selected])),
                            "false_seams_per_min_realized_median": float(np.median([
                                r["false_seams_per_min_realized"] for r in selected])),
                            "attack_subject_median_hit": float(np.median(subject_hits)),
                            "attack_subject_median_fp_per_min": float(np.median(subject_fps)),
                            "paired_median_hit_minus_auc_half": float(np.median(paired_delta)),
                            "paired_bootstrap_ci95_hit_minus_auc_half": [float(x) for x in interval],
                            "above_poisson_chance_median_p95": bool(np.median(subject_hits) > chance_upper),
                        })
    payload = {
        "status": "E1 hypothetical metadata-only simulation; not EEG G1 validation",
        "seed": 20261008, "subjects_per_session": subjects,
        "reps_per_subject_session": reps,
        "auc_model": "equal-variance Gaussian ROC, one independent background candidate per second",
        "splice_detection": "all one-second candidate bins sampled; bins with >=1 real splice use TPR",
        "false_splices": "independent Bernoulli non-splice bins at assumed false seams/min",
        "attack": "causal 3.5-s no-next-observed-seam gap, target-idle-oracle FP2 thinning",
        "seam_guard_s": 0.0,
        "chance_poisson_hit": p_chance,
        "chance_cohort_median_p95": chance_upper,
        "chance_note": "Poisson reference is not the primary gate; paired AUC=0.5 seam-blind control preserves schedule and FP2 thinning",
        "results": summary,
    }
    output_path = OUT_DIR / "composed_splice_attack.json"
    output_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"Saved {len(rows)} rows, {len(summary)} cells to {output_path}")
    print(f"Poisson hit={p_chance:.4f}, cohort-median 95th percentile={chance_upper:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=54)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    run(args.subjects, args.reps)
