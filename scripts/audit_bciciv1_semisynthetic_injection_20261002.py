"""Bounded semi-synthetic check on native-continuous BCI IV-1 EEG.

This checks evaluation mechanics, not real self-initiated MI performance.
The frozen model/anchor/threshold split matches the 2026-10-02 factorial.
Synthetic events are placed only in known idle runs, and all sham placements
are sampled without looking at EEG scores or alarm outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from uncued_mi.config import ExperimentConfig
from uncued_mi.features import extract_covariances
from uncued_mi.riemannian import tangent_features
from audit_bciciv1_calibration_factorial_20261002 import (
    TARGET_FP, ma_from_raw, stage_masks,
)
from revalidate_bciciv1_continuous_20261002 import chronological_threshold, rearmed
from run_bciciv1_riemannian_sanity import (
    REAL_SUBJECTS, SFREQ, STEP_SECONDS, WINDOW_SECONDS, calibration_masks,
    downsample_true_labels, regularized_reference, support_labels,
)

SEED = 20261002
EVENT_SECONDS = 3.0
EVENTS_PER_SUBJECT = 20
AMPLITUDE_ATTENUATION = 0.5
TOLERANCE_SECONDS = 2.0
MIN_IDLE_RUN_SECONDS = 6.0


def eligible_idle_events(labels: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Candidate intervals with 1.5 s lead and at least 1.5 s trail."""
    idle = np.isfinite(labels) & (labels == 0)
    edges = np.diff(np.r_[False, idle, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = []
    for start, stop in zip(starts, stops):
        if (stop - start) / sfreq >= MIN_IDLE_RUN_SECONDS:
            onset = start / sfreq + 1.5
            result.append((onset, onset + EVENT_SECONDS))
    return np.asarray(result, dtype=float).reshape(-1, 2)


def inject_sensorimotor_erd(eeg: np.ndarray, events: np.ndarray,
                            channels: tuple[str, ...],
                            amplitude: float = AMPLITUDE_ATTENUATION,
                            sfreq: float = SFREQ) -> np.ndarray:
    """Subtract a phase-aligned 8-30 Hz component in motor channels during events.

    The half-cosine 0.3 s entry/exit avoids an abrupt broadband boundary.
    Zero-phase filtering is used only to manufacture synthetic raw EEG; the
    detector's filtering and decisions remain causal. This manipulation has
    no claim of physiological realism.
    """
    if not 0 <= amplitude <= 1:
        raise ValueError("amplitude must lie in [0, 1]")
    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim != 2 or eeg.shape[1] != len(channels):
        raise ValueError("EEG shape and channel names disagree")
    lookup = {name.casefold(): i for i, name in enumerate(channels)}
    chosen = [lookup[name.casefold()] for name in ("C3", "Cz", "C4")]
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
    component = sosfiltfilt(sos, eeg[:, chosen], axis=0)
    envelope = np.zeros(len(eeg), dtype=float)
    ramp = int(round(0.3 * sfreq))
    for start, stop in np.asarray(events, dtype=float).reshape(-1, 2):
        first, last = int(round(start * sfreq)), int(round(stop * sfreq))
        if first < 0 or last > len(eeg) or last - first < 2 * ramp:
            raise ValueError("event lies outside supported EEG")
        envelope[first:last] = 1.0
        edge = (1 - np.cos(np.linspace(0, np.pi, ramp))) / 2
        envelope[first:first + ramp] = edge
        envelope[last - ramp:last] = edge[::-1]
    modified = eeg.copy()
    modified[:, chosen] -= amplitude * envelope[:, None] * component
    return modified


def hit_map(times: np.ndarray, alarms: np.ndarray, onsets: np.ndarray,
            offsets: np.ndarray, tolerance: float = TOLERANCE_SECONDS) -> np.ndarray:
    """Candidate-event hits for unshifted and circularly shifted alarm trains."""
    result = np.zeros((len(onsets), len(offsets) + 1), dtype=bool)
    for column, offset in enumerate(np.r_[0, offsets]):
        alarm_times = times[np.flatnonzero(np.roll(alarms, int(offset)))]
        left = np.searchsorted(alarm_times, onsets, side="left")
        right = np.searchsorted(alarm_times, onsets + tolerance, side="right")
        result[:, column] = right > left
    return result


def placebo_audit(hits: np.ndarray, placements: int, events_per_draw: int,
                  seed: int) -> dict:
    """Conditional random-placement audit of the circular-alarm-shift null."""
    if hits.shape[0] < events_per_draw or hits.shape[1] < 2:
        raise ValueError("insufficient candidate events or null shifts")
    rng = np.random.default_rng(seed)
    obs, null_median, p_values, inside_band = [], [], [], []
    for _ in range(placements):
        selected = rng.choice(hits.shape[0], events_per_draw, replace=False)
        rates = np.mean(hits[selected], axis=0)
        observed, null = float(rates[0]), rates[1:]
        obs.append(observed)
        null_median.append(float(np.median(null)))
        p_values.append(float((1 + np.sum(null >= observed)) / (len(null) + 1)))
        bounds = np.quantile(null, [0.025, 0.975])
        inside_band.append(bool(bounds[0] <= observed <= bounds[1]))
    excess = np.asarray(obs) - np.asarray(null_median)
    return {
        "draws": placements,
        "conditional_rejection_fraction_p_le_0_05": float(np.mean(np.asarray(p_values) <= 0.05)),
        "null_predictive_95pct_band_coverage": float(np.mean(inside_band)),
        "placebo_excess_median": float(np.median(excess)),
        "placebo_excess_q025_q975": np.quantile(excess, [0.025, 0.975]).tolist(),
        "interpretation": "Repeated placements share one fixed EEG/alarm stream; this is conditional diagnostic, not an independent-sample Type-I or CI-coverage estimate.",
    }


def support_preserving_placebo_audit(candidate_hits: np.ndarray,
                                     placements: int, events_per_draw: int,
                                     seed: int,
                                     reference_draws: int = 10000) -> dict:
    """Randomize event assignment only among the same eligible idle runs."""
    candidate_hits = np.asarray(candidate_hits, dtype=bool)
    if candidate_hits.ndim != 1 or len(candidate_hits) < events_per_draw:
        raise ValueError("invalid support-preserving candidate set")
    rng = np.random.default_rng(seed)
    # Uniform random-key subsets are without replacement within each draw.
    reference_indices = np.argpartition(
        rng.random((reference_draws, len(candidate_hits))),
        events_per_draw - 1, axis=1,
    )[:, :events_per_draw]
    reference_rates = np.mean(candidate_hits[reference_indices], axis=1)
    observed_indices = np.argpartition(
        rng.random((placements, len(candidate_hits))),
        events_per_draw - 1, axis=1,
    )[:, :events_per_draw]
    observed_rates = np.mean(candidate_hits[observed_indices], axis=1)
    p_values = np.asarray([
        (1 + np.sum(reference_rates >= value)) / (reference_draws + 1)
        for value in observed_rates
    ])
    lo, hi = np.quantile(reference_rates, [0.025, 0.975])
    return {
        "draws": placements,
        "reference_draws": reference_draws,
        "conditional_rejection_fraction_p_le_0_05": float(np.mean(p_values <= 0.05)),
        "null_predictive_95pct_band_coverage": float(np.mean(
            (observed_rates >= lo) & (observed_rates <= hi))),
        "placebo_excess_median": float(np.median(observed_rates - np.median(reference_rates))),
        "reference_hit_rate_median": float(np.median(reference_rates)),
        "exact_conditional": exact_support_preserving_null(
            candidate_hits, events_per_draw,
        ),
        "interpretation": "Event placements and reference both use identical idle-run candidate support; still conditional on one fixed EEG stream and discrete event hits.",
    }


def exact_support_preserving_null(candidate_hits: np.ndarray,
                                  events_per_draw: int) -> dict:
    """Exact random-subset null conditional on a fixed alarm train and runs.

    Sampling n distinct runs from N with K fixed alarm hits gives a
    hypergeometric hit count. This removes Monte Carlo error but does not
    create independent EEG recordings or validate physiological specificity.
    """
    candidate_hits = np.asarray(candidate_hits, dtype=bool)
    if candidate_hits.ndim != 1 or not 1 <= events_per_draw <= len(candidate_hits):
        raise ValueError("invalid candidate hits or event count")
    population = len(candidate_hits)
    successes = int(candidate_hits.sum())
    sample = int(events_per_draw)
    support = np.arange(max(0, sample - (population - successes)),
                        min(sample, successes) + 1)
    mass = hypergeom.pmf(support, population, successes, sample)
    upper_tail = hypergeom.sf(support - 1, population, successes, sample)
    low, high = hypergeom.ppf([.025, .975], population, successes, sample)
    return {
        "eligible_run_count": population,
        "candidate_alarm_hit_count": successes,
        "events_per_draw": sample,
        "expected_hit_fraction": successes / population,
        "hit_count_support": support.tolist(),
        "hit_count_probability": mass.tolist(),
        "one_sided_p_by_hit_count": upper_tail.tolist(),
        "exact_rejection_probability_p_le_0_05": float(
            mass[upper_tail <= .05].sum()),
        "central_95pct_predictive_hit_count_interval": [int(low), int(high)],
        "exact_central_95pct_predictive_coverage": float(
            mass[(support >= low) & (support <= high)].sum()),
    }


def frozen_model(root: Path, subject: str):
    calib = loadmat(root / "mat" / f"BCICIV_calib_ds1{subject}.mat",
                    struct_as_record=False, squeeze_me=True)
    channels = tuple(str(v) for v in np.atleast_1d(calib["nfo"].clab))
    config = ExperimentConfig(native_sfreq=SFREQ, target_sfreq=SFREQ,
                              window_seconds=WINDOW_SECONDS,
                              feature_step_seconds=STEP_SECONDS,
                              transient_seconds=2.0)
    times, cov = extract_covariances(calib["cnt"].astype(float) * 0.1,
                                     SFREQ, channels, config,
                                     config.sensorimotor_channels)
    active, idle = calibration_masks(times, np.atleast_1d(calib["mrk"].pos), SFREQ)
    train, anchor, threshold_stage, _ = stage_masks(times, np.atleast_1d(calib["mrk"].pos))
    ref = regularized_reference(cov[idle & train])
    features = tangent_features(cov, ref)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000,
                            solver="liblinear", random_state=SEED).fit(
        np.r_[features[active & train], features[idle & train]],
        np.r_[np.ones(np.sum(active & train)), np.zeros(np.sum(idle & train))],
    )
    raw = lr.decision_function(features)
    mean = float(np.mean(raw[idle & anchor]))
    variance = float(np.var(raw[idle & anchor], ddof=1))
    stage_ma = ma_from_raw(raw[threshold_stage], mean, variance, "ewma")
    threshold, calibration_fp = chronological_threshold(
        stage_ma, idle[threshold_stage], TARGET_FP,
    )
    return channels, config, ref, lr, mean, variance, threshold, calibration_fp


def subject_audit(root: Path, subject: str, *, placements: int, shifts: int) -> dict:
    channels, config, ref, lr, mean, variance, threshold, calibration_fp = frozen_model(root, subject)
    evaluation = loadmat(root / "mat" / f"BCICIV_eval_ds1{subject}.mat",
                         struct_as_record=False, squeeze_me=True)
    label_1000 = loadmat(root / "labels" / f"BCICIV_eval_ds1{subject}_1000Hz_true_y.mat",
                         squeeze_me=True)["true_y"]
    labels = downsample_true_labels(label_1000)
    eeg = evaluation["cnt"].astype(float) * 0.1
    candidates = eligible_idle_events(labels)
    if len(candidates) < EVENTS_PER_SUBJECT:
        raise ValueError("too few eligible independent idle runs")
    rng = np.random.default_rng(SEED + ord(subject))
    selected = np.sort(rng.choice(len(candidates), EVENTS_PER_SUBJECT, replace=False))
    events = candidates[selected]
    times, cov = extract_covariances(eeg, SFREQ, channels, config,
                                     config.sensorimotor_channels)
    baseline_raw = lr.decision_function(tangent_features(cov, ref))
    baseline_ma = ma_from_raw(baseline_raw, mean, variance, "ewma")
    baseline_alarm = rearmed(baseline_ma, threshold)
    modified = inject_sensorimotor_erd(eeg, events, channels)
    modified_times, modified_cov = extract_covariances(
        modified, SFREQ, channels, config, config.sensorimotor_channels,
    )
    assert np.array_equal(times, modified_times)
    motor_indices = [config.sensorimotor_channels.index(name) for name in ("C3", "Cz", "C4")]
    steady_event = np.zeros(len(times), dtype=bool)
    for start, stop in events:
        steady_event |= (times >= start + 1.0) & (times <= stop - 0.3)
    baseline_band_power = np.mean(
        np.diagonal(cov, axis1=1, axis2=2)[:, motor_indices], axis=1,
    )
    modified_band_power = np.mean(
        np.diagonal(modified_cov, axis1=1, axis2=2)[:, motor_indices], axis=1,
    )
    power_ratio = float(np.median(
        modified_band_power[steady_event] / baseline_band_power[steady_event],
    ))
    modified_raw = lr.decision_function(tangent_features(modified_cov, ref))
    modified_ma = ma_from_raw(modified_raw, mean, variance, "ewma")
    modified_alarm = rearmed(modified_ma, threshold)
    minimum = int(np.ceil(20.0 / STEP_SECONDS))
    offsets = rng.choice(np.arange(minimum, len(times) - minimum),
                         size=shifts, replace=False)
    base_hits = hit_map(times, baseline_alarm, candidates[:, 0], offsets)
    injected_hits = hit_map(times, modified_alarm, events[:, 0], offsets)
    placebo = placebo_audit(base_hits, placements, EVENTS_PER_SUBJECT,
                             SEED + 1000 + ord(subject))
    placement_placebo = support_preserving_placebo_audit(
        base_hits[:, 0], placements, EVENTS_PER_SUBJECT,
        SEED + 2000 + ord(subject),
    )
    fixed_selected_hits = int(base_hits[selected, 0].sum())
    placement_placebo["fixed_selected_hit_count"] = fixed_selected_hits
    placement_placebo["fixed_selected_exact_one_sided_p"] = float(hypergeom.sf(
        fixed_selected_hits - 1, len(candidates), int(base_hits[:, 0].sum()),
        EVENTS_PER_SUBJECT,
    ))
    base_event_tpr = float(np.mean(base_hits[selected, 0]))
    injected_event_tpr = float(np.mean(injected_hits[:, 0]))
    # The injection manipulation itself is never counted as an idle FP.
    ends = np.rint(times * SFREQ).astype(int)
    strict = support_labels(labels, ends, int(SFREQ))
    outside_windows = np.isfinite(strict) & (strict == 0)
    for start, stop in events:
        outside_windows &= ~((times >= start - WINDOW_SECONDS)
                             & (times <= stop + WINDOW_SECONDS))
    exposure = float(np.sum(outside_windows) * STEP_SECONDS)
    return {
        "subject": subject,
        "threshold": float(threshold), "calibration_fp_per_min": float(calibration_fp),
        "eligible_idle_runs": int(len(candidates)),
        "injected_events": events.tolist(),
        "injection": {
            "channels": ["C3", "Cz", "C4"], "band_hz": [8, 30],
            "amplitude_attenuation": AMPLITUDE_ATTENUATION,
            "event_seconds": EVENT_SECONDS, "edge_ramp_seconds": 0.3,
            "median_causal_motor_covariance_power_ratio": power_ratio,
            "baseline_hit_2s": base_event_tpr,
            "injected_hit_2s": injected_event_tpr,
            "paired_hit_difference": injected_event_tpr - base_event_tpr,
            "baseline_null_median": float(np.median(np.mean(base_hits[selected, 1:], axis=0))),
            "injected_null_median": float(np.median(np.mean(injected_hits[:, 1:], axis=0))),
            "outside_event_idle_seconds": exposure,
            "outside_event_idle_fp_baseline": float(60 * np.sum(baseline_alarm & outside_windows) / exposure),
            "outside_event_idle_fp_injected": float(60 * np.sum(modified_alarm & outside_windows) / exposure),
        },
        "placebo": placebo,
        "support_preserving_placebo": placement_placebo,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/bciciv1"))
    parser.add_argument("--placements", type=int, default=1000)
    parser.add_argument("--shifts", type=int, default=200)
    parser.add_argument("--subjects", nargs="+", default=list(REAL_SUBJECTS))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/bciciv1_semisynthetic_injection_20261002.json"))
    args = parser.parse_args()
    rows = []
    for subject in args.subjects:
        if subject not in REAL_SUBJECTS:
            raise ValueError("subject must be one of four real BCI IV-1 subjects")
        row = subject_audit(args.root, subject, placements=args.placements,
                            shifts=args.shifts)
        rows.append(row)
        print(f"done subject={subject}", flush=True)
    result = {
        "protocol": {
            "scope": "semi-synthetic mechanics check on native continuous acoustically-cued BCI IV-1; not physiological MI evidence",
            "model": "frozen disjoint 80/40/80 cue source/anchor/threshold procedure; EWMA chronological FP2 threshold",
            "selection": "20 of eligible >=6 s official idle runs chosen by fixed seed, without EEG-score inspection",
            "negative": f"{args.placements} random placebo event sets on the unchanged EEG/alarm stream",
            "positive": "single fixed 50% amplitude attenuation of phase-aligned 8-30 Hz component at C3/Cz/C4 for 3 s, 0.3 s smooth edges; offline injection only",
            "null": f"{args.shifts} circular alarm shifts >=20 s versus 10000 idle-run-support-preserving event placements; conditional calibration diagnostic only",
            "limitation": "n=4, one injection level, artificial events, repeated placements on the same EEG, acoustic cues and original task remain",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({r["subject"]: {
        "placebo_rejection": r["placebo"]["conditional_rejection_fraction_p_le_0_05"],
        "baseline_hit": r["injection"]["baseline_hit_2s"],
        "injected_hit": r["injection"]["injected_hit_2s"],
    } for r in rows}, indent=2))


if __name__ == "__main__":
    main()
