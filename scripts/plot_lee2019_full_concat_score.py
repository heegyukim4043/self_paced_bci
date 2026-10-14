"""Plot the complete rest-injected stream and rest-to-trial aligned score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.adaptation import ewma_normalize_and_decide
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, reference, rolling_covariance,
)

RELATIVE = np.arange(-8., 4.0001, STEP)  # last 5 s rest + 3 s fixation + task


def score_subject(subject, stream_root, raw_root, cov_root, mil_root):
    with np.load(cov_root / f"sub-{subject:02d}.npz", allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        ct, it = data["train_command_trial"], data["train_idle_iti_trial"]
        command = tangent_features(data["train_command_covariance"][ct < 50], ref)
        iti = tangent_features(data["train_idle_iti_covariance"][it < 50], ref)
    scaler = StandardScaler().fit(np.r_[command, iti])
    model = LogisticRegression(class_weight="balanced", solver="liblinear",
                               max_iter=2000, random_state=20260811).fit(
        scaler.transform(np.r_[command, iti]),
        np.r_[np.ones(len(command)), np.zeros(len(iti))])
    with np.load(raw_root / f"sub-{subject:02d}.npz", allow_pickle=False) as data:
        anchor_cov, _ = rolling_covariance(data["pre_train"])
    anchor = model.decision_function(scaler.transform(tangent_features(anchor_cov, ref)))
    mean, variance = float(anchor.mean()), float(anchor.var(ddof=1))
    anchor_z = (anchor-mean)/np.sqrt(variance)
    point = tune_source_operating_points(
        [calibration(anchor_z)], "ma_dwell", [2.], STEP,
        ma_seconds=.5, dwell_seconds=.3, refractory_seconds=1., grid_size=160)[2.]
    with np.load(stream_root / f"sub-{subject:02d}.npz", allow_pickle=False) as data:
        signal, events, state = data["signal"], data["events"], data["state"]
    covariance, ends = rolling_covariance(signal)
    raw_score = model.decision_function(scaler.transform(tangent_features(covariance, ref)))
    times = ends / SFREQ
    statistic = ewma_normalize_and_decide(
        raw_score, initial_mean=mean, initial_variance=variance,
        threshold=point.threshold, step_seconds=STEP, half_life_seconds=30.,
        update_mode="all", ma_seconds=.5, dwell_seconds=.3,
        refractory_seconds=1., sigma_floor_fraction=.25).statistic
    aligned = []
    for row in events:
        cue = row[5] / SFREQ
        indices = np.searchsorted(times, cue + RELATIVE)
        indices = np.clip(indices, 0, len(times)-1)
        aligned.append(statistic[indices])
    with np.load(mil_root / f"sub-{subject:02d}_mil_raw.npz", allow_pickle=False) as data:
        original_signal = data["signal"]
        command_intervals = data["command_intervals"]
        command_trials = data["command_trials"]
    original_covariance, original_ends = rolling_covariance(original_signal)
    original_raw = model.decision_function(
        scaler.transform(tangent_features(original_covariance, ref)))
    original_times = original_ends / SFREQ
    original_statistic = ewma_normalize_and_decide(
        original_raw, initial_mean=mean, initial_variance=variance,
        threshold=point.threshold, step_seconds=STEP, half_life_seconds=30.,
        update_mode="all", ma_seconds=.5, dwell_seconds=.3,
        refractory_seconds=1., sigma_floor_fraction=.25).statistic
    original_aligned = []
    for interval, trial in zip(command_intervals, command_trials):
        if trial < 50:
            continue
        cue = float(interval[0] - .5)
        indices = np.searchsorted(original_times, cue + RELATIVE)
        indices = np.clip(indices, 0, len(original_times)-1)
        original_aligned.append(original_statistic[indices])
    return (times, statistic, events, state, np.asarray(aligned), point.threshold,
            np.asarray(original_aligned))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-root", type=Path,
                        default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest_raw/session1"))
    parser.add_argument("--cov-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest/session1"))
    parser.add_argument("--mil-root", type=Path,
                        default=Path("data/interim/lee2019_mil_raw/session1"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--example-subject", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_full_concat_score.png"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("data/processed/lee2019_seam_ramp_audit.json"))
    args = parser.parse_args()
    subjects = args.subjects or sorted(
        int(p.stem.split("-")[1]) for p in args.stream_root.glob("sub-*.npz"))
    subject_means = []
    original_subject_means = []
    changes = []
    example = None
    for index, subject in enumerate(subjects, 1):
        print(f"concat plot {index}/{len(subjects)} subject={subject}", flush=True)
        result = score_subject(subject, args.stream_root, args.raw_root, args.cov_root,
                               args.mil_root)
        subject_means.append(result[4].mean(axis=0))
        original_subject_means.append(result[6].mean(axis=0))
        fixation = (RELATIVE >= -1.) & (RELATIVE <= 0.)
        strict_mi = (RELATIVE >= 1.5) & (RELATIVE <= 3.5)
        stable_rest = (RELATIVE >= -8.) & (RELATIVE <= -4.)
        changes.append({
            "subject": subject,
            "concat_mi_minus_fixation": float(
                result[4][:, strict_mi].mean()-result[4][:, fixation].mean()),
            "original_mi_minus_fixation": float(
                result[6][:, strict_mi].mean()-result[6][:, fixation].mean()),
            "concat_mi_minus_stable_injected_rest": float(
                result[4][:, strict_mi].mean()-result[4][:, stable_rest].mean()),
        })
        if subject == args.example_subject:
            example = result
    subject_means = np.asarray(subject_means)
    original_subject_means = np.asarray(original_subject_means)
    center = np.mean(subject_means, axis=0)
    sem = np.std(subject_means, axis=0, ddof=1)/np.sqrt(len(subject_means))
    original_center = np.mean(original_subject_means, axis=0)
    original_sem = np.std(original_subject_means, axis=0, ddof=1)/np.sqrt(len(original_subject_means))

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [1, 1.3]})
    ax = axes[0]
    ax.plot(RELATIVE, center, color="#167a3a", lw=2)
    ax.fill_between(RELATIVE, center-1.96*sem, center+1.96*sem,
                    color="#167a3a", alpha=.18)
    ax.plot(RELATIVE, original_center, color="#7b3294", lw=1.7,
            label="original continuous trial block")
    ax.fill_between(RELATIVE, original_center-1.96*original_sem,
                    original_center+1.96*original_sem, color="#7b3294", alpha=.10)
    ax.axvspan(-8, -3, color="0.75", alpha=.35, label="injected rest (last 5 s)")
    ax.axvspan(-3, 0, color="#4c78a8", alpha=.10, label="fixation")
    ax.axvspan(.5, 3.5, color="#a40018", alpha=.12, label="nominal MI")
    ax.axvline(-3, color="black", ls=":", label="artificial rest→trial seam")
    ax.axvline(0, color="black", ls="--", label="cue")
    ax.set(xlabel="Time from cue (s); score window end", ylabel="EWMA-normalized Riemannian score",
           title=f"Cohort mean, n={len(subjects)} (last 5 s of variable-length rest retained)")
    ax.legend(ncol=5, fontsize=8, loc="upper left")

    if example is None:
        raise ValueError("example subject not included")
    times, score, events, _, _, threshold, _ = example
    ax = axes[1]
    ax.plot(times/60, score, color="#167a3a", lw=.45, rasterized=True)
    for row in events:
        rest_a, rest_b = row[2]/SFREQ/60, row[3]/SFREQ/60
        mi_a, mi_b = row[6]/SFREQ/60, row[7]/SFREQ/60
        ax.axvspan(rest_a, rest_b, color="0.7", alpha=.16)
        ax.axvspan(mi_a, mi_b, color="#a40018", alpha=.12)
    ax.axhline(threshold, color="#d62728", ls="--", lw=1, label="anchor FP=2 threshold")
    ax.set(xlabel="Complete concatenated stream (minutes)", ylabel="EWMA-normalized score",
           title=f"Subject {args.example_subject}: all 50 random-rest + trial segments")
    ax.legend(loc="upper right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    concat_change = np.asarray([x["concat_mi_minus_fixation"] for x in changes])
    original_change = np.asarray([x["original_mi_minus_fixation"] for x in changes])
    stable_change = np.asarray([x["concat_mi_minus_stable_injected_rest"] for x in changes])
    summary = {
        "subjects": len(changes),
        "concat_mi_minus_fixation_median": float(np.median(concat_change)),
        "original_mi_minus_fixation_median": float(np.median(original_change)),
        "concat_minus_original_change_median": float(np.median(concat_change-original_change)),
        "concat_mi_minus_stable_injected_rest_median": float(np.median(stable_change)),
        "seam_location_from_cue_seconds": -3.0,
        "guard2_interval_from_cue_seconds": [-5.0, -1.0],
        "interpretation": (
            "A comparable original-block MI-minus-fixation change supports a task-locked "
            "effect; a concat-only change indicates seam/segment contamination."
        ),
    }
    args.summary_output.write_text(json.dumps({"summary": summary, "rows": changes}, indent=2),
                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
