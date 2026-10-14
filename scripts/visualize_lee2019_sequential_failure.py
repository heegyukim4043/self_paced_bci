"""Visual and numeric audit of where the strict sequential gate fires."""

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
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import tune_source_operating_points
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_pseudo_uncued_riemannian import (
    SFREQ, STEP, WINDOW, calibration, contiguous_intervals, reference,
    rolling_covariance,
)

TARGETS = (.25, .5, 1., 2., 4., 8., 16.)
GUARD_WIDTH = 3.
RELATIVE = np.arange(-2., 5.0001, STEP)


def interp_threshold(points):
    pairs = sorted((value["fp"], value["threshold"]) for value in points)
    x = np.asarray([a for a, _ in pairs])
    y = np.asarray([b for _, b in pairs])
    unique = np.unique(x)
    y = np.asarray([np.median(y[x == value]) for value in unique])
    x = unique
    if len(x) < 2 or 2. < x.min() or 2. > x.max():
        return None
    return float(np.interp(2., x, y))


def moving_average(values, allowed, steps=5):
    output = np.full(len(values), np.nan)
    history = []
    for index, value in enumerate(values):
        if not allowed[index]:
            history.clear()
            continue
        history.append(float(value))
        if len(history) > steps:
            history.pop(0)
        if len(history) == steps:
            output[index] = np.mean(history)
    return output


def acf_segmented(values, valid, max_lag):
    centered = values[valid] - np.mean(values[valid])
    denominator = np.sum(centered ** 2)
    variance = denominator / max(len(centered), 1)
    output = np.full(max_lag + 1, np.nan)
    output[0] = 1.
    indices = np.flatnonzero(valid)
    for lag in range(1, max_lag + 1):
        keep = valid[:-lag] & valid[lag:]
        if not np.any(keep) or variance <= 0:
            continue
        left = values[:-lag][keep]
        right = values[lag:][keep]
        output[lag] = np.mean((left-left.mean()) * (right-right.mean())) / variance
    return output


def subject_data(subject, stream_root, raw_root, cov_root):
    with np.load(cov_root / f"sub-{subject:02d}.npz", allow_pickle=False) as data:
        ref = reference(data["pre_train_covariance"])
        command = tangent_features(
            data["train_command_covariance"][data["train_command_trial"] < 50], ref)
        iti = tangent_features(
            data["train_idle_iti_covariance"][data["train_idle_iti_trial"] < 50], ref)
    scaler = StandardScaler().fit(np.r_[command, iti])
    model = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=2000,
        random_state=20260811,
    ).fit(scaler.transform(np.r_[command, iti]),
          np.r_[np.ones(len(command)), np.zeros(len(iti))])
    with np.load(raw_root / f"sub-{subject:02d}.npz", allow_pickle=False) as data:
        anchor_covariance, _ = rolling_covariance(data["pre_train"])
    anchor = model.decision_function(
        scaler.transform(tangent_features(anchor_covariance, ref)))
    initial_mean = float(anchor.mean())
    initial_variance = float(anchor.var(ddof=1))
    anchor_z = (anchor-initial_mean) / np.sqrt(initial_variance)
    operating = tune_source_operating_points(
        [calibration(anchor_z)], "ma_dwell", TARGETS, STEP,
        ma_seconds=.5, dwell_seconds=.3, refractory_seconds=1., grid_size=160)

    with np.load(stream_root / f"sub-{subject:02d}.npz", allow_pickle=False) as data:
        signal, events = data["signal"], data["events"]
    covariance, ends = rolling_covariance(signal)
    raw_score = model.decision_function(
        scaler.transform(tangent_features(covariance, ref)))
    times = ends / SFREQ
    starts = (ends-int(WINDOW*SFREQ)) / SFREQ
    seams = np.unique(np.r_[events[:, 4], events[1:, 2]]) / SFREQ
    guard = np.zeros(len(times), dtype=bool)
    for seam in seams:
        guard |= (starts < seam + GUARD_WIDTH) & (times > seam)
    strict = np.zeros(len(times), dtype=bool)
    for row in events:
        strict |= (starts >= row[6]/SFREQ) & (times <= row[7]/SFREQ)
    idle = ~strict & ~guard
    idle_intervals = contiguous_intervals(idle, times)
    commands = np.asarray([(row[6]/SFREQ + WINDOW, row[7]/SFREQ) for row in events])

    points = []
    for target, point in operating.items():
        decision = ewma_normalize_and_decide(
            raw_score, initial_mean=initial_mean, initial_variance=initial_variance,
            threshold=point.threshold, step_seconds=STEP, half_life_seconds=30.,
            update_mode="all", update_mask=~guard, ma_seconds=.5,
            dwell_seconds=.3, refractory_seconds=1., sigma_floor_fraction=.25)
        metric = evaluate_alarms(times, decision.alarms, idle_intervals, commands)
        points.append({"target": target, "fp": metric.fp_per_minute,
                       "threshold": float(point.threshold)})
    threshold = interp_threshold(points)
    if threshold is None:
        return {"subject": subject, "bracketed": False}
    decision = ewma_normalize_and_decide(
        raw_score, initial_mean=initial_mean, initial_variance=initial_variance,
        threshold=threshold, step_seconds=STEP, half_life_seconds=30.,
        update_mode="all", update_mask=~guard, ma_seconds=.5,
        dwell_seconds=.3, refractory_seconds=1., sigma_floor_fraction=.25)
    score = decision.statistic
    ma = moving_average(score, ~guard, int(np.ceil(.5/STEP)))
    active = ma >= threshold

    trials = []
    exceed_relative = []
    counts = {name: 0 for name in (
        "flat", "raw_spike", "ma_short", "detected",
        "late_pre_seam", "post_seam_only")}
    for trial_index, row in enumerate(events):
        cue = row[5] / SFREQ
        sample_indices = np.searchsorted(times, cue + RELATIVE)
        sample_indices = np.clip(sample_indices, 0, len(times)-1)
        raw_margin = score[sample_indices] - threshold
        ma_margin = ma[sample_indices] - threshold
        alarm = decision.alarms[sample_indices]
        command = (RELATIVE >= 1.5) & (RELATIVE <= 3.5)
        late = (RELATIVE > 3.5) & (RELATIVE <= 4.)
        post_seam = (RELATIVE > 4.) & (RELATIVE <= 5.)
        allowed_relative = ~guard[sample_indices]
        raw_command = raw_margin[command] >= 0
        ma_command = np.nan_to_num(ma_margin[command], nan=-np.inf) >= 0
        alarm_command = alarm[command]
        if np.any(alarm_command):
            category = "detected"
        elif np.any(ma_command):
            category = "ma_short"
        elif np.any(raw_command):
            category = "raw_spike"
        elif np.any((raw_margin >= 0) & late & allowed_relative):
            category = "late_pre_seam"
        elif np.any((raw_margin >= 0) & post_seam):
            category = "post_seam_only"
        else:
            category = "flat"
        counts[category] += 1
        trials.append({
            "subject": subject, "trial": trial_index, "category": category,
            "raw": raw_margin.tolist(), "ma": ma_margin.tolist(),
            "alarm": alarm.astype(int).tolist(),
            "max_ma": float(np.nanmax(ma_margin[command])) if np.any(np.isfinite(ma_margin[command])) else -99.,
        })
        valid = ~guard & (score >= threshold)
        nearby = valid & (times >= cue-2.) & (times <= cue+5.)
        exceed_relative.extend((times[nearby]-cue).tolist())

    acf = acf_segmented(score, idle, int(10./STEP))
    below = np.flatnonzero(acf <= np.exp(-1))
    correlation_time = float(below[0]*STEP) if len(below) else None
    return {
        "subject": subject, "bracketed": True, "threshold": threshold,
        "counts": counts, "trials": trials, "exceed_relative": exceed_relative,
        "acf": acf.tolist(), "correlation_time_1e_seconds": correlation_time,
        "stream": {"times": times.tolist(), "score": (score-threshold).tolist(),
                   "ma": (ma-threshold).tolist(), "alarms": decision.alarms.astype(int).tolist(),
                   "events": events.tolist()},
    }


def choose_trials(rows, category, count):
    candidates = [trial for row in rows if row.get("bracketed")
                  for trial in row["trials"] if trial["category"] == category]
    candidates.sort(key=lambda trial: trial["max_ma"], reverse=True)
    output, used = [], set()
    for trial in candidates:
        if trial["subject"] not in used:
            output.append(trial); used.add(trial["subject"])
        if len(output) == count:
            break
    return output


def plot(rows, output):
    detected = choose_trials(rows, "detected", 12)
    nonresponse = choose_trials(rows, "raw_spike", 12)
    if len(nonresponse) < 12:
        nonresponse += choose_trials(rows, "flat", 12-len(nonresponse))
    fig = plt.figure(figsize=(18, 20), constrained_layout=True)
    outer = fig.add_gridspec(4, 1, height_ratios=[5.5, 1.4, 1.4, 2.2])
    trial_grid = outer[0].subgridspec(4, 6)
    examples = [(trial, "detected") for trial in detected] + [
        (trial, "non-response") for trial in nonresponse]
    for index in range(24):
        ax = fig.add_subplot(trial_grid[index//6, index%6])
        if index >= len(examples):
            ax.axis("off"); continue
        trial, label = examples[index]
        ax.plot(RELATIVE, trial["raw"], color="#377eb8", lw=1, label="score")
        ax.plot(RELATIVE, trial["ma"], color="#e41a1c", lw=1.2, label="0.5-s MA")
        ax.axhline(0, color="black", lw=.8)
        ax.axvspan(1.5, 3.5, color="#f4a3b5", alpha=.25)
        alarm_times = RELATIVE[np.asarray(trial["alarm"], dtype=bool)]
        ax.scatter(alarm_times, np.zeros(len(alarm_times)), marker="v", color="black", s=12)
        ax.set_title(f"{label}: S{trial['subject']} T{trial['trial']}", fontsize=8)
        ax.set_xlim(-2, 5); ax.set_ylim(-4, 4)
        if index == 0: ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Guard-3 sequential timing audit", fontsize=15)

    ax_hist = fig.add_subplot(outer[1])
    relative = np.asarray([value for row in rows if row.get("bracketed")
                           for value in row["exceed_relative"]])
    ax_hist.hist(relative, bins=np.arange(-2, 5.1, .1), color="#756bb1")
    ax_hist.axvspan(1.5, 3.5, color="#f4a3b5", alpha=.25, label="strict command")
    ax_hist.axvline(0, color="black", ls="--")
    ax_hist.set(xlabel="Time from cue (s)", ylabel="Above-threshold samples",
                title="(b) Location of all valid guard-3 threshold exceedances")
    ax_hist.legend()

    ax_acf = fig.add_subplot(outer[2])
    acfs = np.asarray([row["acf"] for row in rows if row.get("bracketed")])
    median = np.nanmedian(acfs, axis=0)
    lower, upper = np.nanquantile(acfs, [.25, .75], axis=0)
    lags = np.arange(acfs.shape[1]) * STEP
    ax_acf.plot(lags, median, color="#238b45")
    ax_acf.fill_between(lags, lower, upper, color="#74c476", alpha=.35)
    ax_acf.axhline(np.exp(-1), color="black", ls="--", lw=.8, label="1/e")
    ax_acf.set(xlabel="Lag (s)", ylabel="Autocorrelation",
               title="(c) Valid-idle Riemannian score ACF (median and IQR)")
    ax_acf.legend()

    available = [row for row in rows if row.get("bracketed")]
    representative = max(available, key=lambda row: row["counts"]["detected"])
    stream = representative["stream"]
    times = np.asarray(stream["times"])
    start = times[len(times)//3]
    keep = (times >= start) & (times <= start+180.)
    ax_stream = fig.add_subplot(outer[3])
    ax_stream.plot(times[keep]-start, np.asarray(stream["score"])[keep], lw=.7,
                   color="#377eb8", label="score-threshold")
    ax_stream.plot(times[keep]-start, np.asarray(stream["ma"])[keep], lw=.8,
                   color="#e41a1c", label="MA-threshold")
    ax_stream.axhline(0, color="black", lw=.8)
    for row in np.asarray(stream["events"]):
        begin, end = row[6]/SFREQ + WINDOW, row[7]/SFREQ
        if end >= start and begin <= start+180.:
            ax_stream.axvspan(max(begin, start)-start, min(end, start+180.)-start,
                              color="#f4a3b5", alpha=.25)
    alarms = np.asarray(stream["alarms"], dtype=bool) & keep
    ax_stream.scatter(times[alarms]-start, np.zeros(np.sum(alarms)), marker="v",
                      color="black", s=20, label="alarm")
    ax_stream.set(xlabel="Time in 3-min excerpt (s)", ylabel="Margin",
                  title=f"(d) Continuous excerpt, subject {representative['subject']}")
    ax_stream.legend(ncol=4, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-root", type=Path,
                        default=Path("data/interim/lee2019_pseudo_uncued_raw/session1"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest_raw/session1"))
    parser.add_argument("--cov-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest/session1"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--json", type=Path,
                        default=Path("data/processed/lee2019_sequential_failure_timing.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("data/processed/lee2019_sequential_failure_timing.png"))
    args = parser.parse_args()
    subjects = args.subjects or list(range(1, 55))
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"timing audit {index}/{len(subjects)} subject={subject}", flush=True)
        rows.append(subject_data(subject, args.stream_root, args.raw_root, args.cov_root))
    aggregate = {name: int(sum(row["counts"][name] for row in rows if row.get("bracketed")))
                 for name in ("flat", "raw_spike", "ma_short", "detected",
                              "late_pre_seam", "post_seam_only")}
    correlation_times = [row["correlation_time_1e_seconds"] for row in rows
                         if row.get("bracketed") and row["correlation_time_1e_seconds"] is not None]
    exceed_by_subject = [len(row["exceed_relative"]) for row in rows if row.get("bracketed")]
    summary = {
        "bracketed_subjects": int(sum(row.get("bracketed", False) for row in rows)),
        "trial_categories": aggregate,
        "valid_exceedances_near_cue": int(sum(exceed_by_subject)),
        "subjects_with_any_exceedance": int(sum(value > 0 for value in exceed_by_subject)),
        "exceedance_subject_gini_proxy_top10_fraction": float(
            sum(sorted(exceed_by_subject, reverse=True)[:max(1, len(exceed_by_subject)//10)])
            / max(sum(exceed_by_subject), 1)),
        "idle_acf_1e_time_median_seconds": float(np.median(correlation_times)),
        "idle_acf_1e_time_iqr_seconds": np.quantile(correlation_times, [.25, .75]).tolist(),
    }
    plot(rows, args.figure)
    compact_rows = []
    for row in rows:
        if not row.get("bracketed"):
            compact_rows.append(row)
            continue
        compact_rows.append({
            "subject": row["subject"], "bracketed": True,
            "threshold": row["threshold"], "counts": row["counts"],
            "exceed_relative": row["exceed_relative"], "acf": row["acf"],
            "correlation_time_1e_seconds": row["correlation_time_1e_seconds"],
            "trial_categories": [trial["category"] for trial in row["trials"]],
        })
    args.json.write_text(json.dumps({"summary": summary, "rows": compact_rows}, indent=2),
                         encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
