"""Plot one rest-to-command transition for three sequential detectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import instantaneous, moving_average_dwell
from uncued_mi.hsmm import GaussianEmission
from uncued_mi.operating_point import ScoredSequence
from uncued_mi.rest_injected import build_rest_injected_stream
from uncued_mi.riemannian import tangent_features
from scripts.run_lee2019_sequential_onset import (
    RESTS, STEP, calibration_sequence, finite_horizon_glr, fit_score,
    hsmm_statistic, reference, tune,
)


def build(subject, session, cache_root):
    path = cache_root / f"session{session}" / f"sub-{subject:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        pre = data["pre_train_covariance"]
        split = len(pre) // 2
        ref = reference(pre[:split])
        command = tangent_features(data["train_command_covariance"], ref)
        iti = tangent_features(data["train_idle_iti_covariance"], ref)
        rest = {
            name: tangent_features(data[f"{name}_covariance"], ref)
            for name in RESTS
        }
        command_trials = data["train_command_trial"]
        iti_trials = data["train_idle_iti_trial"]
        scaler, model = fit_score(
            command[command_trials < 50],
            iti[iti_trials < 50],
            rest["pre_train"][:split],
        )
        predict = lambda values: model.decision_function(
            scaler.transform(values)
        )
        fit_mi = predict(command[command_trials < 50])
        fit_idle = np.r_[
            predict(iti[iti_trials < 50]),
            predict(rest["pre_train"][:split]),
        ]
        emission = GaussianEmission.fit(fit_idle, fit_mi)
        rest_scores = {name: predict(value) for name, value in rest.items()}
        calibration = [
            rest_scores[name][:len(rest_scores[name])//2]
            for name in RESTS if name != "pre_train"
        ]
        evaluation = [
            value[len(value)//2:] for value in rest_scores.values()
        ]
        command_scores = predict(command)
        trials = [
            command_scores[command_trials == trial]
            for trial in range(50, 100)
        ]
    stream, _ = build_rest_injected_stream(
        evaluation, trials, step_seconds=STEP,
        minimum_rest_seconds=5, maximum_rest_seconds=11,
        guard_seconds=1, seed=20260728 + 100 * session + subject,
    )
    ma_point = tune(
        [calibration_sequence(value) for value in calibration], "ma_dwell"
    )
    glr_calibration = [
        ScoredSequence(
            item.times, finite_horizon_glr(item.scores, emission, 10),
            item.idle_intervals, item.command_intervals,
        )
        for item in map(calibration_sequence, calibration)
    ]
    hsmm_calibration = [
        ScoredSequence(
            item.times, hsmm_statistic(item.scores, emission),
            item.idle_intervals, item.command_intervals,
        )
        for item in map(calibration_sequence, calibration)
    ]
    glr_point = tune(glr_calibration, "glr_1s")
    hsmm_point = tune(hsmm_calibration, "hsmm")

    ma = moving_average_dwell(
        stream.scores, ma_point.threshold, STEP, 0.5, 0.3, 1.0
    )
    glr_values = finite_horizon_glr(stream.scores, emission, 10)
    glr = instantaneous(glr_values, glr_point.threshold, STEP, 1.0)
    hsmm_values = hsmm_statistic(stream.scores, emission)
    hsmm = instantaneous(hsmm_values, hsmm_point.threshold, STEP, 1.0)
    return stream, {
        "ma": (ma.statistic, ma.alarms, ma_point.threshold),
        "glr": (glr_values, glr.alarms, glr_point.threshold),
        "hsmm": (hsmm_values, hsmm.alarms, hsmm_point.threshold),
    }


def choose_event(stream, results):
    times = stream.times
    ma_alarm = times[results["ma"][1]]
    hsmm_alarm = times[results["hsmm"][1]]
    for index, (start, stop) in enumerate(stream.command_intervals):
        hsmm_hit = hsmm_alarm[
            (hsmm_alarm >= start) & (hsmm_alarm <= start + 0.5)
        ]
        ma_hit = ma_alarm[(ma_alarm >= start) & (ma_alarm <= stop)]
        if len(hsmm_hit) and (not len(ma_hit) or ma_hit[0] > hsmm_hit[0]):
            return index, start, stop
    return 0, *stream.command_intervals[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=2)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_four_rest")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/sequential_onset_example.png"),
    )
    args = parser.parse_args()
    stream, results = build(args.subject, args.session, args.cache_root)
    event, start, stop = choose_event(stream, results)
    selected = (stream.times >= start - 4.0) & (stream.times <= stop + 1.0)
    relative = stream.times[selected] - start

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    specifications = (
        ("ma", "MA 0.5 s + dwell 0.3 s", "tab:blue"),
        ("glr", "Finite-horizon LLR-GLR (1 s)", "tab:orange"),
        ("hsmm", "Causal HSMM posterior", "tab:green"),
    )
    for axis, (key, title, color) in zip(axes, specifications):
        statistic, alarms, threshold = results[key]
        visible = statistic[selected].copy()
        guard = ((relative >= -1.0) & (relative < 0.0)) | (
            (relative > stop - start) & (relative <= stop - start + 1.0)
        )
        visible[guard] = np.nan
        axis.plot(relative, visible, color=color, lw=1.5)
        axis.axhline(threshold, color="black", ls="--", lw=1, label="threshold")
        alarm_times = stream.times[alarms] - start
        alarm_times = alarm_times[
            (alarm_times >= relative.min()) & (alarm_times <= relative.max())
        ]
        for alarm in alarm_times:
            axis.axvline(alarm, color="crimson", lw=1.5, label="alarm")
        axis.axvspan(-1.0, 0.0, color="0.7", alpha=0.25)
        axis.axvspan(
            stop - start, stop - start + 1.0, color="0.7", alpha=0.25
        )
        axis.axvspan(0, stop - start, color="tab:red", alpha=0.10)
        axis.axvline(0, color="tab:red", lw=1.2)
        axis.set_ylabel("statistic")
        axis.set_title(title, loc="left", fontsize=10)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), loc="upper left")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel(
        "Time relative to simulated command-score segment onset (s)"
    )
    fig.suptitle(
        f"Sequential onset example — Lee2019 S{args.session}, "
        f"subject {args.subject}, event {event}"
    )
    fig.text(
        0.5, 0.01,
        "White: rest; gray: simulation reset guard; red shading: "
        "command-score segment; red vertical line: onset; crimson: alarm",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)
    print({"event": event, "start": start, "stop": stop})


if __name__ == "__main__":
    main()
