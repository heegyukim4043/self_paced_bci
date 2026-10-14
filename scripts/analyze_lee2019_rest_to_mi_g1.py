"""Lee2019 Phase-1 physiological gate: direction-corrected mu/beta ERD.

This is an offline, zero-phase, cue-locked physiology analysis.  It is not a
causal detector and it does not identify a self-initiated intention onset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.data import lee2019_path, load_lee2019_run


BANDS = {
    "mu_8_13": (8., 13.),
    "beta_low_13_20": (13., 20.),
    "beta_high_20_30": (20., 30.),
}
CHANNELS = ("C3", "C4")
PHASES = {
    "visual_0_0p5": (0., .5),
    "initiation_0p5_1p5": (.5, 1.5),
    "sustained_1p5_4": (1.5, 4.),
    "full_mi_0p5_4": (.5, 4.),
}
CURVE_START = -1.
CURVE_STOP = 4.
CURVE_STEP = .02


def bootstrap_median(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(values), size=(repeats, len(values)))
    draws = np.median(values[chosen], axis=1)
    return np.quantile(draws, [.025, .975]).tolist()


def summarize(values: np.ndarray, direction: str, seed: int, repeats: int) -> dict:
    values = np.asarray(values, float)
    alternative = "less" if direction == "negative" else "greater"
    test = wilcoxon(values, zero_method="pratt", alternative=alternative)
    return {
        "median_db": float(np.median(values)),
        "bootstrap_95ci": bootstrap_median(values, seed, repeats),
        "wilcoxon_one_sided_p": float(test.pvalue),
        "expected_direction": direction,
        "negative_zero_positive": [
            int(np.sum(values < 0)), int(np.sum(values == 0)), int(np.sum(values > 0))
        ],
    }


def car_selected(eeg: np.ndarray, selected: list[int]) -> np.ndarray:
    common = np.mean(eeg, axis=1, dtype=np.float64)
    return np.asarray(eeg[:, selected], np.float64) - common[:, None]


def analytic_power(signal: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, signal, axis=0)
    return np.abs(hilbert(filtered, axis=0)) ** 2


def analyze_one(data_root: Path, subject: int, session: int) -> tuple[dict, dict]:
    run = load_lee2019_run(lee2019_path(data_root, subject, session), "train")
    lookup = {name.casefold(): i for i, name in enumerate(run.channels)}
    selected = [lookup[name.casefold()] for name in CHANNELS]
    sfreq = float(run.sfreq)
    signal = car_selected(run.eeg, selected)
    rest = car_selected(run.pre_rest, selected)
    labels = np.asarray(run.labels, int)
    offsets = np.arange(
        int(round(CURVE_START * sfreq)),
        int(round(CURVE_STOP * sfreq)),
        max(1, int(round(CURVE_STEP * sfreq))),
    )
    curve_times = offsets / sfreq
    row = {"subject": subject, "session": session, "bands": {}}
    curves: dict[str, np.ndarray] = {}

    for band_index, (band, (lo, hi)) in enumerate(BANDS.items()):
        power = analytic_power(signal, sfreq, lo, hi)
        rest_power = analytic_power(rest, sfreq, lo, hi)
        trim = int(round(2 * sfreq))
        rest_reference = np.mean(rest_power[trim:-trim], axis=0)
        trial_curves = []
        phase_values = {
            baseline: {phase: [] for phase in PHASES}
            for baseline in ("precue", "true_rest")
        }

        for cue, label in zip(np.asarray(run.cue_samples, int), labels):
            baseline_power = np.mean(power[cue - int(sfreq):cue], axis=0)
            sampled = power[cue + offsets]
            trial_curves.append(
                10 * np.log10(np.maximum(sampled, 1e-20) / np.maximum(baseline_power, 1e-20))
            )
            for phase, (start, stop) in PHASES.items():
                segment = power[
                    cue + int(round(start * sfreq)):cue + int(round(stop * sfreq))
                ]
                task_power = np.mean(segment, axis=0)
                for baseline, reference in (
                    ("precue", baseline_power), ("true_rest", rest_reference)
                ):
                    db = 10 * np.log10(
                        np.maximum(task_power, 1e-20) / np.maximum(reference, 1e-20)
                    )
                    # The MAT `class` field is authoritative:
                    # label 1 = right hand (contra C3), label 2 = left hand
                    # (contra C4).
                    contra_index = 0 if label == 1 else 1
                    ipsi_index = 1 - contra_index
                    phase_values[baseline][phase].append({
                        "label": int(label),
                        "c3": float(db[0]),
                        "c4": float(db[1]),
                        "contra": float(db[contra_index]),
                        "ipsi": float(db[ipsi_index]),
                    })

        trial_curves = np.asarray(trial_curves)
        contra_curves = np.where(
                labels[:, None] == 1, trial_curves[:, :, 0], trial_curves[:, :, 1]
        )
        ipsi_curves = np.where(
                labels[:, None] == 1, trial_curves[:, :, 1], trial_curves[:, :, 0]
        )
        curves[f"{band}_contra"] = np.median(contra_curves, axis=0)
        curves[f"{band}_ipsi"] = np.median(ipsi_curves, axis=0)
        band_row = {}
        for baseline, phases in phase_values.items():
            band_row[baseline] = {}
            for phase, trials in phases.items():
                contra = np.asarray([item["contra"] for item in trials])
                ipsi = np.asarray([item["ipsi"] for item in trials])
                left = [item for item in trials if item["label"] == 2]
                right = [item for item in trials if item["label"] == 1]
                band_row[baseline][phase] = {
                    "contra_db": float(np.median(contra)),
                    "ipsi_db": float(np.median(ipsi)),
                    "ipsi_minus_contra_db": float(np.median(ipsi - contra)),
                    "left_c4_db": float(np.median([item["c4"] for item in left])),
                    "left_c3_db": float(np.median([item["c3"] for item in left])),
                    "right_c3_db": float(np.median([item["c3"] for item in right])),
                    "right_c4_db": float(np.median([item["c4"] for item in right])),
                    "n_left": len(left),
                    "n_right": len(right),
                }
        row["bands"][band] = band_row
        del power, rest_power, trial_curves, contra_curves, ipsi_curves

    curves["times"] = curve_times
    return row, curves


def make_summary(rows: list[dict], repeats: int) -> dict:
    summary = {}
    measures = {
        "contra_db": "negative",
        "ipsi_db": "negative",
        "left_c4_db": "negative",
        "right_c3_db": "negative",
        "ipsi_minus_contra_db": "positive",
    }
    for session in (1, 2):
        selected = [row for row in rows if row["session"] == session]
        session_out = {}
        for band_index, band in enumerate(BANDS):
            band_out = {}
            for baseline_index, baseline in enumerate(("precue", "true_rest")):
                baseline_out = {}
                for phase_index, phase in enumerate(PHASES):
                    phase_out = {}
                    for measure_index, (measure, direction) in enumerate(measures.items()):
                        values = np.asarray([
                            row["bands"][band][baseline][phase][measure]
                            for row in selected
                        ])
                        phase_out[measure] = summarize(
                            values, direction,
                            100000 * session + 10000 * band_index
                            + 1000 * baseline_index + 100 * phase_index + measure_index,
                            repeats,
                        )
                    baseline_out[phase] = phase_out
                band_out[baseline] = baseline_out
            session_out[band] = band_out
        summary[str(session)] = session_out
    return summary


def correct_legacy_label_mapping(rows: list[dict], curve_rows: list[dict]) -> None:
    """Correct caches made under the reversed label-1/label-2 assumption."""
    for row in rows:
        for band in BANDS:
            for baseline in ("precue", "true_rest"):
                for phase in PHASES:
                    item = row["bands"][band][baseline][phase]
                    old = dict(item)
                    item["contra_db"] = old["ipsi_db"]
                    item["ipsi_db"] = old["contra_db"]
                    item["ipsi_minus_contra_db"] = -old["ipsi_minus_contra_db"]
                    item["left_c4_db"] = old["right_c4_db"]
                    item["left_c3_db"] = old["right_c3_db"]
                    item["right_c3_db"] = old["left_c3_db"]
                    item["right_c4_db"] = old["left_c4_db"]
                    item["n_left"] = old["n_right"]
                    item["n_right"] = old["n_left"]
    for row in curve_rows:
        for band in BANDS:
            contra = np.asarray(row[f"{band}_contra"]).copy()
            row[f"{band}_contra"] = np.asarray(row[f"{band}_ipsi"]).copy()
            row[f"{band}_ipsi"] = contra


def plot_curves(curve_rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=False)
    for row_index, session in enumerate((1, 2)):
        selected = [row for row in curve_rows if row["session"] == session]
        times = selected[0]["times"]
        for col_index, band in enumerate(BANDS):
            ax = axes[row_index, col_index]
            for key, label, color in (
                ("contra", "contralateral", "#2878b5"),
                ("ipsi", "ipsilateral", "#e36c21"),
            ):
                values = np.stack([row[f"{band}_{key}"] for row in selected])
                median = np.median(values, axis=0)
                lo, hi = np.quantile(values, [.25, .75], axis=0)
                ax.plot(times, median, color=color, lw=1.8, label=label)
                ax.fill_between(times, lo, hi, color=color, alpha=.17)
            ax.axvspan(0, .5, color="0.75", alpha=.2, label="visual" if row_index == 0 and col_index == 0 else None)
            ax.axvspan(.5, 1.5, color="#f4b183", alpha=.13, label="initiation" if row_index == 0 and col_index == 0 else None)
            ax.axvspan(1.5, 4., color="#e98b9a", alpha=.12, label="sustained MI" if row_index == 0 and col_index == 0 else None)
            ax.axhline(0, color="0.25", lw=.8)
            ax.axvline(0, color="0.35", ls="--", lw=.8)
            ax.grid(alpha=.2)
            ax.set_title(f"S{session}: {band.replace('_', ' ')}")
            ax.set_ylabel("ERD/ERS vs −1–0 s baseline (dB)")
            ax.set_xlabel("Time from cue (s)")
            if row_index == 0 and col_index == 0:
                ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Lee2019 G1: direction-corrected sensorimotor ERD\n"
        "54 subjects × 100 trials/session; subject medians with interquartile bands",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, .93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g1_checkpoint.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g1.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("data/processed/lee2019_rest_to_mi_g1.png"),
    )
    args = parser.parse_args()

    rows: list[dict] = []
    curve_rows: list[dict] = []
    mapping_version = "mat_class_field_v2_label1_right_label2_left"
    if args.checkpoint.exists():
        prior = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        rows = prior.get("rows", [])
        curve_rows = [
            {key: np.asarray(value) if key != "session" and key != "subject" else value
             for key, value in item.items()}
            for item in prior.get("curves", [])
        ]
        if prior.get("mapping_version") != mapping_version:
            correct_legacy_label_mapping(rows, curve_rows)
            args.checkpoint.write_text(json.dumps({
                "mapping_version": mapping_version,
                "rows": rows,
                "curves": [
                    {key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in item.items()}
                    for item in curve_rows
                ],
            }), encoding="utf-8")
            print("corrected legacy Lee label mapping: 1=right, 2=left", flush=True)
    complete = {(int(row["subject"]), int(row["session"])) for row in rows}
    for session in args.sessions:
        for index, subject in enumerate(args.subjects, start=1):
            if (subject, session) in complete:
                print(f"G1 S{session} sub-{subject:02d} cached", flush=True)
                continue
            row, curves = analyze_one(args.data_root, subject, session)
            rows.append(row)
            curve_rows.append({"subject": subject, "session": session, **curves})
            checkpoint = {
                "mapping_version": mapping_version,
                "rows": rows,
                "curves": [
                    {key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in item.items()}
                    for item in curve_rows
                ],
            }
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
            print(f"G1 S{session} sub-{subject:02d} [{index}/{len(args.subjects)}]", flush=True)

    expected = {(subject, session) for session in args.sessions for subject in args.subjects}
    rows = [row for row in rows if (int(row["subject"]), int(row["session"])) in expected]
    curve_rows = [row for row in curve_rows if (int(row["subject"]), int(row["session"])) in expected]
    summary = make_summary(rows, args.bootstrap)
    payload = {
        "protocol": {
            "dataset": "Lee2019/OpenBMI train phase, 54 subjects x 2 sessions",
            "analysis": "offline cue-locked physiology; zero-phase filtering and Hilbert power",
            "reference": "common average reference before selecting C3/C4",
            "bands_hz": BANDS,
            "primary_baseline": "trial-specific cue -1 to 0 s",
            "secondary_baseline": "pre-train 60-s eyes-open rest, edge-trimmed 2 s",
            "phases_seconds": PHASES,
            "direction": "MAT class field: label 1 right=>C3 contra; label 2 left=>C4 contra",
            "unit": "subject/session median across 100 trials",
            "inference": "one-sided Wilcoxon and 10,000-draw subject bootstrap median CI",
            "claim_limit": "cue-locked ERD validation; not causal and not self-initiated onset",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_curves(curve_rows, args.figure)
    print(args.output)
    print(args.figure)


if __name__ == "__main__":
    main()
