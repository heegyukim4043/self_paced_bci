"""Audit whether Lee2019/Cho2017 can support an MRCP detection experiment.

This is a signal/data-identifiability audit, not a classifier evaluation.
Zero-phase filtering is used only to visualize offline event-locked slow
potentials; any later detector must use causal preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.data import channel_indices, load_lee2019_run

CHO_CHANNELS = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)
CENTRAL = ("FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4")


def car(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def slow_filter(x: np.ndarray, sfreq: float) -> np.ndarray:
    sos = butter(3, [0.05, 4.0], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def band_power_ratio(x: np.ndarray, sfreq: float) -> dict[str, float]:
    frequencies, density = welch(
        x,
        fs=sfreq,
        axis=-1,
        nperseg=min(int(round(20 * sfreq)), x.shape[-1]),
        noverlap=min(int(round(10 * sfreq)), x.shape[-1] // 2),
    )
    mean_density = np.mean(density, axis=tuple(range(density.ndim - 1)))

    def integrate(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies <= high)
        return float(np.trapezoid(mean_density[mask], frequencies[mask]))

    low = integrate(0.05, 4.0)
    erd = integrate(8.0, 30.0)
    return {
        "power_0p05_4": low,
        "power_8_30": erd,
        "low_to_erd_ratio": low / max(erd, np.finfo(float).tiny),
    }


def baseline_correct(
    epochs: np.ndarray, times: np.ndarray, interval: tuple[float, float]
) -> np.ndarray:
    mask = (times >= interval[0]) & (times < interval[1])
    return epochs - np.mean(epochs[..., mask], axis=-1, keepdims=True)


def lee_payload(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    run = load_lee2019_run(path, phase="train")
    selected = tuple(
        name for name in CENTRAL if name.casefold() in {item.casefold() for item in run.channels}
    )
    indices = channel_indices(run.channels, selected)
    continuous = car(run.eeg[:, indices].T)
    filtered = slow_filter(continuous, run.sfreq)
    start, stop = -2.0, 5.0
    offsets = np.arange(
        int(round(start * run.sfreq)), int(round(stop * run.sfreq))
    )
    epochs = []
    for cue in run.cue_samples:
        samples = cue + offsets
        if samples[0] >= 0 and samples[-1] < filtered.shape[-1]:
            epochs.append(filtered[:, samples])
    epochs = np.asarray(epochs)
    times = offsets / run.sfreq
    epochs = baseline_correct(epochs, times, (-2.0, -1.5))
    trace = np.median(np.mean(epochs, axis=1), axis=0)
    audit = {
        "dataset": "Lee2019",
        "subject": 1,
        "sfreq": run.sfreq,
        "channels": list(selected),
        "n_events": len(epochs),
        "stored_filter_metadata": False,
        "cue_time_seconds": 0.0,
        "nominal_mi_onset_seconds": 0.5,
        "independent_mi_onset_timestamp": False,
        "cue_mi_offset_is_fixed_seconds": 0.5,
        "mrcp_vs_cue_slow_potential_identifiable": False,
        "identifiability_reason": (
            "MI onset is defined as cue+0.5 s rather than independently observed; "
            "cue-locked and nominal-MI-locked averages are time-shifted views of "
            "the same event."
        ),
        "spectral_presence_diagnostic": band_power_ratio(continuous, run.sfreq),
    }
    return times, trace, audit


def cho_payload(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    eeg = loadmat(
        path,
        variable_names=["eeg"],
        squeeze_me=True,
        struct_as_record=False,
        verify_compressed_data_integrity=False,
    )["eeg"]
    sfreq = float(eeg.srate)
    n_trials = int(eeg.n_imagery_trials)
    samples_per_trial = int(round(7.0 * sfreq))
    indices = channel_indices(CHO_CHANNELS, CENTRAL)
    classes = []
    for values in (eeg.imagery_left, eeg.imagery_right):
        raw = np.asarray(values, dtype=float)[:64]
        if raw.shape[1] != n_trials * samples_per_trial:
            raise ValueError("Cho2017 imagery matrix is not the documented 7-s trial concatenation")
        trials = raw.reshape(64, n_trials, samples_per_trial).transpose(1, 0, 2)
        classes.append(trials[:, indices])
    trials = car(np.concatenate(classes, axis=0))
    filtered = np.stack([slow_filter(trial, sfreq) for trial in trials])
    # Stored frame is [-2, +5] s relative to the visual instruction at t=0.
    times = np.arange(samples_per_trial) / sfreq - 2.0
    filtered = baseline_correct(filtered, times, (-2.0, -1.5))
    trace = np.median(np.mean(filtered, axis=1), axis=0)
    continuous_for_psd = trials.transpose(1, 0, 2).reshape(len(indices), -1)
    audit = {
        "dataset": "Cho2017",
        "subject": 1,
        "sfreq": sfreq,
        "channels": list(CENTRAL),
        "n_events": len(trials),
        "stored_filter_metadata": False,
        "recording_system": "BioSemi ActiveTwo (paper/loader documentation)",
        "cue_time_seconds": 0.0,
        "nominal_mi_onset_seconds": 0.0,
        "independent_mi_onset_timestamp": False,
        "cue_mi_offset_is_fixed_seconds": 0.0,
        "mrcp_vs_cue_slow_potential_identifiable": False,
        "identifiability_reason": (
            "The visual instruction defines the nominal MI onset and no independent "
            "behavioral/physiological onset timestamp is stored."
        ),
        "seam_warning": "Trials are stored as exact 7-s concatenations; filter per trial.",
        "spectral_presence_diagnostic": band_power_ratio(continuous_for_psd, sfreq),
    }
    return times, trace, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lee",
        type=Path,
        default=Path(
            "data/raw/lee2019_mi/session1/s1/"
            "sess01_subj01_EEG_MI.mat"
        ),
    )
    parser.add_argument(
        "--cho", type=Path, default=Path("data/raw/cho2017/s01.mat")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/mrcp_feasibility_audit.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("data/processed/mrcp_feasibility_audit.png"),
    )
    args = parser.parse_args()

    lee_times, lee_trace, lee_audit = lee_payload(args.lee)
    cho_times, cho_trace, cho_audit = cho_payload(args.cho)

    payload = {
        "purpose": "signal-presence and onset-identifiability audit; not classifier evaluation",
        "offline_visualization_filter": "zero-phase 0.05-4 Hz only for audit",
        "deployment_filter_requirement": "causal",
        "datasets": [lee_audit, cho_audit],
        "decision": {
            "low_frequency_samples_present": True,
            "acquisition_high_pass_verified_from_mat_metadata": False,
            "mrcp_can_be_separated_from_cue_locked_slow_potential": False,
            "reason": (
                "Neither dataset stores an independently observed MI-intention onset. "
                "The nominal onset is fixed to the visual cue, so an MRCP detector "
                "cannot be validated as uncued rather than cue-locked on these data."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for axis, times, trace, title in (
        (axes[0], lee_times, lee_trace, "Lee2019 S1 — central slow potential"),
        (axes[1], cho_times, cho_trace, "Cho2017 S1 — central slow potential"),
    ):
        axis.plot(times, trace, color="#1f5a92", linewidth=1.2)
        axis.axvline(0.0, color="#b33a3a", linestyle="--", label="visual cue")
        axis.axhline(0.0, color="0.7", linewidth=0.8)
        axis.set_ylabel("Amplitude (a.u.)")
        axis.set_title(title)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time from cue (s)")
    figure.tight_layout()
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
