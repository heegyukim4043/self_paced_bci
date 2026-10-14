"""Test whether cue-aligned 55--90-Hz power tracks EMG envelope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uncued_mi.data import lee2019_path

ENDS = np.arange(-1., 4.0001, .1)


def log_rms_windows(values, cue, fs):
    width = int(fs)
    output = []
    for relative in ENDS:
        end = int(round(cue + relative*fs))
        segment = values[end-width:end]
        output.append(np.log(max(float(np.sqrt(np.mean(segment**2))), 1e-12)))
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 55)))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_gamma_emg_correlation.json"))
    args = parser.parse_args()
    rows = []
    pre = (ENDS >= -.5) & (ENDS <= 0.)
    mi = (ENDS >= 1.5) & (ENDS <= 3.5)
    for index, subject in enumerate(args.subjects, 1):
        path = lee2019_path(args.data_root, subject, 1)
        mat = loadmat(path, variable_names=["EEG_MI_train"], squeeze_me=True,
                      struct_as_record=False, verify_compressed_data_integrity=False)["EEG_MI_train"]
        eeg = np.asarray(mat.x, float); emg = np.asarray(mat.EMG, float)
        fs = float(mat.fs); cues = np.asarray(mat.t, int).reshape(-1)
        channels = [str(np.squeeze(x).item()) for x in np.ravel(mat.chan)]
        lookup = {n.casefold(): i for i, n in enumerate(channels)}
        central = [lookup[n.casefold()] for n in ("C3", "Cz", "C4")]
        car = eeg[:, central] - eeg.mean(axis=1, keepdims=True)
        gamma = sosfilt(butter(4, [55., 90.], btype="bandpass", fs=fs, output="sos"), car, axis=0)
        muscle = sosfilt(butter(4, [20., 200.], btype="bandpass", fs=fs, output="sos"), emg, axis=0)
        gamma_trials = np.asarray([log_rms_windows(gamma, cue, fs) for cue in cues])
        emg_trials = np.asarray([log_rms_windows(muscle, cue, fs) for cue in cues])
        gamma_delta = gamma_trials[:, mi].mean(1)-gamma_trials[:, pre].mean(1)
        emg_delta = emg_trials[:, mi].mean(1)-emg_trials[:, pre].mean(1)
        within = [spearmanr(g, e).statistic for g, e in zip(gamma_trials, emg_trials)]
        rows.append({
            "subject": subject,
            "gamma_delta_median": float(np.median(gamma_delta)),
            "emg_delta_median": float(np.median(emg_delta)),
            "within_trial_gamma_emg_rho_median": float(np.nanmedian(within)),
            "trial_delta_gamma_emg_rho": float(spearmanr(gamma_delta, emg_delta).statistic),
            "gamma_trajectory": gamma_trials.mean(0).tolist(),
            "emg_trajectory": emg_trials.mean(0).tolist(),
        })
        print(f"gamma-EMG {index}/{len(args.subjects)} subject={subject}", flush=True)
    gamma_delta = np.asarray([r["gamma_delta_median"] for r in rows])
    emg_delta = np.asarray([r["emg_delta_median"] for r in rows])
    within = np.asarray([r["within_trial_gamma_emg_rho_median"] for r in rows])
    between = spearmanr(gamma_delta, emg_delta)
    summary = {
        "subjects": len(rows),
        "gamma_mi_minus_precue_log_rms_median": float(np.median(gamma_delta)),
        "gamma_change_wilcoxon_p": float(wilcoxon(gamma_delta).pvalue),
        "emg_mi_minus_precue_log_rms_median": float(np.median(emg_delta)),
        "emg_change_wilcoxon_p": float(wilcoxon(emg_delta).pvalue),
        "within_trial_gamma_emg_rho_subject_median": float(np.median(within)),
        "subject_delta_gamma_emg_spearman_rho": float(between.statistic),
        "subject_delta_gamma_emg_spearman_p": float(between.pvalue),
        "interpretation_rule": (
            "Gamma is considered EMG-linked only if EMG also rises and either "
            "within-trial or across-subject coupling is materially positive."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
