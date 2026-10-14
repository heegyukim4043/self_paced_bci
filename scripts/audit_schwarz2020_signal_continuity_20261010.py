"""Descriptive EEG-sample discontinuity audit of three BNCI 001-2020 files.

This cannot prove acquisition chronology: unmarked gaps or re-filtered joins may
leave little amplitude jump. It only tests whether marked run joins are visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("participant", choices=("G01", "V01", "H01"))
    participant = parser.parse_args().participant
    path = ROOT / "data" / "raw" / "schwarz2020_audit_20261010" / f"{participant}.mat"
    audit = json.loads((ROOT / "data" / "processed" /
                        f"schwarz2020_{participant.lower()}_raw_audit_20261010_v6.json")
                       .read_text(encoding="utf-8"))
    out = ROOT / "data" / "processed" / f"schwarz2020_{participant.lower()}_continuity_20261010_v3.json"
    if out.exists():
        raise FileExistsError(out)
    mat = loadmat(path, variable_names=["signal", "header"], squeeze_me=True,
                  struct_as_record=False)
    signal, h = mat["signal"], mat["header"]
    eeg_ids = np.ravel(h.channels_eeg).astype(int) - 1
    labels = [str(v) for v in np.ravel(h.channels_labels)]
    eeg_ids = np.asarray([j for j in eeg_ids if j < len(labels) and labels[j] != "A2"], dtype=int)
    if not len(eeg_ids) or np.min(eeg_ids) < 0 or np.max(eeg_ids) >= signal.shape[0]:
        raise AssertionError("invalid EEG channel indices")
    selected = eeg_ids[np.unique(np.rint(np.linspace(0, len(eeg_ids)-1,
                                                min(12, len(eeg_ids)))).astype(int))]
    x = signal[selected, :]
    fs = float(h.sample_rate)
    n_steps = x.shape[1] - 1
    sampled = np.arange(0, n_steps, 100, dtype=np.int64)
    baseline_delta = np.abs(x[:, sampled+1] - x[:, sampled])
    scale = np.median(baseline_delta, axis=1)
    if np.any(scale <= 0):
        raise AssertionError("zero EEG derivative scale")
    sampled_metric = np.median(baseline_delta / scale[:, None], axis=0)
    sample_quantiles = np.quantile(sampled_metric, [.5, .9, .99, .999])
    candidate_indices = []
    for start in range(0, n_steps, 50_000):
        end = min(start + 50_000, n_steps)
        diff = np.abs(x[:, start+1:end+1] - x[:, start:end])
        metric = np.median(diff / scale[:, None], axis=0)
        take = min(25, len(metric))
        local = np.argpartition(metric, -take)[-take:]
        candidate_indices.extend((float(metric[j]), int(start+j)) for j in local)
    top = sorted(candidate_indices, reverse=True)[:30]

    def boundary_record(sample_1based: int, kind: str, label: str):
        # Between 1-based samples `sample_1based` and `sample_1based+1`.
        j = sample_1based - 1
        if j < 0 or j >= n_steps:
            raise AssertionError(f"boundary {sample_1based} outside signal")
        value = float(np.median(np.abs(x[:, j+1]-x[:, j]) / scale))
        return {"kind": kind, "label": label, "after_sample_1based": sample_1based,
                "time_s": float(sample_1based/fs), "robust_jump_score": value,
                "reference_sample_percentile": float(np.mean(sampled_metric <= value))}

    boundaries = []
    for i, run in enumerate(audit["run_details"][:-1]):
        boundaries.append(boundary_record(run["end_sample_1based"], "header_run_join",
                                          f'{run["task_code"]}->{audit["run_details"][i+1]["task_code"]}'))
    for i, rest in enumerate(audit["rest_pairs_chronological"]):
        boundaries.append(boundary_record(rest["onset_sample_1based"] - 1,
                                          "rest_onset_marker", f"rest_{i+1}"))
    result = {
        "participant": participant,
        "status": "descriptive original signal jump audit; not chronological proof",
        "selected_eeg_rows_0based": [int(j) for j in selected],
        "selected_labels": [labels[j] for j in selected],
        "baseline_abs_sample_derivative_uv": [float(j) for j in scale],
        "reference_jump_score_quantiles_50_90_99_99_9": [float(j) for j in sample_quantiles],
        "boundaries": boundaries,
        "top_30_unrestricted_jump_samples": [{"after_sample_1based": j+1,
                                               "time_s": (j+1)/fs,
                                               "robust_jump_score": v} for v,j in top],
        "limitation": "A large join jump suggests discontinuity; a small jump does not establish original recording continuity or run order. Physiologic/artifact jumps can be large too.",
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(participant, "ref q", result["reference_jump_score_quantiles_50_90_99_99_9"],
          "run joins", [(v["label"], round(v["robust_jump_score"], 2),
                          round(v["reference_sample_percentile"], 4))
                         for v in boundaries if v["kind"] == "header_run_join"], flush=True)


if __name__ == "__main__":
    main()
