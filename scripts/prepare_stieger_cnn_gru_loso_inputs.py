"""Prepare reusable, session-normalized Stieger inputs for CNN-GRU LOSO.

The raw MAT files are expensive to filter repeatedly.  This script performs
the frozen causal preprocessing once per subject-session and stores a flat
memory-mappable signal plus trial metadata.  Normalization uses only the
chronological first-half preparation period and never uses active/rest labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.config import ExperimentConfig
from scripts.prepare_stieger_cnn_gru_score_cache import (
    BANDS,
    STEP_SECONDS,
    TARGET_SFREQ,
    WINDOW_SECONDS,
    _field,
    _parse_subject_session,
    filter_cell,
    normalize_signals,
    preparation_stats,
    selected_indices,
)


PROTOCOL_VERSION = 1


def prepare_one(source: Path, target: Path, *, overwrite: bool = False) -> dict[str, Any]:
    metadata_path = target / "metadata.npz"
    signal_path = target / "signals.npy"
    if metadata_path.exists() and signal_path.exists() and not overwrite:
        with np.load(metadata_path, allow_pickle=False) as saved:
            audit = json.loads(str(saved["audit_json"]))
        if int(audit.get("protocol_version", -1)) == PROTOCOL_VERSION:
            return audit

    started = time.time()
    loaded = loadmat(
        source, struct_as_record=False, squeeze_me=True, variable_names=["BCI"]
    )
    bci = loaded["BCI"]
    data = np.atleast_1d(_field(bci, "data"))
    trials = np.atleast_1d(_field(bci, "TrialData"))
    labels = [str(value) for value in np.atleast_1d(_field(bci.chaninfo, "label"))]
    native_sfreq = float(_field(bci, "SRATE"))
    subject, session = _parse_subject_session(source)
    config = ExperimentConfig(
        window_seconds=WINDOW_SECONDS, feature_step_seconds=STEP_SECONDS
    )
    indices = selected_indices(labels, config.sensorimotor_channels)
    total_trials = len(data)
    split_index = total_trials // 2

    targets = np.asarray(
        [int(_field(trial, "targetnumber")) for trial in trials], dtype=np.int8
    )
    artifacts = np.asarray(
        [int(_field(trial, "artifact")) for trial in trials], dtype=np.int8
    )
    trial_lengths = np.asarray(
        [float(_field(trial, "triallength")) for trial in trials], dtype=np.float32
    )
    run_numbers = np.asarray(
        [int(_field(trial, "runnumber")) for trial in trials], dtype=np.int16
    )
    task_numbers = np.asarray(
        [int(_field(trial, "tasknumber")) for trial in trials], dtype=np.int8
    )
    result = np.asarray(
        [float(_field(trial, "result")) for trial in trials], dtype=np.float32
    )
    forced_result = np.asarray(
        [int(_field(trial, "forcedresult")) for trial in trials], dtype=np.int8
    )

    signals: list[np.ndarray | None] = []
    for index, (raw, artifact) in enumerate(zip(data, artifacts)):
        signals.append(
            None
            if artifact
            else filter_cell(raw, indices, native_sfreq, config.filter_order)
        )
        if (index + 1) % 100 == 0:
            print(f"    filtered {index + 1}/{total_trials}", flush=True)

    mean, std, normalization_samples = preparation_stats(
        signals, artifacts, split_index
    )
    signals = normalize_signals(signals, mean, std)
    lengths = np.asarray(
        [0 if signal is None else signal.shape[-1] for signal in signals],
        dtype=np.int64,
    )
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    flat = np.empty((int(offsets[-1]), 3, 20), dtype=np.float32)
    for index, signal in enumerate(signals):
        if signal is not None:
            flat[offsets[index] : offsets[index + 1]] = np.moveaxis(signal, -1, 0)

    if target.exists() and overwrite:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    np.save(signal_path, flat, allow_pickle=False)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "source": str(source.resolve()),
        "target": str(target.resolve()),
        "subject": subject,
        "session": session,
        "trial_cells": total_trials,
        "split_index": split_index,
        "stored_samples": int(len(flat)),
        "normalization_samples_per_band_channel": int(normalization_samples),
        "normalization": "target-session first-half preparation only; labels unused",
        "preprocessing": {
            "reference": "CAR over all EEG channels before sensorimotor20 selection",
            "bands_hz": [list(band) for band in BANDS],
            "filter": f"causal Butterworth order {config.filter_order}",
            "native_sfreq_hz": native_sfreq,
            "target_sfreq_hz": TARGET_SFREQ,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    np.savez_compressed(
        metadata_path,
        trial_offsets=offsets,
        trial_target_number=targets,
        trial_artifact=artifacts,
        trial_length_s=trial_lengths,
        trial_run_number=run_numbers,
        trial_task_number=task_numbers,
        trial_result=result,
        trial_forced_result=forced_result,
        normalization_mean=mean,
        normalization_std=std,
        channels=np.asarray(config.sensorimotor_channels),
        bands_hz=np.asarray(BANDS, dtype=np.float32),
        audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/stieger2021"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/stieger2021_cnn_gru_loso_inputs"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_loso_input_manifest.json"),
    )
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--sessions", type=int, nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.raw_root.glob("S*_Session_*.mat"))
    if args.subjects:
        keep = set(args.subjects)
        paths = [path for path in paths if _parse_subject_session(path)[0] in keep]
    if args.sessions:
        keep_sessions = set(args.sessions)
        paths = [
            path for path in paths
            if _parse_subject_session(path)[1] in keep_sessions
        ]
    if not paths:
        raise FileNotFoundError(f"no Stieger MAT files under {args.raw_root}")

    rows = []
    for index, source in enumerate(paths, start=1):
        subject, session = _parse_subject_session(source)
        print(f"[{index}/{len(paths)}] S{subject} session {session}", flush=True)
        row = prepare_one(
            source,
            args.output_root / f"session{session}" / f"sub-{subject:02d}",
            overwrite=args.overwrite,
        )
        rows.append(row)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "protocol": {
                        "version": PROTOCOL_VERSION,
                        "purpose": "reusable causal inputs for subject-LOSO CNN-GRU",
                        "normalization": "session first-half preparation; no active/rest labels",
                    },
                    "completed_subject_sessions": len(rows),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"    samples={row['stored_samples']} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
