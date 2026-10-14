"""Build frozen-protocol, trial-wise CNN-GRU score caches for Stieger2021.

Stieger's public MAT files contain independent trial cells.  Every cell is
filtered and scored independently, and CNN-GRU hidden state is reset at every
cell boundary.  The first chronological half is supervised training; the
second half is locked for first-10 screening and future evaluation.

This is a partial external-validation cache.  It must not be interpreted as a
continuous long-idle self-paced stream.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.config import ExperimentConfig
from uncued_mi.features import (
    _causal_band_signal,
    _causal_downsample,
    common_average_reference,
)
from scripts.audit_stieger_cnn_gru_compatibility import (
    FIRST_ACTIVE_SCREENING,
    chronological_split_counts,
)
from scripts.run_lee2019_cnn_gru import CausalCNNGRU


TARGET_SFREQ = 250.0
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.1
FILTER_WARMUP_SECONDS = 2.0
CELL_TARGET_ONSET_SECONDS = 2.0
CELL_FEEDBACK_ONSET_SECONDS = 4.0
BANDS = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
DEFAULT_SEEDS = (20260729, 20260730, 20260731)


def _field(struct: Any, name: str) -> Any:
    if not hasattr(struct, name):
        raise ValueError(f"missing required field: {name}")
    return getattr(struct, name)


def _parse_subject_session(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"S(\d+)_Session_(\d+)\.mat", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unexpected Stieger filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def selected_indices(labels: list[str], selected: tuple[str, ...]) -> np.ndarray:
    lookup = {label.casefold(): index for index, label in enumerate(labels)}
    missing = [channel for channel in selected if channel.casefold() not in lookup]
    if missing:
        raise ValueError(f"missing sensorimotor channels: {missing}")
    return np.asarray([lookup[channel.casefold()] for channel in selected], dtype=np.int64)


def filter_cell(
    eeg_channels_by_samples: np.ndarray,
    indices: np.ndarray,
    native_sfreq: float,
    filter_order: int,
) -> np.ndarray:
    """Return causal 3-band signal as band x channel x sample at 250 Hz."""
    raw = np.asarray(eeg_channels_by_samples, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("cell EEG must have shape channels x samples")
    referenced = common_average_reference(raw.T)
    selected = referenced[:, indices]
    output = []
    for low, high in BANDS:
        filtered = _causal_band_signal(
            selected, native_sfreq, low, high, filter_order
        )
        downsampled = _causal_downsample(filtered, native_sfreq, TARGET_SFREQ)
        output.append(downsampled.T.astype(np.float32, copy=False))
    return np.asarray(output, dtype=np.float32)


def cell_window_ends(
    trial_length_s: float,
    signal_samples: int,
    *,
    sfreq: float = TARGET_SFREQ,
    window_seconds: float = WINDOW_SECONDS,
    step_seconds: float = STEP_SECONDS,
    warmup_seconds: float = FILTER_WARMUP_SECONDS,
    target_onset_seconds: float = CELL_TARGET_ONSET_SECONDS,
    feedback_onset_seconds: float = CELL_FEEDBACK_ONSET_SECONDS,
) -> np.ndarray:
    """Causal window ends from valid preparation support through feedback.

    A window may straddle feedback onset. Its full support must begin after the
    filter warm-up, and its end may not exceed the feedback interval.
    """
    window = int(round(window_seconds * sfreq))
    step = int(round(step_seconds * sfreq))
    first_end = int(round((warmup_seconds + window_seconds) * sfreq))
    # With a -2 s cell start, this is target+1 s, i.e. feedback-1 s.
    if first_end < int(round(target_onset_seconds * sfreq)) + window:
        raise AssertionError("first window violates post-warmup support")
    feedback_stop = int(
        np.floor((feedback_onset_seconds + float(trial_length_s)) * sfreq + 1e-9)
    )
    last_end = min(int(signal_samples), feedback_stop)
    if first_end > last_end:
        return np.empty(0, dtype=np.int64)
    ends = np.arange(first_end, last_end + 1, step, dtype=np.int64)
    assert np.all(ends - window >= int(round(warmup_seconds * sfreq)))
    return ends


def times_from_feedback(ends: np.ndarray) -> np.ndarray:
    return np.asarray(ends, dtype=np.float32) / TARGET_SFREQ - CELL_FEEDBACK_ONSET_SECONDS


def window_labels(times_s: np.ndarray, target_number: int) -> np.ndarray:
    """Active feedback endpoints are positive; preparation/rest are negative."""
    return ((np.asarray(times_s) > 0.0) & (int(target_number) != 4)).astype(np.float32)


def make_windows(signal: np.ndarray, ends: np.ndarray) -> np.ndarray:
    window = int(round(WINDOW_SECONDS * TARGET_SFREQ))
    return np.asarray(
        [signal[:, :, int(end) - window : int(end)] for end in ends],
        dtype=np.float32,
    )


def preparation_stats(
    signals: list[np.ndarray | None],
    artifacts: np.ndarray,
    split_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit band/channel normalization on first-half preparation samples only."""
    start = int(round(CELL_TARGET_ONSET_SECONDS * TARGET_SFREQ))
    stop = int(round(CELL_FEEDBACK_ONSET_SECONDS * TARGET_SFREQ))
    total = None
    total_sq = None
    count = 0
    for index in range(split_index):
        signal = signals[index]
        if signal is None or artifacts[index]:
            continue
        segment = np.asarray(signal[..., start:stop], dtype=np.float64)
        if segment.shape[-1] != stop - start:
            raise ValueError(f"cell {index}: incomplete preparation normalization support")
        values = segment.sum(axis=-1)
        squares = np.square(segment).sum(axis=-1)
        total = values if total is None else total + values
        total_sq = squares if total_sq is None else total_sq + squares
        count += segment.shape[-1]
    if total is None or total_sq is None or count < 2:
        raise ValueError("no valid first-half preparation samples for normalization")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean[..., None].astype(np.float32), np.sqrt(variance)[..., None].astype(np.float32), count


def normalize_signals(
    signals: list[np.ndarray | None], mean: np.ndarray, std: np.ndarray
) -> list[np.ndarray | None]:
    return [
        None if signal is None else ((signal - mean) / std).astype(np.float32)
        for signal in signals
    ]


def build_sequences(
    signals: list[np.ndarray | None],
    targets: np.ndarray,
    artifacts: np.ndarray,
    trial_lengths: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ends_by_trial: list[np.ndarray] = []
    times_by_trial: list[np.ndarray] = []
    labels_by_trial: list[np.ndarray] = []
    for signal, target, artifact, length in zip(
        signals, targets, artifacts, trial_lengths
    ):
        if signal is None or artifact:
            ends = np.empty(0, dtype=np.int64)
        else:
            ends = cell_window_ends(float(length), signal.shape[-1])
        times = times_from_feedback(ends)
        ends_by_trial.append(ends)
        times_by_trial.append(times)
        labels_by_trial.append(window_labels(times, int(target)))
    return ends_by_trial, times_by_trial, labels_by_trial


def _length_buckets(indices: np.ndarray, lengths: np.ndarray, batch_size: int) -> list[np.ndarray]:
    ordered = indices[np.argsort(lengths[indices], kind="stable")]
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def train_trialwise(
    signals: list[np.ndarray | None],
    ends_by_trial: list[np.ndarray],
    labels_by_trial: list[np.ndarray],
    train_indices: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    sequence_batch_size: int,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = CausalCNNGRU(20, 32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    eligible = np.asarray(
        [index for index in train_indices if len(ends_by_trial[int(index)])],
        dtype=np.int64,
    )
    if not len(eligible):
        raise ValueError("no training sequences")
    lengths = np.asarray([len(values) for values in ends_by_trial], dtype=int)
    positives = int(sum(labels_by_trial[int(i)].sum() for i in eligible))
    total = int(sum(len(labels_by_trial[int(i)]) for i in eligible))
    negatives = total - positives
    if positives == 0 or negatives == 0:
        raise ValueError("training labels contain only one class")
    pos_weight = torch.tensor(max(negatives / positives, 1.0), device=device)

    batches = _length_buckets(eligible, lengths, sequence_batch_size)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        batch_order = rng.permutation(len(batches))
        epoch_loss: list[float] = []
        for batch_index in batch_order:
            chosen = np.asarray(batches[int(batch_index)], dtype=np.int64).copy()
            rng.shuffle(chosen)
            common_length = int(min(lengths[chosen]))
            x = np.stack(
                [
                    make_windows(signals[int(i)], ends_by_trial[int(i)])[:common_length]
                    for i in chosen
                ]
            )
            y = np.stack(
                [labels_by_trial[int(i)][:common_length] for i in chosen]
            ).astype(np.float32)
            tx = torch.from_numpy(x).to(device)
            ty = torch.from_numpy(y).to(device)
            optimizer.zero_grad(set_to_none=True)
            # hidden=None is deliberate: every batch row is an independent cell.
            logits, _ = model(tx, hidden=None)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, ty, pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_loss)))

    return model, {
        "training_sequences": int(len(eligible)),
        "training_windows": total,
        "training_positive_windows": positives,
        "training_negative_windows": negatives,
        "positive_class_weight": float(pos_weight.detach().cpu()),
        "epoch_mean_loss": losses,
        "sequence_batch_size": int(sequence_batch_size),
        "length_bucket_crop_rule": "sort by sequence length; crop each batch to shortest cell",
    }


@torch.no_grad()
def predict_windows(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    *,
    chunk_size: int = 100,
) -> np.ndarray:
    model.eval()
    hidden = None
    output: list[np.ndarray] = []
    for start in range(0, len(values), chunk_size):
        batch = torch.from_numpy(values[start : start + chunk_size][None]).to(device)
        logits, hidden = model(batch, hidden)
        hidden = hidden.detach()
        output.append(logits[0].float().cpu().numpy())
    return np.concatenate(output).astype(np.float32) if output else np.empty(0, np.float32)


def score_all_cells(
    model: nn.Module,
    signals: list[np.ndarray | None],
    ends_by_trial: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for signal, ends in zip(signals, ends_by_trial):
        if signal is None or not len(ends):
            output.append(np.empty(0, dtype=np.float32))
            continue
        # A new call always starts with hidden=None: no state crosses a seam.
        output.append(predict_windows(model, make_windows(signal, ends), device))
    return output


@torch.no_grad()
def model_causality_and_reset_audit(
    model: nn.Module,
    signals: list[np.ndarray | None],
    ends_by_trial: list[np.ndarray],
    device: torch.device,
) -> dict[str, float | int | bool]:
    candidates = [i for i, ends in enumerate(ends_by_trial) if len(ends) >= 4]
    if len(candidates) < 2:
        raise ValueError("need at least two scored cells for model audit")
    first, second = candidates[:2]
    x1 = make_windows(signals[first], ends_by_trial[first])
    x2 = make_windows(signals[second], ends_by_trial[second])
    prefix = min(4, len(x1))
    model.eval()
    full, _ = model(torch.from_numpy(x1[None]).to(device), hidden=None)
    short, _ = model(torch.from_numpy(x1[:prefix][None]).to(device), hidden=None)
    prefix_error = float(
        torch.max(torch.abs(full[0, :prefix] - short[0])).detach().cpu()
    )

    common = min(len(x1), len(x2))
    stacked = torch.from_numpy(np.stack([x1[:common], x2[:common]])).to(device)
    batched, _ = model(stacked, hidden=None)
    independent1, _ = model(stacked[0:1], hidden=None)
    independent2, _ = model(stacked[1:2], hidden=None)
    reset_error = float(
        max(
            torch.max(torch.abs(batched[0] - independent1[0])).detach().cpu(),
            torch.max(torch.abs(batched[1] - independent2[0])).detach().cpu(),
        )
    )
    tolerance = 1e-5
    return {
        "prefix_steps": int(prefix),
        "prefix_causality_max_abs_error": prefix_error,
        "independent_vs_batched_reset_max_abs_error": reset_error,
        "tolerance": tolerance,
        "passed": bool(prefix_error <= tolerance and reset_error <= tolerance),
    }


def _flatten_metadata(
    values_by_trial: list[np.ndarray], trial_values: np.ndarray, dtype: Any
) -> np.ndarray:
    return np.concatenate(
        [np.full(len(values), trial_values[i], dtype=dtype) for i, values in enumerate(values_by_trial)]
    )


def cache_one(
    source: Path,
    target: Path,
    *,
    epochs: int,
    seeds: tuple[int, ...],
    device: torch.device,
    sequence_batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    loaded = loadmat(
        source,
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["BCI"],
    )
    bci = loaded["BCI"]
    data = np.atleast_1d(_field(bci, "data"))
    trials = np.atleast_1d(_field(bci, "TrialData"))
    labels = [str(value) for value in np.atleast_1d(_field(bci.chaninfo, "label"))]
    native_sfreq = float(_field(bci, "SRATE"))
    subject, session = _parse_subject_session(source)
    config = ExperimentConfig(window_seconds=WINDOW_SECONDS, feature_step_seconds=STEP_SECONDS)
    indices = selected_indices(labels, config.sensorimotor_channels)
    total_trials = len(data)
    split_index = total_trials // 2

    targets = np.asarray([int(_field(trial, "targetnumber")) for trial in trials], dtype=np.int8)
    artifacts = np.asarray([int(_field(trial, "artifact")) for trial in trials], dtype=np.int8)
    trial_lengths = np.asarray([float(_field(trial, "triallength")) for trial in trials], dtype=np.float32)
    run_numbers = np.asarray([int(_field(trial, "runnumber")) for trial in trials], dtype=np.int16)
    task_numbers = np.asarray([int(_field(trial, "tasknumber")) for trial in trials], dtype=np.int8)
    result = np.asarray([float(_field(trial, "result")) for trial in trials], dtype=np.float32)
    forced_result = np.asarray([int(_field(trial, "forcedresult")) for trial in trials], dtype=np.int8)

    signals: list[np.ndarray | None] = []
    for index, (raw, artifact) in enumerate(zip(data, artifacts)):
        if artifact:
            signals.append(None)
            continue
        signal = filter_cell(raw, indices, native_sfreq, config.filter_order)
        signals.append(signal)
        if (index + 1) % 100 == 0:
            print(f"    filtered {index + 1}/{total_trials} cells", flush=True)

    mean, std, normalization_samples = preparation_stats(signals, artifacts, split_index)
    signals = normalize_signals(signals, mean, std)
    ends_by_trial, times_by_trial, labels_by_trial = build_sequences(
        signals, targets, artifacts, trial_lengths
    )
    split = chronological_split_counts(
        targets, artifacts, trial_lengths, split_index=split_index
    )
    train_indices = np.arange(split_index, dtype=np.int64)

    seed_scores: list[np.ndarray] = []
    training_audits: list[dict[str, Any]] = []
    model_audits: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds, start=1):
        print(
            f"    train seed {seed_index}/{len(seeds)} ({seed}), epochs={epochs}",
            flush=True,
        )
        model, training_audit = train_trialwise(
            signals,
            ends_by_trial,
            labels_by_trial,
            train_indices,
            seed=seed,
            epochs=epochs,
            device=device,
            sequence_batch_size=sequence_batch_size,
        )
        model_audit = model_causality_and_reset_audit(
            model, signals, ends_by_trial, device
        )
        if not model_audit["passed"]:
            raise AssertionError(f"causality/reset audit failed for seed {seed}: {model_audit}")
        scores_by_trial = score_all_cells(model, signals, ends_by_trial, device)
        seed_scores.append(np.concatenate(scores_by_trial).astype(np.float32))
        training_audits.append({"seed": seed, **training_audit})
        model_audits.append({"seed": seed, **model_audit})
        del model, scores_by_trial
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_score_array = np.asarray(seed_scores, dtype=np.float32)
    scores = seed_score_array.mean(axis=0).astype(np.float32)
    trial_offsets = np.r_[0, np.cumsum([len(values) for values in times_by_trial])].astype(np.int32)
    flat_times = np.concatenate(times_by_trial).astype(np.float32)
    flat_labels = np.concatenate(labels_by_trial).astype(np.int8)
    trial_index = _flatten_metadata(times_by_trial, np.arange(total_trials), np.int16)
    target_number = _flatten_metadata(times_by_trial, targets, np.int8)
    run_number = _flatten_metadata(times_by_trial, run_numbers, np.int16)
    task_number = _flatten_metadata(times_by_trial, task_numbers, np.int8)
    split_role = (trial_index >= split_index).astype(np.int8)
    feedback_mask = (flat_times > 0).astype(np.int8)
    preparation_mask = (flat_times <= 0).astype(np.int8)

    screening_cutoff = split["screening_cutoff_trial_index"]
    trial_protocol_role = np.zeros(total_trials, dtype=np.int8)
    trial_protocol_role[split_index : int(screening_cutoff) + 1] = 1
    trial_protocol_role[int(screening_cutoff) + 1 :] = 2

    report = {
        "source": str(source.resolve()),
        "target": str(target.resolve()),
        "subject": subject,
        "session": session,
        "model": "frozen Lee2019 CausalCNNGRU(20 channels, hidden=32)",
        "parameter_count": int(sum(p.numel() for p in CausalCNNGRU(20, 32).parameters())),
        "training_regime": "chronological within-subject first half",
        "locked_regime": "second half; first10 screening then future",
        "trial_cells": total_trials,
        "split_index": split_index,
        "score_windows": int(len(scores)),
        "training_protocol_counts": split,
        "normalization": {
            "source": "first-half target/preparation samples only",
            "samples_per_band_channel": int(normalization_samples),
            "future_samples_used": False,
        },
        "preprocessing": {
            "reference": "CAR over all 62 EEG channels before selecting sensorimotor20",
            "bands_hz": [list(band) for band in BANDS],
            "filter": f"causal Butterworth order {config.filter_order}",
            "native_sfreq_hz": native_sfreq,
            "target_sfreq_hz": TARGET_SFREQ,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "filter_warmup_seconds": FILTER_WARMUP_SECONDS,
        },
        "training": {
            "epochs": epochs,
            "seeds": list(seeds),
            "optimizer": "AdamW(lr=5e-4, weight_decay=1e-4)",
            "gradient_clip": 5.0,
            "seed_audits": training_audits,
        },
        "causality_reset_audits": model_audits,
        "seam_contract": {
            "hidden_reset_each_cell": True,
            "normalizer_fit_on_first_half_only": True,
            "window_crosses_cell_seam": False,
            "ewma_ma_dwell_not_applied_in_score_cache": True,
        },
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        scores=scores,
        seed_scores=seed_score_array,
        time_from_feedback_s=flat_times,
        window_label_active_feedback=flat_labels,
        feedback_mask=feedback_mask,
        preparation_mask=preparation_mask,
        trial_index=trial_index,
        target_number=target_number,
        run_number=run_number,
        task_number=task_number,
        split_role=split_role,
        trial_offsets=trial_offsets,
        trial_target_number=targets,
        trial_artifact=artifacts,
        trial_length_s=trial_lengths,
        trial_run_number=run_numbers,
        trial_task_number=task_numbers,
        trial_result=result,
        trial_forced_result=forced_result,
        trial_protocol_role=trial_protocol_role,
        normalization_mean=mean,
        normalization_std=std,
        channels=np.asarray(config.sensorimotor_channels),
        bands_hz=np.asarray(BANDS, dtype=np.float32),
        seeds=np.asarray(seeds, dtype=np.int64),
        audit_json=np.asarray(json.dumps(report, sort_keys=True)),
    )
    del loaded, bci, data, trials, signals
    gc.collect()
    return report


def validate_existing(path: Path, seeds: tuple[int, ...], epochs: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            audit = json.loads(str(cache["audit_json"]))
            if tuple(int(value) for value in cache["seeds"]) != tuple(seeds):
                return None
            if int(audit["training"]["epochs"]) != int(epochs):
                return None
            if not all(row["passed"] for row in audit["causality_reset_audits"]):
                return None
            if len(cache["scores"]) != cache["seed_scores"].shape[1]:
                return None
            return audit
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/stieger2021"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/stieger2021_cnn_gru_scores"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/stieger2021_cnn_gru_score_cache_manifest.json"),
    )
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--sessions", type=int, nargs="*")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seeds = tuple(args.seeds)
    paths = sorted(args.raw_root.glob("S*_Session_*.mat"))
    if args.subjects:
        selected = set(args.subjects)
        paths = [path for path in paths if _parse_subject_session(path)[0] in selected]
    if args.sessions:
        selected_sessions = set(args.sessions)
        paths = [path for path in paths if _parse_subject_session(path)[1] in selected_sessions]
    if not paths:
        raise FileNotFoundError(f"no Stieger MAT files under {args.raw_root}")

    rows: list[dict[str, Any]] = []
    if args.manifest.exists() and not args.overwrite:
        try:
            rows = json.loads(args.manifest.read_text(encoding="utf-8")).get("rows", [])
        except Exception:
            rows = []
    for file_index, source in enumerate(paths, start=1):
        subject, session = _parse_subject_session(source)
        target = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
        existing = None if args.overwrite else validate_existing(target, seeds, args.epochs)
        if existing is not None:
            print(f"[{file_index}/{len(paths)}] valid-cache skip {source.name}", flush=True)
            report = existing
        else:
            print(f"[{file_index}/{len(paths)}] build {source.name}", flush=True)
            report = cache_one(
                source,
                target,
                epochs=args.epochs,
                seeds=seeds,
                device=device,
                sequence_batch_size=args.sequence_batch_size,
            )
        rows = [
            row for row in rows
            if not (row.get("subject") == subject and row.get("session") == session)
        ]
        rows.append(report)
        rows.sort(key=lambda row: (row["session"], row["subject"]))
        payload = {
            "protocol": {
                "model": "Lee2019 frozen CausalCNNGRU architecture",
                "training": "Stieger chronological within-subject first half",
                "locked": "second-half first10 screening plus future",
                "epochs": args.epochs,
                "seeds": list(seeds),
                "trial_boundary_reset": True,
                "continuous_long_idle_claim": False,
                "device": args.device,
            },
            "completed_subject_sessions": len(rows),
            "rows": rows,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"    done S{subject} session {session}: windows={report['score_windows']} "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
