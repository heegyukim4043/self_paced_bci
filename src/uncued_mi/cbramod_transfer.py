"""Causal, source-only Lee2019 feature extraction for pretrained CBraMod.

The upstream backbone is deliberately an external dependency. Its published
checkpoint is loaded strictly; no random-weight fallback is allowed. Every
prediction uses only the two seconds ending at that decision time.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt

from .data import LeeRun


SFREQ = 200
PATCH_SAMPLES = 200
CONTEXT_SECONDS = 2
WINDOW = SFREQ * CONTEXT_SECONDS
STEP = 20  # 0.1 s, identical to the frozen original-continuous score grid.
MODEL_NAME = "CBraMod_pretrained_frozen_linear_probe"
PROTOCOL_VERSION = "cbramod_lee_v1_car_causal_0p3_40_2s_motorprobe"
FINETUNED_MODEL_NAME = "CBraMod_pretrained_finetuned_LOSO"
FINETUNE_PROTOCOL = "cbramod_lee_v1_full_backbone_3epoch_adamw_source0_49"

# Upstream TUEG pretraining used this 19-channel order. Lee names modern
# temporal electrodes T7/T8/P7/P8 rather than T3/T4/T5/T6.
PRETRAIN_CHANNELS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)
LEE_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
FEATURE_DIM = 800  # Global + C3 + Cz + C4, each 200-dimensional.


def selected_channel_indices(channels: tuple[str, ...]) -> np.ndarray:
    lookup = {name.casefold(): i for i, name in enumerate(channels)}
    indices = []
    missing = []
    for name in PRETRAIN_CHANNELS:
        candidate = name if name.casefold() in lookup else LEE_ALIASES.get(name, name)
        if candidate.casefold() not in lookup:
            missing.append(f"{name}/{candidate}")
        else:
            indices.append(lookup[candidate.casefold()])
    if missing:
        raise ValueError(f"Lee recording lacks CBraMod montage channels: {missing}")
    if len(set(indices)) != len(PRETRAIN_CHANNELS):
        raise AssertionError("duplicate mapped EEG channels")
    return np.asarray(indices, dtype=np.int64)


def causal_lee_200hz(samples: np.ndarray, channels: tuple[str, ...],
                     native_sfreq: float, *, input_unit: str) -> np.ndarray:
    """Return [19, time] microvolt EEG with no future-dependent operations.

    Lee's 1000 -> 200 Hz conversion is intentionally integer decimation after
    causal IIR antialias filtering. A non-integer rate must be audited, not
    silently passed through a zero-phase/Fourier resampler.
    """
    if samples.ndim != 2 or samples.shape[1] != len(channels):
        raise ValueError("expected [time, channels] Lee EEG")
    ratio = native_sfreq / SFREQ
    if not np.isclose(ratio, round(ratio), atol=1e-9) or ratio < 2:
        raise ValueError(f"unsupported native sampling frequency: {native_sfreq}")
    if input_unit not in ("uV", "V"):
        raise ValueError("input_unit must be explicitly 'uV' or 'V'")
    selected = samples[:, selected_channel_indices(channels)].astype(np.float64)
    if input_unit == "V":
        selected *= 1e6
    if not np.all(np.isfinite(selected)):
        raise ValueError("non-finite input EEG")
    probe = selected[::max(1, len(selected) // 5000)]
    median_abs = float(np.median(np.abs(probe)))
    if not 0.01 <= median_abs <= 5000.0:
        raise ValueError(f"implausible microvolt scale (median={median_abs:g}); check --input-unit")
    # Match the upstream BCIC-IV-2a MI preprocessing: a per-timepoint common
    # average reference. This operation is causal (no time neighbours used).
    selected -= selected.mean(axis=1, keepdims=True)
    # The upstream MI downstream pipeline uses 0.3-40 Hz; the causal IIR and
    # integer decimation here replace its whole-trial Fourier resampling.
    sos = butter(4, (0.3, 40.0), btype="bandpass", fs=native_sfreq,
                 output="sos")
    filtered = sosfilt(sos, selected, axis=0)
    return np.asarray(filtered[::int(round(ratio))].T, dtype=np.float32)


def trailing_windows(signal: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """[batch, channels, 2 patches, 200 points] with support <= each end."""
    ends = np.asarray(ends, dtype=np.int64)
    if signal.ndim != 2 or signal.shape[0] != len(PRETRAIN_CHANNELS):
        raise ValueError("expected 19-channel signal")
    if np.any(ends < WINDOW) or np.any(ends > signal.shape[1]):
        raise ValueError("window endpoint outside causal support")
    return np.stack([signal[:, end - WINDOW:end].reshape(
        len(PRETRAIN_CHANNELS), CONTEXT_SECONDS, PATCH_SAMPLES)
        for end in ends]).astype(np.float32)


def training_ends(cue_times: np.ndarray, n_samples: int,
                  *, first_trials: int = 50, max_per_class: int = 128
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use source trials 0..49 only; positives have full MI window support.

    Negatives are fully supported by fixation or late ITI, never pre-rest or
    evaluation trials. This changes the training domain from the old simulated
    CNN-GRU recipe, and is documented as a distinct representation experiment.
    """
    cue_times = np.asarray(cue_times, dtype=float)
    if len(cue_times) < first_trials + 1 or max_per_class < 1:
        raise ValueError("insufficient trials or invalid class cap")
    positive: list[tuple[int, int]] = []
    negative: list[tuple[int, int]] = []
    for trial in range(first_trials):
        cue = float(cue_times[trial])
        next_cue = float(cue_times[trial + 1])
        # Full 2-s input is after the excluded cue-onset half-second.
        for offset in (2.5, 3.0, 3.5):
            positive.append((int(round((cue + offset) * SFREQ)), trial))
        # Full input lies in the cue-3..cue fixation interval.
        for offset in (-1.0, -0.6, -0.2):
            negative.append((int(round((cue + offset) * SFREQ)), trial))
        # Full input lies after MI and before the next fixation onset.
        for offset in (7.0, 7.4):
            if cue + offset <= next_cue - 3.2:
                negative.append((int(round((cue + offset) * SFREQ)), trial))

    def select(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        pairs = [(end, trial) for end, trial in pairs if WINDOW <= end <= n_samples]
        pairs = sorted(set(pairs))
        if len(pairs) < max_per_class:
            raise ValueError(f"only {len(pairs)} supported training windows")
        chosen = np.linspace(0, len(pairs) - 1, max_per_class, dtype=np.int64)
        return [pairs[int(i)] for i in chosen]

    pos, neg = select(positive), select(negative)
    entries = [(end, 1, trial) for end, trial in pos] + [
        (end, 0, trial) for end, trial in neg]
    entries.sort()
    ends = np.asarray([item[0] for item in entries], dtype=np.int64)
    labels = np.asarray([item[1] for item in entries], dtype=np.int8)
    trial_ids = np.asarray([item[2] for item in entries], dtype=np.int8)
    if np.any(trial_ids >= first_trials) or len(np.unique(ends)) != len(ends):
        raise AssertionError("source training trial split or endpoint overlap")
    return ends, labels, trial_ids


def decision_ends(times: np.ndarray, n_samples: int) -> np.ndarray:
    ends = np.rint(np.asarray(times, dtype=float) * SFREQ).astype(np.int64)
    if len(ends) == 0 or np.any(np.diff(ends) <= 0):
        raise ValueError("non-increasing decision times")
    if np.max(np.abs(ends / SFREQ - times)) > 0.00251:
        raise ValueError("decision grid incompatible with 200 Hz")
    if ends[0] < WINDOW or ends[-1] > n_samples:
        raise ValueError("decision grid outside filtered signal")
    return ends


def regular_ends(n_samples: int) -> np.ndarray:
    return np.arange(WINDOW, n_samples + 1, STEP, dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError(f"CBraMod repo is not a versioned checkout: {repo}") from None


def load_frozen_backbone(repo: Path, checkpoint: Path, device: torch.device):
    """Load the official upstream model and published checkpoint strictly."""
    repo, checkpoint = repo.resolve(), checkpoint.resolve()
    if not (repo / "models" / "cbramod.py").is_file():
        raise FileNotFoundError(f"official CBraMod source missing under {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"pretrained CBraMod checkpoint missing: {checkpoint}")
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("models.cbramod")
    finally:
        sys.path.pop(0)
    if not Path(module.__file__).resolve().is_relative_to(repo):
        raise ImportError("a different 'models.cbramod' module was imported")
    cls = module.CBraMod
    model = cls()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.proj_out = torch.nn.Identity()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def motor_features(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.ndim != 4 or encoded.shape[1:] != (
            len(PRETRAIN_CHANNELS), CONTEXT_SECONDS, 200):
        raise AssertionError(f"unexpected CBraMod feature shape: {tuple(encoded.shape)}")
    last = encoded[:, :, -1, :]
    return torch.cat((
        last.mean(dim=1),
        last[:, PRETRAIN_CHANNELS.index("C3")],
        last[:, PRETRAIN_CHANNELS.index("Cz")],
        last[:, PRETRAIN_CHANNELS.index("C4")],
    ), dim=1)


class CBraModBinaryClassifier(torch.nn.Module):
    """Published backbone plus a small, trainable MI-vs-task-idle head."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(FEATURE_DIM, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.head(motor_features(self.backbone(windows))).squeeze(-1)


@torch.inference_mode()
def encode_ends(model, signal: np.ndarray, ends: np.ndarray,
                device: torch.device, *, batch_size: int = 32) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    features = []
    for start in range(0, len(ends), batch_size):
        windows = trailing_windows(signal, ends[start:start + batch_size])
        encoded = model(torch.from_numpy(windows).to(device))
        features.append(motor_features(encoded).float().cpu().numpy())
    if not features:
        raise ValueError("no decision endpoints")
    result = np.concatenate(features).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite CBraMod features")
    return result


@torch.inference_mode()
def predict_ends(model: CBraModBinaryClassifier, signal: np.ndarray,
                 ends: np.ndarray, device: torch.device,
                 *, batch_size: int = 32) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    scores = []
    for start in range(0, len(ends), batch_size):
        windows = trailing_windows(signal, ends[start:start + batch_size])
        scores.append(model(torch.from_numpy(windows).to(device)).float().cpu().numpy())
    if not scores:
        raise ValueError("no decision endpoints")
    result = np.concatenate(scores).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite fine-tuned CBraMod scores")
    return result


def preprocess_run(run: LeeRun, *, input_unit: str, stop_seconds: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    stop_native = int(round(stop_seconds * run.sfreq))
    if stop_native > len(run.eeg):
        raise ValueError("requested stop lies after Lee recording")
    signal = causal_lee_200hz(run.eeg[:stop_native], run.channels, run.sfreq,
                              input_unit=input_unit)
    anchor = causal_lee_200hz(run.pre_rest, run.channels, run.sfreq,
                              input_unit=input_unit)
    return signal, anchor
