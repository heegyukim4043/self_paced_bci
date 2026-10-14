"""Past-only Lee2019 adapter for the published LaBraM base checkpoint.

The official model is an external, versioned checkout. A failed or partial
checkpoint load is an error; it must never silently become a random backbone.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, iirnotch, sosfilt, lfilter

from .cbramod_transfer import (
    SFREQ, PATCH_SAMPLES, CONTEXT_SECONDS, WINDOW, STEP, PRETRAIN_CHANNELS,
    selected_channel_indices, training_ends, decision_ends, regular_ends,
    trailing_windows, sha256_file, upstream_commit,
)
from .data import LeeRun


MODEL_NAME = "LaBraM_base_pretrained_frozen_linear_probe"
PROTOCOL_VERSION = "labram_lee_v1_causal_0p1_75_notch50_2s_motorprobe"
FEATURE_DIM = 800  # Global + C3/Cz/C4 final-patch embeddings, 200 each.

# Indices are from the official utils.standard_1020 list. Lee's T7/T8/P7/P8
# electrodes are addressed by their modern names, not the T3/T4/T5/T6 aliases
# used to locate the same physical channels in the CBraMod input order.
POSITION_INDEX = {
    "FP1": 0, "FP2": 2, "F3": 17, "F4": 21, "C3": 39, "C4": 43,
    "P3": 61, "P4": 65, "O1": 80, "O2": 82, "F7": 15, "F8": 23,
    "T3": 37, "T4": 45, "T5": 59, "T6": 67,
    "FZ": 19, "CZ": 41, "PZ": 63,
}


def input_channel_positions() -> list[int]:
    """Official LaBraM get_input_chans convention: CLS=0, channels=index+1."""
    positions = [0] + [POSITION_INDEX[name.upper()] + 1 for name in PRETRAIN_CHANNELS]
    if len(positions) != 20 or len(set(positions)) != 20 or max(positions) >= 129:
        raise AssertionError("invalid LaBraM spatial position indices")
    return positions


def causal_lee_200hz(samples: np.ndarray, channels: tuple[str, ...],
                     native_sfreq: float, *, input_unit: str) -> np.ndarray:
    """Return [19, time] microvolt data; no acausal filtering or resampling.

    This implements the official 0.1–75 Hz / 50 Hz notch / 200 Hz / uV target
    with causal IIR filters and integer decimation, a documented departure
    from the upstream offline zero-phase processing. No future EEG is used.
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
        raise ValueError(f"implausible microvolt scale (median={median_abs:g})")
    sos = butter(4, (0.1, 75.0), btype="bandpass", fs=native_sfreq,
                 output="sos")
    band = sosfilt(sos, selected, axis=0)
    b, a = iirnotch(50.0, Q=30.0, fs=native_sfreq)
    filtered = lfilter(b, a, band, axis=0)
    return np.asarray(filtered[::int(round(ratio))].T, dtype=np.float32)


def _student_weights(checkpoint: object, model: torch.nn.Module) -> dict:
    """Select only pretrained student encoder keys and verify full coverage."""
    if not isinstance(checkpoint, dict):
        raise ValueError("LaBraM checkpoint must be a state-dict container")
    state = checkpoint.get("model", checkpoint.get("module", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("LaBraM checkpoint contains no model state dict")
    prefixes = ("student.", "module.student.")
    if any(key.startswith(prefixes) for key in state):
        state = {key.split("student.", 1)[1]: value for key, value in state.items()
                 if key.startswith(prefixes)}
    target = model.state_dict()
    weights = {}
    for key, tensor in state.items():
        if key in target:
            if not isinstance(tensor, torch.Tensor) or tensor.shape != target[key].shape:
                raise ValueError(f"LaBraM checkpoint shape mismatch: {key}")
            weights[key] = tensor
    missing = set(target) - set(weights)
    if missing:
        raise ValueError(f"incomplete LaBraM pretrained encoder: {sorted(missing)[:12]}")
    if not any(key.startswith("blocks.11.") for key in weights):
        raise ValueError("LaBraM final transformer block missing")
    return weights


def _make_official_backbone(module):
    """Match the published base pretraining/fine-tuning layer-scale setting."""
    return module.labram_base_patch200_200(
        pretrained=False, EEG_size=WINDOW, num_classes=0,
        use_mean_pooling=False, use_abs_pos_emb=True,
        use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
        qkv_bias=False, init_values=0.1,
    )


def load_frozen_backbone(repo: Path, checkpoint: Path, device: torch.device):
    repo, checkpoint = repo.resolve(), checkpoint.resolve()
    source = repo / "modeling_finetune.py"
    if not source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"official LaBraM source/checkpoint missing: {source}, {checkpoint}")
    spec = importlib.util.spec_from_file_location("_official_labram_finetune", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official LaBraM source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = _make_official_backbone(module)
    try:
        # Official training checkpoints may retain their argparse.Namespace.
        # Allowlist only that standard-library container, never arbitrary code.
        with torch.serialization.safe_globals([argparse.Namespace]):
            raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "LaBraM checkpoint cannot be read safely with weights_only=True; "
            "verify or convert the official checkpoint to a tensor-only state dict"
        ) from exc
    weights = _student_weights(raw, model)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def motor_features(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1:] != (38, 200):
        raise AssertionError(f"unexpected LaBraM patch-token shape: {tuple(tokens.shape)}")
    last = tokens.reshape(tokens.shape[0], 19, 2, 200)[:, :, -1, :]
    return torch.cat((last.mean(1), last[:, PRETRAIN_CHANNELS.index("C3")],
                      last[:, PRETRAIN_CHANNELS.index("Cz")],
                      last[:, PRETRAIN_CHANNELS.index("C4")]), dim=1)


@torch.inference_mode()
def encode_ends(model, signal: np.ndarray, ends: np.ndarray,
                device: torch.device, *, batch_size: int = 32) -> np.ndarray:
    if batch_size < 1 or len(ends) == 0:
        raise ValueError("batch size and endpoints must be nonempty")
    model.eval()
    parts = []
    for start in range(0, len(ends), batch_size):
        windows = trailing_windows(signal, ends[start:start + batch_size])
        tokens = model(torch.from_numpy(windows).to(device),
                       input_chans=input_channel_positions(),
                       return_patch_tokens=True)
        parts.append(motor_features(tokens).float().cpu().numpy())
    features = np.concatenate(parts).astype(np.float32)
    if features.shape != (len(ends), FEATURE_DIM) or not np.all(np.isfinite(features)):
        raise AssertionError("invalid LaBraM features")
    return features


def preprocess_run(run: LeeRun, *, input_unit: str,
                   stop_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    stop_native = int(round(stop_seconds * run.sfreq))
    if stop_native > len(run.eeg):
        raise ValueError("requested stop lies after Lee recording")
    signal = causal_lee_200hz(run.eeg[:stop_native], run.channels, run.sfreq,
                              input_unit=input_unit)
    anchor = causal_lee_200hz(run.pre_rest, run.channels, run.sfreq,
                              input_unit=input_unit)
    return signal, anchor
