"""I6 LOSO CNN-GRU original-continuous score cache, with reset ablation.

Uses the frozen source-pooled training recipe; does not save/interpret EEG metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path, load_lee2019_run
from scripts.run_fbcnet_validation import filter_banks
from scripts.run_lee2019_cnn_gru import CausalCNNGRU
from scripts.run_lee2019_cnn_gru_loso_arm_a_stream_v2 import (
    SFREQ, STEP_SAMPLES, WINDOW, _cached_source_training, _normalize_banks,
    _window_ends, predict_signal, train_source_pool,
)


@torch.no_grad()
def predict_window_reset(model, signal, device, ends, batch_size=100):
    """One-window GRU state only, equivalent to independent forward calls."""
    model.eval()
    out = []
    for off in range(0, len(ends), batch_size):
        selected = ends[off:off + batch_size]
        batch = np.stack([signal[:, :, end-WINDOW:end] for end in selected]).astype(np.float32)
        encoded = model.encode(torch.from_numpy(batch).to(device))
        states, _ = model.gru(encoded[:, None, :])
        out.append(model.output(states[:, 0, :]).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def target_input(data_root, subject, session, stop_seconds):
    run = load_lee2019_run(lee2019_path(data_root, subject, session), "train")
    selected = ExperimentConfig().sensorimotor_channels
    stop_sample = int(round(stop_seconds * run.sfreq))
    if stop_sample > len(run.eeg):
        raise AssertionError("tier-0 endpoint outside original EEG")
    # Causal filtering makes this prefix identical to filtering the full run.
    raw = filter_banks(run.eeg[:stop_sample], run.channels, run.sfreq, selected)
    pre = filter_banks(run.pre_rest, run.channels, run.sfreq, selected)
    signal, anchor, _, _ = _normalize_banks(raw, pre)
    cues = np.asarray(run.cue_samples / run.sfreq, dtype=float)
    return signal, anchor, cues


def preflight_inference():
    torch.manual_seed(20261012)
    model = CausalCNNGRU(20, 32).eval()
    device = torch.device("cpu")
    rng = np.random.default_rng(20261012)
    signal = rng.normal(size=(3, 20, 750)).astype(np.float32)
    ends = np.arange(250, 751, 25)
    fast_reset = predict_window_reset(model, signal, device, ends)
    manual = []
    with torch.no_grad():
        for end in ends:
            window = torch.from_numpy(signal[:, :, end-WINDOW:end][None, None])
            logits, _ = model(window)
            manual.append(float(logits[0, 0]))
    if not np.allclose(fast_reset, manual, atol=2e-6, rtol=0):
        raise AssertionError("window-reset vectorization differs from independent forwards")
    original = predict_signal(model, signal, device, ends=ends)
    changed = signal.copy()
    changed[:, :, 500:] += 3.0
    future_changed = predict_signal(model, changed, device, ends=ends)
    if not np.allclose(original[ends <= 500], future_changed[ends <= 500], atol=1e-7, rtol=0):
        raise AssertionError("stateful inference responds to future input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260729, 20260730, 20260731])
    ap.add_argument("--max-source-windows", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    ap.add_argument("--train-cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    ap.add_argument("--prepared-source-root", type=Path, default=Path("data/interim/lee2019_cnn_gru_loso_prepared_source"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--output-root", type=Path, default=Path("data/interim/lee2019_i6_cnn_gru_original"))
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    preflight_inference()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")
    for session in args.sessions:
        source_pool = {
            source: _cached_source_training(args.data_root, args.train_cache_root,
                                            args.prepared_source_root, source, session,
                                            args.max_source_windows)
            for source in range(1, 55)
        }
        for subject in args.subjects:
            outpath = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if outpath.exists():
                with np.load(outpath, allow_pickle=False) as previous:
                    if tuple(previous["seeds"].tolist()) != tuple(args.seeds) or int(previous["epochs"]) != args.epochs:
                        raise AssertionError(f"Existing cache protocol mismatch: {outpath}")
                print(f"cache hit S{session} sub-{subject:02d}", flush=True)
                continue
            with np.load(args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz") as t:
                tier_times = np.asarray(t["continuous_times"])
                tier_cues = np.asarray(t["cue_samples"] / float(t["native_sfreq"]))
            signal, anchor, cues = target_input(args.data_root, subject, session, float(tier_times[-1]))
            ends = np.rint(tier_times * SFREQ).astype(np.int64)
            pre_ends = _window_ends(anchor.shape[-1], first_end=3*SFREQ)
            times = ends / SFREQ
            pre_times = pre_ends / SFREQ
            if len(times) != len(tier_times) or not np.allclose(times, tier_times, atol=.001, rtol=0):
                raise AssertionError("CNN original time grid != tier-0 grid")
            if ends[-1] > signal.shape[-1] or ends[0] < WINDOW:
                raise AssertionError("original CNN window endpoint outside signal")
            if not np.allclose(cues, tier_cues, atol=.005, rtol=0):
                raise AssertionError("CNN cue times != tier-0 cue times")
            source_data = [source_pool[s] for s in range(1, 55) if s != subject]
            scores, reset_scores, pre_scores, pre_reset_scores = [], [], [], []
            for seed in args.seeds:
                model = train_source_pool(source_data, seed, args.epochs, device)
                scores.append(predict_signal(model, signal, device, ends=ends))
                reset_scores.append(predict_window_reset(model, signal, device, ends))
                pre_scores.append(predict_signal(model, anchor, device, ends=pre_ends))
                pre_reset_scores.append(predict_window_reset(model, anchor, device, pre_ends))
                del model
            outpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                outpath, subject=np.int16(subject), session=np.int8(session),
                times=times, pre_times=pre_times, cue_times=cues,
                raw_scores=np.mean(scores, axis=0).astype(np.float32),
                reset_raw_scores=np.mean(reset_scores, axis=0).astype(np.float32),
                pre_raw_scores=np.mean(pre_scores, axis=0).astype(np.float32),
                pre_reset_raw_scores=np.mean(pre_reset_scores, axis=0).astype(np.float32),
                seeds=np.asarray(args.seeds), epochs=np.int16(args.epochs),
                provenance_json=json.dumps({
                    "source_subjects": [s for s in range(1, 55) if s != subject],
                    "target_trials_used_for_fit": False,
                    "source_training": "original frozen contiguous LOSO source-pooled protocol",
                    "stateful": "continuous hidden state across original trial sequence",
                    "reset": "GRU hidden reset at every 1-s window; ablation only",
                }),
            )
            print(f"saved I6 S{session} sub-{subject:02d} windows={len(ends)} device={device}", flush=True)


if __name__ == "__main__":
    main()
