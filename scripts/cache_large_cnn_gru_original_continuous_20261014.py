"""Exploratory Lee LOSO score cache for one larger causal CNN-GRU.

This intentionally reuses I6's source sequences, optimizer, seed/epoch budget,
pre-rest normalization, and original-continuous target inference.  Both CNN/GRU
capacity and within-window normalization differ from I6; this is not an
isolated parameter-count intervention.  No target subject enters fitting.  The
score cache is NOT a performance result or a self-paced MI validation.
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

from uncued_mi.large_causal_cnn_gru import LargeCausalCNNGRU
from scripts.cache_i6_cnn_gru_original_continuous_20261012 import (
    predict_window_reset, target_input,
)
from scripts.run_lee2019_cnn_gru_loso_arm_a_stream_v2 import (
    SFREQ, WINDOW, _cached_source_training, _window_ends,
    predict_signal, train_source_pool,
)


def validate_args(args: argparse.Namespace) -> None:
    if not args.subjects or any(s < 1 or s > 54 for s in args.subjects):
        raise ValueError("subjects must be within 1..54")
    if not args.sessions or any(s not in (1, 2) for s in args.sessions):
        raise ValueError("sessions must be 1 or 2")
    if not args.seeds or args.epochs < 1 or args.max_source_windows < 1:
        raise ValueError("seeds, epochs and max-source-windows must be positive")
    if args.output_root == args.baseline_root:
        raise ValueError("large-model output must not overwrite the I6 baseline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 55)))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[20260729, 20260730, 20260731])
    ap.add_argument("--max-source-windows", type=int, default=100)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    ap.add_argument("--train-cache-root", type=Path,
                    default=Path("data/interim/lee2019_mil_raw"))
    ap.add_argument("--prepared-source-root", type=Path,
                    default=Path("data/interim/lee2019_cnn_gru_loso_prepared_source"))
    ap.add_argument("--tier0-root", type=Path,
                    default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--baseline-root", type=Path,
                    default=Path("data/interim/lee2019_i6_cnn_gru_original"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/interim/lee2019_large_cnn_gru_original"))
    args = ap.parse_args()
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    parameters = sum(p.numel() for p in LargeCausalCNNGRU().parameters())
    print(f"LargeCausalCNNGRU parameters={parameters}", flush=True)

    for session in args.sessions:
        # Source construction is model-independent.  The target is excluded
        # below *before* passing blocks into the training loop.
        source_pool = {
            s: _cached_source_training(
                args.data_root, args.train_cache_root, args.prepared_source_root,
                s, session, args.max_source_windows,
            ) for s in range(1, 55)
        }
        for subject in args.subjects:
            out = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if out.exists():
                with np.load(out, allow_pickle=False) as old:
                    if (str(old["model_name"]) != "LargeCausalCNNGRU"
                            or list(old["seeds"]) != args.seeds
                            or int(old["epochs"]) != args.epochs
                            or int(old["max_source_windows"]) != args.max_source_windows):
                        raise AssertionError(f"existing cache protocol mismatch: {out}")
                print(f"cache hit S{session} sub-{subject:02d}", flush=True)
                continue
            tier = args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
            if not tier.exists():
                raise FileNotFoundError(f"generate tier-0 cache first: {tier}")
            with np.load(tier, allow_pickle=False) as t:
                tier_times = np.asarray(t["continuous_times"])
                tier_cues = np.asarray(t["cue_samples"] / float(t["native_sfreq"]))
            signal, anchor, cues = target_input(
                args.data_root, subject, session, float(tier_times[-1])
            )
            ends = np.rint(tier_times * SFREQ).astype(np.int64)
            pre_ends = _window_ends(anchor.shape[-1], first_end=3 * SFREQ)
            times = ends / SFREQ
            if (not np.allclose(times, tier_times, atol=.001, rtol=0)
                    or ends[0] < WINDOW or ends[-1] > signal.shape[-1]):
                raise AssertionError("target time grid or window support mismatch")
            if not np.allclose(cues, tier_cues, atol=.005, rtol=0):
                raise AssertionError("cue time mismatch")
            donors = [s for s in range(1, 55) if s != subject]
            if len(donors) != 53 or subject in donors:
                raise AssertionError("LOSO donor isolation failed")
            blocks = [source_pool[s] for s in donors]
            scores, reset_scores, pre_scores, pre_reset_scores = [], [], [], []
            for seed in args.seeds:
                model = train_source_pool(
                    blocks, seed, args.epochs, device,
                    model_class=LargeCausalCNNGRU, hidden=128,
                )
                scores.append(predict_signal(model, signal, device, ends=ends))
                reset_scores.append(predict_window_reset(model, signal, device, ends))
                pre_scores.append(predict_signal(model, anchor, device, ends=pre_ends))
                pre_reset_scores.append(predict_window_reset(model, anchor, device, pre_ends))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out, subject=np.int16(subject), session=np.int8(session),
                times=times, pre_times=pre_ends / SFREQ, cue_times=cues,
                raw_scores=np.mean(scores, axis=0).astype(np.float32),
                reset_raw_scores=np.mean(reset_scores, axis=0).astype(np.float32),
                pre_raw_scores=np.mean(pre_scores, axis=0).astype(np.float32),
                pre_reset_raw_scores=np.mean(pre_reset_scores, axis=0).astype(np.float32),
                seeds=np.asarray(args.seeds), epochs=np.int16(args.epochs),
                max_source_windows=np.int16(args.max_source_windows),
                model_name=np.asarray("LargeCausalCNNGRU"),
                parameter_count=np.int32(parameters),
                provenance_json=np.asarray(json.dumps({
                    "source_subjects": donors,
                    "target_trials_used_for_fit": False,
                    "source_training": "I6 frozen source-pooled simulated-sequence protocol",
                    "target_inference": "original continuous, chronological stateful GRU",
                    "reset_scores": "window-reset ablation only",
                    "interpretation": "exploratory capacity comparison; no MI specificity claim",
                }, sort_keys=True)),
            )
            print(f"saved S{session} sub-{subject:02d} windows={len(ends)}", flush=True)


if __name__ == "__main__":
    main()
