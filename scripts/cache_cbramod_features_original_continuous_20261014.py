"""Cache published CBraMod features on original, unsliced Lee2019 recordings.

This stage performs no target-specific fitting. The same frozen backbone is
applied to all 54 subjects in each session. Source training examples are
strictly from trials 0..49; continuous scores are extracted separately at
the existing tier-0 decision times. Outputs are features, not EEG results.
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

from uncued_mi.cbramod_transfer import (
    MODEL_NAME, PRETRAIN_CHANNELS, PROTOCOL_VERSION, SFREQ, decision_ends, encode_ends,
    load_frozen_backbone, preprocess_run, regular_ends, sha256_file,
    training_ends, trailing_windows, upstream_commit,
)
from uncued_mi.data import lee2019_path, load_lee2019_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 55)))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--upstream-repo", type=Path, default=Path("third_party/CBraMod"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("third_party/CBraMod/pretrained_weights/pretrained_weights.pth"))
    ap.add_argument("--input-unit", choices=("uV", "V"), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/interim/lee2019_cbramod_frozen_features"))
    args = ap.parse_args()
    if (not args.subjects or any(s < 1 or s > 54 for s in args.subjects)
            or not args.sessions or any(s not in (1, 2) for s in args.sessions)
            or args.batch_size < 1 or args.torch_threads < 1):
        raise ValueError("invalid subjects, sessions, batch size or thread count")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(args.torch_threads)
    checkpoint_sha = sha256_file(args.checkpoint)
    commit = upstream_commit(args.upstream_repo)
    model = load_frozen_backbone(args.upstream_repo, args.checkpoint, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{MODEL_NAME}: {parameters} frozen backbone parameters", flush=True)

    for session in args.sessions:
        for subject in args.subjects:
            output = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if output.exists():
                with np.load(output, allow_pickle=False) as old:
                    if (str(old["checkpoint_sha256"]) != checkpoint_sha
                            or str(old["upstream_commit"]) != commit
                            or str(old["input_unit"]) != args.input_unit
                            or str(old["protocol_version"]) != PROTOCOL_VERSION):
                        raise AssertionError(f"existing cache uses a different protocol: {output}")
                print(f"cache hit S{session} sub-{subject:02d}", flush=True)
                continue
            tier_path = args.tier0_root / f"session{session}" / f"sub-{subject:02d}_tier0_features.npz"
            if not tier_path.is_file():
                raise FileNotFoundError(f"prepare tier-0 decision grid first: {tier_path}")
            with np.load(tier_path, allow_pickle=False) as tier:
                times = np.asarray(tier["continuous_times"], dtype=float)
                tier_cues = np.asarray(tier["cue_samples"], dtype=float) / float(tier["native_sfreq"])
            run = load_lee2019_run(lee2019_path(args.data_root, subject, session), "train")
            if len(run.cue_samples) != 100:
                raise AssertionError("expected 100 chronological Lee training trials")
            cue_times = np.asarray(run.cue_samples / run.sfreq, dtype=float)
            if not np.allclose(cue_times, tier_cues, atol=0.005, rtol=0):
                raise AssertionError("tier-0 cue schedule and original recording differ")
            signal, anchor = preprocess_run(run, input_unit=args.input_unit,
                                            stop_seconds=float(times[-1]))
            signal_q99 = float(np.percentile(np.abs(signal), 99))
            anchor_q99 = float(np.percentile(np.abs(anchor), 99))
            signal_gt100 = float(np.mean(np.abs(signal) > 100.0))
            ends = decision_ends(times, signal.shape[-1])
            pre_ends = regular_ends(anchor.shape[-1])
            source_ends, source_labels, source_trials = training_ends(
                cue_times, signal.shape[-1])
            source_features = encode_ends(model, signal, source_ends, device,
                                          batch_size=args.batch_size)
            source_windows = trailing_windows(signal, source_ends).astype(np.float16)
            pre_features = encode_ends(model, anchor, pre_ends, device,
                                       batch_size=args.batch_size)
            continuous_features = encode_ends(model, signal, ends, device,
                                               batch_size=args.batch_size)
            metadata = {
                "model": MODEL_NAME,
                "source_trial_ids": "0..49 only",
                "target_fit": False,
                "raw_eeg_unit": args.input_unit,
                "preprocessing": "samplewise CAR; causal 0.3-40 Hz IIR; integer decimation to 200 Hz",
                "backbone": "published checkpoint, strictly loaded and frozen",
                "feature": "global mean + C3/Cz/C4 final-patch embeddings after 2-s past-only input",
                "channel_order": list(PRETRAIN_CHANNELS),
                "limitations": "Lee is cue-paced; causal filtering differs from upstream zero-phase pretraining",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output, subject=np.int16(subject), session=np.int8(session),
                times=ends / SFREQ, pre_times=pre_ends / SFREQ,
                cue_times=cue_times, train_ends=source_ends,
                train_trial_ids=source_trials, train_labels=source_labels,
                train_features=source_features,
                train_windows=source_windows,
                pre_features=pre_features,
                continuous_features=continuous_features,
                signal_abs_q99_uV=np.float32(signal_q99),
                anchor_abs_q99_uV=np.float32(anchor_q99),
                signal_fraction_abs_gt100_uV=np.float32(signal_gt100),
                input_unit=np.asarray(args.input_unit),
                protocol_version=np.asarray(PROTOCOL_VERSION),
                checkpoint_sha256=np.asarray(checkpoint_sha),
                upstream_commit=np.asarray(commit),
                backbone_parameters=np.int64(parameters),
                protocol_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            print(f"saved S{session} sub-{subject:02d}: "
                  f"train={len(source_ends)} pre={len(pre_ends)} "
                  f"decisions={len(ends)} Q99={signal_q99:.1f} uV", flush=True)


if __name__ == "__main__":
    main()
