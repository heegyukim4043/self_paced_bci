"""Extract frozen LaBraM embeddings from original, unspliced Lee recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.data import lee2019_path, load_lee2019_run
from uncued_mi.labram_transfer import (
    MODEL_NAME, PRETRAIN_CHANNELS, PROTOCOL_VERSION, SFREQ, decision_ends,
    encode_ends, load_frozen_backbone, preprocess_run, regular_ends,
    sha256_file, training_ends, upstream_commit,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 55)))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--upstream-repo", type=Path, default=Path("third_party/LaBraM"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("third_party/LaBraM/checkpoints/labram-base.pth"))
    ap.add_argument("--input-unit", choices=("uV", "V"), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/interim/lee2019_labram_frozen_features"))
    args = ap.parse_args()
    if (not args.subjects or any(s < 1 or s > 54 for s in args.subjects)
            or not args.sessions or any(s not in (1, 2) for s in args.sessions)
            or args.batch_size < 1 or args.torch_threads < 1):
        raise ValueError("invalid case selection or compute configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(args.torch_threads)
    checkpoint_sha = sha256_file(args.checkpoint)
    commit = upstream_commit(args.upstream_repo)
    model = load_frozen_backbone(args.upstream_repo, args.checkpoint, device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{MODEL_NAME}: {parameters} frozen parameters", flush=True)

    for session in args.sessions:
        for subject in args.subjects:
            output = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
            tier_path = (args.tier0_root / f"session{session}"
                         / f"sub-{subject:02d}_tier0_features.npz")
            if not tier_path.is_file():
                raise FileNotFoundError(f"prepare tier-0 decision grid first: {tier_path}")
            with np.load(tier_path, allow_pickle=False) as tier:
                times = np.asarray(tier["continuous_times"], dtype=float)
                tier_cues = (np.asarray(tier["cue_samples"], dtype=float)
                             / float(tier["native_sfreq"]))
            if output.exists():
                with np.load(output, allow_pickle=False) as old:
                    if (str(old["checkpoint_sha256"]) != checkpoint_sha
                            or str(old["upstream_commit"]) != commit
                            or str(old["input_unit"]) != args.input_unit
                            or str(old["protocol_version"]) != PROTOCOL_VERSION
                            or not np.allclose(old["times"], times, atol=0.00251, rtol=0)):
                        raise AssertionError(f"existing cache uses a different protocol: {output}")
                print(f"cache hit S{session} sub-{subject:02d}", flush=True)
                continue
            run = load_lee2019_run(lee2019_path(args.data_root, subject, session), "train")
            if len(run.cue_samples) != 100:
                raise AssertionError("expected 100 chronological Lee trials")
            cues = np.asarray(run.cue_samples / run.sfreq, dtype=float)
            if not np.allclose(cues, tier_cues, atol=0.005, rtol=0):
                raise AssertionError("tier-0 cue schedule differs from original recording")
            signal, anchor = preprocess_run(run, input_unit=args.input_unit,
                                            stop_seconds=float(times[-1]))
            ends = decision_ends(times, signal.shape[-1])
            pre_ends = regular_ends(anchor.shape[-1])
            train_ends, labels, trials = training_ends(cues, signal.shape[-1])
            train_features = encode_ends(model, signal, train_ends, device,
                                         batch_size=args.batch_size)
            pre_features = encode_ends(model, anchor, pre_ends, device,
                                       batch_size=args.batch_size)
            continuous_features = encode_ends(model, signal, ends, device,
                                              batch_size=args.batch_size)
            metadata = {
                "model": MODEL_NAME, "target_fit": False,
                "source_trial_ids": "0..49 only", "raw_eeg_unit": args.input_unit,
                "preprocessing": "causal 0.1-75 Hz IIR, causal 50 Hz notch, integer decimation to 200 Hz, no CAR",
                "feature": "2-s past-only LaBraM global+C3/Cz/C4 last-patch embeddings",
                "channel_order": list(PRETRAIN_CHANNELS),
                "limitations": "Lee is cued; causal IIR differs from official offline preprocessing",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output, subject=np.int16(subject), session=np.int8(session),
                times=ends / SFREQ, pre_times=pre_ends / SFREQ, cue_times=cues,
                train_ends=train_ends, train_trial_ids=trials, train_labels=labels,
                train_features=train_features, pre_features=pre_features,
                continuous_features=continuous_features,
                signal_abs_q99_uV=np.float32(np.percentile(np.abs(signal), 99)),
                anchor_abs_q99_uV=np.float32(np.percentile(np.abs(anchor), 99)),
                input_unit=np.asarray(args.input_unit),
                protocol_version=np.asarray(PROTOCOL_VERSION),
                checkpoint_sha256=np.asarray(checkpoint_sha),
                upstream_commit=np.asarray(commit),
                backbone_parameters=np.int64(parameters),
                protocol_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            print(f"saved S{session} sub-{subject:02d}: train={len(train_ends)} "
                  f"pre={len(pre_ends)} decisions={len(ends)}", flush=True)


if __name__ == "__main__":
    main()
