"""Full-backbone CBraMod LOSO fine-tuning on original Lee source trials.

This is a separate, costly arm from the frozen linear probe. For every target
subject/session, published weights are restarted and trained for a fixed
number of epochs on trials 0..49 of the other 53 subjects. There is no
target-label fit, target-driven epoch choice, or simulated source replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.cbramod_transfer import (
    CBraModBinaryClassifier, FINETUNE_PROTOCOL, FINETUNED_MODEL_NAME,
    PROTOCOL_VERSION, SFREQ, decision_ends,
    load_frozen_backbone, predict_ends, preprocess_run, regular_ends,
    sha256_file, upstream_commit,
)
from uncued_mi.data import lee2019_path, load_lee2019_run


MODEL_NAME = FINETUNED_MODEL_NAME


def load_training_windows(path: Path, subject: int, session: int,
                          checkpoint_sha: str, commit: str, input_unit: str):
    with np.load(path, allow_pickle=False) as data:
        if ((int(data["subject"]), int(data["session"])) != (subject, session)
                or str(data["protocol_version"]) != PROTOCOL_VERSION
                or str(data["checkpoint_sha256"]) != checkpoint_sha
                or str(data["upstream_commit"]) != commit
                or str(data["input_unit"]) != input_unit):
            raise AssertionError(f"source cache identity/protocol mismatch: {path}")
        windows = np.asarray(data["train_windows"], dtype=np.float32)
        labels = np.asarray(data["train_labels"], dtype=np.float32)
        trials = np.asarray(data["train_trial_ids"], dtype=np.int64)
    if (windows.shape != (256, 19, 2, 200) or labels.shape != (256,)
            or int(labels.sum()) != 128 or np.any(trials < 0) or np.any(trials >= 50)
            or not np.all(np.isfinite(windows))):
        raise AssertionError(f"source-only training windows invalid: {path}")
    return windows, labels


def train_one(model: CBraModBinaryClassifier, windows: np.ndarray,
              labels: np.ndarray, *, seed: int, epochs: int, batch_size: int,
              device: torch.device) -> list[float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.backbone.requires_grad_(True)
    model.train()
    dataset = TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, generator=generator, drop_last=False)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for x_cpu, y_cpu in loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(y)
            count += len(y)
        losses.append(total / count)
        print(f"epoch {epoch+1}/{epochs} source BCE={losses[-1]:.4f}", flush=True)
    return losses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 55)))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--upstream-repo", type=Path, default=Path("third_party/CBraMod"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("third_party/CBraMod/pretrained_weights/pretrained_weights.pth"))
    ap.add_argument("--input-unit", choices=("uV", "V"), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--inference-batch-size", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    ap.add_argument("--tier0-root", type=Path, default=Path("data/interim/lee2019_tier0"))
    ap.add_argument("--feature-root", type=Path,
                    default=Path("data/interim/lee2019_cbramod_frozen_features"))
    ap.add_argument("--model-root", type=Path,
                    default=Path("data/models/lee2019_cbramod_finetuned_loso"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/interim/lee2019_cbramod_finetuned_loso_scores"))
    args = ap.parse_args()
    if (not args.subjects or any(s < 1 or s > 54 for s in args.subjects)
            or not args.sessions or any(s not in (1, 2) for s in args.sessions)
            or args.epochs != 3 or min(args.batch_size, args.inference_batch_size,
                                        args.torch_threads) < 1):
        raise ValueError("invalid subject/session or fixed fine-tune protocol")
    if len({args.feature_root.resolve(), args.model_root.resolve(),
            args.output_root.resolve()}) != 3:
        raise ValueError("feature, model and score directories must differ")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(args.torch_threads)
    checkpoint_sha = sha256_file(args.checkpoint)
    commit = upstream_commit(args.upstream_repo)

    for session in args.sessions:
        source = {}
        for subject in range(1, 55):
            path = args.feature_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"complete source-window cache required: {path}")
            source[subject] = load_training_windows(
                path, subject, session, checkpoint_sha, commit, args.input_unit)
        for target in args.subjects:
            score_path = args.output_root / f"session{session}" / f"sub-{target:02d}.npz"
            model_path = args.model_root / f"session{session}" / f"sub-{target:02d}.pt"
            if score_path.exists():
                if not model_path.exists():
                    raise AssertionError(f"score cache without trained model: {score_path}")
                with np.load(score_path, allow_pickle=False) as old:
                    if (str(old["model_name"]) != MODEL_NAME
                            or str(old["protocol_version"]) != FINETUNE_PROTOCOL
                            or str(old["checkpoint_sha256"]) != checkpoint_sha):
                        raise AssertionError(f"existing score protocol mismatch: {score_path}")
                    previous = json.loads(str(old["provenance_json"]))
                    if (previous["epochs"] != args.epochs
                            or previous["train_batch_size"] != args.batch_size
                            or previous["input_unit"] != args.input_unit
                            or previous["upstream_commit"] != commit):
                        raise AssertionError(f"existing score hyperparameter mismatch: {score_path}")
                print(f"cache hit S{session} sub-{target:02d}", flush=True)
                continue
            seed = 20261014 + session * 100 + target
            torch.manual_seed(seed)
            donors = [subject for subject in range(1, 55) if subject != target]
            if len(donors) != 53 or target in donors:
                raise AssertionError("LOSO donor split failed")
            windows = np.concatenate([source[s][0] for s in donors], axis=0)
            labels = np.concatenate([source[s][1] for s in donors], axis=0)
            backbone = load_frozen_backbone(args.upstream_repo, args.checkpoint, device)
            model = CBraModBinaryClassifier(backbone).to(device)
            metadata = {
                "model_name": MODEL_NAME, "protocol_version": FINETUNE_PROTOCOL,
                "feature_protocol": PROTOCOL_VERSION,
                "checkpoint_sha256": checkpoint_sha, "upstream_commit": commit,
                "input_unit": args.input_unit,
                "source_subjects": donors, "target_trials_used_for_fit": False,
                "source_trials": "0..49 original continuous; 128 MI + 128 idle per donor",
                "epochs": args.epochs, "seed": seed, "train_batch_size": args.batch_size,
                "optimizer": "AdamW backbone lr=1e-4, head lr=1e-3, wd=0.01; grad clip=1",
                "selection": "fixed 3 epochs; no target or future validation",
                "interpretation": "exploratory Lee cue-paced; no uncued MI specificity claim",
            }
            if model_path.exists():
                saved = torch.load(model_path, map_location="cpu", weights_only=True)
                if saved["metadata"] != metadata:
                    raise AssertionError(f"partial model cache protocol mismatch: {model_path}")
                model.load_state_dict(saved["state_dict"], strict=True)
                losses = saved["source_bce_by_epoch"]
                print(f"resume trained S{session} sub-{target:02d}", flush=True)
            else:
                losses = train_one(model, windows, labels, seed=seed,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   device=device)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"metadata": metadata,
                            "source_bce_by_epoch": losses,
                            "state_dict": {key: value.detach().cpu()
                                           for key, value in model.state_dict().items()}},
                           model_path)
            del windows, labels
            tier_path = args.tier0_root / f"session{session}" / f"sub-{target:02d}_tier0_features.npz"
            with np.load(tier_path, allow_pickle=False) as tier:
                times = np.asarray(tier["continuous_times"], dtype=float)
                tier_cues = np.asarray(tier["cue_samples"], dtype=float) / float(tier["native_sfreq"])
            run = load_lee2019_run(lee2019_path(args.data_root, target, session), "train")
            cues = np.asarray(run.cue_samples / run.sfreq, dtype=float)
            if not np.allclose(cues, tier_cues, atol=0.005, rtol=0):
                raise AssertionError("target cue/tier-0 mismatch")
            signal, anchor = preprocess_run(run, input_unit=args.input_unit,
                                            stop_seconds=float(times[-1]))
            ends = decision_ends(times, signal.shape[-1])
            pre_ends = regular_ends(anchor.shape[-1])
            raw = predict_ends(model, signal, ends, device,
                               batch_size=args.inference_batch_size)
            pre_raw = predict_ends(model, anchor, pre_ends, device,
                                   batch_size=args.inference_batch_size)
            score_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                score_path, subject=np.int16(target), session=np.int8(session),
                model_name=np.asarray(MODEL_NAME),
                protocol_version=np.asarray(FINETUNE_PROTOCOL),
                times=ends / SFREQ, pre_times=pre_ends / SFREQ,
                cue_times=cues, raw_scores=raw, pre_raw_scores=pre_raw,
                checkpoint_sha256=np.asarray(checkpoint_sha),
                upstream_commit=np.asarray(commit),
                source_bce_by_epoch=np.asarray(losses, dtype=np.float32),
                provenance_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            print(f"saved fine-tuned S{session} sub-{target:02d}: "
                  f"donors=53 decisions={len(raw)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
