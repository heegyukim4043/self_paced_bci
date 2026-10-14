"""Train subject-LOSO CNN-GRU models and score Stieger trial cells.

One three-seed ensemble is trained per held-out subject and reused for every
available session of that subject.  Every source subject contributes the same
maximum number of chronological first-half cells, so subjects with a second
session are not overweighted.  Target first-half preparation is used only for
label-free input normalization and later score/threshold calibration.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_stieger_cnn_gru_compatibility import chronological_split_counts
from scripts.prepare_stieger_cnn_gru_score_cache import (
    DEFAULT_SEEDS,
    cell_window_ends,
    make_windows,
    times_from_feedback,
    window_labels,
)
from scripts.run_lee2019_cnn_gru import CausalCNNGRU


PROTOCOL_VERSION = 1
ACTIVE_FRACTION = 0.75


def parse_shard(path: Path) -> tuple[int, int]:
    session = int(path.parent.name.removeprefix("session"))
    subject = int(path.name.removeprefix("sub-"))
    return subject, session


def load_shard(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.npz"
    signal_path = path / "signals.npy"
    with np.load(metadata_path, allow_pickle=False) as saved:
        result = {key: np.asarray(saved[key]) for key in saved.files if key != "audit_json"}
        result["input_audit"] = json.loads(str(saved["audit_json"]))
    result["signal"] = np.load(signal_path, mmap_mode="r")
    result["path"] = path
    result["subject"], result["session"] = parse_shard(path)
    result["split_index"] = len(result["trial_target_number"]) // 2
    return result


def cell_signal(shard: dict[str, Any], trial: int) -> np.ndarray:
    start, stop = shard["trial_offsets"][trial : trial + 2].astype(int)
    return np.moveaxis(np.asarray(shard["signal"][start:stop]), 0, -1)


def cell_ends(shard: dict[str, Any], trial: int) -> np.ndarray:
    if bool(shard["trial_artifact"][trial]):
        return np.empty(0, dtype=np.int64)
    start, stop = shard["trial_offsets"][trial : trial + 2].astype(int)
    return cell_window_ends(
        float(shard["trial_length_s"][trial]), int(stop - start)
    )


def sample_source_refs(
    shards: list[dict[str, Any]],
    held_out_subject: int,
    max_cells_per_subject: int,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], int]], dict[str, Any]]:
    grouped: dict[int, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for shard in shards:
        if shard["subject"] == held_out_subject:
            continue
        for trial in range(shard["split_index"]):
            if len(cell_ends(shard, trial)):
                grouped[shard["subject"]].append((shard, trial))

    selected: list[tuple[dict[str, Any], int]] = []
    per_subject = []
    for subject in sorted(grouped):
        rng = np.random.default_rng(seed + subject * 1009)
        candidates = grouped[subject]
        active = [ref for ref in candidates if int(ref[0]["trial_target_number"][ref[1]]) != 4]
        rest = [ref for ref in candidates if int(ref[0]["trial_target_number"][ref[1]]) == 4]
        if max_cells_per_subject <= 0:
            chosen = candidates.copy()
        else:
            active_quota = int(round(max_cells_per_subject * ACTIVE_FRACTION))
            rest_quota = max_cells_per_subject - active_quota
            chosen = []
            for values, quota in ((active, active_quota), (rest, rest_quota)):
                if values:
                    take = min(quota, len(values))
                    chosen.extend([values[i] for i in rng.choice(len(values), take, replace=False)])
            remaining = max_cells_per_subject - len(chosen)
            if remaining > 0:
                chosen_ids = {(id(ref[0]), ref[1]) for ref in chosen}
                pool = [ref for ref in candidates if (id(ref[0]), ref[1]) not in chosen_ids]
                if pool:
                    take = min(remaining, len(pool))
                    chosen.extend([pool[i] for i in rng.choice(len(pool), take, replace=False)])
        rng.shuffle(chosen)
        selected.extend(chosen)
        per_subject.append(
            {
                "subject": subject,
                "available": len(candidates),
                "selected": len(chosen),
                "selected_active": int(sum(int(ref[0]["trial_target_number"][ref[1]]) != 4 for ref in chosen)),
                "selected_rest": int(sum(int(ref[0]["trial_target_number"][ref[1]]) == 4 for ref in chosen)),
                "sessions": sorted({int(ref[0]["session"]) for ref in chosen}),
            }
        )
    if not selected:
        raise ValueError("no LOSO source cells selected")
    return selected, {
        "held_out_subject": held_out_subject,
        "source_subjects": len(grouped),
        "selected_cells": len(selected),
        "max_cells_per_source_subject": max_cells_per_subject,
        "active_fraction_target": ACTIVE_FRACTION,
        "per_source_subject": per_subject,
    }


def ref_data(ref: tuple[dict[str, Any], int]) -> tuple[np.ndarray, np.ndarray]:
    shard, trial = ref
    ends = cell_ends(shard, trial)
    times = times_from_feedback(ends)
    x = make_windows(cell_signal(shard, trial), ends)
    y = window_labels(times, int(shard["trial_target_number"][trial]))
    return x, y


def train_source_pool(
    refs: list[tuple[dict[str, Any], int]],
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

    lengths = np.asarray([len(cell_ends(*ref)) for ref in refs], dtype=int)
    positives = 0
    total = 0
    for shard, trial in refs:
        ends = cell_ends(shard, trial)
        labels = window_labels(times_from_feedback(ends), int(shard["trial_target_number"][trial]))
        positives += int(labels.sum())
        total += len(labels)
    negatives = total - positives
    if positives == 0 or negatives == 0:
        raise ValueError("source labels contain only one class")
    pos_weight = torch.tensor(max(negatives / positives, 1.0), device=device)
    ordered = np.argsort(lengths, kind="stable")
    buckets = [ordered[start : start + sequence_batch_size] for start in range(0, len(ordered), sequence_batch_size)]

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for bucket_index in rng.permutation(len(buckets)):
            chosen = np.asarray(buckets[int(bucket_index)]).copy()
            rng.shuffle(chosen)
            common = int(min(lengths[chosen]))
            batch = [ref_data(refs[int(index)]) for index in chosen]
            x = np.stack([values[0][:common] for values in batch])
            y = np.stack([values[1][:common] for values in batch]).astype(np.float32)
            tx = torch.from_numpy(x).to(device)
            ty = torch.from_numpy(y).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tx, hidden=None)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, ty, pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"      epoch {epoch + 1}/{epochs} loss={losses[-1]:.4f}", flush=True)
    return model, {
        "training_sequences": len(refs),
        "training_windows": total,
        "training_positive_windows": positives,
        "training_negative_windows": negatives,
        "positive_class_weight": float(pos_weight.detach().cpu()),
        "epoch_mean_loss": losses,
        "sequence_batch_size": sequence_batch_size,
        "batch_rule": "length buckets; crop independent cells to shortest sequence",
    }


@torch.no_grad()
def predict_cell(model: nn.Module, x: np.ndarray, device: torch.device, chunk: int = 100) -> np.ndarray:
    model.eval()
    hidden = None
    output = []
    for start in range(0, len(x), chunk):
        tx = torch.from_numpy(x[start : start + chunk][None]).to(device)
        logits, hidden = model(tx, hidden=hidden)
        hidden = hidden.detach()
        output.append(logits[0].float().cpu().numpy())
    return np.concatenate(output).astype(np.float32) if output else np.empty(0, np.float32)


def score_shard(model: nn.Module, shard: dict[str, Any], device: torch.device) -> list[np.ndarray]:
    output = []
    for trial in range(len(shard["trial_target_number"])):
        ends = cell_ends(shard, trial)
        output.append(
            predict_cell(model, make_windows(cell_signal(shard, trial), ends), device)
            if len(ends)
            else np.empty(0, np.float32)
        )
    return output


@torch.no_grad()
def causality_reset_audit(model: nn.Module, shard: dict[str, Any], device: torch.device) -> dict[str, Any]:
    candidates = [i for i in range(len(shard["trial_target_number"])) if len(cell_ends(shard, i)) >= 4]
    x1, _ = ref_data((shard, candidates[0]))
    x2, _ = ref_data((shard, candidates[1]))
    prefix = 4
    model.eval()
    full, _ = model(torch.from_numpy(x1[None]).to(device), hidden=None)
    short, _ = model(torch.from_numpy(x1[:prefix][None]).to(device), hidden=None)
    prefix_error = float(torch.max(torch.abs(full[0, :prefix] - short[0])).cpu())
    common = min(len(x1), len(x2))
    stacked = torch.from_numpy(np.stack([x1[:common], x2[:common]])).to(device)
    batched, _ = model(stacked, hidden=None)
    independent1, _ = model(stacked[:1], hidden=None)
    independent2, _ = model(stacked[1:], hidden=None)
    reset_error = float(max(torch.max(torch.abs(batched[0] - independent1[0])).cpu(), torch.max(torch.abs(batched[1] - independent2[0])).cpu()))
    return {
        "prefix_causality_max_abs_error": prefix_error,
        "independent_vs_batched_reset_max_abs_error": reset_error,
        "tolerance": 1e-5,
        "passed": bool(max(prefix_error, reset_error) <= 1e-5),
    }


def flatten_metadata(values: list[np.ndarray], trial_values: np.ndarray, dtype: Any) -> np.ndarray:
    return np.concatenate([np.full(len(row), trial_values[i], dtype=dtype) for i, row in enumerate(values)])


def save_score_cache(
    target: Path,
    shard: dict[str, Any],
    seed_scores_by_trial: list[list[np.ndarray]],
    report: dict[str, Any],
    seeds: tuple[int, ...],
) -> None:
    times_by_trial = [times_from_feedback(cell_ends(shard, trial)) for trial in range(len(shard["trial_target_number"]))]
    labels_by_trial = [window_labels(times, int(shard["trial_target_number"][trial])) for trial, times in enumerate(times_by_trial)]
    seed_scores = np.asarray([np.concatenate(rows) for rows in seed_scores_by_trial], dtype=np.float32)
    scores = seed_scores.mean(axis=0).astype(np.float32)
    offsets = np.r_[0, np.cumsum([len(row) for row in times_by_trial])].astype(np.int32)
    flat_times = np.concatenate(times_by_trial).astype(np.float32)
    trial_index = flatten_metadata(times_by_trial, np.arange(len(times_by_trial)), np.int16)
    split_index = shard["split_index"]
    split = chronological_split_counts(
        shard["trial_target_number"], shard["trial_artifact"], shard["trial_length_s"], split_index=split_index
    )
    screening_cutoff = int(split["screening_cutoff_trial_index"])
    roles = np.zeros(len(times_by_trial), dtype=np.int8)
    roles[split_index : screening_cutoff + 1] = 1
    roles[screening_cutoff + 1 :] = 2
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        scores=scores,
        seed_scores=seed_scores,
        time_from_feedback_s=flat_times,
        window_label_active_feedback=np.concatenate(labels_by_trial).astype(np.int8),
        feedback_mask=(flat_times > 0).astype(np.int8),
        preparation_mask=(flat_times <= 0).astype(np.int8),
        trial_index=trial_index,
        target_number=flatten_metadata(times_by_trial, shard["trial_target_number"], np.int8),
        run_number=flatten_metadata(times_by_trial, shard["trial_run_number"], np.int16),
        task_number=flatten_metadata(times_by_trial, shard["trial_task_number"], np.int8),
        split_role=(trial_index >= split_index).astype(np.int8),
        trial_offsets=offsets,
        trial_target_number=shard["trial_target_number"],
        trial_artifact=shard["trial_artifact"],
        trial_length_s=shard["trial_length_s"],
        trial_run_number=shard["trial_run_number"],
        trial_task_number=shard["trial_task_number"],
        trial_result=shard["trial_result"],
        trial_forced_result=shard["trial_forced_result"],
        trial_protocol_role=roles,
        normalization_mean=shard["normalization_mean"],
        normalization_std=shard["normalization_std"],
        channels=shard["channels"],
        bands_hz=shard["bands_hz"],
        seeds=np.asarray(seeds, dtype=np.int64),
        audit_json=np.asarray(json.dumps(report, sort_keys=True)),
    )


def valid_existing(path: Path, seeds: tuple[int, ...], epochs: int, max_cells: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            audit = json.loads(str(cache["audit_json"]))
            if audit.get("protocol_version") != PROTOCOL_VERSION:
                return None
            if tuple(cache["seeds"].astype(int)) != seeds:
                return None
            if audit["training"]["epochs"] != epochs or audit["source_sampling"]["max_cells_per_source_subject"] != max_cells:
                return None
            return audit
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("data/interim/stieger2021_cnn_gru_loso_inputs"))
    parser.add_argument("--output-root", type=Path, default=Path("data/interim/stieger2021_cnn_gru_loso_scores"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/stieger2021_cnn_gru_loso_score_cache_manifest.json"))
    parser.add_argument("--subjects", type=int, nargs="*")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-source-cells-per-subject", type=int, default=64)
    parser.add_argument("--sequence-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seeds = tuple(args.seeds)
    shard_paths = sorted(path.parent for path in args.input_root.glob("session*/sub-*/metadata.npz"))
    if len(shard_paths) != 30:
        raise ValueError(f"expected 30 input shards, found {len(shard_paths)}")
    shards = [load_shard(path) for path in shard_paths]
    by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for shard in shards:
        by_subject[shard["subject"]].append(shard)
    subjects = sorted(by_subject)
    if args.subjects:
        requested = set(args.subjects)
        subjects = [subject for subject in subjects if subject in requested]

    rows = []
    if args.manifest.exists() and not args.overwrite:
        try:
            rows = json.loads(args.manifest.read_text(encoding="utf-8")).get("rows", [])
        except Exception:
            rows = []
    for fold_index, subject in enumerate(subjects, start=1):
        targets = sorted(by_subject[subject], key=lambda row: row["session"])
        expected = [args.output_root / f"session{row['session']}" / f"sub-{subject:02d}.npz" for row in targets]
        existing = [None if args.overwrite else valid_existing(path, seeds, args.epochs, args.max_source_cells_per_subject) for path in expected]
        if all(row is not None for row in existing):
            print(f"[{fold_index}/{len(subjects)}] target S{subject}: valid caches skip", flush=True)
            continue
        started = time.time()
        refs, sampling = sample_source_refs(
            shards, subject, args.max_source_cells_per_subject, 20260921
        )
        print(f"[{fold_index}/{len(subjects)}] target S{subject}: {len(refs)} source cells from {sampling['source_subjects']} subjects", flush=True)
        session_seed_scores: dict[int, list[list[np.ndarray]]] = {row["session"]: [] for row in targets}
        training_audits = []
        model_audits = []
        for seed_index, seed in enumerate(seeds, start=1):
            print(f"    seed {seed_index}/{len(seeds)} ({seed})", flush=True)
            model, training = train_source_pool(
                refs, seed=seed, epochs=args.epochs, device=device,
                sequence_batch_size=args.sequence_batch_size,
            )
            audit = causality_reset_audit(model, targets[0], device)
            if not audit["passed"]:
                raise AssertionError(f"causality/reset failed: {audit}")
            for target_shard in targets:
                session_seed_scores[target_shard["session"]].append(score_shard(model, target_shard, device))
            training_audits.append({"seed": seed, **training})
            model_audits.append({"seed": seed, **audit})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = float(time.time() - started)
        for target_shard, target_path in zip(targets, expected):
            report = {
                "protocol_version": PROTOCOL_VERSION,
                "subject": subject,
                "session": target_shard["session"],
                "target": str(target_path.resolve()),
                "model": "CausalCNNGRU(20 channels, hidden=32)",
                "parameter_count": int(sum(p.numel() for p in CausalCNNGRU(20, 32).parameters())),
                "training_regime": "subject-LOSO source-pooled; target subject all sessions excluded",
                "target_normalization": "target first-half preparation only; label-free",
                "target_threshold_calibration": "target first-half intentional rest; independent of model fitting",
                "source_sampling": sampling,
                "training": {"epochs": args.epochs, "seeds": list(seeds), "seed_audits": training_audits},
                "causality_reset_audits": model_audits,
                "seam_contract": {"hidden_reset_each_cell": True, "window_crosses_cell_seam": False},
                "elapsed_fold_seconds": elapsed,
                "device": str(device),
            }
            save_score_cache(target_path, target_shard, session_seed_scores[target_shard["session"]], report, seeds)
            rows = [row for row in rows if not (row.get("subject") == subject and row.get("session") == target_shard["session"])]
            rows.append(report)
        rows.sort(key=lambda row: (row["session"], row["subject"]))
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({
            "protocol": {
                "model": "CausalCNNGRU subject-LOSO source pool",
                "target_exclusion": "all sessions",
                "target_normalization": "first-half preparation, label-free",
                "source_sampling": f"balanced max {args.max_source_cells_per_subject} cells per source subject",
                "epochs": args.epochs,
                "seeds": list(seeds),
            },
            "completed_subject_sessions": len(rows),
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"    fold complete elapsed={elapsed:.1f}s", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
