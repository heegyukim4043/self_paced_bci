"""Fit source-only LOSO linear probes on frozen LaBraM feature caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.fit_cbramod_loso_original_continuous_20261014 import fit_loso_probe
from uncued_mi.labram_transfer import FEATURE_DIM, MODEL_NAME, PROTOCOL_VERSION


def load_source_feature(path: Path, subject: int, session: int):
    with np.load(path, allow_pickle=False) as data:
        if (int(data["subject"]), int(data["session"])) != (subject, session):
            raise AssertionError(f"feature cache identity mismatch: {path}")
        features = np.asarray(data["train_features"], np.float32)
        labels = np.asarray(data["train_labels"], np.int8)
        trials = np.asarray(data["train_trial_ids"], np.int8)
        signature = (str(data["checkpoint_sha256"]), str(data["upstream_commit"]),
                     str(data["input_unit"]), str(data["protocol_version"]))
    if (features.shape != (256, FEATURE_DIM) or labels.shape != (256,)
            or np.any(trials < 0) or np.any(trials >= 50)
            or int(np.sum(labels)) != 128 or not np.all(np.isfinite(features))
            or signature[-1] != PROTOCOL_VERSION):
        raise AssertionError(f"invalid source-only LaBraM feature cache: {path}")
    return features, labels, signature


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 55)))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--feature-root", type=Path,
                    default=Path("data/interim/lee2019_labram_frozen_features"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/interim/lee2019_labram_frozen_loso_scores"))
    args = ap.parse_args()
    if (not args.subjects or any(s < 1 or s > 54 for s in args.subjects)
            or not args.sessions or any(s not in (1, 2) for s in args.sessions)
            or args.feature_root.resolve() == args.output_root.resolve()):
        raise ValueError("invalid subjects, sessions, or output root")
    for session in args.sessions:
        source, signatures = {}, set()
        for subject in range(1, 55):
            path = args.feature_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"complete 54-subject feature cache required: {path}")
            features, labels, signature = load_source_feature(path, subject, session)
            source[subject] = (features, labels)
            signatures.add(signature)
        if len(signatures) != 1:
            raise AssertionError("mixed LaBraM checkpoint, upstream revision, unit or protocol")
        checkpoint_sha, upstream_sha, input_unit, _ = next(iter(signatures))
        for target in args.subjects:
            output = args.output_root / f"session{session}" / f"sub-{target:02d}.npz"
            if output.exists():
                with np.load(output, allow_pickle=False) as old:
                    if (str(old["model_name"]) != MODEL_NAME
                            or str(old["checkpoint_sha256"]) != checkpoint_sha
                            or str(old["protocol_version"]) != PROTOCOL_VERSION):
                        raise AssertionError(f"existing score protocol mismatch: {output}")
                print(f"cache hit S{session} sub-{target:02d}", flush=True)
                continue
            model, donors, n_train = fit_loso_probe(source, target)
            path = args.feature_root / f"session{session}" / f"sub-{target:02d}.npz"
            with np.load(path, allow_pickle=False) as data:
                continuous = np.asarray(data["continuous_features"], np.float32)
                pre = np.asarray(data["pre_features"], np.float32)
                times = np.asarray(data["times"], dtype=float)
                pre_times = np.asarray(data["pre_times"], dtype=float)
                cues = np.asarray(data["cue_times"], dtype=float)
                parameters = int(data["backbone_parameters"])
            if (continuous.ndim != 2 or continuous.shape[1] != FEATURE_DIM
                    or pre.ndim != 2 or pre.shape[1] != FEATURE_DIM
                    or len(continuous) != len(times) or len(pre) != len(pre_times)
                    or len(cues) != 100):
                raise AssertionError(f"invalid target feature geometry: {path}")
            raw = model.decision_function(continuous).astype(np.float32)
            pre_raw = model.decision_function(pre).astype(np.float32)
            if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(pre_raw)):
                raise AssertionError("non-finite LaBraM probe scores")
            provenance = {
                "source_subjects": donors, "source_trial_ids": "0..49",
                "target_trials_used_for_fit": False,
                "source_training_windows": n_train,
                "input_unit": input_unit,
                "classifier": "source-only StandardScaler + balanced logistic regression C=1",
                "feature": "frozen LaBraM 2-s trailing-window global+C3/Cz/C4 embeddings",
                "interpretation": "exploratory cue-paced Lee transfer; no uncued or MI-specific claim",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output, subject=np.int16(target), session=np.int8(session),
                model_name=np.asarray(MODEL_NAME), times=times, pre_times=pre_times,
                cue_times=cues, raw_scores=raw, pre_raw_scores=pre_raw,
                protocol_version=np.asarray(PROTOCOL_VERSION),
                checkpoint_sha256=np.asarray(checkpoint_sha),
                upstream_commit=np.asarray(upstream_sha),
                backbone_parameters=np.int64(parameters),
                provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
            )
            print(f"saved S{session} sub-{target:02d}: donors=53 train={n_train} "
                  f"decisions={len(raw)}", flush=True)


if __name__ == "__main__":
    main()
