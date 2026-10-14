"""Fit a source-only linear probe and score Lee continuous EEG (54 x 2).

This is a frozen-backbone transfer experiment, not full CBraMod fine-tuning.
All source windows belong to trials 0..49 of the other 53 people; the target
person's labels and feature statistics are never used in the fitted probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from uncued_mi.cbramod_transfer import FEATURE_DIM, MODEL_NAME, PROTOCOL_VERSION


def fit_loso_probe(source: dict[int, tuple[np.ndarray, np.ndarray]],
                   target: int):
    donors = [subject for subject in sorted(source) if subject != target]
    if len(donors) != 53 or target in donors:
        raise AssertionError("LOSO requires exactly 53 non-target donors")
    features = np.concatenate([source[subject][0] for subject in donors])
    labels = np.concatenate([source[subject][1] for subject in donors])
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"CBraMod feature dimension must be {FEATURE_DIM}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("source examples must include both classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                           max_iter=1000, random_state=20261014),
    )
    model.fit(features, labels)
    return model, donors, int(len(labels))


def load_source_feature(path: Path, subject: int, session: int):
    with np.load(path, allow_pickle=False) as data:
        if (int(data["subject"]), int(data["session"])) != (subject, session):
            raise AssertionError(f"feature cache identity mismatch: {path}")
        features = np.asarray(data["train_features"], np.float32)
        labels = np.asarray(data["train_labels"], np.int8)
        trial_ids = np.asarray(data["train_trial_ids"], np.int8)
        sha = str(data["checkpoint_sha256"])
        commit = str(data["upstream_commit"])
        protocol = str(data["protocol_version"])
    if (features.shape != (256, FEATURE_DIM) or labels.shape != (256,)
            or np.any(trial_ids < 0) or np.any(trial_ids >= 50)
            or int(np.sum(labels)) != 128 or not np.all(np.isfinite(features))):
        raise AssertionError(f"invalid source-only feature cache: {path}")
    if protocol != PROTOCOL_VERSION:
        raise AssertionError(f"feature cache protocol mismatch: {path}")
    return features, labels, sha, commit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 55)))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--feature-root", type=Path,
                    default=Path("data/interim/lee2019_cbramod_frozen_features"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("data/interim/lee2019_cbramod_frozen_loso_scores"))
    args = ap.parse_args()
    if (not args.subjects or any(s < 1 or s > 54 for s in args.subjects)
            or not args.sessions or any(s not in (1, 2) for s in args.sessions)):
        raise ValueError("invalid subject/session selection")
    if args.feature_root.resolve() == args.output_root.resolve():
        raise ValueError("feature and score output directories must differ")
    for session in args.sessions:
        source = {}
        signatures = set()
        for subject in range(1, 55):
            path = args.feature_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"complete 54-subject feature cache required: {path}")
            features, labels, sha, commit = load_source_feature(path, subject, session)
            source[subject] = (features, labels)
            signatures.add((sha, commit))
        if len(signatures) != 1:
            raise AssertionError("mixed pretrained checkpoints or upstream revisions")
        checkpoint_sha, upstream_sha = next(iter(signatures))
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
                raise AssertionError("non-finite linear-probe scores")
            metadata = {
                "source_subjects": donors,
                "source_trial_ids": "0..49",
                "target_trials_used_for_fit": False,
                "source_training_windows": n_train,
                "classifier": "source-only StandardScaler + balanced logistic regression C=1",
                "feature": "frozen CBraMod 2-s trailing-window global+C3/Cz/C4 embeddings",
                "interpretation": "exploratory cue-paced Lee transfer; no uncued or MI-specific claim",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output, subject=np.int16(target), session=np.int8(session),
                model_name=np.asarray(MODEL_NAME), times=times, pre_times=pre_times,
                protocol_version=np.asarray(PROTOCOL_VERSION),
                cue_times=cues, raw_scores=raw, pre_raw_scores=pre_raw,
                checkpoint_sha256=np.asarray(checkpoint_sha),
                upstream_commit=np.asarray(upstream_sha),
                backbone_parameters=np.int64(parameters),
                provenance_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            print(f"saved S{session} sub-{target:02d}: "
                  f"donors=53 train={n_train} decisions={len(raw)}", flush=True)


if __name__ == "__main__":
    main()
