"""Audit whether cross-block rest seams are locally detectable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feature(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # A causal gate can exploit changes in channel scale/mean at a boundary.
    eps = 1e-6
    return np.r_[
        right.mean(0) - left.mean(0),
        np.log(right.var(0) + eps) - np.log(left.var(0) + eps),
    ]


def subject_examples(path: Path, subject: int, half_window_seconds: float):
    with np.load(path, allow_pickle=False) as data:
        sfreq = float(data["sfreq"])
        blocks = [data[name] for name in ("post_train", "pre_test", "post_test")]
    half = int(round(half_window_seconds * sfreq))
    rng = np.random.default_rng(20260804 + subject)
    positives, negatives = [], []
    # Cross-recording seams: end of one rest recording to start of another.
    for left_id, right_id in ((0, 1), (1, 2), (2, 0)):
        positives.append(feature(blocks[left_id][-half:], blocks[right_id][:half]))
    # Matched continuous-rest controls, three per block.
    for block in blocks:
        centers = rng.integers(half, len(block) - half, size=3)
        negatives.extend(feature(block[c-half:c], block[c:c+half]) for c in centers)
    x = np.asarray(positives + negatives)
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    return x, y, np.full(len(y), subject)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/interim/lee2019_four_rest_raw/session1"))
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--half-window-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/lee2019_pseudo_uncued_seam_audit.json"))
    args = parser.parse_args()
    paths = sorted(args.cache_root.glob("sub-*.npz"))
    if args.subjects is not None:
        wanted = set(args.subjects)
        paths = [p for p in paths if int(p.stem.split("-")[1]) in wanted]
    rows = [subject_examples(p, int(p.stem.split("-")[1]), args.half_window_seconds)
            for p in paths]
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    groups = np.concatenate([r[2] for r in rows])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", solver="liblinear", random_state=20260804))
    probability = cross_val_predict(model, x, y, groups=groups,
                                    cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, probability))
    rng = np.random.default_rng(20260804)
    null = np.empty(5000)
    for i in range(len(null)):
        permuted = y.copy()
        for group in np.unique(groups):
            mask = groups == group
            permuted[mask] = rng.permutation(permuted[mask])
        null[i] = roc_auc_score(permuted, probability)
    result = {
        "subjects": len(paths), "positive_cross_block_seams": int(y.sum()),
        "continuous_rest_controls": int((y == 0).sum()),
        "half_window_seconds": args.half_window_seconds,
        "leave_one_subject_out_auc": auc,
        "permutation_p_greater": float((1 + np.sum(null >= auc)) / (len(null) + 1)),
        "gate": "PASS only if cross-block seam localizability is not above chance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
