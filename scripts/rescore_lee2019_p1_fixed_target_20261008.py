"""Prepare P1 rest scores under each *fixed target* LOSO model/reference.

This is score preparation, not the P1 threshold/FP analysis. In particular,
K_F is already in the locked target stream and is not scored here. No model
is trained by ``--dry-run``. Full execution requires an explicit subject list
or ``--all-subjects`` and never uses target future labels for fitting.

CNN-GRU source subjects contribute the first half of their pre-rest to model
training. The output therefore marks only windows whose entire support lies
in the second half as out-of-training donor calibration candidates. At 0.1 s
steps this leaves about 29 s, so a 36 s K_X comparison is not valid for CNN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SUBJECTS = tuple(range(1, 55))
SEEDS = (20260729, 20260730, 20260731)
STEP_SAMPLES = 25
WINDOW_SAMPLES = 250
SFREQ = 250


def anchor_grid(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Use the original 3 s first end and retain actual 1 s window support."""
    ends = np.arange(3 * SFREQ, n_samples + 1, STEP_SAMPLES, dtype=np.int64)
    return (ends - WINDOW_SAMPLES) / SFREQ, ends / SFREQ


def out_of_training_mask(starts: np.ndarray, midpoint_seconds: float) -> np.ndarray:
    """Exclude even windows that straddle the source's training-rest half."""
    return np.asarray(starts >= midpoint_seconds - 1e-9, dtype=bool)


def check_original_anchor(new: np.ndarray, old: np.ndarray, *, name: str) -> float:
    if new.shape != old.shape or not np.all(np.isfinite(new)):
        raise AssertionError(f"{name}: non-finite or shape mismatch {new.shape} != {old.shape}")
    difference = float(np.max(np.abs(new.astype(float) - old.astype(float))))
    if difference > 1e-3:
        raise AssertionError(f"{name}: original target-anchor mismatch {difference:.6g} > 0.001")
    return difference


def required_paths(args: argparse.Namespace, session: int, subject: int) -> list[Path]:
    stem = Path(f"session{session}") / f"sub-{subject:02d}.npz"
    paths: list[Path] = []
    if args.model in ("riemannian", "both"):
        paths += [args.cov_root / stem, args.riem_cache_root / stem]
    if args.model in ("cnn-gru", "both"):
        paths += [
            args.cnn_cache_root / stem,
            args.prepared_source_root / f"session{session}"
            / f"sub-{subject:02d}_n{args.max_source_windows}.npz",
        ]
    return paths


def rescore_riemannian(
    session: int, subject: int, domains_by_session: dict[int, dict], args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    from scripts.evaluate_lee2019_revalidation_guard_rearm_smoke import fit_loso
    from scripts.run_lee2019_pseudo_uncued_riemannian import reference
    from uncued_mi.riemannian import tangent_features

    own = domains_by_session[session]
    other_session = 3 - session
    ref = reference(own[subject]["pre"])
    scaler, model = fit_loso(own, subject, ref)

    def score(covariance: np.ndarray) -> np.ndarray:
        features = tangent_features(covariance, ref)
        return np.asarray(model.decision_function(scaler.transform(features)), np.float32)

    target_anchor = score(own[subject]["pre"])
    old_path = args.riem_cache_root / f"session{session}" / f"sub-{subject:02d}.npz"
    with np.load(old_path, allow_pickle=False) as old:
        max_error = check_original_anchor(
            target_anchor, np.asarray(old["anchor_raw_scores"]), name=str(old_path),
        )
    donors = tuple(s for s in SUBJECTS if s != subject)
    donor_scores = np.stack([score(own[s]["pre"]) for s in donors])
    other_anchor = score(domains_by_session[other_session][subject]["pre"])
    if len(set([len(target_anchor), len(other_anchor), *map(len, donor_scores)])) != 1:
        raise AssertionError("pre-rest score grids differ across sessions/subjects")
    n_steps = len(target_anchor)
    starts, times = anchor_grid(3 * SFREQ + (n_steps - 1) * STEP_SAMPLES)
    if len(starts) != len(target_anchor):
        # Covariance cache has one element per 0.1 s score step; this relation
        # is checked explicitly instead of silently truncating source scores.
        raise AssertionError("Riemannian covariance grid does not match anchor grid")
    return {
        "target_anchor_scores": target_anchor,
        "other_session_anchor_scores": other_anchor,
        "donor_anchor_scores": donor_scores,
        "donor_subjects": np.asarray(donors, np.int16),
        "anchor_window_starts_seconds": starts,
        "anchor_window_ends_seconds": times,
        "donor_out_of_training_mask": np.ones(len(starts), np.uint8),
        "target_anchor_cache_max_abs_error": np.asarray(max_error),
        "source_training_overlap_warning": np.asarray("none: donor pre-rest is not used to fit Riemannian LR"),
    }


def rescore_cnn(
    session: int, subject: int, source_pool: dict[int, dict], args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    import torch
    from scripts.run_lee2019_cnn_gru_loso_arm_a_stream_v2 import (
        _cached_source_training, _window_ends, predict_signal, train_source_pool,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use it")
    donors = tuple(s for s in SUBJECTS if s != subject)
    target_signal = source_pool[subject]["anchor_signal"]
    other = _cached_source_training(
        args.data_root, args.train_cache_root, args.prepared_source_root,
        subject, 3 - session, args.max_source_windows,
    )
    other_signal = other["anchor_signal"]
    starts, times = anchor_grid(target_signal.shape[-1])
    ends = _window_ends(target_signal.shape[-1], first_end=3 * SFREQ)
    if not np.array_equal(times, ends / SFREQ):
        raise AssertionError("CNN pre-rest grid differs from the locked scorer")
    donor_mask = out_of_training_mask(
        starts, float(source_pool[donors[0]]["anchor_midpoint_seconds"]),
    )
    target_by_seed = []
    other_by_seed = []
    donor_by_seed = []
    for seed in args.seeds:
        model = train_source_pool([source_pool[s] for s in donors], seed, args.epochs, device)
        target_by_seed.append(predict_signal(model, target_signal, device, ends=ends))
        other_by_seed.append(predict_signal(model, other_signal, device, ends=ends))
        donor_by_seed.append(np.stack([
            predict_signal(model, source_pool[s]["anchor_signal"], device, ends=ends)
            for s in donors
        ]))
        del model
    target_seed = np.asarray(target_by_seed, np.float32)
    other_seed = np.asarray(other_by_seed, np.float32)
    donor_seed = np.asarray(donor_by_seed, np.float32)
    target_anchor = np.mean(target_seed, axis=0)
    old_path = args.cnn_cache_root / f"session{session}" / f"sub-{subject:02d}.npz"
    with np.load(old_path, allow_pickle=False) as old:
        if not np.array_equal(old["seeds"], np.asarray(args.seeds)):
            raise AssertionError(f"{old_path}: seed list differs from locked cache")
        max_error = check_original_anchor(
            target_anchor, np.asarray(old["anchor_scores"]), name=str(old_path),
        )
    for donor in donors:
        donor_starts, donor_times = anchor_grid(source_pool[donor]["anchor_signal"].shape[-1])
        if not np.array_equal(starts, donor_starts) or not np.array_equal(times, donor_times):
            raise AssertionError(f"donor sub-{donor:02d}: anchor grid mismatch")
        mask = out_of_training_mask(
            donor_starts, float(source_pool[donor]["anchor_midpoint_seconds"]),
        )
        if not np.array_equal(donor_mask, mask):
            raise AssertionError(f"donor sub-{donor:02d}: training-rest boundary mismatch")
    return {
        "target_anchor_scores": target_anchor.astype(np.float32),
        "other_session_anchor_scores": np.mean(other_seed, axis=0).astype(np.float32),
        "donor_anchor_scores": np.mean(donor_seed, axis=0).astype(np.float32),
        "target_anchor_seed_scores": target_seed,
        "other_session_anchor_seed_scores": other_seed,
        "donor_anchor_seed_scores": donor_seed,
        "seeds": np.asarray(args.seeds, np.int64),
        "donor_subjects": np.asarray(donors, np.int16),
        "anchor_window_starts_seconds": starts,
        "anchor_window_ends_seconds": times,
        "donor_out_of_training_mask": donor_mask.astype(np.uint8),
        "target_anchor_cache_max_abs_error": np.asarray(max_error),
        "source_training_overlap_warning": np.asarray(
            "CNN donor pre-rest <=30 s was in source training; only mask==1 is eligible for K_X"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("riemannian", "cnn-gru", "both"), required=True)
    subjects = parser.add_mutually_exclusive_group(required=True)
    subjects.add_argument("--subjects", type=int, nargs="+")
    subjects.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--sessions", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--dry-run", action="store_true", help="inventory only; no fitting or output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/interim/p1_fixed_target_rest_scores_20261008")
    parser.add_argument("--cov-root", type=Path, default=ROOT / "data/interim/lee2019_revalidation_norm_cov")
    parser.add_argument("--riem-cache-root", type=Path, default=ROOT / "data/interim/lee2019_revalidation_riemannian_scores")
    parser.add_argument("--cnn-cache-root", type=Path, default=ROOT / "data/interim/lee2019_cnn_gru_chronological_score_cache")
    parser.add_argument("--prepared-source-root", type=Path, default=ROOT / "data/interim/lee2019_cnn_gru_loso_prepared_source")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/lee2019_mi")
    parser.add_argument("--train-cache-root", type=Path, default=ROOT / "data/interim/lee2019_mil_raw")
    parser.add_argument("--max-source-windows", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.subjects = list(SUBJECTS) if args.all_subjects else args.subjects
    if any(s not in SUBJECTS for s in args.subjects):
        parser.error("subjects must be in 1..54")
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("duplicate subject ID")
    if args.epochs < 1 or not args.seeds or args.max_source_windows < 1:
        parser.error("epochs, max-source-windows, and seed count must be positive")
    return args


def main() -> None:
    args = parse_args()
    models = ("riemannian", "cnn-gru") if args.model == "both" else (args.model,)
    sessions = tuple(sorted(set(args.sessions)))
    required = set()
    for session in sessions:
        for subject in SUBJECTS:
            required.update(required_paths(args, session, subject))
        if args.model in ("riemannian", "both"):
            other = 3 - session
            for subject in args.subjects:
                required.add(args.cov_root / f"session{other}" / f"sub-{subject:02d}.npz")
        if args.model in ("cnn-gru", "both"):
            other = 3 - session
            for subject in args.subjects:
                required.add(args.prepared_source_root / f"session{other}"
                             / f"sub-{subject:02d}_n{args.max_source_windows}.npz")
    missing = sorted(path for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(f"{len(missing)} required inputs absent; first: {missing[0]}")
    jobs = [(model, session, subject) for model in models for session in sessions for subject in args.subjects]
    print(f"P1 fixed-target rest rescore: {len(jobs)} jobs; dry_run={args.dry_run}", flush=True)
    print("No thresholds, FP, SEN, or inferential statistics are computed.", flush=True)
    if args.dry_run:
        return
    for session in sessions:
        riem_domains = None
        cnn_pool = None
        if "riemannian" in models:
            from scripts.evaluate_lee2019_revalidation_guard_rearm_smoke import source_domains
            riem_domains = {
                session: source_domains(session, args.cov_root),
                3 - session: source_domains(3 - session, args.cov_root),
            }
        if "cnn-gru" in models:
            from scripts.run_lee2019_cnn_gru_loso_arm_a_stream_v2 import _cached_source_training
            cnn_pool = {
                source: _cached_source_training(
                    args.data_root, args.train_cache_root, args.prepared_source_root,
                    source, session, args.max_source_windows,
                )
                for source in SUBJECTS
            }
        for subject in args.subjects:
            for model in models:
                destination = args.output_root / model / f"session{session}" / f"sub-{subject:02d}.npz"
                if destination.exists() and not args.overwrite:
                    print(f"skip existing {destination}", flush=True)
                    continue
                if model == "riemannian":
                    assert riem_domains is not None
                    scores = rescore_riemannian(session, subject, riem_domains, args)
                else:
                    assert cnn_pool is not None
                    scores = rescore_cnn(session, subject, cnn_pool, args)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination, subject=np.asarray(subject, np.int16),
                    session=np.asarray(session, np.int8),
                    protocol=np.asarray("P1 fixed-target LOSO rest rescore; diagnostic simulation only"),
                    **scores,
                )
                print(f"saved {destination}", flush=True)


if __name__ == "__main__":
    main()
