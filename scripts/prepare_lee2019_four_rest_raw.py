"""Prepare normalized causal raw arrays for all four Lee2019 rest blocks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import lee2019_path, load_lee2019_run
from scripts.prepare_lee2019_mil_raw_cache import preprocess


def cache_one(source: Path, target: Path, config: ExperimentConfig) -> None:
    train = load_lee2019_run(source, "train")
    test = load_lee2019_run(source, "test")
    raw = {
        "pre_train": preprocess(
            train.pre_rest, train.channels, train.sfreq,
            config.sensorimotor_channels, config,
        ),
        "post_train": preprocess(
            train.post_rest, train.channels, train.sfreq,
            config.sensorimotor_channels, config,
        ),
        "pre_test": preprocess(
            test.pre_rest, test.channels, test.sfreq,
            config.sensorimotor_channels, config,
        ),
        "post_test": preprocess(
            test.post_rest, test.channels, test.sfreq,
            config.sensorimotor_channels, config,
        ),
    }
    transient = int(round(config.transient_seconds * config.target_sfreq))
    anchor = raw["pre_train"][transient:]
    mean = anchor.mean(axis=0)
    std = np.maximum(anchor.std(axis=0, ddof=1), 1e-6)
    payload = {
        name: ((values - mean) / std).astype(np.float32)
        for name, values in raw.items()
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, **payload,
        sfreq=np.float32(config.target_sfreq),
        channels=np.asarray(config.sensorimotor_channels),
        transient_samples=np.int32(transient),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--output-root", type=Path, default=Path("data/interim/lee2019_four_rest_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(window_seconds=1.0, feature_step_seconds=0.1)
    for session in args.sessions:
        for subject in args.subjects:
            target = args.output_root / f"session{session}" / f"sub-{subject:02d}.npz"
            if not target.exists() or args.overwrite:
                cache_one(lee2019_path(args.data_root, subject, session), target, config)
            print(f"rest-raw session={session} subject={subject}", flush=True)


if __name__ == "__main__":
    main()
