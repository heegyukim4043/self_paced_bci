"""Prepare identical Riemannian LR score streams for HSMM and MA+dwell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.hsmm import GaussianEmission
from uncued_mi.mil import SEEDS
from uncued_mi.riemannian import tangent_features, window_covariances

SUBSET20 = (
    2, 3, 9, 10, 12, 13, 14, 17, 19, 20,
    23, 25, 26, 29, 38, 40, 42, 51, 53, 54,
)
SFREQ = 250
WINDOW = 250
DECISION_STEP = 0.5


def unpad_positive(data, stop: int) -> np.ndarray:
    values = []
    for trial in range(stop):
        size = int(data["positive_bag_sizes"][trial])
        values.extend(data["positive_end_samples"][trial, :size])
    return np.asarray(values, dtype=np.int64)


def interval_mask(
    times: np.ndarray, intervals: np.ndarray
) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in np.asarray(intervals).reshape(-1, 2):
        mask |= (times >= start) & (times < stop)
    return mask


def build_one(source: Path, target: Path, train_stop: int) -> None:
    with np.load(source, allow_pickle=False) as data:
        signal = np.asarray(data["signal"])
        pre = np.asarray(data["pre_rest"])
        positive = unpad_positive(data, train_stop)
        iti_trials = np.asarray(data["iti_instance_trials"])
        idle = np.asarray(data["iti_end_samples"])[iti_trials < train_stop]
        pre_ends = np.r_[
            data["pre_rest_fit_end_samples"],
            data["pre_rest_validation_end_samples"],
        ]
        command_intervals = np.asarray(data["command_intervals"])
        iti_intervals = np.asarray(data["iti_intervals"])
        iti_interval_trials = np.asarray(data["iti_trials"])

    reference = window_covariances(pre, pre_ends, WINDOW).mean(axis=0)
    reference += (
        np.eye(reference.shape[0])
        * np.trace(reference)
        / reference.shape[0]
        * 1e-6
    )
    train_ends = np.r_[positive, idle]
    train_y = np.r_[
        np.ones(len(positive), dtype=np.int8),
        np.zeros(len(idle), dtype=np.int8),
    ]
    train_x = tangent_features(
        window_covariances(signal, train_ends, WINDOW), reference
    )
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEEDS[0],
    ).fit(scaler.transform(train_x), train_y)

    first_end = int(3.0 * SFREQ)
    step = int(DECISION_STEP * SFREQ)
    decision_ends = np.arange(
        first_end, len(signal) + 1, step, dtype=np.int64
    )
    decision_x = tangent_features(
        window_covariances(signal, decision_ends, WINDOW), reference
    )
    scores = model.decision_function(scaler.transform(decision_x))
    times = decision_ends / SFREQ

    fit_commands = command_intervals[:train_stop]
    fit_idle = iti_intervals[iti_interval_trials < train_stop]
    command_mask = interval_mask(times, fit_commands)
    idle_mask = interval_mask(times, fit_idle)
    emission = GaussianEmission.fit(scores[idle_mask], scores[command_mask])

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        times=times.astype(np.float32),
        scores=scores.astype(np.float32),
        command_intervals=command_intervals.astype(np.float32),
        iti_intervals=iti_intervals.astype(np.float32),
        iti_trials=iti_interval_trials.astype(np.int16),
        train_stop=np.int16(train_stop),
        emission=np.asarray(
            [
                emission.idle_mean,
                emission.idle_std,
                emission.mi_mean,
                emission.mi_std,
            ],
            dtype=np.float64,
        ),
        audit_train_positive=np.int32(len(positive)),
        audit_train_idle=np.int32(len(idle)),
        audit_emission_idle=np.int32(np.sum(idle_mask)),
        audit_emission_mi=np.int32(np.sum(command_mask)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/interim/lee2019_mil_raw")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/interim/hsmm_scores")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--train-stops", type=int, nargs="+", default=[40, 50])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for session in args.sessions:
        for subject in args.subjects:
            source = (
                args.raw_root
                / f"session{session}"
                / f"sub-{subject:02d}_mil_raw.npz"
            )
            for train_stop in args.train_stops:
                target = (
                    args.output_root
                    / f"session{session}"
                    / f"sub-{subject:02d}_train-{train_stop}.npz"
                )
                if target.exists() and not args.overwrite:
                    print(f"exists {target}", flush=True)
                    continue
                build_one(source, target, train_stop)
                print(
                    f"cached session={session} subject={subject} "
                    f"train_stop={train_stop}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
