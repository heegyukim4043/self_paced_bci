"""Cache 1 s covariances for all four Lee2019 rest blocks and fixation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import LeeRun, build_intervals, lee2019_path
from uncued_mi.features import extract_covariances


def channel_names(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(np.squeeze(value).item()) for value in np.ravel(values))


def make_run(struct) -> LeeRun:
    return LeeRun(
        eeg=np.asarray(struct.x, dtype=np.float64),
        cue_samples=np.asarray(struct.t, dtype=np.int64).reshape(-1),
        labels=np.asarray(struct.y_dec, dtype=np.int64).reshape(-1),
        channels=channel_names(struct.chan),
        sfreq=float(struct.fs),
        pre_rest=np.asarray(struct.pre_rest, dtype=np.float64),
        post_rest=np.asarray(struct.post_rest, dtype=np.float64),
    )


def collect(
    times: np.ndarray,
    covariances: np.ndarray,
    intervals,
    kind: str,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, trials, relative_ends = [], [], []
    for interval in intervals:
        if interval.kind != kind:
            continue
        mask = (
            (times - window_seconds >= interval.start - 1e-9)
            & (times <= interval.stop + 1e-9)
        )
        values.append(covariances[mask])
        trials.append(np.full(mask.sum(), int(interval.trial), dtype=np.int16))
        relative_ends.append((times[mask] - (interval.stop if kind == "fixation" else interval.start)))
    return (
        np.concatenate(values),
        np.concatenate(trials),
        np.concatenate(relative_ends).astype(np.float32),
    )


def rest_covariances(
    signal: np.ndarray,
    sfreq: float,
    channels: tuple[str, ...],
    config: ExperimentConfig,
) -> np.ndarray:
    _, covariance = extract_covariances(
        signal, sfreq, channels, config, config.sensorimotor_channels
    )
    return covariance


def cache_one(source: Path, target: Path, config: ExperimentConfig) -> dict:
    mat = loadmat(
        source,
        variable_names=["EEG_MI_train", "EEG_MI_test"],
        squeeze_me=True,
        struct_as_record=False,
        verify_compressed_data_integrity=False,
    )
    runs = {
        "train": make_run(mat["EEG_MI_train"]),
        "test": make_run(mat["EEG_MI_test"]),
    }
    payload: dict[str, np.ndarray] = {}
    audit: dict[str, object] = {
        "source": str(source.resolve()),
        "window_seconds": config.window_seconds,
        "step_seconds": config.feature_step_seconds,
        "rest_blocks": {},
        "phase_windows": {},
    }
    for phase, run in runs.items():
        times, covariance = extract_covariances(
            run.eeg,
            run.sfreq,
            run.channels,
            config,
            config.sensorimotor_channels,
        )
        intervals = build_intervals(run, config)
        for kind in ("command", "idle_iti", "fixation"):
            values, trials, relative = collect(
                times, covariance, intervals, kind, config.window_seconds
            )
            prefix = f"{phase}_{kind}"
            payload[f"{prefix}_covariance"] = values.astype(np.float32)
            payload[f"{prefix}_trial"] = trials
            payload[f"{prefix}_relative_end"] = relative
            audit["phase_windows"][prefix] = int(len(values))
        for position, signal in (
            ("pre", run.pre_rest),
            ("post", run.post_rest),
        ):
            name = f"{position}_{phase}"
            values = rest_covariances(
                signal, run.sfreq, run.channels, config
            ).astype(np.float32)
            payload[f"{name}_covariance"] = values
            audit["rest_blocks"][name] = int(len(values))

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **payload,
        channels=np.asarray(config.sensorimotor_channels),
        audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/lee2019_mi")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/lee2019_four_rest"),
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(
        window_seconds=1.0, feature_step_seconds=0.1
    )
    for session in args.sessions:
        for subject in args.subjects:
            target = (
                args.output_root
                / f"session{session}"
                / f"sub-{subject:02d}.npz"
            )
            if target.exists() and not args.overwrite:
                print(f"exists session={session} subject={subject}", flush=True)
                continue
            audit = cache_one(
                lee2019_path(args.data_root, subject, session),
                target,
                config,
            )
            print(
                f"cached session={session} subject={subject} "
                f"rest={audit['rest_blocks']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
