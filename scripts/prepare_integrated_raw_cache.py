from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import build_intervals, lee2019_path, load_lee2019_run
from uncued_mi.features import extract_covariances, extract_log_power


def interval_array(intervals, kind: str) -> np.ndarray:
    return np.asarray(
        [(x.start, x.stop) for x in intervals if x.kind == kind], dtype=np.float64
    ).reshape(-1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--output-root", type=Path, default=Path("data/interim/lee2019_integrated"))
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    parser.add_argument("--windows", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    base = ExperimentConfig()
    for session in args.sessions:
        for subject in args.subjects:
            target = args.output_root / f"session{session}" / f"sub-{subject:02d}_integrated.npz"
            if target.exists() and not args.overwrite:
                print(f"exists: {target}")
                continue
            source = lee2019_path(args.data_root, subject, session)
            payload: dict[str, np.ndarray] = {}
            for phase in ("train", "test"):
                run = load_lee2019_run(source, phase)
                intervals = build_intervals(run, base)
                for kind in ("command", "fixation", "idle_iti", "idle_break"):
                    payload[f"{phase}_{kind}_intervals"] = interval_array(intervals, kind)
                for window_seconds in args.windows:
                    config = replace(base, window_seconds=window_seconds)
                    suffix = str(window_seconds).replace(".", "p")
                    band = extract_log_power(
                        run.eeg, run.sfreq, run.channels, config,
                        output_channels=config.selected_channels,
                    )
                    cov_times, covariance = extract_covariances(
                        run.eeg, run.sfreq, run.channels, config,
                        output_channels=config.sensorimotor_channels,
                    )
                    payload[f"{phase}_w{suffix}_times"] = band.times.astype(np.float32)
                    payload[f"{phase}_w{suffix}_bandpower"] = band.log_power.astype(np.float32)
                    payload[f"{phase}_w{suffix}_covariance"] = covariance
                    pre_band = extract_log_power(
                        run.pre_rest, run.sfreq, run.channels, config,
                        output_channels=config.selected_channels,
                    )
                    pre_cov_times, pre_covariance = extract_covariances(
                        run.pre_rest, run.sfreq, run.channels, config,
                        output_channels=config.sensorimotor_channels,
                    )
                    payload[f"{phase}_w{suffix}_pre_times"] = pre_band.times.astype(np.float32)
                    payload[f"{phase}_w{suffix}_pre_bandpower"] = pre_band.log_power.astype(np.float32)
                    payload[f"{phase}_w{suffix}_pre_covariance"] = pre_covariance
                    if not np.allclose(band.times, cov_times):
                        raise AssertionError("Bandpower and covariance timestamps differ")
                    if not np.allclose(pre_band.times, pre_cov_times):
                        raise AssertionError("Pre-rest feature timestamps differ")
                payload[f"{phase}_labels"] = run.labels.astype(np.int8)
                payload[f"{phase}_cue_samples"] = run.cue_samples
            payload["bandpower_channels"] = np.asarray(base.selected_channels)
            payload["covariance_channels"] = np.asarray(base.sensorimotor_channels)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **payload)
            print(f"cached: {target}")


if __name__ == "__main__":
    main()
