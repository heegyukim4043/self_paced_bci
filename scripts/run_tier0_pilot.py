from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import (
    build_intervals,
    channel_indices,
    lee2019_path,
    load_lee2019_run,
    select_intervals,
    validate_mat_container,
)
from uncued_mi.features import (
    ScoreNormalizer,
    decision_subsample,
    extract_log_power,
    sample_mask,
    tier0_score,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("data/interim/tier0_pilot.json")
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    path = lee2019_path(args.data_root, args.subject, args.session)
    validate_mat_container(path)
    run = load_lee2019_run(path, phase="train")
    indices = channel_indices(run.channels, config.selected_channels)
    selected_names = tuple(run.channels[index] for index in indices)

    rest_stream = extract_log_power(
        run.pre_rest,
        run.sfreq,
        run.channels,
        config,
        output_channels=selected_names,
    )
    normalizer = ScoreNormalizer.fit(rest_stream.log_power, config.epsilon)
    rest_score = tier0_score(rest_stream, normalizer, config.gate_channels)

    continuous_stream = extract_log_power(
        run.eeg,
        run.sfreq,
        run.channels,
        config,
        output_channels=selected_names,
    )
    continuous_score = tier0_score(
        continuous_stream, normalizer, config.gate_channels
    )
    decision_times, decision_scores = decision_subsample(
        continuous_stream.times,
        continuous_score,
        config.decision_step_seconds,
    )

    intervals = build_intervals(run, config)
    masks = {
        kind: sample_mask(decision_times, select_intervals(intervals, kind))
        for kind in ("fixation", "command", "idle_iti")
    }
    summary = {
        "subject": args.subject,
        "session": args.session,
        "file": str(path),
        "sfreq": run.sfreq,
        "continuous_seconds": run.eeg.shape[0] / run.sfreq,
        "n_trials": len(run.cue_samples),
        "channels": list(selected_names),
        "rest_feature_windows": len(rest_score),
        "decision_step_seconds": config.decision_step_seconds,
        "interval_seconds": {
            kind: float(np.sum(mask) * config.decision_step_seconds)
            for kind, mask in masks.items()
        },
        "score": {
            kind: {
                "n": int(np.sum(mask)),
                "mean": float(np.mean(decision_scores[mask])),
                "std": float(np.std(decision_scores[mask])),
                "q95": float(np.quantile(decision_scores[mask], 0.95)),
            }
            for kind, mask in masks.items()
        },
        "pre_rest_score": {
            "mean": float(np.mean(rest_score)),
            "std": float(np.std(rest_score)),
            "q95": float(np.quantile(rest_score, 0.95)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
