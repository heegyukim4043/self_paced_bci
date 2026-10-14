from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.features import decision_subsample
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, apply_decision


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def pooled_rate_ratio(
    fixation_counts: np.ndarray,
    late_counts: np.ndarray,
    exposure_seconds: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    observed = float(np.sum(fixation_counts) / np.sum(late_counts)) if np.sum(
        late_counts
    ) else None
    ratios = []
    for _ in range(permutations):
        swap = rng.random(len(fixation_counts)) < 0.5
        first = np.where(swap, late_counts, fixation_counts)
        second = np.where(swap, fixation_counts, late_counts)
        if np.sum(second):
            ratios.append(float(np.sum(first) / np.sum(second)))
    if observed is None or observed <= 0:
        p_value = None
    else:
        ratios_array = np.asarray(ratios)
        p_value = float(
            (1 + np.sum(np.abs(np.log(ratios_array)) >= abs(np.log(observed))))
            / (1 + len(ratios_array))
        )
    return {
        "rate_ratio_fixation_vs_late_iti": observed,
        "paired_permutation_p_value": p_value,
        "fixation_events": int(np.sum(fixation_counts)),
        "late_iti_events": int(np.sum(late_counts)),
        "exposure_seconds_per_condition": float(np.sum(exposure_seconds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/vep_within_trial_late_iti.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for input_path in args.inputs:
        for row in json.loads(input_path.read_text(encoding="utf-8")):
            if row["calibration"] != "idle_feature_alignment":
                continue
            cache = load_cache(
                args.cache_root
                / f"session{row['session']}"
                / f"sub-{row['subject']:02d}_tier0_features.npz"
            )
            model = LocationScale.fit(cache["pre_features"], axis=0)
            channels = tuple(str(value) for value in cache["channels"])
            raw_scores = tier0_from_features(
                cache["continuous_features"], channels, model
            )
            times, scores = decision_subsample(
                cache["continuous_times"], raw_scores, args.decision_step
            )
            sequence = ScoredSequence(
                times=times,
                scores=scores,
                idle_intervals=cache["iti_intervals"],
                command_intervals=cache["command_intervals"],
            )
            point = row["operating_point"]
            alarms = apply_decision(
                sequence,
                row["decision"],
                point["threshold"],
                args.decision_step,
                drift=point["drift"],
                ma_seconds=0.5,
                dwell_seconds=0.3,
                refractory_seconds=1.0,
            ).alarms
            # Pair fixation n+1 with the final second of ITI n. The two
            # conditions are adjacent in time and have equal exposure.
            fixation = np.asarray(cache["fixation_intervals"][1:], dtype=float)
            fixation[:, 1] = fixation[:, 0] + 1.0
            late_iti = np.asarray(cache["iti_intervals"], dtype=float).copy()
            late_iti[:, 0] = late_iti[:, 1] - 1.0
            fixation_metrics = evaluate_alarms(
                times, alarms, fixation, np.empty((0, 2))
            )
            late_metrics = evaluate_alarms(
                times, alarms, late_iti, np.empty((0, 2))
            )
            key = (
                row["session"],
                row["decision"],
                row["target_fp_setting"],
            )
            groups[key].append(
                {
                    "subject": row["subject"],
                    "fixation_events": fixation_metrics.false_alarms,
                    "late_iti_events": late_metrics.false_alarms,
                    "exposure_seconds": fixation_metrics.idle_seconds,
                }
            )

    output = []
    for key, rows in sorted(groups.items()):
        result = pooled_rate_ratio(
            np.asarray([row["fixation_events"] for row in rows]),
            np.asarray([row["late_iti_events"] for row in rows]),
            np.asarray([row["exposure_seconds"] for row in rows]),
            args.permutations,
            rng,
        )
        output.append(
            {
                "session": key[0],
                "decision": key[1],
                "target_fp_setting": key[2],
                "subjects": len(rows),
                **result,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
