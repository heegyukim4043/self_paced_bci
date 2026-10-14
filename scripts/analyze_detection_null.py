from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.features import decision_subsample
from uncued_mi.metrics import chance_tpr, evaluate_alarms
from uncued_mi.operating_point import ScoredSequence, apply_decision


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_aligned_sequence(cache: dict[str, np.ndarray], step: float) -> ScoredSequence:
    channels = tuple(str(value) for value in cache["channels"])
    normalizer = LocationScale.fit(cache["pre_features"], axis=0)
    scores = tier0_from_features(
        cache["continuous_features"], channels, normalizer
    )
    times, scores = decision_subsample(
        cache["continuous_times"], scores, step
    )
    return ScoredSequence(
        times=times,
        scores=scores,
        idle_intervals=cache["iti_intervals"],
        command_intervals=cache["command_intervals"],
    )


def second_half(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    return intervals[len(intervals) // 2 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument(
        "--evaluation-half",
        choices=["all", "second"],
        default="all",
        help="Use the same held-out second half as the within-subject baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/detection_circular_shift_null.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for input_path in args.inputs:
        for row in json.loads(input_path.read_text(encoding="utf-8")):
            if row["calibration"] != "idle_feature_alignment":
                continue
            cache_path = (
                args.cache_root
                / f"session{row['session']}"
                / f"sub-{row['subject']:02d}_tier0_features.npz"
            )
            cache = load_cache(cache_path)
            sequence = make_aligned_sequence(cache, args.decision_step)
            idle_intervals = (
                second_half(sequence.idle_intervals)
                if args.evaluation_half == "second"
                else sequence.idle_intervals
            )
            command_intervals = (
                second_half(sequence.command_intervals)
                if args.evaluation_half == "second"
                else sequence.command_intervals
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
            observed = evaluate_alarms(
                sequence.times,
                alarms,
                idle_intervals,
                command_intervals,
            )
            offsets = rng.integers(1, len(alarms), size=args.shifts)
            null_tpr = np.empty(args.shifts)
            for index, offset in enumerate(offsets):
                shifted = np.roll(alarms, int(offset))
                null_tpr[index] = evaluate_alarms(
                    sequence.times,
                    shifted,
                    idle_intervals,
                    command_intervals,
                ).true_positive_rate
            p_value = float(
                (1 + np.sum(null_tpr >= observed.true_positive_rate))
                / (args.shifts + 1)
            )
            delays = np.asarray(observed.delay_seconds)
            uniform_p = (
                float(kstest(delays / 3.0, "uniform").pvalue)
                if len(delays) >= 5
                else None
            )
            rows.append(
                {
                    "subject": row["subject"],
                    "evaluation_half": args.evaluation_half,
                    "session": row["session"],
                    "decision": row["decision"],
                    "target_fp_setting": row["target_fp_setting"],
                    "observed_tpr": observed.true_positive_rate,
                    "null_tpr_median": float(np.median(null_tpr)),
                    "null_tpr_95": [
                        float(value) for value in np.quantile(null_tpr, [0.025, 0.975])
                    ],
                    "circular_shift_p_value": p_value,
                    "observed_minus_null_median": float(
                        observed.true_positive_rate - np.median(null_tpr)
                    ),
                    "observed_over_null_median": (
                        float(observed.true_positive_rate / np.median(null_tpr))
                        if np.median(null_tpr) > 0
                        else None
                    ),
                    "observed_fp_per_minute": observed.fp_per_minute,
                    "naive_poisson_chance_tpr": chance_tpr(
                        observed.fp_per_minute,
                        float(
                            np.median(
                                command_intervals[:, 1]
                                - command_intervals[:, 0]
                            )
                        ),
                    ),
                    "detected_events": observed.detected_events,
                    "delay_uniform_ks_p_descriptive": uniform_p,
                }
            )

    summaries = []
    keys = sorted(
        {
            (row["session"], row["decision"], row["target_fp_setting"])
            for row in rows
        }
    )
    for session, decision, target_fp in keys:
        group = [
            row
            for row in rows
            if (
                row["session"],
                row["decision"],
                row["target_fp_setting"],
            )
            == (session, decision, target_fp)
        ]
        summaries.append(
            {
                "session": session,
                "evaluation_half": args.evaluation_half,
                "decision": decision,
                "target_fp_setting": target_fp,
                "subjects": len(group),
                "significant_fraction_p_lt_0_05": float(
                    np.mean([row["circular_shift_p_value"] < 0.05 for row in group])
                ),
                "observed_minus_null_median": float(
                    np.median(
                        [row["observed_minus_null_median"] for row in group]
                    )
                ),
                "observed_tpr_median": float(
                    np.median([row["observed_tpr"] for row in group])
                ),
                "null_tpr_median": float(
                    np.median([row["null_tpr_median"] for row in group])
                ),
                "observed_over_null_ratio_of_medians": (
                    float(
                        np.median([row["observed_tpr"] for row in group])
                        / np.median([row["null_tpr_median"] for row in group])
                    )
                    if np.median([row["null_tpr_median"] for row in group]) > 0
                    else None
                ),
                "naive_poisson_chance_median": float(
                    np.median([row["naive_poisson_chance_tpr"] for row in group])
                ),
                "circular_over_naive_ratio": (
                    float(
                        np.median([row["null_tpr_median"] for row in group])
                        / np.median(
                            [row["naive_poisson_chance_tpr"] for row in group]
                        )
                    )
                    if np.median(
                        [row["naive_poisson_chance_tpr"] for row in group]
                    )
                    > 0
                    else None
                ),
            }
        )

    payload = {"summary": summaries, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
