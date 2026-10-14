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


def alarm_count(times: np.ndarray, alarms: np.ndarray, intervals: np.ndarray) -> int:
    return int(
        sum(
            np.count_nonzero(alarms & (times >= start) & (times < stop))
            for start, stop in intervals
        )
    )


def interval_seconds(intervals: np.ndarray) -> float:
    return float(np.sum(intervals[:, 1] - intervals[:, 0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--bin-seconds", type=float, default=0.2)
    parser.add_argument("--shifts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cue_locked_sensitivity.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/processed/cue_locked_psth.png"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    edges = np.arange(-4.0, 6.0 + args.bin_seconds / 2, args.bin_seconds)
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for input_path in args.inputs:
        for result in json.loads(input_path.read_text(encoding="utf-8")):
            if result["calibration"] != "idle_feature_alignment":
                continue
            cache = load_cache(
                args.cache_root
                / f"session{result['session']}"
                / f"sub-{result['subject']:02d}_tier0_features.npz"
            )
            model = LocationScale.fit(cache["pre_features"], axis=0)
            channels = tuple(str(value) for value in cache["channels"])
            scores = tier0_from_features(
                cache["continuous_features"], channels, model
            )
            times, scores = decision_subsample(
                cache["continuous_times"], scores, args.decision_step
            )
            sequence = ScoredSequence(
                times=times,
                scores=scores,
                idle_intervals=cache["iti_intervals"],
                command_intervals=cache["command_intervals"],
            )
            point = result["operating_point"]
            alarms = apply_decision(
                sequence,
                result["decision"],
                point["threshold"],
                args.decision_step,
                drift=point["drift"],
                ma_seconds=0.5,
                dwell_seconds=0.3,
                refractory_seconds=1.0,
            ).alarms
            command = np.asarray(cache["command_intervals"], dtype=float)
            cue = command[:, 0] - 0.5
            extended_command = command.copy()
            extended_command[:, 0] = cue - 1.0
            peri_early = np.c_[cue - 3.0, cue - 1.0]
            peri_cue = np.c_[cue - 1.0, cue]
            original = evaluate_alarms(
                times, alarms, cache["iti_intervals"], command
            )
            extended = evaluate_alarms(
                times, alarms, cache["iti_intervals"], extended_command
            )
            null_original = np.empty(args.shifts)
            null_extended = np.empty(args.shifts)
            for index, offset in enumerate(
                rng.integers(1, len(alarms), size=args.shifts)
            ):
                shifted = np.roll(alarms, int(offset))
                null_original[index] = evaluate_alarms(
                    times, shifted, cache["iti_intervals"], command
                ).true_positive_rate
                null_extended[index] = evaluate_alarms(
                    times, shifted, cache["iti_intervals"], extended_command
                ).true_positive_rate
            bin_counts = [
                alarm_count(times, alarms, np.c_[cue + left, cue + right])
                for left, right in zip(edges[:-1], edges[1:])
            ]
            key = (
                result["session"],
                result["decision"],
                result["target_fp_setting"],
            )
            groups[key].append(
                {
                    "subject": result["subject"],
                    "original_tpr": original.true_positive_rate,
                    "original_null": float(np.median(null_original)),
                    "original_excess": float(
                        original.true_positive_rate - np.median(null_original)
                    ),
                    "extended_tpr": extended.true_positive_rate,
                    "extended_null": float(np.median(null_extended)),
                    "extended_excess": float(
                        extended.true_positive_rate - np.median(null_extended)
                    ),
                    "idle_iti_fp_per_minute": (
                        alarm_count(times, alarms, cache["iti_intervals"])
                        / interval_seconds(cache["iti_intervals"])
                        * 60.0
                    ),
                    "idle_peri_early_fp_per_minute": (
                        alarm_count(times, alarms, peri_early)
                        / interval_seconds(peri_early)
                        * 60.0
                    ),
                    "peri_cue_fp_per_minute": (
                        alarm_count(times, alarms, peri_cue)
                        / interval_seconds(peri_cue)
                        * 60.0
                    ),
                    "bin_counts": bin_counts,
                    "trials": len(cue),
                }
            )

    summaries = []
    for key, rows in sorted(groups.items()):
        counts = np.sum([row["bin_counts"] for row in rows], axis=0)
        trials = int(np.sum([row["trials"] for row in rows]))
        summaries.append(
            {
                "session": key[0],
                "decision": key[1],
                "target_fp_setting": key[2],
                "subjects": len(rows),
                "original_tpr_median": float(
                    np.median([row["original_tpr"] for row in rows])
                ),
                "original_null_median": float(
                    np.median([row["original_null"] for row in rows])
                ),
                "original_excess_median": float(
                    np.median([row["original_excess"] for row in rows])
                ),
                "extended_tpr_median": float(
                    np.median([row["extended_tpr"] for row in rows])
                ),
                "extended_null_median": float(
                    np.median([row["extended_null"] for row in rows])
                ),
                "extended_excess_median": float(
                    np.median([row["extended_excess"] for row in rows])
                ),
                "paired_extended_minus_original_excess_median": float(
                    np.median(
                        [
                            row["extended_excess"] - row["original_excess"]
                            for row in rows
                        ]
                    )
                ),
                "idle_iti_fp_per_minute_median": float(
                    np.median([row["idle_iti_fp_per_minute"] for row in rows])
                ),
                "idle_peri_early_fp_per_minute_median": float(
                    np.median(
                        [row["idle_peri_early_fp_per_minute"] for row in rows]
                    )
                ),
                "peri_cue_fp_per_minute_median": float(
                    np.median([row["peri_cue_fp_per_minute"] for row in rows])
                ),
                "psth": {
                    "bin_edges_seconds_from_cue": [
                        float(value) for value in edges
                    ],
                    "alarm_rate_per_minute": [
                        float(value / (trials * args.bin_seconds) * 60.0)
                        for value in counts
                    ],
                    "trials": trials,
                },
            }
        )
    payload = {"summary": summaries, "rows": [row for rows in groups.values() for row in rows]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        for row in summaries:
            axis = axes[row["session"] - 1, 0 if row["decision"] == "threshold" else 1]
            bin_edges = np.asarray(row["psth"]["bin_edges_seconds_from_cue"])
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            axis.plot(
                centers,
                row["psth"]["alarm_rate_per_minute"],
                label=f"FP target {row['target_fp_setting']:g}",
            )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.axvspan(-1, 0, color="#DD8452", alpha=0.12)
            axis.set_title(f"Session {row['session']} · {row['decision']}")
            axis.grid(alpha=0.2)
        for axis in axes[-1]:
            axis.set_xlabel("Time from cue (s)")
        for axis in axes[:, 0]:
            axis.set_ylabel("Alarm rate (/min)")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps([{k: v for k, v in row.items() if k != "psth"} for row in summaries], indent=2))


if __name__ == "__main__":
    main()
