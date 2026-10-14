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
from uncued_mi.operating_point import ScoredSequence, apply_decision


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def interval_alarm_count(
    times: np.ndarray, alarms: np.ndarray, intervals: np.ndarray
) -> int:
    return int(
        sum(
            np.count_nonzero(alarms & (times >= start) & (times < stop))
            for start, stop in intervals
        )
    )


def paired_permutation(
    early: np.ndarray,
    late: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    if np.sum(late) == 0:
        return None, None
    observed = float(np.sum(early) / np.sum(late))
    permuted = np.empty(permutations)
    for index in range(permutations):
        swap = rng.random(len(early)) < 0.5
        first = np.where(swap, late, early)
        second = np.where(swap, early, late)
        permuted[index] = np.sum(first) / np.sum(second) if np.sum(second) else np.inf
    finite = (permuted > 0) & np.isfinite(permuted)
    p_value = float(
        (1 + np.sum(np.abs(np.log(permuted[finite])) >= abs(np.log(observed))))
        / (1 + np.sum(finite))
    )
    return observed, p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--bin-seconds", type=float, default=0.2)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/fixation_psth.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("data/processed/fixation_psth.png"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    edges = np.arange(-1.0, 3.0 + args.bin_seconds / 2, args.bin_seconds)

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
            onset = np.asarray(cache["fixation_intervals"])[:, 0]
            early = np.c_[onset, onset + 1.0]
            late = np.c_[onset + 2.0, onset + 3.0]
            bin_counts = []
            for left, right in zip(edges[:-1], edges[1:]):
                intervals = np.c_[onset + left, onset + right]
                bin_counts.append(interval_alarm_count(times, alarms, intervals))
            key = (
                row["session"],
                row["decision"],
                row["target_fp_setting"],
            )
            groups[key].append(
                {
                    "subject": row["subject"],
                    "early_count": interval_alarm_count(times, alarms, early),
                    "late_count": interval_alarm_count(times, alarms, late),
                    "bin_counts": bin_counts,
                    "trials": len(onset),
                    "fixation_total_count": interval_alarm_count(
                        times, alarms, cache["fixation_intervals"]
                    ),
                    "iti_total_count": interval_alarm_count(
                        times, alarms, cache["iti_intervals"]
                    ),
                    "fixation_seconds": float(
                        np.sum(
                            cache["fixation_intervals"][:, 1]
                            - cache["fixation_intervals"][:, 0]
                        )
                    ),
                    "iti_seconds": float(
                        np.sum(
                            cache["iti_intervals"][:, 1]
                            - cache["iti_intervals"][:, 0]
                        )
                    ),
                }
            )

    output = []
    for key, rows in sorted(groups.items()):
        early = np.asarray([row["early_count"] for row in rows])
        late = np.asarray([row["late_count"] for row in rows])
        ratio, p_value = paired_permutation(
            early, late, args.permutations, rng
        )
        counts = np.sum([row["bin_counts"] for row in rows], axis=0)
        trials = int(np.sum([row["trials"] for row in rows]))
        iti_rates = np.asarray(
            [
                row["iti_total_count"] / row["iti_seconds"] * 60.0
                for row in rows
            ]
        )
        continuous_idle_rates = np.asarray(
            [
                (row["iti_total_count"] + row["fixation_total_count"])
                / (row["iti_seconds"] + row["fixation_seconds"])
                * 60.0
                for row in rows
            ]
        )
        output.append(
            {
                "session": key[0],
                "decision": key[1],
                "target_fp_setting": key[2],
                "subjects": len(rows),
                "early_fixation_events": int(np.sum(early)),
                "late_fixation_events": int(np.sum(late)),
                "rate_ratio_early_vs_late": ratio,
                "paired_permutation_p_value": p_value,
                "iti_only_fp_per_minute_median": float(np.median(iti_rates)),
                "iti_plus_fixation_fp_per_minute_median": float(
                    np.median(continuous_idle_rates)
                ),
                "continuous_idle_minus_iti_fp_per_minute_median": float(
                    np.median(continuous_idle_rates - iti_rates)
                ),
                "psth": {
                    "bin_edges_seconds": [float(value) for value in edges],
                    "alarm_counts": [int(value) for value in counts],
                    "alarm_rate_per_minute": [
                        float(value / (trials * args.bin_seconds) * 60.0)
                        for value in counts
                    ],
                    "trials": trials,
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
        for row in output:
            axis = axes[row["session"] - 1, 0 if row["decision"] == "threshold" else 1]
            bin_edges = np.asarray(row["psth"]["bin_edges_seconds"])
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            axis.plot(
                centers,
                row["psth"]["alarm_rate_per_minute"],
                label=f"FP target {row['target_fp_setting']:g}",
            )
            axis.axvline(0, color="black", linewidth=0.8, alpha=0.5)
            axis.set_title(
                f"Session {row['session']} · {row['decision']}"
            )
            axis.grid(alpha=0.2)
        for axis in axes[-1]:
            axis.set_xlabel("Time from fixation onset (s)")
        for axis in axes[:, 0]:
            axis.set_ylabel("Alarm rate (/min)")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps([{k: v for k, v in row.items() if k != "psth"} for row in output], indent=2))


if __name__ == "__main__":
    main()
