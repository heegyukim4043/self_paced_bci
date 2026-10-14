from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.calibration import LocationScale, tier0_from_features
from uncued_mi.decision import moving_average_dwell
from uncued_mi.features import decision_subsample, sample_mask


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def select_thresholds(path: Path, session: int, target_fp: float) -> dict[int, float]:
    return {
        int(row["subject"]): float(row["operating_point"]["threshold"])
        for row in json.loads(path.read_text(encoding="utf-8"))
        if row["session"] == session
        and row["decision"] == "ma_dwell"
        and row["calibration"] == "idle_feature_alignment"
        and float(row["target_fp_setting"]) == target_fp
    }


def interval_metric(
    times: np.ndarray,
    alarms: np.ndarray,
    idle: np.ndarray,
    command: np.ndarray,
) -> tuple[int, int, float]:
    alarm_times = times[np.asarray(alarms, dtype=bool)]
    detected = sum(
        bool(np.any((alarm_times >= start) & (alarm_times < stop)))
        for start, stop in np.asarray(command).reshape(-1, 2)
    )
    false_alarms = sum(
        int(np.count_nonzero((alarm_times >= start) & (alarm_times < stop)))
        for start, stop in np.asarray(idle).reshape(-1, 2)
    )
    idle_seconds = float(
        np.sum(
            np.asarray(idle).reshape(-1, 2)[:, 1]
            - np.asarray(idle).reshape(-1, 2)[:, 0]
        )
    )
    return detected, false_alarms, idle_seconds


def summarize(records: list[dict], shifts: int) -> dict:
    observed_events = sum(row["detected"] for row in records)
    total_events = len(records)
    null_counts = np.sum([row["_null_detected"] for row in records], axis=0)
    fp = sum(row["false_alarms"] for row in records)
    idle_seconds = sum(row["idle_seconds"] for row in records)
    observed_tpr = observed_events / total_events if total_events else None
    null_tpr = null_counts / total_events if total_events else np.zeros(shifts)
    return {
        "trials": total_events,
        "subjects": len({row["subject"] for row in records}),
        "calibration_seconds_median": float(
            np.median([row["calibration_seconds"] for row in records])
        ),
        "fp_per_minute": 60.0 * fp / idle_seconds if idle_seconds else None,
        "observed_tpr": observed_tpr,
        "null_tpr_median": float(np.median(null_tpr)),
        "excess_tpr": (
            float(observed_tpr - np.median(null_tpr))
            if observed_tpr is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/interim/lee2019_tier0")
    )
    parser.add_argument(
        "--session1-results",
        type=Path,
        default=Path("data/processed/tier0_revised_session1.json"),
    )
    parser.add_argument(
        "--session2-results",
        type=Path,
        default=Path("data/processed/tier0_revised_session2.json"),
    )
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--target-fp", type=float, default=2.0)
    parser.add_argument("--decision-step", type=float, default=0.1)
    parser.add_argument("--shifts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cumulative_iti_calibration.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    all_records = []

    for session in args.sessions:
        result_path = (
            args.session1_results if session == 1 else args.session2_results
        )
        thresholds = select_thresholds(result_path, session, args.target_fp)
        paths = sorted(
            (args.cache_root / f"session{session}").glob(
                "sub-*_tier0_features.npz"
            )
        )
        for path in paths:
            cache = load_cache(path)
            subject = int(cache["subject"])
            channels = tuple(str(value) for value in cache["channels"])
            times = cache["continuous_times"]
            features = cache["continuous_features"]
            command = np.asarray(cache["command_intervals"])
            iti = np.asarray(cache["iti_intervals"])
            fixation = np.asarray(cache["fixation_intervals"])
            trial_iti = []
            for trial_index in range(len(command)):
                next_fixation = (
                    fixation[trial_index + 1, 0]
                    if trial_index + 1 < len(fixation)
                    else np.inf
                )
                matched = iti[
                    (iti[:, 0] >= command[trial_index, 1])
                    & (iti[:, 1] <= next_fixation)
                ]
                trial_iti.append(matched[:1])
            completed_idle_features: list[np.ndarray] = []
            completed_idle_seconds = 0.0
            pre_step = float(np.median(np.diff(cache["pre_times"])))
            pre_seconds = len(cache["pre_features"]) * pre_step

            for trial_index in range(len(command)):
                calibration_values = np.concatenate(
                    [cache["pre_features"], *completed_idle_features], axis=0
                )
                cycle_start = fixation[trial_index, 0]
                cycle_stop = (
                    fixation[trial_index + 1, 0]
                    if trial_index + 1 < len(fixation)
                    else command[trial_index, 1]
                )
                keep = (times >= cycle_start) & (times < cycle_stop)
                cycle_times = times[keep]
                trial_command = command[trial_index : trial_index + 1]
                trial_idle = (
                    trial_iti[trial_index]
                )
                for condition, values in (
                    ("pre_rest_only", cache["pre_features"]),
                    ("pre_rest_plus_past_iti", calibration_values),
                ):
                    normalizer = LocationScale.fit(values, axis=0)
                    scored_times, cycle_scores = decision_subsample(
                        cycle_times,
                        tier0_from_features(features[keep], channels, normalizer),
                        args.decision_step,
                    )
                    decision = moving_average_dwell(
                        cycle_scores,
                        thresholds[subject],
                        args.decision_step,
                        ma_seconds=0.5,
                        dwell_seconds=0.3,
                        refractory_seconds=1.0,
                    )
                    detected, false_alarms, idle_seconds = interval_metric(
                        scored_times,
                        decision.alarms,
                        trial_idle,
                        trial_command,
                    )
                    null_detected = np.zeros(args.shifts, dtype=np.int8)
                    if len(decision.alarms):
                        for shift_index, offset in enumerate(
                            rng.integers(1, len(decision.alarms), size=args.shifts)
                        ):
                            shifted = np.roll(decision.alarms, int(offset))
                            null_detected[shift_index] = interval_metric(
                                scored_times,
                                shifted,
                                trial_idle,
                                trial_command,
                            )[0]
                    all_records.append(
                        {
                            "subject": subject,
                            "session": session,
                            "condition": condition,
                            "trial": trial_index + 1,
                            "completed_iti_count": trial_index,
                            "available_idle_seconds": (
                                pre_seconds + completed_idle_seconds
                            ),
                            "calibration_seconds": (
                                pre_seconds
                                if condition == "pre_rest_only"
                                else pre_seconds + completed_idle_seconds
                            ),
                            "detected": detected,
                            "false_alarms": false_alarms,
                            "idle_seconds": idle_seconds,
                            "_null_detected": null_detected,
                        }
                    )
                if len(trial_iti[trial_index]):
                    idle_mask = sample_mask(
                        times, trial_iti[trial_index].tolist()
                    )
                    completed_idle_features.append(features[idle_mask])
                    completed_idle_seconds += float(
                        trial_iti[trial_index][0, 1]
                        - trial_iti[trial_index][0, 0]
                    )

    duration_edges = [0, 90, 150, 210, 270, 330, 390, 450, 510, np.inf]
    duration_curve = []
    for session in args.sessions:
        for lower, upper in zip(duration_edges[:-1], duration_edges[1:]):
            selected = [
                row
                for row in all_records
                if row["session"] == session
                and row["condition"] == "pre_rest_plus_past_iti"
                and lower <= row["calibration_seconds"] < upper
            ]
            if selected:
                duration_curve.append(
                    {
                        "session": session,
                        "calibration_seconds_bin": [
                            lower,
                            None if np.isinf(upper) else upper,
                        ],
                        **summarize(selected, args.shifts),
                    }
                )
    trial_curve = []
    for session in args.sessions:
        for condition in ("pre_rest_only", "pre_rest_plus_past_iti"):
            for trial in range(1, 101):
                selected = [
                    row
                    for row in all_records
                    if row["session"] == session
                    and row["condition"] == condition
                    and row["trial"] == trial
                ]
                trial_curve.append(
                    {
                        "session": session,
                        "condition": condition,
                        "trial": trial,
                        **summarize(selected, args.shifts),
                    }
                )
    trial_block_curve = []
    for session in args.sessions:
        for condition in ("pre_rest_only", "pre_rest_plus_past_iti"):
            for start in range(1, 101, 10):
                selected = [
                    row
                    for row in all_records
                    if row["session"] == session
                    and row["condition"] == condition
                    and start <= row["trial"] < start + 10
                ]
                trial_block_curve.append(
                    {
                        "session": session,
                        "condition": condition,
                        "trial_block": [start, min(100, start + 9)],
                        **summarize(selected, args.shifts),
                    }
                )
    serializable_records = [
        {key: value for key, value in row.items() if key != "_null_detected"}
        for row in all_records
    ]
    payload = {
        "protocol": {
            "target_fp": args.target_fp,
            "source_threshold_fixed": True,
            "initial_calibration": "train pre-rest only",
            "updates": "completed guard-applied ITI only; decision state reset per trial",
        },
        "duration_curve": duration_curve,
        "trial_curve": trial_curve,
        "trial_block_curve": trial_block_curve,
        "rows": serializable_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(duration_curve, indent=2))


if __name__ == "__main__":
    main()
