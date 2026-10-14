"""Run preregistered validation and locked HSMM decision-layer comparison."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uncued_mi.decision import instantaneous, moving_average_dwell
from uncued_mi.hsmm import GaussianEmission, causal_two_state_hsmm
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.mil import SEEDS
from uncued_mi.operating_point import ScoredSequence, tune_source_operating_points

SUBSET20 = (
    2, 3, 9, 10, 12, 13, 14, 17, 19, 20,
    23, 25, 26, 29, 38, 40, 42, 51, 53, 54,
)
STEP_SECONDS = 0.5
REFRACTORY_SECONDS = 1.0
ENTRY_PROBABILITY = 0.01
MINIMUM_SECONDS = (0.5, 1.0)
EXPECTED_SECONDS = (0.5, 1.0, 1.5)
MAXIMUM_SECONDS = 1.5


def intervals_for_trials(
    command: np.ndarray,
    iti: np.ndarray,
    iti_trials: np.ndarray,
    low: int,
    high: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        iti[(iti_trials >= low) & (iti_trials < high)],
        command[low:high],
    )


def emission_from_array(values: np.ndarray) -> GaussianEmission:
    return GaussianEmission(*[float(value) for value in values])


def candidate_grid() -> list[dict]:
    candidates = []
    for minimum in MINIMUM_SECONDS:
        for expected in EXPECTED_SECONDS:
            if expected + 1e-9 < minimum:
                continue
            candidates.append(
                {
                    "minimum_seconds": minimum,
                    "expected_seconds": expected,
                    "maximum_seconds": MAXIMUM_SECONDS,
                }
            )
    return candidates


def hsmm_statistic(scores: np.ndarray, emission: GaussianEmission, candidate: dict) -> np.ndarray:
    return causal_two_state_hsmm(
        scores,
        emission,
        minimum_mi_steps=int(round(candidate["minimum_seconds"] / STEP_SECONDS)),
        expected_mi_steps=candidate["expected_seconds"] / STEP_SECONDS,
        maximum_mi_steps=int(round(candidate["maximum_seconds"] / STEP_SECONDS)),
        entry_probability=ENTRY_PROBABILITY,
    )


def threshold_and_alarms(
    times: np.ndarray,
    statistic: np.ndarray,
    idle: np.ndarray,
    command: np.ndarray,
    decision: str,
) -> tuple[float, np.ndarray, dict]:
    sequence = ScoredSequence(times, statistic, idle, command)
    if decision == "hsmm":
        point = tune_source_operating_points(
            [sequence],
            "threshold",
            [2.0],
            STEP_SECONDS,
            refractory_seconds=REFRACTORY_SECONDS,
            grid_size=160,
        )[2.0]
        result = instantaneous(
            statistic,
            point.threshold,
            STEP_SECONDS,
            REFRACTORY_SECONDS,
        )
    elif decision == "ma_dwell":
        point = tune_source_operating_points(
            [sequence],
            "ma_dwell",
            [2.0],
            STEP_SECONDS,
            ma_seconds=0.5,
            dwell_seconds=0.3,
            refractory_seconds=REFRACTORY_SECONDS,
            grid_size=160,
        )[2.0]
        result = moving_average_dwell(
            statistic,
            point.threshold,
            STEP_SECONDS,
            ma_seconds=0.5,
            dwell_seconds=0.3,
            refractory_seconds=REFRACTORY_SECONDS,
        )
    else:
        raise ValueError(decision)
    observed = evaluate_alarms(times, result.alarms, idle, command)
    return point.threshold, result.alarms, asdict(observed)


def null_adjust(
    times: np.ndarray,
    alarms: np.ndarray,
    idle: np.ndarray,
    command: np.ndarray,
    *,
    shifts: int,
    seed: int,
) -> tuple[float, float, list[int]]:
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, len(alarms), size=shifts)
    null = np.asarray(
        [
            evaluate_alarms(times, np.roll(alarms, int(offset)), idle, command)
            .true_positive_rate
            for offset in offsets
        ]
    )
    observed = evaluate_alarms(times, alarms, idle, command).true_positive_rate
    return float(np.median(null)), float(observed - np.median(null)), offsets.tolist()


def evaluate_one(
    path: Path,
    session: int,
    subject: int,
    low: int,
    high: int,
    candidate: dict,
    shifts: int,
) -> dict:
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=float)
        scores = np.asarray(data["scores"], dtype=float)
        idle, command = intervals_for_trials(
            np.asarray(data["command_intervals"], dtype=float),
            np.asarray(data["iti_intervals"], dtype=float),
            np.asarray(data["iti_trials"], dtype=int),
            low,
            high,
        )
        emission = emission_from_array(data["emission"])

    hsmm_scores = hsmm_statistic(scores, emission, candidate)
    hsmm_threshold, hsmm_alarms, hsmm_observed = threshold_and_alarms(
        times, hsmm_scores, idle, command, "hsmm"
    )
    ma_threshold, ma_alarms, ma_observed = threshold_and_alarms(
        times, scores, idle, command, "ma_dwell"
    )
    seed = SEEDS[0] + session * 1000 + subject
    hsmm_null, hsmm_excess, offsets = null_adjust(
        times, hsmm_alarms, idle, command, shifts=shifts, seed=seed
    )
    # Use identical circular offsets for the paired comparator.
    ma_null_values = np.asarray(
        [
            evaluate_alarms(times, np.roll(ma_alarms, int(offset)), idle, command)
            .true_positive_rate
            for offset in offsets
        ]
    )
    ma_null = float(np.median(ma_null_values))
    ma_excess = float(ma_observed["true_positive_rate"] - ma_null)
    return {
        "session": session,
        "subject": subject,
        "candidate": candidate,
        "hsmm": {
            "threshold": hsmm_threshold,
            **hsmm_observed,
            "null_tpr_median": hsmm_null,
            "excess_tpr": hsmm_excess,
        },
        "ma_dwell": {
            "threshold": ma_threshold,
            **ma_observed,
            "null_tpr_median": ma_null,
            "excess_tpr": ma_excess,
        },
        "paired_delta": hsmm_excess - ma_excess,
    }


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repeats, len(values)))]
    return [
        float(value)
        for value in np.quantile(np.median(samples, axis=1), [0.025, 0.975])
    ]


def summarize(rows: list[dict], seed: int, bootstrap: int) -> dict:
    hsmm = np.asarray([row["hsmm"]["excess_tpr"] for row in rows])
    ma = np.asarray([row["ma_dwell"]["excess_tpr"] for row in rows])
    delta = hsmm - ma
    return {
        "subjects": len(rows),
        "hsmm_fp_per_minute_median": float(
            np.median([row["hsmm"]["fp_per_minute"] for row in rows])
        ),
        "hsmm_observed_tpr_median": float(
            np.median([row["hsmm"]["true_positive_rate"] for row in rows])
        ),
        "hsmm_null_tpr_median": float(
            np.median([row["hsmm"]["null_tpr_median"] for row in rows])
        ),
        "hsmm_excess_tpr_median": float(np.median(hsmm)),
        "hsmm_excess_bootstrap_95_ci": bootstrap_ci(hsmm, seed, bootstrap),
        "ma_dwell_excess_tpr_median": float(np.median(ma)),
        "paired_delta_median": float(np.median(delta)),
        "paired_delta_bootstrap_95_ci": bootstrap_ci(
            delta, seed + 10_000, bootstrap
        ),
        "hsmm_latency_median": float(
            np.nanmedian([row["hsmm"]["median_delay_seconds"] for row in rows])
        ),
        "ma_dwell_latency_median": float(
            np.nanmedian([row["ma_dwell"]["median_delay_seconds"] for row in rows])
        ),
    }


def candidate_sort_key(item: dict) -> tuple:
    candidate = item["candidate"]
    return (
        -item["summary"]["hsmm_excess_tpr_median"],
        abs(candidate["minimum_seconds"] - 0.5),
        abs(candidate["expected_seconds"] - 1.0),
        candidate["expected_seconds"],
    )


def select_candidate(results: list[dict]) -> tuple[dict | None, dict]:
    admissible = [
        item
        for item in results
        if 1.5 <= item["summary"]["hsmm_fp_per_minute_median"] <= 2.5
    ]
    audit = {
        "candidate_count": len(results),
        "admissible_count": len(admissible),
        "calibration_failure": not admissible,
        "tie_break_used": False,
    }
    if not admissible:
        return None, audit
    best_excess = max(
        item["summary"]["hsmm_excess_tpr_median"] for item in admissible
    )
    tied = [
        item
        for item in admissible
        if best_excess - item["summary"]["hsmm_excess_tpr_median"] < 0.01
    ]
    audit["tie_break_used"] = len(tied) > 1
    selected = sorted(tied, key=candidate_sort_key)[0]
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path, default=Path("data/interim/hsmm_scores")
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBSET20)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/hsmm_gate.json"),
    )
    args = parser.parse_args()

    validation_results = []
    for candidate in candidate_grid():
        rows = []
        for subject in args.subjects:
            path = (
                args.score_root
                / "session1"
                / f"sub-{subject:02d}_train-40.npz"
            )
            rows.append(
                evaluate_one(
                    path, 1, subject, 40, 50, candidate, args.shifts
                )
            )
        validation_results.append(
            {
                "candidate": candidate,
                "summary": summarize(
                    rows, SEEDS[0] + 100, args.bootstrap
                ),
                "rows": rows,
            }
        )
        print(f"validated {candidate}", flush=True)

    selected, selection_audit = select_candidate(validation_results)
    if selected is None:
        payload = {
            "protocol": {"status": "calibration_failure"},
            "validation": validation_results,
            "selection_audit": selection_audit,
            "locked": [],
            "binding": {
                "passed": False,
                "reason": "No admissible validation candidate",
            },
        }
    else:
        chosen = selected["candidate"]
        locked_results = []
        session_passes = []
        for session in (1, 2):
            rows = []
            for subject in args.subjects:
                path = (
                    args.score_root
                    / f"session{session}"
                    / f"sub-{subject:02d}_train-50.npz"
                )
                rows.append(
                    evaluate_one(
                        path,
                        session,
                        subject,
                        50,
                        100,
                        chosen,
                        args.shifts,
                    )
                )
            summary = summarize(
                rows, SEEDS[0] + session * 100, args.bootstrap
            )
            absolute = summary["hsmm_excess_bootstrap_95_ci"][0] > 0.10
            paired = summary["paired_delta_bootstrap_95_ci"][0] > 0.0
            session_passes.append(absolute and paired)
            locked_results.append(
                {
                    "session": session,
                    "summary": summary,
                    "absolute_pass": absolute,
                    "paired_pass": paired,
                    "rows": rows,
                }
            )
            print(f"locked session={session}", flush=True)
        payload = {
            "protocol": {
                "selection_session": 1,
                "validation_trials": [40, 50],
                "locked_trials": [50, 100],
                "decision_step_seconds": STEP_SECONDS,
                "entry_probability": ENTRY_PROBABILITY,
                "fp_axis": "held-out diagnostic oracle at FP=2",
                "null_shifts": args.shifts,
                "bootstrap": args.bootstrap,
            },
            "validation": validation_results,
            "selection_audit": selection_audit,
            "selected_candidate": chosen,
            "locked": locked_results,
            "binding": {
                "passed": bool(all(session_passes)),
                "rule": (
                    "both sessions require HSMM excess CI lower >0.10 and "
                    "paired HSMM-MA+dwell CI lower >0"
                ),
                "session_passes": session_passes,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selection_audit": payload["selection_audit"],
                "selected_candidate": payload.get("selected_candidate"),
                "locked": [
                    {
                        "session": item["session"],
                        **item["summary"],
                        "absolute_pass": item["absolute_pass"],
                        "paired_pass": item["paired_pass"],
                    }
                    for item in payload["locked"]
                ],
                "binding": payload["binding"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
