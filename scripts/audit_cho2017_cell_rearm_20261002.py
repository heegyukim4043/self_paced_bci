"""Fixed-threshold boundary audit on 50 Cho2017 Tier-0 short-cell caches.

The historical LOSO source thresholds are kept fixed. This isolates the
target-side decision rule but is not a re-tuned deployment comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from analyze_cho2017_transfer import (
    STEP, load_cache, subject_sequences,
)
from uncued_mi.calibration import LocationScale
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from uncued_mi.metrics import evaluate_alarms
from uncued_mi.operating_point import aggregate_metrics


def evaluate_rearmed(sequences, threshold: float):
    rows = []
    for item in sequences:
        allowed = np.ones(len(item.scores), bool)
        ma = guard_local_moving_average(item.scores, allowed, ma_steps=5)
        alarms = guard_rearmed_alarms(
            ma, threshold, allowed, dwell_steps=3, refractory_steps=10,
        )
        rows.append(evaluate_alarms(
            item.times, alarms, item.idle_intervals, item.command_intervals,
        ))
    return aggregate_metrics(rows)


def run(cache_root: Path, historical: Path) -> dict:
    old = json.loads(historical.read_text(encoding="utf-8"))
    paths = sorted(cache_root.glob("sub-*_tier0_features.npz"))
    caches = {int(load_cache(path)["subject"]): load_cache(path) for path in paths}
    if len(caches) != 50:
        raise ValueError(f"expected 50 subjects, found {len(caches)}")
    selected = [row for row in old["rows"] if float(row["target_fp_setting"]) == 2.0]
    if len(selected) != 100:
        raise ValueError(f"expected 100 historical FP2 rows, found {len(selected)}")
    result_rows = []
    for subject in sorted(caches):
        own = LocationScale.fit(caches[subject]["rest_calibration_features"], axis=0)
        other = [value for key, value in caches.items() if key != subject]
        pooled = LocationScale.fit(np.concatenate(
            [value["rest_calibration_features"] for value in other], axis=0,
        ), axis=0)
        models = {"no_calibration": pooled, "idle_feature_alignment": own}
        for calibration, model in models.items():
            old_row, = [row for row in selected if int(row["subject"]) == subject
                        and row["calibration"] == calibration]
            threshold = float(old_row["operating_point"]["threshold"])
            rest, trials, _ = subject_sequences(caches[subject], model)
            rest_metrics = evaluate_rearmed([rest], threshold)
            command_metrics = evaluate_rearmed(trials, threshold)
            result_rows.append({
                "subject": subject, "calibration": calibration,
                "threshold": threshold,
                "historical_rest_fp_per_min": old_row["rest"]["fp_per_minute"],
                "rearmed_rest_fp_per_min": rest_metrics.fp_per_minute,
                "historical_command_tpr": old_row["command"]["true_positive_rate"],
                "rearmed_command_tpr": command_metrics.true_positive_rate,
                "historical_rest_alarms": old_row["rest"]["false_alarms"],
                "rearmed_rest_alarms": rest_metrics.false_alarms,
                "historical_detected_events": old_row["command"]["detected_events"],
                "rearmed_detected_events": command_metrics.detected_events,
            })
        print(f"done subject={subject}", flush=True)
    summary = {}
    for calibration in models:
        subset = [r for r in result_rows if r["calibration"] == calibration]
        summary[calibration] = {
            "subjects": len(subset),
            "historical_rest_fp_median": float(np.median([
                r["historical_rest_fp_per_min"] for r in subset])),
            "rearmed_rest_fp_median": float(np.median([
                r["rearmed_rest_fp_per_min"] for r in subset])),
            "historical_command_tpr_median": float(np.median([
                r["historical_command_tpr"] for r in subset])),
            "rearmed_command_tpr_median": float(np.median([
                r["rearmed_command_tpr"] for r in subset])),
            "rest_alarm_count_reduced_subjects": int(np.sum([
                r["rearmed_rest_alarms"] < r["historical_rest_alarms"] for r in subset])),
            "command_hits_reduced_subjects": int(np.sum([
                r["rearmed_detected_events"] < r["historical_detected_events"]
                for r in subset])),
        }
    return {
        "protocol": {
            "scope": "50 subjects, FP2 historical source threshold fixed, Tier-0 score",
            "decision": "7-s cells; old MA5/dwell3 vs rearmed MA5/dwell3, refractory10",
            "claim_limit": "short-cell boundary audit; not a continuous rare-FP or self-initiated MI test",
        },
        "summary": summary, "rows": result_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/cho2017_tier0"))
    parser.add_argument("--historical", type=Path, default=Path("data/processed/cho2017_transfer.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/cho2017_cell_rearm_audit_20261002.json"))
    args = parser.parse_args()
    payload = run(args.cache_root, args.historical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
