"""Subject-cluster summary of a diagnostic matched-FP rest-only sham run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 replicates: int = 10000) -> list[float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("expected one value per subject")
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = np.median(values[indices], axis=1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.input.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(args.input.with_suffix(".npz"), allow_pickle=False) as data:
        counts = data["matched_fp_count"].astype(float)
        observed = data["observed_hits2"].astype(float)
        sham = data["sham_hits2"].astype(float)
    if counts.shape != observed.shape or counts.shape != sham.shape:
        raise AssertionError("result arrays do not align")
    provenance = metadata["provenance"]
    rng = np.random.default_rng(20261003)
    rows = []
    summary = {}
    for session in (1, 2):
        session_rows = []
        for subject in range(1, counts.shape[1] + 1):
            slot = (session - 1, subject - 1)
            keep = counts[slot] >= 0
            if not np.any(keep):
                session_rows.append({"subject": subject, "session": session,
                                     "matched_draws": 0, "excess_median": None})
                continue
            count = counts[slot][keep]
            obs = observed[slot][keep] / 30.
            null = sham[slot][keep] / 30.
            paired = obs - null
            source = next(x for x in provenance if x["subject"] == subject
                          and x["session"] == session)
            row = {
                "subject": subject, "session": session,
                "matched_draws": int(keep.sum()),
                "desired_fp_count": source["desired_false_alarm_count"],
                "matched_fp_count_median": float(np.median(count)),
                "target_count_abs_error_median": float(np.median(
                    np.abs(count - source["desired_false_alarm_count"]))),
                "observed_sen2_median": float(np.median(obs)),
                "null_sen2_median": float(np.median(null)),
                "excess_median": float(np.median(paired)),
                "excess_mean": float(np.mean(paired)),
                "positive_difference_fraction": float(np.mean(paired > 0.)),
            }
            session_rows.append(row)
            rows.append(row)
        complete = [x for x in session_rows if x["excess_median"] is not None]
        excess = np.asarray([x["excess_median"] for x in complete])
        summary[str(session)] = {
            "matched_subjects": len(complete),
            "subjects_total": len(session_rows),
            "matched_draws_total": sum(x["matched_draws"] for x in session_rows),
            "unmatched_draws_total": int(np.sum(counts[session - 1] < 0)),
            "matched_fp_count_median_across_subjects": float(np.median([
                x["matched_fp_count_median"] for x in complete])),
            "target_count_abs_error_median_across_subjects": float(np.median([
                x["target_count_abs_error_median"] for x in complete])),
            "observed_sen2_median_across_subjects": float(np.median([
                x["observed_sen2_median"] for x in complete])),
            "null_sen2_median_across_subjects": float(np.median([
                x["null_sen2_median"] for x in complete])),
            "paired_excess_median_across_subjects": float(np.median(excess)),
            "paired_excess_median_subject_bootstrap95": bootstrap_ci(excess, rng),
            "paired_excess_mean_across_subjects": float(np.mean(excess)),
            "subjects_excess_positive": int(np.sum(excess > 0.)),
            "subjects_excess_zero": int(np.sum(excess == 0.)),
            "subjects_excess_negative": int(np.sum(excess < 0.)),
        }
    payload = {
        "status": "diagnostic matched-achieved-FP sham; not deployable and not MI-specific",
        "input": str(args.input),
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
