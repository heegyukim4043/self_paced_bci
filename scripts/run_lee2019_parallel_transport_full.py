"""Full paired evaluation of the pilot-selected 30-s transport+EWMA arm."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lee2019_parallel_transport_smoke import analyze


def bootstrap(values, repeats=10_000, seed=20260814):
    """Subject-cluster bootstrap over paired session rows."""
    rng = np.random.default_rng(seed)
    by_subject = {subject: np.asarray([value for s, value in values if s == subject]) for subject in range(1, 55)}
    draws = []
    for _ in range(repeats):
        subjects = rng.integers(1, 55, 54)
        sample = np.concatenate([by_subject[int(subject)] for subject in subjects])
        draws.append(float(np.median(sample)))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main():
    pairs = [(session, subject) for session in (1, 2) for subject in range(1, 55)]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(analyze, subject, session, (30.0,), False): (session, subject)
            for session, subject in pairs
        }
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(f"done session{row['session']} sub-{row['subject']:02d} ({len(rows)}/108)", flush=True)
    rows.sort(key=lambda row: (row["session"], row["subject"]))
    base = "none_plus_ewma"; transport = "transport_30s_plus_ewma"
    excess_delta = [(row["subject"], row["arms"][transport]["excess_sensitivity_2s"] - row["arms"][base]["excess_sensitivity_2s"]) for row in rows]
    flat_delta = [(row["subject"], row["arms"][transport]["screen_flat_fraction"] - row["arms"][base]["screen_flat_fraction"]) for row in rows]
    fp_delta = [(row["subject"], row["arms"][transport]["actual_fp_per_min"] - row["arms"][base]["actual_fp_per_min"]) for row in rows]
    summary = {
        "n_subject_sessions": len(rows),
        "baseline": {
            key: float(np.median([row["arms"][base][key] for row in rows]))
            for key in ("screen_flat_fraction", "actual_fp_per_min", "excess_sensitivity_2s")
        },
        "transport_30s_plus_ewma": {
            key: float(np.median([row["arms"][transport][key] for row in rows]))
            for key in ("screen_flat_fraction", "actual_fp_per_min", "excess_sensitivity_2s")
        },
        "paired": {
            "excess_delta_median": float(np.median([v for _, v in excess_delta])),
            "excess_delta_subject_cluster_ci": bootstrap(excess_delta),
            "flat_delta_median": float(np.median([v for _, v in flat_delta])),
            "flat_delta_subject_cluster_ci": bootstrap(flat_delta, seed=20260815),
            "fp_delta_median": float(np.median([v for _, v in fp_delta])),
            "fp_delta_subject_cluster_ci": bootstrap(fp_delta, seed=20260816),
            "excess_improved_n": int(sum(v > 0 for _, v in excess_delta)),
            "excess_worsened_n": int(sum(v < 0 for _, v in excess_delta)),
            "flat_improved_n": int(sum(v < 0 for _, v in flat_delta)),
            "flat_worsened_n": int(sum(v > 0 for _, v in flat_delta)),
        },
    }
    payload = {
        "protocol": {
            "selection": "30-s transport+EWMA chosen after fixed random 10-session pilot; 5/10 s stopped for FP-budget violation",
            "update": "causal baseline-inactive covariance buffer; Riemannian mean every 1 s; current window scored before update",
            "comparison": "paired against LOSO Arm A none+EWMA; fixed nominal FP4 event threshold and FP2 screening threshold",
            "inference": "subject-cluster bootstrap across two sessions",
        },
        "summary": summary,
        "rows": rows,
    }
    output = Path("data/processed/lee2019_parallel_transport_full.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
