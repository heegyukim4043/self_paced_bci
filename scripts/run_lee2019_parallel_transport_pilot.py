"""Fixed random 10-session pilot for causal parallel-transport adaptation."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lee2019_parallel_transport_smoke import analyze


def main():
    rng = np.random.default_rng(20260814)
    population = [(session, subject) for session in (1, 2) for subject in range(1, 55)]
    chosen = [population[index] for index in rng.choice(len(population), 10, replace=False)]
    rows = []
    # Two workers is intentionally conservative for a laptop CPU and memory.
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(analyze, subject, session): (session, subject) for session, subject in chosen}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"done session{row['session']} sub-{row['subject']:02d}", flush=True)
    rows.sort(key=lambda row: (row["session"], row["subject"]))
    baseline = np.asarray([row["arms"]["none_plus_ewma"]["excess_sensitivity_2s"] for row in rows])
    summary = {"n": len(rows), "baseline_excess_median": float(np.median(baseline)), "arms": {}}
    for arm in (f"transport_{window}s_plus_ewma" for window in (5, 10, 30)):
        excess = np.asarray([row["arms"][arm]["excess_sensitivity_2s"] for row in rows])
        flat = np.asarray([row["arms"][arm]["screen_flat_fraction"] for row in rows])
        baseline_flat = np.asarray([row["arms"]["none_plus_ewma"]["screen_flat_fraction"] for row in rows])
        fp = np.asarray([row["arms"][arm]["actual_fp_per_min"] for row in rows])
        summary["arms"][arm] = {
            "excess_median": float(np.median(excess)),
            "paired_excess_delta_median": float(np.median(excess - baseline)),
            "improved_n": int(np.sum(excess > baseline)),
            "worsened_n": int(np.sum(excess < baseline)),
            "flat_delta_median": float(np.median(flat - baseline_flat)),
            "actual_fp_median": float(np.median(fp)),
        }
    payload = {
        "protocol": {
            "selection": "10 subject-sessions sampled without replacement using seed 20260814 before pilot results",
            "workers": 2,
            "inference": "diagnostic stage gate only; no population claim",
        },
        "chosen": [{"session": session, "subject": subject} for session, subject in chosen],
        "summary": summary,
        "rows": rows,
    }
    output = Path("data/processed/lee2019_parallel_transport_pilot.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
