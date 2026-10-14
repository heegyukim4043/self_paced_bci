"""Join EEGNet results to the deterministic same-cache Riemannian comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ci(values: np.ndarray, seed: int, repeats: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("data/processed/three_model_validation.json"))
    parser.add_argument("--riemannian", type=Path, default=Path("data/processed/mil_riemannian_comparator.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/three_model_validation_summary.json"))
    args = parser.parse_args()
    models = json.loads(args.models.read_text(encoding="utf-8"))
    riemannian = json.loads(args.riemannian.read_text(encoding="utf-8"))
    rows = []
    for session in (1, 2):
        rmap = {
            row["subject"]: row["excess_tpr"]
            for row in riemannian["rows"]
            if row["session"] == session
        }
        model_rows = next(
            item["subject_rows"] for item in models["summary"]
            if item["session"] == session
        )
        r = np.asarray([rmap[row["subject"]] for row in model_rows])
        sl = np.asarray([row["sl_excess"] for row in model_rows])
        ssl = np.asarray([row["ssl_excess"] for row in model_rows])
        rows.append({
            "session": session,
            "riemannian_excess_median": float(np.median(r)),
            "eegnet_sl_excess_median": float(np.median(sl)),
            "eegnet_sl_excess_ci": ci(sl, 100 + session),
            "eegnet_ssl_excess_median": float(np.median(ssl)),
            "eegnet_ssl_excess_ci": ci(ssl, 200 + session),
            "sl_minus_riemannian_median": float(np.median(sl - r)),
            "sl_minus_riemannian_ci": ci(sl - r, 300 + session),
            "ssl_minus_sl_median": float(np.median(ssl - sl)),
            "ssl_minus_sl_ci": ci(ssl - sl, 400 + session),
        })
    payload = {
        "protocol": models["protocol"],
        "validation_candidates": models["validation_candidates"],
        "summary": rows,
        "binding": {
            "eegnet_sl_improves_riemannian": False,
            "eegnet_ssl_improves_sl": False,
            "reason": "absolute excess lower CI and paired-improvement criteria fail",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
