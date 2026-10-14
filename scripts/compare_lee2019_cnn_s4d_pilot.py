"""Paired frozen-protocol comparison of CNN-S4D and CNN-GRU pilot rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_median(values, seed, repeats=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s4d", type=Path,
        default=Path("data/processed/lee2019_cnn_s4d_pilot5.json"),
    )
    parser.add_argument(
        "--gru", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_prospective_anchor_subset20.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_s4d_pilot5_paired.json"),
    )
    args = parser.parse_args()
    s4d = json.loads(args.s4d.read_text(encoding="utf-8"))
    gru = json.loads(args.gru.read_text(encoding="utf-8"))
    gru_map = {(row["subject"], row["session"]): row for row in gru["rows"]}
    rows = []
    for row in s4d["rows"]:
        key = (row["subject"], row["session"])
        reference = gru_map[key]
        comparison = {"subject": key[0], "session": key[1], "metrics": {}}
        for mode in ("common_fp", "transferred"):
            comparison["metrics"][mode] = {}
            for metric in ("fp", "raw_tpr_1s", "null_tpr_1s", "excess_1s", "delay"):
                new = float(row["metrics"][mode][metric])
                old = float(reference["metrics"][mode][metric])
                comparison["metrics"][mode][metric] = {
                    "s4d": new, "gru": old, "delta": new - old,
                }
        rows.append(comparison)

    summary = {}
    for session in (1, 2):
        selected = [row for row in rows if row["session"] == session]
        item = {"n": len(selected), "modes": {}}
        for mode_index, mode in enumerate(("common_fp", "transferred")):
            item["modes"][mode] = {}
            for metric_index, metric in enumerate(
                ("fp", "raw_tpr_1s", "null_tpr_1s", "excess_1s", "delay")
            ):
                old = [row["metrics"][mode][metric]["gru"] for row in selected]
                new = [row["metrics"][mode][metric]["s4d"] for row in selected]
                delta = np.asarray(new) - np.asarray(old)
                item["modes"][mode][metric] = {
                    "gru_median": float(np.median(old)),
                    "s4d_median": float(np.median(new)),
                    "paired_delta_median": float(np.median(delta)),
                    "paired_delta_ci95": bootstrap_median(
                        delta, 202609130 + session * 1000 + mode_index * 100 + metric_index
                    ),
                }
        summary[f"session{session}"] = item
    payload = {
        "protocol": "same five subjects; exact CNN/training/evaluation protocol",
        "promotion_gate": (
            "common-FP excess paired CI lower bound >0 in both sessions; "
            "then test idle/MI-ratio robustness"
        ),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
