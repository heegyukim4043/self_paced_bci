"""Paired comparison of CNN-selective-SSM and CNN-GRU pilot results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap(values, seed, repeats=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, default=Path("data/processed/lee2019_cnn_selective_ssm_pilot5.json"))
    parser.add_argument("--gru", type=Path, default=Path("data/processed/lee2019_cnn_gru_prospective_anchor_subset20.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lee2019_cnn_selective_ssm_pilot5_paired.json"))
    args = parser.parse_args()
    new = json.loads(args.new.read_text(encoding="utf-8"))["rows"]
    old = {(r["subject"], r["session"]): r for r in json.loads(args.gru.read_text(encoding="utf-8"))["rows"]}
    rows = []
    for row in new:
        ref = old[(row["subject"], row["session"])]
        item = {"subject": row["subject"], "session": row["session"], "metrics": {}}
        for mode in ("common_fp", "transferred"):
            item["metrics"][mode] = {}
            for metric in ("fp", "raw_tpr_1s", "null_tpr_1s", "excess_1s", "delay"):
                a = float(row["metrics"][mode][metric]); b = float(ref["metrics"][mode][metric])
                item["metrics"][mode][metric] = {"selective_ssm": a, "gru": b, "delta": a - b}
        rows.append(item)
    summary = {}
    for session in (1, 2):
        selected = [r for r in rows if r["session"] == session]
        modes = {}
        for mi, mode in enumerate(("common_fp", "transferred")):
            modes[mode] = {}
            for ji, metric in enumerate(("fp", "raw_tpr_1s", "null_tpr_1s", "excess_1s", "delay")):
                delta = np.asarray([r["metrics"][mode][metric]["delta"] for r in selected])
                modes[mode][metric] = {
                    "gru_median": float(np.median([r["metrics"][mode][metric]["gru"] for r in selected])),
                    "selective_ssm_median": float(np.median([r["metrics"][mode][metric]["selective_ssm"] for r in selected])),
                    "paired_delta_median": float(np.median(delta)),
                    "paired_delta_ci95": bootstrap(delta, 202609150 + session * 100 + mi * 10 + ji),
                }
        summary[str(session)] = {"n": len(selected), "modes": modes}
    payload = {
        "protocol": "same five subjects x two sessions; exact CNN/training/evaluation protocol",
        "promotion_gate": "common-FP excess paired CI lower bound >0 in both sessions",
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
