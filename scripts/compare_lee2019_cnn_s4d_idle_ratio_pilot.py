"""Paired idle/MI-ratio robustness comparison for CNN-S4D versus CNN-GRU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RATIOS = (0.75, 1.5, 3.0)
PILOT = {3, 10, 20, 25, 54}


def ci(values, seed, repeats=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975]).tolist()


def indexed(rows):
    output = {}
    for row in rows:
        if row["subject"] not in PILOT:
            continue
        for condition in row["conditions"]:
            output[(row["subject"], row["session"], condition["repeat"], condition["idle_mi_ratio"])] = condition
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s4d", type=Path,
        default=Path("data/processed/lee2019_cnn_s4d_idle_ratio_pilot5.json"),
    )
    parser.add_argument(
        "--gru", type=Path,
        default=Path("data/processed/lee2019_cnn_gru_eval_idle_mi_ratio_subset20.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_s4d_idle_ratio_pilot5_paired.json"),
    )
    args = parser.parse_args()
    s4d = indexed(json.loads(args.s4d.read_text(encoding="utf-8"))["rows"])
    gru = indexed(json.loads(args.gru.read_text(encoding="utf-8"))["rows"])
    keys = sorted({key[:3] for key in s4d} & {key[:3] for key in gru})
    rows = []
    for subject, session, repeat in keys:
        item = {"subject": subject, "session": session, "repeat": repeat}
        for model, source in (("s4d", s4d), ("gru", gru)):
            conditions = [source[(subject, session, repeat, ratio)] for ratio in RATIOS]
            fp = np.asarray([c["metrics"]["transferred"]["fp"] for c in conditions])
            common_excess = np.asarray(
                [c["metrics"]["common_fp"]["excess_1s"] for c in conditions]
            )
            item[model] = {
                "transferred_fp_span": float(np.max(fp) - np.min(fp)),
                "transferred_fp_mean": float(np.mean(fp)),
                "common_excess_span": float(np.max(common_excess) - np.min(common_excess)),
                "common_excess_mean": float(np.mean(common_excess)),
            }
        rows.append(item)

    summary = {}
    for session in (1, 2):
        selected = [row for row in rows if row["session"] == session]
        summary[str(session)] = {"n_subject_repeat": len(selected)}
        for metric_index, metric in enumerate(
            ("transferred_fp_span", "transferred_fp_mean", "common_excess_span", "common_excess_mean")
        ):
            old = np.asarray([row["gru"][metric] for row in selected])
            new = np.asarray([row["s4d"][metric] for row in selected])
            delta = new - old
            summary[str(session)][metric] = {
                "gru_median": float(np.median(old)),
                "s4d_median": float(np.median(new)),
                "paired_delta_median": float(np.median(delta)),
                "paired_delta_ci95": ci(delta, 202609140 + session * 100 + metric_index),
            }
    payload = {
        "metric_definition": {
            "transferred_fp_span": "max ratio FP/min minus min ratio FP/min within paired composition",
            "common_excess_span": "max ratio excess minus min ratio excess within paired composition",
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
