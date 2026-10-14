"""Secondary random idle/MI-ratio robustness pilot for CNN-S4D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_lee2019_cnn_gru_eval_idle_spacing import (
    IDLE_MI_RATIOS,
    analyze,
    summarize,
)
from scripts.run_lee2019_cnn_s4d import CausalCNNS4D
from scripts.run_lee2019_stateful_gru import SEEDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[3, 10, 20, 25, 54])
    parser.add_argument("--sessions", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_s4d_idle_ratio_pilot5.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            print(f"S4D idle-ratio S{session} sub-{subject:02d}", flush=True)
            rows.append(
                analyze(
                    subject, session, args.epochs, tuple(args.seeds), device,
                    args.shifts, args.repeats, model_class=CausalCNNS4D,
                )
            )
    payload = {
        "protocol": {
            "model_training": "causal CNN-S4D frozen across ratio conditions",
            "evaluation_idle_mi_ratios": IDLE_MI_RATIOS,
            "minimum_idle_seconds": 0.5,
            "random_compositions": args.repeats,
            "paired_trial_order": True,
            "secondary_only": True,
        },
        "summary": summarize(rows, args.bootstrap),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
