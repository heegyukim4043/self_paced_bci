"""Causal CNN selective-SSM (Mamba-style) pilot.

The CNN front-end and the full evaluation protocol are inherited from the
CNN-GRU runner.  The temporal head uses input-dependent continuous-time
decay (selective forgetting) and input gating, followed by a causal diagonal
state scan.  This is a local Mamba-style reference implementation, not the
official ``mamba_ssm`` package or a claim of reproducing Mamba-1/2/3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_lee2019_cnn_gru import CausalCNNGRU, analyze, summarize
from scripts.run_lee2019_stateful_gru import SEEDS


class SelectiveDiagonalSSMLayer(nn.Module):
    """Input-selective causal diagonal state-space layer.

    The current input controls both the effective time step/forgetting rate
    and the amount written into the state.  State output is then projected by
    fixed diagonal readout coefficients.
    """

    def __init__(self, width: int, state_size: int = 8, dropout: float = 0.1):
        super().__init__()
        tau = torch.logspace(np.log10(5.0), np.log10(300.0), state_size)
        self.log_tau = nn.Parameter(tau.log()[None].repeat(width, 1))
        self.input_gain = nn.Parameter(torch.ones(width, state_size))
        self.output_gain = nn.Parameter(
            torch.randn(width, state_size) / np.sqrt(state_size)
        )
        self.skip = nn.Parameter(torch.ones(width))
        # Zero-initialized projections recover the non-selective baseline.
        self.delta_projection = nn.Linear(width, width)
        self.gate_projection = nn.Linear(width, width)
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.constant_(self.delta_projection.bias, 0.0)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)
        self.mix = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor, state=None):
        batch, length, width = values.shape
        tau = self.log_tau.exp().clamp(1.0, 10_000.0)
        if state is None:
            state = values.new_zeros(batch, width, tau.shape[1])
        outputs = []
        for index in range(length):
            current = values[:, index]
            # Positive input-dependent delta controls selective forgetting.
            delta = torch.nn.functional.softplus(
                self.delta_projection(current) + 0.54132485
            )
            decay = torch.exp(-delta[..., None] / tau[None])
            write_gate = 2.0 * torch.sigmoid(self.gate_projection(current))
            drive = (1.0 - decay) * self.input_gain[None] * current[..., None]
            state = decay * state + write_gate[..., None] * drive
            readout = torch.sum(state * self.output_gain[None], dim=-1)
            outputs.append(readout + self.skip[None] * current)
        output = torch.stack(outputs, dim=1)
        transformed = self.mix(torch.nn.functional.gelu(output))
        return self.norm(values + self.dropout(transformed)), state


class CausalCNNSelectiveSSM(CausalCNNGRU):
    """Same CNN encoder with one parameter-matched selective SSM head."""

    def __init__(self, channels: int = 20, hidden: int = 32):
        super().__init__(channels, hidden)
        del self.gru
        self.input_projection = nn.Linear(24, hidden)
        self.ssm = SelectiveDiagonalSSMLayer(hidden, state_size=8)

    def forward(self, values: torch.Tensor, hidden=None):
        batch, steps = values.shape[:2]
        encoded = self.encode(values.reshape(-1, *values.shape[2:]))
        sequence = self.input_projection(encoded.reshape(batch, steps, -1))
        state = None if hidden is None else hidden[0]
        sequence, state = self.ssm(sequence, state)
        return self.output(sequence).squeeze(-1), (state,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/lee2019_mi"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/interim/lee2019_mil_raw"))
    parser.add_argument("--subjects", type=int, nargs="+", default=[3, 10, 20, 25, 54])
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shifts", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--score-cache-root", type=Path,
        default=Path("data/interim/lee2019_cnn_selective_ssm_scores_pilot5"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_selective_ssm_pilot5.json"),
    )
    args = parser.parse_args()
    parameter_counts = {
        "cnn_gru": int(sum(p.numel() for p in CausalCNNGRU(20, 32).parameters())),
        "cnn_selective_ssm": int(sum(p.numel() for p in CausalCNNSelectiveSSM(20, 32).parameters())),
    }
    print(f"parameters: {parameter_counts}", flush=True)
    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            print(f"CNN-selective-SSM S{session} sub-{subject:02d}", flush=True)
            rows.append(
                analyze(
                    args.data_root, args.cache_root, subject, session,
                    args.epochs, tuple(args.seeds), args.device, args.shifts,
                    score_cache_root=args.score_cache_root,
                    model_class=CausalCNNSelectiveSSM,
                )
            )
    payload = {
        "protocol": {
            "comparison": "same CNN/training/evaluation; GRU replaced by selective SSM",
            "head": "one-layer causal diagonal selective SSM",
            "selection": "input-dependent delta/forgetting and write gate",
            "causal": True,
            "bidirectional": False,
            "claim_limit": "local Mamba-style reference, not official mamba_ssm reproduction",
        },
        "parameter_counts": parameter_counts,
        "summary": summarize(rows, args.sessions, args.bootstrap),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
