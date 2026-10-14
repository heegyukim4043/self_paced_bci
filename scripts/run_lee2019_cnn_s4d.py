"""Causal CNN-S4D-style pilot under the frozen CNN-GRU protocol.

The EEG filter bank, CNN encoder, sequence construction, loss, seeds, and
evaluation are inherited from ``run_lee2019_cnn_gru.py``.  Only the GRU head is
replaced by a stable real diagonal state-space head.  This is an S4D-style
ablation, not a claim of reproducing a particular external S4/S5 package.
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

from scripts.run_lee2019_cnn_gru import (
    CausalCNNGRU,
    analyze,
    summarize,
)
from scripts.run_lee2019_stateful_gru import SEEDS


class DiagonalSSMLayer(nn.Module):
    """Stable causal diagonal SSM with FFT training and recurrent state."""

    def __init__(self, width: int, state_size: int = 8, dropout: float = 0.1):
        super().__init__()
        # Initial discrete time constants span 0.5--30 s at a 0.1-s step.
        tau = torch.logspace(np.log10(5.0), np.log10(300.0), state_size)
        self.log_tau = nn.Parameter(tau.log()[None].repeat(width, 1))
        self.input_gain = nn.Parameter(torch.ones(width, state_size))
        self.output_gain = nn.Parameter(
            torch.randn(width, state_size) / np.sqrt(state_size)
        )
        self.skip = nn.Parameter(torch.ones(width))
        self.mix = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor, state=None):
        # values: batch, time, width; state: batch, width, state_size
        batch, length, width = values.shape
        tau = self.log_tau.exp().clamp(1.0, 10_000.0)
        decay = torch.exp(-1.0 / tau)
        drive = (1.0 - decay) * self.input_gain
        if state is None:
            state = values.new_zeros(batch, width, decay.shape[1])

        lag = torch.arange(length, device=values.device, dtype=values.dtype)
        powers = decay[..., None].pow(lag)
        kernel = torch.sum(
            self.output_gain[..., None] * drive[..., None] * powers, dim=1
        )
        nfft = 2 * length
        signal_f = torch.fft.rfft(values.transpose(1, 2), n=nfft)
        kernel_f = torch.fft.rfft(kernel, n=nfft)
        response = torch.fft.irfft(signal_f * kernel_f[None], n=nfft)[..., :length]
        response = response.transpose(1, 2)

        time_power = decay[..., None].pow(lag + 1.0)
        initial = torch.einsum(
            "bhn,hnt,hn->bth", state, time_power, self.output_gain
        )
        output = response + initial + values * self.skip

        reverse_power = decay[..., None].pow(
            torch.arange(length - 1, -1, -1, device=values.device, dtype=values.dtype)
        )
        driven_state = torch.einsum(
            "bth,hnt,hn->bhn", values, reverse_power, drive
        )
        final_state = state * decay.pow(length)[None] + driven_state
        transformed = self.mix(torch.nn.functional.gelu(output))
        return self.norm(values + self.dropout(transformed)), final_state


class CausalCNNS4D(CausalCNNGRU):
    """Identical CNN front-end with a parameter-matched diagonal SSM head."""

    def __init__(self, channels: int = 20, hidden: int = 32):
        super().__init__(channels, hidden)
        del self.gru
        self.input_projection = nn.Linear(24, hidden)
        self.ssm = nn.ModuleList(
            [DiagonalSSMLayer(hidden, state_size=8) for _ in range(2)]
        )

    def forward(self, values: torch.Tensor, hidden=None):
        batch, steps = values.shape[:2]
        encoded = self.encode(values.reshape(-1, *values.shape[2:]))
        sequence = self.input_projection(encoded.reshape(batch, steps, -1))
        if hidden is None:
            hidden = (None,) * len(self.ssm)
        next_hidden = []
        for layer, state in zip(self.ssm, hidden):
            sequence, state = layer(sequence, state)
            next_hidden.append(state)
        return self.output(sequence).squeeze(-1), tuple(next_hidden)


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
        default=Path("data/interim/lee2019_cnn_s4d_scores_pilot5"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/lee2019_cnn_s4d_pilot5.json"),
    )
    args = parser.parse_args()

    probe_gru = CausalCNNGRU(20, 32)
    probe_s4d = CausalCNNS4D(20, 32)
    parameter_counts = {
        "cnn_gru": int(sum(p.numel() for p in probe_gru.parameters())),
        "cnn_s4d": int(sum(p.numel() for p in probe_s4d.parameters())),
    }
    print(f"parameters: {parameter_counts}", flush=True)

    rows = []
    for session in args.sessions:
        for subject in args.subjects:
            print(f"CNN-S4D S{session} sub-{subject:02d}", flush=True)
            rows.append(
                analyze(
                    args.data_root, args.cache_root, subject, session,
                    args.epochs, args.seeds, args.device, args.shifts,
                    score_cache_root=args.score_cache_root,
                    model_class=CausalCNNS4D,
                )
            )
    payload = {
        "protocol": {
            "comparison": "same CNN/training/evaluation; GRU head replaced only",
            "head": "two-layer stable real diagonal S4D-style SSM",
            "time_constants": "initialized 0.5--30 s at 0.1-s score step",
            "causal": True,
            "bidirectional": False,
            "subjects": args.subjects,
            "sessions": args.sessions,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "claim_limit": "architecture ablation, not external S4/S5 reproduction",
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
