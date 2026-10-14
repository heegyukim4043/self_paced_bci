"""Single pre-specified, larger causal CNN-GRU candidate for Lee2019.

The model consumes the same 3-band, 20-channel, 250-sample windows as the
frozen CNN-GRU.  No attention, bidirectionality, or future windows are used.
It is an *untrained candidate*, not a validated improvement.
"""

from __future__ import annotations

import torch
from torch import nn


class LargeCausalCNNGRU(nn.Module):
    """Local filter-bank encoder plus a two-layer chronological GRU."""

    def __init__(self, channels: int = 20, hidden: int = 128):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(3, 24, (1, 25), groups=3, bias=False),
            # Per-window normalization: unlike BatchNorm, training statistics
            # cannot couple this window to later windows in the same batch.
            nn.GroupNorm(6, 24),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(24, 48, (channels, 1), groups=24, bias=False),
            nn.GroupNorm(8, 48),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(48, 64, (1, 1), bias=False),
            nn.GroupNorm(8, 64),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.gru = nn.GRU(64, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.output = nn.Linear(hidden, 1)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if (values.ndim != 4 or values.shape[1] != 3
                or values.shape[2] != self.spatial[0].kernel_size[0]
                or values.shape[3] != 250):
            raise ValueError("expected [windows, 3 bands, channels, 250 samples]")
        return self.projection(self.spatial(self.temporal(values)))

    def forward(self, values: torch.Tensor, hidden: torch.Tensor | None = None):
        if values.ndim != 5:
            raise ValueError("expected [batch, chronological windows, 3, channels, samples]")
        batch, steps = values.shape[:2]
        encoded = self.encode(values.reshape(-1, *values.shape[2:]))
        encoded = encoded.reshape(batch, steps, -1)
        state, hidden = self.gru(encoded, hidden)
        return self.output(state).squeeze(-1), hidden
