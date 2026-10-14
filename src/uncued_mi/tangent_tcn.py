from __future__ import annotations

import torch
from torch import nn


class TangentTCN(nn.Module):
    def __init__(self, features: int, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(features, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Conv1d(hidden, hidden, 3),
            nn.ELU(),
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, values):
        if values.ndim != 3:
            raise ValueError("values must be (batch, steps, features)")
        latent = self.network(values.transpose(1, 2))[:, :, -1]
        return self.classifier(latent).squeeze(1)

