from __future__ import annotations

import torch
from torch import nn


class CompactFBCNet(nn.Module):
    """Filter-bank spatial filters followed by segmented log variance."""

    def __init__(self, bands: int, channels: int, spatial_filters: int = 4, segments: int = 4):
        super().__init__()
        self.bands = bands
        self.segments = segments
        self.spatial = nn.Parameter(torch.empty(bands, spatial_filters, channels))
        nn.init.xavier_uniform_(self.spatial)
        self.batch_norm = nn.BatchNorm1d(bands * spatial_filters)
        self.classifier = nn.Linear(bands * spatial_filters * segments, 1)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("values must be (batch, bands, channels, samples)")
        projected = torch.einsum("nbct,bkc->nbkt", values, self.spatial)
        batch, bands, filters, samples = projected.shape
        projected = self.batch_norm(projected.reshape(batch, bands * filters, samples))
        usable = samples - samples % self.segments
        chunks = projected[:, :, :usable].reshape(
            batch, bands * filters, self.segments, usable // self.segments
        )
        variance = chunks.var(dim=-1, unbiased=False).clamp_min(1e-6).log()
        return variance.flatten(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(values)).squeeze(1)

