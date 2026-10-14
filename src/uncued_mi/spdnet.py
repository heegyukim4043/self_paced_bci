from __future__ import annotations

import torch
from torch import nn


class CompactMultiBandSPDNet(nn.Module):
    def __init__(self, bands: int, channels: int, output_dim: int = 8):
        super().__init__()
        weights = []
        for _ in range(bands):
            q, _ = torch.linalg.qr(torch.randn(channels, output_dim))
            weights.append(q.T)
        self.bimap = nn.Parameter(torch.stack(weights))
        self.output_dim = output_dim
        features = bands * output_dim * (output_dim + 1) // 2
        self.norm = nn.BatchNorm1d(features)
        self.classifier = nn.Linear(features, 1)
        self.register_buffer("triangle", torch.triu_indices(output_dim, output_dim))

    def encode(self, covariance: torch.Tensor) -> torch.Tensor:
        if covariance.ndim != 4:
            raise ValueError("covariance must be (batch, bands, channels, channels)")
        mapped = torch.einsum("boc,nbcd,bpd->nbop", self.bimap, covariance, self.bimap)
        eigenvalues, eigenvectors = torch.linalg.eigh(mapped)
        eigenvalues = eigenvalues.clamp_min(1e-5).log()
        logarithm = torch.einsum(
            "nbik,nbk,nbjk->nbij", eigenvectors, eigenvalues, eigenvectors
        )
        vector = logarithm[:, :, self.triangle[0], self.triangle[1]].flatten(1)
        return self.norm(vector)

    def forward(self, covariance: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(covariance)).squeeze(1)

    @torch.no_grad()
    def project_bimap(self) -> None:
        for band in range(len(self.bimap)):
            q, _ = torch.linalg.qr(self.bimap[band].T)
            self.bimap[band].copy_(q.T)

