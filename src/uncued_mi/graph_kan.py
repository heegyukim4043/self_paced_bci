"""Small causal spatial graph heads for filter-bank EEG windows.

The graph is instantaneous within each past-only one-second window.  Temporal
state remains in the downstream GRU, allowing a clean spatial-block ablation
against the established CNN-GRU.
"""

from __future__ import annotations

import math

import torch
from torch import nn


SENSORIMOTOR_CHANNELS = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)


def sensorimotor_coordinates() -> torch.Tensor:
    """Return simple left-right/anterior-posterior montage coordinates."""
    positions = {
        "FC5": (-3.0, 1.0), "FC3": (-2.0, 1.0), "FC1": (-1.0, 1.0),
        "FC2": (1.0, 1.0), "FC4": (2.0, 1.0), "FC6": (3.0, 1.0),
        "C5": (-3.0, 0.0), "C3": (-2.0, 0.0), "C1": (-1.0, 0.0),
        "Cz": (0.0, 0.0), "C2": (1.0, 0.0), "C4": (2.0, 0.0),
        "C6": (3.0, 0.0),
        "CP5": (-3.0, -1.0), "CP3": (-2.0, -1.0), "CP1": (-1.0, -1.0),
        "CPz": (0.0, -1.0), "CP2": (1.0, -1.0), "CP4": (2.0, -1.0),
        "CP6": (3.0, -1.0),
    }
    return torch.tensor([positions[name] for name in SENSORIMOTOR_CHANNELS])


def geodesic_prior(k: int = 4) -> torch.Tensor:
    """Sparse symmetric Gaussian adjacency with self loops and normalization."""
    coordinates = sensorimotor_coordinates().float()
    distance = torch.cdist(coordinates, coordinates)
    nearest = torch.argsort(distance, dim=1)[:, 1:k + 1]
    mask = torch.eye(len(coordinates), dtype=torch.bool)
    mask.scatter_(1, nearest, True)
    mask = mask | mask.T
    neighbor_distance = distance[mask & ~torch.eye(len(coordinates), dtype=torch.bool)]
    sigma = torch.median(neighbor_distance).clamp_min(1e-6)
    adjacency = torch.exp(-(distance ** 2) / (2.0 * sigma ** 2)) * mask
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1e-8)
    inverse = degree.rsqrt()
    return inverse[:, None] * adjacency * inverse[None, :]


class FourierKANLinear(nn.Module):
    """A real Fourier KAN transform; unlike the pasted sketch, bases are used."""

    def __init__(self, input_dim: int, output_dim: int, harmonics: int = 1):
        super().__init__()
        self.base = nn.Linear(input_dim, output_dim)
        self.harmonics = int(harmonics)
        self.coefficients = nn.Parameter(
            torch.empty(output_dim, input_dim, 2 * self.harmonics)
        )
        nn.init.normal_(self.coefficients, std=0.02 / math.sqrt(input_dim))
        self.input_scale = nn.Parameter(torch.ones(input_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scaled = values * self.input_scale
        frequencies = torch.arange(
            1, self.harmonics + 1, device=values.device, dtype=values.dtype
        )
        angles = math.pi * scaled[..., :, None] * frequencies
        basis = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        nonlinear = torch.einsum("...dk,odk->...o", basis, self.coefficients)
        return self.base(values) + nonlinear


class SpatialGraphLayer(nn.Module):
    def __init__(self, width: int, kind: str):
        super().__init__()
        if kind not in {"gcn", "gkan"}:
            raise ValueError(f"Unknown graph layer: {kind}")
        transform = nn.Linear if kind == "gcn" else FourierKANLinear
        self.message = transform(width, width)
        self.update = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        messages = self.message(nodes)
        aggregate = torch.einsum("ij,...jd->...id", adjacency, messages)
        update = torch.nn.functional.gelu(
            self.update(torch.cat((nodes, aggregate), dim=-1))
        )
        return self.norm(nodes + update)


class CausalCNNGraphGRU(nn.Module):
    """Channel-wise temporal CNN, spatial GCN/G-KAN, then stateful GRU."""

    def __init__(
        self, channels: int = 20, hidden: int = 32, *, kind: str = "gkan",
        node_width: int = 6,
    ):
        super().__init__()
        if channels != len(SENSORIMOTOR_CHANNELS):
            raise ValueError("The frozen graph prior is defined for 20 sensorimotor channels")
        self.temporal = nn.Sequential(
            nn.Conv2d(3, 12, (1, 25), groups=3, bias=False),
            nn.BatchNorm2d(12),
            nn.ELU(),
        )
        self.node_projection = nn.Linear(12, node_width)
        self.graph1 = SpatialGraphLayer(node_width, kind)
        self.graph2 = SpatialGraphLayer(node_width, kind)
        self.readout = nn.Linear(2 * node_width, 24)
        self.gru = nn.GRU(24, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
        self.register_buffer("geo_adjacency", geodesic_prior(), persistent=True)
        # Low-rank symmetric reweighting of existing physical edges only.
        self.adjacency_residual = nn.Parameter(torch.randn(channels, 2) * 0.01)

        x = sensorimotor_coordinates()[:, 0]
        self.register_buffer("left_mask", x < 0, persistent=False)
        self.register_buffer("right_mask", x > 0, persistent=False)

    def adjacency(self) -> torch.Tensor:
        residual = self.adjacency_residual @ self.adjacency_residual.T
        adjacency = self.geo_adjacency * torch.exp(residual.clamp(-2.0, 2.0))
        degree = adjacency.sum(dim=1).clamp_min(1e-8)
        inverse = degree.rsqrt()
        return inverse[:, None] * adjacency * inverse[None, :]

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        # values: windows x bands x channels x past samples
        temporal = self.temporal(values).mean(dim=-1).transpose(1, 2)
        nodes = self.node_projection(temporal)
        adjacency = self.adjacency()
        nodes = self.graph1(nodes, adjacency)
        nodes = self.graph2(nodes, adjacency)
        global_mean = nodes.mean(dim=1)
        # Unsigned hemispheric asymmetry preserves detection information while
        # avoiding a requirement to classify left versus right MI.
        left = nodes[:, self.left_mask].mean(dim=1)
        right = nodes[:, self.right_mask].mean(dim=1)
        spatial = torch.cat((global_mean, torch.abs(left - right)), dim=-1)
        return self.readout(spatial)

    def forward(self, values: torch.Tensor, hidden=None):
        batch, steps = values.shape[:2]
        encoded = self.encode(values.reshape(-1, *values.shape[2:]))
        sequence = encoded.reshape(batch, steps, -1)
        state, hidden = self.gru(sequence, hidden)
        return self.output(state).squeeze(-1), hidden


class CausalCNNGCNGRU(CausalCNNGraphGRU):
    def __init__(self, channels: int = 20, hidden: int = 32):
        super().__init__(channels, hidden, kind="gcn")


class CausalCNNGKANGRU(CausalCNNGraphGRU):
    def __init__(self, channels: int = 20, hidden: int = 32):
        super().__init__(channels, hidden, kind="gkan")
