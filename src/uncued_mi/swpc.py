from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class EEGNetGate(nn.Module):
    """Compact EEGNet-style binary gate for (B, C, T) causal windows."""

    def __init__(self, channels: int, samples: int, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            size = int(np.prod(self._features_4d(dummy).shape[1:]))
        self.classifier = nn.Linear(size, 1)

    def _features_4d(self, values: torch.Tensor) -> torch.Tensor:
        return self.separable(self.spatial(self.temporal(values)))

    def encode(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("windows must have shape (batch, channels, samples)")
        return self._features_4d(windows[:, None]).flatten(1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(windows)).squeeze(1)


def balanced_gate_loss(
    positive_logits: torch.Tensor,
    rest_logits: torch.Tensor,
    iti_logits: torch.Tensor,
) -> torch.Tensor:
    positive = F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    )
    rest = F.binary_cross_entropy_with_logits(
        rest_logits, torch.zeros_like(rest_logits)
    )
    iti = F.binary_cross_entropy_with_logits(
        iti_logits, torch.zeros_like(iti_logits)
    )
    return 0.5 * positive + 0.25 * rest + 0.25 * iti


def augment_windows(values: torch.Tensor) -> torch.Tensor:
    """Causal-safe amplitude, noise and short masking augmentations."""
    batch, _, samples = values.shape
    scale = 0.9 + 0.2 * torch.rand(batch, 1, 1, device=values.device)
    noise_scale = values.std(dim=-1, keepdim=True).clamp_min(1e-4)
    output = values * scale + torch.randn_like(values) * noise_scale * 0.02
    width = max(1, samples // 20)
    starts = torch.randint(0, samples - width + 1, (batch,), device=values.device)
    for index, start in enumerate(starts.tolist()):
        output[index, :, start : start + width] = 0
    return output


def swpc_consistency_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    variance_floor: float = 0.5,
) -> torch.Tensor:
    student = F.normalize(student_features, dim=1)
    teacher = F.normalize(teacher_features.detach(), dim=1)
    invariance = (1.0 - (student * teacher).sum(dim=1)).mean()
    spread = torch.sqrt(student_features.var(dim=0) + 1e-4)
    variance = F.relu(variance_floor - spread).mean()
    return invariance + 0.1 * variance


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for source, target in zip(student.parameters(), teacher.parameters()):
        target.mul_(momentum).add_(source, alpha=1.0 - momentum)


def make_teacher(model: EEGNetGate) -> EEGNetGate:
    teacher = copy.deepcopy(model)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher

