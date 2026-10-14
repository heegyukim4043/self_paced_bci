from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEEDS = (20260725, 20260726, 20260727, 20260728, 20260729)


def normalized_lse(logits: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    count = logits.shape[dim]
    return tau * (torch.logsumexp(logits / tau, dim=dim) - np.log(count))


class CausalConv1d(nn.Conv1d):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        left = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(values, (left, 0)))


class CompactCausalEncoder(nn.Module):
    """Small causal 1-second EEG encoder returning one logit per window."""

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            CausalConv1d(
                hidden, hidden, kernel_size=15, groups=hidden, bias=False
            ),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            CausalConv1d(
                hidden,
                hidden,
                kernel_size=9,
                dilation=2,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("windows must have shape (batch, channels, samples)")
        features = self.network(windows).mean(dim=-1)
        return self.classifier(features).squeeze(-1)


def short_continuity_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() < 2:
        return logits.new_zeros(())
    return F.smooth_l1_loss(logits[1:], logits[:-1])


def mil_loss(
    positive_logits: list[torch.Tensor],
    rest_logits: torch.Tensor,
    iti_logits: torch.Tensor,
    *,
    tau: float,
    false_alarm_weight: float,
    continuity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not positive_logits:
        raise ValueError("At least one positive bag is required")
    event_scores = torch.stack(
        [normalized_lse(values, tau=tau) for values in positive_logits]
    )
    event = F.binary_cross_entropy_with_logits(
        event_scores, torch.ones_like(event_scores)
    )
    rest = F.binary_cross_entropy_with_logits(
        rest_logits, torch.zeros_like(rest_logits)
    )
    iti = F.binary_cross_entropy_with_logits(
        iti_logits, torch.zeros_like(iti_logits)
    )
    false_alarm = 0.5 * (rest + iti)
    continuity = torch.stack(
        [short_continuity_loss(values) for values in positive_logits]
    ).mean()
    total = (
        event
        + false_alarm_weight * false_alarm
        + continuity_weight * continuity
    )
    return total, {
        "event": event.detach(),
        "rest": rest.detach(),
        "iti": iti.detach(),
        "false_alarm": false_alarm.detach(),
        "continuity": continuity.detach(),
    }


@dataclass(frozen=True)
class ValidationCandidate:
    tau: float
    false_alarm_weight: float
    continuity_weight: float
    median_fp_per_minute: float
    median_excess_tpr: float


def select_validation_candidate(
    candidates: list[ValidationCandidate],
) -> tuple[ValidationCandidate, bool]:
    """Apply the preregistered admissibility and simplicity rules."""
    if not candidates:
        raise ValueError("No validation candidates")
    admissible = [
        item
        for item in candidates
        if 1.5 <= item.median_fp_per_minute <= 2.5
    ]
    calibration_failure = not admissible
    if calibration_failure:
        selected = min(
            candidates,
            key=lambda item: (
                abs(item.median_fp_per_minute - 2.0),
                item.continuity_weight != 0,
                -item.tau,
                item.false_alarm_weight,
            ),
        )
        return selected, True

    best_excess = max(item.median_excess_tpr for item in admissible)
    tied = [
        item
        for item in admissible
        if best_excess - item.median_excess_tpr < 0.01
    ]
    selected = min(
        tied,
        key=lambda item: (
            item.continuity_weight != 0,
            -item.tau,
            item.false_alarm_weight,
        ),
    )
    return selected, False
