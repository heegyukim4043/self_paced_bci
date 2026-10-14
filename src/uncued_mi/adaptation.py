from __future__ import annotations

import numpy as np

from .decision import AlarmResult


def ewma_normalize_and_decide(
    scores: np.ndarray,
    *,
    initial_mean: float,
    initial_variance: float,
    threshold: float,
    step_seconds: float,
    half_life_seconds: float | None,
    update_mode: str,
    update_mask: np.ndarray | None = None,
    ma_seconds: float = 0.5,
    dwell_seconds: float = 0.3,
    refractory_seconds: float = 1.0,
    anchor_prior_n0: float = 0.0,
    between_variance: bool = False,
    sigma_floor_fraction: float = 0.25,
) -> AlarmResult:
    """Causal score normalization with optional idle-only EWMA updating."""
    if update_mode not in {"static", "all", "freeze_only", "idle_only"}:
        raise ValueError("invalid update_mode")
    values = np.asarray(scores, dtype=float)
    allowed = np.ones(len(values), dtype=bool) if update_mask is None else np.asarray(update_mask, bool)
    alpha = 0.0 if half_life_seconds is None else 1 - 2 ** (-step_seconds / half_life_seconds)
    anchor_mean = float(initial_mean)
    anchor_variance = max(float(initial_variance), 1e-8)
    mean, variance = anchor_mean, anchor_variance
    weight_sum, squared_weight_sum = 0.0, 0.0
    sigma_floor = sigma_floor_fraction * np.sqrt(anchor_variance)
    normalized = np.empty(len(values), dtype=float)
    alarms = np.zeros(len(values), dtype=bool)
    ma_steps = max(1, int(np.ceil(ma_seconds / step_seconds)))
    dwell_steps = max(1, int(np.ceil(dwell_seconds / step_seconds)))
    refractory_steps = max(0, int(np.ceil(refractory_seconds / step_seconds)))
    history: list[float] = []
    count, blocked_until = 0, -1
    for index, score in enumerate(values):
        effective_n = weight_sum * weight_sum / squared_weight_sum if squared_weight_sum > 0 else 0.0
        if anchor_prior_n0 > 0:
            anchor_weight = anchor_prior_n0 / (anchor_prior_n0 + effective_n)
            blended_mean = anchor_weight * anchor_mean + (1 - anchor_weight) * mean
            blended_variance = anchor_weight * anchor_variance + (1 - anchor_weight) * variance
            if between_variance:
                blended_variance += anchor_weight * (1 - anchor_weight) * (anchor_mean - mean) ** 2
        else:
            blended_mean, blended_variance = mean, variance
        z = (score - blended_mean) / max(np.sqrt(max(blended_variance, 0.0)), sigma_floor)
        normalized[index] = z
        if not allowed[index]:
            history.clear(); count = 0
            continue
        history.append(z)
        if len(history) > ma_steps:
            history.pop(0)
        active = len(history) == ma_steps and float(np.mean(history)) >= threshold
        count = count + 1 if active else 0
        if count == dwell_steps and index > blocked_until:
            alarms[index] = True
            blocked_until = index + refractory_steps
        update = update_mode == "all"
        if update_mode == "freeze_only":
            update = index > blocked_until and not active
        if update_mode == "idle_only":
            # Deployable proxy: do not learn from high-score/active/refractory
            # periods. This uses no ground-truth MI labels.
            update = index > blocked_until and z < threshold and not active
        if update and alpha > 0:
            delta = score - mean
            mean += alpha * delta
            variance = (1 - alpha) * (variance + alpha * delta * delta)
            weight_sum = (1 - alpha) * weight_sum + 1.0
            squared_weight_sum = (1 - alpha) ** 2 * squared_weight_sum + 1.0
    return AlarmResult(alarms=alarms, statistic=normalized)
