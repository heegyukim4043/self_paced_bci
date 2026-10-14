from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianEmission:
    idle_mean: float
    idle_std: float
    mi_mean: float
    mi_std: float

    @classmethod
    def fit(
        cls,
        idle_scores: np.ndarray,
        mi_scores: np.ndarray,
        minimum_std: float = 1e-3,
    ) -> "GaussianEmission":
        idle = np.asarray(idle_scores, dtype=float)
        mi = np.asarray(mi_scores, dtype=float)
        if not len(idle) or not len(mi):
            raise ValueError("Both idle and MI scores are required")
        if not np.all(np.isfinite(idle)) or not np.all(np.isfinite(mi)):
            raise ValueError("Emission scores must be finite")
        return cls(
            idle_mean=float(np.mean(idle)),
            idle_std=max(float(np.std(idle, ddof=1)), minimum_std),
            mi_mean=float(np.mean(mi)),
            mi_std=max(float(np.std(mi, ddof=1)), minimum_std),
        )


def duration_pmf(
    minimum_steps: int,
    expected_steps: float,
    maximum_steps: int,
) -> np.ndarray:
    """Maximum-entropy discrete duration PMF on [minimum, maximum]."""
    if minimum_steps < 1 or maximum_steps < minimum_steps:
        raise ValueError("Invalid duration support")
    if not minimum_steps <= expected_steps <= maximum_steps:
        raise ValueError("Expected duration must lie inside the support")
    support = np.arange(minimum_steps, maximum_steps + 1, dtype=float)
    if np.isclose(expected_steps, minimum_steps):
        weights = np.zeros(len(support))
        weights[0] = 1.0
    elif np.isclose(expected_steps, maximum_steps):
        weights = np.zeros(len(support))
        weights[-1] = 1.0
    else:
        low, high = -50.0, 50.0
        for _ in range(100):
            middle = (low + high) / 2.0
            raw = np.exp(np.clip(middle * support, -700, 700))
            mean = float(np.sum(support * raw) / np.sum(raw))
            if mean < expected_steps:
                low = middle
            else:
                high = middle
        raw = np.exp(np.clip(((low + high) / 2.0) * support, -700, 700))
        weights = raw / np.sum(raw)
    result = np.zeros(maximum_steps, dtype=float)
    result[minimum_steps - 1 :] = weights
    return result


def _hazards(pmf: np.ndarray) -> np.ndarray:
    survival = np.flip(np.cumsum(np.flip(pmf)))
    hazards = np.divide(
        pmf,
        survival,
        out=np.ones_like(pmf),
        where=survival > np.finfo(float).tiny,
    )
    hazards[-1] = 1.0
    return hazards


def _normal_logpdf(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (values - mean) / std
    return -0.5 * z * z - np.log(std) - 0.5 * np.log(2.0 * np.pi)


def causal_two_state_hsmm(
    scores: np.ndarray,
    emission: GaussianEmission,
    *,
    minimum_mi_steps: int,
    expected_mi_steps: float,
    maximum_mi_steps: int,
    entry_probability: float = 0.01,
) -> np.ndarray:
    """Return causal posterior for a mature MI state.

    The hidden state is idle or MI-with-age. MI durations use an explicit
    truncated distribution. The returned statistic sums MI ages at or beyond
    ``minimum_mi_steps``; no future observations or backward smoothing are used.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional array")
    if not 0.0 < entry_probability < 1.0:
        raise ValueError("entry_probability must lie in (0, 1)")
    pmf = duration_pmf(
        minimum_mi_steps, expected_mi_steps, maximum_mi_steps
    )
    hazards = _hazards(pmf)
    idle_ll = _normal_logpdf(values, emission.idle_mean, emission.idle_std)
    mi_ll = _normal_logpdf(values, emission.mi_mean, emission.mi_std)

    # Probability vector: [idle, MI age 1, ..., MI age maximum].
    posterior = np.zeros(maximum_mi_steps + 1, dtype=float)
    posterior[0] = 1.0
    statistic = np.zeros(len(values), dtype=float)
    mature_start = minimum_mi_steps

    for index, (idle_loglik, mi_loglik) in enumerate(zip(idle_ll, mi_ll)):
        predicted = np.zeros_like(posterior)
        predicted[0] += posterior[0] * (1.0 - entry_probability)
        predicted[1] += posterior[0] * entry_probability
        for age in range(1, maximum_mi_steps + 1):
            exit_probability = hazards[age - 1]
            predicted[0] += posterior[age] * exit_probability
            if age < maximum_mi_steps:
                predicted[age + 1] += posterior[age] * (1.0 - exit_probability)

        offset = max(idle_loglik, mi_loglik)
        likelihood = np.r_[
            np.exp(idle_loglik - offset),
            np.full(maximum_mi_steps, np.exp(mi_loglik - offset)),
        ]
        posterior = predicted * likelihood
        total = np.sum(posterior)
        if not np.isfinite(total) or total <= np.finfo(float).tiny:
            raise FloatingPointError("HSMM forward filter normalization failed")
        posterior /= total
        statistic[index] = np.sum(posterior[mature_start:])
    return statistic

