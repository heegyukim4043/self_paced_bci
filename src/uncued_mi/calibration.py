from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocationScale:
    location: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls, values: np.ndarray, axis: int | tuple[int, ...] = 0, epsilon: float = 1e-8
    ) -> "LocationScale":
        location = np.mean(values, axis=axis)
        scale = np.std(values, axis=axis, ddof=1)
        return cls(location, np.maximum(scale, epsilon))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.location) / self.scale

    @classmethod
    def fit_median_mad(
        cls, values: np.ndarray, axis: int | tuple[int, ...] = 0, epsilon: float = 1e-8
    ) -> "LocationScale":
        location = np.median(values, axis=axis)
        scale = 1.4826 * np.median(np.abs(values - location), axis=axis)
        return cls(location, np.maximum(scale, epsilon))


def tier0_from_features(
    features: np.ndarray,
    channels: tuple[str, ...],
    normalizer: LocationScale,
    gate_channels: tuple[str, ...] = ("C3", "C4"),
) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError("features must have shape (windows, channels, bands)")
    z = normalizer.transform(features)
    lookup = {name.casefold(): index for index, name in enumerate(channels)}
    indices = [lookup[name.casefold()] for name in gate_channels]
    return np.max(-np.mean(z[:, indices, :], axis=-1), axis=1)


def scalar_align_to_reference(
    target_scores: np.ndarray,
    target_idle_scores: np.ndarray,
    reference_idle_scores: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    target_model = LocationScale.fit(target_idle_scores, epsilon=epsilon)
    reference_model = LocationScale.fit(reference_idle_scores, epsilon=epsilon)
    standardized = target_model.transform(target_scores)
    return standardized * reference_model.scale + reference_model.location


def empirical_quantile_map(
    values: np.ndarray,
    source_idle: np.ndarray,
    target_idle: np.ndarray,
) -> np.ndarray:
    if len(source_idle) < 2 or len(target_idle) < 2:
        raise ValueError("At least two idle samples are required")
    source = np.sort(np.asarray(source_idle, dtype=float))
    target = np.sort(np.asarray(target_idle, dtype=float))
    probabilities = np.linspace(0.0, 1.0, len(target))
    ranks = np.interp(values, target, probabilities, left=0.0, right=1.0)
    source_probabilities = np.linspace(0.0, 1.0, len(source))
    return np.interp(ranks, source_probabilities, source)
