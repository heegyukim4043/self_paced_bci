from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter

from .config import ExperimentConfig


@dataclass(frozen=True)
class FeatureStream:
    times: np.ndarray
    log_power: np.ndarray
    channels: tuple[str, ...]
    bands: tuple[str, ...]


@dataclass(frozen=True)
class ScoreNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, epsilon: float = 1e-12) -> "ScoreNormalizer":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0, ddof=1)
        return cls(mean=mean, std=np.maximum(std, epsilon))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    if eeg.ndim != 2:
        raise ValueError("EEG must have shape (samples, channels)")
    return eeg - np.mean(eeg, axis=1, keepdims=True)


def _causal_band_signal(
    eeg: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
    order: int,
) -> np.ndarray:
    b, a = butter(order, [low, high], btype="bandpass", fs=sfreq)
    return lfilter(b, a, eeg, axis=0)


def _causal_downsample(
    signal: np.ndarray, sfreq: float, target_sfreq: float
) -> np.ndarray:
    """Sample a filtered signal on a target clock using past samples only."""
    if target_sfreq > sfreq:
        raise ValueError("Causal upsampling is not supported")
    ratio = sfreq / target_sfreq
    if np.isclose(ratio, round(ratio)):
        return signal[:: int(round(ratio))]
    output_length = int(np.floor((len(signal) - 1) * target_sfreq / sfreq)) + 1
    source_indices = np.floor(
        np.arange(output_length, dtype=float) * sfreq / target_sfreq
    ).astype(np.int64)
    return signal[source_indices]


def extract_log_power(
    eeg: np.ndarray,
    sfreq: float,
    channels: tuple[str, ...],
    config: ExperimentConfig,
    output_channels: tuple[str, ...] | None = None,
) -> FeatureStream:
    referenced = common_average_reference(eeg)
    if output_channels is None:
        output_channels = channels
    lookup = {name.casefold(): index for index, name in enumerate(channels)}
    missing = [
        name for name in output_channels if name.casefold() not in lookup
    ]
    if missing:
        raise ValueError(f"Missing output channels: {missing}")
    output_indices = np.asarray(
        [lookup[name.casefold()] for name in output_channels], dtype=np.int64
    )
    referenced = referenced[:, output_indices]
    window = int(round(config.window_seconds * config.target_sfreq))
    step = int(round(config.feature_step_seconds * config.target_sfreq))
    transient = int(round(config.transient_seconds * config.target_sfreq))
    if (
        window <= 0
        or step <= 0
        or referenced.shape[0] < window * sfreq / config.target_sfreq
    ):
        raise ValueError("Recording is too short for configured windows")

    per_band: list[np.ndarray] = []
    ends: np.ndarray | None = None
    for _, low, high in config.bands:
        filtered = _causal_downsample(
            _causal_band_signal(
                referenced, sfreq, low, high, config.filter_order
            ),
            sfreq,
            config.target_sfreq,
        )
        squared = filtered * filtered
        cumulative = np.vstack(
            [np.zeros((1, squared.shape[1])), np.cumsum(squared, axis=0)]
        )
        # A complete analysis window must begin after the causal filter warm-up.
        # Starting at max(window, transient) would still include transient samples.
        first_end = transient + window
        band_ends = np.arange(first_end, len(filtered) + 1, step)
        starts = band_ends - window
        power = (cumulative[band_ends] - cumulative[starts]) / window
        per_band.append(np.log(np.maximum(power, config.epsilon)))
        if ends is None:
            ends = band_ends
        elif not np.array_equal(ends, band_ends):
            raise AssertionError("Band feature timestamps differ")

    assert ends is not None
    values = np.stack(per_band, axis=-1)
    return FeatureStream(
        times=ends / config.target_sfreq,
        log_power=values,
        channels=output_channels,
        bands=tuple(name for name, _, _ in config.bands),
    )


def tier0_score(
    stream: FeatureStream,
    normalizer: ScoreNormalizer,
    gate_channels: tuple[str, ...] = ("C3", "C4"),
) -> np.ndarray:
    z = normalizer.transform(stream.log_power)
    lookup = {name.casefold(): index for index, name in enumerate(stream.channels)}
    indices = [lookup[name.casefold()] for name in gate_channels]
    channel_erd = -np.mean(z[:, indices, :], axis=-1)
    return np.max(channel_erd, axis=1)


def sample_mask(
    times: np.ndarray, intervals: list[tuple[float, float]]
) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in intervals:
        mask |= (times >= start) & (times < stop)
    return mask


def decision_subsample(
    times: np.ndarray, values: np.ndarray, decision_step_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(times) != len(values):
        raise ValueError("times and values must have equal length")
    selected = np.zeros(len(times), dtype=bool)
    next_time = times[0]
    for index, time in enumerate(times):
        if time + 1e-9 >= next_time:
            selected[index] = True
            next_time = time + decision_step_seconds
    return times[selected], values[selected]


def extract_covariances(
    eeg: np.ndarray,
    sfreq: float,
    channels: tuple[str, ...],
    config: ExperimentConfig,
    output_channels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Causal 8–30 Hz rolling covariance at the configured feature step."""
    lookup = {name.casefold(): index for index, name in enumerate(channels)}
    indices = np.asarray([lookup[name.casefold()] for name in output_channels])
    referenced = common_average_reference(eeg)[:, indices]
    filtered = _causal_downsample(
        _causal_band_signal(
            referenced, sfreq, 8.0, 30.0, config.filter_order
        ),
        sfreq,
        config.target_sfreq,
    )
    window = int(round(config.window_seconds * config.target_sfreq))
    step = int(round(config.feature_step_seconds * config.target_sfreq))
    transient = int(round(config.transient_seconds * config.target_sfreq))
    ends = np.arange(transient + window, len(filtered) + 1, step)
    covariances = np.empty(
        (len(ends), len(output_channels), len(output_channels)), dtype=np.float32
    )
    for index, end in enumerate(ends):
        segment = filtered[end - window : end]
        segment = segment - np.mean(segment, axis=0, keepdims=True)
        covariances[index] = (segment.T @ segment / (window - 1)).astype(np.float32)
    return ends / config.target_sfreq, covariances
