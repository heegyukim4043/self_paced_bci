from __future__ import annotations

import numpy as np


def augmented_window_covariances(
    signal: np.ndarray,
    end_samples: np.ndarray,
    window_samples: int,
    lag_samples: tuple[int, ...] = (0, 25),
) -> np.ndarray:
    """Covariance of causal time-lag augmented observations.

    For each analysis time ``t``, the observation is
    ``[x(t-lag) for lag in lag_samples]``. Every returned matrix therefore
    uses samples no later than its corresponding end sample.
    """
    signal = np.asarray(signal)
    ends = np.asarray(end_samples, dtype=np.int64)
    lags = tuple(int(value) for value in lag_samples)
    if signal.ndim != 2:
        raise ValueError("signal must have shape (samples, channels)")
    if not lags or min(lags) < 0 or len(set(lags)) != len(lags):
        raise ValueError("lag_samples must contain unique non-negative lags")
    if window_samples < 2:
        raise ValueError("window_samples must be at least 2")
    if len(ends) and np.min(ends) - window_samples - max(lags) < 0:
        raise ValueError("an end sample lacks sufficient causal lag history")

    dimension = signal.shape[1] * len(lags)
    result = np.empty((len(ends), dimension, dimension), dtype=np.float64)
    for index, end in enumerate(ends):
        segment = np.concatenate(
            [
                np.asarray(
                    signal[
                        end - window_samples - lag : end - lag
                    ],
                    dtype=np.float64,
                )
                for lag in lags
            ],
            axis=1,
        )
        segment -= segment.mean(axis=0, keepdims=True)
        result[index] = segment.T @ segment / (window_samples - 1)
    return result
