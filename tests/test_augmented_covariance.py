import numpy as np
import pytest

from uncued_mi.augmented_covariance import augmented_window_covariances


def test_augmented_covariance_matches_manual_construction():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(80, 3))
    covariance = augmented_window_covariances(
        signal, np.asarray([50]), 20, (0, 5)
    )[0]
    segment = np.c_[signal[30:50], signal[25:45]]
    expected = np.cov(segment, rowvar=False, ddof=1)
    np.testing.assert_allclose(covariance, expected, atol=1e-12)


def test_augmented_covariance_is_causal():
    rng = np.random.default_rng(8)
    signal = rng.normal(size=(100, 2))
    changed = signal.copy()
    changed[60:] += 1000
    original = augmented_window_covariances(signal, [60], 25, (0, 10))
    modified = augmented_window_covariances(changed, [60], 25, (0, 10))
    np.testing.assert_allclose(original, modified)


def test_augmented_covariance_rejects_insufficient_history():
    with pytest.raises(ValueError, match="history"):
        augmented_window_covariances(
            np.zeros((50, 2)), [20], 15, (0, 10)
        )
