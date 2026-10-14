import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.riemannian import (
    riemannian_mean, tangent_features, transport_covariance, window_covariances,
)


class RiemannianTests(unittest.TestCase):
    def test_rank_deficient_covariance_remains_finite(self):
        covariance = np.zeros((1, 3, 3), dtype=float)
        reference = np.eye(3)
        self.assertTrue(np.isfinite(tangent_features(covariance, reference)).all())
    def test_identity_reference_maps_identity_to_zero(self):
        values = np.eye(3)[None, ...]
        result = tangent_features(values, np.eye(3))
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_window_covariance_shape(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(size=(30, 4))
        result = window_covariances(signal, np.array([10, 20, 30]), 10)
        self.assertEqual(result.shape, (3, 4, 4))
        np.testing.assert_allclose(result, np.swapaxes(result, 1, 2))

    def test_riemannian_mean_of_identical_matrices(self):
        covariance = np.diag([2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            riemannian_mean(np.stack([covariance, covariance])), covariance,
            atol=1e-9,
        )

    def test_transport_maps_current_reference_to_anchor(self):
        current = np.diag([1.0, 4.0, 9.0])
        anchor = np.diag([4.0, 9.0, 16.0])
        np.testing.assert_allclose(
            transport_covariance(current, current, anchor), anchor, atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
