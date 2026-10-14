import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.hsmm import (
    GaussianEmission,
    causal_two_state_hsmm,
    duration_pmf,
)


class HSMMTests(unittest.TestCase):
    def test_duration_pmf_has_requested_mean(self):
        pmf = duration_pmf(1, 2.0, 3)
        support = np.arange(1, 4)
        self.assertAlmostEqual(float(np.sum(pmf)), 1.0)
        self.assertAlmostEqual(float(np.sum(pmf * support)), 2.0, places=6)

    def test_duration_endpoints_are_point_masses(self):
        self.assertEqual(duration_pmf(1, 1.0, 3).tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(duration_pmf(1, 3.0, 3).tolist(), [0.0, 0.0, 1.0])

    def test_filter_is_causal(self):
        emission = GaussianEmission(0.0, 1.0, 3.0, 1.0)
        prefix = np.zeros(8)
        first = causal_two_state_hsmm(
            np.r_[prefix, np.zeros(4)],
            emission,
            minimum_mi_steps=1,
            expected_mi_steps=2.0,
            maximum_mi_steps=3,
        )
        second = causal_two_state_hsmm(
            np.r_[prefix, np.full(4, 3.0)],
            emission,
            minimum_mi_steps=1,
            expected_mi_steps=2.0,
            maximum_mi_steps=3,
        )
        np.testing.assert_allclose(first[: len(prefix)], second[: len(prefix)])

    def test_mi_evidence_increases_posterior(self):
        emission = GaussianEmission(0.0, 0.5, 3.0, 0.5)
        statistic = causal_two_state_hsmm(
            np.r_[np.zeros(8), np.full(4, 3.0)],
            emission,
            minimum_mi_steps=1,
            expected_mi_steps=1.0,
            maximum_mi_steps=3,
        )
        self.assertLess(float(np.max(statistic[:8])), 0.1)
        self.assertGreater(float(np.max(statistic[8:])), 0.9)

    def test_minimum_duration_delays_mature_posterior(self):
        emission = GaussianEmission(0.0, 0.5, 3.0, 0.5)
        statistic = causal_two_state_hsmm(
            np.r_[np.zeros(4), np.full(4, 3.0)],
            emission,
            minimum_mi_steps=2,
            expected_mi_steps=2.0,
            maximum_mi_steps=3,
        )
        self.assertLess(statistic[4], 1e-6)
        self.assertGreater(statistic[5], 0.9)


if __name__ == "__main__":
    unittest.main()
