"""Boundary and reproducibility checks for the post-G1 diagnostic."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_e3_g1_failure_attribution_20261009 import in_post_interval


class TestOutputCensoring(unittest.TestCase):
    def test_interval_is_causal_and_inclusive(self):
        times = np.array([0.99, 1.0, 1.2, 1.4, 1.401])
        np.testing.assert_array_equal(
            in_post_interval(times, np.array([1.0]), 0.4),
            [False, True, True, True, False],
        )

    def test_any_reference_and_empty_cases(self):
        times = np.array([0.1, 2.2, 3.1])
        np.testing.assert_array_equal(
            in_post_interval(times, np.array([0.0, 2.0]), 0.3),
            [True, True, False],
        )
        np.testing.assert_array_equal(
            in_post_interval(times, np.array([]), 0.3),
            [False, False, False],
        )
        self.assertEqual(len(in_post_interval(np.array([]), np.array([1.0]), 0.3)), 0)


if __name__ == "__main__":
    unittest.main()
