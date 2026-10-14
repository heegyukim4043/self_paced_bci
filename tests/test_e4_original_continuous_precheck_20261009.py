"""E4 metadata geometry: operational endpoints versus strict EEG support."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from precheck_e4_original_continuous_20261009 import following_iti, in_intervals


class TestE4Precheck(unittest.TestCase):
    def test_endpoint_can_contain_prior_context_but_strict_cannot(self):
        ends = np.array([10.0, 10.1, 11.9, 12.0, 12.1, 13.0])
        interval = np.array([[10.0, 13.0]])
        np.testing.assert_array_equal(in_intervals(ends, interval),
                                      [True, True, True, True, True, False])
        np.testing.assert_array_equal(in_intervals(ends, interval, full_support=True),
                                      [False, False, False, True, True, False])

    def test_following_task_iti_is_not_an_unrelated_rest(self):
        intervals = np.array([[14.5, 17.], [29.5, 32.]])
        np.testing.assert_array_equal(following_iti(intervals, 10.),
                                      [[14.5, 17.]])
        self.assertEqual(following_iti(intervals, 12.).shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
