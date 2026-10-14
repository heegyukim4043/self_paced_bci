"""No boundary-crossing windows or duplicate EEG exposure in summaries."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_lee2019_original_continuous_strict_windows_20261009 import (
    full_support_mask, support_union_seconds,
)


class TestStrictSupport(unittest.TestCase):
    def test_whole_window_not_endpoint_only(self):
        ends = np.array([2.4, 2.5, 3.0, 3.5, 3.6])
        np.testing.assert_array_equal(
            full_support_mask(ends, np.array([[.5, 3.5]])),
            [False, True, True, True, False],
        )

    def test_union_counts_overlap_once(self):
        self.assertAlmostEqual(support_union_seconds(np.array([2., 2.1, 2.2])), 2.2)
        self.assertAlmostEqual(support_union_seconds(np.array([2., 5.])), 4.0)
        self.assertEqual(support_union_seconds(np.array([])), 0.0)


if __name__ == "__main__":
    unittest.main()
