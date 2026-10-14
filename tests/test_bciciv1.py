import unittest

import numpy as np

from scripts.run_bciciv1_riemannian_sanity import (
    contiguous_events,
    downsample_true_labels,
    support_labels,
)
from scripts.evaluate_bciciv1_locked_selfpaced_metrics import ground_truth_intervals


class BCICIV1Tests(unittest.TestCase):
    def test_label_downsample_preserves_guard(self):
        labels = np.r_[np.zeros(10), np.full(10, np.nan), np.ones(10)]
        output = downsample_true_labels(labels)
        self.assertEqual(output[0], 0)
        self.assertTrue(np.isnan(output[1]))
        self.assertEqual(output[2], 1)

    def test_support_label_requires_whole_window(self):
        labels = np.r_[np.zeros(5), np.ones(5), np.ones(5)]
        output = support_labels(labels, np.array([5, 10, 15]), 5)
        np.testing.assert_array_equal(output, [0, 1, 1])
        self.assertTrue(np.isnan(support_labels(labels, np.array([8]), 5)[0]))

    def test_contiguous_events(self):
        events = contiguous_events([False, True, True, False, True])
        self.assertEqual([event.tolist() for event in events], [[1, 2], [4]])

    def test_ground_truth_intervals_use_original_sample_clock(self):
        labels = np.array([0, 1, 1, 0, np.nan, 2, 2, 2, 0], dtype=float)
        np.testing.assert_allclose(
            ground_truth_intervals(labels),
            [[0.01, 0.03], [0.05, 0.08]],
        )


if __name__ == "__main__":
    unittest.main()
