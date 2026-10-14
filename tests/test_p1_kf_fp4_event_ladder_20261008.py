"""Support and aggregation checks for the post-hoc K_F event-count ladder."""

import unittest

import numpy as np

from scripts.audit_p1_kf_fp4_event_ladder_20261008 import (
    balanced_calibration_mask, bootstrap_summary,
)


class P1KFFP4EventLadderTests(unittest.TestCase):
    def test_balanced_mask_has_disjoint_equal_source_support(self):
        masks = [np.zeros(1200, dtype=bool) for _ in range(3)]
        masks[0][0:400] = True
        masks[1][400:800] = True
        masks[2][800:1200] = True
        for length in (15, 30, 60):
            selected, sources = balanced_calibration_mask(masks, 1, length)
            self.assertEqual(sources, (0, 2))
            self.assertEqual(int(selected.sum()), length * 10)
            self.assertEqual(int((selected & masks[0]).sum()), length * 5)
            self.assertEqual(int((selected & masks[2]).sum()), length * 5)
            self.assertFalse(np.any(selected & masks[1]))

    def test_two_blocks_must_have_enough_unique_steps(self):
        masks = [np.zeros(1200, dtype=bool) for _ in range(3)]
        masks[0][0:200] = True
        masks[1][400:800] = True
        masks[2][800:1200] = True
        with self.assertRaises(ValueError):
            balanced_calibration_mask(masks, 1, 60)

    def test_bias_variance_decomposition(self):
        matrix = np.array([[-1., 0., 1.], [1., 2., 3.]])
        result = bootstrap_summary(matrix, 200, 42)
        self.assertAlmostEqual(result["cell_sd"] ** 2,
                               result["between_subject_sd"] ** 2
                               + result["within_subject_cell_sd"] ** 2)
        self.assertAlmostEqual(result["mean_subject_signed_fp_rate_bias"], 1.)


if __name__ == "__main__":
    unittest.main()
