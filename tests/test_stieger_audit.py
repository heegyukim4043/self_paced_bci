import unittest

import numpy as np

from scripts.audit_stieger_raw import (
    feedback_speed,
    phase_masks,
    robust_range_overlap,
)


class StiegerAuditTests(unittest.TestCase):
    def test_documented_phase_masks_do_not_overlap(self):
        time_ms = np.arange(-2000, 7001)
        masks = phase_masks(time_ms, trial_length_s=4.0)
        stacked = np.stack(list(masks.values()))
        self.assertTrue(np.all(stacked.sum(axis=0) == 1))
        self.assertEqual(masks["pre_target_idle"].sum(), 2000)
        self.assertEqual(masks["target_only"].sum(), 2000)
        self.assertEqual(masks["feedback"].sum(), 4000)
        self.assertEqual(masks["post_feedback"].sum(), 1001)

    def test_feedback_speed_uses_only_inside_mask_differences(self):
        x = np.arange(6, dtype=float)
        y = np.zeros(6)
        mask = np.array([False, True, True, True, False, False])
        speed = feedback_speed(x, y, mask, sfreq=10.0)
        np.testing.assert_allclose(speed, [10.0, 10.0])

    def test_feedback_speed_accepts_one_dimensional_cursor(self):
        x = np.arange(6, dtype=float)
        y = np.full(6, np.nan)
        mask = np.array([False, True, True, True, False, False])
        speed = feedback_speed(x, y, mask, sfreq=10.0)
        np.testing.assert_allclose(speed, [10.0, 10.0])

    def test_robust_range_overlap(self):
        first = np.linspace(0, 10, 101)
        second = np.linspace(5, 15, 101)
        self.assertGreater(robust_range_overlap(first, second), 0)
        self.assertEqual(
            robust_range_overlap(np.arange(10), np.arange(10) + 100),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
