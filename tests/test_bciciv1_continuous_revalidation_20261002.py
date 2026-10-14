"""Structural checks for the native-continuous BCI IV-1 revalidation."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from revalidate_bciciv1_continuous_20261002 import (
    chronological_threshold, rearmed,
)
from audit_bciciv1_calibration_factorial_20261002 import stage_masks
from uncued_mi.decision import guard_local_moving_average


class ContinuousRevalidationTests(unittest.TestCase):
    def test_constant_stream_never_fires_at_any_level(self):
        for value, threshold in ((0.0, -1.0), (0.0, 0.0), (1.0, 2.0)):
            ma = guard_local_moving_average(
                np.full(100, value), np.ones(100, bool), ma_steps=5,
            )
            self.assertFalse(np.any(rearmed(ma, threshold)))

    def test_calibration_counts_only_idle_alarms_but_keeps_full_timeline(self):
        # The high-score excursion in the middle is non-idle. It must remain
        # in the decision history, but cannot count as a calibration FP.
        ma = np.r_[np.zeros(30), np.ones(30) * 5.0,
                   np.zeros(30), np.ones(30) * 4.0, np.zeros(30)]
        idle = np.r_[np.ones(30, bool), np.zeros(30, bool),
                     np.ones(30, bool), np.zeros(30, bool),
                     np.ones(30, bool)]
        threshold, achieved = chronological_threshold(ma, idle, target_fp=2.0)
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(achieved, 0.0)
        self.assertEqual(int(np.sum(rearmed(ma, threshold) & idle)), 0)

    def test_three_way_split_excludes_cross_boundary_window_support(self):
        cues = np.arange(200) * 1000 + 2000  # 100 Hz, 10 s cue spacing
        times = np.arange(10, 2010, 0.1)
        train, anchor, threshold, boundaries = stage_masks(times, cues)
        self.assertFalse(np.any(train & anchor))
        self.assertFalse(np.any(anchor & threshold))
        self.assertTrue(np.all(times[anchor] - 1.0 >= boundaries[0]))
        self.assertTrue(np.all(times[anchor] <= boundaries[1]))
        self.assertTrue(np.all(times[threshold] - 1.0 >= boundaries[1]))


if __name__ == "__main__":
    unittest.main()
