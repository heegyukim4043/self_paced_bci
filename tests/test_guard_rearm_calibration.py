import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from uncued_mi.guard_rearm_calibration import tune_guard_rearmed_anchor


class GuardRearmCalibrationTests(unittest.TestCase):
    def test_anchor_count_is_replayed_by_exact_evaluation_rule(self):
        z = np.array([-1.] * 5 + [1.] * 5 + [-1.] * 5 + [1.] * 5)
        point = tune_guard_rearmed_anchor(
            z, target_fp_per_min=6., step_seconds=.5,
            ma_steps=1, dwell_steps=3, refractory_steps=0,
        )
        allowed = np.ones(len(z), dtype=bool)
        ma = guard_local_moving_average(z, allowed, ma_steps=1)
        replay = guard_rearmed_alarms(
            ma, point.threshold, allowed, dwell_steps=3, refractory_steps=0,
        )
        self.assertEqual(int(replay.sum()), point.anchor_alarm_count)
        self.assertEqual(point.anchor_fp_per_min, point.anchor_alarm_count * 6.)
        self.assertEqual(point.count_resolution_fp_per_min, 6.)

    def test_constant_anchor_reports_no_attainable_positive_count(self):
        point = tune_guard_rearmed_anchor(
            np.ones(20), target_fp_per_min=2., step_seconds=.1,
        )
        self.assertEqual(point.attainable_alarm_counts, (0,))
        self.assertEqual(point.anchor_fp_per_min, 0.)
        self.assertFalse(point.within_target_band)

    def test_short_or_nonfinite_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            tune_guard_rearmed_anchor(np.ones(4), 2., .1)
        with self.assertRaises(ValueError):
            tune_guard_rearmed_anchor(np.array([0., np.nan, 1.]), 2., .1)


if __name__ == "__main__":
    unittest.main()
