"""A0 structural controls for guard-local MA and edge-armed alarms."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_lee2019_locked_selfpaced_metrics import alarms_from_ma
from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms


class GuardRearmA0Tests(unittest.TestCase):
    def setUp(self):
        self.allowed = np.array([False] * 4 + [True] * 10 + [False] * 4 + [True] * 10)

    def test_constant_score_never_fires_at_any_finite_threshold(self):
        for value in (-2.0, 0.0, 1.0):
            for threshold in (-3.0, 0.0, 0.5, 1.0, 3.0):
                with self.subTest(value=value, threshold=threshold):
                    ma = guard_local_moving_average(
                        np.full(len(self.allowed), value), self.allowed, ma_steps=5
                    )
                    alarm = guard_rearmed_alarms(
                        ma, threshold, self.allowed, dwell_steps=3, refractory_steps=10
                    )
                    self.assertFalse(alarm.any())

    def test_legacy_constant_score_fires_once_per_allowed_segment(self):
        ma = np.ones(len(self.allowed))
        legacy = alarms_from_ma(ma, 0.5, self.allowed, dwell=3, refractory=10)
        self.assertEqual(np.flatnonzero(legacy).tolist(), [6, 20])

    def test_ma_has_no_guard_crossing_support(self):
        scores = np.r_[np.full(4, 100.0), np.arange(10.0),
                       np.full(4, 100.0), np.arange(10.0)]
        ma = guard_local_moving_average(scores, self.allowed, ma_steps=5)
        self.assertTrue(np.isnan(ma[:8]).all())
        self.assertEqual(ma[8], 2.0)
        self.assertTrue(np.isnan(ma[14:22]).all())
        self.assertEqual(ma[22], 2.0)

    def test_high_at_guard_release_requires_new_below_then_rise(self):
        ma = np.r_[np.ones(4), np.ones(10), np.ones(4), np.ones(10)]
        self.assertFalse(guard_rearmed_alarms(ma, 0.5, self.allowed).any())
        ma[7] = 0.0
        alarm = guard_rearmed_alarms(ma, 0.5, self.allowed, dwell_steps=3)
        self.assertEqual(np.flatnonzero(alarm).tolist(), [10])

    def test_low_then_step_after_guard_fires_only_after_dwell(self):
        ma = np.r_[np.ones(4), np.zeros(3), np.ones(7),
                   np.ones(4), np.zeros(3), np.ones(7)]
        alarm = guard_rearmed_alarms(ma, 0.5, self.allowed, dwell_steps=3)
        self.assertEqual(np.flatnonzero(alarm).tolist(), [9, 23])

    def test_complete_path_uses_five_valid_ma_then_three_high_ma(self):
        allowed = np.r_[np.zeros(4, dtype=bool), np.ones(10, dtype=bool)]
        scores = np.r_[np.ones(4), np.zeros(5), np.ones(5)]
        ma = guard_local_moving_average(scores, allowed, ma_steps=5)
        alarm = guard_rearmed_alarms(ma, 0.5, allowed, dwell_steps=3)
        self.assertEqual(np.flatnonzero(alarm).tolist(), [13])

    def test_no_boundary_alarm_for_fluctuating_high_input(self):
        # Fluctuation alone is not a failure; this input never goes below tau.
        ma = np.r_[np.zeros(4), [0.8, 1.2, 0.7, 1.4, 0.6, 1.1, 0.9, 1.3, 0.8, 1.2],
                   np.zeros(4), [1.0, 0.9, 1.2, 0.6, 1.4, 0.8, 1.3, 0.7, 1.0, 1.2]]
        self.assertFalse(guard_rearmed_alarms(ma, 0.5, self.allowed).any())

    def test_equal_to_threshold_does_not_arm(self):
        ma = np.r_[np.zeros(4), np.full(10, 0.5),
                   np.zeros(4), np.full(10, 0.5)]
        self.assertFalse(guard_rearmed_alarms(ma, 0.5, self.allowed).any())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            guard_rearmed_alarms(np.ones(3), np.nan, np.ones(3, dtype=bool))
        with self.assertRaises(ValueError):
            guard_local_moving_average(np.ones(3), np.ones(2, dtype=bool))


if __name__ == "__main__":
    unittest.main()
