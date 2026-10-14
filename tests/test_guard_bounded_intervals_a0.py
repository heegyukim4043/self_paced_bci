"""Regression controls for interval boundaries in the pseudo-uncued replay."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_lee2019_locked_selfpaced_metrics import intervals_from_alarm


class GuardBoundedIntervalsA0Tests(unittest.TestCase):
    def test_release_at_guard_stops_at_last_allowed_sample(self):
        times = np.arange(20) * .1
        allowed = np.r_[np.ones(10, bool), np.zeros(5, bool), np.ones(5, bool)]
        alarms = np.zeros(20, bool)
        alarms[7] = True
        result = intervals_from_alarm(times, alarms, np.ones(20), .5, allowed)
        np.testing.assert_allclose(result, [[.7, .9]])

    def test_gap_merge_never_bridges_guard(self):
        times = np.arange(15) * .1
        allowed = np.r_[np.ones(7, bool), np.zeros(1, bool), np.ones(7, bool)]
        alarms = np.zeros(15, bool)
        alarms[[5, 9]] = True
        result = intervals_from_alarm(times, alarms, np.ones(15), .5, allowed,
                                      gap=.5)
        np.testing.assert_allclose(result, [[.5, .6], [.9, 1.4]])

    def test_alarm_inside_guard_is_rejected(self):
        times = np.arange(5) * .1
        alarms = np.array([False, True, False, False, False])
        allowed = np.array([True, False, False, True, True])
        with self.assertRaises(ValueError):
            intervals_from_alarm(times, alarms, np.ones(5), .5, allowed)


if __name__ == "__main__":
    unittest.main()
