import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.event_evaluation import (
    circular_shift_onset_null, false_alarms_per_minute, interval_metrics,
    onset_metrics, predicted_intervals,
)


class EventEvaluationTests(unittest.TestCase):
    def test_onset_tolerance_is_explicit(self):
        result = onset_metrics(np.arange(0, 6, .5), np.array([0,0,0,1,0,0,0,0,0,0,0,0], bool), [1.0], [.4, .5])
        self.assertEqual(result["0.4"].sensitivity, 0.0)
        self.assertEqual(result["0.5"].sensitivity, 1.0)

    def test_false_alarm_rate_uses_valid_idle_exposure(self):
        alarms = np.array([1, 0, 1, 1], bool)
        idle = np.array([1, 1, 0, 1], bool)
        self.assertAlmostEqual(false_alarms_per_minute(alarms, idle, .5), 80.0)

    def test_interval_metrics_separate_coverage_from_binary_f1(self):
        result = interval_metrics([[1, 2]], [[1, 5]], .7, .7)
        self.assertAlmostEqual(result.coverage, .25)
        self.assertEqual(result.f1, 0.0)

    def test_interval_reconstruction_uses_post_alarm_statistic(self):
        times = np.arange(0, 1, .1)
        alarms = np.zeros(10, bool); alarms[2] = True
        statistic = np.array([0,0,2,2,2,1,0,0,0,0], float)
        intervals = predicted_intervals(times, alarms, statistic, 1.5)
        self.assertEqual(intervals.shape, (1, 2))
        self.assertGreater(intervals[0, 1], intervals[0, 0])

    def test_circular_null_is_deterministic(self):
        times = np.arange(100, dtype=float) * .1
        alarms = np.zeros(100, bool); alarms[[10, 40, 70]] = True
        first = circular_shift_onset_null(times, alarms, [1, 4, 7], [1.0], shifts=10, minimum_shift_seconds=1, seed=4)
        second = circular_shift_onset_null(times, alarms, [1, 4, 7], [1.0], shifts=10, minimum_shift_seconds=1, seed=4)
        self.assertEqual(first, second)

    def test_rejects_noncausal_time_order(self):
        with self.assertRaises(ValueError):
            onset_metrics([0, 2, 1], [False, True, False], [1], [1])


if __name__ == "__main__":
    unittest.main()
