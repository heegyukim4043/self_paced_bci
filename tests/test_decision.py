import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.decision import cusum_style, fp_per_minute, moving_average_dwell


class DecisionTests(unittest.TestCase):
    def test_cusum_alarm_and_reset(self):
        result = cusum_style(
            np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
            threshold=1.5,
            drift=0.25,
            step_seconds=0.5,
            refractory_seconds=0.5,
        )
        self.assertEqual(
            result.alarms.tolist(), [False, False, True, False, False]
        )

    def test_cusum_forced_reset_limits_memory(self):
        result = cusum_style(
            np.full(10, 0.6),
            threshold=2.0,
            drift=0.5,
            step_seconds=1.0,
            refractory_seconds=0.0,
            max_accumulation_seconds=3.0,
        )
        self.assertFalse(np.any(result.alarms))

    def test_moving_average_dwell(self):
        result = moving_average_dwell(
            np.array([0.0, 1.0, 1.0, 1.0]),
            threshold=0.5,
            step_seconds=0.5,
            ma_seconds=0.5,
            dwell_seconds=1.0,
        )
        self.assertEqual(result.alarms.tolist(), [False, False, True, False])

    def test_fp_per_minute(self):
        self.assertEqual(fp_per_minute(2, 120.0), 1.0)

    def test_instantaneous_counts_crossing_once(self):
        from uncued_mi.decision import instantaneous

        result = instantaneous(
            np.array([0.0, 2.0, 2.0, 0.0, 2.0]),
            threshold=1.0,
            step_seconds=0.1,
            refractory_seconds=0.2,
        )
        self.assertEqual(result.alarms.tolist(), [False, True, False, False, True])


if __name__ == "__main__":
    unittest.main()
