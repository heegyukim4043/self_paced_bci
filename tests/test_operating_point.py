import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.operating_point import (
    ScoredSequence,
    tune_source_operating_point,
    tune_source_operating_points,
)


class OperatingPointTests(unittest.TestCase):
    def test_threshold_tuning_returns_finite_value(self):
        sequence = ScoredSequence(
            times=np.arange(0.0, 20.0, 0.5),
            scores=np.linspace(-1.0, 2.0, 40),
            idle_intervals=np.array([[0.0, 10.0]]),
            command_intervals=np.array([[10.0, 20.0]]),
        )
        result = tune_source_operating_point(
            [sequence],
            decision="threshold",
            target_fp_per_minute=6.0,
            step_seconds=0.5,
            grid_size=20,
        )
        self.assertTrue(np.isfinite(result.threshold))

    def test_multiple_targets_share_curve(self):
        sequence = ScoredSequence(
            times=np.arange(0.0, 20.0, 0.5),
            scores=np.linspace(-1.0, 2.0, 40),
            idle_intervals=np.array([[0.0, 10.0]]),
            command_intervals=np.array([[10.0, 20.0]]),
        )
        results = tune_source_operating_points(
            [sequence],
            decision="threshold",
            target_fp_per_minute=[1.0, 2.0, 4.0],
            step_seconds=0.5,
            grid_size=20,
        )
        self.assertEqual(set(results), {1.0, 2.0, 4.0})


if __name__ == "__main__":
    unittest.main()
