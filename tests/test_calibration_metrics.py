import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.calibration import LocationScale, empirical_quantile_map
from uncued_mi.metrics import chance_tpr, evaluate_alarms


class CalibrationMetricTests(unittest.TestCase):
    def test_location_scale(self):
        model = LocationScale.fit(np.array([1.0, 2.0, 3.0]))
        transformed = model.transform(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(np.mean(transformed)), 0.0)

    def test_median_mad_resists_single_outlier(self):
        values = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        robust = LocationScale.fit_median_mad(values)
        classical = LocationScale.fit(values)
        self.assertAlmostEqual(float(robust.location), 2.0)
        self.assertLess(float(robust.scale), float(classical.scale))

    def test_quantile_map(self):
        mapped = empirical_quantile_map(
            np.array([10.0, 20.0]),
            source_idle=np.array([0.0, 1.0]),
            target_idle=np.array([10.0, 20.0]),
        )
        np.testing.assert_allclose(mapped, [0.0, 1.0])

    def test_event_metrics(self):
        result = evaluate_alarms(
            times=np.array([1.0, 5.5, 10.5]),
            alarms=np.array([True, True, True]),
            idle_intervals=np.array([[0.0, 2.0], [9.0, 11.0]]),
            command_intervals=np.array([[5.0, 7.0]]),
        )
        self.assertEqual(result.false_alarms, 2)
        self.assertEqual(result.detected_events, 1)
        self.assertAlmostEqual(result.fp_per_minute, 30.0)
        self.assertAlmostEqual(result.median_delay_seconds, 0.5)
        self.assertEqual(result.delay_seconds, (0.5,))

    def test_chance_tpr(self):
        self.assertAlmostEqual(chance_tpr(1.0, 3.0), 1.0 - np.exp(-0.05))

    def test_command_only_sequence_has_undefined_fp_rate(self):
        result = evaluate_alarms(
            times=np.array([1.0]),
            alarms=np.array([True]),
            idle_intervals=np.empty((0, 2)),
            command_intervals=np.array([[0.0, 2.0]]),
        )
        self.assertTrue(np.isnan(result.fp_per_minute))
        self.assertEqual(result.detected_events, 1)


if __name__ == "__main__":
    unittest.main()
