import unittest

import numpy as np

from scripts.analyze_stieger_cnn_gru_rest_exposure_curve import (
    calibration_alarm_table,
    cluster_bootstrap_median,
    select_threshold_for_exposure,
)


class StiegerRestExposureTests(unittest.TestCase):
    def test_select_threshold_hits_exact_rate(self):
        thresholds = np.asarray([0.0, 1.0, 2.0])
        alarms = [np.asarray([1.0, 10.0]), np.asarray([1.0]), np.asarray([])]
        threshold, rate, count = select_threshold_for_exposure(
            thresholds, alarms, exposure_seconds=30.0, target_fp=2.0
        )
        self.assertEqual(threshold, 1.0)
        self.assertEqual(rate, 2.0)
        self.assertEqual(count, 1)

    def test_select_threshold_tie_is_conservative(self):
        thresholds = np.asarray([0.0, 1.0])
        alarms = [np.asarray([1.0]), np.asarray([])]
        threshold, rate, _ = select_threshold_for_exposure(
            thresholds, alarms, exposure_seconds=15.0, target_fp=2.0
        )
        self.assertEqual(threshold, 1.0)
        self.assertEqual(rate, 0.0)

    def test_alarm_table_preserves_trial_reset_and_offsets(self):
        trial = {
            "active": False,
            "artifact": False,
            "length": 1.0,
            "level": np.asarray([0.0, 2.0, 2.0, 2.0]),
            "time": np.asarray([-0.1, 0.1, 0.2, 0.3]),
        }
        total, positions = calibration_alarm_table(
            [trial, dict(trial)], [0, 1], np.asarray([1.0])
        )
        self.assertEqual(total, 2.0)
        np.testing.assert_allclose(positions[0], [0.3, 1.3])

    def test_subject_cluster_bootstrap_is_finite(self):
        rows = [
            {"subject": 1, "value": 1.0},
            {"subject": 1, "value": 2.0},
            {"subject": 2, "value": 3.0},
        ]
        ci = cluster_bootstrap_median(
            rows, lambda row: row["value"], seed=7, draws=100
        )
        self.assertEqual(len(ci), 2)
        self.assertTrue(np.all(np.isfinite(ci)))
        self.assertLessEqual(ci[0], ci[1])


if __name__ == "__main__":
    unittest.main()
