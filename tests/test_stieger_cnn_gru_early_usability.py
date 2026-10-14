import unittest

import numpy as np

from scripts.analyze_stieger_cnn_gru_early_usability import (
    causal_ma,
    ewma_trial,
    shifted_null,
    trial_alarms,
)


class StiegerCnnGruEarlyUsabilityTests(unittest.TestCase):
    def test_trial_ewma_resets_deterministically(self):
        raw = np.array([0.0, 0.5, 1.0, 0.0])
        first = ewma_trial(raw, 0.0, 1.0)
        second = ewma_trial(raw, 0.0, 1.0)
        np.testing.assert_allclose(first, second)

    def test_alarm_requires_three_feedback_steps(self):
        times = np.arange(-1.0, 0.61, 0.1)
        level = np.ones(len(times)) * 2.0
        alarms = trial_alarms(level, times, threshold=1.0)
        alarm_times = times[alarms]
        self.assertEqual(len(alarm_times), 1)
        self.assertAlmostEqual(float(alarm_times[0]), 0.3, places=6)

    def test_preparation_never_emits_alarm(self):
        times = np.arange(-1.0, 0.01, 0.1)
        alarms = trial_alarms(np.ones(len(times)) * 10.0, times, threshold=1.0)
        self.assertFalse(np.any(alarms))

    def test_causal_ma_has_expected_warmup(self):
        level = causal_ma(np.arange(10, dtype=float), seconds=0.5)
        self.assertTrue(np.all(np.isnan(level[:4])))
        self.assertAlmostEqual(float(level[4]), 2.0)

    def test_run_shift_preserves_label_count(self):
        hit = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
        active = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
        run = np.array([1, 1, 1, 2, 2, 2])
        null = shifted_null(hit, active, run, shifts=20, seed=7)
        self.assertEqual(len(null), 20)
        self.assertTrue(np.all(np.isfinite(null)))


if __name__ == "__main__":
    unittest.main()
