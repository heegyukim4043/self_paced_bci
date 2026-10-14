import unittest

import numpy as np

from scripts.run_stieger_riemannian_smoke import (
    grouped_alarm_summary,
    speed_matched_indices,
)


class StiegerSmokeTests(unittest.TestCase):
    def test_decision_state_resets_at_trial_seam(self):
        trials = np.repeat([1, 2], 5)
        scores = np.ones(10)
        summary = grouped_alarm_summary(trials, scores, threshold=0.0)
        self.assertEqual(set(summary), {1, 2})
        self.assertEqual(summary[1]["alarm_count"], summary[2]["alarm_count"])

    def test_speed_matching_balances_labels(self):
        speed = np.tile(np.linspace(0, 1, 20), 2)
        labels = np.repeat([0, 1], 20)
        selected = speed_matched_indices(speed, labels, bins=4)
        self.assertEqual(np.sum(labels[selected] == 0), np.sum(labels[selected] == 1))


if __name__ == "__main__":
    unittest.main()
