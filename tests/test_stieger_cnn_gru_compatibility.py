import unittest

import numpy as np

from scripts.audit_stieger_cnn_gru_compatibility import (
    chronological_split_counts,
    complete_pre_event_window_count,
)


class StiegerCnnGruCompatibilityTests(unittest.TestCase):
    def test_pre_event_windows_respect_warmup_and_event(self):
        count = complete_pre_event_window_count(
            cell_start_s=-2.0,
            event_s=2.0,
            sample_rate_hz=1000.0,
        )
        self.assertEqual(count, 11)

    def test_no_pre_event_window_when_support_is_too_short(self):
        count = complete_pre_event_window_count(
            cell_start_s=0.0,
            event_s=2.5,
            sample_rate_hz=1000.0,
        )
        self.assertEqual(count, 0)

    def test_chronological_first10_cutoff_and_future_counts(self):
        # 40 training cells, then a locked sequence alternating active/rest.
        targets = np.array([1, 4] * 40)
        artifacts = np.zeros(80, dtype=int)
        lengths = np.full(80, 2.0)
        result = chronological_split_counts(
            targets,
            artifacts,
            lengths,
            split_index=40,
        )
        self.assertEqual(result["training"]["active_trials"], 20)
        self.assertEqual(result["screening_first10_active"]["active_trials"], 10)
        self.assertEqual(result["screening_first10_active"]["intentional_rest_trials"], 9)
        self.assertEqual(result["screening_cutoff_trial_index"], 58)
        self.assertEqual(result["future_after_screening"]["active_trials"], 10)
        self.assertEqual(result["future_after_screening"]["intentional_rest_trials"], 11)

    def test_subwindow_feedback_is_onset_usable_but_substep_is_not(self):
        targets = np.array([1, 4, 1, 4, 1, 4, 1, 4])
        artifacts = np.array([0, 0, 0, 0, 0, 1, 0, 0])
        lengths = np.array([2.0, 2.0, 2.0, 2.0, 0.05, 2.0, 0.5, 2.0])
        result = chronological_split_counts(
            targets,
            artifacts,
            lengths,
            split_index=4,
            first_active=1,
        )
        self.assertEqual(result["screening_cutoff_trial_index"], 6)
        self.assertEqual(result["future_after_screening"]["intentional_rest_trials"], 1)
        self.assertEqual(result["unusable_artifact_trials"], 1)
        self.assertEqual(result["unusable_feedback_shorter_than_one_step"], 1)
        self.assertEqual(result["feedback_shorter_than_full_window_trials"], 2)


if __name__ == "__main__":
    unittest.main()
