import unittest

import numpy as np

from scripts.prepare_stieger_riemannian_cache import (
    feedback_window_ends,
    selected_indices,
    window_cursor_speeds,
)


class StiegerCacheTests(unittest.TestCase):
    def test_feedback_windows_have_full_support(self):
        ends = feedback_window_ends(3.44, 250.0)
        self.assertEqual(ends[0], 1250)
        self.assertLessEqual(ends[-1], int((4.0 + 3.44) * 250))
        self.assertTrue(np.all(np.diff(ends) == 25))
        self.assertTrue(np.all(ends - 250 >= 1000))

    def test_short_feedback_has_no_window(self):
        self.assertEqual(len(feedback_window_ends(0.8, 250.0)), 0)

    def test_cursor_speed_is_window_aligned(self):
        x = np.arange(40, dtype=float)
        y = np.zeros(40)
        speeds = window_cursor_speeds(
            x, y, native_sfreq=10.0, target_sfreq=10.0,
            end_samples=np.array([20, 30]), window_samples=10,
        )
        np.testing.assert_allclose(speeds, [10.0, 10.0])

    def test_cursor_speed_accepts_unused_nan_axis(self):
        x = np.arange(40, dtype=float)
        y = np.full(40, np.nan)
        speeds = window_cursor_speeds(
            x, y, native_sfreq=10.0, target_sfreq=10.0,
            end_samples=np.array([20]), window_samples=10,
        )
        np.testing.assert_allclose(speeds, [10.0])

    def test_channel_matching_is_case_insensitive(self):
        indices = selected_indices(["FC3", "cz", "CP4"], ("fc3", "Cz"))
        np.testing.assert_array_equal(indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
