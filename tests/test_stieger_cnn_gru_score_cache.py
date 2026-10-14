import unittest

import numpy as np
import torch

from scripts.prepare_stieger_cnn_gru_score_cache import (
    CausalCNNGRU,
    cell_window_ends,
    model_causality_and_reset_audit,
    times_from_feedback,
    window_labels,
)


class StiegerCnnGruScoreCacheTests(unittest.TestCase):
    def test_windows_begin_after_warmup_and_straddle_feedback(self):
        ends = cell_window_ends(0.5, signal_samples=1200)
        np.testing.assert_array_equal(ends[:3], [750, 775, 800])
        self.assertEqual(ends[-1], 1125)
        times = times_from_feedback(ends)
        self.assertAlmostEqual(float(times[0]), -1.0)
        self.assertAlmostEqual(float(times[-1]), 0.5)

    def test_fast_feedback_retains_post_onset_step(self):
        ends = cell_window_ends(0.1, signal_samples=1100)
        times = times_from_feedback(ends)
        self.assertTrue(np.any(times > 0))
        self.assertAlmostEqual(float(times[-1]), 0.1, places=6)

    def test_labels_only_active_feedback_as_positive(self):
        times = np.array([-1.0, 0.0, 0.1, 0.2])
        np.testing.assert_array_equal(window_labels(times, 1), [0, 0, 1, 1])
        np.testing.assert_array_equal(window_labels(times, 4), [0, 0, 0, 0])

    def test_architecture_prefix_and_reset_equivalence(self):
        rng = np.random.default_rng(7)
        signals = [
            rng.normal(size=(3, 20, 1200)).astype(np.float32),
            rng.normal(size=(3, 20, 1200)).astype(np.float32),
        ]
        ends = [cell_window_ends(0.5, 1200), cell_window_ends(0.5, 1200)]
        model = CausalCNNGRU(20, 32)
        audit = model_causality_and_reset_audit(
            model, signals, ends, torch.device("cpu")
        )
        self.assertTrue(audit["passed"])
        self.assertLessEqual(audit["prefix_causality_max_abs_error"], 1e-5)
        self.assertLessEqual(audit["independent_vs_batched_reset_max_abs_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
