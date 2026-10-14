"""Cheap P1 rescoring safeguards; no EEG model training is run here."""

import unittest

import numpy as np

from scripts.rescore_lee2019_p1_fixed_target_20261008 import (
    anchor_grid,
    check_original_anchor,
    out_of_training_mask,
)
from scripts.decompose_calibration_error_20261008 import (
    alarms_exact, compress_selected_ma, tune_pre_rest,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from uncued_mi.decision import guard_rearmed_alarms
from uncued_mi.guard_rearm_calibration import tune_guard_rearmed_anchor


class P1FixedTargetRestRescoreTests(unittest.TestCase):
    def test_locked_pre_rest_grid_has_571_scores(self):
        starts, ends = anchor_grid(60 * 250)
        self.assertEqual(len(ends), 571)
        self.assertAlmostEqual(ends[0], 3.0)
        self.assertAlmostEqual(ends[-1], 60.0)
        self.assertAlmostEqual(starts[0], 2.0)

    def test_cnn_donor_mask_excludes_training_half_and_straddling_windows(self):
        starts, ends = anchor_grid(60 * 250)
        eligible = out_of_training_mask(starts, 30.0)
        self.assertEqual(int(eligible.sum()), 291)
        self.assertTrue(np.all(starts[eligible] >= 30.0))
        self.assertTrue(np.all(ends[~eligible] < 31.0))
        self.assertLess(eligible.sum() * .1, 36.0)

    def test_original_anchor_mismatch_stops_scoring(self):
        original = np.array([0.0, 1.0, 2.0])
        self.assertEqual(check_original_anchor(original.copy(), original, name="test"), 0.0)
        with self.assertRaises(AssertionError):
            check_original_anchor(original + .01, original, name="test")
        with self.assertRaises(AssertionError):
            check_original_anchor(original[:2], original, name="test")

    def test_future_rest_compression_preserves_guard_rearmed_count(self):
        ma = np.asarray([
            np.nan, -.5, .8, .9, 1.2, -.4, .7, .8, 1.1,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            -.6, .9, 1.0, 1.2, -.5, 1.1, 1.2, 1.3,
        ])
        allowed = np.isfinite(ma)
        selected = allowed.copy()
        compressed_ma, compressed_allowed = compress_selected_ma(ma, selected)
        for threshold in (-.2, .5, .75, 1., 1.5):
            full = guard_rearmed_alarms(ma, threshold, allowed, 3, 10)
            compressed = guard_rearmed_alarms(
                compressed_ma, threshold, compressed_allowed, 3, 10,
            )
            self.assertEqual(int(full.sum()), int(compressed.sum()))

    def test_optional_jit_alarm_loop_matches_reference(self):
        rng = np.random.default_rng(42)
        statistic = rng.normal(size=300)
        statistic[30:36] = np.nan
        allowed = rng.random(300) > .12
        for threshold in (-2., -.5, 0., .8, 2.):
            self.assertTrue(np.array_equal(
                alarms_exact(statistic, threshold, allowed),
                guard_rearmed_alarms(statistic, threshold, allowed, 3, 10),
            ))

    def test_pre_rest_threshold_matches_original_exact_sweep(self):
        raw = np.random.default_rng(43).normal(size=280)
        z = ewma_z(raw, 0., 1., np.ones(len(raw), dtype=bool))
        original = tune_guard_rearmed_anchor(z, 2., .1, 5, 3, 10)
        replacement = tune_pre_rest(raw, 0., 1., 28, 2.)
        self.assertEqual(replacement["threshold"], original.threshold)
        self.assertEqual(replacement["calibration_alarm_count"], original.anchor_alarm_count)


if __name__ == "__main__":
    unittest.main()
