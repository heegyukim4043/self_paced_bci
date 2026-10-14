"""E4 pre-result checks: full trials, carryover masks, and state continuity."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preflight_e4_full_trial_blocks_20261009 import choose_block, trial_masks
from run_e4_original_continuous_fp_20261009 import _count_masked, _trial_strata, _union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uncued_mi.decision import guard_rearmed_alarms


class TestE4FullTrial(unittest.TestCase):
    def test_full_trial_selection_near_uses_latest_trials(self):
        trials = {}
        for i in range(10, 20):
            fix = np.zeros(100, bool)
            early = np.zeros(100, bool)
            late = np.zeros(100, bool)
            fix[(i - 10) * 10:(i - 10) * 10 + 3] = True
            early[(i - 10) * 10 + 3:(i - 10) * 10 + 5] = True
            late[(i - 10) * 10 + 5:(i - 10) * 10 + 10] = True
            trials[i] = {"fixation": fix, "iti_early": early, "iti_late": late}
        # 56 steps required by production; here each whole trial is 10 steps.
        # A selection of 6 trials must retain whole units, ending at trial 19.
        import preflight_e4_full_trial_blocks_20261009 as geom
        old = geom.MIN_STEPS
        try:
            geom.MIN_STEPS = 56
            near = choose_block(trials, 10, 19, True, False)
            self.assertEqual(near["trial_ids"], [14, 15, 16, 17, 18, 19])
            self.assertEqual(near["endpoint_steps"], 60)
            self.assertAlmostEqual(near["components_seconds"]["iti_early"], 1.2)
        finally:
            geom.MIN_STEPS = old

    def test_early_iti_conservative_boundary(self):
        times = np.round(np.arange(0, 15, .1), 1)
        strata = trial_masks(times, 5., np.array([2., 5.]),
                             np.array([[9.5, 13.]]))
        # Nominal MI ended at 9.0; 2-s window overlap ends at 11.0.
        # Sensitivity excludes through 11.5, including a 0.5-s buffer.
        self.assertTrue(strata["iti_early"][np.where(times == 11.4)[0][0]])
        self.assertFalse(strata["iti_early"][np.where(times == 11.5)[0][0]])
        self.assertTrue(strata["iti_late"][np.where(times == 11.5)[0][0]])
        self.assertFalse(_union(strata, True)[np.where(times == 10.)[0][0]])

    def test_count_only_mask_does_not_reset_detector(self):
        ma = np.array([-1., 1., 1., 1., 1., -1., 1., 1., 1.])
        mask = np.array([False, False, False, True, False,
                         False, False, False, True])
        expected = guard_rearmed_alarms(ma, 0., np.ones(len(ma), bool),
                                        dwell_steps=3, refractory_steps=10)
        self.assertEqual(int(np.sum(expected & mask)),
                         _count_masked(ma, mask, 0., len(ma)))
        # The first alarm at index 3 occurs after 3 above-threshold values,
        # although most of those values are outside the count mask.
        self.assertTrue(expected[3])

    def test_ewma_and_detector_do_not_inspect_future_for_calibration(self):
        ma = np.array([-1., .2, .5, .8, -1., .1, .3, .6])
        mask = np.array([True, True, True, True, False, False, False, False])
        original = _count_masked(ma, mask, .4, 4)
        changed = ma.copy()
        changed[4:] = 100.
        self.assertEqual(original, _count_masked(changed, mask, .4, 4))

    def test_jit_alarm_count_matches_reference_for_random_thresholds(self):
        rng = np.random.default_rng(20261009)
        ma = rng.normal(size=200)
        ma[:4] = np.nan
        mask = rng.random(200) < .35
        for threshold in (-2., -.5, 0., .5, 2.):
            reference = guard_rearmed_alarms(
                ma, threshold, np.ones(len(ma), bool),
                dwell_steps=3, refractory_steps=10)
            self.assertEqual(int(np.sum(reference & mask)),
                             _count_masked(ma, mask, threshold, len(ma)))


if __name__ == "__main__":
    unittest.main()
