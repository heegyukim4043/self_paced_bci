"""Cheap E2 estimator/state-machine tests; no cohort score calculation."""

import unittest

import numpy as np

from scripts.decompose_calibration_error_20261008 import _tune_ma
from scripts.run_e2_pooled_alarm_offset_20261008 import (
    DELTA_GRID, beta_from_points, choose_offset, choose_point,
    episode_diagnostics, exact_sweep, preflight_rearm,
)


class E2PooledAlarmOffsetTests(unittest.TestCase):
    def test_exact_sweep_matches_frozen_p1_tie_rule(self):
        rng = np.random.default_rng(5)
        ma = rng.normal(size=60)
        ma[:4] = np.nan
        sweep = exact_sweep(ma, seconds=6)
        for target in (2.0, 16.0):
            expected = _tune_ma(ma, np.ones(len(ma), dtype=bool), 6, target)
            observed = choose_point(sweep, target)
            self.assertEqual(observed["threshold"], expected["threshold"])
            self.assertEqual(observed["alarm_count"], expected["calibration_alarm_count"])

    def test_offset_uses_pooled_alarm_count_and_conservative_tie(self):
        counts = np.full((2, len(DELTA_GRID)), 3)
        counts[:, 1:] = 1
        counts[:, 2:] = 0
        point = choose_offset(counts, donor_seconds=60)
        self.assertTrue(point["valid"])
        self.assertEqual(point["delta"], 0.05)
        self.assertEqual(point["pooled_alarm_count"], 2)

    def test_continuous_run_rearms_only_once(self):
        preflight_rearm()
        ma = np.r_[np.full(5, -1.0), np.full(50, 1.0), np.full(5, -1.0)]
        self.assertEqual(episode_diagnostics(ma, 0.0)["alarm_count"], 1)
        self.assertEqual(episode_diagnostics(ma, 0.0)["merged_1s_episodes"], 1)
        self.assertEqual(episode_diagnostics(ma, 0.0)["multi_alarm_episode_fraction"], 0.0)

    def test_rearm_with_short_dip_can_fire_twice_in_one_merged_episode(self):
        ma = np.r_[np.full(5, -1.0), np.full(5, 1.0), np.full(8, -1.0),
                   np.full(5, 1.0), np.full(5, -1.0)]
        diag = episode_diagnostics(ma, 0.0)
        self.assertEqual(diag["alarm_count"], 2)
        self.assertEqual(diag["merged_1s_episodes"], 1)
        self.assertEqual(diag["multi_alarm_episode_fraction"], 1.0)

    def test_beta_requires_three_distinct_thresholds(self):
        points = {rate: {"threshold": 1.0, "achieved_rate": rate}
                  for rate in (8.0, 12.0, 16.0, 24.0)}
        self.assertTrue(np.isnan(beta_from_points(points)))


if __name__ == "__main__":
    unittest.main()
