"""Synthetic-only regression checks for E4b update timing and decision state."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_e4b_task_idle_20261010 import rearmed_with_updates
from uncued_mi.decision import guard_rearmed_alarms


class E4bUpdateTests(unittest.TestCase):
    def test_no_update_matches_frozen_detector(self):
        ma = np.array([-1., 0., 2., 2., 2., -1., 2., 2., 2.,
                       np.nan, -1., 2., 2., 2.])
        got = rearmed_with_updates(ma, np.array([1.]), np.empty(0, dtype=np.int64))
        expected = guard_rearmed_alarms(ma, 1., np.ones(len(ma), dtype=bool), 3, 10)
        np.testing.assert_array_equal(got, expected)

    def test_preupdate_identity_and_sham_reset(self):
        ma = np.array([-1., 2., 2., 2., -1., 2., 2., 2.,
                       -1., 2., 2., 2., -1., 2., 2., 2.,
                       -1., 2., 2., 2.])
        at = np.array([8, 16], dtype=np.int64)
        static = guard_rearmed_alarms(ma, 1., np.ones(len(ma), dtype=bool), 3, 10)
        sham = rearmed_with_updates(ma, np.array([1., 1., 1.]), at)
        changed = rearmed_with_updates(ma, np.array([1., 3., 1.]), at)
        np.testing.assert_array_equal(static[:at[0]], sham[:at[0]])
        np.testing.assert_array_equal(static[:at[0]], changed[:at[0]])
        self.assertFalse(changed[8:16].any())

    def test_update_cannot_fire_from_existing_above_threshold_run(self):
        # First post-update values are high. The detector must see below the
        # new threshold before it can count a new dwell.
        ma = np.array([-1., 2., 2., 2., 2., 2., 2., 2.,
                       2., 2., 2., -1., 2., 2., 2.])
        got = rearmed_with_updates(ma, np.array([1., 1.]), np.array([8], dtype=np.int64))
        self.assertFalse(got[8:11].any())
        self.assertTrue(got[14])

    def test_refractory_survives_update(self):
        ma = np.array([-1., 2., 2., 2., -1., 2., 2., 2.,
                       -1., 2., 2., 2., -1.])
        got = rearmed_with_updates(ma, np.array([1., 1.]), np.array([8], dtype=np.int64))
        self.assertTrue(got[3])
        self.assertFalse(got[11])  # still within original 10-step refractory


if __name__ == "__main__":
    unittest.main()
