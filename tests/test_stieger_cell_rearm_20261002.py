import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_stieger_cell_rearm_20261002 import cell_alarms


class CellRearmTests(unittest.TestCase):
    def test_constant_high_cell_does_not_fire(self):
        times = np.round(np.arange(-1.0, 2.0, 0.1), 1)
        for pre in (True, False):
            self.assertFalse(np.any(cell_alarms(
                np.ones(len(times)), times, 0.5,
                arm_before_feedback=pre,
            )))

    def test_pre_feedback_baseline_changes_early_eligibility(self):
        times = np.round(np.arange(-1.0, 2.0, 0.1), 1)
        scores = np.ones(len(times))
        scores[times < 0] = 0.0
        pre = cell_alarms(scores, times, 0.5, arm_before_feedback=True)
        post = cell_alarms(scores, times, 0.5, arm_before_feedback=False)
        self.assertTrue(np.any(pre))
        self.assertFalse(np.any(post))


if __name__ == "__main__":
    unittest.main()
