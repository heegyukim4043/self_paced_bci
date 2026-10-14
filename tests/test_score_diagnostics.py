import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_tier0_scores.py"
SPEC = importlib.util.spec_from_file_location("diagnose_tier0_scores", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ScoreDiagnosticTests(unittest.TestCase):
    def test_auc_direction(self):
        self.assertEqual(MODULE.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])), 1.0)
        self.assertEqual(MODULE.auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
