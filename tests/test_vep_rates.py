import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_vep_rates.py"
SPEC = importlib.util.spec_from_file_location("analyze_vep_rates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class VepRateTests(unittest.TestCase):
    def test_equal_pooled_rates(self):
        result = MODULE.rate_ratio(
            fixation_counts=np.array([1, 2]),
            fixation_exposure=np.array([10.0, 20.0]),
            rest_counts=np.array([2, 1]),
            rest_exposure=np.array([20.0, 10.0]),
            bootstrap=100,
            seed=1,
        )
        self.assertAlmostEqual(result["rate_ratio_fixation_vs_post_rest"], 1.0)


if __name__ == "__main__":
    unittest.main()
