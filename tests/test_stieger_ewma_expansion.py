import unittest

import numpy as np

from scripts.run_stieger_ewma_expansion import causal_ewma_z


class StiegerEwmaTests(unittest.TestCase):
    def test_constant_anchor_stays_zero(self):
        values = np.full(100, 3.0)
        result = causal_ewma_z(values, 3.0, 4.0, 30.0)
        np.testing.assert_allclose(result, 0.0)

    def test_future_change_does_not_modify_past(self):
        first = np.linspace(-1.0, 1.0, 50)
        a = causal_ewma_z(np.r_[first, np.zeros(20)], 0.0, 1.0, 30.0)
        b = causal_ewma_z(np.r_[first, np.full(20, 100.0)], 0.0, 1.0, 30.0)
        np.testing.assert_allclose(a[: len(first)], b[: len(first)])


if __name__ == "__main__":
    unittest.main()
