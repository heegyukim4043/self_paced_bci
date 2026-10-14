import unittest

import numpy as np

from scripts.prepare_lee2019_four_rest_cache import collect
from uncued_mi.data import Interval


class FourRestTests(unittest.TestCase):
    def test_collect_requires_complete_window_support(self):
        times = np.arange(0.5, 5.1, 0.5)
        cov = np.arange(len(times))[:, None, None]
        intervals = [Interval("fixation", 1.0, 4.0, trial=2)]
        values, trials, relative = collect(times, cov, intervals, "fixation", 1.0)
        np.testing.assert_array_equal(values[:, 0, 0], [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(trials, np.full(5, 2))
        np.testing.assert_allclose(relative, [-2, -1.5, -1, -0.5, 0])


if __name__ == "__main__":
    unittest.main()
