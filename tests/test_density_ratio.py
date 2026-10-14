import unittest

import numpy as np

from uncued_mi.density_ratio import DiagonalGMMRatio, ShrinkageGaussianRatio


class DensityRatioTests(unittest.TestCase):
    def test_scores_follow_positive_direction(self):
        rng = np.random.default_rng(3)
        negative = rng.normal(-1, 0.5, (200, 4))
        positive = rng.normal(1, 0.5, (200, 4))
        points = np.asarray([[-1] * 4, [1] * 4], dtype=float)
        for model in (ShrinkageGaussianRatio(), DiagonalGMMRatio(3)):
            model.fit(positive, negative)
            scores = model.decision_function(points)
            self.assertGreater(scores[1], scores[0])


if __name__ == "__main__":
    unittest.main()
