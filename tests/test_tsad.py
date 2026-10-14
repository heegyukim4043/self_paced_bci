import unittest

import numpy as np
import torch

from uncued_mi.tsad import (
    CausalTCNPredictor,
    RobustMahalanobis,
    make_contexts,
    zscore_from_reference,
)


class TSADTests(unittest.TestCase):
    def test_mahalanobis_scores_far_point_higher(self):
        rng = np.random.default_rng(1)
        model = RobustMahalanobis().fit(rng.normal(size=(200, 4)))
        scores = model.score_samples(np.asarray([[0, 0, 0, 0], [8, 8, 8, 8]]))
        self.assertGreater(scores[1], scores[0])

    def test_contexts_do_not_use_target_or_future(self):
        values = np.arange(20, dtype=np.float32).reshape(10, 2)
        x, y = make_contexts(values, 4)
        np.testing.assert_array_equal(x[0], values[:4])
        np.testing.assert_array_equal(y[0], values[4])

    def test_tcn_shape(self):
        model = CausalTCNPredictor(7)
        self.assertEqual(tuple(model(torch.randn(5, 4, 7)).shape), (5, 7))

    def test_robust_zscore(self):
        reference = np.asarray([-1.0, 0.0, 1.0])
        self.assertGreater(zscore_from_reference(np.asarray([3.0]), reference)[0], 0)


if __name__ == "__main__":
    unittest.main()
