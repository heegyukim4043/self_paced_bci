import unittest

import torch

from uncued_mi.spdnet import CompactMultiBandSPDNet


class SPDNetTests(unittest.TestCase):
    def test_shape_and_projection(self):
        model = CompactMultiBandSPDNet(3, 20, 8)
        x = torch.randn(5, 3, 20, 20)
        covariance = x @ x.transpose(-1, -2) + torch.eye(20)
        self.assertEqual(tuple(model(covariance).shape), (5,))
        model.project_bimap()
        identity = model.bimap @ model.bimap.transpose(-1, -2)
        self.assertTrue(torch.allclose(identity, torch.eye(8)[None], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
