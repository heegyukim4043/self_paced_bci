import unittest
import torch
from uncued_mi.tangent_tcn import TangentTCN


class TangentTCNTests(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(tuple(TangentTCN(210)(torch.randn(4, 3, 210)).shape), (4,))


if __name__ == "__main__":
    unittest.main()
