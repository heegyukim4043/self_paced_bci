import unittest

import torch

from uncued_mi.fbcnet import CompactFBCNet


class FBCNetTests(unittest.TestCase):
    def test_output_shape(self):
        model = CompactFBCNet(3, 20)
        self.assertEqual(tuple(model(torch.randn(8, 3, 20, 250)).shape), (8,))

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            CompactFBCNet(3, 20).encode(torch.randn(3, 20, 250))


if __name__ == "__main__":
    unittest.main()
