"""No-data tests for the exploratory large causal CNN-GRU."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.large_causal_cnn_gru import LargeCausalCNNGRU
from scripts.run_lee2019_cnn_gru import CausalCNNGRU


class LargeCausalCNNGRUTest(unittest.TestCase):
    def test_capacity_shape_and_backward(self):
        torch.manual_seed(10)
        model = LargeCausalCNNGRU()
        baseline = CausalCNNGRU()
        self.assertGreater(sum(p.numel() for p in model.parameters()), sum(
            p.numel() for p in baseline.parameters()
        ))
        model.train()
        x = torch.randn(2, 4, 3, 20, 250)
        logits, hidden = model(x)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(hidden.shape), (2, 2, 128))
        logits.square().mean().backward()
        self.assertIsNotNone(model.output.weight.grad)

    def test_future_change_cannot_modify_past_scores(self):
        torch.manual_seed(11)
        model = LargeCausalCNNGRU().eval()
        x = torch.randn(1, 6, 3, 20, 250)
        changed = x.clone()
        changed[:, 4:] += 8
        with torch.no_grad():
            original, _ = model(x)
            perturbed, _ = model(changed)
            first, hidden = model(x[:, :3])
            second, _ = model(x[:, 3:], hidden)
        torch.testing.assert_close(original[:, :4], perturbed[:, :4], atol=1e-6, rtol=0)
        torch.testing.assert_close(original, torch.cat([first, second], dim=1), atol=1e-6, rtol=0)

    def test_invalid_window_shape_rejected(self):
        model = LargeCausalCNNGRU()
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2, 3, 20, 100))


if __name__ == "__main__":
    unittest.main()
