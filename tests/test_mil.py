import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.mil import (
    CompactCausalEncoder,
    ValidationCandidate,
    mil_loss,
    normalized_lse,
    select_validation_candidate,
)


class MilTests(unittest.TestCase):
    def test_lse_small_tau_approaches_max(self):
        values = torch.tensor([0.0, 1.0, 3.0])
        self.assertAlmostEqual(
            float(normalized_lse(values, tau=0.001)), 3.0, places=2
        )

    def test_lse_large_tau_approaches_mean(self):
        values = torch.tensor([0.0, 1.0, 3.0])
        self.assertAlmostEqual(
            float(normalized_lse(values, tau=1000.0)),
            float(values.mean()),
            places=2,
        )

    def test_encoder_shape(self):
        model = CompactCausalEncoder(channels=20)
        result = model(torch.randn(4, 20, 250))
        self.assertEqual(result.shape, (4,))

    def test_selection_prefers_simple_candidate_within_tolerance(self):
        complex_best = ValidationCandidate(0.3, 1.0, 0.1, 2.0, 0.205)
        simple_close = ValidationCandidate(1.0, 0.1, 0.0, 2.1, 0.200)
        selected, failed = select_validation_candidate(
            [complex_best, simple_close]
        )
        self.assertFalse(failed)
        self.assertEqual(selected, simple_close)

    def test_calibration_failure_selects_closest_fp(self):
        close = ValidationCandidate(1.0, 0.1, 0.0, 2.8, 0.2)
        far = ValidationCandidate(1.0, 0.1, 0.0, 4.0, 0.4)
        selected, failed = select_validation_candidate([far, close])
        self.assertTrue(failed)
        self.assertEqual(selected, close)

    def test_false_alarm_subtypes_are_averaged_equally(self):
        _, parts = mil_loss(
            [torch.tensor([2.0, 1.0])],
            torch.zeros(2),
            torch.zeros(20),
            tau=1.0,
            false_alarm_weight=1.0,
            continuity_weight=0.0,
        )
        self.assertAlmostEqual(
            float(parts["false_alarm"]),
            0.5 * (float(parts["rest"]) + float(parts["iti"])),
        )


if __name__ == "__main__":
    unittest.main()
