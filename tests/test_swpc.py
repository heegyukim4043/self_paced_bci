import unittest

import torch

from uncued_mi.swpc import (
    EEGNetGate,
    balanced_gate_loss,
    make_teacher,
    swpc_consistency_loss,
    update_ema,
)


class SWPCTests(unittest.TestCase):
    def test_eegnet_shape(self):
        model = EEGNetGate(20, 250)
        self.assertEqual(tuple(model(torch.randn(7, 20, 250)).shape), (7,))

    def test_subtype_balanced_loss(self):
        zero = torch.zeros(4)
        loss = balanced_gate_loss(zero, zero[:1], zero)
        self.assertAlmostEqual(float(loss), 0.693147, places=5)

    def test_consistency_is_small_for_identical_features(self):
        values = torch.randn(8, 16)
        self.assertLess(float(swpc_consistency_loss(values, values)), 0.1)

    def test_teacher_is_frozen_and_updates(self):
        model = EEGNetGate(4, 64)
        teacher = make_teacher(model)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        with torch.no_grad():
            next(model.parameters()).add_(1)
        before = next(teacher.parameters()).clone()
        update_ema(model, teacher, 0.5)
        self.assertFalse(torch.equal(before, next(teacher.parameters())))


if __name__ == "__main__":
    unittest.main()
