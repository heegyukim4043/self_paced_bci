"""CPU-only structural checks; the real checkpoint is tested on the server."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.fit_cbramod_loso_original_continuous_20261014 import fit_loso_probe
from scripts.finetune_cbramod_loso_original_continuous_20261014 import train_one
from uncued_mi.cbramod_transfer import (
    CBraModBinaryClassifier, FEATURE_DIM, PRETRAIN_CHANNELS, SFREQ, WINDOW,
    causal_lee_200hz, encode_ends,
    selected_channel_indices, training_ends, trailing_windows,
)


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        # Preserve CBraMod's [batch, channel, patch, feature] output geometry.
        return data * self.scale


class CBraModTransferTest(unittest.TestCase):
    def test_pretrained_montage_order_and_aliases(self):
        names = tuple({"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(
            name, name) for name in PRETRAIN_CHANNELS)
        order = names[::-1]
        indices = selected_channel_indices(order)
        self.assertEqual(tuple(order[i] for i in indices), names)
        with self.assertRaisesRegex(ValueError, "lacks"):
            selected_channel_indices(names[:-1])

    def test_causal_filter_prefix_is_unchanged_by_future(self):
        rng = np.random.default_rng(14)
        samples = rng.normal(0, 15, size=(2000, 19))
        names = tuple({"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(
            name, name) for name in PRETRAIN_CHANNELS)
        prefix = causal_lee_200hz(samples[:1000], names, 1000., input_unit="uV")
        full = causal_lee_200hz(samples, names, 1000., input_unit="uV")
        np.testing.assert_array_equal(prefix, full[:, :len(prefix[0])])
        with self.assertRaisesRegex(ValueError, "input_unit"):
            causal_lee_200hz(samples, names, 1000., input_unit="auto")

    def test_two_second_trailing_input_cannot_see_future(self):
        rng = np.random.default_rng(15)
        samples = rng.normal(size=(19, 1400)).astype(np.float32)
        ends = np.asarray([400, 600, 800], dtype=np.int64)
        windows = trailing_windows(samples, ends)
        self.assertEqual(windows.shape, (3, 19, 2, 200))
        before = encode_ends(DummyBackbone(), samples, ends,
                             torch.device("cpu"), batch_size=2)
        changed = samples.copy()
        changed[:, 801:] += 1e6
        after = encode_ends(DummyBackbone(), changed, ends,
                            torch.device("cpu"), batch_size=2)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(before.shape, (3, FEATURE_DIM))
        with self.assertRaisesRegex(ValueError, "outside"):
            trailing_windows(samples, np.asarray([WINDOW - 1]))

    def test_source_ends_use_only_first_50_trials(self):
        cues = 4.0 + np.arange(100) * 11.05
        ends, labels, trials = training_ends(cues, int(1200 * SFREQ))
        self.assertEqual(len(ends), 256)
        self.assertEqual(int(np.sum(labels)), 128)
        self.assertTrue(np.all(trials < 50))
        for end, label, trial in zip(ends, labels, trials):
            start_s = (end - WINDOW) / SFREQ
            end_s = end / SFREQ
            cue = cues[trial]
            if label:
                self.assertGreaterEqual(start_s + 1e-8, cue + .5)
                self.assertLessEqual(end_s, cue + 3.5 + 1e-8)
            else:
                fixation = start_s >= cue - 3.0 - 1e-8 and end_s <= cue + 1e-8
                late_iti = start_s >= cue + 5.0 - 1e-8 and end_s <= cues[trial+1] - 3.2 + 1e-8
                self.assertTrue(fixation or late_iti)

    def test_full_finetune_path_updates_backbone_and_head(self):
        model = CBraModBinaryClassifier(DummyBackbone())
        inputs = torch.randn(3, 19, 2, 200)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (3,))
        logits.square().mean().backward()
        self.assertIsNotNone(model.backbone.scale.grad)
        self.assertIsNotNone(model.head.weight.grad)

    def test_finetune_training_loop_smoke(self):
        rng = np.random.default_rng(21)
        data = rng.normal(size=(8, 19, 2, 200)).astype(np.float32)
        labels = np.asarray([0, 1] * 4, dtype=np.float32)
        model = CBraModBinaryClassifier(DummyBackbone())
        previous = float(model.backbone.scale.detach())
        losses = train_one(model, data, labels, seed=21, epochs=1,
                           batch_size=4, device=torch.device("cpu"))
        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses[0]))
        self.assertNotEqual(float(model.backbone.scale.detach()), previous)

    def test_target_features_and_labels_cannot_affect_loso_head(self):
        source = {}
        for subject in range(1, 55):
            features = np.zeros((2, FEATURE_DIM), dtype=np.float32)
            features[:, 0] = [-1.0 - subject / 100, 1.0 + subject / 100]
            source[subject] = (features, np.asarray([0, 1], np.int8))
        model_a, donors, _ = fit_loso_probe(source, 3)
        probe = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        score_a = model_a.decision_function(probe)
        self.assertEqual(len(donors), 53)
        self.assertNotIn(3, donors)
        source[3] = (np.full((2, FEATURE_DIM), 1e6, np.float32),
                     np.asarray([1, 1], np.int8))
        model_b, _, _ = fit_loso_probe(source, 3)
        np.testing.assert_array_equal(score_a, model_b.decision_function(probe))


if __name__ == "__main__":
    unittest.main()
