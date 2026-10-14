"""Structural LaBraM checks without the external checkpoint or Lee raw EEG."""

from __future__ import annotations

import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from uncued_mi.labram_transfer import (
    FEATURE_DIM, PRETRAIN_CHANNELS, SFREQ, WINDOW, _student_weights,
    _make_official_backbone, causal_lee_200hz, encode_ends, input_channel_positions,
    training_ends, trailing_windows,
)


class DummyLaBraM(torch.nn.Module):
    def forward(self, windows, *, input_chans, return_patch_tokens):
        assert return_patch_tokens
        assert len(input_chans) == 20
        return windows.reshape(windows.shape[0], 38, 200)


class LaBraMTransferTest(unittest.TestCase):
    def test_factory_uses_official_base_layer_scale(self):
        class FakeOfficialModule:
            @staticmethod
            def labram_base_patch200_200(**kwargs):
                # The upstream Block raises if init_values is None.
                if not kwargs["init_values"] > 0:
                    raise TypeError("LaBraM base requires positive layer scale")
                return kwargs

        settings = _make_official_backbone(FakeOfficialModule())
        self.assertEqual(settings["init_values"], 0.1)
        self.assertFalse(settings["use_mean_pooling"])

    def test_entrypoints_import_without_data(self):
        for script in (
            "cache_labram_features_original_continuous_20261014.py",
            "fit_labram_loso_original_continuous_20261014.py",
            "run_labram_original_continuous_20261014.py",
        ):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), "--help"],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_official_channel_positions(self):
        self.assertEqual(input_channel_positions(), [
            0, 1, 3, 18, 22, 40, 44, 62, 66, 81, 83,
            16, 24, 38, 46, 60, 68, 20, 42, 64,
        ])

    def test_causal_preprocessing_and_two_second_support(self):
        rng = np.random.default_rng(44)
        samples = rng.normal(0, 15, size=(3000, 19))
        channels = tuple({"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(
            name, name) for name in PRETRAIN_CHANNELS)
        prefix = causal_lee_200hz(samples[:2000], channels, 1000., input_unit="uV")
        full = causal_lee_200hz(samples, channels, 1000., input_unit="uV")
        np.testing.assert_array_equal(prefix, full[:, :prefix.shape[1]])
        ends = np.asarray([400, 500], dtype=np.int64)
        windows = trailing_windows(full, ends)
        self.assertEqual(windows.shape, (2, 19, 2, 200))
        before = encode_ends(DummyLaBraM(), full, ends, torch.device("cpu"))
        changed = full.copy()
        changed[:, 501:] += 1e5
        after = encode_ends(DummyLaBraM(), changed, ends, torch.device("cpu"))
        np.testing.assert_array_equal(before, after)
        self.assertEqual(before.shape, (2, FEATURE_DIM))
        with self.assertRaisesRegex(ValueError, "outside"):
            trailing_windows(full, np.asarray([WINDOW - 1]))

    def test_source_only_training_support(self):
        cues = 4.0 + np.arange(100) * 11.05
        ends, labels, trials = training_ends(cues, int(1200 * SFREQ))
        self.assertEqual(len(ends), 256)
        self.assertEqual(int(np.sum(labels)), 128)
        self.assertTrue(np.all(trials < 50))

    def test_checkpoint_rejects_partial_encoder(self):
        model = torch.nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _student_weights({"model": {"student.weight": torch.ones(1, 2)}}, model)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _student_weights({"model": {"student.weight": torch.ones(3, 2)}}, model)


if __name__ == "__main__":
    unittest.main()
