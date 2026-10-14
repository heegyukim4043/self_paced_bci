import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncued_mi.config import ExperimentConfig
from uncued_mi.data import LeeRun, build_intervals
from uncued_mi.features import _causal_downsample, extract_log_power


class FeatureTests(unittest.TestCase):
    def test_fractional_causal_downsample_uses_only_past_samples(self):
        signal = np.arange(12, dtype=float)[:, None]
        downsampled = _causal_downsample(signal, 5.0, 3.0)
        np.testing.assert_array_equal(
            downsampled[:, 0],
            np.asarray([0.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0]),
        )

    def test_block_break_is_not_task_iti(self):
        run = LeeRun(
            eeg=np.zeros((3000, 3)),
            cue_samples=np.asarray([500, 1500, 2600]),
            labels=np.asarray([1, 2, 1]),
            channels=("C3", "Cz", "C4"),
            sfreq=10.0,
            pre_rest=np.zeros((600, 3)),
            post_rest=np.zeros((600, 3)),
        )
        config = ExperimentConfig(max_task_iti_seconds=6.0)
        intervals = build_intervals(run, config)
        self.assertTrue(any(item.kind == "idle_break" for item in intervals))
        self.assertFalse(
            any(
                item.kind == "idle_iti" and item.stop - item.start > 6.0
                for item in intervals
            )
        )

    def test_first_window_starts_after_filter_transient(self):
        config = ExperimentConfig(
            native_sfreq=100.0,
            target_sfreq=100.0,
            window_seconds=2.0,
            feature_step_seconds=0.5,
            transient_seconds=1.0,
            bands=(("test", 8.0, 13.0),),
        )
        rng = np.random.default_rng(1)
        eeg = rng.normal(size=(1000, 3))
        stream = extract_log_power(
            eeg,
            100.0,
            ("C3", "Cz", "C4"),
            config,
        )
        self.assertAlmostEqual(stream.times[0], 3.0)
        self.assertEqual(stream.log_power.shape[1:], (3, 1))


if __name__ == "__main__":
    unittest.main()
