"""Construction-level E3 tests; these do not certify G1 or EEG performance."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_lee2019_v2_random_precue_20261008 import (  # noqa: E402
    FADE, SFREQ, append_causal_fade, bounded_rest_partition,
)


class V2ConstructionTests(unittest.TestCase):
    def test_rest_partition_preserves_every_sample(self):
        rng = np.random.default_rng(23)
        for length in range(5 * SFREQ, 60 * SFREQ, 29):
            parts = bounded_rest_partition(length, rng)
            self.assertEqual(int(parts.sum()), length)
            if len(parts) > 1:
                self.assertTrue(np.all(parts >= 4 * SFREQ))
                self.assertTrue(np.all(parts <= 7 * SFREQ))

    def test_causal_fade_does_not_read_later_incoming_samples(self):
        old = np.arange(200, dtype=np.float32)[:, None]
        first = np.arange(100, dtype=np.float32)[:, None]
        second = first.copy()
        second[FADE:] = 9999
        p1, p2 = [old.copy()], [old.copy()]
        append_causal_fade(p1, first, len(old), [])
        append_causal_fade(p2, second, len(old), [])
        np.testing.assert_array_equal(p1[-1][:FADE], p2[-1][:FADE])
        np.testing.assert_array_equal(p1[-1][FADE:], first[FADE:])

    def test_fade_has_no_boundary_jump_for_constant_segments(self):
        p = [np.ones((100, 2), dtype=np.float32)]
        fade = []
        append_causal_fade(p, np.zeros((100, 2), dtype=np.float32), 100, fade)
        self.assertEqual(fade, [(100, 100 + FADE)])
        self.assertTrue(np.all(p[1][0] > 0))
        self.assertTrue(np.all(p[1][FADE - 1] == 0))


if __name__ == "__main__":
    unittest.main()
