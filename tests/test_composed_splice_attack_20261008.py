"""Unit checks for the metadata-only E1 simulation, not an EEG G1 test."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simulate_composed_splice_attack_20261008 import (  # noqa: E402
    chance_reference,
    composed_attack,
    detect_splice_bins,
    make_timeline,
    splice_sensitivity,
)


class ComposedSpliceAttackTests(unittest.TestCase):
    def test_chance_auc_recovers_candidate_fpr(self):
        for false_rate in (0.5, 2.0, 16.0):
            self.assertAlmostEqual(splice_sensitivity(0.5, false_rate), false_rate / 60.0)

    def test_sensitivity_increases_with_auc_at_fixed_false_rate(self):
        self.assertLess(splice_sensitivity(0.55, 2.0), splice_sensitivity(0.70, 2.0))

    def test_auc_half_is_independent_of_where_splices_are(self):
        marks = np.linspace(0.0, 0.99, 20)
        sensitivity = splice_sensitivity(0.5, 4.0)
        early = detect_splice_bins(np.array([1.5]), 20.0, marks, sensitivity, 4.0)
        late = detect_splice_bins(np.array([17.5]), 20.0, marks, sensitivity, 4.0)
        self.assertTrue(np.array_equal(early, late))

    def test_timeline_keeps_all_splices_indistinguishable_to_detector(self):
        seams, onsets, idle, horizon = make_timeline(
            np.array([10.0, 12.0]), 0.0, 3.5, np.random.default_rng(1))
        self.assertTrue(np.array_equal(seams, np.array([0.0, 10.0, 17.0, 29.0])))
        self.assertTrue(np.array_equal(onsets, np.array([13.5, 32.5])))
        self.assertEqual(horizon, 36.0)
        self.assertEqual(float(np.sum(idle[:, 1] - idle[:, 0])), 22.0)

    def test_attack_without_detected_seams_has_no_hits(self):
        idle = np.array([[0.0, 10.0]])
        self.assertEqual(composed_attack(np.array([]), np.array([13.5]), idle, 17.0),
                         (0.0, 0.0, 0, 0))

    def test_chance_reference_not_subject_pseudoreplication(self):
        p, upper = chance_reference(3, np.random.default_rng(1), n_subjects=54)
        self.assertAlmostEqual(p, 1 - np.exp(-2 * 2 / 60))
        self.assertGreater(upper, p)


if __name__ == "__main__":
    unittest.main()
