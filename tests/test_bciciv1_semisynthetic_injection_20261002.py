"""Deterministic structural tests for the BCI IV-1 semi-synthetic audit."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from audit_bciciv1_semisynthetic_injection_20261002 import (
    eligible_idle_events, exact_support_preserving_null, hit_map,
    inject_sensorimotor_erd, placebo_audit,
    support_preserving_placebo_audit,
)


class SemiSyntheticTests(unittest.TestCase):
    def test_eligible_runs_are_nonoverlapping_and_have_margin(self):
        labels = np.r_[np.ones(100), np.zeros(600), np.ones(100),
                       np.zeros(599), np.ones(100), np.zeros(700)]
        events = eligible_idle_events(labels, sfreq=100)
        self.assertEqual(events.shape, (2, 2))
        self.assertTrue(np.allclose(events, [[2.5, 5.5], [16.49, 19.49]]))

    def test_injection_is_local_and_attenuates_band(self):
        sfreq = 100
        time = np.arange(1200) / sfreq
        eeg = np.column_stack([np.sin(2 * np.pi * 10 * time),
                               np.sin(2 * np.pi * 10 * time),
                               np.sin(2 * np.pi * 10 * time),
                               np.sin(2 * np.pi * 10 * time)])
        changed = inject_sensorimotor_erd(eeg, np.array([[4.0, 7.0]]),
                                           ("C3", "Cz", "C4", "Pz"), sfreq=sfreq)
        self.assertTrue(np.array_equal(changed[:400], eeg[:400]))
        self.assertTrue(np.array_equal(changed[700:], eeg[700:]))
        self.assertTrue(np.array_equal(changed[:, 3], eeg[:, 3]))
        self.assertLess(np.std(changed[460:640, 0]), 0.7 * np.std(eeg[460:640, 0]))
        sham = inject_sensorimotor_erd(eeg, np.array([[4.0, 7.0]]),
                                        ("C3", "Cz", "C4", "Pz"),
                                        amplitude=0.0, sfreq=sfreq)
        self.assertTrue(np.array_equal(sham, eeg))

    def test_hit_map_matches_expected_collars(self):
        times = np.arange(0, 10, 0.1)
        alarms = np.zeros(len(times), bool)
        alarms[20] = True
        got = hit_map(times, alarms, np.array([1.0, 2.5, 7.0]),
                      np.array([10]), tolerance=1.0)
        self.assertEqual(got[:, 0].tolist(), [True, False, False])
        self.assertEqual(got[:, 1].tolist(), [False, True, False])

    def test_placebo_audit_reports_conditional_fraction(self):
        hits = np.zeros((30, 5), bool)
        out = placebo_audit(hits, placements=10, events_per_draw=5, seed=1)
        self.assertEqual(out["conditional_rejection_fraction_p_le_0_05"], 0.0)
        self.assertEqual(out["null_predictive_95pct_band_coverage"], 1.0)

    def test_support_preserving_null_stays_within_candidate_pool(self):
        hits = np.r_[np.zeros(20, bool), np.ones(20, bool)]
        out = support_preserving_placebo_audit(
            hits, placements=500, events_per_draw=20,
            reference_draws=2000, seed=3,
        )
        self.assertGreaterEqual(out["null_predictive_95pct_band_coverage"], 0.94)
        self.assertLessEqual(out["conditional_rejection_fraction_p_le_0_05"], 0.08)
        self.assertEqual(out["reference_hit_rate_median"], 0.5)

    def test_exact_support_null_captures_discrete_tail(self):
        hits = np.r_[np.ones(2, bool), np.zeros(8, bool)]
        exact = exact_support_preserving_null(hits, events_per_draw=2)
        self.assertEqual(exact["hit_count_support"], [0, 1, 2])
        self.assertAlmostEqual(sum(exact["hit_count_probability"]), 1.0)
        self.assertAlmostEqual(exact["exact_rejection_probability_p_le_0_05"],
                               1 / 45)
        self.assertEqual(exact["central_95pct_predictive_hit_count_interval"],
                         [0, 1])
        self.assertAlmostEqual(exact["exact_central_95pct_predictive_coverage"],
                               44 / 45)

    def test_exact_support_null_handles_no_alarm_hits(self):
        exact = exact_support_preserving_null(np.zeros(30, bool),
                                              events_per_draw=20)
        self.assertEqual(exact["hit_count_support"], [0])
        self.assertEqual(exact["exact_rejection_probability_p_le_0_05"], 0.0)
        self.assertEqual(exact["exact_central_95pct_predictive_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
