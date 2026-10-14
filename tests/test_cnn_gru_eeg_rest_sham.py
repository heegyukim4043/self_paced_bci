"""Structural tests for the EEG-level rest replacement used with CNN-GRU."""

import unittest

import numpy as np

from scripts.audit_lee2019_cnn_gru_eeg_sham_pilot import (
    draw_cue_preserving_eeg_sham,
    draw_rest_eeg_sham,
    draw_rest_rest_splice_eeg_sham,
    original_rest_library,
)


class TestEEGRestSham(unittest.TestCase):
    def setUp(self):
        self.events = np.zeros((3, 8), dtype=np.int64)
        self.signal = np.full((3, 20, 3 * 3750), -5., dtype=np.float32)
        for row in range(3):
            start = row * 3750
            self.events[row, 2:5] = start, start + 2000, start + 2000
            self.signal[..., start:start + 2000] = row + np.arange(2000) / 2000.
        self.ids = np.arange(3)
        self.spans = np.tile([0, 2000], (3, 1))

    def test_trial_only_replacement_and_exact_rest_lineage(self):
        blocks, starts, difference = original_rest_library(
            self.signal, self.events, self.ids, self.spans, 1750,
        )
        self.assertEqual(difference, 0.)
        self.assertEqual([len(x) for x in starts], [251, 251, 251])
        sham, source = draw_rest_eeg_sham(
            self.signal, self.events, blocks, starts,
            np.random.default_rng(7),
        )
        for row in range(3):
            rest_start, trial_start = map(int, self.events[row, [2, 4]])
            trial_stop = (int(self.events[row + 1, 2])
                          if row + 1 < 3 else self.signal.shape[-1])
            np.testing.assert_array_equal(
                sham[..., rest_start:trial_start],
                self.signal[..., rest_start:trial_start],
            )
            block, offset = map(int, source[row])
            np.testing.assert_array_equal(
                sham[..., trial_start:trial_stop],
                blocks[block][..., offset:offset + 1750],
            )

    def test_repeated_rest_source_must_be_identical(self):
        events = np.vstack([self.events, self.events[2]])
        events[3, 2:5] = 3 * 3750, 3 * 3750 + 2000, 3 * 3750 + 2000
        signal = np.concatenate([self.signal, self.signal[..., 2 * 3750:]], axis=-1)
        ids = np.array([0, 1, 2, 0])
        spans = np.tile([0, 2000], (4, 1))
        with self.assertRaises(AssertionError):
            original_rest_library(signal, events, ids, spans, 1750)

    def test_cue_preserving_sham_keeps_prefix_and_replaces_suffix(self):
        events = self.events.copy()
        events[:, 5] = events[:, 4] + 750
        blocks, starts, _ = original_rest_library(
            self.signal, events, self.ids, self.spans, 1750,
        )
        sham, source = draw_cue_preserving_eeg_sham(
            self.signal, events, blocks, starts, np.random.default_rng(8),
            preserve_after_cue_seconds=.5, crossfade_seconds=.2,
        )
        for row in range(3):
            target_start = int(events[row, 5] + 125)
            target_stop = (int(events[row + 1, 2]) if row + 1 < 3
                           else self.signal.shape[-1])
            np.testing.assert_array_equal(
                sham[..., :target_start] if row == 0 else
                sham[..., int(events[row, 4]):target_start],
                self.signal[..., :target_start] if row == 0 else
                self.signal[..., int(events[row, 4]):target_start],
            )
            block, offset = map(int, source[row])
            relative_start = target_start - int(events[row, 4])
            np.testing.assert_array_equal(
                sham[..., target_start + 50:target_stop],
                blocks[block][..., offset + relative_start + 50:offset + 1750],
            )

    def test_paired_shams_use_same_rest_samples_after_crossfade(self):
        events = self.events.copy()
        events[:, 5] = events[:, 4] + 750
        blocks, starts, _ = original_rest_library(
            self.signal, events, self.ids, self.spans, 1750,
        )
        rest, rest_source = draw_rest_eeg_sham(
            self.signal, events, blocks, starts, np.random.default_rng(19),
        )
        cue, cue_source = draw_cue_preserving_eeg_sham(
            self.signal, events, blocks, starts, np.random.default_rng(19),
        )
        np.testing.assert_array_equal(rest_source, cue_source)
        for row in range(len(events)):
            replacement_start = int(events[row, 5] + 125)
            trial_stop = (int(events[row + 1, 2]) if row + 1 < len(events)
                          else self.signal.shape[-1])
            np.testing.assert_array_equal(
                cue[..., replacement_start + 50:trial_stop],
                rest[..., replacement_start + 50:trial_stop],
            )

    def test_rest_rest_splice_uses_paired_post_rest_and_no_trial_eeg(self):
        events = self.events.copy()
        events[:, 5] = events[:, 4] + 750
        blocks, starts, _ = original_rest_library(
            self.signal, events, self.ids, self.spans, 1750,
        )
        rest, rest_source = draw_rest_eeg_sham(
            self.signal, events, blocks, starts, np.random.default_rng(19),
        )
        splice, post_source, pre_source = draw_rest_rest_splice_eeg_sham(
            self.signal, events, blocks, starts, np.random.default_rng(19),
            np.random.default_rng(20),
        )
        np.testing.assert_array_equal(rest_source, post_source)
        self.assertTrue(np.all(np.any(post_source != pre_source, axis=1)))
        for row in range(len(events)):
            trial_start = int(events[row, 4])
            splice_start = int(events[row, 5] + 125)
            trial_stop = (int(events[row + 1, 2]) if row + 1 < len(events)
                          else self.signal.shape[-1])
            pre_block, pre_offset = map(int, pre_source[row])
            np.testing.assert_array_equal(
                splice[..., trial_start:splice_start],
                blocks[pre_block][..., pre_offset:pre_offset + splice_start - trial_start],
            )
            np.testing.assert_array_equal(
                splice[..., splice_start + 50:trial_stop],
                rest[..., splice_start + 50:trial_stop],
            )


if __name__ == "__main__":
    unittest.main()
