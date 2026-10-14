import unittest

import numpy as np

from uncued_mi.rest_injected import (
    build_rest_injected_stream,
    random_partition,
)


class RestInjectedTests(unittest.TestCase):
    def test_partition_is_non_overlapping_and_complete(self):
        chunks = random_partition(285, 50, 110, np.random.default_rng(1))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 285)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(chunks, chunks[1:])))

    def test_stream_uses_each_rest_window_at_most_once(self):
        rests = [np.arange(120), np.arange(200, 320)]
        commands = [np.ones(20) * value for value in range(10)]
        sequence, audit = build_rest_injected_stream(
            rests,
            commands,
            step_seconds=0.1,
            minimum_rest_seconds=5,
            maximum_rest_seconds=7,
            seed=2,
        )
        idle_seconds = np.sum(
            sequence.idle_intervals[:, 1] - sequence.idle_intervals[:, 0]
        )
        self.assertAlmostEqual(idle_seconds, audit.rest_windows * 0.1)
        self.assertLessEqual(audit.rest_windows, 240)
        self.assertEqual(len(sequence.command_intervals), audit.command_events)
        self.assertTrue(
            np.all(sequence.command_intervals[:, 0] >= sequence.idle_intervals[:, 1])
        )


if __name__ == "__main__":
    unittest.main()
