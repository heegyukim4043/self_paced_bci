"""Synthetic boundary and circular-null checks for the E6 follow-up."""

import numpy as np

from scripts.run_e6_strict_e6r_20261010 import circular_counts, metric


def test_circular_counts_agree_with_explicit_roll():
    binary = np.zeros(23, dtype=np.int8)
    binary[[0, 4, 21]] = 1
    starts = np.array([0, 7, 20])
    ends = np.array([3, 11, 23])
    offsets = np.arange(23, dtype=np.int32)
    got = circular_counts(binary, starts, ends, offsets)
    expected = np.array([[np.roll(binary, k)[a:b].sum() for a, b in zip(starts, ends)]
                         for k in offsets])
    np.testing.assert_array_equal(got, expected)


def test_occupancy_counts_sustained_high_without_new_late_alarm():
    # One early alarm, but supra-threshold MA stays high in the late collar.
    alarms = np.zeros(40, dtype=np.int8)
    alarms[12] = 1
    ma = np.zeros(40)
    ma[10:25] = 2.0
    threshold = 1.0
    starts = np.array([18])
    ends = np.array([23])
    offsets = np.arange(1, 40, dtype=np.int32)
    keep = np.ones(len(offsets), dtype=bool)
    a = metric(alarms, starts, ends, offsets, keep)
    o = metric((ma >= threshold).astype(np.int8), starts, ends, offsets, keep, True)
    assert a["observed_any"] == 0.0
    assert o["observed_any"] == 1.0
    assert o["observed_fraction"] == 1.0


def test_strict_geometry_tier0_and_riemannian():
    onset = 5.5
    for window, first_alarm in [(2., 7.5), (1., 6.5)]:
        assert np.isclose(first_alarm - window, onset)
        assert np.isclose(first_alarm + (3. - window), 8.5)
