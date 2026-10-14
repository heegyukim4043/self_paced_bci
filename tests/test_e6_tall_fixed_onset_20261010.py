"""Synthetic boundary checks for E6 event accounting and shift null."""

import numpy as np

from scripts.run_e6_tall_fixed_onset_20261010 import (
    collar_indices, hit_vector, null_by_all_shifts,
)


def test_collar_includes_exact_two_second_endpoint():
    times = np.arange(0, 8.1, .1)
    starts, ends = collar_indices(times, np.array([2.0]))
    alarms = np.zeros(len(times), dtype=bool)
    alarms[np.searchsorted(times, 4.0)] = True
    assert hit_vector(alarms, starts, ends).tolist() == [True]
    alarms[:] = False
    alarms[np.searchsorted(times, 4.1)] = True
    assert hit_vector(alarms, starts, ends).tolist() == [False]


def test_no_information_train_has_zero_null_and_excess():
    segment = np.zeros(1600, dtype=bool)
    times = np.arange(len(segment)) * .1
    mi_starts, mi_ends = collar_indices(times, np.array([20., 40., 60.]))
    fix_starts, fix_ends = collar_indices(times, np.array([16., 36., 56.]))
    out = null_by_all_shifts(segment, mi_starts, mi_ends,
                             fix_starts, fix_ends, period=20.)
    assert out["mi_primary_mean"] == 0.
    assert out["fixation_primary_mean"] == 0.
    assert out["mi_period_excluded_mean"] == 0.
    assert out["n_period_sensitivity_offsets"] < out["n_primary_offsets"]


def test_shift_null_keeps_both_event_collars_and_excludes_periodic_offsets():
    segment = np.zeros(1800, dtype=bool)
    segment[[250, 450, 650]] = True
    times = np.arange(len(segment)) * .1
    mi_starts, mi_ends = collar_indices(times, np.array([25., 45., 65.]))
    fix_starts, fix_ends = collar_indices(times, np.array([21., 41., 61.]))
    out = null_by_all_shifts(segment, mi_starts, mi_ends,
                             fix_starts, fix_ends, period=20.)
    assert out["n_primary_offsets"] > 1000
    assert out["period_excluded_offsets"] > 0
    assert out["n_period_sensitivity_offsets"] > 0
    assert 0 <= out["mi_primary_mean"] <= 1
    assert 0 <= out["fixation_primary_mean"] <= 1
