import numpy as np

from scripts.audit_lee2019_ar1_no_eeg_stage2 import (
    anchor_ar1_parameters, draw_stationary_ar1,
)
from scripts.audit_lee2019_rearm_blind_time_stage2 import unarmed_valid_samples


def test_unarmed_resets_only_at_guard_and_arms_after_below():
    ma = np.array([np.nan, 2., 2., .5, 2., np.nan, 3., 3., .4, 2.])
    allowed = np.array([False, True, True, True, True,
                        False, True, True, True, True])
    result = unarmed_valid_samples(ma, allowed, threshold=1.)
    assert np.array_equal(result, [False, True, True, True, False,
                                   False, True, True, True, False])


def test_ar1_control_is_seeded_and_stationary():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = draw_stationary_ar1(100_000, 4., 2., .8, rng1)
    b = draw_stationary_ar1(100_000, 4., 2., .8, rng2)
    assert np.array_equal(a, b)
    mean, sd, phi = anchor_ar1_parameters(a)
    assert abs(mean - 4.) < .08
    assert abs(sd - 2.) < .08
    assert abs(phi - .8) < .02


def test_ar1_parameter_estimation_handles_constant_anchor():
    mean, sd, phi = anchor_ar1_parameters(np.ones(10))
    assert mean == 1.
    assert sd == 1e-8
    assert phi == 0.
