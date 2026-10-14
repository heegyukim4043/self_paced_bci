import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from uncued_mi.decision import guard_local_moving_average, guard_rearmed_alarms
from uncued_mi.rest_sham import (
    ewma_z_many,
    guard_local_ma_many,
    guard_rearmed_alarms_many,
)
from scripts.evaluate_lee2019_locked_selfpaced_metrics import ewma_z
from scripts.evaluate_lee2019_rest_sham_a1 import evaluate_raw
from scripts.evaluate_lee2019_rest_sham_matched_fp import (
    compute_guard_local_ma, matched_curve,
)


def test_batch_sham_processing_matches_scalar():
    rng = np.random.default_rng(9147)
    raw = rng.normal(size=(7, 103))
    allowed = np.ones(103, dtype=bool)
    allowed[:5] = False
    allowed[30:36] = False
    allowed[70:75] = False
    z_batch = ewma_z_many(raw, .2, 1.3, allowed)
    ma_batch = guard_local_ma_many(z_batch, allowed, ma_steps=5)
    for threshold in (-1.0, 0.0, 1.0):
        alarms_batch = guard_rearmed_alarms_many(
            ma_batch, threshold, allowed, dwell_steps=3, refractory_steps=10
        )
        for row in range(len(raw)):
            z = ewma_z(raw[row], .2, 1.3, allowed)
            ma = guard_local_moving_average(z, allowed, ma_steps=5)
            alarms = guard_rearmed_alarms(
                ma, threshold, allowed, dwell_steps=3, refractory_steps=10
            )
            np.testing.assert_allclose(z_batch[row], z, rtol=0, atol=1e-12)
            np.testing.assert_allclose(ma_batch[row], ma, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(alarms_batch[row], alarms)


def test_accelerated_sham_metrics_match_scalar():
    rng = np.random.default_rng(1905)
    length = 301
    allowed = np.ones(length, dtype=bool)
    allowed[:8] = False
    allowed[112:127] = False
    allowed[235:248] = False
    times = np.arange(length, dtype=float) * .1
    onsets = np.asarray([3., 15., 26.])
    task = np.zeros(length, dtype=bool)
    for onset in onsets:
        task |= (times >= onset) & (times <= onset + 3.5)
    idle = allowed & ~task
    for raw in (rng.normal(size=length), np.ones(length),
                np.where(task, 1.5, -.7)):
        for threshold in (-.5, .1, .8):
            z = ewma_z(raw, .15, .9, allowed)
            ma = guard_local_moving_average(z, allowed, ma_steps=5)
            alarms = guard_rearmed_alarms(
                ma, threshold, allowed, dwell_steps=3, refractory_steps=10
            )
            expected_hits = sum(np.any(
                alarms & (times >= onset) & (times <= onset + 2.)
            ) for onset in onsets)
            expected_fp = int(np.sum(alarms & idle))
            actual_hits, actual_fp = evaluate_raw(
                raw, .15, .9, allowed, idle, times, onsets, threshold
            )
            assert (actual_hits, actual_fp) == (expected_hits, expected_fp)


def test_matched_curve_matches_scalar_decision():
    rng = np.random.default_rng(4815)
    length = 401
    raw = rng.normal(size=length)
    times = np.arange(length, dtype=float) * .1
    allowed = np.ones(length, dtype=bool)
    allowed[:8] = False
    allowed[170:185] = False
    future = times >= 18.5
    onsets = (21., 30.)
    hit_label = np.full(length, -1, dtype=np.int32)
    task = np.zeros(length, dtype=bool)
    for trial, onset in enumerate(onsets):
        hit_label[(times >= onset) & (times <= onset + 2.)] = trial
        task |= (times >= onset) & (times <= onset + 3.5)
    idle = future & allowed & ~task
    ma = compute_guard_local_ma(raw, .1, .8, allowed)
    expected_ma = guard_local_moving_average(
        ewma_z(raw, .1, .8, allowed), allowed, ma_steps=5
    )
    np.testing.assert_allclose(ma, expected_ma, atol=1e-12, rtol=0)
    curve = matched_curve(ma, allowed, future, idle, hit_label, 51)
    assert len(curve) > 1
    for count, (threshold, hits) in curve.items():
        alarms = guard_rearmed_alarms(
            ma, threshold, allowed, dwell_steps=3, refractory_steps=10
        )
        assert int(np.sum(alarms & idle)) == count
        assert sum(np.any(alarms & (times >= onset) & (times <= onset + 2.))
                   for onset in onsets) == hits
