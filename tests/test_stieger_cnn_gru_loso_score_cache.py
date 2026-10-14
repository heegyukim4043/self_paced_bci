import numpy as np

from scripts.prepare_stieger_cnn_gru_loso_score_cache import parse_shard


def test_parse_shard():
    from pathlib import Path

    assert parse_shard(Path("x/session2/sub-07")) == (7, 2)


def test_loso_scores_average_on_seed_axis():
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(values.mean(axis=0), [2.0, 3.0])
