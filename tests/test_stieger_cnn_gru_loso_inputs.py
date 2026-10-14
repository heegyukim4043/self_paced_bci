import numpy as np

from scripts.prepare_stieger_cnn_gru_loso_inputs import PROTOCOL_VERSION


def test_protocol_version_is_positive():
    assert PROTOCOL_VERSION >= 1


def test_flat_signal_round_trip_shape():
    signal = np.arange(3 * 20 * 7, dtype=np.float32).reshape(3, 20, 7)
    flat = np.moveaxis(signal, -1, 0)
    assert flat.shape == (7, 3, 20)
    np.testing.assert_array_equal(np.moveaxis(flat, 0, -1), signal)
