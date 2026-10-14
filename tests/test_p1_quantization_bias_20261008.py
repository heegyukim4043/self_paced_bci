"""Synthetic checks for the post-hoc P1 quantization/bias audit."""

import unittest

import numpy as np

from scripts.audit_p1_quantization_bias_20261008 import (
    effective_error, summarize_matrix,
)


class P1QuantizationBiasTests(unittest.TestCase):
    def test_effective_error_uses_attained_not_nominal_rate(self):
        row = {
            "calibration_fp_per_min": 60. / 28.,
            "evaluation_rest_seconds": 84.,
            "future_fp_count": 3,
        }
        self.assertAlmostEqual(effective_error(row), np.log(3.5 / 3.))
        self.assertAlmostEqual(effective_error(row, 2.), np.log(3.5 / 2.8))

    def test_zero_calibration_rate_is_not_silently_logged(self):
        with self.assertRaises(ValueError):
            effective_error({"calibration_fp_per_min": 0.,
                             "evaluation_rest_seconds": 84.,
                             "future_fp_count": 0})

    def test_between_within_variance_reconstructs_total(self):
        matrix = np.array([[1., 2., 3.], [4., 5., 6.]])
        result = summarize_matrix(matrix, 200, 42)
        self.assertAlmostEqual(result["between_subject_sd"] ** 2
                               + result["within_subject_cell_sd"] ** 2,
                               result["cell_sd"] ** 2)
        self.assertAlmostEqual(result["median_subject_mean_signed_log_error"], 3.5)


if __name__ == "__main__":
    unittest.main()
