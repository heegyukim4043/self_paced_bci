import unittest

from scripts.build_lee_stieger_cnn_gru_integrated_validation import (
    lee_loso_event_rows,
    usability_rows,
)


class IntegratedValidationTests(unittest.TestCase):
    def test_lee_event_extraction_keeps_sessions_separate(self):
        rows = []
        for session, value in ((1, 0.1), (2, 0.2)):
            rows.append(
                {
                    "session": session,
                    "target_fp_per_min": 2.0,
                    "cnn_gru": {
                        "actual_fp_per_min": 2.0,
                        "onset_sensitivity": {"1.0": value, "2.0": value + .1},
                        "circular_shift_null_median": {"1.0": 0.0, "2.0": 0.0},
                        "circular_shift_excess": {"1.0": value, "2.0": value + .1},
                        "coverage": value,
                    },
                }
            )
        result = lee_loso_event_rows({"rows": rows})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["sensitivity_1s"], 0.1)
        self.assertEqual(result[1]["sensitivity_1s"], 0.2)

    def test_usability_rows_preserve_training_regime(self):
        def block(value):
            models = {}
            for model in ("M0_detection_fraction", "M1_morphology", "M2_combined"):
                models[model] = {
                    "spearman_rho": value,
                    "mae": 0.1,
                    "usable_auroc": 0.5,
                }
            return {
                "target_median": 0.0,
                "models": models,
                "bootstrap": {"rho_ci95": {model: [-0.1, 0.1] for model in models}},
            }

        lee = {"summary": {"future_excess_fp2": block(0.4)}}
        stieger = {"prediction": {"future_common_fp_excess_fp2": block(0.0)}}
        rows = usability_rows(lee, stieger)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["training"], "source-pooled LOSO")
        self.assertEqual(rows[-1]["training"], "within-subject first-half")
        self.assertEqual(rows[0]["transport_gate"], "PASS")
        self.assertEqual(rows[-1]["transport_gate"], "FAIL")


if __name__ == "__main__":
    unittest.main()
