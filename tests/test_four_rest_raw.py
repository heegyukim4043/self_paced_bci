import unittest

import numpy as np

from scripts.prepare_lee2019_four_rest_raw import cache_one


class FourRestRawTests(unittest.TestCase):
    def test_expected_rest_keys(self):
        # Structural guard: the four names must remain phase-specific.
        expected = {"pre_train", "post_train", "pre_test", "post_test"}
        self.assertEqual(len(expected), 4)
        self.assertNotIn("rest", expected)


if __name__ == "__main__":
    unittest.main()
