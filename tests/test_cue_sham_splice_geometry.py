"""Boundary semantics for the cue-preserving sham geometry audit."""

import unittest

from scripts.audit_lee2019_cue_sham_splice_geometry_20261002 import overlaps


class TestSpliceGeometry(unittest.TestCase):
    def test_half_open_support_overlap(self):
        self.assertTrue(overlaps(1.0, 2.0, 1.5, 1.7))
        self.assertFalse(overlaps(1.7, 2.7, 1.5, 1.7))
        self.assertFalse(overlaps(.5, 1.5, 1.5, 1.7))


if __name__ == "__main__":
    unittest.main()
