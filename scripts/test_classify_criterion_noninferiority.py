#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("classify_criterion_noninferiority.py")
SPEC = importlib.util.spec_from_file_location("criterion_classifier", SCRIPT)
classifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(classifier)


class ClassificationBoundaryTests(unittest.TestCase):
    def test_pass_requires_every_upper_endpoint_at_or_below_threshold(self):
        intervals = [(-0.01, 0.05), (0.00, 0.049), (0.01, 0.05)]
        self.assertEqual(classifier.classify(intervals), "PASS")

    def test_one_upper_endpoint_above_threshold_is_not_pass(self):
        intervals = [(-0.01, 0.05), (0.00, 0.050001), (0.01, 0.05)]
        self.assertEqual(classifier.classify(intervals), "INCONCLUSIVE")

    def test_fail_requires_at_least_two_lower_endpoints_strictly_above_threshold(self):
        intervals = [(0.050001, 0.06), (0.050001, 0.07), (-0.01, 0.01)]
        self.assertEqual(classifier.classify(intervals), "FAIL")

    def test_lower_endpoint_equal_to_threshold_does_not_count_as_fail(self):
        intervals = [(0.05, 0.06), (0.050001, 0.07), (-0.01, 0.01)]
        self.assertEqual(classifier.classify(intervals), "INCONCLUSIVE")

    def test_b_over_a_interval_is_inverted_to_candidate_over_baseline(self):
        lower, upper, point = classifier.invert_interval(-0.10, -0.06, -0.08)
        self.assertAlmostEqual(lower, 1.0 / 0.94 - 1.0)
        self.assertAlmostEqual(upper, 1.0 / 0.90 - 1.0)
        self.assertAlmostEqual(point, 1.0 / 0.92 - 1.0)

    def test_sentinel_is_valid_when_interval_touches_band_boundaries(self):
        self.assertFalse(classifier.sentinel_breached(0.05, 0.08))
        self.assertFalse(classifier.sentinel_breached(-0.08, -0.05))

    def test_sentinel_is_invalid_for_wholly_outside_interval_in_either_direction(self):
        self.assertTrue(classifier.sentinel_breached(0.050001, 0.08))
        self.assertTrue(classifier.sentinel_breached(-0.08, -0.050001))


if __name__ == "__main__":
    unittest.main()
