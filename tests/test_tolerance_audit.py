import unittest

from generators import tolerance_audit


class ToleranceAuditTests(unittest.TestCase):
    def test_proposed_tolerance_uses_safety_factor_and_rounds_up(self) -> None:
        proposed = tolerance_audit.propose_tolerance(
            observed_max=2.3e-14,
            safety_factor=1e3,
            floor=1e-15,
        )

        self.assertEqual(proposed, 1e-10)

    def test_proposed_tolerance_respects_floor_for_zero_residual(self) -> None:
        proposed = tolerance_audit.propose_tolerance(
            observed_max=0.0,
            safety_factor=1e3,
            floor=1e-15,
        )

        self.assertEqual(proposed, 1e-15)

    def test_needs_tightening_only_when_more_than_ten_orders_looser(self) -> None:
        self.assertTrue(
            tolerance_audit.needs_tightening(
                current=1e-3,
                observed_max=1e-14,
                looseness_orders=10,
            )
        )
        self.assertFalse(
            tolerance_audit.needs_tightening(
                current=1e-6,
                observed_max=1e-8,
                looseness_orders=10,
            )
        )


if __name__ == "__main__":
    unittest.main()
