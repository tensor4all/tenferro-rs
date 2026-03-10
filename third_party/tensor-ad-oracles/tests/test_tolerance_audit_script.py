import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from generators.tolerance_audit import FamilyAudit
from scripts import check_tolerances


class CheckTolerancesScriptTests(unittest.TestCase):
    def test_main_prints_summary_when_no_family_needs_tightening(self) -> None:
        audit = FamilyAudit(
            op="solve",
            family="identity",
            dtype="float64",
            current_rtol=1e-8,
            current_atol=1e-9,
            max_rel_residual=1e-8,
            max_abs_residual=1e-9,
            proposed_rtol=1e-8,
            proposed_atol=1e-9,
            tighten_rtol=False,
            tighten_atol=False,
            current_second_order_rtol=None,
            current_second_order_atol=None,
            max_second_order_rel_residual=None,
            max_second_order_abs_residual=None,
            proposed_second_order_rtol=None,
            proposed_second_order_atol=None,
            tighten_second_order_rtol=False,
            tighten_second_order_atol=False,
        )

        stdout = io.StringIO()
        with patch.object(check_tolerances, "audit_case_tree", return_value=[audit]):
            with redirect_stdout(stdout):
                exit_code = check_tolerances.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("audited_family_tolerances=1", stdout.getvalue())

    def test_main_fails_when_family_tolerance_is_obviously_too_loose(self) -> None:
        audit = FamilyAudit(
            op="svd",
            family="u_abs",
            dtype="float64",
            current_rtol=1e-6,
            current_atol=1e-7,
            max_rel_residual=1e-16,
            max_abs_residual=1e-17,
            proposed_rtol=1e-13,
            proposed_atol=1e-14,
            tighten_rtol=True,
            tighten_atol=True,
            current_second_order_rtol=None,
            current_second_order_atol=None,
            max_second_order_rel_residual=None,
            max_second_order_abs_residual=None,
            proposed_second_order_rtol=None,
            proposed_second_order_atol=None,
            tighten_second_order_rtol=False,
            tighten_second_order_atol=False,
        )

        with patch.object(check_tolerances, "audit_case_tree", return_value=[audit]):
            with self.assertRaisesRegex(SystemExit, "svd/u_abs/float64"):
                check_tolerances.main()

    def test_main_fails_when_second_order_tolerance_is_obviously_too_loose(self) -> None:
        audit = FamilyAudit(
            op="solve",
            family="identity",
            dtype="float64",
            current_rtol=1e-8,
            current_atol=1e-9,
            max_rel_residual=1e-8,
            max_abs_residual=1e-9,
            proposed_rtol=1e-8,
            proposed_atol=1e-9,
            tighten_rtol=False,
            tighten_atol=False,
            current_second_order_rtol=1e-3,
            current_second_order_atol=1e-4,
            max_second_order_rel_residual=1e-15,
            max_second_order_abs_residual=1e-16,
            proposed_second_order_rtol=1e-12,
            proposed_second_order_atol=1e-13,
            tighten_second_order_rtol=True,
            tighten_second_order_atol=True,
        )

        with patch.object(check_tolerances, "audit_case_tree", return_value=[audit]):
            with self.assertRaisesRegex(SystemExit, "second_order"):
                check_tolerances.main()


if __name__ == "__main__":
    unittest.main()
