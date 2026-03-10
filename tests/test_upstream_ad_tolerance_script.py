import io
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from unittest.mock import patch

from scripts import check_upstream_ad_tolerances


@dataclass(frozen=True)
class FakeAudit:
    case_key: str
    order: str
    observed_rtol: float
    observed_atol: float
    upstream_rtol: float
    upstream_atol: float


class CheckUpstreamAdTolerancesScriptTests(unittest.TestCase):
    def test_main_prints_summary_when_all_observed_residuals_fit_upstream_bounds(self) -> None:
        audits = [
            FakeAudit(
                case_key="solve/identity/float64",
                order="first_order",
                observed_rtol=1e-9,
                observed_atol=1e-10,
                upstream_rtol=1e-3,
                upstream_atol=1e-5,
            )
        ]

        stdout = io.StringIO()
        with patch.object(
            check_upstream_ad_tolerances,
            "audit_against_upstream_ad_tolerances",
            return_value=audits,
        ):
            with redirect_stdout(stdout):
                exit_code = check_upstream_ad_tolerances.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("upstream_ad_tolerance_audits=1", stdout.getvalue())

    def test_main_fails_when_observed_residual_exceeds_upstream_bound(self) -> None:
        audits = [
            FakeAudit(
                case_key="svd/s/float64",
                order="second_order",
                observed_rtol=2e-3,
                observed_atol=2e-5,
                upstream_rtol=1e-3,
                upstream_atol=1e-5,
            )
        ]

        with patch.object(
            check_upstream_ad_tolerances,
            "audit_against_upstream_ad_tolerances",
            return_value=audits,
        ):
            with self.assertRaisesRegex(SystemExit, "svd/s/float64"):
                check_upstream_ad_tolerances.main()


if __name__ == "__main__":
    unittest.main()
