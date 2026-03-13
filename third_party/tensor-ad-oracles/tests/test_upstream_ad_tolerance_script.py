import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
    def _encoded_scalar(self, *, dtype: str, value: float) -> list[float] | list[list[float]]:
        if dtype.startswith("complex"):
            return [[value, 0.0]]
        return [value]

    def _success_record(self, *, dtype: str, pytorch_value: float, fd_value: float) -> dict:
        return {
            "case_id": f"abs_identity_{dtype}",
            "op": "abs",
            "family": "identity",
            "dtype": dtype,
            "expected_behavior": "success",
            "probes": [
                {
                    "pytorch_ref": {
                        "jvp": {
                            "value": {
                                "dtype": dtype,
                                "shape": [1],
                                "order": "row_major",
                                "data": self._encoded_scalar(dtype=dtype, value=pytorch_value),
                            }
                        },
                        "vjp": {
                            "a": {
                                "dtype": dtype,
                                "shape": [1],
                                "order": "row_major",
                                "data": self._encoded_scalar(dtype=dtype, value=1.0),
                            }
                        },
                    },
                    "fd_ref": {
                        "step": 1e-6,
                        "jvp": {
                            "value": {
                                "dtype": dtype,
                                "shape": [1],
                                "order": "row_major",
                                "data": self._encoded_scalar(dtype=dtype, value=fd_value),
                            }
                        },
                    },
                }
            ],
        }

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

    def test_audit_skips_single_precision_records(self) -> None:
        spec_index = {
            ("abs", "identity"): SimpleNamespace(
                upstream_name="abs",
                upstream_variant_name="",
                inventory_kind="scalar",
            )
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "abs").mkdir()
            path = root / "abs" / "identity.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(self._success_record(dtype="float64", pytorch_value=1.0, fd_value=1.0)),
                        json.dumps(self._success_record(dtype="float32", pytorch_value=1.0, fd_value=10.0)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(check_upstream_ad_tolerances, "build_case_spec_index", return_value=spec_index),
                patch.object(
                    check_upstream_ad_tolerances,
                    "resolve_upstream_scalar_ad_tolerance",
                    return_value={"rtol": 1e-3, "atol": 1e-5},
                ),
            ):
                audits = check_upstream_ad_tolerances.audit_against_upstream_ad_tolerances(root)

        self.assertEqual([audit.case_key for audit in audits], ["abs/identity/float64"])

    def test_audit_filters_upstream_violation_checks_by_dtype(self) -> None:
        spec_index = {
            ("abs", "identity"): SimpleNamespace(
                upstream_name="abs",
                upstream_variant_name="",
                inventory_kind="scalar",
            )
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "abs").mkdir()
            path = root / "abs" / "identity.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(self._success_record(dtype="float64", pytorch_value=1.0, fd_value=1.0)),
                        json.dumps(
                            self._success_record(dtype="complex128", pytorch_value=1.0, fd_value=10.0)
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(check_upstream_ad_tolerances, "build_case_spec_index", return_value=spec_index),
                patch.object(
                    check_upstream_ad_tolerances,
                    "resolve_upstream_scalar_ad_tolerance",
                    return_value={"rtol": 1e-3, "atol": 1e-5},
                ),
            ):
                audits = check_upstream_ad_tolerances.audit_against_upstream_ad_tolerances(root)

        audit_map = {audit.case_key: audit for audit in audits}
        self.assertIn("abs/identity/float64", audit_map)
        self.assertFalse(audit_map["abs/identity/float64"].violates_upstream)


if __name__ == "__main__":
    unittest.main()
