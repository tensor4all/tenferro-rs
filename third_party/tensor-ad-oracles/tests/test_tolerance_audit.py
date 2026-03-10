import unittest
import json
import tempfile
from pathlib import Path

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

    def test_audit_case_tree_tracks_second_order_tolerances_when_hvp_is_present(self) -> None:
        record = {
            "schema_version": 1,
            "case_id": "solve_f64_identity_001",
            "op": "solve",
            "dtype": "float64",
            "family": "identity",
            "expected_behavior": "success",
            "inputs": {
                "a": {"dtype": "float64", "shape": [1], "order": "row_major", "data": [1.0]},
            },
            "observable": {"kind": "identity"},
            "comparison": {
                "first_order": {"kind": "allclose", "rtol": 1e-8, "atol": 1e-9},
                "second_order": {"kind": "allclose", "rtol": 1e-6, "atol": 1e-7},
            },
            "probes": [
                {
                    "probe_id": "p0",
                    "direction": {
                        "a": {
                            "dtype": "float64",
                            "shape": [1],
                            "order": "row_major",
                            "data": [1.0],
                        }
                    },
                    "cotangent": {
                        "value": {
                            "dtype": "float64",
                            "shape": [1],
                            "order": "row_major",
                            "data": [1.0],
                        }
                    },
                    "pytorch_ref": {
                        "jvp": {
                            "value": {
                                "dtype": "float64",
                                "shape": [1],
                                "order": "row_major",
                                "data": [1.0],
                            }
                        },
                        "vjp": {
                            "a": {
                                "dtype": "float64",
                                "shape": [1],
                                "order": "row_major",
                                "data": [1.0],
                            }
                        },
                        "hvp": {
                            "a": {
                                "dtype": "float64",
                                "shape": [1],
                                "order": "row_major",
                                "data": [0.1],
                            }
                        },
                    },
                    "fd_ref": {
                        "method": "central_difference",
                        "stencil_order": 2,
                        "step": 1e-6,
                        "jvp": {
                            "value": {
                                "dtype": "float64",
                                "shape": [1],
                                "order": "row_major",
                                "data": [1.0],
                            }
                        },
                        "hvp": {
                            "a": {
                                "dtype": "float64",
                                "shape": [1],
                                "order": "row_major",
                                "data": [0.1001],
                            }
                        },
                    },
                }
            ],
            "provenance": {
                "source_repo": "pytorch",
                "source_file": "torch/testing/_internal/opinfo/definitions/linalg.py",
                "source_function": "sample_inputs_linalg_solve",
                "source_commit": "deadbeef",
                "generator": "python-pytorch-v1",
                "seed": 17,
                "torch_version": "2.10.0",
                "fd_policy_version": "v1",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "solve"
            case_dir.mkdir(parents=True)
            (case_dir / "identity.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            audits = tolerance_audit.audit_case_tree(Path(tmpdir))

        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0].current_second_order_rtol, 1e-6)
        self.assertEqual(audits[0].current_second_order_atol, 1e-7)
        self.assertGreater(audits[0].max_second_order_abs_residual, 0.0)
        self.assertFalse(
            tolerance_audit.needs_tightening(
                current=1e-6,
                observed_max=1e-8,
                looseness_orders=10,
            )
        )


if __name__ == "__main__":
    unittest.main()
