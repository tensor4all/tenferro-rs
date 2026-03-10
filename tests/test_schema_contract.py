import json
import unittest
from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "case.schema.json"


class SchemaContractTests(unittest.TestCase):
    def test_observable_enum_matches_v1_registry(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        observable_enum = schema["$defs"]["observable_kind"]["enum"]

        self.assertEqual(
            observable_enum,
            [
                "identity",
                "svd_u_abs",
                "svd_s",
                "svd_vh_abs",
                "svd_uvh_product",
                "eigh_values_vectors_abs",
                "eig_values_vectors_abs",
            ],
        )

    def test_schema_accepts_minimal_success_case(self) -> None:
        try:
            import jsonschema
        except ModuleNotFoundError as exc:
            self.skipTest(f"jsonschema unavailable: {exc}")

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        case = {
            "schema_version": 1,
            "case_id": "solve_f64_identity_001",
            "op": "solve",
            "dtype": "float64",
            "family": "identity",
            "expected_behavior": "success",
            "inputs": {
                "a": {"dtype": "float64", "shape": [1, 1], "order": "row_major", "data": [1.0]},
                "b": {"dtype": "float64", "shape": [1], "order": "row_major", "data": [2.0]},
            },
            "observable": {"kind": "identity"},
            "comparison": {"kind": "allclose", "rtol": 1e-10, "atol": 1e-10},
            "probes": [
                {
                    "probe_id": "p0",
                    "direction": {
                        "a": {"dtype": "float64", "shape": [1, 1], "order": "row_major", "data": [1.0]},
                        "b": {"dtype": "float64", "shape": [1], "order": "row_major", "data": [1.0]},
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
                                "shape": [1, 1],
                                "order": "row_major",
                                "data": [1.0],
                            },
                            "b": {"dtype": "float64", "shape": [1], "order": "row_major", "data": [1.0]},
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

        jsonschema.validate(case, schema)

    def test_schema_accepts_zero_sized_success_case(self) -> None:
        try:
            import jsonschema
        except ModuleNotFoundError as exc:
            self.skipTest(f"jsonschema unavailable: {exc}")

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        case = {
            "schema_version": 1,
            "case_id": "solve_f64_identity_000",
            "op": "solve",
            "dtype": "float64",
            "family": "identity",
            "expected_behavior": "success",
            "inputs": {
                "a": {"dtype": "float64", "shape": [0, 0], "order": "row_major", "data": []},
                "b": {"dtype": "float64", "shape": [0], "order": "row_major", "data": []},
            },
            "observable": {"kind": "identity"},
            "comparison": {"kind": "allclose", "rtol": 1e-8, "atol": 1e-9},
            "probes": [
                {
                    "probe_id": "p0",
                    "direction": {
                        "a": {"dtype": "float64", "shape": [0, 0], "order": "row_major", "data": []},
                        "b": {"dtype": "float64", "shape": [0], "order": "row_major", "data": []},
                    },
                    "cotangent": {
                        "value": {
                            "dtype": "float64",
                            "shape": [0],
                            "order": "row_major",
                            "data": [],
                        }
                    },
                    "pytorch_ref": {
                        "jvp": {
                            "value": {
                                "dtype": "float64",
                                "shape": [0],
                                "order": "row_major",
                                "data": [],
                            }
                        },
                        "vjp": {
                            "a": {
                                "dtype": "float64",
                                "shape": [0, 0],
                                "order": "row_major",
                                "data": [],
                            },
                            "b": {"dtype": "float64", "shape": [0], "order": "row_major", "data": []},
                        },
                    },
                    "fd_ref": {
                        "method": "central_difference",
                        "stencil_order": 2,
                        "step": 1e-6,
                        "jvp": {
                            "value": {
                                "dtype": "float64",
                                "shape": [0],
                                "order": "row_major",
                                "data": [],
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

        jsonschema.validate(case, schema)


if __name__ == "__main__":
    unittest.main()
