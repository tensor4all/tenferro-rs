import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_regeneration, validate_schema, verify_cases


class VerifyCasesTests(unittest.TestCase):
    def test_find_duplicate_case_ids_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a.jsonl"
            second = root / "b.jsonl"
            first.write_text(json.dumps({"case_id": "dup"}) + "\n", encoding="utf-8")
            second.write_text(json.dumps({"case_id": "dup"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                verify_cases.find_duplicate_case_ids([first, second])

    def test_load_jsonl_records_reads_multiple_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"case_id": "a"}),
                        json.dumps({"case_id": "b"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = verify_cases.load_jsonl_records(path)

            self.assertEqual([record["case_id"] for record in records], ["a", "b"])


class CheckRegenerationTests(unittest.TestCase):
    def test_compare_case_trees_allows_numeric_drift_within_case_tolerance(self) -> None:
        expected_record = {
            "schema_version": 1,
            "case_id": "solve_f64_identity_001",
            "op": "solve",
            "dtype": "float64",
            "family": "identity",
            "expected_behavior": "success",
            "inputs": {
                "a": {
                    "dtype": "float64",
                    "shape": [1],
                    "order": "row_major",
                    "data": [1.0],
                }
            },
            "observable": {"kind": "identity"},
            "comparison": {
                "first_order": {"kind": "allclose", "rtol": 1e-8, "atol": 1e-9}
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
                "source_file": "x.py",
                "source_function": "sample_inputs_linalg_solve",
                "source_commit": "deadbeef",
                "generator": "python-pytorch-v1",
                "seed": 17,
                "torch_version": "2.10.0",
                "fd_policy_version": "v1",
            },
        }
        actual_record = json.loads(json.dumps(expected_record))
        actual_record["probes"][0]["pytorch_ref"]["jvp"]["value"]["data"][0] = 1.0 + 5e-9

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "expected"
            actual = root / "actual"
            (expected / "solve").mkdir(parents=True)
            (actual / "solve").mkdir(parents=True)
            (expected / "solve" / "identity.jsonl").write_text(
                json.dumps(expected_record) + "\n",
                encoding="utf-8",
            )
            (actual / "solve" / "identity.jsonl").write_text(
                json.dumps(actual_record) + "\n",
                encoding="utf-8",
            )

            check_regeneration.compare_case_trees(expected, actual)

    def test_compare_case_trees_detects_content_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "expected"
            actual = root / "actual"
            (expected / "solve").mkdir(parents=True)
            (actual / "solve").mkdir(parents=True)
            (expected / "solve" / "identity.jsonl").write_text(
                json.dumps({"case_id": "solve_f64_identity_001"}) + "\n",
                encoding="utf-8",
            )
            (actual / "solve" / "identity.jsonl").write_text(
                json.dumps({"case_id": "solve_f64_identity_999"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "content mismatch"):
                check_regeneration.compare_case_trees(expected, actual)

    def test_compare_case_trees_ignores_regenerated_comparison_threshold_drift(self) -> None:
        expected_record = {
            "schema_version": 1,
            "case_id": "eig_f64_values_vectors_abs_001",
            "op": "eig",
            "dtype": "float64",
            "family": "values_vectors_abs",
            "expected_behavior": "success",
            "inputs": {},
            "observable": {"kind": "eig_values_vectors_abs"},
            "comparison": {
                "first_order": {"kind": "allclose", "rtol": 1e-2, "atol": 1e-6}
            },
            "probes": [],
            "provenance": {},
        }
        actual_record = json.loads(json.dumps(expected_record))
        actual_record["comparison"]["first_order"]["rtol"] = 1e-3
        actual_record["comparison"]["first_order"]["atol"] = 1e-7

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "expected"
            actual = root / "actual"
            (expected / "eig").mkdir(parents=True)
            (actual / "eig").mkdir(parents=True)
            (expected / "eig" / "values_vectors_abs.jsonl").write_text(
                json.dumps(expected_record) + "\n",
                encoding="utf-8",
            )
            (actual / "eig" / "values_vectors_abs.jsonl").write_text(
                json.dumps(actual_record) + "\n",
                encoding="utf-8",
            )

            check_regeneration.compare_case_trees(expected, actual)


class ValidateSchemaTests(unittest.TestCase):
    def test_require_jsonschema_dependency_raises_clear_error(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ModuleNotFoundError("No module named 'jsonschema'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "jsonschema is required"):
                validate_schema.require_jsonschema()


if __name__ == "__main__":
    unittest.main()
