import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import (
    check_docs_site,
    check_math_registry,
    check_regeneration,
    check_replay,
    validate_schema,
    verify_cases,
)


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

    def test_compare_case_trees_treats_nan_payloads_as_equal(self) -> None:
        expected_record = {
            "schema_version": 1,
            "case_id": "nanmean_f32_identity_001",
            "op": "nanmean",
            "dtype": "float32",
            "family": "identity",
            "expected_behavior": "success",
            "inputs": {
                "a": {
                    "dtype": "float32",
                    "shape": [2],
                    "order": "row_major",
                    "data": [1.0, float("nan")],
                }
            },
            "observable": {"kind": "identity"},
            "comparison": {
                "first_order": {"kind": "allclose", "rtol": 1e-4, "atol": 1e-6}
            },
            "probes": [],
            "provenance": {},
        }
        actual_record = json.loads(json.dumps(expected_record))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "expected"
            actual = root / "actual"
            (expected / "nanmean").mkdir(parents=True)
            (actual / "nanmean").mkdir(parents=True)
            (expected / "nanmean" / "identity.jsonl").write_text(
                json.dumps(expected_record) + "\n",
                encoding="utf-8",
            )
            (actual / "nanmean" / "identity.jsonl").write_text(
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


class CheckReplayScriptTests(unittest.TestCase):
    def test_main_reports_checked_count(self) -> None:
        with patch.object(
            check_replay,
            "replay_case_tree",
            return_value=type("ReplayResult", (), {"checked": 7, "failures": []})(),
        ):
            self.assertEqual(check_replay.main(), 0)

    def test_main_raises_on_failures(self) -> None:
        with patch.object(
            check_replay,
            "replay_case_tree",
            return_value=type(
                "ReplayResult",
                (),
                {"checked": 3, "failures": ["bad_case: mismatch"]},
            )(),
        ):
            with self.assertRaisesRegex(SystemExit, "bad_case: mismatch"):
                check_replay.main()


class CheckMathRegistryScriptTests(unittest.TestCase):
    def test_main_reports_success_for_valid_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "docs" / "math").mkdir(parents=True)
            (root / "cases" / "solve").mkdir(parents=True)
            (root / "docs" / "math" / "solve.md").write_text(
                "<a id=\"family-identity\"></a>\n",
                encoding="utf-8",
            )
            (root / "docs" / "math" / "registry.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "op": "solve",
                                "family": "identity",
                                "note_path": "docs/math/solve.md",
                                "anchor": "family-identity",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "cases" / "solve" / "identity.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )

            with patch.object(check_math_registry, "REPO_ROOT", root):
                self.assertEqual(check_math_registry.main(), 0)

    def test_main_raises_on_invalid_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "docs" / "math").mkdir(parents=True)
            (root / "docs" / "math" / "registry.json").write_text(
                json.dumps({"version": 1, "entries": []}),
                encoding="utf-8",
            )
            (root / "cases" / "solve").mkdir(parents=True)
            (root / "cases" / "solve" / "identity.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )

            with patch.object(check_math_registry, "REPO_ROOT", root):
                with self.assertRaisesRegex(SystemExit, "missing registry entries"):
                    check_math_registry.main()


class CheckDocsSiteScriptTests(unittest.TestCase):
    def test_main_reports_success_for_valid_site(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_root = Path(tmpdir)
            (site_root / "math").mkdir(parents=True)
            (site_root / "index.html").write_text("<h1>Home</h1>\n", encoding="utf-8")
            (site_root / "math" / "index.html").write_text("<h1>Math</h1>\n", encoding="utf-8")
            (site_root / "math" / "svd.html").write_text("<h1>SVD</h1>\n", encoding="utf-8")
            (site_root / "math" / "registry.json").write_text("{}", encoding="utf-8")

            self.assertEqual(check_docs_site.main(["--site-root", str(site_root)]), 0)

    def test_main_raises_for_missing_required_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_root = Path(tmpdir)
            (site_root / "math").mkdir(parents=True)
            (site_root / "index.html").write_text("<h1>Home</h1>\n", encoding="utf-8")
            (site_root / "math" / "index.html").write_text("<h1>Math</h1>\n", encoding="utf-8")
            (site_root / "math" / "registry.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "math/svd.html"):
                check_docs_site.main(["--site-root", str(site_root)])


if __name__ == "__main__":
    unittest.main()
