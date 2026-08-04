#!/usr/bin/env python3
"""Focused tests for the storage element-access baseline verifier."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify-storage-element-access-baseline.py"
BENCHMARK_PATH = "crates/tenferro-tensor/benches/element_access.rs"
REQUIRED_CASES = (
    "element_access/2d/col_major/direct_slice/4096",
    "element_access/2d/col_major/direct_slice_mut/4096",
    "element_access/2d/col_major/get/4096",
    "element_access/2d/col_major/get_unchecked/4096",
    "element_access/2d/col_major/get_mut/4096",
    "rank_fixed/2d/col_major/get2/4096",
    "rank_fixed/3d/col_major/get3/4096",
    "linear_iteration/col_major/as_slice_iter",
    "linear_iteration/col_major/dynamic_tensor_iter",
    "linear_iteration/col_major/tensor_iter",
    "linear_iteration/col_major/dynamic_tensor_iter_mut",
    "strided_traversal/rectangular_transpose/logical_order_get/3840",
)


class BaselineVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "tests@example.invalid"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Baseline Tests"],
            cwd=self.repo,
            check=True,
        )
        benchmark = self.repo / BENCHMARK_PATH
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text("fn main() {}\n", encoding="utf-8")
        subprocess.run(["git", "add", BENCHMARK_PATH], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "benchmark"],
            cwd=self.repo,
            check=True,
        )
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.report_path = self.repo / "baseline.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def report(self) -> dict[str, object]:
        return {
            "schema": "tenferro.storage-element-access-baseline.v1",
            "measured_commit": self.commit,
            "benchmark": {
                "path": BENCHMARK_PATH,
                "crate": "tenferro-tensor",
                "target": "element_access",
                "profile": "bench",
                "cargo_features": "default",
                "warm_up_seconds": 2.0,
                "measurement_seconds": 5.0,
                "sample_size": 100,
                "time_unit": "ns",
            },
            "environment": {
                "rustc_version": "rustc 1.97.1 (test)",
                "rustc_host": "x86_64-unknown-linux-gnu",
                "compilation_target": "x86_64-unknown-linux-gnu",
                "cargo_version": "cargo 1.97.1 (test)",
                "RUSTFLAGS": "",
                "CARGO_ENCODED_RUSTFLAGS": "",
                "os": "Linux 6.test",
                "architecture": "x86_64",
                "cpu_model": "Test CPU",
                "logical_cpu_count": 8,
                "cpu_affinity": [2],
                "thread_environment": {
                    "RAYON_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
            },
            "cases": [
                {
                    "id": case_id,
                    "estimate_ns": 100.0 + index,
                    "confidence_interval_ns": {
                        "confidence_level": 0.95,
                        "lower_bound": 99.0 + index,
                        "upper_bound": 101.0 + index,
                    },
                    "standard_error_ns": 0.5,
                }
                for index, case_id in enumerate(REQUIRED_CASES)
            ],
        }

    def run_verifier(self, report: dict[str, object]) -> subprocess.CompletedProcess[str]:
        self.report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return subprocess.run(
            [
                sys.executable,
                str(VERIFIER),
                "--root",
                str(self.repo),
                "--report",
                str(self.report_path),
            ],
            capture_output=True,
            text=True,
        )

    def test_accepts_measured_report_with_tracked_commit_provenance(self) -> None:
        result = self.run_verifier(self.report())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "storage-element-access-baseline-ok\n")
        self.assertEqual(result.stderr, "")

    def test_rejects_revision_alias_instead_of_canonical_commit(self) -> None:
        report = self.report()
        report["measured_commit"] = "HEAD"

        result = self.run_verifier(report)

        self.assertEqual(result.returncode, 1)
        self.assertIn("canonical Git object ID", result.stderr)

    def test_rejects_benchmark_path_missing_at_measured_commit(self) -> None:
        subprocess.run(["git", "rm", "-q", BENCHMARK_PATH], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "remove benchmark"],
            cwd=self.repo,
            check=True,
        )
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        result = self.run_verifier(self.report())

        self.assertEqual(result.returncode, 1)
        self.assertIn("not tracked at measured_commit", result.stderr)

    def test_rejects_benchmark_path_that_is_not_a_file_blob(self) -> None:
        benchmark = self.repo / BENCHMARK_PATH
        benchmark.unlink()
        benchmark.mkdir()
        (benchmark / "source").write_text("not a benchmark file\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "replace benchmark with directory"],
            cwd=self.repo,
            check=True,
        )
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        result = self.run_verifier(self.report())

        self.assertEqual(result.returncode, 1)
        self.assertIn("must be a tracked file", result.stderr)

    def test_rejects_missing_required_benchmark_case(self) -> None:
        report = self.report()
        cases = report["cases"]
        assert isinstance(cases, list)
        cases.pop()

        result = self.run_verifier(report)

        self.assertEqual(result.returncode, 1)
        self.assertIn("missing required benchmark cases", result.stderr)

    def test_rejects_non_finite_or_non_positive_timing_statistics(self) -> None:
        mutations = (
            ("estimate_ns", 0.0),
            ("standard_error_ns", float("inf")),
            ("confidence_interval_ns", {"confidence_level": 0.95, "lower_bound": -1.0, "upper_bound": 101.0}),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                report = self.report()
                case = report["cases"][0]
                assert isinstance(case, dict)
                case[field] = value

                result = self.run_verifier(report)

                self.assertEqual(result.returncode, 1)
                self.assertIn("finite", result.stderr)

    def test_accepts_zero_standard_error(self) -> None:
        report = self.report()
        cases = report["cases"]
        assert isinstance(cases, list)
        case = cases[0]
        assert isinstance(case, dict)
        case["standard_error_ns"] = 0.0

        result = self.run_verifier(report)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_missing_or_malformed_environment_metadata(self) -> None:
        mutations = (
            ("missing_cpu_model", lambda environment: environment.pop("cpu_model")),
            ("zero_logical_cpu_count", lambda environment: environment.__setitem__("logical_cpu_count", 0)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                report = self.report()
                environment = report["environment"]
                assert isinstance(environment, dict)
                mutate(environment)

                result = self.run_verifier(report)

                self.assertEqual(result.returncode, 1)
                self.assertIn("environment", result.stderr)


if __name__ == "__main__":
    unittest.main()
