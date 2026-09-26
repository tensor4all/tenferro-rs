from pathlib import Path
import json
import os
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from scripts.ci.gpu_test_partition import (
    filter_expression,
    main,
    read_partition,
    report,
    validate_inventory,
)

ROOT = Path(__file__).resolve().parents[3]


class GpuTestPartitionTests(unittest.TestCase):
    def setUp(self):
        self.partition = {("crate", "host_case"): "host", ("crate", "device_case"): "gpu"}
        self.inventory = {
            "test-count": 2,
            "rust-suites": {"crate": {"testcases": {"host_case": {}, "device_case": {}}}},
        }

    def test_exact_inventory_accepts_both_lanes(self):
        validate_inventory(self.inventory, self.partition)
        self.assertEqual(
            filter_expression(self.partition, "gpu"),
            "(package(=crate) & binary(=crate) & (test(=device_case)))",
        )
        self.assertEqual(
            filter_expression(self.partition, "host"),
            "not ((package(=crate) & binary(=crate) & (test(=device_case))))",
        )

    def test_new_removed_or_renamed_tests_fail_closed(self):
        for names in ({"host_case", "device_case", "new_case"}, {"host_case"}, {"host_case", "renamed"}):
            with self.subTest(names=names):
                inventory = {"test-count": len(names), "rust-suites": {"crate": {"testcases": dict.fromkeys(names, {})}}}
                with self.assertRaisesRegex(ValueError, "audit test bodies/helpers"):
                    validate_inventory(inventory, self.partition)

    def test_empty_and_inconsistent_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "empty or inconsistent"):
            validate_inventory({"test-count": 0, "rust-suites": {}}, {})
        self.inventory["test-count"] = 3
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_inventory(self.inventory, self.partition)

    def test_duplicate_assignments_and_unknown_lane_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.tsv"
            for text in ("host\tx\tt\ngpu\tx\tt\n", "guess\tx\tt\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_partition(path)

    def test_binary_identity_is_part_of_membership(self):
        partition = {("crate::integration", "device_case"): "gpu"}
        with self.assertRaises(ValueError):
            validate_inventory(self.inventory, partition)
        self.assertIn("binary(=integration)", filter_expression(partition, "gpu"))

    def test_reviewed_exclusions_are_reported_not_passed(self):
        cuda = read_partition(ROOT / "scripts/ci/cuda_test_partition.tsv")
        pjrt = read_partition(ROOT / "scripts/ci/pjrt_test_partition.tsv")
        self.assertEqual(sum(v == "two-gpu" for v in cuda.values()), 1)
        self.assertEqual(sum(v == "compile-only" for v in cuda.values()), 2)
        self.assertEqual(sum(v == "benchmark" for v in cuda.values()), 1)
        self.assertEqual(sum(v == "external-tool" for v in pjrt.values()), 3)
        self.assertIn("NOT RUN (two-gpu):", report(cuda, "gpu"))
        self.assertIn("NOT RUN (external-tool):", report(pjrt, "host"))
        self.assertNotIn("PASS", report(cuda, "gpu"))

    def test_host_execution_is_required_before_paid_lifecycle_even_on_cache_hit(self):
        parent = (ROOT / ".github/workflows/runpod-gpu-test.yml").read_text()
        child = (ROOT / ".github/workflows/runpod-gpu-execute.yml").read_text()
        host_step = parent.split("      - name: Run device-independent CUDA and PJRT tests", 1)[1].split("      - uses:", 1)[0]
        self.assertNotIn("if:", host_step)
        installer = parent.split("      - uses: taiki-e/install-action@", 1)[1].split("      - name:", 1)[0]
        self.assertIn("tool: nextest", installer)
        self.assertNotIn("if:", installer)
        self.assertLess(parent.index("      - uses: taiki-e/install-action@"), parent.index("      - name: Run device-independent CUDA and PJRT tests"))
        for kind in ("cuda", "pjrt"):
            self.assertIn(f"gpu_test_partition.py --kind {kind} --lane host", host_step)
            self.assertIn(f"gpu_test_partition.py --kind {kind} --lane gpu", child)
        self.assertIn("needs: [authorize, runpod-contract, pre-runpod-gate, cuda-archive]", parent)
        self.assertNotIn("cargo nextest run", child)

    def test_hardware_classification_is_not_a_name_heuristic(self):
        cuda = read_partition(ROOT / "scripts/ci/cuda_test_partition.tsv")
        self.assertEqual(cuda[("tenferro-gpu::storage_provider_cuda", "cuda_provider_does_not_expose_safe_unscoped_raw_access")], "host")
        self.assertEqual(cuda[("tenferro-linalg::integration", "determinant_extremes::cuda_complex_determinant_retains_existing_support")], "gpu")
        self.assertEqual(cuda[("tenferro-linalg::integration", "determinant_extremes::determinant_extremes_preserve_value_and_sign")], "host")

    def test_gpu_timeout_is_twice_the_observed_maximum(self):
        config = tomllib.loads((ROOT / "scripts/ci/gpu_nextest.toml").read_text())
        self.assertEqual(config["profile"]["gpu-ci"]["slow-timeout"], {
            "period": "100s", "terminate-after": 2, "grace-period": "10s",
        })

    def test_only_gpu_execution_selects_the_timeout_profile(self):
        for kind in ("cuda", "pjrt"):
            for lane in ("host", "gpu"):
                with self.subTest(kind=kind, lane=lane), \
                     patch("sys.argv", ["gpu_test_partition.py", "--kind", kind,
                                        "--lane", lane, "--archive-file", "tests.tar.zst"]), \
                     patch.dict(os.environ, {}, clear=True), \
                     patch("scripts.ci.gpu_test_partition.read_partition", return_value=self.partition), \
                     patch("scripts.ci.gpu_test_partition.subprocess.run") as run:
                    run.return_value = subprocess.CompletedProcess([], 0, json.dumps(self.inventory))
                    self.assertEqual(main(), 0)
                    command = run.call_args.args[0]
                    self.assertEqual(command[:3], ["cargo", "nextest", "run"])
                    if lane == "gpu":
                        self.assertEqual(command[command.index("--profile") + 1], "gpu-ci")
                        config_path = Path(command[command.index("--config-file") + 1])
                        self.assertEqual(config_path.resolve(), ROOT / "scripts/ci/gpu_nextest.toml")
                        self.assertEqual(run.call_args.kwargs["env"]["TENFERRO_REQUIRE_GPU"], "1")
                    else:
                        self.assertNotIn("--profile", command)
                        self.assertNotIn("--config-file", command)


if __name__ == "__main__":
    unittest.main()
