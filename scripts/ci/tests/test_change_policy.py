import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.ci.change_policy import ChangeClass, classify_paths


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "change_policy.py"


class ChangePolicyTests(unittest.TestCase):
    def test_docs_only_runs_docs_without_rust_or_gpu(self) -> None:
        policy = classify_paths(["docs/guides/cpu.md", "README.md"])
        self.assertIs(policy.change_class, ChangeClass.DOCS_ONLY)
        self.assertTrue(policy.run_docs)
        self.assertFalse(policy.run_rust)
        self.assertFalse(policy.run_extensions)
        self.assertFalse(policy.run_gpu)

    def test_ci_and_docs_runs_both_lightweight_suites(self) -> None:
        policy = classify_paths(
            [".github/workflows/ci.yml", "docs/worklogs/ci.md"]
        )
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertTrue(policy.run_docs)
        self.assertFalse(policy.run_rust)

    def test_runpod_control_plane_requires_gpu(self) -> None:
        for path in (
            "scripts/ci/runpod_client.py",
            "scripts/ci/runpod_config.json",
            ".github/workflows/runpod-gpu-test.yml",
            ".github/workflows/CI_gpu.yml",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
                self.assertTrue(policy.run_ci_config)
                self.assertTrue(policy.run_gpu)

    def test_unrelated_ci_change_does_not_require_gpu(self) -> None:
        policy = classify_paths([".github/workflows/docs.yml"])
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertFalse(policy.run_gpu)

    def test_unknown_and_empty_diffs_fall_back_to_code(self) -> None:
        for paths in ([], ["new-top-level-policy.toml"]):
            with self.subTest(paths=paths):
                policy = classify_paths(paths)
                self.assertIs(policy.change_class, ChangeClass.CODE)
                self.assertTrue(policy.run_rust)
                self.assertTrue(policy.run_extensions)
                self.assertTrue(policy.run_gpu)

    def test_push_to_main_forces_comprehensive_non_gpu_lanes(self) -> None:
        policy = classify_paths(["README.md"], event="push")
        self.assertIs(policy.change_class, ChangeClass.CODE)
        self.assertTrue(policy.run_rust)
        self.assertTrue(policy.run_extensions)
        self.assertTrue(policy.run_docs)
        self.assertTrue(policy.run_ci_config)

    def test_paths_are_normalized_and_deduplicated(self) -> None:
        policy = classify_paths([" ./README.md ", "README.md", ""])
        self.assertIs(policy.change_class, ChangeClass.DOCS_ONLY)
        self.assertEqual(policy.reasons, ("docs: README.md",))

    def test_cli_writes_stable_github_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            env = os.environ | {"GITHUB_OUTPUT": str(output)}
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--path", "README.md"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["classification"], "docs-only")
            values = dict(
                line.split("=", 1) for line in output.read_text().splitlines()
            )
            self.assertEqual(values["classification"], "docs-only")
            self.assertEqual(values["run_docs"], "true")
            self.assertEqual(values["run_rust"], "false")
            self.assertIn("README.md", values["reason"])


if __name__ == "__main__":
    unittest.main()
