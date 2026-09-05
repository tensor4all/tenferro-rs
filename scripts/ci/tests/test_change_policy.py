import json
import os
import shutil
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
        self.assertFalse(policy.run_macos)

    def test_repository_rules_are_docs_but_do_not_hide_code(self) -> None:
        paths = ["AGENTS.md", "REPOSITORY_RULES.md"]
        self.assertIs(classify_paths(paths).change_class, ChangeClass.DOCS_ONLY)
        for extra in ("src/lib.rs", "unknown.md"):
            with self.subTest(extra=extra):
                self.assertIs(
                    classify_paths(paths + [extra]).change_class, ChangeClass.CODE
                )

    def test_ci_and_docs_runs_both_lightweight_suites(self) -> None:
        policy = classify_paths(
            [".github/workflows/ci.yml", "docs/worklogs/ci.md"]
        )
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertTrue(policy.run_docs)
        self.assertFalse(policy.run_rust)
        self.assertFalse(policy.run_macos)

    def test_macos_control_plane_requires_macos(self) -> None:
        for path in (
            ".github/workflows/ci-pr-workspace-tests.yml",
            "scripts/ci/change_policy.py",
            "scripts/ci/run_profile.py",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
                self.assertTrue(policy.run_ci_config)
                self.assertTrue(policy.run_macos)

    def test_inventory_checker_uses_ci_config_lane(self) -> None:
        policy = classify_paths(["scripts/check-public-boundary-inventory.py"])
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertFalse(policy.run_docs)
        self.assertFalse(policy.run_rust)

    def test_rule_review_tests_use_ci_config_lane(self) -> None:
        policy = classify_paths(["scripts/test-repository-rules-review.py"])
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertFalse(policy.run_rust)

    def test_unrelated_ci_change_does_not_require_macos(self) -> None:
        policy = classify_paths([".github/workflows/docs.yml"])
        self.assertFalse(policy.run_macos)

    def test_runpod_control_plane_requires_gpu(self) -> None:
        for path in (
            "scripts/ci/runpod_client.py",
            "scripts/ci/runpod_config.json",
            ".github/workflows/runpod-gpu-test.yml",
            ".github/workflows/CI_gpu.yml",
            ".github/workflows/ci-cache-publish.yml",
            "scripts/ci/find_archive_artifact.py",
            "scripts/ci/install_cuda_toolkit_hosted.sh",
            "scripts/ci/install_cuda_runtime_tree.sh",
            "scripts/ci/install_cutensor.sh",
            "scripts/ci/runpod_pricing.py",
            "scripts/ci/runpod_provision.py",
            "scripts/ci/cuda_smoke_test.py",
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
                self.assertTrue(policy.run_macos)

    def test_push_to_main_forces_comprehensive_lanes(self) -> None:
        policy = classify_paths(["README.md"], event="push")
        self.assertIs(policy.change_class, ChangeClass.CODE)
        self.assertTrue(policy.run_rust)
        self.assertTrue(policy.run_extensions)
        self.assertTrue(policy.run_docs)
        self.assertTrue(policy.run_ci_config)
        self.assertTrue(policy.run_macos)

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
            self.assertEqual(values["run_macos"], "false")
            self.assertIn("README.md", values["reason"])

    def test_cli_classifies_deleted_documentation_as_docs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo,
                check=True,
            )
            docs = repo / "docs"
            docs.mkdir()
            guide = docs / "guide.md"
            guide.write_text("guide\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "add guide"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            base = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            guide.unlink()
            subprocess.run(["git", "add", "-u"], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "remove guide"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base",
                    base,
                    "--head",
                    head,
                ],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(result.stdout)["classification"], "docs-only")


class LocalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        (self.repo / "scripts" / "ci").mkdir(parents=True)
        (self.repo / "scripts" / "lib").mkdir(parents=True)
        shutil.copy2(ROOT / "scripts" / "check-pr-fast.sh", self.repo / "scripts")
        shutil.copy2(
            ROOT / "scripts" / "ci" / "change_policy.py",
            self.repo / "scripts" / "ci",
        )
        shutil.copy2(
            ROOT / "scripts" / "ci" / "run_profile.py",
            self.repo / "scripts" / "ci",
        )
        shutil.copy2(
            ROOT / "scripts" / "lib" / "python.sh",
            self.repo / "scripts" / "lib",
        )
        bin_dir = self.repo / "bin"
        bin_dir.mkdir()
        cargo = bin_dir / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" >> \"$CARGO_MARKER\"\n"
        )
        cargo.chmod(0o755)
        subprocess.run(["git", "init", "-b", "test-branch"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.repo,
            check=True,
        )
        (self.repo / "README.md").write_text("baseline\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "baseline"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        self.marker = self.repo / "cargo-calls"
        self.env = os.environ | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CARGO_MARKER": str(self.marker),
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_gate(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "scripts/check-pr-fast.sh",
                "--base",
                "HEAD",
                "--no-fetch",
                "--skip-doc-snippets",
                *args,
            ],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def write_change(self, path: str) -> None:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n")

    def test_code_change_requires_a_focused_command(self) -> None:
        self.write_change("src/lib.rs")
        result = self.run_gate("--coverage-reviewed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("focused verification command required", result.stderr)

    def test_code_change_runs_focused_command_incrementally(self) -> None:
        self.write_change("src/lib.rs")
        result = self.run_gate("--coverage-reviewed", "--test", "true")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("classification: code", result.stdout)
        self.assertTrue(self.marker.exists(), "code changes should run cargo fmt")

    def test_clean_base_head_is_a_valid_no_change_gate_under_bin_bash(self) -> None:
        compatible_python_dir = Path(sys.executable).resolve().parent
        env = self.env | {
            "PATH": f"{compatible_python_dir}:{self.env['PATH']}",
        }
        result = subprocess.run(
            [
                "/bin/bash",
                "scripts/check-pr-fast.sh",
                "--base",
                "HEAD",
                "--no-fetch",
                "--skip-doc-snippets",
                "--coverage-reviewed",
                "--test",
                "true",
            ],
            cwd=self.repo,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("changed files: none", result.stdout)

    def test_docs_only_change_skips_cargo_and_coverage_acknowledgement(self) -> None:
        self.write_change("docs/guide.md")
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("classification: docs-only", result.stdout)
        self.assertFalse(self.marker.exists(), "docs-only must not invoke Cargo")

    def test_repository_rules_skip_cargo_and_coverage_acknowledgement(self) -> None:
        self.write_change("AGENTS.md")
        self.write_change("REPOSITORY_RULES.md")
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("classification: docs-only", result.stdout)
        self.assertFalse(self.marker.exists(), "rule docs must not invoke Cargo")

    def test_untracked_docs_with_trailing_whitespace_fail(self) -> None:
        target = self.repo / "docs" / "guide.md"
        target.parent.mkdir(parents=True)
        target.write_text("trailing whitespace   \n")
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trailing whitespace", result.stdout + result.stderr)

    def test_ci_only_change_requires_a_focused_command(self) -> None:
        self.write_change("scripts/ci/example.py")
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("focused verification command required", result.stderr)

    def test_ci_only_change_accepts_an_explicit_ci_profile(self) -> None:
        self.write_change("scripts/ci/example.py")
        result = self.run_gate(
            "--ci-profile", "ci-config", "--ci-profile-dry-run"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("python3 -m unittest discover", result.stdout)


if __name__ == "__main__":
    unittest.main()
