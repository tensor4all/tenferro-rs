import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read(path: str) -> str:
    return (ROOT / path).read_text()


class WorkflowContractTests(unittest.TestCase):
    def test_fast_ci_uses_shared_policy_and_profiles(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertIn("python3 scripts/ci/change_policy.py", text)
        self.assertIn("python3 scripts/ci/run_profile.py blas-inject", text)
        self.assertIn("python3 scripts/ci/run_profile.py coverage", text)
        self.assertIn("python3 scripts/ci/run_profile.py docs", text)
        self.assertIn("name: CI configuration checks", text)

    def test_required_names_remain_stable(self) -> None:
        fast = read(".github/workflows/ci.yml")
        heavy = read(".github/workflows/ci-pr-workspace-tests.yml")
        for name in (
            "rustfmt",
            "clippy",
            "coverage",
            "docs-site",
            "cargo test (blas inject)",
        ):
            with self.subTest(name=name):
                self.assertIn(f"name: {name}", fast)
        self.assertIn("name: CI gate (PR workspace tests)", heavy)

    def test_fast_required_jobs_fail_if_policy_fails(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertGreaterEqual(text.count("needs.policy.result"), 6)
        self.assertIn("Change classification failed", text)

    def test_heavy_workflow_has_explicit_noop_matrix_and_gate_contract(self) -> None:
        text = read(".github/workflows/ci-pr-workspace-tests.yml")
        self.assertIn('"backend":"not-required"', text)
        self.assertIn("RUN_WORKSPACE", text)
        self.assertIn("RUN_EXTENSIONS", text)
        self.assertIn("Workspace tests not required", text)
        self.assertIn("python3 scripts/ci/run_profile.py", text)
        self.assertNotIn("grep -qE", text)

    def test_ci_config_installs_a_pinned_actionlint(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertIn(
            "github.com/rhysd/actionlint/cmd/actionlint@v1.7.7", text
        )


if __name__ == "__main__":
    unittest.main()
