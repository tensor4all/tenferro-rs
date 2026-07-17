import io
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ci.run_profile import commands_for, expand_profiles, run_profiles


ROOT = Path(__file__).resolve().parents[3]


class RunProfileTests(unittest.TestCase):
    def test_default_local_profiles_are_incremental_and_unoptimized(self) -> None:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        expected = {
            "opt-level": 0,
            "debug": 0,
            "debug-assertions": True,
            "overflow-checks": True,
            "incremental": True,
        }
        self.assertEqual(manifest["profile"]["dev"], expected)
        self.assertEqual(manifest["profile"]["test"], expected)

    def test_non_incremental_local_gate_profile_is_removed(self) -> None:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        self.assertNotIn("local-gate", manifest["profile"])
        with self.assertRaisesRegex(ValueError, "unknown CI profile"):
            commands_for("local-gate")

    def test_hosted_full_profile_does_not_include_local_gate(self) -> None:
        self.assertNotIn("local-gate", expand_profiles(["full"]))

    def test_workspace_blas_matches_ci_feature_contract(self) -> None:
        self.assertEqual(
            commands_for("workspace-blas"),
            (
                "cargo nextest run --workspace --release --no-default-features "
                "--features cpu-blas --no-fail-fast",
                "cargo test --doc --workspace --release --no-default-features "
                "--features cpu-blas",
            ),
        )

    def test_full_profile_expands_named_profiles_once(self) -> None:
        expanded = expand_profiles(["full"])
        self.assertEqual(
            expanded,
            (
                "workspace-faer",
                "workspace-blas",
                "blas-inject",
                "extensions",
                "docs",
                "coverage",
                "ci-config",
            ),
        )
        self.assertEqual(len(expanded), len(set(expanded)))

    def test_duplicate_profile_composition_is_deduplicated(self) -> None:
        expanded = expand_profiles(["workspace-faer", "full"])
        self.assertEqual(expanded[0], "workspace-faer")
        self.assertEqual(expanded.count("workspace-faer"), 1)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown CI profile"):
            expand_profiles(["expensive-typo"])

    def test_dry_run_prints_commands_without_executing(self) -> None:
        output = io.StringIO()
        with patch("scripts.ci.run_profile.subprocess.run") as run:
            run_profiles(["blas-inject"], dry_run=True, output=output)
        run.assert_not_called()
        self.assertIn("+ cargo test -p tenferro-cpu", output.getvalue())

    def test_workspace_blas_uses_linker_flags_only_for_that_profile(self) -> None:
        calls: list[dict[str, str]] = []

        def record_run(*_args: object, **kwargs: object) -> None:
            calls.append(kwargs["env"])  # type: ignore[arg-type]

        with patch.dict("os.environ", {}, clear=True), patch(
            "scripts.ci.run_profile.subprocess.run", side_effect=record_run
        ):
            run_profiles(
                ["workspace-faer", "workspace-blas"],
                dry_run=False,
                output=io.StringIO(),
            )
        self.assertNotIn("RUSTFLAGS", calls[0])
        self.assertEqual(
            calls[2]["RUSTFLAGS"], "-l dylib=openblas -l dylib=lapack"
        )

    def test_fast_preflight_delegates_profiles_without_redefining_them(self) -> None:
        source = (ROOT / "scripts" / "check-pr-fast.sh").read_text()
        self.assertIn("--ci-profile", source)
        self.assertIn('python3 scripts/ci/run_profile.py "${profile_args[@]}"', source)
        self.assertNotIn("cargo nextest run --workspace", source)

    def test_create_pr_uses_local_gate_instead_of_release_suite(self) -> None:
        source = (ROOT / "scripts" / "create-pr.sh").read_text()
        self.assertIn("bash scripts/check-pr-fast.sh", source)
        self.assertIn("--ci-profile local-gate", source)
        self.assertIn("python3 scripts/repository-rules-review.py", source)
        self.assertNotIn("cargo nextest run --workspace --release", source)
        self.assertNotIn("cargo llvm-cov", source)

    def test_remediation_workflow_uses_local_gate_before_pr(self) -> None:
        source = (
            ROOT / "ai" / "contribution-workflows" / "repository-remediation.md"
        ).read_text()
        self.assertIn("bash scripts/check-pr-fast.sh", source)
        self.assertIn("--ci-profile local-gate", source)
        self.assertNotIn("cargo test --workspace --release", source)
        self.assertNotIn("cargo llvm-cov --workspace --release", source)

    def test_sccache_policy_preserves_incremental_ai_development(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text()
        contributing = (ROOT / "CONTRIBUTING.md").read_text()
        design = (
            ROOT
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-07-17-local-pr-gate-design.md"
        ).read_text()

        for source in (agents, contributing, design):
            self.assertIn("AI-assisted edit-test loops", source)
            self.assertRegex(source, r"Cargo\s+incremental compilation")

        self.assertNotIn(
            "Before the first workspace-wide local Rust build", agents
        )


if __name__ == "__main__":
    unittest.main()
