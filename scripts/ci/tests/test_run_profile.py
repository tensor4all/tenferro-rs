import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
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

    def test_hosted_ci_profile_is_non_incremental_and_stripped(self) -> None:
        root = tomllib.loads((ROOT / "Cargo.toml").read_text())["profile"]["ci"]
        self.assertEqual(root["inherits"], "test")
        self.assertFalse(root["incremental"])
        self.assertEqual(root["strip"], "symbols")

        nested_expected = {
            "inherits": "test",
            "debug": 0,
            "incremental": False,
            "strip": "symbols",
        }
        for relative in (
            "ext/tropical/Cargo.toml",
            "ext/sparse/Cargo.toml",
            "ext/tenferro-cpu-tblis/Cargo.toml",
            "samples/kdv-pinn/Cargo.toml",
        ):
            manifest = tomllib.loads((ROOT / relative).read_text())
            with self.subTest(manifest=relative):
                self.assertEqual(manifest["profile"]["ci"], nested_expected)

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
                "cargo-nextest nextest run --workspace --cargo-profile ci "
                "--no-default-features "
                "--features cpu-blas --no-fail-fast",
                "cargo test --doc --workspace --profile ci --no-default-features "
                "--features cpu-blas",
            ),
        )

    def test_fmt_profile_covers_workspace_and_standalone_extensions(self) -> None:
        self.assertEqual(
            commands_for("fmt"),
            (
                "cargo fmt --all --check",
                "cargo fmt --manifest-path ext/tropical/Cargo.toml --all --check",
                "cargo fmt --manifest-path ext/sparse/Cargo.toml --all --check",
                "cargo fmt --manifest-path ext/tenferro-cpu-tblis/Cargo.toml --all --check",
            ),
        )

    def test_clippy_profile_matches_hosted_ci_contract(self) -> None:
        self.assertEqual(
            commands_for("clippy"),
            (
                "cargo clippy --workspace --all-targets -- -D warnings "
                "-D clippy::missing_errors_doc -D clippy::missing_panics_doc",
                "cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets "
                "-- -D warnings -D clippy::missing_errors_doc "
                "-D clippy::missing_panics_doc",
                "cargo clippy --manifest-path ext/sparse/Cargo.toml --all-targets "
                "-- -D warnings -D clippy::missing_errors_doc "
                "-D clippy::missing_panics_doc",
                "cargo clippy --manifest-path ext/tenferro-cpu-tblis/Cargo.toml "
                "--all-targets -- -D warnings -D clippy::missing_errors_doc "
                "-D clippy::missing_panics_doc",
            ),
        )

    def test_hosted_profiles_use_cargo_ci_profile_not_release(self) -> None:
        for name in (
            "workspace-faer",
            "workspace-blas",
            "blas-inject",
            "extensions",
            "coverage",
        ):
            joined = "\n".join(commands_for(name))
            with self.subTest(profile=name):
                self.assertNotIn("--release", joined)
                self.assertTrue(
                    "--cargo-profile ci" in joined or "--profile ci" in joined
                )

    def test_docs_profile_checks_dependency_footprint_generator(self) -> None:
        commands = commands_for("docs")

        self.assertIn("python3 scripts/test-gen-dep-graph.py", commands)
        self.assertLess(
            commands.index("python3 scripts/test-gen-dep-graph.py"),
            commands.index("bash scripts/build_docs_site.sh"),
        )

    def test_ci_config_checks_storage_ownership_contract_ledger(self) -> None:
        commands = commands_for("ci-config")
        self.assertIn(
            "python3 scripts/test-storage-ownership-contracts-v2.py", commands
        )
        self.assertIn("python3 scripts/check-storage-ownership-contracts.py", commands)
        self.assertLess(
            commands.index("python3 scripts/test-storage-ownership-contracts-v2.py"),
            commands.index("python3 scripts/check-storage-ownership-contracts.py"),
        )

    def test_ci_config_dry_run_prints_storage_ownership_checker(self) -> None:
        output = io.StringIO()
        with patch("scripts.ci.run_profile.subprocess.run") as run:
            run_profiles(["ci-config"], dry_run=True, output=output)
        run.assert_not_called()
        lines = output.getvalue().splitlines()
        self.assertIn(
            "+ python3 scripts/test-storage-ownership-contracts-v2.py", lines
        )
        self.assertIn(
            "+ python3 scripts/check-storage-ownership-contracts.py",
            lines,
        )
        self.assertLess(
            lines.index("+ python3 scripts/test-storage-ownership-contracts-v2.py"),
            lines.index("+ python3 scripts/check-storage-ownership-contracts.py"),
        )

    def test_ci_config_base_is_appended_once_and_shell_quoted(self) -> None:
        output = io.StringIO()
        base = "refs/heads/base branch;not-a-command"
        try:
            run_profiles(
                ["ci-config"],
                dry_run=True,
                output=output,
                storage_ownership_base=base,
            )
        except TypeError as error:
            self.fail(f"run_profiles does not accept a storage ownership base: {error}")

        checker_lines = [
            line
            for line in output.getvalue().splitlines()
            if "scripts/check-storage-ownership-contracts.py" in line
        ]
        self.assertEqual(
            checker_lines,
            [
                "+ python3 scripts/check-storage-ownership-contracts.py "
                f"--base-commit {shlex.quote(base)}"
            ],
        )

    def test_storage_ownership_base_requires_ci_config(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "storage ownership base requires the ci-config profile"
        ):
            run_profiles(
                ["fmt"],
                dry_run=True,
                output=io.StringIO(),
                storage_ownership_base="base-commit",
            )

    def test_hosted_ci_config_supplies_event_base_with_full_history(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        ci_config_job = workflow.split("\n  ci-config:\n", maxsplit=1)[1]

        self.assertIn("fetch-depth: 0", ci_config_job)
        self.assertIn("EVENT_NAME: ${{ github.event_name }}", ci_config_job)
        self.assertIn(
            "PR_BASE_SHA: ${{ github.event.pull_request.base.sha }}", ci_config_job
        )
        self.assertIn("PUSH_BASE_SHA: ${{ github.event.before }}", ci_config_job)
        self.assertIn('if [ "${EVENT_NAME}" = pull_request ]; then', ci_config_job)
        self.assertIn('elif [ "${EVENT_NAME}" = push ]; then', ci_config_job)
        self.assertIn('BASE_SHA="${PR_BASE_SHA}"', ci_config_job)
        self.assertIn('BASE_SHA="${PUSH_BASE_SHA}"', ci_config_job)
        invocation = (
            "python3 scripts/ci/run_profile.py ci-config "
            '--storage-ownership-base "${BASE_SHA}"'
        )
        self.assertEqual(ci_config_job.count(invocation), 1)

    def test_full_profile_expands_named_profiles_once(self) -> None:
        expanded = expand_profiles(["full"])
        self.assertEqual(
            expanded,
            (
                "fmt",
                "clippy",
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
        for call in calls[:2]:
            self.assertNotIn("RUSTFLAGS", call)
            self.assertNotIn("TENFERRO_TRYBUILD_RUSTFLAGS", call)
            self.assertNotIn("CARGO", call)
        for call in calls[2:]:
            self.assertEqual(
                call["RUSTFLAGS"], "-l dylib=openblas -l dylib=lapack"
            )
            self.assertEqual(
                call["TENFERRO_TRYBUILD_RUSTFLAGS"],
                "-l dylib=openblas -l dylib=lapack",
            )
            self.assertEqual(
                call["CARGO"],
                str((ROOT / "scripts" / "ci" / "trybuild-cargo.py").resolve()),
            )

    def test_trybuild_cargo_preserves_and_augments_complete_rustflags(self) -> None:
        build_flags = [
            "--cfg",
            "trybuild",
            "--verbose",
            "--diagnostic-width=140",
            "-A",
            "dead_code",
            "-C",
            "instrument-coverage",
        ]
        target_flags = [
            "--cfg",
            "target-specific",
            "-C",
            "target-cpu=native",
        ]
        blas_flags = ["-l", "dylib=openblas", "-l", "dylib=lapack"]

        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            capture = temp / "capture.json"
            fake_cargo = temp / "cargo"
            fake_cargo.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

Path(os.environ["TRYBUILD_CARGO_CAPTURE"]).write_text(
    json.dumps({"args": sys.argv[1:], "cargo": os.environ.get("CARGO")})
)
"""
            )
            fake_cargo.chmod(0o755)

            environment = os.environ.copy()
            environment["PATH"] = f"{temp}{os.pathsep}{environment['PATH']}"
            environment["CARGO"] = "must-not-recurse"
            environment["TENFERRO_TRYBUILD_RUSTFLAGS"] = " ".join(blas_flags)
            environment["TRYBUILD_CARGO_CAPTURE"] = str(capture)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "ci" / "trybuild-cargo.py"),
                    "check",
                    "--config=build.rustflags=" + json.dumps(build_flags),
                    "--config=target.x86_64-unknown-linux-gnu.rustflags="
                    + json.dumps(target_flags),
                    "--offline",
                ],
                check=True,
                env=environment,
            )

            result = json.loads(capture.read_text())

        build_argument = next(
            argument
            for argument in result["args"]
            if argument.startswith("--config=build.rustflags=")
        )
        target_argument = next(
            argument
            for argument in result["args"]
            if argument.startswith(
                "--config=target.x86_64-unknown-linux-gnu.rustflags="
            )
        )
        self.assertEqual(
            json.loads(build_argument.split("=", 2)[2]),
            build_flags + blas_flags,
        )
        self.assertEqual(
            json.loads(target_argument.split("=", 2)[2]),
            target_flags + blas_flags,
        )
        self.assertIn("--offline", result["args"])
        self.assertIsNone(result["cargo"])

    def test_trybuild_cargo_runs_without_tomllib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "tomllib.py").write_text(
                'raise ModuleNotFoundError("No module named tomllib")\n'
            )
            fake_cargo = temp / "cargo"
            fake_cargo.write_text("#!/bin/sh\nexit 0\n")
            fake_cargo.chmod(0o755)

            environment = os.environ.copy()
            environment["PATH"] = f"{temp}{os.pathsep}{environment['PATH']}"
            environment["PYTHONPATH"] = str(temp)
            environment["TENFERRO_TRYBUILD_RUSTFLAGS"] = "-l dylib=openblas"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "ci" / "trybuild-cargo.py"),
                    "check",
                    '--config=build.rustflags=["--cfg","trybuild"]',
                ],
                check=True,
                env=environment,
            )

    def test_fast_preflight_delegates_profiles_without_redefining_them(self) -> None:
        source = (ROOT / "scripts" / "check-pr-fast.sh").read_text()
        self.assertIn("--ci-profile", source)
        self.assertIn("python3 scripts/ci/run_profile.py fmt", source)
        self.assertIn('python3 scripts/ci/run_profile.py "${profile_args[@]}"', source)
        self.assertNotIn("cargo nextest run --workspace", source)

    def test_fast_preflight_runs_ci_parity_clippy_for_code_changes(self) -> None:
        source = (ROOT / "scripts" / "check-pr-fast.sh").read_text()
        self.assertIn('if [[ "${change_class}" == "code" ]]; then', source)
        self.assertIn("python3 scripts/ci/run_profile.py clippy", source)
        self.assertIn("has_ci_profile clippy || has_ci_profile full", source)

    def test_fast_preflight_avoids_bash4_only_mapfile(self) -> None:
        source = (ROOT / "scripts" / "check-pr-fast.sh").read_text()
        self.assertNotIn("mapfile", source)
        self.assertIn("while IFS= read -r field", source)

    def test_create_pr_forwards_focused_tests(self) -> None:
        source = (ROOT / "scripts" / "create-pr.sh").read_text()
        self.assertIn("bash scripts/check-pr-fast.sh", source)
        self.assertIn("FOCUSED_TESTS=()", source)
        self.assertIn("--test)", source)
        self.assertIn('fast_gate_args+=(--test "$command")', source)
        self.assertIn("Focused local verification", source)
        self.assertIn("python3 scripts/repository-rules-review.py", source)
        self.assertNotIn("--ci-profile local-gate", source)
        self.assertNotIn("cargo nextest run --workspace --release", source)
        self.assertNotIn("cargo llvm-cov", source)

    def test_create_pr_pushes_the_named_branch_not_its_base_upstream(self) -> None:
        source = (ROOT / "scripts" / "create-pr.sh").read_text()
        self.assertIn('git push -u origin "$current_branch"', source)
        self.assertNotIn(
            "git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'",
            source,
        )

    def test_remediation_workflow_uses_focused_tests_before_pr(self) -> None:
        source = (
            ROOT / "ai" / "contribution-workflows" / "repository-remediation.md"
        ).read_text()
        self.assertIn("bash scripts/check-pr-fast.sh", source)
        self.assertIn("--test 'cargo test -p tenferro-tensor", source)
        self.assertNotIn("--ci-profile local-gate", source)
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

    def test_policy_assigns_comprehensive_validation_to_hosted_ci(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text()
        contributing = (ROOT / "CONTRIBUTING.md").read_text()
        old_design = (
            ROOT
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-07-17-local-pr-gate-design.md"
        ).read_text()

        for source in (agents, contributing):
            self.assertIn("incremental=true", source)
            self.assertIn("documentation-only", source)
            self.assertIn("focused", source)
            self.assertIn("Hosted CI", source)
            self.assertNotIn("--ci-profile local-gate", source)

        self.assertIn("Superseded", old_design)
        self.assertIn("2026-07-17-lightweight-local-pr-gate-design.md", old_design)


if __name__ == "__main__":
    unittest.main()
