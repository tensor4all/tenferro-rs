#!/usr/bin/env python3
"""Contract tests for the Phase 2E provenance-bound build orchestrator."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def old_root_cargo() -> bytes:
    lines = []
    for dependency in build.STRIDED_DEPENDENCIES:
        lines.append(
            f'{dependency} = {{ git = "https://example.invalid/strided-rs", '
            f'rev = "{build.OLD_STRIDED}" }}\n'
        )
    return "".join(lines).encode()


def source_fixtures():
    baseline = {
        build.ROOT_CARGO_PATH: old_root_cargo(),
        build.AD_CARGO_PATH: b"[package]\nname = \"tenferro-ad\"\n",
    }
    harness = {
        build.AD_CARGO_PATH: baseline[build.AD_CARGO_PATH]
        + build.BENCH_STANZA.encode(),
        build.BENCH_SOURCE_PATH: b"fn main() {}\n",
    }
    direct = {
        build.AD_CARGO_PATH: harness[build.AD_CARGO_PATH],
        build.BENCH_SOURCE_PATH: harness[build.BENCH_SOURCE_PATH],
    }
    normalized = dict(direct)
    normalized[build.ROOT_CARGO_PATH] = old_root_cargo().replace(
        build.OLD_STRIDED.encode(), build.COMMON_STRIDED.encode()
    )
    return baseline, harness, direct, normalized


def write_fake_tool(directory: pathlib.Path, name: str, payload: bytes | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload or f"{name}-tool\n".encode())
    path.chmod(0o755)
    return path


def controlled_tool_path(root: pathlib.Path) -> str:
    git_directory = root / "git-bin"
    rust_directory = root / "rust-bin"
    write_fake_tool(git_directory, "git")
    write_fake_tool(rust_directory, "cargo")
    write_fake_tool(rust_directory, "rustc")
    return os.pathsep.join((str(git_directory), str(rust_directory)))


def system_tool_path() -> str:
    directories = []
    for name in ("git", "cargo", "rustc"):
        if name == "git":
            discovered = shutil.which(name)
            if discovered is None:
                raise unittest.SkipTest("git is required for the real-Git test")
        else:
            rustup = shutil.which("rustup")
            if rustup is None:
                raise unittest.SkipTest("rustup is required for the real-Git test")
            completed = subprocess.run(
                [rustup, "which", name],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise unittest.SkipTest(f"rustup cannot resolve {name}")
            discovered = completed.stdout.strip()
        directory = pathlib.Path(discovered).resolve(strict=True).parent
        if directory not in directories:
            directories.append(directory)
    return os.pathsep.join(map(str, directories))


def write_probe_fixture(repository: pathlib.Path) -> pathlib.Path:
    source = repository / "scripts/phase2e/allocation-probe"
    (source / "src").mkdir(parents=True)
    (source / "Cargo.toml.in").write_text(
        """[package]
name = "phase2e-allocation-probe"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
tenferro-ad = { path = "__TENFERRO_REPOSITORY_ROOT__/crates/tenferro-ad", default-features = false, features = ["cpu-faer"] }
tenferro-cpu = { path = "__TENFERRO_REPOSITORY_ROOT__/crates/tenferro-cpu", default-features = false, features = ["cpu-faer"] }
tenferro-tensor = { path = "__TENFERRO_REPOSITORY_ROOT__/crates/tenferro-tensor", default-features = false }
"""
    )
    (source / "src/main.rs").write_text("fn main() {}\n")
    (source / "src/tests.rs").write_text("#[test]\nfn probe() {}\n")
    for name in ("tenferro-ad", "tenferro-cpu", "tenferro-tensor"):
        (repository / "crates" / name).mkdir(parents=True)
    return source


class FakeProbeRunner:
    def __init__(
        self,
        *,
        fail_at: str | None = None,
        interrupt: BaseException | None = None,
        failure_reason: str = "nonzero-exit",
    ):
        self.fail_at = fail_at
        self.interrupt = interrupt
        self.failure_reason = failure_reason
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def __call__(self, argv, **kwargs):
        argv = tuple(argv)
        self.calls.append((argv, dict(kwargs)))
        if "fmt" in argv:
            name = "fmt"
        elif "test" in argv:
            name = "test"
        elif "clippy" in argv:
            name = "clippy"
        elif "build" in argv:
            name = "build"
        else:
            name = "list-cases"
        if self.interrupt is not None and name == self.fail_at:
            raise self.interrupt
        cwd = pathlib.Path(kwargs["cwd"])
        environment = kwargs["environment"]
        if name == "test":
            (cwd / "Cargo.lock").write_bytes(b"frozen-lock\n")
        elif name == "build":
            binary = pathlib.Path(environment["CARGO_TARGET_DIR"]) / "release" / build.ALLOCATION_PROBE_BINARY
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"probe-binary")
            binary.chmod(0o755)
        stdout = ""
        if name == "list-cases":
            stdout = json.dumps(list(protocol.CANONICAL_CASES), separators=(",", ":")) + "\n"
        returncode = 9 if name == self.fail_at else 0
        return build.CommandResult(
            argv=argv,
            cwd=str(cwd),
            environment=dict(environment),
            deadline_seconds=kwargs["deadline_seconds"],
            returncode=returncode,
            stdout=stdout,
            stderr="failed" if returncode else "",
            validity_state="COMPLETE" if returncode == 0 else "INCONCLUSIVE",
            failure_reason=None if returncode == 0 else self.failure_reason,
            terminated=returncode != 0 and self.failure_reason == "deadline-exceeded",
            killed=returncode != 0 and self.failure_reason == "deadline-exceeded",
        )


class IdentityAndDeltaTests(unittest.TestCase):
    def test_immutable_identities_commands_and_field_axes(self) -> None:
        self.assertEqual(
            build.IMPLEMENTATION_BASELINE,
            "85855e272b1495611deb601a9ee06f3546772c3c",
        )
        self.assertEqual(
            build.HARNESS_COMMIT, "4471d6145c4d8793de3a96f8d99400c24ca8c6d1"
        )
        self.assertEqual(
            build.OLD_STRIDED, "10fc972d3c0f8cdfd4ecb45d21d815aebfd7d1f2"
        )
        self.assertEqual(
            build.COMMON_STRIDED, "6b0b4a46b7dd9a9ea1677a0d596c0b4adab1acbc"
        )
        self.assertEqual(
            build.BENCH_COMMAND,
            (
                "cargo",
                "bench",
                "--locked",
                "--no-run",
                "-p",
                "tenferro-ad",
                "--bench",
                "eager_dispatch_baseline",
                "--no-default-features",
                "--features",
                "cpu-faer",
            ),
        )
        self.assertEqual(
            build.INVARIANT_FIELDS,
            frozenset(
                {
                    "protocol_version",
                    "toolchain",
                    "target",
                    "profile",
                    "requested_features",
                    "provider",
                    "benchmark_sha256",
                    "benchmark_stanza_sha256",
                    "command_template",
                    "config_chain_sha256",
                }
            ),
        )
        self.assertEqual(
            build.ROLE_FIELDS,
            frozenset(
                {
                    "role",
                    "head",
                    "tracked_tree_sha256",
                    "resolved_features_sha256",
                    "lock_sha256",
                    "worktree",
                    "target_dir",
                    "executable",
                    "executable_sha256",
                }
            ),
        )

    def test_direct_delta_is_exact_frozen_harness_only(self) -> None:
        baseline, harness, direct, _normalized = source_fixtures()
        build.validate_source_delta(
            "direct-current-main-baseline", baseline, harness, direct
        )
        extra = dict(direct)
        extra[pathlib.Path("README.md")] = b"unexpected\n"
        with self.assertRaises(protocol.ProtocolError):
            build.validate_source_delta(
                "direct-current-main-baseline", baseline, harness, extra
            )
        changed_hunk = dict(direct)
        changed_hunk[build.AD_CARGO_PATH] += b"# extra hunk\n"
        with self.assertRaises(protocol.ProtocolError):
            build.validate_source_delta(
                "direct-current-main-baseline", baseline, harness, changed_hunk
            )

    def test_normalized_delta_adds_exactly_five_strided_pins(self) -> None:
        baseline, harness, _direct, normalized = source_fixtures()
        build.validate_source_delta(
            "common-lock-normalized-baseline", baseline, harness, normalized
        )
        four_pins = dict(normalized)
        four_pins[build.ROOT_CARGO_PATH] = four_pins[build.ROOT_CARGO_PATH].replace(
            build.COMMON_STRIDED.encode(), build.OLD_STRIDED.encode(), 1
        )
        with self.assertRaises(protocol.ProtocolError):
            build.validate_source_delta(
                "common-lock-normalized-baseline", baseline, harness, four_pins
            )
        with self.assertRaises(protocol.ProtocolError):
            build.normalized_root_cargo(old_root_cargo() + old_root_cargo())

    def test_git_source_control_materializes_and_revalidates_both_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repository"
            repository.mkdir()

            def git(*arguments):
                completed = subprocess.run(
                    ["git", *arguments],
                    cwd=repository,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                return completed.stdout.strip()

            git("init", "--quiet")
            git("config", "user.name", "Phase 2E Test")
            git("config", "user.email", "phase2e@example.invalid")
            (repository / ".gitignore").write_text("Cargo.lock\n")
            (repository / build.ROOT_CARGO_PATH).write_bytes(old_root_cargo())
            ad_cargo = repository / build.AD_CARGO_PATH
            ad_cargo.parent.mkdir(parents=True)
            ad_cargo.write_bytes(b"[package]\nname = \"tenferro-ad\"\n")
            git("add", ".")
            git("commit", "--quiet", "-m", "baseline")
            baseline_commit = git("rev-parse", "HEAD")

            ad_cargo.write_bytes(ad_cargo.read_bytes() + build.BENCH_STANZA.encode())
            benchmark = repository / build.BENCH_SOURCE_PATH
            benchmark.parent.mkdir(parents=True)
            benchmark.write_bytes(b"fn main() {}\n")
            git("add", ".")
            git("commit", "--quiet", "-m", "harness")
            harness_commit = git("rev-parse", "HEAD")

            home = root / "home"
            home.mkdir()
            source = build.GitSourceControl(
                repository,
                path=system_tool_path(),
                home=home,
                implementation_baseline=baseline_commit,
                harness_commit=harness_commit,
            )
            owned_lock = root / "owned.Cargo.lock"
            owned_lock.write_bytes(b"owned lock\n")
            for role in (
                "direct-current-main-baseline",
                "common-lock-normalized-baseline",
            ):
                worktree = root / role
                build.prepare_fresh_worktree_destination(worktree)
                spec = build.WorktreeSpec(role, worktree, baseline_commit)
                source.create_worktree(spec)
                measurement_commit = source.materialize_baseline(spec)
                build.install_root_owned_lock(owned_lock, worktree)
                proof = source.validate_worktree(
                    worktree, measurement_commit, owned_lock
                )
                self.assertEqual(proof.head, measurement_commit)
                self.assertEqual(
                    proof.benchmark_sha256,
                    sha256_bytes(b"fn main() {}\n"),
                )
                root_cargo = (worktree / build.ROOT_CARGO_PATH).read_bytes()
                if role == "direct-current-main-baseline":
                    self.assertEqual(root_cargo, old_root_cargo())
                else:
                    self.assertNotIn(build.OLD_STRIDED.encode(), root_cargo)
                    self.assertEqual(
                        root_cargo.count(build.COMMON_STRIDED.encode()), 5
                    )

    def test_real_git_persisted_validation_rejects_every_source_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            git_path = pathlib.Path(shutil.which("git") or "").resolve(strict=True)

            def git(cwd, *arguments):
                completed = subprocess.run(
                    [str(git_path), *arguments],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                return completed.stdout.strip()

            git(repository, "init", "--quiet")
            git(repository, "config", "user.name", "Phase 2E Test")
            git(repository, "config", "user.email", "phase2e@example.invalid")
            (repository / ".gitignore").write_text("Cargo.lock\nignored.tmp\n")
            (repository / build.ROOT_CARGO_PATH).write_bytes(old_root_cargo())
            ad_cargo = repository / build.AD_CARGO_PATH
            ad_cargo.parent.mkdir(parents=True)
            ad_cargo.write_bytes(b"[package]\nname = \"tenferro-ad\"\n")
            git(repository, "add", ".")
            git(repository, "commit", "--quiet", "-m", "baseline")
            baseline_commit = git(repository, "rev-parse", "HEAD")

            ad_cargo.write_bytes(ad_cargo.read_bytes() + build.BENCH_STANZA.encode())
            benchmark = repository / build.BENCH_SOURCE_PATH
            benchmark.parent.mkdir(parents=True)
            benchmark.write_bytes(b"fn main() {}\n")
            git(repository, "add", ".")
            git(repository, "commit", "--quiet", "-m", "harness")
            harness_commit = git(repository, "rev-parse", "HEAD")

            (repository / build.ROOT_CARGO_PATH).write_bytes(
                build.normalized_root_cargo(old_root_cargo())
            )
            git(repository, "add", str(build.ROOT_CARGO_PATH))
            git(repository, "commit", "--quiet", "-m", "candidate")
            candidate_commit = git(repository, "rev-parse", "HEAD")

            evidence = repository / "evidence"
            evidence.mkdir()
            home = root / "home"
            cargo_home = root / "cargo-home"
            home.mkdir()
            cargo_home.mkdir()
            (cargo_home / "registry").mkdir()
            (cargo_home / "git").mkdir()
            config = build.BuildConfig(
                repository=repository,
                evidence_root=evidence,
                scratch_root=root / "scratch",
                candidate_commit=candidate_commit,
                path=system_tool_path(),
                home=home,
                cargo_home=cargo_home,
            )

            class LocalGitSource(build.GitSourceControl):
                def create_worktree(self, spec):
                    start = (
                        candidate_commit
                        if spec.role == "candidate"
                        else baseline_commit
                    )
                    super().create_worktree(
                        build.WorktreeSpec(spec.role, spec.path, start)
                    )

                def materialize_baseline(self, spec):
                    return super().materialize_baseline(
                        build.WorktreeSpec(spec.role, spec.path, baseline_commit)
                    )

            source = LocalGitSource(
                repository,
                path=config.path,
                home=home,
                implementation_baseline=baseline_commit,
                harness_commit=harness_commit,
            )
            result = build._build_all_with_dependencies(
                config,
                source_control=source,
                command_runner=FakeCargoRunner(),
            )
            self.assertEqual(result.validity_state, "COMPLETE")
            build._validate_build_set_with_source_control(
                config, source, command_runner=FakeCargoRunner()
            )

            candidate = config.scratch_root / "candidate"
            candidate_bench = candidate / build.BENCH_SOURCE_PATH
            original_bench = candidate_bench.read_bytes()
            candidate_bench.write_bytes(original_bench + b"// dirty\n")
            with self.subTest("modified worktree"):
                with self.assertRaises(protocol.ProtocolError):
                    build._validate_build_set_with_source_control(
                        config, source, command_runner=FakeCargoRunner()
                    )
            candidate_bench.write_bytes(original_bench)

            moved_candidate = root / "moved-candidate"
            candidate.rename(moved_candidate)
            with self.subTest("deleted worktree"):
                with self.assertRaises(protocol.ProtocolError):
                    build._validate_build_set_with_source_control(
                        config, source, command_runner=FakeCargoRunner()
                    )
            moved_candidate.rename(candidate)

            ignored = candidate / "ignored.tmp"
            ignored.write_text("forbidden ignored file\n")
            with self.subTest("extra ignored file"):
                with self.assertRaises(protocol.ProtocolError):
                    build._validate_build_set_with_source_control(
                        config, source, command_runner=FakeCargoRunner()
                    )
            ignored.unlink()

            cargo_config = candidate / ".cargo/config.toml"
            cargo_config.parent.mkdir()
            cargo_config.write_text("[build]\ntarget = \"forged-target\"\n")
            with self.subTest("config change"):
                with self.assertRaises(protocol.ProtocolError):
                    build._validate_build_set_with_source_control(
                        config, source, command_runner=FakeCargoRunner()
                    )
            cargo_config.unlink()
            cargo_config.parent.rmdir()

            candidate_manifest_path = (
                evidence / build.BUILD_MANIFEST_PATHS["candidate"]
            )
            original_candidate_manifest = candidate_manifest_path.read_bytes()
            candidate_manifest = json.loads(original_candidate_manifest)
            candidate_manifest["tracked_tree_sha256"] = "f" * 64
            candidate_manifest_path.chmod(0o644)
            candidate_manifest_path.write_text(json.dumps(candidate_manifest))
            candidate_manifest_path.chmod(0o444)
            with self.subTest("forged tracked hash"):
                with self.assertRaises(protocol.ProtocolError):
                    build._validate_build_set_with_source_control(
                        config, source, command_runner=FakeCargoRunner()
                    )
            candidate_manifest_path.chmod(0o644)
            candidate_manifest_path.write_bytes(original_candidate_manifest)
            candidate_manifest_path.chmod(0o444)

            wrong_candidate = build.BuildConfig(
                repository=config.repository,
                evidence_root=config.evidence_root,
                scratch_root=config.scratch_root,
                candidate_commit="f" * 40,
                path=config.path,
                home=config.home,
                cargo_home=config.cargo_home,
            )
            with self.subTest("candidate mismatch"):
                with self.assertRaises(protocol.ProtocolError):
                    build._validate_build_set_with_source_control(
                        wrong_candidate,
                        source,
                        command_runner=FakeCargoRunner(),
                    )

            direct_role = "direct-current-main-baseline"
            direct = config.scratch_root / direct_role
            direct_manifest_path = evidence / build.BUILD_MANIFEST_PATHS[direct_role]
            original_direct_manifest = direct_manifest_path.read_bytes()
            direct_manifest = json.loads(original_direct_manifest)
            unexpected = direct / "unexpected.txt"
            unexpected.write_text("wrong baseline delta\n")
            git(direct, "add", "unexpected.txt")
            git(direct, "commit", "--quiet", "-m", "wrong delta")
            bad_head = git(direct, "rev-parse", "HEAD")
            direct_manifest["head"] = bad_head
            direct_manifest_path.chmod(0o644)
            direct_manifest_path.write_text(json.dumps(direct_manifest))
            direct_manifest_path.chmod(0o444)
            with self.subTest("wrong baseline delta"):
                with self.assertRaises(protocol.ProtocolError):
                    build._validate_build_set_with_source_control(
                        config, source, command_runner=FakeCargoRunner()
                    )
            git(direct, "checkout", "--quiet", "--detach", result.manifests[direct_role]["head"])
            direct_manifest_path.chmod(0o644)
            direct_manifest_path.write_bytes(original_direct_manifest)
            direct_manifest_path.chmod(0o444)


class LockWorktreeAndInventoryTests(unittest.TestCase):
    def test_four_root_owned_lock_copies_are_exact_and_collision_safe(self) -> None:
        self.assertEqual(
            build.LOCK_PATHS,
            {
                "direct": pathlib.Path(
                    "builds/locks/direct-current-main.Cargo.lock"
                ),
                "common": pathlib.Path("builds/locks/common.Cargo.lock"),
                "direct-probe": pathlib.Path(
                    "builds/locks/direct-current-main-probe.Cargo.lock"
                ),
                "common-probe": pathlib.Path(
                    "builds/locks/common-probe.Cargo.lock"
                ),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "evidence"
            root.mkdir()
            for index, lock_name in enumerate(build.LOCK_PATHS):
                source = pathlib.Path(temporary) / f"source-{index}.lock"
                expected = f"lock-{lock_name}\n".encode()
                source.write_bytes(expected)
                destination = build.copy_root_owned_lock(root, lock_name, source)
                self.assertEqual(destination, root / build.LOCK_PATHS[lock_name])
                self.assertEqual(destination.read_bytes(), expected)
                with self.assertRaises(protocol.ProtocolError):
                    build.copy_root_owned_lock(root, lock_name, source)

    def test_root_owned_lock_is_installed_byte_for_byte_in_a_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "evidence"
            worktree = root / "worktree"
            evidence.mkdir()
            worktree.mkdir()
            source = root / "generated.lock"
            source.write_bytes(b"root-owned lock\n")
            owned = build.copy_root_owned_lock(evidence, "common", source)
            (worktree / "Cargo.lock").write_bytes(b"generated input\n")
            installed = build.install_root_owned_lock(owned, worktree)
            self.assertEqual(installed, worktree / "Cargo.lock")
            self.assertEqual(installed.read_bytes(), owned.read_bytes())
            self.assertEqual(protocol.sha256_file(installed), protocol.sha256_file(owned))

    def test_worktree_specs_are_dedicated_and_destinations_must_be_fresh(self) -> None:
        candidate = "c" * 40
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            specs = build.worktree_specs(scratch, candidate)
            self.assertEqual(
                [spec.role for spec in specs],
                [
                    "direct-current-main-baseline",
                    "common-lock-normalized-baseline",
                    "candidate",
                ],
            )
            self.assertEqual(
                [spec.start_commit for spec in specs],
                [build.IMPLEMENTATION_BASELINE, build.IMPLEMENTATION_BASELINE, candidate],
            )
            self.assertEqual(len({spec.path for spec in specs}), 3)
            for spec in specs:
                build.prepare_fresh_worktree_destination(spec.path)
                self.assertTrue(spec.path.is_dir())
                (spec.path / "occupied").write_text("x")
                with self.assertRaises(protocol.ProtocolError):
                    build.prepare_fresh_worktree_destination(spec.path)

    def test_filesystem_inventory_allows_only_tracked_paths_git_and_root_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / ".git").write_text("gitdir: elsewhere\n")
            tracked = pathlib.Path("src/lib.rs")
            (root / tracked.parent).mkdir()
            (root / tracked).write_text("tracked\n")
            (root / "Cargo.lock").write_bytes(b"lock")
            build.validate_filesystem_inventory(
                root, {pathlib.Path("Cargo.lock")}, tracked_paths={tracked}
            )
            (root / "target").mkdir()
            with self.assertRaises(protocol.ProtocolError):
                build.validate_filesystem_inventory(
                    root, {pathlib.Path("Cargo.lock")}, tracked_paths={tracked}
                )

    def test_ignored_inventory_is_an_exact_root_lock_allowlist(self) -> None:
        build.validate_ignored_inventory(
            {pathlib.Path("Cargo.lock")}, {pathlib.Path("Cargo.lock")}
        )
        with self.assertRaises(protocol.ProtocolError):
            build.validate_ignored_inventory(
                {pathlib.Path("Cargo.lock"), pathlib.Path("target/cache")},
                {pathlib.Path("Cargo.lock")},
            )


class CargoEnvironmentAndCommandTests(unittest.TestCase):
    def test_allocation_probe_command_plan_is_exact_and_bounded(self) -> None:
        manifest = pathlib.Path("/tmp/generated/Cargo.toml")
        binary = pathlib.Path("/tmp/target/bench/phase2e-allocation-probe")
        plan = build.allocation_probe_command_plan(manifest, binary, "/bin/cargo")
        self.assertEqual(
            [(step.name, step.deadline_seconds) for step in plan],
            [
                ("fmt", 300),
                ("test", 1800),
                ("clippy", 1800),
                ("build", 1800),
                ("list-cases", 30),
            ],
        )
        self.assertEqual(
            [step.argv for step in plan],
            [
                ("/bin/cargo", "fmt", "--manifest-path", str(manifest), "--", "--check"),
                ("/bin/cargo", "test", "--manifest-path", str(manifest)),
                (
                    "/bin/cargo",
                    "clippy",
                    "--manifest-path",
                    str(manifest),
                    "--locked",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                ),
                (
                    "/bin/cargo",
                    "build",
                    "--locked",
                    "--profile",
                    "bench",
                    "--manifest-path",
                    str(manifest),
                ),
                (str(binary), "--list-cases"),
            ],
        )

    def test_cargo_environment_is_controlled_and_drops_ambient_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controlled = pathlib.Path(temporary) / "controlled-bin"
            controlled.mkdir()
            env = protocol.cargo_environment(
                path=str(controlled),
                home="/tmp/phase2e-home",
                cargo_home="/tmp/phase2e-cargo-home",
                target_dir="/tmp/phase2e-target",
            )
            self.assertEqual(
                set(env),
                {
                    "PATH",
                    "HOME",
                    "LC_ALL",
                    "TZ",
                    *protocol.THREAD_ENV,
                    "CARGO_HOME",
                    "CARGO_TARGET_DIR",
                    "CARGO_INCREMENTAL",
                    "CARGO_NET_OFFLINE",
                },
            )
            self.assertEqual(env["CARGO_INCREMENTAL"], "0")
            self.assertEqual(env["CARGO_NET_OFFLINE"], "true")
            for forbidden in (
                "LD_PRELOAD",
                "RUSTFLAGS",
                "RUSTC_WRAPPER",
                "CARGO_PROFILE_BENCH_LTO",
                "CARGO_BUILD_TARGET",
            ):
                self.assertNotIn(forbidden, env)
            with self.assertRaises(protocol.ProtocolError):
                protocol.cargo_environment(
                    path="relative/bin",
                    home="/tmp/home",
                    cargo_home="/tmp/cargo",
                    target_dir="/tmp/target",
                )
            with self.assertRaises(protocol.ProtocolError):
                protocol.cargo_environment(
                    path=os.pathsep.join((str(controlled), "relative/bin")),
                    home="/tmp/home",
                    cargo_home="/tmp/cargo",
                    target_dir="/tmp/target",
                )

    def test_controlled_cargo_home_rejects_config_and_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cargo_home = pathlib.Path(temporary)
            (cargo_home / "registry").mkdir()
            (cargo_home / "git").mkdir()
            build.validate_controlled_cargo_home(cargo_home)
            (cargo_home / "config.toml").write_text("[net]\noffline = false\n")
            with self.assertRaises(protocol.ProtocolError):
                build.validate_controlled_cargo_home(cargo_home)

    def test_feature_query_and_deadlines_match_the_actual_build(self) -> None:
        target = "x86_64-unknown-linux-gnu"
        expected_tree = (
            "cargo",
            "tree",
            "--locked",
            "--target",
            target,
            "-p",
            "tenferro-ad",
            "--no-default-features",
            "--features",
            "cpu-faer",
            "-e",
            "features",
        )
        self.assertEqual(build.REQUESTED_FEATURES, ("cpu-faer",))
        self.assertEqual(build.timing_feature_command(target), expected_tree)
        with tempfile.TemporaryDirectory() as temporary:
            tools = build.resolve_toolchain(
                controlled_tool_path(pathlib.Path(temporary))
            )
            plan = build.build_command_plan(target, tools.cargo)
            self.assertEqual(
                [(step.name, step.deadline_seconds) for step in plan],
                [("metadata", 300), ("features", 300), ("build", 1800)],
            )
            self.assertEqual(plan[1].argv[0], str(tools.cargo.path))
            self.assertEqual(plan[1].argv[1:], expected_tree[1:])
            self.assertEqual(plan[2].argv[0], str(tools.cargo.path))
            self.assertEqual(plan[2].argv[1:], build.BENCH_COMMAND[1:])
        with self.assertRaises(protocol.ProtocolError):
            build.validate_feature_query(
                ("cargo", "tree", "-e", "features"),
                target=target,
                package="tenferro-ad",
                requested_features=build.REQUESTED_FEATURES,
                no_default_features=True,
            )


class AllocationProbeVerifierTests(unittest.TestCase):
    def fixture(self, root: pathlib.Path):
        repository = root / "repository"
        repository.mkdir()
        write_probe_fixture(repository)
        cargo = write_fake_tool(root / "tools", "cargo")
        cache = root / "cache"
        (cache / "registry").mkdir(parents=True)
        (cache / "git").mkdir()
        owned = root / "owned"

        def make_root():
            owned.mkdir()
            return owned

        identity = build.ResolvedTool("cargo", cargo, protocol.sha256_file(cargo))
        return repository, identity, cache, owned, make_root

    def test_verifier_runs_exact_five_steps_with_sealed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)
            runner = FakeProbeRunner()
            result = build._verify_allocation_probe_with_dependencies(
                repository,
                cargo=cargo,
                command_runner=runner,
                temporary_root_factory=make_root,
                cache_source=cache,
            )
            self.assertEqual(
                [step[0][1] for step in runner.calls[:4]],
                ["fmt", "test", "clippy", "build"],
            )
            self.assertEqual(runner.calls[4][0][1:], ("--list-cases",))
            self.assertEqual(
                [call[1]["deadline_seconds"] for call in runner.calls],
                [300, 1800, 1800, 1800, 30],
            )
            for argv, kwargs in runner.calls[:4]:
                self.assertEqual(argv[0], str(cargo.path))
                self.assertTrue(pathlib.Path(argv[0]).is_absolute())
                self.assertEqual(kwargs["executable_identity"], cargo)
                self.assertEqual(
                    set(kwargs["environment"]),
                    {
                        "PATH",
                        "HOME",
                        "LC_ALL",
                        "TZ",
                        *protocol.THREAD_ENV,
                        "CARGO_HOME",
                        "CARGO_TARGET_DIR",
                        "CARGO_INCREMENTAL",
                        "CARGO_NET_OFFLINE",
                    },
                )
            self.assertIsNone(runner.calls[4][1]["executable_identity"])
            self.assertEqual(result.case_inventory, tuple(protocol.CANONICAL_CASES))
            self.assertEqual(result.source_sha256, result.generated_source_sha256)
            self.assertEqual(
                result.template_sha256,
                protocol.sha256_file(
                    repository
                    / build.ALLOCATION_PROBE_SOURCE_ROOT
                    / build.ALLOCATION_PROBE_TEMPLATE
                ),
            )
            self.assertRegex(result.lock_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(result.binary_sha256, r"^[0-9a-f]{64}$")
            self.assertFalse(owned.exists())

    def test_each_noncomplete_step_stops_immediately_and_cleans(self) -> None:
        for reason in ("nonzero-exit", "deadline-exceeded"):
            for failing_index, name in enumerate(
                ("fmt", "test", "clippy", "build", "list-cases")
            ):
                with self.subTest(name=name, reason=reason), tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    repository, cargo, cache, owned, make_root = self.fixture(root)
                    runner = FakeProbeRunner(fail_at=name, failure_reason=reason)
                    with self.assertRaisesRegex(protocol.ProtocolError, name):
                        build._verify_allocation_probe_with_dependencies(
                            repository,
                            cargo=cargo,
                            command_runner=runner,
                            temporary_root_factory=make_root,
                            cache_source=cache,
                        )
                    self.assertEqual(len(runner.calls), failing_index + 1)
                    self.assertFalse(owned.exists())

    def test_control_exceptions_preserve_identity_and_cleanup(self) -> None:
        for exception in (KeyboardInterrupt("cancel"), SystemExit(17)):
            with self.subTest(exception=type(exception).__name__), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                repository, cargo, cache, owned, make_root = self.fixture(root)
                runner = FakeProbeRunner(fail_at="clippy", interrupt=exception)
                with self.assertRaises(type(exception)) as caught:
                    build._verify_allocation_probe_with_dependencies(
                        repository,
                        cargo=cargo,
                        command_runner=runner,
                        temporary_root_factory=make_root,
                        cache_source=cache,
                    )
                self.assertIs(caught.exception, exception)
                self.assertFalse(owned.exists())

    def test_cleanup_failure_never_replaces_active_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)
            interruption = KeyboardInterrupt("primary")
            runner = FakeProbeRunner(fail_at="fmt", interrupt=interruption)
            with mock.patch.object(build.shutil, "rmtree", side_effect=OSError("cleanup")):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    build._verify_allocation_probe_with_dependencies(
                        repository,
                        cargo=cargo,
                        command_runner=runner,
                        temporary_root_factory=make_root,
                        cache_source=cache,
                    )
            self.assertIs(caught.exception, interruption)
            self.assertTrue(owned.is_dir())

    def test_cleanup_without_primary_preserves_control_exceptions_raw(self) -> None:
        root = pathlib.Path("/tmp/owned-probe-root")
        for make_interruption in (lambda: KeyboardInterrupt("cleanup"), lambda: SystemExit(23)):
            interruption = make_interruption()
            with self.subTest(kind=type(interruption).__name__):
                def raising_cleanup(_root, error=interruption):
                    raise error

                with mock.patch.object(build.shutil, "rmtree", side_effect=raising_cleanup):
                    try:
                        build._cleanup_probe_root(root, None)
                    except BaseException as caught:
                        self.assertIs(caught, interruption)
                        traceback_names = []
                        traceback = caught.__traceback__
                        while traceback is not None:
                            traceback_names.append(traceback.tb_frame.f_code.co_name)
                            traceback = traceback.tb_next
                    else:
                        self.fail("cleanup control exception was suppressed")
                self.assertIn("raising_cleanup", traceback_names)

    def test_cleanup_without_primary_wraps_ordinary_exception_only(self) -> None:
        root = pathlib.Path("/tmp/owned-probe-root")
        for make_secondary in (lambda: OSError("cleanup"), lambda: RuntimeError("cleanup")):
            secondary = make_secondary()
            with self.subTest(kind=type(secondary).__name__):
                with mock.patch.object(build.shutil, "rmtree", side_effect=secondary):
                    with self.assertRaises(protocol.ProtocolError) as caught:
                        build._cleanup_probe_root(root, None)
                self.assertIs(caught.exception.__cause__, secondary)

    def test_cleanup_secondary_never_replaces_active_primary(self) -> None:
        root = pathlib.Path("/tmp/owned-probe-root")
        for make_primary in (
            lambda: OSError("primary"),
            lambda: RuntimeError("primary"),
            lambda: KeyboardInterrupt("primary"),
            lambda: SystemExit(17),
        ):
            for make_secondary in (
                lambda: OSError("cleanup"),
                lambda: RuntimeError("cleanup"),
                lambda: KeyboardInterrupt("cleanup"),
                lambda: SystemExit(29),
            ):
                primary = make_primary()
                secondary = make_secondary()
                with self.subTest(
                    primary=type(primary).__name__, secondary=type(secondary).__name__
                ):
                    def raise_primary(error=primary):
                        raise error

                    with self.assertRaises(type(primary)) as caught:
                        try:
                            raise_primary()
                        except BaseException as active:
                            with mock.patch.object(
                                build.shutil, "rmtree", side_effect=secondary
                            ):
                                build._cleanup_probe_root(root, active)
                            raise
                    self.assertIs(caught.exception, primary)
                    self.assertTrue(
                        any("cleanup" in note for note in getattr(primary, "__notes__", ()))
                    )

    def test_write_new_regular_preserves_open_write_and_fsync_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            real_open = os.open
            real_write = os.write
            real_fsync = os.fsync
            for stage in ("open", "write", "fsync"):
                for make_primary in (
                    lambda: OSError(f"{stage}-primary"),
                    lambda: RuntimeError(f"{stage}-primary"),
                    lambda: KeyboardInterrupt(f"{stage}-primary"),
                    lambda: SystemExit(31),
                ):
                    primary = make_primary()
                    with self.subTest(stage=stage, primary=type(primary).__name__):
                        path = root / f"{stage}-{type(primary).__name__}"
                        opened: list[int] = []

                        def recording_open(*args, **kwargs):
                            descriptor = real_open(*args, **kwargs)
                            opened.append(descriptor)
                            return descriptor

                        patches = {
                            "open": mock.patch.object(
                                build.os,
                                "open",
                                side_effect=primary if stage == "open" else recording_open,
                            ),
                            "write": mock.patch.object(
                                build.os,
                                "write",
                                side_effect=primary if stage == "write" else real_write,
                            ),
                            "fsync": mock.patch.object(
                                build.os,
                                "fsync",
                                side_effect=primary if stage == "fsync" else real_fsync,
                            ),
                        }
                        with patches["open"], patches["write"], patches["fsync"]:
                            expected = (
                                protocol.ProtocolError
                                if isinstance(primary, Exception)
                                else type(primary)
                            )
                            with self.assertRaises(expected) as caught:
                                build._write_new_regular(path, b"payload")
                        if isinstance(primary, Exception):
                            self.assertIs(caught.exception.__cause__, primary)
                        else:
                            self.assertIs(caught.exception, primary)
                        for descriptor in opened:
                            with self.assertRaises(OSError):
                                os.fstat(descriptor)
                            del descriptor
                        if stage == "open":
                            self.assertFalse(path.exists())
                        elif path.exists():
                            path.unlink()

    def test_write_new_regular_close_exception_matrix_and_fd_reuse(self) -> None:
        real_open = os.open
        real_close = os.close
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for make_secondary in (
                lambda: OSError("close"),
                lambda: RuntimeError("close"),
                lambda: KeyboardInterrupt("close"),
                lambda: SystemExit(37),
            ):
                secondary = make_secondary()
                with self.subTest(secondary=type(secondary).__name__):
                    path = root / f"close-{type(secondary).__name__}"
                    reused: list[int] = []

                    def close_reuse_then_raise(descriptor, error=secondary):
                        real_close(descriptor)
                        reused.append(
                            real_open(
                                root / "reuse", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                            )
                        )
                        self.assertEqual(reused[-1], descriptor)
                        raise error

                    with mock.patch.object(build.os, "close", side_effect=close_reuse_then_raise):
                        if isinstance(secondary, Exception):
                            with self.assertRaises(protocol.ProtocolError) as context:
                                build._write_new_regular(path, b"payload")
                            caught = context.exception
                        else:
                            try:
                                build._write_new_regular(path, b"payload")
                            except BaseException as caught:
                                observed = caught
                                traceback_names = []
                                traceback = caught.__traceback__
                                while traceback is not None:
                                    traceback_names.append(traceback.tb_frame.f_code.co_name)
                                    traceback = traceback.tb_next
                            else:
                                self.fail("close control exception was suppressed")
                            self.assertIn("close_reuse_then_raise", traceback_names)
                    if isinstance(secondary, Exception):
                        self.assertIs(caught.__cause__, secondary)
                    else:
                        self.assertIs(observed, secondary)
                    os.write(reused[-1], b"still-live")
                    real_close(reused[-1])
                    (root / "reuse").unlink()

    def test_write_primary_survives_every_close_secondary_without_double_close(self) -> None:
        real_open = os.open
        real_close = os.close
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for make_primary in (
                lambda: OSError("write"),
                lambda: RuntimeError("write"),
                lambda: KeyboardInterrupt("write"),
                lambda: SystemExit(41),
            ):
                for make_secondary in (
                    lambda: OSError("close"),
                    lambda: RuntimeError("close"),
                    lambda: KeyboardInterrupt("close"),
                    lambda: SystemExit(41),
                ):
                    primary = make_primary()
                    secondary = make_secondary()
                    with self.subTest(
                        primary=type(primary).__name__, secondary=type(secondary).__name__
                    ):
                        path = root / f"primary-{type(primary).__name__}-{type(secondary).__name__}"
                        reused: list[int] = []

                        def close_reuse_then_raise(descriptor, error=secondary):
                            real_close(descriptor)
                            reused.append(
                                real_open(
                                    root / "reuse",
                                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                                    0o600,
                                )
                            )
                            self.assertEqual(reused[-1], descriptor)
                            raise error

                        with mock.patch.object(build.os, "write", side_effect=primary), mock.patch.object(
                            build.os, "close", side_effect=close_reuse_then_raise
                        ):
                            expected = (
                                protocol.ProtocolError
                                if isinstance(primary, Exception)
                                else type(primary)
                            )
                            with self.assertRaises(expected) as caught:
                                build._write_new_regular(path, b"payload")
                        if isinstance(primary, Exception):
                            self.assertIs(caught.exception.__cause__, primary)
                            active = caught.exception
                        else:
                            self.assertIs(caught.exception, primary)
                            active = primary
                        self.assertTrue(
                            any("close" in note for note in getattr(active, "__notes__", ()))
                        )
                        os.write(reused[-1], b"still-live")
                        real_close(reused[-1])
                        (root / "reuse").unlink()
                        path.unlink()

    def test_read_regular_preserves_open_and_read_failures(self) -> None:
        real_open = os.open
        real_read = os.read
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            source.write_bytes(b"payload")
            for stage in ("open", "read"):
                for make_primary in (
                    lambda: OSError(f"{stage}-primary"),
                    lambda: RuntimeError(f"{stage}-primary"),
                    lambda: KeyboardInterrupt(f"{stage}-primary"),
                    lambda: SystemExit(43),
                ):
                    primary = make_primary()
                    with self.subTest(stage=stage, primary=type(primary).__name__):
                        with mock.patch.object(
                            build.os,
                            "open",
                            side_effect=primary if stage == "open" else real_open,
                        ), mock.patch.object(
                            build.os,
                            "read",
                            side_effect=primary if stage == "read" else real_read,
                        ):
                            expected = (
                                protocol.ProtocolError
                                if isinstance(primary, Exception)
                                else type(primary)
                            )
                            with self.assertRaises(expected) as caught:
                                build._read_regular_bytes(source)
                        if isinstance(primary, Exception):
                            self.assertIs(caught.exception.__cause__, primary)
                        else:
                            self.assertIs(caught.exception, primary)

    def test_read_regular_close_exception_matrix_and_fd_reuse(self) -> None:
        real_open = os.open
        real_close = os.close
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            source.write_bytes(b"payload")
            for make_secondary in (
                lambda: OSError("close"),
                lambda: RuntimeError("close"),
                lambda: KeyboardInterrupt("close"),
                lambda: SystemExit(47),
            ):
                secondary = make_secondary()
                with self.subTest(secondary=type(secondary).__name__):
                    reused: list[int] = []

                    def close_reuse_then_raise(descriptor, error=secondary):
                        real_close(descriptor)
                        reused.append(real_open(source, os.O_RDONLY | os.O_CLOEXEC))
                        self.assertEqual(reused[-1], descriptor)
                        raise error

                    with mock.patch.object(build.os, "close", side_effect=close_reuse_then_raise):
                        if isinstance(secondary, Exception):
                            with self.assertRaises(protocol.ProtocolError) as context:
                                build._read_regular_bytes(source)
                            caught = context.exception
                        else:
                            try:
                                build._read_regular_bytes(source)
                            except BaseException as caught:
                                observed = caught
                                traceback_names = []
                                traceback = caught.__traceback__
                                while traceback is not None:
                                    traceback_names.append(traceback.tb_frame.f_code.co_name)
                                    traceback = traceback.tb_next
                            else:
                                self.fail("close control exception was suppressed")
                            self.assertIn("close_reuse_then_raise", traceback_names)
                    if isinstance(secondary, Exception):
                        self.assertIs(caught.__cause__, secondary)
                    else:
                        self.assertIs(observed, secondary)
                    self.assertEqual(os.read(reused[-1], 1), b"p")
                    real_close(reused[-1])

    def test_read_primary_survives_every_close_secondary_without_double_close(self) -> None:
        real_open = os.open
        real_close = os.close
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            source.write_bytes(b"payload")
            for make_primary in (
                lambda: OSError("read"),
                lambda: RuntimeError("read"),
                lambda: KeyboardInterrupt("read"),
                lambda: SystemExit(53),
            ):
                for make_secondary in (
                    lambda: OSError("close"),
                    lambda: RuntimeError("close"),
                    lambda: KeyboardInterrupt("close"),
                    lambda: SystemExit(59),
                ):
                    primary = make_primary()
                    secondary = make_secondary()
                    with self.subTest(
                        primary=type(primary).__name__, secondary=type(secondary).__name__
                    ):
                        reused: list[int] = []

                        def close_reuse_then_raise(descriptor, error=secondary):
                            real_close(descriptor)
                            reused.append(real_open(source, os.O_RDONLY | os.O_CLOEXEC))
                            self.assertEqual(reused[-1], descriptor)
                            raise error

                        with mock.patch.object(build.os, "read", side_effect=primary), mock.patch.object(
                            build.os, "close", side_effect=close_reuse_then_raise
                        ):
                            expected = (
                                protocol.ProtocolError
                                if isinstance(primary, Exception)
                                else type(primary)
                            )
                            with self.assertRaises(expected) as caught:
                                build._read_regular_bytes(source)
                        if isinstance(primary, Exception):
                            self.assertIs(caught.exception.__cause__, primary)
                            active = caught.exception
                        else:
                            self.assertIs(caught.exception, primary)
                            active = primary
                        self.assertTrue(
                            any("close" in note for note in getattr(active, "__notes__", ()))
                        )
                        self.assertEqual(os.read(reused[-1], 1), b"p")
                        real_close(reused[-1])

    def test_source_mutation_and_foreign_inventory_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)
            source = repository / build.ALLOCATION_PROBE_SOURCE_ROOT / "src/main.rs"

            class MutatingRunner(FakeProbeRunner):
                def __call__(self, argv, **kwargs):
                    result = super().__call__(argv, **kwargs)
                    source.write_text("fn changed() {}\n")
                    return result

            with self.assertRaisesRegex(protocol.ProtocolError, "source.*changed"):
                build._verify_allocation_probe_with_dependencies(
                    repository,
                    cargo=cargo,
                    command_runner=MutatingRunner(),
                    temporary_root_factory=make_root,
                    cache_source=cache,
                )
            self.assertFalse(owned.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)
            (repository / build.ALLOCATION_PROBE_SOURCE_ROOT / "extra").write_text("x")
            runner = FakeProbeRunner()
            with self.assertRaisesRegex(protocol.ProtocolError, "inventory"):
                build._verify_allocation_probe_with_dependencies(
                    repository,
                    cargo=cargo,
                    command_runner=runner,
                    temporary_root_factory=make_root,
                    cache_source=cache,
                )
            self.assertEqual(runner.calls, [])
            self.assertFalse(owned.exists())

    def test_template_generated_source_and_symlink_mutations_are_rejected(self) -> None:
        for relative in ("Cargo.toml.in", "src/main.rs"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                repository, cargo, cache, owned, make_root = self.fixture(root)
                tracked = repository / build.ALLOCATION_PROBE_SOURCE_ROOT / relative

                class TrackedMutatingRunner(FakeProbeRunner):
                    def __call__(self, argv, **kwargs):
                        result = super().__call__(argv, **kwargs)
                        tracked.write_bytes(tracked.read_bytes() + b"\n")
                        return result

                with self.assertRaisesRegex(protocol.ProtocolError, "source changed"):
                    build._verify_allocation_probe_with_dependencies(
                        repository,
                        cargo=cargo,
                        command_runner=TrackedMutatingRunner(),
                        temporary_root_factory=make_root,
                        cache_source=cache,
                    )
                self.assertFalse(owned.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)

            class GeneratedMutatingRunner(FakeProbeRunner):
                def __call__(self, argv, **kwargs):
                    result = super().__call__(argv, **kwargs)
                    generated = pathlib.Path(kwargs["cwd"]) / "src/main.rs"
                    generated.write_bytes(generated.read_bytes() + b"\n")
                    return result

            with self.assertRaisesRegex(protocol.ProtocolError, "generated.*source changed"):
                build._verify_allocation_probe_with_dependencies(
                    repository,
                    cargo=cargo,
                    command_runner=GeneratedMutatingRunner(),
                    temporary_root_factory=make_root,
                    cache_source=cache,
                )
            self.assertFalse(owned.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)
            source = repository / build.ALLOCATION_PROBE_SOURCE_ROOT / "src/main.rs"
            source.unlink()
            source.symlink_to("tests.rs")
            runner = FakeProbeRunner()
            with self.assertRaisesRegex(protocol.ProtocolError, "not a regular file"):
                build._verify_allocation_probe_with_dependencies(
                    repository,
                    cargo=cargo,
                    command_runner=runner,
                    temporary_root_factory=make_root,
                    cache_source=cache,
                )
            self.assertEqual(runner.calls, [])
            self.assertFalse(owned.exists())

    def test_manifest_dependency_contract_is_parsed_not_text_matched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)
            template = repository / build.ALLOCATION_PROBE_SOURCE_ROOT / "Cargo.toml.in"
            template.write_text(template.read_text().replace(
                'features = ["cpu-faer"]', 'features = ["cpu-blas"]', 1
            ))
            runner = FakeProbeRunner()
            with self.assertRaisesRegex(protocol.ProtocolError, "features mismatch"):
                build._verify_allocation_probe_with_dependencies(
                    repository,
                    cargo=cargo,
                    command_runner=runner,
                    temporary_root_factory=make_root,
                    cache_source=cache,
                )
            self.assertEqual(runner.calls, [])
            self.assertFalse(owned.exists())

    def test_manifest_rejects_every_noncanonical_toml_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, _, _, _, _ = self.fixture(root)
            template = (
                repository
                / build.ALLOCATION_PROBE_SOURCE_ROOT
                / build.ALLOCATION_PROBE_TEMPLATE
            ).read_bytes()
            valid = build._render_allocation_probe_manifest(template, repository).decode()
            mutations = {
                "top-level key": "foreign = true\n" + valid,
                "top-level table": valid + "\n[foreign]\nvalue = true\n",
                "package unknown": valid.replace(
                    "[package]\n", '[package]\nbuild = "build.rs"\n', 1
                ),
                "package name": valid.replace(
                    'name = "phase2e-allocation-probe"', 'name = "foreign"', 1
                ),
                "package version": valid.replace('version = "0.0.0"', 'version = "1.0.0"', 1),
                "package edition": valid.replace('edition = "2021"', 'edition = "2024"', 1),
                "package publish": valid.replace("publish = false", "publish = true", 1),
                "absolute bin": valid
                + '\n[[bin]]\nname = "probe"\npath = "/tmp/main.rs"\n',
                "tracked bin": valid
                + '\n[[bin]]\nname = "probe"\npath = "src/main.rs"\n',
                "foreign bin": valid
                + '\n[[bin]]\nname = "probe"\npath = "foreign.rs"\n',
                "build dependencies": valid + "\n[build-dependencies]\ncc = \"1\"\n",
                "dev dependencies": valid + "\n[dev-dependencies]\nserde = \"1\"\n",
                "patch": valid + "\n[patch.crates-io]\nserde = { path = \"/tmp/serde\" }\n",
                "replace": valid
                + '\n[replace]\n"serde:1.0.0" = { path = "/tmp/serde" }\n',
                "profile": valid + "\n[profile.bench]\nlto = true\n",
                "workspace members": valid + '\n[workspace]\nmembers = ["foreign"]\n',
                "workspace dependencies": valid
                + '\n[workspace.dependencies]\nserde = "1"\n',
                "features": valid + '\n[features]\ndefault = ["foreign"]\n',
                "target dependencies": valid
                + '\n[target.\'cfg(unix)\'.dependencies]\nserde = "1"\n',
            }
            for label, mutation in mutations.items():
                with self.subTest(label=label), self.assertRaises(protocol.ProtocolError):
                    build._validate_allocation_probe_manifest(
                        mutation.encode(), repository
                    )

    def test_template_accepts_only_the_single_frozen_placeholder_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, _, _, _, _ = self.fixture(root)
            template = (
                repository
                / build.ALLOCATION_PROBE_SOURCE_ROOT
                / build.ALLOCATION_PROBE_TEMPLATE
            ).read_bytes()
            mutations = (
                template.replace(
                    build.ALLOCATION_PROBE_ROOT_PLACEHOLDER.encode(),
                    b"/foreign/root",
                    1,
                ),
                template + build.ALLOCATION_PROBE_ROOT_PLACEHOLDER.encode(),
                template + b"\n# __FOREIGN_PLACEHOLDER__\n",
            )
            for mutation in mutations:
                with self.assertRaises(protocol.ProtocolError):
                    build._render_allocation_probe_manifest(mutation, repository)

    def test_invalid_manifest_is_rejected_before_launch_and_owned_root_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)
            template = repository / build.ALLOCATION_PROBE_SOURCE_ROOT / "Cargo.toml.in"
            template.write_text("foreign = true\n" + template.read_text())
            runner = FakeProbeRunner()
            with self.assertRaisesRegex(protocol.ProtocolError, "top-level"):
                build._verify_allocation_probe_with_dependencies(
                    repository,
                    cargo=cargo,
                    command_runner=runner,
                    temporary_root_factory=make_root,
                    cache_source=cache,
                )
            self.assertEqual(runner.calls, [])
            self.assertFalse(owned.exists())

    def test_lock_mutation_after_creation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository, cargo, cache, owned, make_root = self.fixture(root)

            class LockMutatingRunner(FakeProbeRunner):
                def __call__(self, argv, **kwargs):
                    result = super().__call__(argv, **kwargs)
                    if "clippy" in argv:
                        (pathlib.Path(kwargs["cwd"]) / "Cargo.lock").write_bytes(b"changed")
                    return result

            with self.assertRaisesRegex(protocol.ProtocolError, "Cargo.lock changed"):
                build._verify_allocation_probe_with_dependencies(
                    repository,
                    cargo=cargo,
                    command_runner=LockMutatingRunner(),
                    temporary_root_factory=make_root,
                    cache_source=cache,
                )
            self.assertFalse(owned.exists())

    def test_main_is_import_safe_strict_and_returns_typed_error(self) -> None:
        with mock.patch.object(
            build, "verify_allocation_probe", return_value=mock.sentinel.result
        ) as verify:
            self.assertEqual(
                build.main(
                    ["verify-allocation-probe", "--repository", "/tmp/repository"]
                ),
                0,
            )
            verify.assert_called_once_with(pathlib.Path("/tmp/repository"))
        self.assertNotEqual(build.main([]), 0)
        self.assertNotEqual(build.main(["unknown"]), 0)


class ResolvedToolchainTests(unittest.TestCase):
    def test_tools_are_canonical_regular_executables_and_actual_argv_is_absolute(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            controlled_path = controlled_tool_path(root)
            tools = build.resolve_toolchain(controlled_path)
            self.assertEqual(tools.path, controlled_path)
            for tool in (tools.git, tools.cargo, tools.rustc):
                self.assertTrue(tool.path.is_absolute())
                self.assertEqual(tool.path, tool.path.resolve())
                self.assertEqual(tool.sha256, protocol.sha256_file(tool.path))
            plan = build.build_command_plan(
                "x86_64-unknown-linux-gnu", tools.cargo
            )
            self.assertTrue(all(step.argv[0] == str(tools.cargo.path) for step in plan))
            self.assertEqual(build.BENCH_COMMAND[0], "cargo")

    def test_tool_resolution_rejects_symlink_special_nonexec_and_unneeded_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            valid_path = controlled_tool_path(root)
            git_directory, rust_directory = map(
                pathlib.Path, valid_path.split(os.pathsep)
            )

            with self.subTest("duplicate"):
                with self.assertRaises(protocol.ProtocolError):
                    build.resolve_toolchain(
                        os.pathsep.join((valid_path, str(git_directory)))
                    )

            with self.subTest("noncanonical"):
                noncanonical = git_directory / ".." / git_directory.name
                with self.assertRaises(protocol.ProtocolError):
                    build.resolve_toolchain(
                        os.pathsep.join((str(noncanonical), str(rust_directory)))
                    )

            with self.subTest("unneeded"):
                unused = root / "unused-bin"
                write_fake_tool(unused, "helper")
                with self.assertRaises(protocol.ProtocolError):
                    build.resolve_toolchain(
                        os.pathsep.join((valid_path, str(unused)))
                    )

            cargo = rust_directory / "cargo"
            cargo.unlink()
            cargo.symlink_to(rust_directory / "rustc")
            with self.subTest("symlink"):
                with self.assertRaises(protocol.ProtocolError):
                    build.resolve_toolchain(valid_path)
            cargo.unlink()
            write_fake_tool(rust_directory, "cargo")
            cargo.chmod(0o644)
            with self.subTest("non-executable"):
                with self.assertRaises(protocol.ProtocolError):
                    build.resolve_toolchain(valid_path)
            cargo.unlink()
            fifo = rust_directory / "cargo"
            os.mkfifo(fifo)
            with self.subTest("special"):
                with self.assertRaises(protocol.ProtocolError):
                    build.resolve_toolchain(valid_path)

    def test_tool_replacement_after_resolution_is_rejected_after_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            tools = build.resolve_toolchain(controlled_tool_path(root))
            process = FakeProcess([("cargo 1.90.0", "")])

            def replacing_factory(_argv, **_kwargs):
                tools.cargo.path.write_bytes(b"replacement\n")
                tools.cargo.path.chmod(0o755)
                return process

            with self.assertRaises(protocol.ProtocolError):
                build.run_bounded_command(
                    (str(tools.cargo.path), "--version", "--verbose"),
                    cwd=root,
                    environment={"PATH": tools.path},
                    deadline_seconds=300,
                    process_factory=replacing_factory,
                    executable_identity=tools.cargo,
                )


class ManifestTests(unittest.TestCase):
    def manifest(
        self,
        role: str,
        executable: pathlib.Path,
        target_dir: pathlib.Path,
        *,
        lock_sha256: str,
    ):
        target = "x86_64-unknown-linux-gnu"
        source_delta = {
            "direct-current-main-baseline": ["frozen-benchmark-harness"],
            "common-lock-normalized-baseline": [
                "frozen-benchmark-harness",
                "five-strided-pins",
            ],
            "candidate": [],
        }[role]
        tools = build.resolve_toolchain(
            controlled_tool_path(target_dir.parent / "manifest-tools")
        )
        environment = protocol.cargo_environment(
            path=tools.path,
            home="/tmp/build-home",
            cargo_home="/tmp/cargo-home",
            target_dir=str(target_dir),
        )
        config_chain = [{"path": ".cargo/config.toml", "sha256": "1" * 64}]
        return {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "toolchain": {
                "git": {
                    "path": str(tools.git.path),
                    "sha256": tools.git.sha256,
                },
                "cargo": {
                    "path": str(tools.cargo.path),
                    "sha256": tools.cargo.sha256,
                    "version": "cargo 1.90.0",
                },
                "rustc": {
                    "path": str(tools.rustc.path),
                    "sha256": tools.rustc.sha256,
                    "version": "rustc 1.90.0",
                },
            },
            "target": target,
            "profile": "bench",
            "requested_features": ["cpu-faer"],
            "provider": "Faer",
            "benchmark_sha256": "2" * 64,
            "benchmark_stanza_sha256": "3" * 64,
            "command_template": list(build.BENCH_COMMAND),
            "config_chain_sha256": protocol.sha256_json(config_chain),
            "role": role,
            "head": "4" * 40,
            "tracked_tree_sha256": "5" * 64,
            "resolved_features_sha256": "6" * 64,
            "lock_sha256": lock_sha256,
            "worktree": f"/tmp/{role}",
            "target_dir": str(target_dir),
            "executable": str(executable),
            "executable_sha256": protocol.sha256_file(executable),
            "validity_state": "COMPLETE",
            "source_delta": source_delta,
            "commands": [
                step.to_manifest()
                for step in build.build_command_plan(target, tools.cargo)
            ],
            "environment": environment,
            "cargo_config_chain": config_chain,
        }

    def test_role_comparison_allows_only_predeclared_differences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            baseline_target = root / "target-baseline"
            candidate_target = root / "target-candidate"
            baseline_executable = baseline_target / "release/deps/baseline"
            candidate_executable = candidate_target / "release/deps/candidate"
            baseline_executable.parent.mkdir(parents=True)
            candidate_executable.parent.mkdir(parents=True)
            baseline_executable.write_bytes(b"baseline binary")
            candidate_executable.write_bytes(b"candidate binary")
            baseline = self.manifest(
                "direct-current-main-baseline",
                baseline_executable,
                baseline_target,
                lock_sha256="7" * 64,
            )
            candidate = self.manifest(
                "candidate",
                candidate_executable,
                candidate_target,
                lock_sha256="8" * 64,
            )
            build.validate_pair("direct-current-main", baseline, candidate)
            candidate["requested_features"] = ["cpu-blas"]
            with self.assertRaises(protocol.ProtocolError):
                build.validate_pair("direct-current-main", baseline, candidate)

    def test_normalized_pair_requires_byte_identical_common_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            baseline_target = root / "target-baseline"
            candidate_target = root / "target-candidate"
            baseline_executable = baseline_target / "release/deps/baseline"
            candidate_executable = candidate_target / "release/deps/candidate"
            baseline_executable.parent.mkdir(parents=True)
            candidate_executable.parent.mkdir(parents=True)
            baseline_executable.write_bytes(b"baseline binary")
            candidate_executable.write_bytes(b"candidate binary")
            baseline = self.manifest(
                "common-lock-normalized-baseline",
                baseline_executable,
                baseline_target,
                lock_sha256="9" * 64,
            )
            candidate = self.manifest(
                "candidate",
                candidate_executable,
                candidate_target,
                lock_sha256="9" * 64,
            )
            build.validate_pair("common-lock-normalized", baseline, candidate)
            baseline["lock_sha256"] = "a" * 64
            with self.assertRaises(protocol.ProtocolError):
                build.validate_pair("common-lock-normalized", baseline, candidate)

    def test_manifest_is_strict_and_rehashes_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "target"
            executable = target / "release/deps/bench"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"binary")
            manifest = self.manifest(
                "candidate", executable, target, lock_sha256="b" * 64
            )
            build.validate_build_manifest(manifest)
            manifest["unexpected"] = True
            with self.assertRaises(protocol.ProtocolError):
                build.validate_build_manifest(manifest)
            del manifest["unexpected"]
            executable.write_bytes(b"modified")
            with self.assertRaises(protocol.ProtocolError):
                build.validate_build_manifest(manifest)

    def test_bench_executable_parser_requires_one_path_under_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "target"
            executable = target / "release/deps/eager_dispatch_baseline-deadbeef"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"binary")
            output = (
                "Finished `bench` profile\n"
                f"Executable benches/eager_dispatch_baseline.rs ({executable})\n"
            )
            self.assertEqual(build.parse_bench_executable(output, target), executable)
            with self.assertRaises(protocol.ProtocolError):
                build.parse_bench_executable("Finished\n", target)
            with self.assertRaises(protocol.ProtocolError):
                build.parse_bench_executable(
                    output + f"Executable another ({executable})\n", target
                )
            symlink = target / "release/deps/symlink-bench"
            symlink.symlink_to(executable)
            with self.assertRaises(protocol.ProtocolError):
                build.parse_bench_executable(
                    f"Executable benches/eager_dispatch_baseline.rs ({symlink})\n",
                    target,
                )


class FakeProcess:
    def __init__(self, responses, *, returncode=0) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.responses = list(responses)
        self.timeouts = []

    def communicate(self, timeout=None):
        self.timeouts.append(timeout)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class InterruptingRealProcess:
    def __init__(
        self,
        process: subprocess.Popen[str],
        identity_path: pathlib.Path,
        interruption: BaseException,
    ) -> None:
        self.process = process
        self.identity_path = identity_path
        self.interruption = interruption
        self.interrupted = False

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def communicate(self, timeout=None):
        if not self.interrupted:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.identity_path.exists():
                    self.interrupted = True
                    raise self.interruption
                time.sleep(0.01)
            raise AssertionError("descendant did not publish its process identity")
        return self.process.communicate(timeout=timeout)


class BoundedProcessTests(unittest.TestCase):
    _REAL_GROUP_SCRIPT = """
import os
import signal
import sys

leader_group = os.getpgrp()
reaper = os.fork()
if reaper == 0:
    os.setpgid(0, 0)
    for descriptor in (1, 2):
        try:
            os.close(descriptor)
        except OSError:
            pass
    child = os.fork()
    if child == 0:
        os.setpgid(0, leader_group)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open(sys.argv[1], "w", encoding="utf-8") as identity:
            identity.write(f"{os.getpid()} {os.getpgrp()}\\n")
            identity.flush()
            os.fsync(identity.fileno())
        while True:
            signal.pause()
    os.waitpid(child, 0)
    os._exit(0)

while True:
    signal.pause()
"""

    @staticmethod
    def _read_process_identity(path: pathlib.Path) -> tuple[int, int]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                fields = path.read_text().split()
            except FileNotFoundError:
                fields = []
            if len(fields) == 2:
                return int(fields[0]), int(fields[1])
            time.sleep(0.01)
        raise AssertionError("descendant did not publish its process identity")

    @staticmethod
    def _linux_process_state(pid: int) -> tuple[str, int] | None:
        try:
            suffix = (
                pathlib.Path(f"/proc/{pid}/stat")
                .read_text()
                .rsplit(")", 1)[1]
            )
        except (FileNotFoundError, IndexError, OSError):
            return None
        fields = suffix.split()
        if len(fields) < 3:
            return None
        return fields[0], int(fields[2])

    @classmethod
    def _linux_group_has_member(cls, process_group: int) -> bool:
        try:
            entries = tuple(pathlib.Path("/proc").iterdir())
        except OSError:
            return False
        for entry in entries:
            if not entry.name.isdigit():
                continue
            state = cls._linux_process_state(int(entry.name))
            if state is not None and state[1] == process_group:
                return True
        return False

    @classmethod
    def _wait_for_real_group_exit(cls, leader: int, descendant: int) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (
                cls._linux_process_state(leader) is None
                and cls._linux_process_state(descendant) is None
                and not cls._linux_group_has_member(leader)
            ):
                return
            time.sleep(0.01)
        raise AssertionError("real process group still has live members")

    @staticmethod
    def _cleanup_real_group(
        process: subprocess.Popen[str] | None,
        descendant: int | None,
    ) -> None:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        if descendant is not None:
            try:
                os.kill(descendant, signal.SIGKILL)
            except OSError:
                pass
        if process is not None:
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.communicate(timeout=1)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass

    @unittest.skipUnless(
        os.name == "posix" and sys.platform.startswith("linux"),
        "requires Linux process groups and /proc",
    )
    def test_real_timeout_kills_term_ignoring_descendant_after_leader_exit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            identity_path = root / "descendant.identity"
            process: subprocess.Popen[str] | None = None
            descendant: int | None = None

            def factory(argv, **kwargs):
                nonlocal process
                process = subprocess.Popen(argv, **kwargs)
                return process

            try:
                result = build.run_bounded_command(
                    (
                        sys.executable,
                        "-c",
                        self._REAL_GROUP_SCRIPT,
                        str(identity_path),
                    ),
                    cwd=root,
                    environment={"PATH": os.environ.get("PATH", "")},
                    deadline_seconds=1,
                    process_factory=factory,
                )
                descendant, process_group = self._read_process_identity(identity_path)
                self.assertIsNotNone(process)
                self.assertEqual(process_group, process.pid)
                self.assertEqual(result.validity_state, "INCONCLUSIVE")
                self.assertEqual(result.failure_reason, "deadline-exceeded")
                self.assertTrue(result.terminated)
                self.assertTrue(result.killed)
                self.assertIsNotNone(process.returncode)
                self._wait_for_real_group_exit(process.pid, descendant)
            finally:
                self._cleanup_real_group(process, descendant)

    @unittest.skipUnless(
        os.name == "posix" and sys.platform.startswith("linux"),
        "requires Linux process groups and /proc",
    )
    def test_real_cancellation_kills_group_and_preserves_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            identity_path = root / "descendant.identity"
            interruption = KeyboardInterrupt("cancelled")
            process: subprocess.Popen[str] | None = None
            descendant: int | None = None

            def factory(argv, **kwargs):
                nonlocal process
                process = subprocess.Popen(argv, **kwargs)
                return InterruptingRealProcess(
                    process, identity_path, interruption
                )

            try:
                with self.assertRaises(KeyboardInterrupt) as caught:
                    build.run_bounded_command(
                        (
                            sys.executable,
                            "-c",
                            self._REAL_GROUP_SCRIPT,
                            str(identity_path),
                        ),
                        cwd=root,
                        environment={"PATH": os.environ.get("PATH", "")},
                        deadline_seconds=30,
                        process_factory=factory,
                    )
                self.assertIs(caught.exception, interruption)
                descendant, process_group = self._read_process_identity(identity_path)
                self.assertIsNotNone(process)
                self.assertEqual(process_group, process.pid)
                self.assertIsNotNone(process.returncode)
                self._wait_for_real_group_exit(process.pid, descendant)
            finally:
                self._cleanup_real_group(process, descendant)

    def test_success_uses_a_new_process_group_and_records_actual_inputs(self) -> None:
        process = FakeProcess([("stdout", "stderr")])
        launches = []

        def factory(argv, **kwargs):
            launches.append((argv, kwargs))
            return process

        result = build.run_bounded_command(
            ("cargo", "metadata"),
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=300,
            process_factory=factory,
        )
        self.assertEqual(result.validity_state, "COMPLETE")
        self.assertEqual(result.argv, ("cargo", "metadata"))
        self.assertEqual(result.environment, {"PATH": "/bin"})
        self.assertEqual(process.timeouts, [300])
        self.assertTrue(launches[0][1]["start_new_session"])

    def test_timeout_terminates_group_and_waits_the_five_second_grace(self) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        process = FakeProcess(
            [timeout, ("partial", "timed out"), ("partial", "timed out")]
        )
        signals = []
        result = build.run_bounded_command(
            ("cargo", "metadata"),
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=300,
            process_factory=lambda *_args, **_kwargs: process,
            signal_process_group=lambda pid, sig: signals.append((pid, sig)),
        )
        self.assertEqual(result.validity_state, "INCONCLUSIVE")
        self.assertEqual(result.failure_reason, "deadline-exceeded")
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )

    def test_timeout_kills_process_group_survivors_after_grace(self) -> None:
        first = subprocess.TimeoutExpired(("cargo", "bench"), 1800)
        second = subprocess.TimeoutExpired(("cargo", "bench"), 5)
        process = FakeProcess([first, second, ("partial", "killed")])
        signals = []
        result = build.run_bounded_command(
            build.BENCH_COMMAND,
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=1800,
            process_factory=lambda *_args, **_kwargs: process,
            signal_process_group=lambda pid, sig: signals.append((pid, sig)),
        )
        self.assertEqual(result.validity_state, "INCONCLUSIVE")
        self.assertEqual(process.timeouts, [1800, 5, 5])
        self.assertNotIn(None, process.timeouts)
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )

    def test_timeout_kills_surviving_descendant_after_leader_closes_pipes(self) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        process = FakeProcess(
            [timeout, ("leader drained", ""), ("group drained", "")]
        )
        signals = []

        result = build.run_bounded_command(
            ("cargo", "metadata"),
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=300,
            process_factory=lambda *_args, **_kwargs: process,
            signal_process_group=lambda pid, sig: signals.append((pid, sig)),
        )

        self.assertEqual(result.validity_state, "INCONCLUSIVE")
        self.assertTrue(result.terminated)
        self.assertTrue(result.killed)
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_nonzero_command_is_validity_inconclusive(self) -> None:
        process = FakeProcess([("stdout", "failure")], returncode=9)
        result = build.run_bounded_command(
            ("cargo", "tree"),
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=300,
            process_factory=lambda *_args, **_kwargs: process,
        )
        self.assertEqual(result.validity_state, "INCONCLUSIVE")
        self.assertEqual(result.failure_reason, "nonzero-exit")
        self.assertEqual(result.returncode, 9)

    def test_cancellation_terminates_group_and_preserves_base_exception(self) -> None:
        interruption = KeyboardInterrupt("cancelled")
        process = FakeProcess(
            [interruption, ("partial", "cancelled"), ("partial", "cancelled")]
        )
        signals = []
        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=lambda pid, sig: signals.append((pid, sig)),
            )
        self.assertIs(caught.exception, interruption)
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )

    def test_cancellation_kills_surviving_descendant_after_leader_closes_pipes(
        self,
    ) -> None:
        interruption = KeyboardInterrupt("cancelled")
        process = FakeProcess(
            [interruption, ("leader drained", ""), ("group drained", "")]
        )
        signals = []

        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=lambda pid, sig: signals.append((pid, sig)),
            )

        self.assertIs(caught.exception, interruption)
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_cancellation_preserves_original_when_cleanup_signal_fails(self) -> None:
        interruption = KeyboardInterrupt("cancelled")
        process = FakeProcess([interruption])

        def fail_signal(_pid, _signal):
            raise OSError("signal failure")

        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=fail_signal,
            )
        self.assertIs(caught.exception, interruption)

    def test_post_kill_pipe_timeout_returns_partial_typed_evidence(self) -> None:
        first = subprocess.TimeoutExpired(
            ("cargo", "bench"), 1800, output="started\n", stderr="warning\n"
        )
        grace = subprocess.TimeoutExpired(
            ("cargo", "bench"), 5, output="term partial\n", stderr="term err\n"
        )
        drain = subprocess.TimeoutExpired(
            ("cargo", "bench"), 5, output="kill partial\n", stderr="kill err\n"
        )
        process = FakeProcess([first, grace, drain])
        result = build.run_bounded_command(
            build.BENCH_COMMAND,
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=1800,
            process_factory=lambda *_args, **_kwargs: process,
            signal_process_group=lambda _pid, _sig: None,
        )
        self.assertEqual(result.validity_state, "INCONCLUSIVE")
        self.assertEqual(
            result.failure_reason,
            "deadline-exceeded:post-kill-drain-timeout",
        )
        self.assertEqual(result.stdout, "kill partial\n")
        self.assertEqual(result.stderr, "kill err\n")
        self.assertEqual(process.timeouts, [1800, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_timeout_term_signal_failure_still_kills_and_drains_bounded(self) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        process = FakeProcess([timeout, ("after kill", "term failed")])
        signals = []

        def signal_group(pid, requested_signal):
            signals.append((pid, requested_signal))
            if requested_signal == signal.SIGTERM:
                raise OSError("TERM failed")

        result = build.run_bounded_command(
            ("cargo", "metadata"),
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=300,
            process_factory=lambda *_args, **_kwargs: process,
            signal_process_group=signal_group,
        )
        self.assertEqual(result.validity_state, "INCONCLUSIVE")
        self.assertEqual(
            result.failure_reason,
            "deadline-exceeded:term-signal-failed",
        )
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )
        self.assertEqual(process.timeouts, [300, 5])
        self.assertNotIn(None, process.timeouts)

    def test_timeout_kill_signal_failure_is_typed_and_drain_is_bounded(self) -> None:
        first = subprocess.TimeoutExpired(("cargo", "bench"), 1800)
        grace = subprocess.TimeoutExpired(("cargo", "bench"), 5)
        process = FakeProcess([first, grace, ("after failed kill", "kill failed")])

        def signal_group(_pid, requested_signal):
            if requested_signal == signal.SIGKILL:
                raise OSError("KILL failed")

        result = build.run_bounded_command(
            build.BENCH_COMMAND,
            cwd=pathlib.Path("/tmp/worktree"),
            environment={"PATH": "/bin"},
            deadline_seconds=1800,
            process_factory=lambda *_args, **_kwargs: process,
            signal_process_group=signal_group,
        )
        self.assertEqual(result.validity_state, "INCONCLUSIVE")
        self.assertEqual(
            result.failure_reason,
            "deadline-exceeded:kill-signal-failed",
        )
        self.assertEqual(process.timeouts, [1800, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_cancellation_post_kill_cleanup_is_bounded_and_preserves_original(self) -> None:
        interruption = KeyboardInterrupt("cancelled")
        grace = subprocess.TimeoutExpired(("cargo", "metadata"), 5)
        drain = subprocess.TimeoutExpired(("cargo", "metadata"), 5)
        process = FakeProcess([interruption, grace, drain])
        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=lambda _pid, _sig: None,
            )
        self.assertIs(caught.exception, interruption)
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_timeout_cleanup_interruption_during_term_forces_kill_and_drain(self) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        interruption = KeyboardInterrupt("during TERM")
        process = FakeProcess([timeout, ("drained", "")])
        signals = []

        def signal_group(pid, requested_signal):
            signals.append((pid, requested_signal))
            if requested_signal == signal.SIGTERM:
                raise interruption

        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=signal_group,
            )
        self.assertIs(caught.exception, interruption)
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )
        self.assertEqual(process.timeouts, [300, 5])
        self.assertNotIn(None, process.timeouts)

    def test_timeout_cleanup_interruption_during_grace_forces_kill_and_drain(self) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        interruption = KeyboardInterrupt("during grace")
        process = FakeProcess([timeout, interruption])
        signals = []
        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=lambda pid, sig: signals.append((pid, sig)),
            )
        self.assertIs(caught.exception, interruption)
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_timeout_cleanup_interruption_during_kill_retries_and_drains(self) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        grace = subprocess.TimeoutExpired(("cargo", "metadata"), 5)
        interruption = KeyboardInterrupt("during KILL")
        process = FakeProcess([timeout, grace, ("drained", "")])
        signals = []
        first_kill = True

        def signal_group(pid, requested_signal):
            nonlocal first_kill
            signals.append((pid, requested_signal))
            if requested_signal == signal.SIGKILL and first_kill:
                first_kill = False
                raise interruption

        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=signal_group,
            )
        self.assertIs(caught.exception, interruption)
        self.assertEqual(
            signals,
            [
                (process.pid, signal.SIGTERM),
                (process.pid, signal.SIGKILL),
                (process.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_timeout_cleanup_interruption_during_final_drain_rekills_and_redrains(
        self,
    ) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        grace = subprocess.TimeoutExpired(("cargo", "metadata"), 5)
        interruption = KeyboardInterrupt("during final drain")
        process = FakeProcess(
            [timeout, grace, interruption, ("finally drained", "")]
        )
        signals = []
        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=lambda pid, sig: signals.append((pid, sig)),
            )
        self.assertIs(caught.exception, interruption)
        self.assertEqual(
            signals,
            [
                (process.pid, signal.SIGTERM),
                (process.pid, signal.SIGKILL),
                (process.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(process.timeouts, [300, 5, 5, 5])
        self.assertNotIn(None, process.timeouts)

    def test_timeout_cleanup_preserves_interruption_when_forced_kill_fails(self) -> None:
        timeout = subprocess.TimeoutExpired(("cargo", "metadata"), 300)
        interruption = KeyboardInterrupt("during grace")
        process = FakeProcess([timeout, interruption, ("bounded drain", "")])
        signals = []

        def signal_group(pid, requested_signal):
            signals.append((pid, requested_signal))
            if requested_signal == signal.SIGKILL:
                raise OSError("KILL failed")

        with self.assertRaises(KeyboardInterrupt) as caught:
            build.run_bounded_command(
                ("cargo", "metadata"),
                cwd=pathlib.Path("/tmp/worktree"),
                environment={"PATH": "/bin"},
                deadline_seconds=300,
                process_factory=lambda *_args, **_kwargs: process,
                signal_process_group=signal_group,
            )
        self.assertIs(caught.exception, interruption)
        self.assertEqual(
            signals,
            [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)],
        )
        self.assertEqual(process.timeouts, [300, 5, 5])
        self.assertNotIn(None, process.timeouts)


class FakeSourceControl:
    def __init__(self) -> None:
        self.created = []
        self.materialized = []
        self.heads = {}
        self.validated_sources = []
        self.proof_overrides = {}

    def create_worktree(self, spec) -> None:
        self.created.append(spec)
        if spec.role == "candidate":
            root_cargo = old_root_cargo().replace(
                build.OLD_STRIDED.encode(), build.COMMON_STRIDED.encode()
            )
            ad_cargo = b"[package]\nname = \"tenferro-ad\"\n" + build.BENCH_STANZA.encode()
            bench = b"fn main() {}\n"
        else:
            root_cargo = old_root_cargo()
            ad_cargo = b"[package]\nname = \"tenferro-ad\"\n"
            bench = None
        (spec.path / build.ROOT_CARGO_PATH).write_bytes(root_cargo)
        ad_path = spec.path / build.AD_CARGO_PATH
        ad_path.parent.mkdir(parents=True)
        ad_path.write_bytes(ad_cargo)
        if bench is not None:
            bench_path = spec.path / build.BENCH_SOURCE_PATH
            bench_path.parent.mkdir(parents=True)
            bench_path.write_bytes(bench)
        self.heads[spec.path] = spec.start_commit

    def materialize_baseline(self, spec) -> str:
        self.materialized.append(spec.role)
        ad_path = spec.path / build.AD_CARGO_PATH
        ad_path.write_bytes(ad_path.read_bytes() + build.BENCH_STANZA.encode())
        bench_path = spec.path / build.BENCH_SOURCE_PATH
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench_path.write_bytes(b"fn main() {}\n")
        if spec.role == "common-lock-normalized-baseline":
            cargo = spec.path / build.ROOT_CARGO_PATH
            cargo.write_bytes(build.normalized_root_cargo(cargo.read_bytes()))
        head = (
            "d" * 40
            if spec.role == "direct-current-main-baseline"
            else "e" * 40
        )
        self.heads[spec.path] = head
        return head

    def validate_worktree(self, path, expected_head, expected_lock):
        self.assert_expected(path, expected_head, expected_lock)
        benchmark = pathlib.Path(path) / build.BENCH_SOURCE_PATH
        proof = build.WorktreeProof(
            head=expected_head,
            tracked_tree_sha256=sha256_bytes(str(path).encode()),
            benchmark_sha256=protocol.sha256_file(benchmark),
            benchmark_stanza_sha256=sha256_bytes(build.BENCH_STANZA.encode()),
            cargo_config_chain=(),
        )
        override = self.proof_overrides.get(pathlib.Path(path), {})
        if not override:
            return proof
        return build.WorktreeProof(
            head=override.get("head", proof.head),
            tracked_tree_sha256=override.get(
                "tracked_tree_sha256", proof.tracked_tree_sha256
            ),
            benchmark_sha256=override.get(
                "benchmark_sha256", proof.benchmark_sha256
            ),
            benchmark_stanza_sha256=override.get(
                "benchmark_stanza_sha256", proof.benchmark_stanza_sha256
            ),
            cargo_config_chain=override.get(
                "cargo_config_chain", proof.cargo_config_chain
            ),
        )

    def validate_role_source(self, role, head, expected_candidate):
        self.validated_sources.append((role, head, expected_candidate))
        if role == "candidate":
            if head != expected_candidate:
                raise protocol.ProtocolError("candidate commit mismatch")
            return
        expected = {
            "direct-current-main-baseline": "d" * 40,
            "common-lock-normalized-baseline": "e" * 40,
        }[role]
        if head != expected:
            raise protocol.ProtocolError("baseline measurement commit mismatch")

    def assert_expected(self, path, expected_head, expected_lock) -> None:
        if self.heads[pathlib.Path(path)] != expected_head:
            raise AssertionError("unexpected worktree HEAD")
        if (pathlib.Path(path) / "Cargo.lock").read_bytes() != pathlib.Path(
            expected_lock
        ).read_bytes():
            raise AssertionError("wrong installed root-owned lock")


class FakeCargoRunner:
    def __init__(self, *, timeout_build=False, mismatched_role=None) -> None:
        self.calls = []
        self.timeout_build = timeout_build
        self.mismatched_role = mismatched_role

    def __call__(self, argv, *, cwd, environment, deadline_seconds):
        argv = tuple(argv)
        cwd = pathlib.Path(cwd)
        environment = dict(environment)
        self.calls.append((argv, cwd, environment, deadline_seconds))
        stdout = ""
        stderr = ""
        returncode = 0
        validity = "COMPLETE"
        reason = None
        terminated = False
        killed = False
        tool = pathlib.Path(argv[0]).name
        arguments = argv[1:]
        mismatched = cwd.name == self.mismatched_role
        if tool == "rustc" and arguments == ("--version", "--verbose"):
            version = "1.89.0" if mismatched else "1.90.0"
            host = "aarch64-unknown-linux-gnu" if mismatched else "x86_64-unknown-linux-gnu"
            stdout = f"rustc {version}\nhost: {host}\n"
        elif tool == "cargo" and arguments == ("--version", "--verbose"):
            version = "1.89.0" if mismatched else "1.90.0"
            stdout = f"cargo {version}\n"
        elif tool == "cargo" and arguments == build.LOCK_COMMAND[1:]:
            lock = b"direct lock\n" if "direct" in cwd.name else b"common lock\n"
            (cwd / "Cargo.lock").write_bytes(lock)
        elif tool == "cargo" and arguments == build.METADATA_COMMAND[1:]:
            stdout = json.dumps({"packages": [], "resolve": {"nodes": []}})
        elif tool == "cargo" and arguments[:1] == ("tree",):
            stdout = f'tenferro-ad ({cwd})\ntenferro-cpu feature "cpu-faer"\n'
        elif tool == "cargo" and arguments == build.BENCH_COMMAND[1:]:
            if self.timeout_build:
                validity = "INCONCLUSIVE"
                reason = "deadline-exceeded"
                terminated = True
                killed = True
                returncode = None
            else:
                executable = (
                    pathlib.Path(environment["CARGO_TARGET_DIR"])
                    / "release/deps/eager_dispatch_baseline-deadbeef"
                )
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_bytes(f"binary:{cwd.name}".encode())
                stderr = (
                    "Finished `bench` profile\n"
                    f"Executable benches/eager_dispatch_baseline.rs ({executable})\n"
                )
        else:
            raise AssertionError(f"unexpected command: {argv!r}")
        return build.CommandResult(
            argv=argv,
            cwd=str(cwd),
            environment=environment,
            deadline_seconds=deadline_seconds,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            validity_state=validity,
            failure_reason=reason,
            terminated=terminated,
            killed=killed,
        )


class BuildOrchestratorTests(unittest.TestCase):
    def config(self, root: pathlib.Path):
        repository = root / "repository"
        evidence = repository / "docs/worklogs/phase2e-evidence"
        home = root / "home"
        cargo_home = root / "cargo-home"
        repository.mkdir()
        evidence.mkdir(parents=True)
        home.mkdir()
        cargo_home.mkdir()
        (cargo_home / "registry").mkdir()
        (cargo_home / "git").mkdir()
        return build.BuildConfig(
            repository=repository,
            evidence_root=evidence,
            scratch_root=root / "scratch",
            candidate_commit="c" * 40,
            path=controlled_tool_path(root / "controlled-tools"),
            home=home,
            cargo_home=cargo_home,
        )

    def fake_source(self, config: build.BuildConfig) -> FakeSourceControl:
        source_control = FakeSourceControl()
        source_control.tools = build.resolve_toolchain(config.path)
        return source_control

    def fake_build_set(
        self, root: pathlib.Path
    ) -> tuple[
        build.BuildConfig,
        FakeSourceControl,
        build.BuildSetResult,
    ]:
        config = self.config(root)
        source_control = self.fake_source(config)
        result = build._build_all_with_dependencies(
            config,
            source_control=source_control,
            command_runner=FakeCargoRunner(),
        )
        self.assertEqual(result.validity_state, "COMPLETE")
        return config, source_control, result

    def rewrite_manifests(self, config: build.BuildConfig, mutate) -> None:
        for relative in build.BUILD_MANIFEST_PATHS.values():
            path = config.evidence_root / relative
            manifest = json.loads(path.read_text())
            mutate(manifest)
            path.chmod(0o644)
            path.write_text(json.dumps(manifest))
            path.chmod(0o444)

    def public_validate_with_fake_dependencies(
        self,
        config: build.BuildConfig,
        source_control: FakeSourceControl,
        command_runner: FakeCargoRunner,
    ):
        with mock.patch.object(
            build, "GitSourceControl", return_value=source_control
        ), mock.patch.object(
            build, "run_bounded_command", side_effect=command_runner
        ):
            return build.validate_build_set(config)

    def test_public_build_set_validator_has_no_source_adapter_parameter(self) -> None:
        signature = inspect.signature(build.validate_build_set)
        self.assertEqual(list(signature.parameters), ["config"])

    def test_public_build_orchestrator_has_no_dependency_injection_parameters(
        self,
    ) -> None:
        signature = inspect.signature(build.build_all)
        self.assertEqual(list(signature.parameters), ["config"])

    def test_public_build_orchestrator_cannot_be_fed_fabricated_dependencies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self.config(root)
            with self.assertRaises(TypeError):
                build.build_all(
                    config,
                    source_control=FakeSourceControl(),
                    command_runner=FakeCargoRunner(),
                )

    def test_public_build_orchestrator_returns_only_authoritatively_reopened_manifests(
        self,
    ) -> None:
        private_builder = getattr(build, "_build_all_with_dependencies", None)
        self.assertIsNotNone(private_builder)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self.config(root)
            built = build.BuildSetResult(
                "COMPLETE", {"candidate": {"origin": "builder"}}, None
            )
            reopened = {"candidate": {"origin": "persisted-validator"}}
            with mock.patch.object(
                build,
                "_build_all_with_dependencies",
                return_value=built,
            ) as private_build, mock.patch.object(
                build, "GitSourceControl"
            ) as git_source, mock.patch.object(
                build,
                "validate_build_set",
                return_value=reopened,
            ) as validate:
                result = build.build_all(config)

            self.assertEqual(result, build.BuildSetResult("COMPLETE", reopened, None))
            validate.assert_called_once_with(config)
            self.assertEqual(private_build.call_count, 1)
            self.assertEqual(git_source.call_count, 1)

    def test_public_build_set_validator_cannot_be_fed_a_fabricated_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self.config(root)
            source_control = self.fake_source(config)
            result = build._build_all_with_dependencies(
                config,
                source_control=source_control,
                command_runner=FakeCargoRunner(),
            )
            self.assertEqual(result.validity_state, "COMPLETE")
            candidate_cargo = config.scratch_root / "candidate/Cargo.toml"
            candidate_cargo.write_text("[workspace]\nmembers = []\n")
            with self.assertRaises(TypeError):
                build.validate_build_set(
                    config,
                    source_control=source_control,
                )

    def test_public_persisted_validation_reexecutes_bound_role_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config, source_control, result = self.fake_build_set(root)
            runner = FakeCargoRunner()

            loaded = self.public_validate_with_fake_dependencies(
                config, source_control, runner
            )

            self.assertEqual(loaded, result.manifests)
            self.assertEqual(len(runner.calls), 9)
            tools = build.resolve_toolchain(config.path)
            for role in build.BUILD_MANIFEST_PATHS:
                role_calls = [
                    call for call in runner.calls if call[1].name == role
                ]
                self.assertEqual(len(role_calls), 3)
                self.assertEqual(
                    [pathlib.Path(call[0][0]) for call in role_calls],
                    [tools.rustc.path, tools.cargo.path, tools.cargo.path],
                )
                self.assertTrue(
                    all(call[3] == build.QUERY_DEADLINE_SECONDS for call in role_calls)
                )

    def test_public_persisted_validation_rejects_joint_tool_version_forgery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config, source_control, _result = self.fake_build_set(root)

            def forge_versions(manifest):
                manifest["toolchain"]["cargo"]["version"] = "cargo 99.0.0"
                manifest["toolchain"]["rustc"]["version"] = (
                    "rustc 99.0.0\nhost: x86_64-unknown-linux-gnu"
                )

            self.rewrite_manifests(config, forge_versions)
            with self.assertRaises(protocol.ProtocolError):
                self.public_validate_with_fake_dependencies(
                    config, source_control, FakeCargoRunner()
                )

    def test_public_persisted_validation_rejects_joint_target_command_forgery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config, source_control, _result = self.fake_build_set(root)
            tools = build.resolve_toolchain(config.path)
            forged_target = "aarch64-unknown-linux-gnu"

            def forge_target_and_commands(manifest):
                manifest["target"] = forged_target
                manifest["commands"] = [
                    command.to_manifest()
                    for command in build.build_command_plan(
                        forged_target, tools.cargo
                    )
                ]

            self.rewrite_manifests(config, forge_target_and_commands)
            with self.assertRaises(protocol.ProtocolError):
                self.public_validate_with_fake_dependencies(
                    config, source_control, FakeCargoRunner()
                )

    def test_public_persisted_validation_rejects_resolved_feature_forgery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config, source_control, _result = self.fake_build_set(root)
            candidate_path = (
                config.evidence_root / build.BUILD_MANIFEST_PATHS["candidate"]
            )
            candidate = json.loads(candidate_path.read_text())
            candidate["resolved_features_sha256"] = "0" * 64
            candidate_path.chmod(0o644)
            candidate_path.write_text(json.dumps(candidate))
            candidate_path.chmod(0o444)

            with self.assertRaises(protocol.ProtocolError):
                self.public_validate_with_fake_dependencies(
                    config, source_control, FakeCargoRunner()
                )

    def test_orchestrator_builds_three_roles_and_binds_root_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self.config(root)
            source_control = self.fake_source(config)
            runner = FakeCargoRunner()
            result = build._build_all_with_dependencies(
                config, source_control=source_control, command_runner=runner
            )
            self.assertEqual(result.validity_state, "COMPLETE")
            self.assertIsNone(result.failure)
            self.assertEqual(set(result.manifests), set(build.BUILD_MANIFEST_PATHS))
            self.assertEqual(len(source_control.created), 3)
            self.assertEqual(
                source_control.materialized,
                [
                    "direct-current-main-baseline",
                    "common-lock-normalized-baseline",
                ],
            )
            self.assertEqual(len(runner.calls), 17)
            direct_lock = config.evidence_root / build.LOCK_PATHS["direct"]
            common_lock = config.evidence_root / build.LOCK_PATHS["common"]
            self.assertNotEqual(direct_lock.read_bytes(), common_lock.read_bytes())
            specs = {
                spec.role: spec
                for spec in build.worktree_specs(config.scratch_root, "c" * 40)
            }
            self.assertEqual(
                (specs["direct-current-main-baseline"].path / "Cargo.lock").read_bytes(),
                direct_lock.read_bytes(),
            )
            for role in ("common-lock-normalized-baseline", "candidate"):
                self.assertEqual(
                    (specs[role].path / "Cargo.lock").read_bytes(),
                    common_lock.read_bytes(),
                )
            for role, relative in build.BUILD_MANIFEST_PATHS.items():
                stored = json.loads((config.evidence_root / relative).read_text())
                self.assertEqual(stored, result.manifests[role])
                build.validate_build_manifest(stored)
            build.validate_pair(
                "direct-current-main",
                result.manifests["direct-current-main-baseline"],
                result.manifests["candidate"],
            )
            build.validate_pair(
                "common-lock-normalized",
                result.manifests["common-lock-normalized-baseline"],
                result.manifests["candidate"],
            )
            loaded = build._validate_build_set_with_source_control(
                config,
                source_control,
                command_runner=FakeCargoRunner(),
            )
            self.assertEqual(loaded, result.manifests)
            common_lock.chmod(0o644)
            common_lock.write_bytes(b"tampered common lock\n")
            with self.assertRaises(protocol.ProtocolError):
                build._validate_build_set_with_source_control(
                    config,
                    source_control,
                    command_runner=FakeCargoRunner(),
                )

    def test_role_local_toolchain_probe_mismatch_is_rejected_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self.config(root)
            runner = FakeCargoRunner(
                mismatched_role="direct-current-main-baseline"
            )
            source_control = self.fake_source(config)
            with self.assertRaises(protocol.ProtocolError):
                build._build_all_with_dependencies(
                    config,
                    source_control=source_control,
                    command_runner=runner,
                )
            self.assertFalse(
                any(
                    pathlib.Path(argv[0]).name == "cargo"
                    and argv[1:] == build.LOCK_COMMAND[1:]
                    for argv, _cwd, _environment, _deadline in runner.calls
                )
            )

    def test_persisted_validation_recomputes_fake_proofs_and_binds_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self.config(root)
            source_control = self.fake_source(config)
            result = build._build_all_with_dependencies(
                config,
                source_control=source_control,
                command_runner=FakeCargoRunner(),
            )
            self.assertEqual(result.validity_state, "COMPLETE")
            build._validate_build_set_with_source_control(
                config,
                source_control,
                command_runner=FakeCargoRunner(),
            )
            self.assertEqual(
                {role for role, _head, _candidate in source_control.validated_sources},
                set(build.BUILD_MANIFEST_PATHS),
            )

            candidate_path = config.scratch_root / "candidate"
            source_control.proof_overrides[candidate_path] = {
                "tracked_tree_sha256": "f" * 64
            }
            with self.assertRaises(protocol.ProtocolError):
                build._validate_build_set_with_source_control(
                    config,
                    source_control,
                    command_runner=FakeCargoRunner(),
                )
            source_control.proof_overrides.clear()

            wrong_candidate = build.BuildConfig(
                repository=config.repository,
                evidence_root=config.evidence_root,
                scratch_root=config.scratch_root,
                candidate_commit="f" * 40,
                path=config.path,
                home=config.home,
                cargo_home=config.cargo_home,
            )
            with self.assertRaises(protocol.ProtocolError):
                build._validate_build_set_with_source_control(
                    wrong_candidate,
                    source_control,
                    command_runner=FakeCargoRunner(),
                )

            manifest_path = (
                config.evidence_root / build.BUILD_MANIFEST_PATHS["candidate"]
            )
            manifest = json.loads(manifest_path.read_text())
            manifest["target_dir"] = str(root / "forged-target")
            manifest_path.chmod(0o644)
            manifest_path.write_text(json.dumps(manifest))
            manifest_path.chmod(0o444)
            with self.assertRaises(protocol.ProtocolError):
                build._validate_build_set_with_source_control(
                    config,
                    source_control,
                    command_runner=FakeCargoRunner(),
                )

    def test_build_timeout_returns_validity_inconclusive_without_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self.config(root)
            source_control = self.fake_source(config)
            result = build._build_all_with_dependencies(
                config,
                source_control=source_control,
                command_runner=FakeCargoRunner(timeout_build=True),
            )
            self.assertEqual(result.validity_state, "INCONCLUSIVE")
            self.assertIsNotNone(result.failure)
            self.assertEqual(result.failure.failure_reason, "deadline-exceeded")
            for relative in build.BUILD_MANIFEST_PATHS.values():
                self.assertFalse((config.evidence_root / relative).exists())

    def test_invalid_controlled_path_is_rejected_before_worktree_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            valid = self.config(root)
            config = build.BuildConfig(
                repository=valid.repository,
                evidence_root=valid.evidence_root,
                scratch_root=valid.scratch_root,
                candidate_commit=valid.candidate_commit,
                path="relative/bin",
                home=valid.home,
                cargo_home=valid.cargo_home,
            )
            source_control = self.fake_source(valid)
            with self.assertRaises(protocol.ProtocolError):
                build._build_all_with_dependencies(
                    config,
                    source_control=source_control,
                    command_runner=FakeCargoRunner(),
                )
            self.assertEqual(source_control.created, [])


class AllocationProbeBuildPlanTests(unittest.TestCase):
    def test_three_probe_identities_bind_role_worktrees_locks_and_manifests(self) -> None:
        self.assertEqual(
            build.PROBE_BUILD_MANIFEST_PATHS,
            {
                "direct-current-main-baseline": pathlib.Path(
                    "builds/probes/direct-current-main-baseline.json"
                ),
                "common-lock-normalized-baseline": pathlib.Path(
                    "builds/probes/common-lock-normalized-baseline.json"
                ),
                "candidate": pathlib.Path("builds/probes/candidate.json"),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = BuildOrchestratorTests().config(root)
            manifests = {
                role: {"role": role, "head": character * 40}
                for role, character in (
                    ("direct-current-main-baseline", "d"),
                    ("common-lock-normalized-baseline", "e"),
                    ("candidate", "c"),
                )
            }
            specs = build.allocation_probe_build_specs(config, manifests)
            self.assertEqual([spec.role for spec in specs], list(build.BUILD_MANIFEST_PATHS))
            self.assertEqual(
                [spec.lock_name for spec in specs],
                ["direct-probe", "common-probe", "common-probe"],
            )
            for spec in specs:
                self.assertEqual(spec.repository, config.scratch_root / spec.role)
                self.assertEqual(
                    spec.manifest_path,
                    config.evidence_root / build.PROBE_BUILD_MANIFEST_PATHS[spec.role],
                )
                self.assertEqual(spec.profile, "bench")

    def test_probe_build_command_is_locked_bench_profile_and_lists_cases(self) -> None:
        manifest = pathlib.Path("/probe/Cargo.toml")
        binary = pathlib.Path("/target/release/phase2e-allocation-probe")
        plan = build.allocation_probe_build_only_command_plan(
            manifest, binary, "/tools/cargo"
        )
        self.assertEqual([step.name for step in plan], ["build", "list-cases"])
        self.assertEqual(
            plan[0].argv,
            (
                "/tools/cargo",
                "build",
                "--locked",
                "--profile",
                "bench",
                "--manifest-path",
                str(manifest),
            ),
        )
        self.assertEqual(plan[1].argv, (str(binary), "--list-cases"))

    def test_probe_builder_runs_two_lock_generations_and_three_independent_builds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            write_probe_fixture(repository)
            evidence = root / "evidence"
            evidence.mkdir()
            scratch = root / "scratch"
            for role in build.BUILD_MANIFEST_PATHS:
                for crate in ("tenferro-ad", "tenferro-cpu", "tenferro-tensor"):
                    (scratch / role / "crates" / crate).mkdir(parents=True)
            home = root / "home"
            cargo_home = root / "cargo-home"
            home.mkdir()
            cargo_home.mkdir()
            controlled_path = controlled_tool_path(root)
            tools = build.resolve_toolchain(controlled_path)
            config = build.BuildConfig(
                repository=repository,
                evidence_root=evidence,
                scratch_root=scratch,
                candidate_commit="c" * 40,
                path=controlled_path,
                home=home,
                cargo_home=cargo_home,
            )
            toolchain = {
                name: {
                    "path": str(getattr(tools, name).path),
                    "sha256": getattr(tools, name).sha256,
                    **({} if name == "git" else {"version": f"{name} 1.90.0"}),
                }
                for name in ("git", "cargo", "rustc")
            }
            manifests = {
                role: {
                    "role": role,
                    "head": ("c" if role == "candidate" else "d") * 40,
                    "toolchain": toolchain,
                }
                for role in build.BUILD_MANIFEST_PATHS
            }

            class ProbeBuildRunner:
                def __init__(self):
                    self.calls = []

                def __call__(self, argv, *, cwd, environment, deadline_seconds, **_kwargs):
                    argv = tuple(argv)
                    cwd = pathlib.Path(cwd)
                    self.calls.append(argv)
                    if len(argv) > 1 and argv[1] == "generate-lockfile":
                        (cwd / "Cargo.lock").write_bytes(
                            f"lock:{cwd.name}\n".encode()
                        )
                    elif len(argv) > 1 and argv[1] == "build":
                        binary = (
                            pathlib.Path(environment["CARGO_TARGET_DIR"])
                            / "release"
                            / build.ALLOCATION_PROBE_BINARY
                        )
                        binary.parent.mkdir(parents=True)
                        binary.write_bytes(f"binary:{cwd.parent.name}\n".encode())
                        binary.chmod(0o755)
                    stdout = (
                        json.dumps(list(protocol.CANONICAL_CASES), separators=(",", ":"))
                        + "\n"
                        if argv[0].endswith(build.ALLOCATION_PROBE_BINARY)
                        else ""
                    )
                    return build.CommandResult(
                        argv=argv,
                        cwd=str(cwd),
                        environment=dict(sorted(environment.items())),
                        deadline_seconds=deadline_seconds,
                        returncode=0,
                        stdout=stdout,
                        stderr="",
                        validity_state="COMPLETE",
                        failure_reason=None,
                        terminated=False,
                        killed=False,
                    )

            command_runner = ProbeBuildRunner()
            observed = build._build_allocation_probe_set_with_dependencies(
                config,
                manifests,
                tools=tools,
                command_runner=command_runner,
            )
            self.assertEqual(tuple(observed), tuple(build.BUILD_MANIFEST_PATHS))
            self.assertEqual(len(command_runner.calls), 8)
            self.assertEqual(
                sum(argv[1:2] == ("generate-lockfile",) for argv in command_runner.calls),
                2,
            )
            self.assertEqual(
                sum(argv[1:2] == ("--list-cases",) for argv in command_runner.calls),
                3,
            )
            self.assertEqual(
                observed["candidate"]["lock_sha256"],
                observed["common-lock-normalized-baseline"]["lock_sha256"],
            )
            self.assertNotEqual(
                observed["candidate"]["lock_sha256"],
                observed["direct-current-main-baseline"]["lock_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
