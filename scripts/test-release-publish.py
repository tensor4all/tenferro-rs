#!/usr/bin/env python3
"""Focused tests for the fail-closed crates.io release helper."""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import re
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("release-publish.py")
SPEC = importlib.util.spec_from_file_location("release_publish", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


class ReleaseDocumentationContractTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    SKILL_PATHS = (
        ".agents/skills/tenferro-release-publish/SKILL.md",
        ".claude/skills/tenferro-release-publish/SKILL.md",
        ".kimi/skills/tenferro-release-publish/SKILL.md",
    )

    def test_canonical_workflow_requires_semver_proposal_before_edits(self) -> None:
        text = (self.ROOT / "ai/contribution-workflows/release-publish.md").read_text()
        normalized = " ".join(text.split())
        for phrase in (
            "latest published stable version",
            "before changing manifests",
            "independently versioned",
            "stop for explicit confirmation and a reason",
            "unimplemented accepted issues do not affect",
            "Agents must stop after validation and must never execute a publication.",
        ):
            self.assertIn(phrase, normalized)
        self.assertIn(
            """| Baseline | `breaking` | `feature` | `fix-only` |
   | --- | --- | --- | --- |
   | `0.Y.Z` | `0.(Y+1).0` | `0.(Y+1).0` | `0.Y.(Z+1)` |
   | `X.Y.Z`, `X >= 1` | `(X+1).0.0` | `X.(Y+1).0` | `X.Y.(Z+1)` |""",
            text,
        )
        self.assertLess(
            normalized.index("provenance tag"), normalized.index("one proposed")
        )

    def test_release_adapters_reference_the_proposal_gate_and_human_boundary(self) -> None:
        paths = (*self.SKILL_PATHS, ".opencode/commands/tenferro-release-publish.md")
        for relative in paths:
            text = (self.ROOT / relative).read_text()
            normalized = " ".join(text.split())
            self.assertIn("ai/contribution-workflows/release-publish.md", text)
            self.assertIn("before editing", text)
            self.assertIn("SemVer proposal", text)
            self.assertIn(
                "stop after validation; a human maintainer runs Phase 3 publication from the tag.",
                normalized,
            )
            for contradictory in (
                "Phase 3: publish crates in dependency order from a worktree of the tag",
                "Proceed phase by phase — version-bump PR, tag, dependency-order publish from a worktree of the tag",
                "confirm with the user immediately before the first `cargo publish`",
            ):
                self.assertNotIn(contradictory.lower(), normalized.lower())

    def test_release_skill_adapters_are_byte_identical(self) -> None:
        expected = (self.ROOT / self.SKILL_PATHS[0]).read_bytes()
        for relative in self.SKILL_PATHS[1:]:
            self.assertEqual(expected, (self.ROOT / relative).read_bytes())


class GitDependencyTests(unittest.TestCase):
    def test_parses_commented_multiline_workspace_git_dependencies(self) -> None:
        manifest = """\
[workspace.dependencies]
local = { path = "local" }

# Cargo replaces this source with the registry version during publication.
[workspace.dependencies.strided]
package = "strided-view" # registry package
version = "0.4.0"
git = "https://example.invalid/strided.git"
rev = "0123456789abcdef0123456789abcdef01234567"
"""
        self.assertEqual(
            RELEASE.parse_workspace_git_dependencies(manifest),
            [
                RELEASE.GitDependency(
                    name="strided",
                    package="strided-view",
                    version="0.4.0",
                    git="https://example.invalid/strided.git",
                    rev="0123456789abcdef0123456789abcdef01234567",
                )
            ],
        )

    def test_accepts_cargo_exact_version_operator(self) -> None:
        manifest = """\
[workspace.dependencies.fixture]
git = "https://example.invalid/repo"
version = "=0.4.0"
rev = "0123456789abcdef0123456789abcdef01234567"
"""
        self.assertEqual(
            RELEASE.parse_workspace_git_dependencies(manifest)[0].version,
            "0.4.0",
        )

    def test_rejects_git_dependency_without_exact_registry_identity(self) -> None:
        base = """\
[workspace.dependencies]
strided = { git = "https://example.invalid/repo", %s }
"""
        cases = (
            'rev = "0123456789abcdef0123456789abcdef01234567"',
            'version = "^0.4", rev = "0123456789abcdef0123456789abcdef01234567"',
            'version = "0.4.0", rev = "main"',
        )
        for fields in cases:
            with self.subTest(fields=fields):
                with self.assertRaises(RELEASE.ReleaseError):
                    RELEASE.parse_workspace_git_dependencies(base % fields)

    def test_validates_package_and_workspace_version_at_pinned_revision(self) -> None:
        dependency = RELEASE.GitDependency(
            "view",
            "strided-view",
            "0.4.0",
            "https://example.invalid/strided.git",
            "0123456789abcdef0123456789abcdef01234567",
        )
        manifests = {
            "Cargo.toml": """\
[workspace]
members = ["strided-view"]
[workspace.package]
version = "0.4.0"
""",
            "strided-view/Cargo.toml": """\
[package]
name = "strided-view"
version.workspace = true
""",
        }
        RELEASE.validate_revision_manifest(dependency, manifests)
        with self.assertRaisesRegex(RELEASE.ReleaseError, "declares version 0.3.0"):
            RELEASE.validate_revision_manifest(
                dependency,
                {
                    "strided-view/Cargo.toml": """\
[package]
name = "strided-view"
version = "0.3.0"
"""
                },
            )


class ArchiveTests(unittest.TestCase):
    COMMIT = "abcdef0123456789abcdef0123456789abcdef01"
    ORIGINAL_MANIFEST = b"[package]\nname = \"tenferro-fixture\"\nversion.workspace = true\n"
    NORMALIZED_MANIFEST = b"""\
[package]
name = "tenferro-fixture"
version = "0.4.0"
description = "Fixture package"
license = "MIT OR Apache-2.0"
repository = "https://github.com/tensor4all/tenferro-rs"
homepage = "https://tensor4all.org/tenferro-rs/"
documentation = "https://docs.rs/tenferro-fixture"
readme = "README.md"
rust-version = "1.96"
keywords = ["tensor"]
categories = ["science"]
include = ["src/**", "README.md"]
"""

    def archive(
        self,
        *,
        commit: str = COMMIT,
        dirty: bool = False,
        changes: dict[str, bytes | None] | None = None,
        extra_members: tuple[tarfile.TarInfo, ...] = (),
    ) -> bytes:
        prefix = "tenferro-fixture-0.4.0/"
        files: dict[str, bytes] = {
            "Cargo.toml": self.NORMALIZED_MANIFEST,
            "Cargo.toml.orig": self.ORIGINAL_MANIFEST,
            "Cargo.lock": b"# generated\n",
            "README.md": b"fixture\n",
            "src/lib.rs": b"pub fn fixture() {}\n",
            ".cargo_vcs_info.json": json.dumps(
                {"git": {"sha1": commit, "dirty": dirty}}
            ).encode(),
        }
        for name, data in (changes or {}).items():
            if data is None:
                files.pop(name, None)
            else:
                files[name] = data
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(prefix + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            for info in extra_members:
                archive.addfile(info)
        return output.getvalue()

    @staticmethod
    def reader(path: str) -> bytes | None:
        return {
            "crates/tenferro-fixture/Cargo.toml": ArchiveTests.ORIGINAL_MANIFEST,
            "crates/tenferro-fixture/src/lib.rs": b"pub fn fixture() {}\n",
            "README.md": b"fixture\n",
        }.get(path)

    def inspect(
        self,
        archive: bytes,
        expected_contents: dict[str, bytes] | None = None,
    ) -> RELEASE.ArchiveInspection:
        return RELEASE.inspect_crate_archive(
            archive,
            "tenferro-fixture",
            "0.4.0",
            self.COMMIT,
            "crates/tenferro-fixture",
            self.reader,
            {
                ".cargo_vcs_info.json",
                "Cargo.lock",
                "Cargo.toml",
                "Cargo.toml.orig",
                "README.md",
                "src/lib.rs",
            },
            expected_contents,
        )

    def test_attests_metadata_generated_files_and_tagged_source_bytes(self) -> None:
        inspection = self.inspect(self.archive())
        self.assertEqual(inspection.metadata["rust-version"], "1.96")
        self.assertIn("tenferro-fixture-0.4.0/README.md", inspection.files)
        self.assertEqual(inspection.contents["Cargo.lock"], b"# generated\n")

    def test_registry_archive_must_match_local_generated_files_exactly(self) -> None:
        local = self.inspect(self.archive())
        mutations = (
            {
                "Cargo.toml": self.NORMALIZED_MANIFEST
                + b'\n[dependencies.injected]\nversion = "9"\n'
            },
            {"Cargo.lock": b"not valid = [toml"},
            {"Cargo.lock": b"# different but valid\nversion = 4\n"},
            {
                ".cargo_vcs_info.json": json.dumps(
                    {"git": {"sha1": self.COMMIT, "dirty": False}, "extra": True}
                ).encode()
            },
        )
        for changes in mutations:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(
                    RELEASE.ReleaseError, "differ.*exact local tagged archive"
                ):
                    self.inspect(
                        self.archive(changes=changes),
                        expected_contents=local.contents,
                    )

    def test_rejects_directory_traversal_before_skipping_directory(self) -> None:
        malicious = tarfile.TarInfo("tenferro-fixture-0.4.0/../escape")
        malicious.type = tarfile.DIRTYPE
        with self.assertRaisesRegex(RELEASE.ReleaseError, "invalid crate archive member path"):
            self.inspect(self.archive(extra_members=(malicious,)))

    def test_validates_directory_prefix_and_duplicates(self) -> None:
        outside = tarfile.TarInfo("outside")
        outside.type = tarfile.DIRTYPE
        duplicate = tarfile.TarInfo("tenferro-fixture-0.4.0/src")
        duplicate.type = tarfile.DIRTYPE
        cases = (
            ((outside,), "outside expected prefix"),
            ((duplicate, duplicate), "duplicate crate archive member path"),
        )
        for members, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RELEASE.ReleaseError, message):
                    self.inspect(self.archive(extra_members=members))

    def test_rejects_dirty_wrong_commit_tampered_unmapped_and_missing_source(self) -> None:
        cases = (
            self.archive(dirty=True),
            self.archive(commit="0" * 40),
            self.archive(changes={"src/lib.rs": b"tampered\n"}),
            self.archive(changes={"unexpected.bin": b"unmapped\n"}),
            self.archive(changes={"Cargo.toml.orig": None}),
            self.archive(changes={"README.md": None}),
        )
        for archive in cases:
            with self.subTest():
                with self.assertRaises(RELEASE.ReleaseError):
                    self.inspect(archive)

    def test_rejects_malformed_toml_json_and_archive(self) -> None:
        cases = (
            self.archive(changes={"Cargo.toml": b"not = [toml"}),
            self.archive(changes={".cargo_vcs_info.json": b"{"}),
            b"not a crate archive",
        )
        for archive in cases:
            with self.subTest():
                with self.assertRaisesRegex(RELEASE.ReleaseError, "invalid .crate archive"):
                    self.inspect(archive)


class RegistryAndApprovalTests(unittest.TestCase):
    def test_registry_queries_are_injected_and_do_not_need_network(self) -> None:
        calls: list[str] = []

        def transport(url: str) -> tuple[int, bytes]:
            calls.append(url)
            return (200, b"{}") if url.endswith("/strided-view/0.4.0") else (404, b"")

        client = RELEASE.CratesIoClient(transport)
        self.assertTrue(client.version_exists("strided-view", "0.4.0"))
        self.assertFalse(client.package_exists("new-package"))
        self.assertEqual(len(calls), 2)

    def test_default_transport_converts_incomplete_response_reads(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.side_effect = http.client.IncompleteRead(b"partial", 10)
        with mock.patch.object(RELEASE.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RELEASE.ReleaseError, "crates.io request failed"):
                RELEASE.crates_io_transport("https://example.invalid/archive")

    def test_new_package_approval_supports_exact_published_resume_only(self) -> None:
        packages = {"new": False, "resumed": True, "stale": True}
        versions = {"new": False, "resumed": True, "stale": False}
        with self.assertRaisesRegex(RELEASE.ReleaseError, "new"):
            RELEASE.validate_new_package_approvals(packages, versions, set())
        RELEASE.validate_new_package_approvals(packages, versions, {"new", "resumed"})
        with self.assertRaisesRegex(RELEASE.ReleaseError, "target version does not exist"):
            RELEASE.validate_new_package_approvals(packages, versions, {"new", "stale"})

    def test_transport_failures_are_actionable(self) -> None:
        failures = (
            TimeoutError("offline"),
            http.client.IncompleteRead(b"partial", 10),
            http.client.HTTPException("broken response"),
        )
        for failure in failures:
            with self.subTest(failure=failure):
                def transport(_url: str) -> tuple[int, bytes]:
                    raise failure

                with self.assertRaisesRegex(RELEASE.ReleaseError, "crates.io request failed"):
                    RELEASE.CratesIoClient(transport).package_exists("fixture")


class CommandAndCheckoutTests(unittest.TestCase):
    def test_command_failures_are_converted_and_timeout_is_bounded(self) -> None:
        with mock.patch.object(
            RELEASE.subprocess, "run", side_effect=FileNotFoundError("missing")
        ):
            with self.assertRaisesRegex(RELEASE.ReleaseError, "command not found"):
                RELEASE.run(["missing-command"])
        timeout = subprocess.TimeoutExpired(["cargo", "package"], 1)
        with mock.patch.object(RELEASE.subprocess, "run", side_effect=timeout):
            with self.assertRaisesRegex(RELEASE.ReleaseError, "timed out"):
                RELEASE.run(["cargo", "package"], timeout=1)

    def test_checkout_requires_detached_exact_remote_tag_on_main(self) -> None:
        commit = "a" * 40
        calls: list[tuple[str, ...]] = []

        def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(tuple(command))
            outputs = {
                ("git", "status", "--porcelain"): "",
                ("git", "ls-remote", "--tags", "origin", "refs/tags/v0.4.0", "refs/tags/v0.4.0^{}"): f"{commit}\trefs/tags/v0.4.0\n",
                ("git", "rev-parse", "HEAD"): commit + "\n",
            }
            if command[:4] == ["git", "symbolic-ref", "-q", "HEAD"]:
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, outputs.get(tuple(command), ""), "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text(
                '[workspace.package]\nversion = "0.4.0"\n', encoding="utf-8"
            )
            self.assertEqual(
                RELEASE.verify_release_checkout("0.4.0", runner=runner, root=root),
                commit,
            )
        self.assertIn(("git", "merge-base", "--is-ancestor", commit, "origin/main"), calls)

    def test_checkout_rejects_attached_head(self) -> None:
        def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            stdout = "refs/heads/main\n" if command[1] == "symbolic-ref" else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RELEASE.ReleaseError, "detached"):
                RELEASE.verify_release_checkout("0.4.0", runner=runner, root=Path(directory))

    def test_checkout_rejects_wrong_remote_tag(self) -> None:
        commit = "a" * 40

        def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if command[1] == "symbolic-ref":
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[1] == "ls-remote":
                return subprocess.CompletedProcess(
                    command, 0, f"{commit}\trefs/tags/v0.4.0\n", ""
                )
            if command[1:] == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, "b" * 40 + "\n", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RELEASE.ReleaseError, "not exact pushed remote tag"):
                RELEASE.verify_release_checkout("0.4.0", runner=runner, root=Path(directory))

    def test_cargo_metadata_rejects_malformed_json(self) -> None:
        runner = lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "{", ""
        )
        with self.assertRaisesRegex(RELEASE.ReleaseError, "invalid JSON"):
            RELEASE.cargo_metadata(runner=runner)

    def test_cli_selects_release_root_and_defaults_to_script_root(self) -> None:
        with mock.patch.object(RELEASE, "publish_release") as publish:
            self.assertEqual(RELEASE.main(["0.4.0"]), 0)
            self.assertEqual(publish.call_args.kwargs["root"], RELEASE.ROOT)

            selected = Path("tag-checkout")
            self.assertEqual(
                RELEASE.main(["0.4.0", "--release-root", str(selected)]), 0
            )
            self.assertEqual(publish.call_args.kwargs["root"], selected.resolve())


class FakeClient:
    def __init__(self, package_versions: dict[str, bool], events: list[str]) -> None:
        self.package_versions = package_versions
        self.events = events

    def package_exists(self, package: str) -> bool:
        return package in self.package_versions

    def version_exists(self, package: str, _version: str) -> bool:
        self.events.append(f"exists:{package}")
        return self.package_versions.get(package, False)

    def download(self, package: str, _version: str) -> bytes:
        self.events.append(f"download:{package}")
        return package.encode()


class OrchestrationTests(unittest.TestCase):
    VERSION = "0.4.0"
    COMMIT = "a" * 40

    @staticmethod
    def metadata(root: Path, crates: list[str]) -> dict:
        return {
            "target_directory": str(root / "target"),
            "packages": [
                {
                    "name": crate,
                    "version": "0.4.0",
                    "manifest_path": str(root / "crates" / crate / "Cargo.toml"),
                    "dependencies": [],
                }
                for crate in crates
            ],
        }

    @staticmethod
    def workspace(root: Path) -> None:
        (root / "Cargo.toml").write_text("[workspace.dependencies]\n", encoding="utf-8")
        workflow = root / "ai/contribution-workflows/release-publish.md"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("unused", encoding="utf-8")

    def test_exact_package_inspect_dry_run_publish_wait_order(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            metadata = self.metadata(root, ["crate-a"])
            archive = root / "target/package/crate-a-0.4.0.crate"
            client = FakeClient({}, events)

            def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                if command[:2] == ["cargo", "package"]:
                    events.append("package")
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    archive.write_bytes(b"package")
                elif command[:3] == ["cargo", "publish", "--dry-run"]:
                    events.append("dry-run")
                    for staging in ("tmp-crate", "tmp-registry"):
                        staged = archive.parent / staging / archive.name
                        staged.parent.mkdir()
                        staged.write_bytes(b"dry-run")
                elif command[:2] == ["cargo", "publish"]:
                    events.append("publish")
                    client.package_versions["crate-a"] = True
                return subprocess.CompletedProcess(command, 0, "", "")

            def inspect(data: bytes, *args: object) -> RELEASE.ArchiveInspection:
                events.append(f"inspect:{data.decode()}")
                expected_contents = args[-1]
                if expected_contents is not None:
                    self.assertEqual(expected_contents, {"Cargo.toml": b"generated"})
                return RELEASE.ArchiveInspection(
                    {}, (), {"Cargo.toml": b"generated"}
                )

            RELEASE.publish_release(
                self.VERSION,
                {"crate-a"},
                True,
                root=root,
                runner=runner,
                client=client,
                checkout_verifier=lambda *_args, **_kwargs: self.COMMIT,
                metadata_loader=lambda **_kwargs: metadata,
                order_loader=lambda _text: ["crate-a"],
                package_files_loader=lambda *_args, **_kwargs: {"Cargo.toml"},
                archive_inspector=inspect,
                source_reader_factory=lambda *_args, **_kwargs: lambda _path: None,
                attempts=1,
                delay=0,
            )
        self.assertEqual(
            events,
            [
                "exists:crate-a",
                "package",
                "inspect:package",
                "dry-run",
                "inspect:dry-run",
                "publish",
                "exists:crate-a",
                "download:crate-a",
                "inspect:crate-a",
            ],
        )

    def assert_runtime_bootstrap_commands(self, runtime_exists: bool) -> None:
        cargo_commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'tag "checkout\\special'
            root.mkdir()
            self.workspace(root)
            metadata = self.metadata(root, ["tenferro-runtime", "tenferro-cpu"])
            metadata["packages"][0]["dependencies"] = [
                {"name": "tenferro-cpu", "kind": "dev"}
            ]
            metadata["packages"][1]["dependencies"] = [
                {"name": "tenferro-runtime", "kind": "normal"}
            ]
            client = FakeClient(
                {"tenferro-runtime": runtime_exists, "tenferro-cpu": False}, []
            )

            def runner(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if command[0] == "cargo":
                    cargo_commands.append(command)
                if command[:2] == ["cargo", "package"]:
                    crate = command[command.index("-p") + 1]
                    archive = root / f"target/package/{crate}-0.4.0.crate"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    archive.write_bytes(crate.encode())
                elif command[:3] == ["cargo", "publish", "--dry-run"]:
                    crate = command[command.index("-p") + 1]
                    archive = root / f"target/package/{crate}-0.4.0.crate"
                    staged = archive.parent / "tmp-crate" / archive.name
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_bytes(crate.encode())
                elif command[:2] == ["cargo", "publish"]:
                    crate = command[command.index("-p") + 1]
                    client.package_versions[crate] = True
                return subprocess.CompletedProcess(command, 0, "", "")

            RELEASE.publish_release(
                self.VERSION,
                set(),
                True,
                root=root,
                runner=runner,
                client=client,
                checkout_verifier=lambda *_args, **_kwargs: self.COMMIT,
                metadata_loader=lambda **_kwargs: metadata,
                order_loader=lambda _text: ["tenferro-runtime", "tenferro-cpu"],
                package_files_loader=lambda *_args, **_kwargs: {"Cargo.toml"},
                archive_inspector=lambda *_args: RELEASE.ArchiveInspection(
                    {}, (), {"Cargo.toml": b"generated"}
                ),
                source_reader_factory=lambda *_args, **_kwargs: lambda _path: None,
                attempts=1,
                delay=0,
            )

        patch = (
            "patch.crates-io.tenferro-cpu.path="
            + json.dumps(str((root / "crates/tenferro-cpu").resolve()))
        )
        bootstrap = ["--no-verify", "--config", patch]
        self.assertEqual(
            cargo_commands,
            [
                ["cargo", "package", "-p", "tenferro-runtime", "--locked", *bootstrap],
                *(
                    []
                    if runtime_exists
                    else [
                        [
                            "cargo",
                            "publish",
                            "--dry-run",
                            "-p",
                            "tenferro-runtime",
                            "--locked",
                            "--registry",
                            "crates-io",
                            *bootstrap,
                        ],
                        [
                            "cargo",
                            "publish",
                            "-p",
                            "tenferro-runtime",
                            "--locked",
                            "--registry",
                            "crates-io",
                            *bootstrap,
                        ],
                    ]
                ),
                ["cargo", "package", "-p", "tenferro-cpu", "--locked"],
                [
                    "cargo",
                    "publish",
                    "--dry-run",
                    "-p",
                    "tenferro-cpu",
                    "--locked",
                    "--registry",
                    "crates-io",
                ],
                [
                    "cargo",
                    "publish",
                    "-p",
                    "tenferro-cpu",
                    "--locked",
                    "--registry",
                    "crates-io",
                ],
            ],
        )

    def test_runtime_bootstrap_patches_tag_cpu_while_both_are_missing(self) -> None:
        self.assert_runtime_bootstrap_commands(runtime_exists=False)

    def test_runtime_bootstrap_patches_tag_cpu_on_restart_after_runtime_publish(self) -> None:
        self.assert_runtime_bootstrap_commands(runtime_exists=True)

    def assert_xla_bootstrap_commands(self, xla_exists: bool) -> None:
        cargo_commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'tag "checkout\\special'
            root.mkdir()
            self.workspace(root)
            crates = ["tenferro-xla", "tenferro-einsum", "tenferro-cpu"]
            metadata = self.metadata(root, crates)
            metadata["packages"][0]["dependencies"] = [
                {"name": "tenferro-einsum", "kind": "dev"}
            ]
            client = FakeClient(
                {
                    "tenferro-xla": xla_exists,
                    "tenferro-einsum": False,
                    "tenferro-cpu": True,
                },
                [],
            )

            def runner(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if command[0] == "cargo":
                    cargo_commands.append(command)
                if command[:2] == ["cargo", "package"]:
                    crate = command[command.index("-p") + 1]
                    archive = root / f"target/package/{crate}-0.4.0.crate"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    archive.write_bytes(crate.encode())
                elif command[:3] == ["cargo", "publish", "--dry-run"]:
                    crate = command[command.index("-p") + 1]
                    archive = root / f"target/package/{crate}-0.4.0.crate"
                    staged = archive.parent / "tmp-crate" / archive.name
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_bytes(crate.encode())
                elif command[:2] == ["cargo", "publish"]:
                    crate = command[command.index("-p") + 1]
                    client.package_versions[crate] = True
                return subprocess.CompletedProcess(command, 0, "", "")

            RELEASE.publish_release(
                self.VERSION,
                set(),
                True,
                root=root,
                runner=runner,
                client=client,
                checkout_verifier=lambda *_args, **_kwargs: self.COMMIT,
                metadata_loader=lambda **_kwargs: metadata,
                order_loader=lambda _text: crates,
                package_files_loader=lambda *_args, **_kwargs: {"Cargo.toml"},
                archive_inspector=lambda *_args: RELEASE.ArchiveInspection(
                    {}, (), {"Cargo.toml": b"generated"}
                ),
                source_reader_factory=lambda *_args, **_kwargs: lambda _path: None,
                attempts=1,
                delay=0,
            )

        patch = (
            "patch.crates-io.tenferro-einsum.path="
            + json.dumps(str((root / "crates/tenferro-einsum").resolve()))
        )
        bootstrap = ["--no-verify", "--config", patch]
        xla_commands = [command for command in cargo_commands if "tenferro-xla" in command]
        self.assertEqual(
            xla_commands,
            [
                ["cargo", "package", "-p", "tenferro-xla", "--locked", *bootstrap],
                *(
                    []
                    if xla_exists
                    else [
                        [
                            "cargo",
                            "publish",
                            "--dry-run",
                            "-p",
                            "tenferro-xla",
                            "--locked",
                            "--registry",
                            "crates-io",
                            *bootstrap,
                        ],
                        [
                            "cargo",
                            "publish",
                            "-p",
                            "tenferro-xla",
                            "--locked",
                            "--registry",
                            "crates-io",
                            *bootstrap,
                        ],
                    ]
                ),
            ],
        )
        self.assertTrue(any("tenferro-cpu" in command for command in cargo_commands))
        self.assertTrue(any("tenferro-einsum" in command for command in cargo_commands))
        for command in cargo_commands:
            if "tenferro-xla" not in command:
                self.assertNotIn("--no-verify", command)
                self.assertNotIn(patch, command)

    def test_xla_bootstrap_patches_tag_einsum_while_both_are_missing(self) -> None:
        self.assert_xla_bootstrap_commands(xla_exists=False)

    def test_xla_bootstrap_patches_tag_einsum_on_restart_after_xla_publish(self) -> None:
        self.assert_xla_bootstrap_commands(xla_exists=True)

    def test_rejects_differing_dry_run_staging_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.metadata(root, ["crate-a"])
            archive = root / "target/package/crate-a-0.4.0.crate"

            def runner(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                archives = (("tmp-crate", b"first"), ("tmp-registry", b"second"))
                for staging, contents in archives:
                    staged = archive.parent / staging / archive.name
                    staged.parent.mkdir(parents=True)
                    staged.write_bytes(contents)
                return subprocess.CompletedProcess(command, 0, "", "")

            with self.assertRaisesRegex(RELEASE.ReleaseError, "differing archives"):
                RELEASE.build_and_inspect_archive(
                    metadata,
                    "crate-a",
                    self.VERSION,
                    self.COMMIT,
                    "crates/crate-a",
                    lambda _path: None,
                    {"Cargo.toml"},
                    ["cargo", "publish", "--dry-run"],
                    runner=runner,
                    root=root,
                )

    def test_archive_status_failure_names_failing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.metadata(root, ["crate-a"])
            archive = root / "target/package/crate-a-0.4.0.crate"
            failing_archive = archive.parent / "tmp-crate" / archive.name

            def runner(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                archive.parent.mkdir(parents=True)
                archive.write_bytes(b"valid")
                failing_archive.parent.mkdir()
                failing_archive.write_bytes(b"same")
                return subprocess.CompletedProcess(command, 0, "", "")

            original_stat = Path.stat

            def stat(path: Path, *, follow_symlinks: bool = True) -> object:
                if path == failing_archive:
                    raise OSError("test status failure")
                return original_stat(path, follow_symlinks=follow_symlinks)

            with mock.patch.object(Path, "stat", stat):
                with self.assertRaisesRegex(
                    RELEASE.ReleaseError, re.escape(str(failing_archive))
                ):
                    RELEASE.build_and_inspect_archive(
                        metadata,
                        "crate-a",
                        self.VERSION,
                        self.COMMIT,
                        "crates/crate-a",
                        lambda _path: None,
                        {"Cargo.toml"},
                        ["cargo", "publish", "--dry-run"],
                        runner=runner,
                        root=root,
                        archive_inspector=lambda *_args: RELEASE.ArchiveInspection(
                            {}, (), {}
                        ),
                    )

    def test_archive_read_failure_names_failing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.metadata(root, ["crate-a"])
            archive = root / "target/package/crate-a-0.4.0.crate"
            failing_archive = archive.parent / "tmp-registry" / archive.name

            def runner(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                for staging in ("tmp-crate", "tmp-registry"):
                    staged = archive.parent / staging / archive.name
                    staged.parent.mkdir(parents=True)
                    staged.write_bytes(b"same")
                return subprocess.CompletedProcess(command, 0, "", "")

            original_read_bytes = Path.read_bytes

            def read_bytes(path: Path) -> bytes:
                if path == failing_archive:
                    raise OSError("test read failure")
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", read_bytes):
                with self.assertRaisesRegex(
                    RELEASE.ReleaseError, re.escape(str(failing_archive))
                ):
                    RELEASE.build_and_inspect_archive(
                        metadata,
                        "crate-a",
                        self.VERSION,
                        self.COMMIT,
                        "crates/crate-a",
                        lambda _path: None,
                        {"Cargo.toml"},
                        ["cargo", "publish", "--dry-run"],
                        runner=runner,
                        root=root,
                    )

    def test_resume_accepts_new_package_approval_and_attests_before_skip(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            metadata = self.metadata(root, ["new-crate"])
            archive = root / "target/package/new-crate-0.4.0.crate"

            def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                if command[:2] == ["cargo", "package"]:
                    events.append("package")
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    archive.write_bytes(b"exact-tag")
                return subprocess.CompletedProcess(command, 0, "", "")

            def inspect(data: bytes, *args: object) -> RELEASE.ArchiveInspection:
                expected_contents = args[-1]
                events.append(f"inspect:{data.decode()}")
                if data == b"new-crate":
                    self.assertEqual(
                        expected_contents, {"Cargo.toml": b"exact-tag-generated"}
                    )
                else:
                    self.assertIsNone(expected_contents)
                return RELEASE.ArchiveInspection(
                    {}, (), {"Cargo.toml": b"exact-tag-generated"}
                )

            RELEASE.publish_release(
                self.VERSION,
                {"new-crate"},
                False,
                root=root,
                runner=runner,
                client=FakeClient({"new-crate": True}, events),
                checkout_verifier=lambda *_args, **_kwargs: self.COMMIT,
                metadata_loader=lambda **_kwargs: metadata,
                order_loader=lambda _text: ["new-crate"],
                package_files_loader=lambda *_args, **_kwargs: {"Cargo.toml"},
                archive_inspector=inspect,
                source_reader_factory=lambda *_args, **_kwargs: lambda _path: None,
                attempts=1,
                delay=0,
            )
        self.assertEqual(
            events,
            [
                "exists:new-crate",
                "package",
                "inspect:exact-tag",
                "download:new-crate",
                "inspect:new-crate",
            ],
        )

    def test_resume_packages_only_after_prerequisite_archive_is_attested(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            metadata = self.metadata(root, ["crate-a", "crate-b"])
            metadata["packages"][1]["dependencies"] = [
                {"name": "crate-a", "kind": "normal"}
            ]

            def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                if command[:2] == ["cargo", "package"]:
                    crate = command[command.index("-p") + 1]
                    events.append(f"package:{crate}")
                    archive = root / f"target/package/{crate}-0.4.0.crate"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    archive.write_bytes(f"local:{crate}".encode())
                return subprocess.CompletedProcess(command, 0, "", "")

            def inspect(data: bytes, *args: object) -> RELEASE.ArchiveInspection:
                crate = args[0]
                expected_contents = args[-1]
                events.append(f"inspect:{data.decode()}")
                contents = {"Cargo.toml": f"generated:{crate}".encode()}
                if data.startswith(b"crate-"):
                    self.assertEqual(expected_contents, contents)
                return RELEASE.ArchiveInspection({}, (), contents)

            RELEASE.publish_release(
                self.VERSION,
                set(),
                False,
                root=root,
                runner=runner,
                client=FakeClient({"crate-a": True, "crate-b": True}, events),
                checkout_verifier=lambda *_args, **_kwargs: self.COMMIT,
                metadata_loader=lambda **_kwargs: metadata,
                order_loader=lambda _text: ["crate-a", "crate-b"],
                package_files_loader=lambda *_args, **_kwargs: {"Cargo.toml"},
                archive_inspector=inspect,
                source_reader_factory=lambda *_args, **_kwargs: lambda _path: None,
                attempts=1,
                delay=0,
            )

        prerequisite_attestations = [
            index for index, event in enumerate(events) if event == "inspect:crate-a"
        ]
        self.assertEqual(len(prerequisite_attestations), 2)
        self.assertLess(prerequisite_attestations[-1], events.index("package:crate-b"))

    def test_propagation_retries_then_succeeds(self) -> None:
        events: list[str] = []

        class PropagatingClient(FakeClient):
            def __init__(self) -> None:
                super().__init__({"crate-a": True}, events)
                self.checks = 0

            def version_exists(self, package: str, version: str) -> bool:
                self.checks += 1
                events.append(f"exists:{package}")
                return self.checks == 3

        inspection = RELEASE.await_registry_archive(
            PropagatingClient(),
            "crate-a",
            self.VERSION,
            self.COMMIT,
            "crates/crate-a",
            lambda _path: None,
            {"Cargo.toml"},
            {"Cargo.toml": b"generated"},
            archive_inspector=lambda *_args: RELEASE.ArchiveInspection(
                {}, (), {"Cargo.toml": b"generated"}
            ),
            attempts=3,
            delay=0,
            sleeper=lambda _delay: events.append("sleep"),
        )
        self.assertEqual(
            inspection,
            RELEASE.ArchiveInspection({}, (), {"Cargo.toml": b"generated"}),
        )
        self.assertEqual(events.count("sleep"), 2)
        self.assertEqual(events[-1], "download:crate-a")

    def test_propagation_retries_then_times_out(self) -> None:
        events: list[str] = []
        client = FakeClient({}, events)
        with self.assertRaisesRegex(RELEASE.ReleaseError, "did not become visible"):
            RELEASE.await_registry_archive(
                client,
                "crate-a",
                self.VERSION,
                self.COMMIT,
                "crates/crate-a",
                lambda _path: None,
                {"Cargo.toml"},
                {"Cargo.toml": b"generated"},
                archive_inspector=lambda *_args: RELEASE.ArchiveInspection(
                    {}, (), {"Cargo.toml": b"generated"}
                ),
                attempts=3,
                delay=0,
                sleeper=lambda _delay: events.append("sleep"),
            )
        self.assertEqual(events.count("sleep"), 2)


if __name__ == "__main__":
    unittest.main()
