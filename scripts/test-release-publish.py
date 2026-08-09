#!/usr/bin/env python3
"""Focused tests for the fail-closed crates.io release helper."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("release-publish.py")
SPEC = importlib.util.spec_from_file_location("release_publish", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


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

    def archive(self, *, commit: str = COMMIT, dirty: bool = False) -> bytes:
        manifest = """\
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
        files = {
            "tenferro-fixture-0.4.0/Cargo.toml": manifest.encode(),
            "tenferro-fixture-0.4.0/README.md": b"fixture\n",
            "tenferro-fixture-0.4.0/src/lib.rs": b"",
            "tenferro-fixture-0.4.0/.cargo_vcs_info.json": json.dumps(
                {"git": {"sha1": commit, "dirty": dirty}}
            ).encode(),
        }
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return output.getvalue()

    def test_inspects_archive_metadata_files_and_exact_provenance(self) -> None:
        inspection = RELEASE.inspect_crate_archive(
            self.archive(), "tenferro-fixture", "0.4.0", self.COMMIT
        )
        self.assertEqual(inspection.metadata["rust-version"], "1.96")
        self.assertIn("tenferro-fixture-0.4.0/README.md", inspection.files)

    def test_rejects_dirty_or_wrong_commit_archive(self) -> None:
        for archive in (
            self.archive(dirty=True),
            self.archive(commit="0" * 40),
        ):
            with self.subTest():
                with self.assertRaises(RELEASE.ReleaseError):
                    RELEASE.inspect_crate_archive(
                        archive, "tenferro-fixture", "0.4.0", self.COMMIT
                    )


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

    def test_new_package_approval_must_name_exact_missing_package(self) -> None:
        existing = {"old-package": True, "new-package": False}
        with self.assertRaisesRegex(RELEASE.ReleaseError, "new-package"):
            RELEASE.validate_new_package_approvals(existing, set())
        RELEASE.validate_new_package_approvals(existing, {"new-package"})
        with self.assertRaisesRegex(RELEASE.ReleaseError, "not new"):
            RELEASE.validate_new_package_approvals(existing, {"old-package", "new-package"})


if __name__ == "__main__":
    unittest.main()
