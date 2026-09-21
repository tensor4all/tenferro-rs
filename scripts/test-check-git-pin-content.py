#!/usr/bin/env python3
"""Focused tests for the git-pin content checker."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("check-git-pin-content.py")
SPEC = importlib.util.spec_from_file_location("check_git_pin_content", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

VCS_INFO = b'{"git": {"sha1": "0" * 40}, "path_in_vcs": "crates/fixture"}'


def pinned_package(
    files: dict[str, bytes],
    *,
    dependencies: dict | None = None,
    workspace_dependencies: dict | None = None,
) -> CHECKER.PinnedPackage:
    manifest = {
        "package": {"name": "fixture", "version": "0.4.0"},
        "dependencies": dependencies or {},
    }
    workspace_manifest = {"workspace": {"dependencies": workspace_dependencies or {}}}
    return CHECKER.PinnedPackage("fixture", "0.4.0", manifest, workspace_manifest, files)


def archive(files: dict[str, bytes]) -> dict[str, bytes]:
    """Cargo always packages these members; they must not affect comparison."""

    return {
        "Cargo.toml": b"# normalized\n",
        "Cargo.toml.orig": b"# original\n",
        ".cargo_vcs_info.json": VCS_INFO,
        **files,
    }


class VersionResolutionTests(unittest.TestCase):
    def test_exact_requirement_selects_only_that_version(self) -> None:
        self.assertEqual(
            CHECKER.effective_version("=0.2.0", ("0.2.0", "0.3.0", "0.1.9")), "0.2.0"
        )

    def test_exact_requirement_ignores_prereleases(self) -> None:
        # The registry carries 0.3.0-pre.3; an exact 0.2.0 pin must not accept it.
        self.assertEqual(
            CHECKER.effective_version("=0.2.0", ("0.2.0", "0.3.0-pre.3")), "0.2.0"
        )

    def test_caret_requirement_selects_highest_matching_release(self) -> None:
        # The strided-kernel incident: the pin declares 0.4.0 and resolves 0.4.1.
        self.assertEqual(
            CHECKER.effective_version("0.4.0", ("0.4.0", "0.4.1", "0.5.0")), "0.4.1"
        )

    def test_caret_requirement_respects_zero_minor_bounds(self) -> None:
        self.assertEqual(CHECKER.effective_version("0.0.3", ("0.0.3", "0.0.4")), "0.0.3")

    def test_prerelease_detection(self) -> None:
        self.assertTrue(CHECKER.is_prerelease("0.3.0-pre.3"))
        self.assertTrue(CHECKER.is_prerelease("0.3.0+build.1"))
        self.assertFalse(CHECKER.is_prerelease("0.3.0"))


class DependencyNormalizationTests(unittest.TestCase):
    def test_member_features_union_with_inherited_workspace_features(self) -> None:
        # t4a-cubek-matmul 0.2.1: `half = { workspace = true, features =
        # ["bytemuck"] }` publishes the union of the workspace and member lists.
        manifest = {"dependencies": {"half": {"workspace": True, "features": ["bytemuck"]}}}
        workspace = {
            "workspace": {
                "dependencies": {
                    "half": {
                        "version": "2.5",
                        "default-features": False,
                        "features": ["alloc", "num-traits", "serde"],
                    }
                }
            }
        }
        normalized = CHECKER.normalized_dependencies(manifest, workspace)
        self.assertEqual(
            normalized["dependencies:half"]["features"],
            ["alloc", "bytemuck", "num-traits", "serde"],
        )
        self.assertFalse(normalized["dependencies:half"]["default_features"])

    def test_string_workspace_entry_resolves(self) -> None:
        manifest = {"dependencies": {"num-traits": {"workspace": True}}}
        workspace = {"workspace": {"dependencies": {"num-traits": "0.2"}}}
        normalized = CHECKER.normalized_dependencies(manifest, workspace)
        self.assertEqual(normalized["dependencies:num-traits"]["version"], "0.2")

    def test_build_dependencies_are_compared(self) -> None:
        manifest = {"build-dependencies": {"cc": "1"}}
        self.assertIn("build-dependencies:cc", CHECKER.normalized_dependencies(manifest, {}))

    def test_dev_dependencies_are_not_compared(self) -> None:
        manifest = {"dev-dependencies": {"criterion": "0.5"}}
        self.assertEqual(CHECKER.normalized_dependencies(manifest, {}), {})

    def test_missing_inherited_entry_is_an_error(self) -> None:
        manifest = {"dependencies": {"absent": {"workspace": True}}}
        with self.assertRaises(CHECKER.PinContentError):
            CHECKER.normalized_dependencies(manifest, {"workspace": {"dependencies": {}}})


class ComparisonTests(unittest.TestCase):
    def compare(
        self,
        pinned_files: dict[str, bytes],
        archive_files: dict[str, bytes],
        **kwargs,
    ) -> list[CHECKER.Finding]:
        return CHECKER.compare_pin(
            pinned=pinned_package(pinned_files, **kwargs),
            version="0.4.0",
            archive_files=archive_files,
            archive_manifest={"package": {"name": "fixture"}, "dependencies": {}},
        )

    def test_matching_tree_reports_nothing(self) -> None:
        files = {"src/lib.rs": b"pub fn a() {}\n"}
        self.assertEqual(self.compare(files, archive(files)), [])

    def test_cargo_generated_and_unpackaged_files_are_ignored(self) -> None:
        pinned_files = {
            "src/lib.rs": b"pub fn a() {}\n",
            "benches/bench.rs": b"fn main() {}\n",
            "README.md": b"# fixture\n",
        }
        self.assertEqual(self.compare(pinned_files, archive({"src/lib.rs": b"pub fn a() {}\n"})), [])

    def test_source_file_list_mismatch_is_reported(self) -> None:
        # The strided-kernel 0.4.0 incident: the release has a different src set.
        pinned_files = {"src/lib.rs": b"pub use basic::*;\n", "src/erased.rs": b"//\n"}
        findings = self.compare(
            pinned_files,
            archive({"src/lib.rs": b"pub use basic::*;\n", "src/map_view.rs": b"//\n"}),
        )
        messages = " ".join(finding.message for finding in findings)
        self.assertTrue(all(finding.severity == "error" for finding in findings))
        self.assertIn("lacks pinned files", messages)
        self.assertIn("src/erased.rs", messages)
        self.assertIn("absent from the pinned revision", messages)
        self.assertIn("src/map_view.rs", messages)

    def test_source_content_mismatch_is_reported(self) -> None:
        # The strided-perm 0.4.0 incident: same file list, different content.
        findings = self.compare(
            {"src/plan.rs": b"if dim_a == dim_b || gapped {}\n"},
            archive({"src/plan.rs": b"if dim_a == dim_b {}\n"}),
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("content differs", findings[0].message)
        self.assertIn("src/plan.rs", findings[0].message)

    def test_build_script_is_compared(self) -> None:
        findings = self.compare(
            {"src/lib.rs": b"//\n", "build.rs": b"fn main() {}\n"},
            archive({"src/lib.rs": b"//\n"}),
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("build.rs", findings[0].message)

    def test_dependency_wiring_mismatch_is_reported(self) -> None:
        # The cubek-fft case: the registry package depends on other crates.
        findings = CHECKER.compare_pin(
            pinned=pinned_package(
                {"src/lib.rs": b"//\n"},
                dependencies={"cubecl": {"package": "t4a-cubecl", "version": "=0.10.1"}},
                workspace_dependencies={"cubecl": {"package": "t4a-cubecl", "version": "=0.10.1"}},
            ),
            version="0.2.0",
            archive_files=archive({"src/lib.rs": b"//\n"}),
            archive_manifest={
                "package": {"name": "fixture"},
                "dependencies": {"cubecl": {"version": "0.10.0"}},
            },
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("dependency wiring differs", findings[0].message)
        self.assertIn("dependencies:cubecl", findings[0].message)

    def test_duplicate_published_features_are_normalized(self) -> None:
        pinned_files = {"src/lib.rs": b"//\n"}
        findings = CHECKER.compare_pin(
            pinned=pinned_package(
                pinned_files,
                dependencies={"serde_bytes": {"version": "0.11", "features": ["alloc"]}},
            ),
            version="0.10.1",
            archive_files=archive(pinned_files),
            archive_manifest={
                "package": {"name": "fixture"},
                "dependencies": {"serde_bytes": {"version": "0.11", "features": ["alloc", "alloc"]}},
            },
        )
        self.assertEqual(findings, [])


class ExceptionTests(unittest.TestCase):
    def report(self, message: str) -> CHECKER.Report:
        return CHECKER.Report(
            "fixture", "=0.2.0", "0.2.0", "0.2.0", (CHECKER.Finding("error", message),)
        )

    def test_exception_downgrades_to_warning_with_issue(self) -> None:
        exceptions = {
            "fixture": CHECKER.ExceptionEntry(
                "fixture", "upstream owns the name", "https://example.invalid/1"
            )
        }
        errors, warnings = CHECKER.apply_exceptions([self.report("mismatch")], exceptions)
        self.assertEqual(errors, [])
        self.assertTrue(any("excepted" in warning.message for warning in warnings))
        self.assertTrue(any("https://example.invalid/1" in warning.message for warning in warnings))

    def test_unexcepted_mismatch_is_an_error(self) -> None:
        errors, warnings = CHECKER.apply_exceptions([self.report("mismatch")], {})
        self.assertEqual(len(errors), 1)
        self.assertEqual(warnings, [])

    def test_stale_exception_warns(self) -> None:
        exceptions = {"fixture": CHECKER.ExceptionEntry("fixture", "why", "issue")}
        errors, warnings = CHECKER.apply_exceptions(
            [CHECKER.Report("fixture", "0.2.0", "0.2.0", "0.2.0", ())], exceptions
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("stale" in warning.message for warning in warnings))

    def test_exception_for_unknown_package_is_an_error(self) -> None:
        exceptions = {"absent": CHECKER.ExceptionEntry("absent", "why", "issue")}
        errors, _ = CHECKER.apply_exceptions([], exceptions)
        self.assertEqual(len(errors), 1)
        self.assertIn("unknown package", errors[0].message)


class RepositoryConfigurationTests(unittest.TestCase):
    def test_repository_exceptions_are_justified(self) -> None:
        exceptions = CHECKER.load_exceptions(CHECKER.ROOT)
        for package, entry in exceptions.items():
            self.assertTrue(package)
            self.assertTrue(entry.reason.strip())
            self.assertTrue(entry.issue.startswith("https://github.com/"))


if __name__ == "__main__":
    unittest.main()
