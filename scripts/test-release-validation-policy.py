#!/usr/bin/env python3
"""Focused tests for the release validation lane classifier."""

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("release-validation-policy.py")
SPEC = importlib.util.spec_from_file_location("release_validation_policy", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


MANIFEST = b"""\
[package]
name = "tenferro-fixture"
version = "0.3.0"
description = "Fixture"
license = "MIT OR Apache-2.0"
documentation = "https://docs.rs/tenferro-fixture"
readme = "README.md"
keywords = ["tensor"]
categories = ["science"]

[dependencies]
tenferro-tensor = { path = "../tenferro-tensor", version = "0.3.0" }

[features]
default = []
"""


class LaneClassificationTests(unittest.TestCase):
    def test_helper_only_changes_select_focused_lane(self) -> None:
        cases = (
            "scripts/release-publish.py",
            "scripts/check-publish-layout.py",
            "scripts/release-validation-policy.py",
            "scripts/test-release-publish.py",
            "scripts/ci/run_profile.py",
            "ai/contribution-workflows/release-publish.md",
            ".agents/skills/tenferro-release-publish/SKILL.md",
            ".claude/skills/tenferro-release-publish/SKILL.md",
            ".kimi/skills/tenferro-release-publish/SKILL.md",
            ".opencode/commands/tenferro-release-publish.md",
            ".github/workflows/ci-pr-workspace-tests.yml",
        )
        for path in cases:
            with self.subTest(path=path):
                self.assertEqual(
                    POLICY.classify_change(path, b"old", b"new"),
                    POLICY.LANE_HELPER,
                )

    def test_metadata_only_manifest_change_selects_metadata_lane(self) -> None:
        new = MANIFEST.replace(
            b'version = "0.3.0"\ndescription', b'version = "0.4.0"\ndescription'
        )
        self.assertEqual(
            POLICY.classify_change("crates/tenferro-fixture/Cargo.toml", MANIFEST, new),
            POLICY.LANE_METADATA,
        )
        described = MANIFEST.replace(
            b'description = "Fixture"', b'description = "Renamed fixture"'
        )
        self.assertEqual(
            POLICY.classify_change(
                "crates/tenferro-fixture/Cargo.toml", MANIFEST, described
            ),
            POLICY.LANE_METADATA,
        )

    def test_root_version_bump_is_metadata_only(self) -> None:
        old = b'[workspace]\nmembers = ["crates/a"]\n[workspace.package]\nversion = "0.3.0"\n'
        new = old.replace(b'"0.3.0"', b'"0.4.0"')
        self.assertEqual(
            POLICY.classify_change("Cargo.toml", old, new), POLICY.LANE_METADATA
        )

    def test_semantic_manifest_changes_select_semantic_lane(self) -> None:
        mutations = (
            (b'version = "0.3.0"', b'version = "0.3.1"'),  # dependency requirement
            (b'[features]\ndefault = []', b"[features]\ndefault = [\"cpu-faer\"]"),
            (b'tenferro-tensor = { path = "../tenferro-tensor", version = "0.3.0" }',
             b'tenferro-tensor = { path = "../tenferro-tensor", version = "0.3.0", default-features = false }'),
        )
        for old_fragment, new_fragment in mutations:
            with self.subTest(old_fragment=old_fragment):
                self.assertEqual(
                    POLICY.classify_change(
                        "crates/tenferro-fixture/Cargo.toml",
                        MANIFEST.replace(old_fragment, old_fragment),
                        MANIFEST.replace(old_fragment, new_fragment),
                    ),
                    POLICY.LANE_SEMANTIC,
                )

    def test_added_or_removed_manifest_is_semantic(self) -> None:
        self.assertEqual(
            POLICY.classify_change("crates/tenferro-new/Cargo.toml", None, MANIFEST),
            POLICY.LANE_SEMANTIC,
        )
        self.assertEqual(
            POLICY.classify_change("crates/tenferro-gone/Cargo.toml", MANIFEST, None),
            POLICY.LANE_SEMANTIC,
        )

    def test_malformed_manifest_is_semantic(self) -> None:
        self.assertEqual(
            POLICY.classify_change(
                "crates/tenferro-fixture/Cargo.toml", b"not = [toml", MANIFEST
            ),
            POLICY.LANE_SEMANTIC,
        )

    def test_rust_source_and_ambiguous_paths_select_full_lane(self) -> None:
        for path, old, new in (
            ("crates/tenferro-runtime/src/lib.rs", b"old", b"new"),
            ("crates/tenferro-xla/src/executor.rs", b"old", b"new"),
            ("docs/design/release.md", b"old", b"new"),
            ("REPOSITORY_RULES.md", b"old", b"new"),
            (".github/workflows/CI_gpu.yml", b"old", b"new"),
            ("scripts/test-unrelated-tooling.py", b"old", b"new"),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    POLICY.classify_change(path, old, new), POLICY.LANE_FULL
                )

    def test_mixed_change_sets_select_strongest_lane(self) -> None:
        helper = ("scripts/release-publish.py", b"old", b"new")
        metadata = (
            "crates/tenferro-fixture/Cargo.toml",
            MANIFEST,
            MANIFEST.replace(
                b'version = "0.3.0"\ndescription', b'version = "0.4.0"\ndescription'
            ),
        )
        semantic = (
            "crates/tenferro-fixture/Cargo.toml",
            MANIFEST,
            MANIFEST + b"\n[target.'cfg(unix)'.dependencies]\nlibc = \"0.2\"\n",
        )
        rust = ("crates/tenferro-runtime/src/lib.rs", b"old", b"new")
        self.assertEqual(
            POLICY.classify_changes([helper, metadata]), POLICY.LANE_METADATA
        )
        self.assertEqual(
            POLICY.classify_changes([helper, semantic]), POLICY.LANE_SEMANTIC
        )
        self.assertEqual(POLICY.classify_changes([helper, rust]), POLICY.LANE_FULL)
        self.assertEqual(
            POLICY.classify_changes([metadata, semantic, rust]), POLICY.LANE_FULL
        )

    def test_diff_classification_uses_injected_runner(self) -> None:
        files = {
            "scripts/release-publish.py": b"helper",
            "crates/tenferro-fixture/Cargo.toml": MANIFEST,
            "crates/tenferro-runtime/src/lib.rs": b"code",
        }
        calls: list[tuple[str, ...]] = []

        def runner(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess:
            calls.append(tuple(command))
            if command[:2] == ["git", "diff"]:
                return subprocess.CompletedProcess(
                    command, 0, "M\tscripts/release-publish.py\nM\tcrates/tenferro-fixture/Cargo.toml\nM\tcrates/tenferro-runtime/src/lib.rs\n", ""
                )
            revision, _, path = command[2].partition(":")
            content = files.get(path)
            if content is None:
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, content, "")

        lane = POLICY.classify_diff("BASE", "HEAD", runner=runner)
        self.assertEqual(lane, POLICY.LANE_FULL)
        self.assertIn(("git", "diff", "--name-status", "BASE", "HEAD"), calls)


if __name__ == "__main__":
    unittest.main()
