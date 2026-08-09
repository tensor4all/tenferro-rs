#!/usr/bin/env python3
"""Focused tests for the crates.io publish-layout checker."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check-publish-layout.py")
SPEC = importlib.util.spec_from_file_location("check_publish_layout", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


VALID_ALL_FEATURES = """\
[package]
name = "tenferro-fixture"
rust-version.workspace = true
keywords = ["tensor"]
categories = ["science"]
documentation = "https://docs.rs/tenferro-fixture"
[package.metadata.docs.rs]
rustdoc-args = ["--cfg", "docsrs"]
all-features = true

[features]
cpu-faer = []
autodiff = []
"""
VALID_EXPLICIT_FEATURES = VALID_ALL_FEATURES.replace(
    "all-features = true", 'features = ["cpu-faer", "autodiff"]'
)


class ReleaseOrderTests(unittest.TestCase):
    def metadata(self) -> dict:
        root = CHECKER.ROOT
        packages = [
            {
                "id": "core-id",
                "name": "tenferro-core-ops",
                "manifest_path": str(root / "crates/tenferro-core-ops/Cargo.toml"),
                "publish": None,
                "dependencies": [
                    {"name": "tenferro-runtime", "kind": "dev"},
                ],
            },
            {
                "id": "runtime-id",
                "name": "tenferro-runtime",
                "manifest_path": str(root / "crates/tenferro-runtime/Cargo.toml"),
                "publish": None,
                "dependencies": [
                    {"name": "tenferro-core-ops", "kind": None},
                    {"name": "tenferro-hidden", "kind": None},
                ],
            },
            {
                "id": "hidden-id",
                "name": "tenferro-hidden",
                "manifest_path": str(root / "crates/tenferro-hidden/Cargo.toml"),
                "publish": [],
                "dependencies": [],
            },
        ]
        return {
            "packages": packages,
            "workspace_members": [package["id"] for package in packages],
        }

    def release_text(self, order: list[str]) -> str:
        crates = "\n".join(order)
        return f"## Phase 3 — Publish From The Tag\n\n```text\n{crates}\n```\n"

    def test_accepts_topological_order_excluding_dev_and_nonpublishable(
        self,
    ) -> None:
        errors: list[str] = []
        CHECKER.check_release_order(
            self.metadata(),
            self.release_text(["tenferro-core-ops", "tenferro-runtime"]),
            errors,
        )
        self.assertEqual(errors, [])

    def test_rejects_dependency_after_dependent(self) -> None:
        errors: list[str] = []
        CHECKER.check_release_order(
            self.metadata(),
            self.release_text(["tenferro-runtime", "tenferro-core-ops"]),
            errors,
        )
        self.assertEqual(
            errors,
            [
                "release publish order must place dependency "
                "'tenferro-core-ops' before 'tenferro-runtime'"
            ],
        )

    def test_checks_optional_and_build_dependencies(self) -> None:
        cases = (
            {"name": "tenferro-core-ops", "kind": None, "optional": True},
            {"name": "tenferro-core-ops", "kind": "build", "optional": False},
        )
        for dependency in cases:
            with self.subTest(dependency=dependency):
                metadata = self.metadata()
                metadata["packages"][1]["dependencies"] = [dependency]
                errors: list[str] = []
                CHECKER.check_release_order(
                    metadata,
                    self.release_text(["tenferro-runtime", "tenferro-core-ops"]),
                    errors,
                )
                self.assertEqual(
                    errors,
                    [
                        "release publish order must place dependency "
                        "'tenferro-core-ops' before 'tenferro-runtime'"
                    ],
                )

    def test_rejects_missing_unexpected_and_duplicate_membership(self) -> None:
        cases = {
            "missing": (
                ["tenferro-core-ops"],
                "release publish order is missing crates: ['tenferro-runtime']",
            ),
            "unexpected": (
                ["tenferro-core-ops", "tenferro-runtime", "other"],
                "release publish order has unexpected crates: ['other']",
            ),
            "duplicate": (
                ["tenferro-core-ops", "tenferro-runtime", "tenferro-runtime"],
                "release publish order must not contain duplicate crates",
            ),
        }
        for name, (order, expected) in cases.items():
            with self.subTest(name=name):
                errors: list[str] = []
                CHECKER.check_release_order(
                    self.metadata(), self.release_text(order), errors
                )
                self.assertIn(expected, errors)

    def test_requires_exact_unique_phase_heading(self) -> None:
        valid = self.release_text(["tenferro-core-ops", "tenferro-runtime"])
        cases = (
            valid.replace(
                "## Phase 3 — Publish From The Tag",
                "## Phase 3 — Publish From The Tag Extra",
            ),
            valid + "\n## Phase 3 — Publish From The Tag\n",
        )
        for release_text in cases:
            with self.subTest(release_text=release_text):
                errors: list[str] = []
                CHECKER.check_release_order(self.metadata(), release_text, errors)
                self.assertEqual(
                    errors,
                    ["release workflow must contain exactly one exact Phase 3 heading"],
                )

    def test_stops_phase_at_the_next_heading(self) -> None:
        errors: list[str] = []
        release_text = self.release_text(
            ["tenferro-core-ops", "tenferro-runtime"]
        ) + "\n### Other Section\n\n```text\nunrelated\n```\n"
        CHECKER.check_release_order(self.metadata(), release_text, errors)
        self.assertEqual(errors, [])

    def test_ignores_heading_like_content_inside_non_order_fence(self) -> None:
        errors: list[str] = []
        release_text = """\
## Phase 3 — Publish From The Tag

```bash
# Build command
cargo build
```

```text
tenferro-core-ops
tenferro-runtime
```
"""
        CHECKER.check_release_order(self.metadata(), release_text, errors)
        self.assertEqual(errors, [])

    def test_rejects_missing_ambiguous_and_malformed_text_fences(self) -> None:
        valid = self.release_text(["tenferro-core-ops", "tenferro-runtime"])
        cases = (
            valid.replace("```text", "```bash"),
            valid + "\n```text\nunrelated\n```\n",
            valid.removesuffix("```\n"),
            valid + "\n```\n",
        )
        for release_text in cases:
            with self.subTest(release_text=release_text):
                errors: list[str] = []
                CHECKER.check_release_order(self.metadata(), release_text, errors)
                self.assertEqual(
                    errors,
                    [
                        "release workflow Phase 3 must contain exactly one "
                        "complete text fence"
                    ],
                )


class PublishMetadataTests(unittest.TestCase):
    def check(self, manifest: str) -> list[str]:
        errors: list[str] = []
        CHECKER.check_crate_metadata(
            "tenferro-fixture", manifest, errors, "fixture/Cargo.toml"
        )
        return errors

    def test_accepts_all_features_mode(self) -> None:
        self.assertEqual(self.check(VALID_ALL_FEATURES), [])

    def test_accepts_explicit_features_mode(self) -> None:
        self.assertEqual(self.check(VALID_EXPLICIT_FEATURES), [])

    def test_accepts_crates_io_keyword_syntax(self) -> None:
        manifest = VALID_ALL_FEATURES.replace(
            'keywords = ["tensor"]', 'keywords = ["123", "a_b", "a-b", "a+b"]'
        )
        self.assertEqual(self.check(manifest), [])

    def test_rejects_missing_rust_version_keywords_categories_and_docs_metadata(self) -> None:
        manifest = VALID_ALL_FEATURES.replace("rust-version.workspace = true\n", "")
        manifest = manifest.replace('keywords = ["tensor"]', "keywords = []")
        manifest = manifest.replace('categories = ["science"]', "categories = []")
        manifest = manifest.replace("[package.metadata.docs.rs]", "[package.metadata.missing]")
        errors = self.check(manifest)
        self.assertTrue(any("rust-version" in error for error in errors))
        self.assertTrue(any("keywords" in error for error in errors))
        self.assertTrue(any("categories" in error for error in errors))
        self.assertTrue(any("docs.rs" in error for error in errors))

    def test_rejects_invalid_keyword_and_category_counts(self) -> None:
        manifest = VALID_ALL_FEATURES.replace(
            'keywords = ["tensor"]',
            'keywords = ["a", "b", "c", "d", "e", "f"]',
        ).replace(
            'categories = ["science"]',
            'categories = ["science", "mathematics", "algorithms", "simulation", "data-structures", "no-std"]',
        )
        errors = self.check(manifest)
        self.assertTrue(any("keywords must contain 1-5" in error for error in errors))
        self.assertTrue(any("categories must contain 1-5" in error for error in errors))

    def test_rejects_invalid_keyword_syntax(self) -> None:
        cases = (
            '["tensor words"]',
            '["tensor!"]',
            '["_tensor"]',
            '["ténsor"]',
            '["abcdefghijklmnopqrstu"]',
        )
        for keywords in cases:
            with self.subTest(keywords=keywords):
                errors = self.check(VALID_ALL_FEATURES.replace('keywords = ["tensor"]', f"keywords = {keywords}"))
                self.assertTrue(any("keywords" in error and "syntax" in error for error in errors))

        errors = self.check(
            VALID_ALL_FEATURES.replace(
                'keywords = ["tensor"]', 'keywords = ["tensor", "tensor"]'
            )
        )
        self.assertTrue(any("keywords must be unique" in error for error in errors))

    def test_rejects_invalid_documentation_url(self) -> None:
        errors = self.check(
            VALID_ALL_FEATURES.replace(
                'documentation = "https://docs.rs/tenferro-fixture"',
                'documentation = "https://example.invalid/tenferro-fixture"',
            )
        )
        self.assertTrue(any("documentation" in error for error in errors))

    def test_rejects_invalid_docs_rs_modes(self) -> None:
        cases = (
            VALID_ALL_FEATURES.replace("all-features = true", ""),
            VALID_ALL_FEATURES.replace("all-features = true", "all-features = false"),
            VALID_ALL_FEATURES.replace("all-features = true", 'features = []'),
            VALID_ALL_FEATURES.replace("all-features = true", 'features = "cpu-faer"'),
            VALID_EXPLICIT_FEATURES.replace(
                'features = ["cpu-faer", "autodiff"]',
                'all-features = true\nfeatures = ["cpu-faer"]',
            ),
            VALID_EXPLICIT_FEATURES.replace(
                'features = ["cpu-faer", "autodiff"]',
                'all-features = false\nfeatures = ["cpu-faer"]',
            ),
            VALID_EXPLICIT_FEATURES.replace(
                'features = ["cpu-faer", "autodiff"]',
                'features = ["missing-feature"]',
            ),
        )
        for manifest in cases:
            with self.subTest(manifest=manifest):
                self.assertTrue(
                    any("docs.rs" in error for error in self.check(manifest))
                )

    def test_discovers_publishable_workspace_members_without_heuristics(
        self,
    ) -> None:
        metadata = {
            "workspace_members": ["publishable-id", "hidden-id"],
            "packages": [
                {
                    "id": "publishable-id",
                    "name": "future-crate",
                    "manifest_path": "/anywhere/future/Cargo.toml",
                    "publish": None,
                },
                {
                    "id": "hidden-id",
                    "name": "tenferro-hidden",
                    "manifest_path": "/crates/tenferro-hidden/Cargo.toml",
                    "publish": [],
                },
                {
                    "id": "nonmember-id",
                    "name": "tenferro-nonmember",
                    "manifest_path": str(
                        CHECKER.ROOT / "crates/tenferro-nonmember/Cargo.toml"
                    ),
                    "publish": None,
                },
            ],
        }
        self.assertEqual(CHECKER.publishable_crates(metadata), {"future-crate"})


if __name__ == "__main__":
    unittest.main()
