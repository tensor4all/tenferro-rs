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

    def test_discovers_only_publishable_nested_crates(self) -> None:
        root = CHECKER.ROOT
        metadata = {
            "packages": [
                {
                    "name": "tenferro-fixture",
                    "manifest_path": str(root / "crates/tenferro-fixture/Cargo.toml"),
                    "publish": None,
                },
                {
                    "name": "tenferro-hidden",
                    "manifest_path": str(root / "crates/tenferro-hidden/Cargo.toml"),
                    "publish": [],
                },
                {
                    "name": "tenferro-nested",
                    "manifest_path": str(root / "crates/tenferro-nested/deep/Cargo.toml"),
                    "publish": None,
                },
            ]
        }
        self.assertEqual(CHECKER.publishable_crates(metadata), {"tenferro-fixture"})


if __name__ == "__main__":
    unittest.main()
