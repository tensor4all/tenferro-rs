#!/usr/bin/env python3
"""Unit tests for `scripts/ci/check-twin-paths.py` on synthetic Rust snippets.

Each case builds a throwaway `crates/<crate>/src` tree so the assertions are
about the detector, not about the repository's current contents.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
CHECKER = SCRIPTS / "ci" / "check-twin-paths.py"

DUPLICATED = """
pub fn scaled_sum(values: &[f64], factor: f64) -> f64 {
    let mut total = 0.0;
    for value in values {
        let term = value * factor;
        if term.is_finite() {
            total += term;
        }
    }
    total
}
"""

BOILERPLATE = """
#[derive(Clone)]
pub struct Wrapper {
    pub value: u32,
}

impl Default for Wrapper {
    fn default() -> Self {
        let value = 0u32;
        let scaled = value as u64 * 2u64;
        Self { value: scaled as u32 }
    }
}
"""

MANIFEST = '[package]\nname = "{name}"\nversion = "0.0.0"\n'


def load_checker():
    spec = importlib.util.spec_from_file_location("check_twin_paths", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_twin_paths"] = module
    spec.loader.exec_module(module)
    return module


class TwinPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = load_checker()

    def tree(self, files: dict[str, str]) -> Path:
        root = tempfile.TemporaryDirectory()
        base = Path(root.name)
        crates = {relative.split("/")[1] for relative in files}
        for crate in sorted(crates):
            manifest = base / "crates" / crate / "Cargo.toml"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(MANIFEST.format(name=crate.replace("_", "-")))
        for relative, text in files.items():
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.addCleanup(root.cleanup)
        return base

    def run_checker(self, root: Path, *arguments: str) -> tuple[int, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = self.checker.main(["--root-dir", str(root), *arguments])
        return status, stdout.getvalue()

    # (a) an exact duplicate body is reported
    def test_exact_duplicate_body_is_reported(self):
        root = self.tree(
            {
                "crates/alpha/src/one.rs": DUPLICATED,
                "crates/beta/src/two.rs": DUPLICATED,
            }
        )
        report, _errors = self.checker.build_report(
            root, self.checker.ALLOWLIST
        )
        bodies = [finding for finding in report["findings"] if finding["kind"] == "body"]
        self.assertEqual(len(bodies), 1, report["findings"])
        self.assertEqual(bodies[0]["similarity"], 1.0)
        self.assertEqual(bodies[0]["left"]["name"], "scaled_sum")
        self.assertEqual(bodies[0]["right"]["name"], "scaled_sum")
        self.assertEqual(
            {bodies[0]["left"]["path"], bodies[0]["right"]["path"]},
            {"crates/alpha/src/one.rs", "crates/beta/src/two.rs"},
        )

    # (b) boilerplate trait impls and test items are not twins
    def test_boilerplate_and_test_items_are_ignored(self):
        root = self.tree(
            {
                "crates/alpha/src/one.rs": BOILERPLATE,
                "crates/beta/src/two.rs": BOILERPLATE,
                "crates/alpha/src/three.rs": "#[cfg(test)]\nmod tests {\n"
                + DUPLICATED
                + "}\n",
            }
        )
        report, _errors = self.checker.build_report(
            root, self.checker.ALLOWLIST
        )
        kinds = {finding["kind"] for finding in report["findings"]}
        self.assertNotIn("body", kinds, report["findings"])
        self.assertNotIn("dead", kinds, report["findings"])

    # (c) a `pub` function only its own unit test calls is a dead candidate;
    #     one an example calls is not
    def test_dead_twin_needs_no_consumer(self):
        dead = """
pub fn promote_like(lhs: bool, rhs: bool) -> bool {
    if lhs && rhs {
        return false;
    }
    lhs || rhs
}
"""
        used = """
pub fn documented_helper(lhs: bool, rhs: bool) -> bool {
    if lhs && rhs {
        return false;
    }
    lhs || rhs
}
"""
        root = self.tree(
            {
                "crates/alpha/src/one.rs": dead + used,
                "crates/alpha/src/tests.rs": "fn t() { promote_like(true, false); }\n",
                "crates/alpha/examples/demo.rs": "fn main() { documented_helper(true, false); }\n",
            }
        )
        report, _errors = self.checker.build_report(
            root, self.checker.ALLOWLIST
        )
        dead_names = {
            finding["left"]["name"]
            for finding in report["findings"]
            if finding["kind"] == "dead"
        }
        self.assertEqual(dead_names, {"promote_like"}, report["findings"])

    # (d) `--check` fails on an unlisted twin and passes once it is allowlisted
    def test_check_mode_respects_allowlist(self):
        root = self.tree(
            {
                "crates/alpha/src/one.rs": DUPLICATED,
                "crates/beta/src/two.rs": DUPLICATED,
                # One caller per crate keeps both copies off the `dead` list, so
                # this case is about the allowlist and nothing else.
                "crates/alpha/src/user.rs": (
                    "fn use_it(values: &[f64]) -> f64 {\n"
                    "    super::one::scaled_sum(values, 2.0)\n"
                    "}\n"
                ),
                "crates/beta/src/user.rs": (
                    "fn use_it(values: &[f64]) -> f64 {\n"
                    "    super::two::scaled_sum(values, 2.0)\n"
                    "}\n"
                ),
            }
        )
        allowlist = root / "allow.toml"
        status, _stdout = self.run_checker(root, "--check", "--allowlist", str(allowlist))
        self.assertEqual(status, 1)

        allowlist.write_text(
            "[[allow]]\n"
            'kind = "body"\n'
            'left = "crates/alpha/src/one.rs::scaled_sum"\n'
            'right = "crates/beta/src/two.rs::scaled_sum"\n'
            'reason = "both copies are generated from one template, reviewed by hand"\n'
        )
        status, stdout = self.run_checker(
            root, "--check", "--allowlist", str(allowlist), "--json"
        )
        self.assertEqual(status, 0, stdout)
        status, _stdout = self.run_checker(root, "--report")
        self.assertEqual(status, 0)

    def test_allowlist_entry_without_reason_is_an_error(self):
        root = self.tree({"crates/alpha/src/one.rs": DUPLICATED})
        allowlist = root / "allow.toml"
        allowlist.write_text(
            "[[allow]]\n"
            'kind = "body"\n'
            'left = "crates/alpha/src/one.rs::scaled_sum"\n'
            'right = "crates/beta/src/two.rs::scaled_sum"\n'
        )
        status, _stdout = self.run_checker(
            root, "--report", "--allowlist", str(allowlist)
        )
        self.assertEqual(status, 2)

    def test_renamed_import_counts_as_a_reference(self):
        caller = """
use crate::alpha_impl::helper as renamed_helper;

pub fn caller() -> u32 {
    renamed_helper(3u32)
}
"""
        root = self.tree(
            {
                "crates/alpha/src/one.rs": (
                    "pub fn helper(value: u32) -> u32 {\n"
                    "    let mut total = value;\n"
                    "    total += 1u32;\n"
                    "    total += 2u32;\n"
                    "    total\n"
                    "}\n"
                ),
                "crates/alpha/src/two.rs": caller,
            }
        )
        report, _errors = self.checker.build_report(
            root, self.checker.ALLOWLIST
        )
        dead_names = {
            finding["left"]["name"]
            for finding in report["findings"]
            if finding["kind"] == "dead"
        }
        self.assertNotIn("helper", dead_names, report["findings"])

    def test_lexer_survives_unicode_and_nested_comments(self):
        module = self.checker
        tokens, lines = module.lex(
            "/* outer /* inner */ */ let λ = 'x'; let β = r#\"raw\"#; // tail\n"
            "fn ok() { let s = \"\\\"escaped\\\"\"; }\n"
        )
        self.assertIn("fn", tokens)
        self.assertIn("ok", tokens)
        self.assertEqual(tokens.count("<literal>"), 3)
        self.assertEqual(lines[-1], 2)


if __name__ == "__main__":
    unittest.main()
