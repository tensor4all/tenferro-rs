#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check-public-error-docs.py")
SPEC = importlib.util.spec_from_file_location("check_public_error_docs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PublicErrorDocsTests(unittest.TestCase):
    def audit(self, source: str, filename: str = "sample.rs"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_text(source, encoding="utf-8")
            return MODULE.audit_file(path)

    def test_public_result_requires_errors_section(self) -> None:
        findings = self.audit(
            """
            /// Compute a value.
            pub fn compute() -> Result<(), MyError> { Ok(()) }
            """
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].reason, "missing # Errors")

    def test_errors_section_must_name_concrete_failure(self) -> None:
        findings = self.audit(
            """
            /// Compute a value.
            ///
            /// # Errors
            ///
            /// Returns an error when computation fails.
            pub fn compute() -> Result<(), MyError> { Ok(()) }
            """
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("concrete", findings[0].reason)

    def test_category_only_errors_section_is_not_concrete(self) -> None:
        findings = self.audit(
            """
            /// Compute a value.
            ///
            /// # Errors
            ///
            /// Returns a typed backend/runtime-state error.
            pub fn compute() -> Result<(), MyError> { Ok(()) }
            """
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("concrete", findings[0].reason)

    def test_validation_wrapper_without_payload_is_not_concrete(self) -> None:
        findings = self.audit(
            """
            /// Compute a value.
            ///
            /// # Errors
            ///
            /// Returns `Error::Validation` with a typed validation source.
            pub fn compute() -> Result<(), MyError> { Ok(()) }
            """
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("concrete", findings[0].reason)

    def test_trait_method_and_variant_are_accepted(self) -> None:
        findings = self.audit(
            """
            pub trait Compute {
                /// Compute a value.
                ///
                /// # Errors
                ///
            /// Returns `ValidationError::ShapeMismatch` for incompatible input.
                fn compute(&self) -> Result<(), Error>;
            }
            """
        )
        self.assertEqual(findings, [])

    def test_doc_attributes_are_treated_as_generated_rustdoc(self) -> None:
        findings = self.audit(
            r'''
            #[doc = "Register an extension."]
            #[doc = "\n# Errors\n\nReturns `Error::InvalidArgument` for an invalid family id."]
            pub fn register() -> Result<(), Error> { Ok(()) }
            '''
        )
        self.assertEqual(findings, [])

    def test_traced_deferred_validation_requires_deferred_section(self) -> None:
        findings = self.audit(
            """
            /// Build a symbolic operation.
            ///
            /// # Errors
            ///
            /// Returns `ValidationError::ShapeMismatch` when shape compatibility is checked
            /// during compilation or execution.
            pub fn build() -> Result<(), Error> { Ok(()) }
            """,
            filename="traced.rs",
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("Deferred errors", findings[0].reason)

    def test_traced_deferred_section_is_accepted(self) -> None:
        findings = self.audit(
            """
            /// Build a symbolic operation.
            ///
            /// # Errors
            ///
            /// Returns `ValidationError::ShapeMismatch` for an invalid shape.
            ///
            /// # Deferred errors
            ///
            /// A symbolic mismatch is reported at execution.
            pub fn build() -> Result<(), Error> { Ok(()) }
            """,
            filename="traced.rs",
        )
        self.assertEqual(findings, [])

    def test_non_result_function_is_not_a_finding(self) -> None:
        findings = self.audit(
            """
            /// Return a value.
            pub fn value() -> usize { 1 }
            """
        )
        self.assertEqual(findings, [])

    def test_changed_audit_requires_base_object_in_shallow_checkout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tenferro-error-docs-git-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()

            def git(*args: str, cwd: Path = source) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            git("init", "-b", "main")
            git("config", "user.name", "public-error-docs-test")
            git("config", "user.email", "public-error-docs-test@example.invalid")
            sample = source / "sample.rs"
            sample.write_text(
                """
                /// Compute a value.
                ///
                /// # Errors
                ///
                /// Returns `Error::InvalidArgument` when input is invalid.
                pub fn compute() -> Result<(), Error> { Ok(()) }
                """.lstrip(),
                encoding="utf-8",
            )
            git("add", "sample.rs")
            git("commit", "-m", "base")
            base = git("rev-parse", "HEAD").stdout.strip()

            sample.write_text(
                sample.read_text(encoding="utf-8").replace(
                    "Compute a value.", "Compute another value."
                ),
                encoding="utf-8",
            )
            git("add", "sample.rs")
            git("commit", "-m", "head")

            full = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root-dir",
                    str(source),
                    "--changed-from",
                    base,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(full.returncode, 0, full.stderr)

            shallow = root / "shallow"
            subprocess.run(
                ["git", "clone", "--depth", "1", source.as_uri(), str(shallow)],
                check=True,
                capture_output=True,
                text=True,
            )
            missing_base_diff = subprocess.run(
                ["git", "diff", "--name-only", f"{base}...HEAD", "--", "*.rs"],
                cwd=shallow,
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing_base_diff.returncode, 128, missing_base_diff.stderr)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root-dir",
                    str(shallow),
                    "--changed-from",
                    base,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("128", result.stderr)


if __name__ == "__main__":
    unittest.main()
