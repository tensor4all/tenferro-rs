#!/usr/bin/env python3
"""Unit tests for `scripts/ci/check-duplicate-test-messages.py`.

Each case builds a throwaway `crates/<crate>/...` tree so the assertions are
about the detector, not about the repository's current contents.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
CHECKER = SCRIPTS / "ci" / "check-duplicate-test-messages.py"

SENTENCE = "a preset dtype always has the requested representation"


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_duplicate_test_messages", CHECKER
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_duplicate_test_messages"] = module
    spec.loader.exec_module(module)
    return module


CHECKER_MODULE = load_checker()


class Repository:
    """A temporary repository holding only the files a case needs."""

    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def close(self) -> None:
        self.directory.cleanup()

    def write(self, relative: str, body: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def allowlist(self, body: str) -> Path:
        path = self.root / "allowlist.toml"
        path.write_text(body, encoding="utf-8")
        return path


def test_file(message: str) -> str:
    return f'pub fn case() {{\n    value().expect("{message}");\n}}\n'


def integration_path(crate: str, name: str) -> str:
    return f"crates/{crate}/tests/integration/{name}.rs"


class DuplicateTestMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Repository()
        self.addCleanup(self.repo.close)

    def report(self, max_files: int = 2):
        return CHECKER_MODULE.build_report(
            self.repo.root, Path("unused-allowlist.toml"), max_files
        )

    def test_two_test_files_may_share_a_sentence(self) -> None:
        self.repo.write(integration_path("tenferro-ad", "a"), test_file(SENTENCE))
        self.repo.write(integration_path("tenferro-ad", "b"), test_file(SENTENCE))

        report, errors = self.report()

        self.assertEqual(errors, [])
        self.assertEqual(report["findings"], [])

    def test_three_test_files_sharing_a_sentence_is_a_finding(self) -> None:
        for crate, name in [
            ("tenferro-ad", "a"),
            ("tenferro-ad", "b"),
            ("tenferro-runtime", "c"),
        ]:
            self.repo.write(integration_path(crate, name), test_file(SENTENCE))

        report, errors = self.report()

        self.assertEqual(errors, [])
        self.assertEqual(len(report["findings"]), 1)
        self.assertEqual(report["findings"][0]["literal"], SENTENCE)
        self.assertEqual(report["findings"][0]["occurrences"], 3)

    def test_production_sources_are_out_of_scope(self) -> None:
        for crate in ["tenferro-ad", "tenferro-runtime", "tenferro-cpu"]:
            self.repo.write(f"crates/{crate}/src/lib.rs", test_file(SENTENCE))

        report, errors = self.report()

        self.assertEqual(errors, [])
        self.assertEqual(report["findings"], [])

    def test_a_reviewed_entry_turns_the_finding_into_a_note(self) -> None:
        files = [
            integration_path("tenferro-ad", "a"),
            integration_path("tenferro-ad", "b"),
            integration_path("tenferro-runtime", "c"),
        ]
        for path in files:
            self.repo.write(path, test_file(SENTENCE))
        allowlist = self.repo.allowlist(
            "\n".join(
                [
                    "[[message]]",
                    f'literal = "{SENTENCE}"',
                    "files = [",
                    *[f'  "{path}",' for path in files],
                    "]",
                    'reason = "one reviewed sentence for a shared accessor"',
                ]
            )
            + "\n"
        )

        report, errors = CHECKER_MODULE.build_report(self.repo.root, allowlist, 2)

        self.assertEqual(errors, [])
        self.assertEqual(report["findings"], [])
        self.assertEqual(len(report["reviewed"]), 1)

    def test_an_unrecorded_file_fails_a_reviewed_entry(self) -> None:
        recorded = [
            integration_path("tenferro-ad", "a"),
            integration_path("tenferro-ad", "b"),
        ]
        intruder = integration_path("tenferro-runtime", "c")
        for path in [*recorded, intruder]:
            self.repo.write(path, test_file(SENTENCE))
        allowlist = self.repo.allowlist(
            "\n".join(
                [
                    "[[message]]",
                    f'literal = "{SENTENCE}"',
                    "files = [",
                    *[f'  "{path}",' for path in recorded],
                    "]",
                    'reason = "one reviewed sentence for a shared accessor"',
                ]
            )
            + "\n"
        )

        report, errors = CHECKER_MODULE.build_report(self.repo.root, allowlist, 2)

        self.assertEqual(errors, [])
        self.assertEqual(len(report["findings"]), 1)
        self.assertEqual(report["findings"][0]["unrecorded"], [intruder])

    def test_an_entry_without_a_reason_fails_the_run(self) -> None:
        for crate in ["tenferro-ad", "tenferro-runtime", "tenferro-cpu"]:
            self.repo.write(integration_path(crate, "c"), test_file(SENTENCE))
        allowlist = self.repo.allowlist(
            "\n".join(
                [
                    "[[message]]",
                    f'literal = "{SENTENCE}"',
                    'files = ["crates/tenferro-ad/tests/integration/c.rs"]',
                ]
            )
            + "\n"
        )

        _, errors = CHECKER_MODULE.build_report(self.repo.root, allowlist, 2)

        self.assertTrue(errors)
        self.assertIn("has no reason", errors[0])

    def test_a_stale_entry_is_reported(self) -> None:
        self.repo.write(integration_path("tenferro-ad", "a"), test_file(SENTENCE))
        allowlist = self.repo.allowlist(
            "\n".join(
                [
                    "[[message]]",
                    f'literal = "{SENTENCE}"',
                    "files = [",
                    f'  "{integration_path("tenferro-ad", "a")}",',
                    "  \"crates/tenferro-ad/tests/integration/b.rs\",",
                    "]",
                    'reason = "one reviewed sentence for a shared accessor"',
                ]
            )
            + "\n"
        )

        _, errors = CHECKER_MODULE.build_report(self.repo.root, allowlist, 2)

        self.assertTrue(errors)
        self.assertIn("stale allowlist entry", errors[0])

    def test_short_labels_are_not_invariants(self) -> None:
        for crate in ["tenferro-ad", "tenferro-runtime", "tenferro-cpu"]:
            self.repo.write(integration_path(crate, "c"), test_file("download"))

        report, errors = self.report()

        self.assertEqual(errors, [])
        self.assertEqual(report["findings"], [])

    def test_check_mode_fails_on_a_finding_and_report_mode_does_not(self) -> None:
        for crate in ["tenferro-ad", "tenferro-runtime", "tenferro-cpu"]:
            self.repo.write(integration_path(crate, "c"), test_file(SENTENCE))
        allowlist = self.repo.allowlist("")

        self.assertEqual(
            CHECKER_MODULE.main(
                [
                    "--check",
                    "--root-dir",
                    str(self.repo.root),
                    "--allowlist",
                    str(allowlist),
                ]
            ),
            1,
        )
        self.assertEqual(
            CHECKER_MODULE.main(
                [
                    "--report",
                    "--root-dir",
                    str(self.repo.root),
                    "--allowlist",
                    str(allowlist),
                ]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
