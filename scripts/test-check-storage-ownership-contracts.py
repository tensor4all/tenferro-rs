#!/usr/bin/env python3
"""Tests for the storage ownership contract ledger checker."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check-storage-ownership-contracts.py"


def _entry(
    *,
    entry_id: str = "fixture-ok",
    kind: str = "trybuild-pass",
    status: str = "active",
    gate: str = "G1",
    owner_issue: int | None = 1557,
    path: str | None = "fixtures/ok.rs",
) -> str:
    lines = [
        "[[fixtures]]",
        f'id = "{entry_id}"',
        f'gate = "{gate}"',
        f'kind = "{kind}"',
        f'status = "{status}"',
    ]
    if owner_issue is not None:
        lines.append(f"owner_issue = {owner_issue}")
    if path is not None:
        lines.append(f'path = "{path}"')
    return "\n".join(lines)


def _suite(
    *,
    entry_id: str = "suite-pass",
    kind: str | None = "trybuild-pass",
    root: str | None = "fixtures",
    glob: str | None = "**/*.rs",
    owner_issue: int | None = 1557,
    rationale: str | None = "cover the active trybuild fixtures in this root",
) -> str:
    lines = [
        "[[fixture_suites]]",
        f'id = "{entry_id}"',
    ]
    if kind is not None:
        lines.append(f'kind = "{kind}"')
    if root is not None:
        lines.append(f'root = "{root}"')
    if glob is not None:
        lines.append(f'glob = "{glob}"')
    if owner_issue is not None:
        lines.append(f"owner_issue = {owner_issue}")
    if rationale is not None:
        lines.append(f'rationale = "{rationale}"')
    return "\n".join(lines)


def _inventory(
    *,
    entry_id: str = "source-ok",
    path: str = "src/legacy.rs",
    needle: str | None = "RAW_HANDLE",
    symbol: str | None = None,
    scan: str | None = "raw-handle-scan",
    status: str = "active",
    category: str = "raw-handle-extraction",
    rationale: str = "remove the legacy authority seam",
    disposition: str = "remove",
    removal_issue: int = 1559,
) -> str:
    lines = [
        "[[source_inventory]]",
        f'id = "{entry_id}"',
        'gate = "G5"',
        'kind = "source"',
        f'status = "{status}"',
        "owner_issue = 1557",
        f'path = "{path}"',
        *([f'scan = "{scan}"'] if scan is not None else []),
        *_selector_lines(needle=needle, symbol=symbol).splitlines(),
        f'category = "{category}"',
        f'rationale = "{rationale}"',
        f'disposition = "{disposition}"',
        f"removal_issue = {removal_issue}",
    ]
    return "\n".join(lines)


def _selector_lines(*, needle: str | None, symbol: str | None) -> str:
    lines: list[str] = []
    if needle is not None:
        if "\n" in needle:
            lines.append(f"needle = '''{needle}'''")
        else:
            lines.append(f'needle = "{needle}"')
    if symbol is not None:
        lines.append(f'symbol = "{symbol}"')
    return "\n".join(lines)


def _scan(
    *,
    entry_id: str = "raw-handle-scan",
    needle: str = "RAW_HANDLE",
    status: str | None = "active",
    kind: str | None = "source",
    owner_issue: int | None = 1557,
    rationale: str | None = "track every remaining raw-handle seam",
    root: str = "src",
    glob: str = "**/*.rs",
    mode: str | None = "inventoried",
) -> str:
    lines = [
        "[[source_scans]]",
        f'id = "{entry_id}"',
        'gate = "G5"',
        f'root = "{root}"',
        f'glob = "{glob}"',
        *_selector_lines(needle=needle, symbol=None).splitlines(),
    ]
    if status is not None:
        lines.append(f'status = "{status}"')
    if mode is not None:
        lines.append(f'mode = "{mode}"')
    if kind is not None:
        lines.append(f'kind = "{kind}"')
    if owner_issue is not None:
        lines.append(f"owner_issue = {owner_issue}")
    if rationale is not None:
        lines.append(f'rationale = "{rationale}"')
    return "\n".join(lines)


def _manifest(*sections: str) -> str:
    return "\n\n".join(
        [
            'schema = "tenferro.storage-ownership-contracts.v1"',
            *sections,
        ]
    ) + "\n"


class CheckerTests(unittest.TestCase):
    def run_checker(
        self,
        manifest: str,
        *,
        files: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "storage-ownership-contracts.toml"
            manifest_path.write_text(manifest)
            for relative, contents in (files or {}).items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def run_checker_at_root(
        self, root: Path, manifest: str
    ) -> subprocess.CompletedProcess[str]:
        manifest_path = root / "storage-ownership-contracts.toml"
        manifest_path.write_text(manifest)
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(root),
                "--manifest",
                str(manifest_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def make_symlink_or_skip(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink creation unsupported: {error}")

    def test_clean_fixture_and_exact_source_inventory_pass(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _suite(), _inventory(), _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "const RAW_HANDLE: &str = \"legacy\";\n",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("storage ownership contract ledger: OK", result.stdout)

    def test_duplicate_fixture_and_inventory_ids_are_rejected(self) -> None:
        duplicate_fixture = _entry(entry_id="same-id") + "\n\n" + _entry(
            entry_id="same-id", path="fixtures/other.rs"
        )
        result = self.run_checker(
            _manifest(duplicate_fixture, _inventory(), _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "fixtures/other.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate entry id 'same-id'", result.stderr)

        duplicate_inventory = _inventory(entry_id="same-id") + "\n\n" + _inventory(
            entry_id="same-id", path="src/other.rs"
        )
        result = self.run_checker(
            _manifest(_entry(), duplicate_inventory, _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
                "src/other.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate entry id 'same-id'", result.stderr)

    def test_active_fixture_path_must_exist(self) -> None:
        result = self.run_checker(
            _manifest(_entry(path="fixtures/missing.rs")),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("active fixture 'fixture-ok' path does not exist", result.stderr)

    def test_deferred_fixture_requires_an_owning_issue(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(
                    status="deferred",
                    owner_issue=None,
                    path=None,
                )
            )
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "deferred fixture 'fixture-ok' must declare owner_issue",
            result.stderr,
        )

    def test_clean_active_trybuild_fixture_suite_passes(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _suite()),
            files={"fixtures/ok.rs": "fn fixture() {}\n"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixture_suite_requires_schema_fields(self) -> None:
        cases = (
            ("kind", _suite(kind=None), "fixture suite 'suite-pass' must declare kind"),
            ("root", _suite(root=None), "fixture suite 'suite-pass' must declare root"),
            ("glob", _suite(glob=None), "fixture suite 'suite-pass' must declare glob"),
            (
                "owner_issue",
                _suite(owner_issue=None),
                "fixture suite 'suite-pass' must declare owner_issue",
            ),
            (
                "rationale",
                _suite(rationale=None),
                "fixture suite 'suite-pass' must declare rationale",
            ),
        )
        for name, suite, expected in cases:
            with self.subTest(case=name):
                result = self.run_checker(
                    _manifest(_entry(), suite),
                    files={"fixtures/ok.rs": "fn fixture() {}\n"},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_fixture_suite_rejects_unknown_and_line_number_fields(self) -> None:
        for field, declaration, expected in (
            (
                "description",
                'description = "temporary"',
                "fixture suite 'suite-pass' has unknown field 'description'",
            ),
            (
                "line",
                "line = 12",
                "fixture suite 'suite-pass' must not use line-number key 'line'",
            ),
        ):
            with self.subTest(field=field):
                result = self.run_checker(
                    _manifest(_entry(), _suite() + f"\n{declaration}"),
                    files={"fixtures/ok.rs": "fn fixture() {}\n"},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_fixture_suite_kind_must_be_trybuild_kind(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _suite(kind="source")),
            files={"fixtures/ok.rs": "fn fixture() {}\n"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "fixture suite 'suite-pass' kind must be 'trybuild-fail' or 'trybuild-pass'",
            result.stderr,
        )

    def test_fixture_suite_absolute_glob_fails_without_traceback(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _suite(glob="/tmp/*.rs")),
            files={"fixtures/ok.rs": "fn fixture() {}\n"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "fixture suite 'suite-pass' field 'glob' must be a repository-relative glob",
            result.stderr,
        )
        self.assertNotIn("Traceback", result.stderr)

    def test_fixture_suite_root_must_exist_and_match_a_file(self) -> None:
        missing_root = self.run_checker(
            _manifest(_entry(path="missing/ok.rs"), _suite(root="missing")),
        )
        self.assertNotEqual(missing_root.returncode, 0)
        self.assertIn(
            "active fixture suite 'suite-pass' root does not exist: 'missing'",
            missing_root.stderr,
        )

        empty = self.run_checker(
            _manifest(_entry(path="other/ok.txt"), _suite(root="other")),
            files={"other/ok.txt": "not a Rust fixture\n"},
        )
        self.assertNotEqual(empty.returncode, 0)
        self.assertIn(
            "active fixture suite 'suite-pass' glob '**/*.rs' matches no files",
            empty.stderr,
        )

    def test_suite_file_without_fixture_row_is_rejected(self) -> None:
        result = self.run_checker(
            _manifest(_suite()),
            files={"fixtures/orphan.rs": "fn orphan() {}\n"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "fixture suite 'suite-pass' matched file 'fixtures/orphan.rs' without exactly one active trybuild-pass fixture row",
            result.stderr,
        )

    def test_active_trybuild_fixture_outside_all_matching_suites_is_rejected(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _entry(entry_id="other-fixture", path="other/ok.rs"),
                _suite(root="other"),
            ),
            files={
                "fixtures/ok.rs": "fn outside_suite() {}\n",
                "other/ok.rs": "fn covered() {}\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "active trybuild fixture 'fixture-ok' path 'fixtures/ok.rs' must be covered by exactly one matching fixture suite; found 0",
            result.stderr,
        )

    def test_suite_kind_mismatch_and_wrong_trybuild_directory_are_rejected(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(path="fixtures/pass.rs"),
                _suite(kind="trybuild-fail", root="fixtures"),
            ),
            files={"fixtures/pass.rs": "fn pass() {}\n"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "fixture suite 'suite-pass' matched file 'fixtures/pass.rs' with fixture 'fixture-ok' kind 'trybuild-pass'; expected 'trybuild-fail'",
            result.stderr,
        )
        self.assertIn(
            "active trybuild fixture 'fixture-ok' path 'fixtures/pass.rs' must be covered by exactly one matching fixture suite; found 0",
            result.stderr,
        )

    def test_overlapping_suite_coverage_is_rejected(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _suite(
                    entry_id="suite-pass-overlap",
                    rationale="intentionally overlaps for the test",
                ),
            ),
            files={"fixtures/ok.rs": "fn fixture() {}\n"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "fixture path 'fixtures/ok.rs' is matched by more than one fixture suite: 'suite-pass', 'suite-pass-overlap'",
            result.stderr,
        )

    def test_deferred_trybuild_and_active_nontrybuild_rows_need_no_suite(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(
                    entry_id="deferred-pass",
                    status="deferred",
                    path="deferred/ok.rs",
                ),
                _entry(
                    entry_id="parity-baseline",
                    kind="parity",
                    path="baseline/parity.rs",
                ),
            ),
            files={
                "deferred/ok.rs": "fn deferred() {}\n",
                "baseline/parity.rs": "fn parity() {}\n",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_suite_diagnostics_use_sorted_repository_relative_matches(self) -> None:
        result = self.run_checker(
            _manifest(_suite()),
            files={
                "fixtures/z-last.rs": "fn z_last() {}\n",
                "fixtures/a-first.rs": "fn a_first() {}\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fixtures/a-first.rs", result.stderr)
        self.assertIn("fixtures/z-last.rs", result.stderr)
        first = result.stderr.index("fixtures/a-first.rs")
        second = result.stderr.index("fixtures/z-last.rs")
        self.assertLess(first, second)

    def test_inventory_needle_must_occur_exactly_once_in_declared_file(self) -> None:
        missing = self.run_checker(
            _manifest(_entry(), _inventory(needle="MISSING"), _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("needle 'MISSING' not found", missing.stderr)

        duplicate = self.run_checker(
            _manifest(_entry(), _inventory(), _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\nRAW_HANDLE\n",
            },
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("needle 'RAW_HANDLE' occurs 2 times", duplicate.stderr)

        outside_declared_file = self.run_checker(
            _manifest(_entry(), _inventory(path="src/declared.rs"), _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/declared.rs": "const OTHER: &str = \"declared\";\n",
                "src/other.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(outside_declared_file.returncode, 0)
        self.assertIn(
            "needle 'RAW_HANDLE' not found in declared file 'src/declared.rs'",
            outside_declared_file.stderr,
        )

    def test_inventory_requires_exactly_one_selector(self) -> None:
        for selector in (
            _inventory(needle=None),
            _inventory(needle="RAW_HANDLE", symbol="const RAW_HANDLE"),
        ):
            with self.subTest(selector=selector):
                result = self.run_checker(
                    _manifest(_entry(), selector, _scan()),
                    files={
                        "fixtures/ok.rs": "fn fixture() {}\n",
                        "src/legacy.rs": "const RAW_HANDLE: &str = \"legacy\";\n",
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "must declare exactly one of symbol or needle", result.stderr
                )

    def test_inventory_symbol_must_occur_exactly_once_in_declared_file(self) -> None:
        missing = self.run_checker(
            _manifest(
                _entry(),
                _inventory(symbol="fn missing_signature", needle=None),
                _scan(),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "fn present_signature() {}\n",
            },
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn(
            "symbol 'fn missing_signature' not found in declared file 'src/legacy.rs'",
            missing.stderr,
        )

        duplicate = self.run_checker(
            _manifest(
                _entry(),
                _inventory(symbol="fn duplicated_signature", needle=None),
                _scan(needle="fn duplicated_signature"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": (
                    "fn duplicated_signature() {}\n"
                    "fn duplicated_signature() {}\n"
                ),
            },
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn(
            "symbol 'fn duplicated_signature' occurs 2 times in declared file "
            "'src/legacy.rs'",
            duplicate.stderr,
        )

    def test_symbol_selector_can_account_for_matching_active_scan(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _inventory(symbol="const RAW_HANDLE", needle=None),
                _scan(needle="const RAW_HANDLE"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "const RAW_HANDLE: &str = \"legacy\";\n",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_family_scan_two_occurrences_with_two_bound_rows_passes(self) -> None:
        inventories = _inventory(
            entry_id="family-first",
            needle="fn first_member() { FAMILY_MEMBER(); }",
            scan="family-scan",
        ) + "\n\n" + _inventory(
            entry_id="family-second",
            needle="fn second_member() { FAMILY_MEMBER(); }",
            scan="family-scan",
        )
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                inventories,
                _scan(entry_id="family-scan", needle="FAMILY_MEMBER"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": (
                    "fn first_member() { FAMILY_MEMBER(); }\n"
                    "fn second_member() { FAMILY_MEMBER(); }\n"
                ),
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_family_scan_extra_occurrence_fails_as_drift(self) -> None:
        inventories = _inventory(
            entry_id="family-first",
            needle="fn first_member() { FAMILY_MEMBER(); }",
            scan="family-scan",
        ) + "\n\n" + _inventory(
            entry_id="family-second",
            needle="fn second_member() { FAMILY_MEMBER(); }",
            scan="family-scan",
        )
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                inventories,
                _scan(entry_id="family-scan", needle="FAMILY_MEMBER"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": (
                    "fn first_member() { FAMILY_MEMBER(); }\n"
                    "fn second_member() { FAMILY_MEMBER(); }\n"
                    "fn untracked_member() { FAMILY_MEMBER(); }\n"
                ),
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source scan 'family-scan' path 'src/legacy.rs' needle "
            "'FAMILY_MEMBER' occurs 3 times but has 2 bound inventory rows",
            result.stderr,
        )

    def test_inventory_selector_must_contain_family_needle(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _inventory(
                    entry_id="family-first",
                    needle="fn first_member",
                    scan="family-scan",
                ),
                _scan(entry_id="family-scan", needle="FAMILY_MEMBER"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "fn first_member() {}\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source inventory 'family-first' selector 'fn first_member' must "
            "contain source scan 'family-scan' needle 'FAMILY_MEMBER'",
            result.stderr,
        )

    def test_moving_family_needle_between_scopes_breaks_exact_context(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _inventory(
                    entry_id="family-first",
                    needle="fn first_member() { FAMILY_MEMBER(); }",
                    scan="family-scan",
                ),
                _scan(entry_id="family-scan", needle="FAMILY_MEMBER"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": (
                    "fn first_member() {}\n"
                    "fn second_member() { FAMILY_MEMBER(); }\n"
                ),
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "needle 'fn first_member() { FAMILY_MEMBER(); }' not found in "
            "declared file 'src/legacy.rs'",
            result.stderr,
        )

    def test_family_scan_orphan_occurrence_fails_as_untracked(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _inventory(
                    entry_id="family-first",
                    needle="fn first_member() { FAMILY_MEMBER(); }",
                    scan="family-scan",
                    path="src/legacy.rs",
                ),
                _scan(entry_id="family-scan", needle="FAMILY_MEMBER"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "fn first_member() { FAMILY_MEMBER(); }\n",
                "src/orphan.rs": "fn orphan_member() { FAMILY_MEMBER(); }\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "scanned source seam 'src/orphan.rs' needle 'FAMILY_MEMBER' "
            "is not inventoried",
            result.stderr,
        )

    def test_forbidden_scan_with_zero_occurrences_and_no_inventory_passes(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _scan(
                    entry_id="forbidden-scan",
                    needle="FORBIDDEN_PATTERN",
                    mode="forbidden",
                ),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "fn allowed() {}\n",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_write_enum_rejects_an_added_variant(self) -> None:
        current_enum = """pub enum TensorWrite<'a> {
    Tensor(&'a mut Tensor),
    View(TensorViewMut<'a>),
}"""
        result = self.run_checker(
            _manifest(
                _entry(),
                _inventory(
                    entry_id="tensor-write-enum",
                    needle=current_enum,
                    scan="write-enum-shape",
                ),
                _scan(
                    entry_id="write-enum-shape",
                    needle=current_enum,
                ),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": (
                    "pub enum TensorWrite<'a> {\n"
                    "    Tensor(&'a mut Tensor),\n"
                    "    View(TensorViewMut<'a>),\n"
                    "    External(ExternalWrite<'a>),\n"
                    "}\n"
                ),
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "not found in declared file 'src/legacy.rs'",
            result.stderr,
        )

    def test_source_scan_requires_explicit_mode(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _inventory(), _scan(mode=None)),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source scan 'raw-handle-scan' must declare mode", result.stderr)

    def test_source_scan_rejects_unknown_mode(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _inventory(), _scan(mode="optional")),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source scan 'raw-handle-scan' mode 'optional' must be one of: "
            "forbidden, inventoried",
            result.stderr,
        )

    def test_inventoried_scan_requires_active_inventory(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _scan(mode="inventoried", needle="TRACKED_FAMILY"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "fn allowed() {}\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "inventoried source scan 'raw-handle-scan' must have at least one "
            "active bound inventory row",
            result.stderr,
        )

    def test_forbidden_scan_rejects_bound_inventory(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _inventory(), _scan(mode="forbidden")),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "forbidden source scan 'raw-handle-scan' must not have active bound "
            "inventory rows",
            result.stderr,
        )

    def test_forbidden_scan_rejects_occurrence_without_inventory(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _scan(mode="forbidden", needle="FORBIDDEN_PATTERN"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "fn bad() { FORBIDDEN_PATTERN(); }\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "forbidden source scan 'raw-handle-scan' path 'src/legacy.rs' "
            "needle 'FORBIDDEN_PATTERN' occurs 1 time",
            result.stderr,
        )

    def test_map_write_context_move_is_detected(self) -> None:
        context = "fn map_write(&self) { managed.resource().map_write(); }"
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _inventory(
                    entry_id="map-write-authority",
                    needle=context,
                    scan="map-write-family",
                    category="shared-to-mutable",
                    disposition="replace",
                    removal_issue=1560,
                ),
                _scan(
                    entry_id="map-write-family",
                    needle="managed.resource().map_write()",
                ),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": (
                    "fn map_write(&self) {}\n"
                    "fn unrelated(&self) { managed.resource().map_write(); }\n"
                ),
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            f"needle '{context}' not found in declared file 'src/legacy.rs'",
            result.stderr,
        )

    def test_new_scanned_source_seam_must_be_in_inventory(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _inventory(), _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
                "src/new.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "scanned source seam 'src/new.rs' needle 'RAW_HANDLE' is not inventoried",
            result.stderr,
        )

    def test_each_storage_inventory_category_rejects_an_extra_source_seam(self) -> None:
        cases = (
            ("raw-handle-extraction", "raw-handle-scan", "RAW_HANDLE_DECL"),
            (
                "public-mutable-projection",
                "public-mutable-projection-scan",
                "MUTABLE_PROJECTION",
            ),
            ("shared-to-mutable", "shared-to-mutable-scan", "MUTABLE_TRANSITION"),
            (
                "temporary-migration-adapter",
                "temporary-adapter-scan",
                "TEMP_ADAPTER",
            ),
        )
        for category, scan_id, needle in cases:
            with self.subTest(category=category):
                result = self.run_checker(
                    _manifest(
                        _entry(),
                        _inventory(
                            entry_id=f"{category}-source",
                            needle=needle,
                            scan=scan_id,
                            category=category,
                        ),
                        _scan(entry_id=scan_id, needle=needle),
                    ),
                    files={
                        "fixtures/ok.rs": "fn fixture() {}\n",
                        "src/legacy.rs": f"{needle}\n",
                        "src/extra.rs": f"{needle}\n",
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"scanned source seam 'src/extra.rs' needle '{needle}' is not inventoried",
                    result.stderr,
                )

    def test_each_storage_inventory_category_requires_its_selector_to_remain(self) -> None:
        cases = (
            ("raw-handle-extraction", "raw-handle-scan", "RAW_HANDLE_DECL"),
            (
                "public-mutable-projection",
                "public-mutable-projection-scan",
                "MUTABLE_PROJECTION",
            ),
            ("shared-to-mutable", "shared-to-mutable-scan", "MUTABLE_TRANSITION"),
            (
                "temporary-migration-adapter",
                "temporary-adapter-scan",
                "TEMP_ADAPTER",
            ),
        )
        for category, scan_id, needle in cases:
            with self.subTest(category=category):
                result = self.run_checker(
                    _manifest(
                        _entry(),
                        _inventory(
                            entry_id=f"{category}-source",
                            needle=needle,
                            scan=scan_id,
                            category=category,
                        ),
                        _scan(entry_id=scan_id, needle=needle),
                    ),
                    files={
                        "fixtures/ok.rs": "fn fixture() {}\n",
                        "src/legacy.rs": "fn renamed_source() {}\n",
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"needle '{needle}' not found in declared file 'src/legacy.rs'",
                    result.stderr,
                )

    def test_temporary_migration_adapter_must_be_removed(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _inventory(
                    category="temporary-migration-adapter",
                    disposition="replace",
                ),
                _scan(),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "temporary-migration-adapter disposition must be 'remove'",
            result.stderr,
        )

    def test_source_scan_requires_explicit_status_kind_owner_and_rationale(self) -> None:
        cases = (
            ("status", _scan(status=None), "source scan 'raw-handle-scan' must declare status"),
            ("kind", _scan(kind=None), "source scan 'raw-handle-scan' must declare kind"),
            (
                "owner_issue",
                _scan(owner_issue=None),
                "source scan 'raw-handle-scan' must declare owner_issue",
            ),
            (
                "rationale",
                _scan(rationale=None),
                "source scan 'raw-handle-scan' must declare rationale",
            ),
            (
                "kind-source",
                _scan(kind="property"),
                "source scan 'raw-handle-scan' kind must be 'source'",
            ),
        )
        for name, scan, expected in cases:
            with self.subTest(case=name):
                result = self.run_checker(
                    _manifest(_entry(), _inventory(), scan),
                    files={
                        "fixtures/ok.rs": "fn fixture() {}\n",
                        "src/legacy.rs": "RAW_HANDLE\n",
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_source_scan_rejects_legacy_activation_fields(self) -> None:
        for field in ("enabled", "disabled"):
            with self.subTest(field=field):
                result = self.run_checker(
                    _manifest(
                        _entry(),
                        _inventory(),
                        _scan() + f"\n{field} = false",
                    ),
                    files={
                        "fixtures/ok.rs": "fn fixture() {}\n",
                        "src/legacy.rs": "RAW_HANDLE\n",
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"source scan 'raw-handle-scan' has unknown field '{field}'",
                    result.stderr,
                )

    def test_active_source_scan_must_match_at_least_one_file(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _inventory(), _scan(glob="**/*.missing")),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "active source scan 'raw-handle-scan' glob '**/*.missing' matches no files",
            result.stderr,
        )

    def test_active_inventory_must_reference_an_active_scan(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _inventory(), _scan(status="deferred")),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "active source inventory 'source-ok' may reference only active source scan "
            "'raw-handle-scan'",
            result.stderr,
        )

    def test_active_inventory_must_declare_scan(self) -> None:
        result = self.run_checker(
            _manifest(_entry(), _inventory(scan=None), _scan()),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "active source inventory 'source-ok' must declare scan",
            result.stderr,
        )

    def test_deferred_inventory_may_reference_a_deferred_scan(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _inventory(status="deferred"),
                _scan(status="deferred"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_active_inventory_path_must_be_under_scan_root(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _inventory(path="src/legacy.rs"),
                _scan(root="src/owned"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
                "src/owned/owned.rs": "fn owned() {}\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source inventory 'source-ok' path 'src/legacy.rs' must be under scan root "
            "'src/owned'",
            result.stderr,
        )

    def test_active_inventory_path_must_match_scan_glob(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _inventory(path="src/legacy.rs"),
                _scan(glob="**/owned.rs"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
                "src/owned.rs": "fn owned() {}\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source inventory 'source-ok' path 'src/legacy.rs' does not match scan "
            "'raw-handle-scan' glob '**/owned.rs'",
            result.stderr,
        )

    def test_symlink_escape_in_active_fixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            outside_file = Path(outside) / "fixture.rs"
            outside_file.write_text("fn outside() {}\n")
            link = root / "fixtures" / "escape.rs"
            link.parent.mkdir(parents=True)
            self.make_symlink_or_skip(link, outside_file)
            (root / "fixtures" / "ok.rs").write_text("fn fixture() {}\n")

            result = self.run_checker_at_root(
                root,
                _manifest(_entry(path="fixtures/escape.rs")),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "fixture 'fixture-ok' path 'fixtures/escape.rs' resolves outside repository root",
            result.stderr,
        )

    def test_symlink_escape_in_inventory_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            outside_file = Path(outside) / "legacy.rs"
            outside_file.write_text("RAW_HANDLE\n")
            link = root / "src" / "escape.rs"
            link.parent.mkdir(parents=True)
            self.make_symlink_or_skip(link, outside_file)
            (root / "fixtures" / "ok.rs").parent.mkdir(parents=True, exist_ok=True)
            (root / "fixtures" / "ok.rs").write_text("fn fixture() {}\n")

            result = self.run_checker_at_root(
                root,
                _manifest(_entry(), _inventory(path="src/escape.rs"), _scan()),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source inventory 'source-ok' path 'src/escape.rs' resolves outside repository root",
            result.stderr,
        )

    def test_symlink_escape_in_scan_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            outside_file = Path(outside) / "legacy.rs"
            outside_file.write_text("RAW_HANDLE\n")
            source_root = root / "src"
            source_root.mkdir(parents=True)
            link = source_root / "escape.rs"
            self.make_symlink_or_skip(link, outside_file)

            inventories = _inventory(path="src/legacy.rs") + "\n\n" + _inventory(
                entry_id="source-escape", path="src/escape.rs"
            )
            (root / "fixtures").mkdir(parents=True)
            (root / "fixtures" / "ok.rs").write_text("fn fixture() {}\n")
            (root / "src" / "legacy.rs").write_text("RAW_HANDLE\n")
            result = self.run_checker_at_root(
                root,
                _manifest(_entry(), inventories, _scan()),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "active source scan 'raw-handle-scan' candidate 'src/escape.rs' resolves outside repository root",
            result.stderr,
        )

    def test_symlink_escape_in_fixture_suite_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            outside_root = Path(outside) / "fixtures"
            outside_root.mkdir()
            link = root / "fixtures-link"
            self.make_symlink_or_skip(link, outside_root)

            result = self.run_checker_at_root(
                root,
                _manifest(_entry(path="fixtures-link/ok.rs"), _suite(root="fixtures-link")),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "fixture suite 'suite-pass' root 'fixtures-link' resolves outside repository root",
            result.stderr,
        )

    def test_symlink_escape_in_fixture_suite_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            outside_file = Path(outside) / "orphan.rs"
            outside_file.write_text("fn outside() {}\n")
            fixtures = root / "fixtures"
            fixtures.mkdir()
            link = fixtures / "escape.rs"
            self.make_symlink_or_skip(link, outside_file)

            result = self.run_checker_at_root(root, _manifest(_suite()))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "active fixture suite 'suite-pass' candidate 'fixtures/escape.rs' resolves outside repository root",
            result.stderr,
        )

    def test_inventory_requires_category_rationale_disposition_and_removal_issue(
        self,
    ) -> None:
        for field in ("category", "rationale", "disposition", "removal_issue"):
            with self.subTest(field=field):
                inventory = _inventory()
                if field == "removal_issue":
                    inventory = inventory.replace("removal_issue = 1559", "")
                else:
                    inventory = inventory.replace(
                        f'{field} = "{_inventory_field_value(field)}"\n', ""
                    )
                result = self.run_checker(
                    _manifest(_entry(), inventory, _scan()),
                    files={
                        "fixtures/ok.rs": "fn fixture() {}\n",
                        "src/legacy.rs": "RAW_HANDLE\n",
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"source inventory 'source-ok' must declare {field}",
                    result.stderr,
                )

    def test_inventory_rejects_unknown_category(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _inventory(category="raw-handle"),
                _scan(),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source inventory 'source-ok' category 'raw-handle' must be one of: "
            "public-mutable-projection, raw-handle-extraction, shared-to-mutable, "
            "temporary-migration-adapter",
            result.stderr,
        )

    def test_inventory_rejects_unknown_disposition(self) -> None:
        result = self.run_checker(
            _manifest(
                _entry(),
                _suite(),
                _inventory(disposition="keep"),
                _scan(),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "RAW_HANDLE\n",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "source inventory 'source-ok' disposition 'keep' must be one of: "
            "narrow, remove, replace",
            result.stderr,
        )


def _inventory_field_value(field: str) -> str:
    return {
        "category": "raw-handle-extraction",
        "rationale": "remove the legacy authority seam",
        "disposition": "remove",
    }[field]


if __name__ == "__main__":
    unittest.main()
