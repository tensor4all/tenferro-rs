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


def _inventory(
    *,
    entry_id: str = "source-ok",
    path: str = "src/legacy.rs",
    needle: str | None = "RAW_HANDLE",
    symbol: str | None = None,
    scan: str | None = "raw-handle-scan",
    status: str = "active",
    category: str = "raw-handle",
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
        lines.append(f'needle = "{needle}"')
    if symbol is not None:
        lines.append(f'symbol = "{symbol}"')
    return "\n".join(lines)


def _scan(
    *,
    needle: str = "RAW_HANDLE",
    status: str | None = "active",
    kind: str | None = "source",
    owner_issue: int | None = 1557,
    rationale: str | None = "track every remaining raw-handle seam",
    root: str = "src",
    glob: str = "**/*.rs",
) -> str:
    lines = [
        "[[source_scans]]",
        'id = "raw-handle-scan"',
        'gate = "G5"',
        f'root = "{root}"',
        f'glob = "{glob}"',
        f'needle = "{needle}"',
    ]
    if status is not None:
        lines.append(f'status = "{status}"')
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
            _manifest(_entry(), _inventory(), _scan()),
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
                _inventory(symbol="const RAW_HANDLE", needle=None),
                _scan(needle="const RAW_HANDLE"),
            ),
            files={
                "fixtures/ok.rs": "fn fixture() {}\n",
                "src/legacy.rs": "const RAW_HANDLE: &str = \"legacy\";\n",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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


def _inventory_field_value(field: str) -> str:
    return {
        "category": "raw-handle",
        "rationale": "remove the legacy authority seam",
        "disposition": "remove",
    }[field]


if __name__ == "__main__":
    unittest.main()
