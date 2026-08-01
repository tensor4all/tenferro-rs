#!/usr/bin/env python3
"""Validate the storage ownership contract verification ledger."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "tenferro.storage-ownership-contracts.v1"
GATES = frozenset({f"G{number}" for number in range(1, 8)})
KINDS = frozenset(
    {
        "trybuild-fail",
        "trybuild-pass",
        "parity",
        "source",
        "corruption",
        "miri",
        "property",
    }
)
STATUSES = frozenset({"active", "deferred"})
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
LINE_NUMBER_KEYS = frozenset(
    {
        "line",
        "lines",
        "line_number",
        "line_numbers",
        "line_no",
        "line_num",
        "lineno",
    }
)

TOP_LEVEL_KEYS = frozenset(
    {"schema", "fixtures", "fixture_suites", "source_scans", "source_inventory"}
)
COMMON_KEYS = frozenset({"id", "gate", "kind", "status", "owner_issue"})
FIXTURE_KEYS = COMMON_KEYS | frozenset(
    {"path", "future_path", "rationale", "description"}
)
FIXTURE_SUITE_KEYS = frozenset(
    {"id", "kind", "root", "glob", "owner_issue", "rationale"}
)
SCAN_KEYS = frozenset(
    {
        "id",
        "gate",
        "root",
        "glob",
        "needle",
        "status",
        "owner_issue",
        "kind",
        "rationale",
        "description",
    }
)
INVENTORY_KEYS = COMMON_KEYS | frozenset(
    {
        "path",
        "scan",
        "symbol",
        "needle",
        "category",
        "rationale",
        "disposition",
        "removal_issue",
        "description",
    }
)


@dataclass(frozen=True)
class Fixture:
    entry_id: str
    kind: str
    status: str
    path: str | None


@dataclass(frozen=True)
class FixtureSuite:
    entry_id: str
    kind: str
    root: str
    glob: str
    resolved_root: Path | None


@dataclass(frozen=True)
class Scan:
    entry_id: str
    root: str
    glob: str
    needle: str
    active: bool
    resolved_root: Path | None


@dataclass(frozen=True)
class Inventory:
    entry_id: str
    status: str
    path: str
    scan: str | None
    symbol: str | None
    needle: str | None
    selector: str


def _label(kind: str, row: dict[str, Any]) -> str:
    entry_id = row.get("id")
    if isinstance(entry_id, str) and entry_id:
        return f"{kind} '{entry_id}'"
    return f"{kind} entry"


def _has_line_number_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in LINE_NUMBER_KEYS or normalized.startswith("line_number_")


def _check_line_number_keys(
    row: dict[str, Any], label: str, errors: list[str]
) -> None:
    for key in sorted(row):
        if isinstance(key, str) and _has_line_number_key(key):
            errors.append(f"{label} must not use line-number key '{key}'")


def _check_unknown_keys(
    row: dict[str, Any], allowed: frozenset[str], label: str, errors: list[str]
) -> None:
    for key in sorted(set(row) - allowed):
        errors.append(f"{label} has unknown field '{key}'")


def _entry_id(
    row: dict[str, Any],
    kind: str,
    ids: dict[str, str],
    errors: list[str],
) -> str | None:
    label = _label(kind, row)
    value = row.get("id")
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must declare a non-empty stable id")
        return None
    if not ID_PATTERN.fullmatch(value):
        errors.append(
            f"{kind} '{value}' id must contain only letters, digits, '.', '_' or '-'"
        )
    if value in ids:
        errors.append(f"duplicate entry id '{value}'")
    else:
        ids[value] = kind
    return value


def _required_string(
    row: dict[str, Any], key: str, label: str, errors: list[str]
) -> str | None:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must declare {key}")
        return None
    return value


def _optional_string(
    row: dict[str, Any], key: str, label: str, errors: list[str]
) -> str | None:
    if key not in row:
        return None
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} field '{key}' must be a non-empty string")
        return None
    return value


def _positive_issue(
    row: dict[str, Any], label: str, errors: list[str], *, required: bool
) -> int | None:
    if "owner_issue" not in row:
        if required:
            errors.append(f"{label} must declare owner_issue")
        return None
    value = row["owner_issue"]
    if type(value) is not int or value <= 0:
        errors.append(f"{label} owner_issue must be a positive integer")
        return None
    return value


def _gate(row: dict[str, Any], label: str, errors: list[str]) -> str | None:
    value = _required_string(row, "gate", label, errors)
    if value is not None and value not in GATES:
        errors.append(f"{label} gate must be one of G1..G7, got '{value}'")
        return None
    return value


def _kind(row: dict[str, Any], label: str, errors: list[str]) -> str | None:
    value = _required_string(row, "kind", label, errors)
    if value is not None and value not in KINDS:
        errors.append(f"{label} kind must be one of the supported contract kinds")
        return None
    return value


def _status(
    row: dict[str, Any],
    label: str,
    errors: list[str],
    *,
    default: str | None = None,
) -> str | None:
    if "status" not in row and default is not None:
        return default
    value = _required_string(row, "status", label, errors)
    if value is not None and value not in STATUSES:
        errors.append(f"{label} status must be 'active' or 'deferred'")
        return None
    return value


def _relative_path(
    value: str | None, field: str, label: str, errors: list[str]
) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{label} field '{field}' must be a repository-relative path")
        return None
    normalized = path.as_posix()
    if normalized in {"", "."}:
        errors.append(f"{label} field '{field}' must be a non-empty path")
        return None
    return normalized


def _resolve_under_root(
    root: Path,
    relative_path: str,
    label: str,
    errors: list[str],
    *,
    field: str,
) -> Path | None:
    candidate = root / relative_path
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as error:
        errors.append(
            f"{label} {field} '{relative_path}' cannot be resolved: {error}"
        )
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        errors.append(
            f"{label} {field} '{relative_path}' resolves outside repository root"
        )
        return None
    return resolved


def _file_contents(
    root: Path, relative_path: str, label: str, errors: list[str]
) -> str | None:
    path = _resolve_under_root(
        root, relative_path, label, errors, field="path"
    )
    if path is None:
        return None
    if not path.is_file():
        errors.append(f"{label} declared file '{relative_path}' does not exist")
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        errors.append(
            f"{label} declared file '{relative_path}' cannot be read: {error}"
        )
        return None


def _fixture_rows(
    rows: list[Any], root: Path, ids: dict[str, str], errors: list[str]
) -> list[Fixture]:
    fixtures: list[Fixture] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("fixture entries must be TOML tables")
            continue
        label = _label("fixture", row)
        _check_line_number_keys(row, label, errors)
        _check_unknown_keys(row, FIXTURE_KEYS, label, errors)
        entry_id = _entry_id(row, "fixture", ids, errors)
        _gate(row, label, errors)
        kind = _kind(row, label, errors)
        status = _status(row, label, errors)
        _positive_issue(row, label, errors, required=False)
        if "owner_issue" not in row:
            fixture_name = entry_id or "<unknown>"
            if status == "deferred":
                errors.append(
                    f"deferred fixture '{fixture_name}' must declare owner_issue"
                )
            else:
                errors.append(f"fixture '{fixture_name}' must declare owner_issue")
        path = _relative_path(
            _optional_string(row, "path", label, errors), "path", label, errors
        )
        future_path = _relative_path(
            _optional_string(row, "future_path", label, errors),
            "future_path",
            label,
            errors,
        )
        resolved_path = (
            _resolve_under_root(root, path, label, errors, field="path")
            if path is not None
            else None
        )
        if future_path is not None:
            _resolve_under_root(root, future_path, label, errors, field="future_path")
        _optional_string(row, "rationale", label, errors)

        if status == "active":
            if path is None:
                errors.append(f"active fixture '{entry_id or '<unknown>'}' must declare path")
            elif resolved_path is None:
                pass
            elif not resolved_path.is_file():
                errors.append(
                    f"active fixture '{entry_id or '<unknown>'}' path does not exist: '{path}'"
                )

        if entry_id is not None and kind is not None and status is not None:
            fixtures.append(
                Fixture(
                    entry_id=entry_id,
                    kind=kind,
                    status=status,
                    path=path,
                )
            )
    return fixtures


def _fixture_suite_rows(
    rows: list[Any], root: Path, ids: dict[str, str], errors: list[str]
) -> list[FixtureSuite]:
    suites: list[FixtureSuite] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("fixture suite entries must be TOML tables")
            continue
        label = _label("fixture suite", row)
        _check_line_number_keys(row, label, errors)
        _check_unknown_keys(row, FIXTURE_SUITE_KEYS, label, errors)
        entry_id = _entry_id(row, "fixture suite", ids, errors)
        kind = _required_string(row, "kind", label, errors)
        if kind is not None and kind not in {"trybuild-fail", "trybuild-pass"}:
            errors.append(
                f"{label} kind must be 'trybuild-fail' or 'trybuild-pass'"
            )
        root_value = _relative_path(
            _required_string(row, "root", label, errors), "root", label, errors
        )
        glob_value = _required_string(row, "glob", label, errors)
        _positive_issue(row, label, errors, required=True)
        _required_string(row, "rationale", label, errors)

        valid_glob = glob_value is not None
        if glob_value is not None:
            glob_path = Path(glob_value)
            if glob_path.is_absolute() or ".." in glob_path.parts:
                errors.append(
                    f"{label} field 'glob' must be a repository-relative glob"
                )
                valid_glob = False

        resolved_root = (
            _resolve_under_root(root, root_value, label, errors, field="root")
            if root_value is not None
            else None
        )
        if (
            entry_id is not None
            and kind in {"trybuild-fail", "trybuild-pass"}
            and root_value is not None
            and glob_value is not None
            and valid_glob
        ):
            if resolved_root is not None and not resolved_root.is_dir():
                errors.append(
                    f"active fixture suite '{entry_id}' root does not exist: '{root_value}'"
                )
            suites.append(
                FixtureSuite(
                    entry_id=entry_id,
                    kind=kind,
                    root=root_value,
                    glob=glob_value,
                    resolved_root=resolved_root,
                )
            )
    return suites


def _scan_rows(
    rows: list[Any], root: Path, ids: dict[str, str], errors: list[str]
) -> list[Scan]:
    scans: list[Scan] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("source scan entries must be TOML tables")
            continue
        label = _label("source scan", row)
        _check_line_number_keys(row, label, errors)
        _check_unknown_keys(row, SCAN_KEYS, label, errors)
        entry_id = _entry_id(row, "source scan", ids, errors)
        _gate(row, label, errors)
        kind = _kind(row, label, errors)
        if kind is not None and kind != "source":
            errors.append(f"{label} kind must be 'source'")
        status = _status(row, label, errors)
        _positive_issue(row, label, errors, required=True)
        root_value = _relative_path(
            _required_string(row, "root", label, errors), "root", label, errors
        )
        glob_value = _required_string(row, "glob", label, errors)
        needle = _required_string(row, "needle", label, errors)
        _required_string(row, "rationale", label, errors)

        if glob_value is not None:
            glob_path = Path(glob_value)
            if glob_path.is_absolute() or ".." in glob_path.parts:
                errors.append(
                    f"{label} field 'glob' must be a repository-relative glob"
                )

        resolved_root = (
            _resolve_under_root(root, root_value, label, errors, field="root")
            if root_value is not None
            else None
        )
        active = status == "active" and entry_id is not None
        if active and resolved_root is not None and not resolved_root.is_dir():
            errors.append(
                f"active source scan '{entry_id}' root does not exist: '{root_value}'"
            )
        if entry_id is not None and root_value is not None and glob_value and needle:
            scans.append(
                Scan(
                    entry_id=entry_id,
                    root=root_value,
                    glob=glob_value,
                    needle=needle,
                    active=active,
                    resolved_root=resolved_root,
                )
            )
    return scans


def _check_inventory_scan_path(
    root: Path,
    path: str,
    scan: Scan,
    label: str,
    errors: list[str],
) -> None:
    try:
        Path(path).relative_to(Path(scan.root))
    except ValueError:
        errors.append(
            f"{label} path '{path}' must be under scan root '{scan.root}'"
        )
        return

    if scan.resolved_root is None:
        return
    scan_root = root / scan.root
    try:
        candidates = scan_root.glob(scan.glob)
        declared_path = root / path
        if not any(candidate == declared_path for candidate in candidates):
            errors.append(
                f"{label} path '{path}' does not match scan '{scan.entry_id}' "
                f"glob '{scan.glob}'"
            )
    except (OSError, ValueError) as error:
        errors.append(
            f"active source scan '{scan.entry_id}' cannot scan "
            f"root '{scan.root}' with glob '{scan.glob}': {error}"
        )


def _inventory_rows(
    rows: list[Any],
    root: Path,
    ids: dict[str, str],
    scans_by_id: dict[str, Scan],
    errors: list[str],
) -> list[Inventory]:
    inventories: list[Inventory] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("source inventory entries must be TOML tables")
            continue
        label = _label("source inventory", row)
        _check_line_number_keys(row, label, errors)
        _check_unknown_keys(row, INVENTORY_KEYS, label, errors)
        entry_id = _entry_id(row, "source inventory", ids, errors)
        _gate(row, label, errors)
        kind = _kind(row, label, errors)
        if kind is not None and kind != "source":
            errors.append(f"{label} kind must be 'source'")
        status = _status(row, label, errors)
        _positive_issue(row, label, errors, required=True)

        path = _relative_path(
            _required_string(row, "path", label, errors), "path", label, errors
        )
        scan = _optional_string(row, "scan", label, errors)
        scan_entry = scans_by_id.get(scan) if scan is not None else None
        if scan is not None and scan_entry is None:
            errors.append(f"{label} references unknown source scan '{scan}'")

        if status == "active":
            if scan is None:
                errors.append(f"active {label} must declare scan")
            elif scan_entry is not None and not scan_entry.active:
                errors.append(
                    f"active {label} may reference only active source scan '{scan}'"
                )
            elif scan_entry is not None and path is not None:
                _check_inventory_scan_path(
                    root, path, scan_entry, label, errors
                )

        symbol = _optional_string(row, "symbol", label, errors)
        needle = _optional_string(row, "needle", label, errors)
        if (symbol is None) == (needle is None):
            errors.append(f"{label} must declare exactly one of symbol or needle")
        selector = symbol if symbol is not None else needle

        for field in ("category", "rationale", "disposition"):
            _required_string(row, field, label, errors)
        if "removal_issue" not in row:
            errors.append(f"{label} must declare removal_issue")
        elif type(row["removal_issue"]) is not int or row["removal_issue"] <= 0:
            errors.append(f"{label} must declare a positive removal_issue")

        if entry_id is None or status is None or path is None or selector is None:
            continue
        contents = _file_contents(root, path, label, errors)
        if contents is None:
            continue
        if selector is not None:
            occurrences = contents.count(selector)
            if occurrences == 0:
                errors.append(
                    f"{label} {'symbol' if symbol is not None else 'needle'} "
                    f"'{selector}' not found in declared file '{path}'"
                )
            elif occurrences != 1:
                errors.append(
                    f"{label} {'symbol' if symbol is not None else 'needle'} "
                    f"'{selector}' occurs {occurrences} times in declared file '{path}'"
                )
        inventories.append(
            Inventory(
                entry_id=entry_id,
                status=status,
                path=path,
                scan=scan,
                symbol=symbol,
                needle=needle,
                selector=selector,
            )
        )
    return inventories


def _scan_candidates(
    root: Path, scan: Scan, errors: list[str]
) -> list[tuple[str, Path]]:
    scan_root = root / scan.root
    try:
        candidates = sorted(
            (path for path in scan_root.glob(scan.glob) if path.is_file()),
            key=lambda path: path.as_posix(),
        )
    except (OSError, ValueError) as error:
        errors.append(
            f"active source scan '{scan.entry_id}' cannot scan "
            f"root '{scan.root}' with glob '{scan.glob}': {error}"
        )
        return []

    safe_candidates: list[tuple[str, Path]] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        resolved = _resolve_under_root(
            root,
            relative,
            f"active source scan '{scan.entry_id}'",
            errors,
            field="candidate",
        )
        if resolved is not None:
            safe_candidates.append((relative, resolved))
    return safe_candidates


def _scan_sources(
    root: Path,
    scans: list[Scan],
    inventories: list[Inventory],
    errors: list[str],
) -> None:
    inventory_tuples = [
        (inventory.scan, inventory.path, inventory.selector)
        for inventory in inventories
        if inventory.status == "active"
        and inventory.scan is not None
        and inventory.selector is not None
    ]
    inventoried = set(inventory_tuples)
    seen_inventory_tuples: set[tuple[str, str, str]] = set()
    for item in sorted(inventory_tuples):
        if item in seen_inventory_tuples:
            errors.append(
                "duplicate source inventory tuple "
                f"scan '{item[0]}' path '{item[1]}' needle '{item[2]}'"
            )
        seen_inventory_tuples.add(item)

    for scan in sorted(scans, key=lambda item: item.entry_id):
        if not scan.active:
            continue
        matches = _scan_candidates(root, scan, errors)
        if not matches:
            errors.append(
                f"active source scan '{scan.entry_id}' glob '{scan.glob}' "
                "matches no files"
            )
            continue
        for relative, path in matches:
            try:
                contents = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                errors.append(
                    f"active source scan '{scan.entry_id}' cannot read "
                    f"'{relative}': {error}"
                )
                continue
            if scan.needle in contents and (
                scan.entry_id,
                relative,
                scan.needle,
            ) not in inventoried:
                errors.append(
                    f"scanned source seam '{relative}' needle '{scan.needle}' "
                    "is not inventoried "
                    f"(scan '{scan.entry_id}')"
                )


def _fixture_suite_candidates(
    root: Path, suite: FixtureSuite, errors: list[str]
) -> list[str]:
    if suite.resolved_root is None or not suite.resolved_root.is_dir():
        return []

    try:
        candidates = sorted(
            (
                path
                for path in suite.resolved_root.glob(suite.glob)
                if path.is_file()
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    except (NotImplementedError, OSError, ValueError) as error:
        errors.append(
            f"active fixture suite '{suite.entry_id}' cannot scan "
            f"root '{suite.root}' with glob '{suite.glob}': {error}"
        )
        return []

    matches: list[str] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        resolved = _resolve_under_root(
            root,
            relative,
            f"active fixture suite '{suite.entry_id}'",
            errors,
            field="candidate",
        )
        if resolved is not None:
            matches.append(relative)
    return matches


def _check_fixture_suites(
    root: Path,
    fixtures: list[Fixture],
    suites: list[FixtureSuite],
    errors: list[str],
) -> None:
    active_by_path: dict[str, list[Fixture]] = {}
    for fixture in fixtures:
        if fixture.status == "active" and fixture.path is not None:
            active_by_path.setdefault(fixture.path, []).append(fixture)
    for rows in active_by_path.values():
        rows.sort(key=lambda fixture: fixture.entry_id)

    suites_by_path: dict[str, list[FixtureSuite]] = {}
    for suite in sorted(suites, key=lambda item: item.entry_id):
        matches = _fixture_suite_candidates(root, suite, errors)
        if not matches:
            if suite.resolved_root is not None and suite.resolved_root.is_dir():
                errors.append(
                    f"active fixture suite '{suite.entry_id}' glob '{suite.glob}' "
                    "matches no files"
                )
            continue

        for relative in matches:
            suites_by_path.setdefault(relative, []).append(suite)
            matching = [
                fixture
                for fixture in active_by_path.get(relative, [])
                if fixture.kind == suite.kind
            ]
            if len(matching) == 1:
                continue
            if not matching:
                wrong_kind = active_by_path.get(relative, [])
                if wrong_kind:
                    for fixture in wrong_kind:
                        errors.append(
                            f"fixture suite '{suite.entry_id}' matched file "
                            f"'{relative}' with fixture '{fixture.entry_id}' kind "
                            f"'{fixture.kind}'; expected '{suite.kind}'"
                        )
                else:
                    errors.append(
                        f"fixture suite '{suite.entry_id}' matched file "
                        f"'{relative}' without exactly one active {suite.kind} "
                        "fixture row"
                    )
            else:
                errors.append(
                    f"fixture suite '{suite.entry_id}' matched file '{relative}' "
                    f"with {len(matching)} active {suite.kind} fixture rows"
                )

    for relative in sorted(suites_by_path):
        matching_suites = suites_by_path[relative]
        if len(matching_suites) > 1:
            suite_ids = ", ".join(
                f"'{suite.entry_id}'"
                for suite in sorted(matching_suites, key=lambda item: item.entry_id)
            )
            errors.append(
                f"fixture path '{relative}' is matched by more than one "
                f"fixture suite: {suite_ids}"
            )

    for fixture in sorted(fixtures, key=lambda item: item.entry_id):
        if (
            fixture.status != "active"
            or fixture.kind not in {"trybuild-fail", "trybuild-pass"}
            or fixture.path is None
        ):
            continue
        matching_suites = [
            suite
            for suite in suites_by_path.get(fixture.path, [])
            if suite.kind == fixture.kind
        ]
        if len(matching_suites) != 1:
            errors.append(
                f"active trybuild fixture '{fixture.entry_id}' path "
                f"'{fixture.path}' must be covered by exactly one matching "
                f"fixture suite; found {len(matching_suites)}"
            )


def _manifest_rows(
    data: dict[str, Any], root: Path, errors: list[str]
) -> None:
    for key in sorted(set(data) - TOP_LEVEL_KEYS):
        errors.append(f"manifest has unknown top-level field '{key}'")
    schema = data.get("schema")
    if schema != SCHEMA:
        errors.append(f"manifest schema must be '{SCHEMA}'")

    arrays: dict[str, list[Any]] = {}
    for key in ("fixtures", "fixture_suites", "source_scans", "source_inventory"):
        value = data.get(key, [])
        if not isinstance(value, list):
            errors.append(f"manifest field '{key}' must be an array of tables")
            arrays[key] = []
        else:
            arrays[key] = value

    ids: dict[str, str] = {}
    fixtures = _fixture_rows(arrays["fixtures"], root, ids, errors)
    suites = _fixture_suite_rows(arrays["fixture_suites"], root, ids, errors)
    scans = _scan_rows(arrays["source_scans"], root, ids, errors)
    inventories = _inventory_rows(
        arrays["source_inventory"],
        root,
        ids,
        {scan.entry_id: scan for scan in scans},
        errors,
    )
    _check_fixture_suites(root, fixtures, suites, errors)
    _scan_sources(root, scans, inventories, errors)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except OSError as error:
        print(f"error: cannot read manifest '{path}': {error}", file=sys.stderr)
        return None
    except tomllib.TOMLDecodeError as error:
        print(f"error: invalid TOML in manifest '{path}': {error}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"error: manifest '{path}' must contain a TOML table", file=sys.stderr)
        return None
    return data


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "scripts" / "storage-ownership-contracts.toml",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    manifest = manifest.resolve()

    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"repository root does not exist or is not a directory: '{root}'")
    data = _load_manifest(manifest)
    if data is None:
        return 1
    _manifest_rows(data, root, errors)

    for error in sorted(set(errors)):
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print("storage ownership contract ledger: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
