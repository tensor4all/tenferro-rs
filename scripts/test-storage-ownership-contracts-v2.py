#!/usr/bin/env python3
"""Executable RED specification for the v2 storage-ownership ledger.

This file is a contract test, not the checker or runner.  The v2 checker and
runner are intentionally absent at this checkpoint.  The tests therefore
describe the required green behavior and are expected to fail until the
implementation phase lands.  Keeping the tests executable prevents the
ledger contract from becoming prose that can silently drift.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check-storage-ownership-contracts.py"
RUNNER = ROOT / "scripts" / "run-storage-ownership-contracts.py"
PRODUCTION_MANIFEST = ROOT / "scripts" / "storage-ownership-contracts.toml"
LEGACY_V1_MANIFEST_FIXTURE = (
    ROOT / "scripts" / "fixtures" / "storage-ownership-contracts-v1.toml"
)
V1_TEST_SUITE = ROOT / "scripts" / "test-check-storage-ownership-contracts.py"
V2_RED_SUITE = ROOT / "scripts" / "test-storage-ownership-contracts-v2.py"
CHECKER_RELATIVE = CHECKER.relative_to(ROOT).as_posix()
LEGACY_FIXTURE_RELATIVE = LEGACY_V1_MANIFEST_FIXTURE.relative_to(ROOT).as_posix()
V1_TEST_SUITE_RELATIVE = V1_TEST_SUITE.relative_to(ROOT).as_posix()
V2_RED_SUITE_RELATIVE = V2_RED_SUITE.relative_to(ROOT).as_posix()

SCHEMA = "tenferro.storage-ownership-contracts.v2"
LEGACY_SCHEMA = "tenferro.storage-ownership-contracts.v1"
GATES = tuple(f"G{number}" for number in range(1, 8))
OBSERVATION_SCHEMA = "tenferro.storage-ownership-observation.v1"
OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "command_id",
        "process_argv",
        "normalized_process_argv",
        "cwd",
        "artifact_path",
        "artifact_sha256",
        "executable",
        "interpreter",
        "nonce",
        "challenge",
    }
)
EXECUTABLE_IDENTITY_FIELDS = frozenset({"requested", "resolved", "sha256"})
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "base_commit",
        "candidate_commit",
        "base_manifest_sha256",
        "candidate_manifest_sha256",
        "executions",
    }
)
RECEIPT_EXECUTION_FIELDS = frozenset(
    {
        "obligation_id",
        "artifact_id",
        "command_id",
        "candidate_commit",
        "exit_code",
        "artifact_sha256",
        "command_sha256",
        "argv",
        "cwd",
        "artifact_path",
        "executable",
        "observation_nonce",
        "observation_challenge",
    }
)

CLI_CONTRACT_SCHEMA = "tenferro.storage-ownership-cli-contract.v1"
CLI_CONTRACT_PROBE = "--contract-schema"
CHECKER_CLI_CONTRACT = {
    "schema": CLI_CONTRACT_SCHEMA,
    "tool": "check-storage-ownership-contracts",
    "role": "checker",
    "manifest_schema": SCHEMA,
    "probe": CLI_CONTRACT_PROBE,
    "options": [
        "--root",
        "--manifest",
        "--base-commit",
        "--receipt",
        "--summary-json",
        "--diagnostics-json",
    ],
}
RUNNER_CLI_CONTRACT = {
    "schema": CLI_CONTRACT_SCHEMA,
    "tool": "run-storage-ownership-contracts",
    "role": "runner",
    "manifest_schema": SCHEMA,
    "probe": CLI_CONTRACT_PROBE,
    "options": [
        "--root",
        "--manifest",
        "--base-commit",
        "--receipt-out",
        "--diagnostics-json",
    ],
}

# These are temporary RED-only sentinels for the exact pre-migration quartet.
# The atomic v2 implementation commit must delete this predicate and all four
# frozen values; they are not a v1 compatibility surface.
LEGACY_V1_QUARTET_SHA256 = (
    (
        "manifest",
        PRODUCTION_MANIFEST,
        "7694da2a07fb702cdc0e2003eeff6b2610d1b8714cd19f78a04b07e4c9082fcf",
    ),
    (
        "checker",
        CHECKER,
        "91ab78217adbb74f8f6bf55a48ec6bb0c6c7eea17b9c51251dcdc092627dc718",
    ),
    (
        "suite",
        V1_TEST_SUITE,
        "e4dbf32d274f7671430a7a1e474016337b60fcab555087e2d111d093acccbdfe",
    ),
    (
        "fixture",
        LEGACY_V1_MANIFEST_FIXTURE,
        "fed8c80e0e5b8969f18a46f729644bad267adeb8a137499638d3a4926ed1b2ec",
    ),
)

CHECKER_CAUSE = "v2-checker-not-implemented"
RUNNER_CAUSE = "v2-runner-not-implemented"
FUTURE_ARTIFACT_CAUSE = "future-production-proof-artifact-not-landed"
MIGRATION_CAUSE = "v2-atomic-migration-not-landed"


class _InventoryAllowlistEntry(NamedTuple):
    kind: str
    relative_path: str
    token: str
    purpose: str
    expected_count: int


# This is an exact, path-scoped allowance for intentional negative evidence in
# the v2 RED suite and the schema-only fixture.  Every entry has a purpose and
# an expected occurrence count; post-migration verification rejects drift.
STORAGE_TOOLING_INVENTORY_ALLOWLIST = (
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        LEGACY_SCHEMA,
        "v2 RED rejects the legacy manifest schema",
        3,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "TOP_LEVEL_KEYS = frozenset",
        "v2 RED names the removed parser surface and tests its rejection",
        4,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "def _fixture_rows(",
        "v2 RED names and tests the removed fixture parser for rejection",
        2,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "def _fixture_suite_rows(",
        "v2 RED names and tests the removed fixture-suite parser for rejection",
        2,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "def _scan_rows(",
        "v2 RED names and tests the removed source-scan parser for rejection",
        2,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "def _inventory_rows(",
        "v2 RED names and tests the removed source-inventory parser for rejection",
        2,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "class FixtureSuite",
        "v2 RED names and tests the removed fixture-suite model for rejection",
        2,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "class CheckerTests(unittest.TestCase)",
        "v2 RED names and tests the removed checker suite for rejection",
        4,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        '"fixtures"',
        "v2 RED enumerates and tests the removed fixtures table",
        7,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        '"fixture_suites"',
        "v2 RED enumerates and tests the removed fixture_suites table",
        6,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        '"source_scans"',
        "v2 RED enumerates and tests the removed source_scans table",
        5,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        '"source_inventory"',
        "v2 RED enumerates and tests the removed source_inventory table",
        5,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "compatibility_mode",
        "v2 RED proves and tests the legacy compatibility mode is absent",
        5,
    ),
    _InventoryAllowlistEntry(
        "content",
        V2_RED_SUITE_RELATIVE,
        "test-check-storage-ownership-contracts.py",
        "v2 RED names and rejects the old suite as a migration-removal target",
        2,
    ),
    _InventoryAllowlistEntry(
        "content",
        LEGACY_FIXTURE_RELATIVE,
        LEGACY_SCHEMA,
        "schema-only negative v1 fixture",
        1,
    ),
    _InventoryAllowlistEntry(
        "path",
        LEGACY_FIXTURE_RELATIVE,
        "v1",
        "schema-only negative v1 fixture path",
        1,
    ),
    _InventoryAllowlistEntry(
        "path",
        CHECKER_RELATIVE,
        "check-storage-ownership-contracts.py",
        "canonical checker path is retained for the v2 implementation",
        1,
    ),
)

STORAGE_TOOLING_SPECIFIC_CONTENT_RULES = (
    (LEGACY_SCHEMA, "legacy manifest schema"),
)
STORAGE_TOOLING_ANCHORED_CONTENT_RULES = (
    ("TOP_LEVEL_KEYS = frozenset", "legacy top-level parser key set"),
    ("class FixtureSuite", "legacy fixture-suite model"),
    ("def _fixture_rows(", "legacy fixture parser"),
    ("def _fixture_suite_rows(", "legacy fixture-suite parser"),
    ("def _scan_rows(", "legacy source-scan parser"),
    ("def _inventory_rows(", "legacy source-inventory parser"),
    ('"fixtures"', "legacy fixtures parser/table key"),
    ('"fixture_suites"', "legacy fixture_suites parser/table key"),
    ('"source_scans"', "legacy source_scans parser/table key"),
    ('"source_inventory"', "legacy source_inventory parser/table key"),
    ("[[fixtures]]", "legacy fixtures TOML table"),
    ("[[fixture_suites]]", "legacy fixture_suites TOML table"),
    ("[[source_scans]]", "legacy source_scans TOML table"),
    ("[[source_inventory]]", "legacy source_inventory TOML table"),
    ("--compatibility", "legacy compatibility flag"),
    ("--compatibility-mode", "legacy compatibility mode flag"),
    ("--compat-mode", "legacy compatibility mode alias"),
    ("--legacy", "legacy compatibility flag"),
    ("--v1", "legacy schema-selection flag"),
    ("compatibility_mode", "legacy compatibility mode"),
    ("legacy_mode", "legacy mode variable"),
    ("allow_legacy", "legacy opt-in variable"),
    ("v1_compat", "legacy compatibility variable"),
    ("class CheckerTests(unittest.TestCase)", "legacy v1 test suite"),
)
STORAGE_TOOLING_SOURCE_ANCHORS = (
    "storage-ownership-contracts",
    "storage_ownership_contracts",
    "tenferro.storage-ownership",
)


class RedExpectedFailure(AssertionError):
    """An intentional RED assertion with a machine-readable cause."""

    def __init__(self, cause: str) -> None:
        if not cause or not cause.replace("-", "").isalnum():
            raise ValueError("expected RED cause must be a stable slug")
        self.cause = cause
        super().__init__(f"intentional RED: {cause}")


class RedExpectedError(RuntimeError):
    """An intentional RED execution error with a machine-readable cause."""

    def __init__(self, cause: str) -> None:
        if not cause or not cause.replace("-", "").isalnum():
            raise ValueError("expected RED cause must be a stable slug")
        self.cause = cause
        super().__init__(f"intentional RED: {cause}")


def _probe_cli_contract(
    tool: Path, expected: dict[str, object]
) -> bool:
    """Accept availability only after a successful exact JSON CLI probe."""
    try:
        result = subprocess.run(
            [sys.executable, str(tool), CLI_CONTRACT_PROBE],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return False
    if result.returncode != 0 or result.stderr != "":
        return False
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return False
    return type(payload) is dict and payload == expected


def _v2_checker_unavailable_cause() -> str | None:
    """Prove checker availability through its machine-readable CLI contract."""
    return (
        None
        if _probe_cli_contract(CHECKER, CHECKER_CLI_CONTRACT)
        else CHECKER_CAUSE
    )


def _v2_runner_unavailable_cause() -> str | None:
    """Prove runner availability through its machine-readable CLI contract."""
    return (
        None
        if _probe_cli_contract(RUNNER, RUNNER_CLI_CONTRACT)
        else RUNNER_CAUSE
    )


def _legacy_tooling_is_current() -> bool:
    """Prove the exact legacy-tool state by frozen bytes, not source shape."""
    for _, path, expected in LEGACY_V1_QUARTET_SHA256:
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        if actual != expected:
            return False
    return True


def _post_migration_sentinel_violations(source: str) -> list[str]:
    """Return RED-only migration sentinels still present in the v2 suite."""
    targets = (
        "7694da2a07fb702c" + "dc0e2003eeff6b2610d1b8714cd19f78a04b07e4c9082fcf",
        "91ab78217adbb74f" + "8f6bf55a48ec6bb0c6c7eea17b9c51251dcdc092627dc718",
        "e4dbf32d274f7671" + "430a7a1e474016337b60fcab555087e2d111d093acccbdfe",
        "fed8c80e0e5b8969" + "f18a46f729644bad267adeb8a137499638d3a4926ed1b2ec",
        "LEGACY_V1_" + "QUARTET_SHA256",
        "_legacy_tooling_" + "is_current",
        "MIGRATION_" + "CAUSE",
        "v2-atomic-migration-" + "not-landed",
    )
    return [target for target in targets if target in source]


def _post_migration_red_event_violations(
    registry: dict[str, object], atomic_test_name: str
) -> list[str]:
    """Return the atomic migration test if its temporary RED event remains."""
    return [atomic_test_name] if atomic_test_name in registry else []


def _inventory_allowlisted(kind: str, relative_path: str, token: str) -> bool:
    return any(
        entry.kind == kind
        and entry.relative_path == relative_path
        and entry.token == token
        for entry in STORAGE_TOOLING_INVENTORY_ALLOWLIST
    )


def _storage_tooling_inventory(root: Path) -> list[tuple[str, str]]:
    """Lexically inventory removed storage tooling; never parse or execute it."""
    scripts_root = root / "scripts"
    violations: list[tuple[str, str]] = []
    checker_path = root / CHECKER_RELATIVE
    if not checker_path.is_file():
        violations.append((CHECKER_RELATIVE, "<missing-canonical-v2-checker>"))
    if not scripts_root.is_dir():
        return violations
    for path in sorted(scripts_root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_file() or path.suffix not in {".py", ".toml"}:
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path == V1_TEST_SUITE_RELATIVE:
            violations.append((relative_path, relative_path))
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            violations.append((relative_path, "<unreadable>"))
            continue
        is_schema_only_negative_fixture = (
            relative_path == LEGACY_FIXTURE_RELATIVE
            and source.strip() == f'schema = "{LEGACY_SCHEMA}"'
        )
        storage_anchored = any(
            anchor in relative_path or anchor in source
            for anchor in STORAGE_TOOLING_SOURCE_ANCHORS
        )
        for token, _purpose in STORAGE_TOOLING_SPECIFIC_CONTENT_RULES:
            if is_schema_only_negative_fixture and token == LEGACY_SCHEMA:
                continue
            if token in source and not _inventory_allowlisted(
                "content", relative_path, token
            ):
                violations.append((relative_path, token))
        if storage_anchored:
            for token, _purpose in STORAGE_TOOLING_ANCHORED_CONTENT_RULES:
                if token in source and not _inventory_allowlisted(
                    "content", relative_path, token
                ):
                    violations.append((relative_path, token))
            for token in ("v1", "legacy"):
                if token in relative_path and not _inventory_allowlisted(
                    "path", relative_path, token
                ):
                    violations.append((relative_path, token))
    return violations


def _storage_tooling_allowlist_drift(
    root: Path,
) -> list[tuple[str, str, str]]:
    """Return path/token/purpose records when intentional evidence changes."""
    drift: list[tuple[str, str, str]] = []
    for entry in STORAGE_TOOLING_INVENTORY_ALLOWLIST:
        path = root / entry.relative_path
        if entry.kind == "path":
            actual_count = (
                entry.relative_path.count(entry.token) if path.is_file() else 0
            )
        else:
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                actual_count = -1
            else:
                actual_count = source.count(entry.token)
        if actual_count != entry.expected_count:
            drift.append(
                (
                    entry.relative_path,
                    entry.token,
                    f"{entry.purpose}: expected {entry.expected_count}, "
                    f"found {actual_count}",
                )
            )
    return drift


def _require_v2_checker() -> None:
    cause = _v2_checker_unavailable_cause()
    if cause is not None:
        raise RedExpectedFailure(cause)


def _require_v2_runner() -> None:
    cause = _v2_runner_unavailable_cause()
    if cause is not None:
        raise RedExpectedError(cause)

# P0 and P1 are independent roots.  P2 has exactly one prerequisite, P1.
# P0 enters the graph only through the atomic CUTOVER cohort.
UNITS = (
    ("P0", 1556, "control-plane"),
    ("P1", 1557, "contract-ledger"),
    ("P2", 1558, "root-claims"),
    ("P3", 1559, "host-ownership"),
    ("P4", 1560, "access-retirement"),
    ("P5", 1561, "allocation-group"),
    ("P6", 1562, "reinterpret"),
    ("P7", 1563, "cuda"),
    ("P8", 1564, "webgpu-metal"),
    ("P9", 1565, "runtime-ad-cutover"),
    ("P10", 1566, "api-normalization"),
    ("P11", 1568, "hardware"),
    ("P12", 1569, "documentation"),
    ("P13-A", 1567, "freeze"),
    ("P13-B", 1567, "closure"),
)

EDGES = (
    ("P1", "P2"),
    ("P2", "P4"),
    ("P4", "P5"),
    ("P3", "P6"),
    ("P9", "P6"),
    ("P6", "P7"),
    ("P6", "P8"),
    ("P7", "P10"),
    ("P8", "P10"),
    ("P10", "P13-A"),
    ("P13-A", "P11"),
    ("P13-A", "P12"),
    ("P11", "P13-B"),
    ("P12", "P13-B"),
)

CUTOVER = {
    "id": "cutover",
    "prerequisites": ("P0", "P5"),
    "members": ("P3", "P9"),
}

# The obligation graph is keyed by unit and gate IDs.  There is deliberately
# no second ownership/fixture/source table: the checker must reject one.
BASE_ACTIVE_OBLIGATIONS = (
    (
        "p0-control-plane",
        "P0",
        ("G3",),
        "artifact-control-plane",
        "crates/tenferro-runtime/tests/execution_engine_identity.rs",
        "rust-test",
        "cmd-control-plane",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-runtime", "--test", "execution_engine_identity"),
        (),
    ),
    (
        "p1-ledger",
        "P1",
        ("G1", "G3", "G5"),
        "artifact-ledger",
        "scripts/storage-ownership-contracts.toml",
        "manifest",
        "cmd-ledger",
        "python-test",
        ("python3", "scripts/check-storage-ownership-contracts.py"),
        ("scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"),
    ),
    (
        "p1-contract-document",
        "P1",
        ("G6",),
        "artifact-contract-document",
        "docs/design/storage-ownership-contracts.md",
        "documentation",
        "cmd-contract-document",
        "doc-check",
        ("python3", "scripts/check-storage-design-docs.py"),
        ("scripts/check-storage-design-docs.py", "docs/design/storage-ownership-contracts.md"),
    ),
    (
        "p1-api-parity",
        "P1",
        ("G4",),
        "artifact-api-parity",
        "crates/tenferro-tensor/tests/storage_api_parity.rs",
        "rust-test",
        "cmd-api-parity",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_api_parity"),
        (),
    ),
    (
        "p2-root-claims",
        "P2",
        ("G1",),
        "artifact-root-claims",
        "crates/tenferro-tensor/tests/storage_root_claims.rs",
        "rust-test",
        "cmd-root-claims",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_root_claims"),
        (),
    ),
)

DEFERRED_OBLIGATIONS = (
    (
        "p4-production-borrow-contract",
        "P4",
        ("G1", "G4"),
        "artifact-production-borrow-contract",
        "crates/tenferro-tensor/tests/storage_borrow_contract.rs",
        "compile-contract",
        "cmd-production-borrow-contract",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_borrow_contract"),
        (),
    ),
    (
        "p3-host-owner",
        "P3",
        ("G4",),
        "artifact-owner-compile",
        "crates/tenferro-tensor/tests/ui/storage/fail/owned_storage_not_clone.rs",
        "trybuild",
        "cmd-owner-compile",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"),
        (),
    ),
    (
        "p3-auto-trait-contract",
        "P3",
        ("G1", "G4"),
        "artifact-storage-auto-traits",
        "crates/tenferro-tensor/tests/storage_auto_traits.rs",
        "compile-contract",
        "cmd-storage-auto-traits",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_auto_traits"),
        (),
    ),
    (
        "p4-access-retirement",
        "P4",
        ("G1", "G3"),
        "artifact-corruption-map",
        "crates/tenferro-tensor/src/storage/tests/corruption_map.rs",
        "corruption-test",
        "cmd-corruption-map",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--lib", "storage::tests::corruption_map"),
        (),
    ),
    (
        "p4-provider-release-lifecycle",
        "P4",
        ("G1", "G3"),
        "artifact-provider-release-lifecycle",
        "crates/tenferro-tensor/src/storage/tests/provider_release_lifecycle.rs",
        "provider-test",
        "cmd-provider-release-lifecycle",
        "cargo-test",
        (
            "cargo",
            "test",
            "-p",
            "tenferro-tensor",
            "--lib",
            "storage::tests::provider_release_lifecycle",
        ),
        (),
    ),
    (
        "p5-allocation-group",
        "P5",
        ("G2", "G5"),
        "artifact-allocation-group",
        "crates/tenferro-tensor/tests/storage_allocation_group.rs",
        "rust-test",
        "cmd-allocation-group",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_allocation_group"),
        (),
    ),
    (
        "p9-submission",
        "P9",
        ("G3",),
        "artifact-submit-compile",
        "crates/tenferro-tensor/tests/ui/storage/pass/consuming_submission.rs",
        "trybuild",
        "cmd-submit-compile",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"),
        (),
    ),
    (
        "p6-reinterpret",
        "P6",
        ("G4",),
        "artifact-reinterpret",
        "crates/tenferro-tensor/tests/storage_reinterpret.rs",
        "rust-test",
        "cmd-reinterpret",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_reinterpret"),
        (),
    ),
    (
        "p7-cuda",
        "P7",
        ("G1", "G3"),
        "artifact-cuda-provider",
        "crates/tenferro-cuda/tests/storage_provider.rs",
        "hardware-test",
        "cmd-cuda-provider",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-cuda", "--test", "storage_provider"),
        (),
    ),
    (
        "p8-webgpu-metal",
        "P8",
        ("G1", "G3"),
        "artifact-webgpu-metal-provider",
        "crates/tenferro-wgpu/tests/storage_provider.rs",
        "hardware-test",
        "cmd-webgpu-metal-provider",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-wgpu", "--test", "storage_provider"),
        (),
    ),
    (
        "p10-api-normalization",
        "P10",
        ("G4",),
        "artifact-api-normalization",
        "crates/tenferro-tensor/tests/storage_public_api.rs",
        "rust-test",
        "cmd-api-normalization",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_public_api"),
        (),
    ),
    (
        "p13-freeze",
        "P13-A",
        ("G1", "G2", "G3", "G4", "G5", "G6", "G7"),
        "artifact-contract-freeze",
        "docs/design/storage-contract-freeze.md",
        "documentation",
        "cmd-contract-freeze",
        "doc-check",
        ("python3", "scripts/check-storage-design-docs.py"),
        ("scripts/check-storage-design-docs.py", "docs/design/storage-contract-freeze.md"),
    ),
    (
        "p11-hardware",
        "P11",
        ("G1", "G3"),
        "artifact-hardware-matrix",
        "docs/testing/storage-hardware-matrix.md",
        "documentation",
        "cmd-hardware-matrix",
        "doc-check",
        ("python3", "scripts/check-storage-design-docs.py"),
        ("scripts/check-storage-design-docs.py", "docs/testing/storage-hardware-matrix.md"),
    ),
    (
        "p12-documentation",
        "P12",
        ("G6",),
        "artifact-storage-guide",
        "docs/storage-ownership.md",
        "documentation",
        "cmd-storage-guide",
        "doc-check",
        ("python3", "scripts/check-storage-design-docs.py"),
        ("scripts/check-storage-design-docs.py", "docs/storage-ownership.md"),
    ),
    (
        "p13-closure",
        "P13-B",
        ("G1", "G2", "G3", "G4", "G5", "G6", "G7"),
        "artifact-storage-closure",
        "docs/worklogs/storage-redesign-closure.md",
        "documentation",
        "cmd-storage-closure",
        "doc-check",
        ("python3", "scripts/check-storage-design-docs.py"),
        ("scripts/check-storage-design-docs.py", "docs/worklogs/storage-redesign-closure.md"),
    ),
)

ALL_OBLIGATIONS = BASE_ACTIVE_OBLIGATIONS + DEFERRED_OBLIGATIONS
CANONICAL_COMMAND_IDS = tuple(dict.fromkeys(row[6] for row in ALL_OBLIGATIONS))
CANONICAL_COMMAND_ARGV = {
    command_id: next(row[8] for row in ALL_OBLIGATIONS if row[6] == command_id)
    for command_id in CANONICAL_COMMAND_IDS
}
CANONICAL_COMMAND_ARGV_COORDINATES = tuple(
    (command_id, index)
    for command_id in CANONICAL_COMMAND_IDS
    for index in range(len(CANONICAL_COMMAND_ARGV[command_id]))
)
COMMAND_ARGV_LENGTH_CASES = (
    "missing-final-argument",
    "appended-extra-argument",
)
CUTOVER_CANDIDATE_OBLIGATIONS = frozenset(
    row[0] for row in DEFERRED_OBLIGATIONS if row[1] in {"P3", "P4", "P5", "P9"}
)
CUTOVER_PARTIAL_OBLIGATIONS = frozenset(
    row[0] for row in DEFERRED_OBLIGATIONS if row[1] in {"P4", "P5", "P9"}
)

# The v2 diagnostic envelope is intentionally narrow.  A checker must emit
# exactly these identifying fields for each code; a broad "all known errors"
# response is not a valid witness for a one-fault RED case.
DIAGNOSTIC_FIELDS = {
    "E_SCHEMA_VERSION": frozenset({"actual"}),
    "E_SCHEMA_PARALLEL_TABLE": frozenset({"table"}),
    "E_SCHEMA_UNKNOWN_TABLE": frozenset({"table"}),
    "E_OBLIGATION_TAGGED_STATE": frozenset({"obligation_id"}),
    "E_UNIT_OBLIGATION_MISSING": frozenset({"unit"}),
    "E_GRAPH_P2_PREREQUISITE": frozenset({"unit"}),
    "E_GRAPH_DUPLICATE_EDGE": frozenset({"from", "to"}),
    "E_GRAPH_UNKNOWN_UNIT": frozenset({"unit"}),
    "E_COHORT_DEFINITION": frozenset({"cohort_id"}),
    "E_COHORT_PARTIAL_PROMOTION": frozenset({"cohort_id"}),
    "E_COHORT_PREREQUISITE_INCOMPLETE": frozenset({"unit", "obligation_id"}),
    "E_OBSOLETE_OWNERSHIP_TABLE": frozenset({"table"}),
    "E_ARTIFACT_SYNTHETIC_TERMINAL": frozenset({"artifact_id"}),
    "E_ARTIFACT_DUPLICATE_TARGET": frozenset({"artifact_id"}),
    "E_ARTIFACT_MISSING": frozenset({"artifact_id"}),
    "E_PATH_ESCAPE": frozenset({"obligation_id"}),
    "E_PATH_SYMLINK_ESCAPE": frozenset({"obligation_id"}),
    "E_DEFERRED_ARTIFACT_EXISTS": frozenset({"obligation_id"}),
    "E_COMMAND_KIND": frozenset({"command_id", "kind"}),
    "E_COMMAND_ARGV": frozenset({"command_id"}),
    "E_COMMAND_ARGV_LENGTH": frozenset({"command_id", "expected", "actual"}),
    "E_COMMAND_ARGV_BINDING": frozenset(
        {"command_id", "index", "expected", "actual"}
    ),
    "E_COMMAND_CWD_ESCAPE": frozenset({"command_id", "cwd"}),
    "E_COMMAND_PATH_ESCAPE": frozenset({"command_id"}),
    "E_COMMAND_ARGV_PATH_ESCAPE": frozenset(
        {"command_id", "index", "argument"}
    ),
    "E_COMMAND_CWD_SYMLINK_ESCAPE": frozenset({"command_id", "cwd"}),
    "E_COMMAND_ARGV_SYMLINK_ESCAPE": frozenset(
        {"command_id", "index", "argument"}
    ),
    "E_COMMAND_ARTIFACT_BINDING": frozenset({"command_id"}),
    "E_COMMAND_TARGET_BINDING": frozenset({"command_id"}),
    "E_COMMAND_ID_CONFLICT": frozenset({"command_id"}),
    "E_COMMAND_FAILED": frozenset({"command_id", "exit_code"}),
    "E_PROMOTION_IDENTITY": frozenset({"obligation_id"}),
    "E_RECEIPT_COMMIT": frozenset({"actual_head"}),
    "E_RECEIPT_SHAPE": frozenset({"field", "expected", "actual"}),
    "E_RECEIPT_MANIFEST_DIGEST": frozenset({"field", "expected", "actual"}),
    "E_RECEIPT_DIGEST": frozenset(
        {"obligation_id", "field", "expected", "actual"}
    ),
    "E_RECEIPT_PATH_IDENTITY": frozenset(
        {"obligation_id", "field", "expected", "actual"}
    ),
    "E_RECEIPT_EXECUTION_BINDING": frozenset(
        {"obligation_id", "field", "expected", "actual"}
    ),
    "E_RECEIPT_OBSERVATION_BINDING": frozenset(
        {"obligation_id", "field", "expected", "actual"}
    ),
    "E_RECEIPT_INCOMPLETE": frozenset({"obligation_id"}),
    "E_TERMINAL_DECLARED": frozenset({"field"}),
}
DIAGNOSTIC_FIELD_TYPES = {
    "E_COMMAND_ARGV_LENGTH": {
        "command_id": str,
        "expected": int,
        "actual": int,
    },
    "E_RECEIPT_MANIFEST_DIGEST": {
        "field": str,
        "expected": str,
        "actual": str,
    },
    "E_RECEIPT_SHAPE": {
        "field": str,
        "expected": str,
        "actual": str,
    },
    "E_RECEIPT_DIGEST": {
        "obligation_id": str,
        "field": str,
        "expected": str,
        "actual": str,
    },
    "E_RECEIPT_PATH_IDENTITY": {
        "obligation_id": str,
        "field": str,
        "expected": str,
        "actual": str,
    },
}

# This is the intentional RED boundary for the current checkpoint.  It is
# machine-readable and exact at test/subtest granularity: a new failure, an
# error instead of an assertion failure, or an unexpected subtest is never
# absorbed by a broad "known RED" label.  The registry must be updated in the
# same change that lands the corresponding checker/runner/artifact.
RED_EXPECTED_FAILURES = {
    "test_atomic_v2_migration_removes_legacy_surface": {
        "cause": "v2-atomic-migration-not-landed",
    },
    "test_checker_cli_schema_probe_is_required": {
        "cause": "v2-checker-not-implemented",
    },
    "test_legacy_v1_fixture_and_source_tables_are_rejected": {
        "cause": "v2-checker-not-implemented",
    },
    "test_artifact_paths_are_unique_repository_relative_and_real": {
        "cause": "v2-checker-not-implemented",
    },
    "test_canonical_future_lifecycle_proof_commands_execute": {
        "cause": "future-production-proof-artifact-not-landed",
        "subtests": [
            {"obligation_id": row[0]} for row in DEFERRED_OBLIGATIONS
        ],
    },
    "test_canonical_graph_keeps_p0_p1_roots_and_p2_only_depends_on_p1": {
        "cause": "v2-checker-not-implemented",
    },
    "test_v2_checker_rejects_legacy_production_manifest_until_migration": {
        "cause": "v2-checker-not-implemented",
    },
    "test_command_allowlist_is_typed_and_fail_closed": {
        "cause": "v2-checker-not-implemented",
    },
    "test_command_argv_exact_allowlist_is_enforced": {
        "cause": "v2-checker-not-implemented",
        "subtests": [
            {"command_id": command_id, "index": index}
            for command_id, index in CANONICAL_COMMAND_ARGV_COORDINATES
        ],
    },
    "test_command_argv_length_is_enforced": {
        "cause": "v2-checker-not-implemented",
        "subtests": [
            {"command_id": command_id, "case": case}
            for command_id in CANONICAL_COMMAND_IDS
            for case in COMMAND_ARGV_LENGTH_CASES
        ],
    },
    "test_command_cwd_confinement_rejects_absolute_and_parent_escape": {
        "cause": "v2-checker-not-implemented",
        "subtests": [
            {"cwd": "/tmp/ledger-command-outside"},
            {"cwd": "../ledger-command-outside"},
        ],
    },
    "test_command_argv_path_escape_ignores_path_args_metadata": {
        "cause": "v2-checker-not-implemented",
        "subtests": [
            {"argument": "/tmp/ledger-command-outside.py"},
            {"argument": "../ledger-command-outside.py"},
        ],
    },
    "test_command_symlink_confinement_rejects_cwd_and_argv_escape": {
        "cause": "v2-checker-not-implemented",
        "subtests": [
            {"case": "cwd-symlink"},
            {"case": "argv-symlink"},
        ],
    },
    "test_command_must_bind_to_exact_artifact_and_target_links": {
        "cause": "v2-checker-not-implemented",
    },
    "test_cutover_is_atomic_and_partial_activation_is_rejected": {
        "cause": "v2-checker-not-implemented",
    },
    "test_cutover_requires_non_vacuous_p0_and_p5_receipt_proof": {
        "cause": "v2-runner-not-implemented",
        "subtests": [{"unit": "P0"}, {"unit": "P5"}],
    },
    "test_deferred_artifact_cannot_be_promoted_by_existing_file_alone": {
        "cause": "v2-checker-not-implemented",
    },
    "test_every_canonical_unit_has_required_obligations": {
        "cause": "v2-checker-not-implemented",
    },
    "test_graph_rejects_duplicate_and_unknown_target_links": {
        "cause": "v2-checker-not-implemented",
    },
    "test_matching_fake_commit_ids_cannot_replace_git_identity": {
        "cause": "v2-runner-not-implemented",
    },
    "test_nominal_v2_manifest_is_green": {
        "cause": "v2-checker-not-implemented",
    },
    "test_one_tagged_obligation_table_replaces_parallel_status_tables": {
        "cause": "v2-checker-not-implemented",
    },
    "test_post_receipt_artifact_mutation_is_rejected": {
        "cause": "v2-runner-not-implemented",
    },
    "test_post_receipt_base_manifest_mutation_digest_is_rejected": {
        "cause": "v2-runner-not-implemented",
    },
    "test_post_receipt_command_path_symlink_retarget_is_rejected": {
        "cause": "v2-runner-not-implemented",
    },
    "test_post_receipt_artifact_symlink_retarget_with_identical_external_bytes_is_rejected": {
        "cause": "v2-runner-not-implemented",
    },
    "test_post_receipt_cwd_symlink_retarget_is_rejected": {
        "cause": "v2-runner-not-implemented",
    },
    "test_promotion_preserves_immutable_identity_and_binds_receipt_to_candidate": {
        "cause": "v2-runner-not-implemented",
    },
    "test_promotion_rejects_artifact_or_command_identity_change": {
        "cause": "v2-checker-not-implemented",
        "subtests": [{"case": "artifact"}, {"case": "command"}],
    },
    "test_real_symlink_escape_is_rejected": {
        "cause": "v2-checker-not-implemented",
    },
    "test_receipt_digests_bind_exact_manifest_artifact_and_command": {
        "cause": "v2-runner-not-implemented",
        "subtests": [
            {"digest_kind": "manifest"},
            {"digest_kind": "artifact"},
            {"digest_kind": "command"},
        ],
    },
    "test_receipt_execution_identity_mutations_are_rejected": {
        "cause": "v2-runner-not-implemented",
        "subtests": [
            {"field": "artifact_id"},
            {"field": "command_id"},
            {"field": "candidate_commit"},
            {"field": "argv"},
            {"field": "cwd"},
            {"field": "artifact_path"},
            {"field": "executable"},
            {"field": "observation_nonce"},
            {"field": "observation_challenge"},
        ],
    },
    "test_receipt_envelope_missing_and_extra_fields_are_rejected": {
        "cause": "v2-runner-not-implemented",
        "subtests": [
            {"case": "missing-field", "field": field}
            for field in sorted(RECEIPT_FIELDS)
        ]
        + [{"case": "extra-field", "field": "terminal"}],
    },
    "test_runner_cli_schema_probe_is_required": {
        "cause": "v2-runner-not-implemented",
    },
    "test_checker_accepts_canonical_cargo_command_with_path_shim": {
        "cause": "v2-checker-not-implemented",
    },
    "test_runner_emits_exact_candidate_bound_receipt": {
        "cause": "v2-runner-not-implemented",
    },
    "test_runner_executes_canonical_cargo_command_with_path_shim": {
        "cause": "v2-runner-not-implemented",
    },
    "test_child_observation_mutations_are_rejected": {
        "cause": "v2-runner-not-implemented",
        "subtests": [
            {"case": "missing"},
            {"case": "duplicate"},
            {"case": "forged"},
            {"case": "swapped"},
        ],
    },
    "test_runner_is_fail_closed_on_command_failure": {
        "cause": "v2-runner-not-implemented",
    },
    "test_runner_receipt_digests_match_independent_sha256_calculations": {
        "cause": "v2-runner-not-implemented",
    },
    "test_source_and_fixture_tables_cannot_reappear_as_parallel_authority": {
        "cause": "v2-checker-not-implemented",
        "subtests": [
            {"table": "fixtures"},
            {"table": "fixture_suites"},
            {"table": "source_scans"},
            {"table": "source_inventory"},
            {"table": "ownerships"},
        ],
    },
    "test_stale_p3_p4_ownership_rows_are_not_a_second_authority": {
        "cause": "v2-checker-not-implemented",
    },
    "test_synthetic_terminal_artifacts_and_terminal_flags_are_rejected": {
        "cause": "v2-checker-not-implemented",
    },
    "test_terminal_state_is_derived_from_obligations_and_receipts": {
        "cause": "v2-checker-not-implemented",
    },
    "test_terminal_true_requires_zero_deferred_and_complete_receipt": {
        "cause": "v2-runner-not-implemented",
    },
    "test_v1_is_rejected_without_compatibility_mode": {
        "cause": "v2-checker-not-implemented",
    },
}

# The cause registry fixes the event shape as well as its prose label.  A
# missing checker is an expected assertion failure, while attempting to invoke
# a missing runner is an expected execution error.  Subtests retain the same
# distinction instead of being collapsed into one generic event kind.
RED_CAUSE_EVENT_SHAPES = {
    CHECKER_CAUSE: ("failure", "RedExpectedFailure"),
    RUNNER_CAUSE: ("error", "RedExpectedError"),
    FUTURE_ARTIFACT_CAUSE: ("failure", "RedExpectedFailure"),
    MIGRATION_CAUSE: ("failure", "RedExpectedFailure"),
}


def _quote(value: str) -> str:
    return json.dumps(value)


def _array(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_quote(value) for value in values) + "]"


def _registry(*, edges: tuple[tuple[str, str], ...] = EDGES) -> str:
    rows = ["[registry]"]
    rows.extend(
        textwrap.dedent(
            f'''\
            [[registry.gates]]
            id = "{gate}"
            title = "{gate} storage ownership gate"
            '''
        ).strip()
        for gate in GATES
    )
    rows.extend(
        textwrap.dedent(
            f'''\
            [[registry.units]]
            id = "{unit}"
            issue = {issue}
            name = "{name}"
            '''
        ).strip()
        for unit, issue, name in UNITS
    )
    rows.extend(
        textwrap.dedent(
            f'''\
            [[registry.edges]]
            from = "{source}"
            to = "{target}"
            '''
        ).strip()
        for source, target in edges
    )
    rows.append(
        textwrap.dedent(
            f'''\
            [[registry.cohorts]]
            id = "{CUTOVER["id"]}"
            prerequisites = {_array(CUTOVER["prerequisites"])}
            members = {_array(CUTOVER["members"])}
            '''
        ).strip()
    )
    return "\n\n".join(rows)


def _command(
    *,
    command_id: str,
    kind: str,
    argv: tuple[str, ...],
    path_args: tuple[str, ...],
    artifact_id: str,
    artifact_path: str,
    marker: bool,
) -> str:
    if marker:
        argv = ("python3", f"markers/{command_id}.py", artifact_path)
        path_args = (f"markers/{command_id}.py", artifact_path)
        kind = "python-test"
    return (
        "command = { "
        f"id = {_quote(command_id)}, "
        f"kind = {_quote(kind)}, "
        f"argv = {_array(argv)}, "
        'cwd = ".", '
        f"path_args = {_array(path_args)}, "
        f"artifact_id = {_quote(artifact_id)} "
        "}"
    )


def _obligation(
    *,
    entry_id: str,
    unit: str,
    gates: tuple[str, ...],
    artifact_id: str,
    artifact_path: str,
    artifact_kind: str,
    command_id: str,
    command_kind: str,
    argv: tuple[str, ...],
    path_args: tuple[str, ...],
    deferred_unit: str | None,
    marker: bool,
) -> str:
    state = (
        'state = { kind = "active" }'
        if deferred_unit is None
        else (
            "state = { "
            f'kind = "deferred", activation_unit = "{deferred_unit}", '
            'promotion = { mode = "activate-in-place" } }'
        )
    )
    return textwrap.dedent(
        f'''\
        [[obligations]]
        id = "{entry_id}"
        unit = "{unit}"
        gates = {_array(gates)}
        artifact = {{ id = "{artifact_id}", kind = "{artifact_kind}", path = "{artifact_path}" }}
        {_command(command_id=command_id, kind=command_kind, argv=argv, path_args=path_args, artifact_id=artifact_id, artifact_path=artifact_path, marker=marker)}
        {state}
        rationale = "The obligation is one immutable artifact-command contract owned by one graph unit."
        '''
    ).strip()


def valid_manifest(
    *,
    schema: str = SCHEMA,
    edges: tuple[tuple[str, str], ...] = EDGES,
    marker: bool = False,
    marker_exclude: frozenset[str] = frozenset(),
    active_override: dict[str, str] | None = None,
    deferred_override: dict[str, str] | None = None,
    promote: frozenset[str] = frozenset(),
) -> str:
    active_override = active_override or {}
    deferred_override = deferred_override or {}
    rows = [f"schema = {_quote(schema)}", _registry(edges=edges)]
    for row in BASE_ACTIVE_OBLIGATIONS:
        entry_id, unit, gates, artifact_id, path, artifact_kind, command_id, command_kind, argv, path_args = row
        replacement = active_override.get(entry_id)
        rows.append(
            _obligation(
                entry_id=entry_id,
                unit=unit,
                gates=gates,
                artifact_id=artifact_id,
                artifact_path=replacement or path,
                artifact_kind=artifact_kind,
                command_id=command_id,
                command_kind=command_kind,
                argv=argv,
                path_args=path_args,
                deferred_unit=None,
                marker=marker and entry_id not in marker_exclude,
            )
        )
    for row in DEFERRED_OBLIGATIONS:
        entry_id, unit, gates, artifact_id, path, artifact_kind, command_id, command_kind, argv, path_args = row
        replacement = deferred_override.get(entry_id)
        rows.append(
            _obligation(
                entry_id=entry_id,
                unit=unit,
                gates=gates,
                artifact_id=artifact_id,
                artifact_path=replacement or path,
                artifact_kind=artifact_kind,
                command_id=command_id,
                command_kind=command_kind,
                argv=argv,
                path_args=path_args,
                deferred_unit=None if entry_id in promote else unit,
                marker=marker and entry_id not in marker_exclude,
            )
        )
    return "\n\n".join(rows) + "\n"


def repository_files() -> dict[str, str]:
    files = {
        "scripts/storage-ownership-contracts.toml": 'schema = "legacy-placeholder"\n',
        "scripts/check-storage-ownership-contracts.py": "raise SystemExit(0)\n",
        "scripts/check-storage-design-docs.py": "raise SystemExit(0)\n",
        "docs/design/storage-ownership-contracts.md": "# Storage ownership contracts\n",
        "crates/tenferro-tensor/tests/storage_api_parity.rs": "fn parity_contract() {}\n",
    }
    for row in BASE_ACTIVE_OBLIGATIONS:
        files[row[4]] = "fn active_contract() {}\n"
    return files


def marker_files(manifest: str | None = None) -> dict[str, str]:
    files = repository_files()
    marker_manifest = tomllib.loads(manifest or valid_manifest(marker=True))
    commands = {
        row["command"]["id"]: row["command"]
        for row in marker_manifest["obligations"]
    }
    for row in ALL_OBLIGATIONS:
        command_id = row[6]
        requested_executable = commands[command_id]["argv"][0]
        files[f"markers/{command_id}.py"] = textwrap.dedent(
            f'''\
            import hashlib
            import json
            import secrets
            import shutil
            import sys
            from pathlib import Path

            command_id = {command_id!r}
            requested_executable = {requested_executable!r}
            resolved_requested = shutil.which(requested_executable)
            if resolved_requested is None:
                raise SystemExit(127)
            executable = Path(resolved_requested).resolve()
            interpreter = Path(sys.executable).resolve()
            artifact = Path(sys.argv[-1]).resolve()
            cwd = Path.cwd().resolve()
            artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
            nonce = secrets.token_hex(16)
            challenge = hashlib.sha256(
                "\\0".join(
                    (command_id, nonce, str(cwd), str(artifact), artifact_sha256)
                ).encode("utf-8")
            ).hexdigest()
            observation = {{
                "schema": {OBSERVATION_SCHEMA!r},
                "command_id": command_id,
                "process_argv": [str(interpreter), *sys.argv],
                "normalized_process_argv": [
                    str(executable),
                    str(Path(sys.argv[0]).resolve()),
                    *sys.argv[1:-1],
                    str(artifact),
                ],
                "cwd": str(cwd),
                "artifact_path": str(artifact),
                "artifact_sha256": artifact_sha256,
                "executable": {{
                    "requested": requested_executable,
                    "resolved": str(executable),
                    "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                }},
                "interpreter": str(interpreter),
                "nonce": nonce,
                "challenge": challenge,
            }}
            observation_dir = Path("observations")
            observation_dir.mkdir(parents=True, exist_ok=True)
            (observation_dir / f"{{command_id}}-{{nonce}}.json").write_text(
                json.dumps(observation, sort_keys=True) + "\\n", encoding="utf-8"
            )
            Path("runner.log").open("a", encoding="utf-8").write("{command_id}\\n")
            '''
        )
    return files


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"replacement target must occur exactly once, found {count}: {old!r}")
    return source.replace(old, new, 1)


def _replace_in_obligation(source: str, obligation_id: str, old: str, new: str) -> str:
    blocks = source.split("\n\n")
    matches = [index for index, block in enumerate(blocks) if f'id = "{obligation_id}"' in block]
    if len(matches) != 1:
        raise AssertionError(f"obligation block must occur exactly once: {obligation_id}")
    index = matches[0]
    blocks[index] = _replace_once(blocks[index], old, new)
    return "\n\n".join(blocks)


def _cargo_path_shim_manifest() -> str:
    """Markerize all fixtures except the existing canonical cargo obligation."""
    return valid_manifest(
        marker=True,
        marker_exclude=frozenset({"p0-control-plane"}),
    )


def _cargo_path_shim_source() -> str:
    row = next(row for row in BASE_ACTIVE_OBLIGATIONS if row[0] == "p0-control-plane")
    command_id = row[6]
    artifact_path = row[4]
    expected_args = list(row[8][1:])
    return textwrap.dedent(
        f'''\
        #!/usr/bin/env python3
        import hashlib
        import json
        import secrets
        import shutil
        import sys
        from pathlib import Path

        command_id = {command_id!r}
        requested_executable = {row[8][0]!r}
        expected_args = {expected_args!r}
        if sys.argv[1:] != expected_args:
            raise SystemExit(64)
        resolved_requested = shutil.which(requested_executable)
        if resolved_requested is None:
            raise SystemExit(127)
        executable = Path(resolved_requested).resolve()
        interpreter = Path(sys.executable).resolve()
        artifact = Path({artifact_path!r}).resolve()
        cwd = Path.cwd().resolve()
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        nonce = secrets.token_hex(16)
        challenge = hashlib.sha256(
            "\\0".join(
                (command_id, nonce, str(cwd), str(artifact), artifact_sha256)
            ).encode("utf-8")
        ).hexdigest()
        observation = {{
            "schema": {OBSERVATION_SCHEMA!r},
            "command_id": command_id,
            "process_argv": [str(interpreter), *sys.argv],
            "normalized_process_argv": [str(executable), *sys.argv[1:]],
            "cwd": str(cwd),
            "artifact_path": str(artifact),
            "artifact_sha256": artifact_sha256,
            "executable": {{
                "requested": requested_executable,
                "resolved": str(executable),
                "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            }},
            "interpreter": str(interpreter),
            "nonce": nonce,
            "challenge": challenge,
        }}
        observation_dir = Path("observations")
        observation_dir.mkdir(parents=True, exist_ok=True)
        (observation_dir / f"{{command_id}}-{{nonce}}.json").write_text(
            json.dumps(observation, sort_keys=True) + "\\n", encoding="utf-8"
        )
        Path("runner.log").open("a", encoding="utf-8").write("{{command_id}}\\n")
        '''
    )


def _command_argv_occurrences(
    source: str, command_id: str
) -> tuple[list[str], list[int], tuple[str, ...]]:
    rows = [row for row in ALL_OBLIGATIONS if row[6] == command_id]
    if not rows:
        raise AssertionError(f"unknown canonical command ID: {command_id}")
    expected_argv = rows[0][8]
    if any(row[8] != expected_argv for row in rows):
        raise AssertionError(
            f"shared command ID has inconsistent canonical argv: {command_id}"
        )
    blocks = source.split("\n\n")
    matches = [
        block_index
        for block_index, block in enumerate(blocks)
        if f'id = "{command_id}"' in block
    ]
    if len(matches) != len(rows):
        raise AssertionError(
            f"command ID occurrence count changed for {command_id}: "
            f"expected {len(rows)}, found {len(matches)}"
        )
    return blocks, matches, expected_argv


def _replace_command_argv_in_all_occurrences(
    source: str, command_id: str, *, index: int, actual: str
) -> tuple[str, str]:
    blocks, matches, expected_argv = _command_argv_occurrences(source, command_id)
    if not 0 <= index < len(expected_argv):
        raise AssertionError(f"argv index out of range for {command_id}: {index}")
    mutated_argv = expected_argv[:index] + (actual,) + expected_argv[index + 1 :]
    old = f"argv = {_array(expected_argv)}"
    new = f"argv = {_array(mutated_argv)}"
    for block_index in matches:
        blocks[block_index] = _replace_once(blocks[block_index], old, new)
    return "\n\n".join(blocks), expected_argv[index]


def _replace_command_argv_length_in_all_occurrences(
    source: str, command_id: str, *, case: str
) -> tuple[str, int, int]:
    blocks, matches, expected_argv = _command_argv_occurrences(source, command_id)
    if not expected_argv:
        raise AssertionError(f"canonical command argv must not be empty: {command_id}")
    if case == "missing-final-argument":
        mutated_argv = expected_argv[:-1]
    elif case == "appended-extra-argument":
        mutated_argv = expected_argv + ("unexpected-extra-argument",)
    else:
        raise AssertionError(f"unknown argv length case: {case}")
    old = f"argv = {_array(expected_argv)}"
    new = f"argv = {_array(mutated_argv)}"
    for block_index in matches:
        blocks[block_index] = _replace_once(blocks[block_index], old, new)
    return "\n\n".join(blocks), len(expected_argv), len(mutated_argv)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout.strip()


def _git_show_bytes(root: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout.decode() + result.stderr.decode())
    return result.stdout


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_resolved_path(root: Path, relative_path: str) -> str:
    return _sha256_bytes((root / relative_path).resolve().read_bytes())


def _sha256_command(command: dict[str, object]) -> str:
    canonical = json.dumps(command, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(canonical)


def _resolve_executable_identity(
    requested: str, search_path: str | None = None
) -> dict[str, str]:
    resolved = shutil.which(requested, path=search_path)
    if resolved is None:
        raise AssertionError(f"fixture executable is not resolvable: {requested}")
    executable = Path(resolved).resolve()
    return {
        "requested": requested,
        "resolved": str(executable),
        "sha256": _sha256_bytes(executable.read_bytes()),
    }


def _resolve_manifest_path(root: Path, value: str) -> str:
    path = Path(value)
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def _expected_fixture_argv(
    root: Path,
    command: dict[str, object],
    *,
    executable_search_path: str | None = None,
) -> list[str]:
    argv = command["argv"]
    path_args = set(command.get("path_args", []))
    identity = _resolve_executable_identity(argv[0], executable_search_path)
    return [
        identity["resolved"],
        *(
            _resolve_manifest_path(root, value) if value in path_args else value
            for value in argv[1:]
        ),
    ]


def _observation_challenge(
    command_id: str,
    nonce: str,
    cwd: str,
    artifact_path: str,
    artifact_sha256: str,
) -> str:
    return _sha256_bytes(
        "\0".join(
            (command_id, nonce, cwd, artifact_path, artifact_sha256)
        ).encode("utf-8")
    )


def _child_observations(root: Path) -> list[tuple[Path, dict[str, object]]]:
    directory = root / "observations"
    if not directory.is_dir():
        return []
    observations: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(directory.glob("*.json"), key=lambda candidate: candidate.name):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AssertionError(f"child observation must be an object: {path}")
        observations.append((path, payload))
    return observations


def _probe_symlink_capability() -> dict[str, object]:
    """Return a structured, non-skipping result for the required symlink tests."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        target = root / "target"
        link = root / "link"
        target.write_bytes(b"symlink capability probe\n")
        try:
            link.symlink_to(target.name)
        except (OSError, NotImplementedError) as error:
            return {
                "available": False,
                "event": "symlink-capability-unavailable",
                "operation": "relative-symlink",
                "platform": sys.platform,
                "error_type": type(error).__name__,
                "errno": getattr(error, "errno", None),
            }
        return {
            "available": True,
            "event": "symlink-capability-available",
            "operation": "relative-symlink",
            "platform": sys.platform,
        }


def _create_required_symlink(link: Path, target: Path, *, operation: str) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        raise RuntimeError(
            json.dumps(
                {
                    "event": "symlink-capability-lost",
                    "required": True,
                    "operation": operation,
                    "platform": sys.platform,
                    "error_type": type(error).__name__,
                    "errno": getattr(error, "errno", None),
                },
                sort_keys=True,
            )
        ) from error


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _init_git_repository(root: Path, manifest: str, *, files: dict[str, str]) -> str:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "ledger-red@example.invalid")
    _git(root, "config", "user.name", "Ledger RED")
    _write_files(root, files)
    manifest_path = root / "scripts/storage-ownership-contracts.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest, encoding="utf-8")
    return _commit(root, "base ledger")


def _materialize_active_artifacts(root: Path, manifest: str) -> None:
    for row in tomllib.loads(manifest)["obligations"]:
        if row["state"]["kind"] != "active":
            continue
        path = root / row["artifact"]["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"artifact for {row['id']}\n", encoding="utf-8")


def _receipt_executions(receipt: dict[str, object]) -> list[dict[str, object]]:
    executions = receipt.get("executions")
    if not isinstance(executions, list):
        raise AssertionError("runner receipt must contain an executions array")
    if not all(isinstance(row, dict) for row in executions):
        raise AssertionError("every runner receipt execution must be an object")
    return executions


def _receipt_execution(receipt: dict[str, object], obligation_id: str) -> dict[str, object]:
    executions = _receipt_executions(receipt)
    matches = [
        row for row in executions if row.get("obligation_id") == obligation_id
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one receipt execution for {obligation_id}")
    return matches[0]


def _remove_receipt_execution(receipt: dict[str, object], obligation_id: str) -> None:
    target = _receipt_execution(receipt, obligation_id)
    executions = _receipt_executions(receipt)
    executions.remove(target)


class StorageOwnershipV2RedTests(unittest.TestCase):
    """The assertions below become green only after the v2 tools land."""

    def run_checker(
        self,
        manifest: str,
        *,
        files: dict[str, str] | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        _require_v2_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(root, files or repository_files())
            manifest_path = root / "ledger.toml"
            manifest_path.write_text(manifest, encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    "ledger.toml",
                    *extra_args,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def run_checker_at_root(
        self,
        root: Path,
        manifest: str,
        *,
        manifest_name: str = "ledger.toml",
        extra_args: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        _require_v2_checker()
        manifest_path = root / manifest_name
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest, encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(root),
                "--manifest",
                manifest_name,
                *extra_args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_production_checker(
        self, *extra_args: str
    ) -> subprocess.CompletedProcess[str]:
        _require_v2_checker()
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(ROOT),
                "--manifest",
                "scripts/storage-ownership-contracts.toml",
                *extra_args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_repository_runner(
        self,
        root: Path,
        base_commit: str,
        receipt_path: Path,
        *,
        diagnostics: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        _require_v2_runner()
        return subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--root",
                str(root),
                "--manifest",
                "scripts/storage-ownership-contracts.toml",
                "--base-commit",
                base_commit,
                "--receipt-out",
                str(receipt_path),
                *(("--diagnostics-json",) if diagnostics else ()),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_repository_checker(
        self,
        root: Path,
        base_commit: str,
        receipt_path: Path,
        output_flag: str,
    ) -> subprocess.CompletedProcess[str]:
        _require_v2_checker()
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(root),
                "--manifest",
                "scripts/storage-ownership-contracts.toml",
                "--base-commit",
                base_commit,
                "--receipt",
                str(receipt_path),
                output_flag,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def emit_runner_receipt(
        self,
        root: Path,
        *,
        promote: frozenset[str] = frozenset(),
        message: str = "candidate",
        active_override: dict[str, str] | None = None,
        manifest_override: str | None = None,
        files_override: dict[str, str] | None = None,
        runner_env: dict[str, str] | None = None,
        executable_paths: tuple[str, ...] = (),
    ) -> tuple[str, str, str, Path, dict[str, object]]:
        if manifest_override is not None:
            if promote or active_override:
                raise AssertionError(
                    "manifest_override cannot be combined with promotion overrides"
                )
            base = candidate = manifest_override
        else:
            base = valid_manifest(marker=True, active_override=active_override)
            candidate = valid_manifest(
                marker=True, active_override=active_override, promote=promote
            )
        base_commit = _init_git_repository(
            root,
            base,
            files=files_override or marker_files(base),
        )
        for relative in executable_paths:
            executable = root / relative
            executable.chmod(0o755)
        _materialize_active_artifacts(root, candidate)
        (root / "scripts/storage-ownership-contracts.toml").write_text(candidate, encoding="utf-8")
        (root / "candidate-note").write_text(f"{message}\n", encoding="utf-8")
        candidate_commit = _commit(root, message)
        receipt_path = root / "receipt.json"
        runner = self.run_repository_runner(
            root, base_commit, receipt_path, env=runner_env
        )
        self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertIsInstance(receipt, dict)
        self.assertEqual(set(receipt), RECEIPT_FIELDS)
        self.assertEqual(receipt.get("schema"), "tenferro.storage-ownership-receipt.v1")
        _receipt_executions(receipt)
        return base_commit, candidate_commit, candidate, receipt_path, receipt

    def parse_diagnostic_payload(
        self, result: subprocess.CompletedProcess[str]
    ) -> dict[str, object]:
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            self.fail(f"tool did not emit structured diagnostics: {result.stdout}{result.stderr}")
        self.assertIsInstance(payload, dict)
        self.assertEqual(set(payload), {"schema", "diagnostics"})
        self.assertEqual(payload.get("schema"), "tenferro.storage-ownership-diagnostics.v1")
        diagnostics = payload.get("diagnostics")
        self.assertIsInstance(diagnostics, list)
        self.assertTrue(diagnostics, payload)
        codes: list[str] = []
        for diagnostic in diagnostics:
            self.assertIsInstance(diagnostic, dict)
            self.assertEqual(set(diagnostic), {"code", "fields", "message"})
            code = diagnostic.get("code")
            self.assertIsInstance(code, str)
            self.assertIn(code, DIAGNOSTIC_FIELDS)
            fields = diagnostic.get("fields")
            self.assertIsInstance(fields, dict)
            self.assertEqual(set(fields), DIAGNOSTIC_FIELDS[code])
            for field, field_type in DIAGNOSTIC_FIELD_TYPES.get(code, {}).items():
                self.assertIs(type(fields[field]), field_type)
            self.assertIsInstance(diagnostic.get("message"), str)
            self.assertTrue(diagnostic["message"].strip(), diagnostic)
            codes.append(code)
        self.assertEqual(len(codes), len(set(codes)), payload)
        return payload

    def assert_checker_error(
        self,
        manifest: str,
        code: str,
        *,
        fields: dict[str, object] | None = None,
        files: dict[str, str] | None = None,
    ) -> None:
        result = self.run_checker(manifest, files=files, extra_args=("--diagnostics-json",))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.parse_diagnostic_payload(result)
        diagnostics = payload["diagnostics"]
        self.assertEqual({item["code"] for item in diagnostics}, {code}, payload)
        expected_fields = fields or {}
        for item in diagnostics:
            self.assertEqual(set(item["fields"]), set(expected_fields), payload)
            for key, value in expected_fields.items():
                self.assertEqual(item["fields"][key], value, payload)

    def assert_result_diagnostic(
        self,
        result: subprocess.CompletedProcess[str],
        code: str,
        *,
        fields: dict[str, object] | None = None,
    ) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.parse_diagnostic_payload(result)
        diagnostics = payload["diagnostics"]
        self.assertEqual({item["code"] for item in diagnostics}, {code}, payload)
        expected_fields = fields or {}
        for item in diagnostics:
            self.assertEqual(set(item["fields"]), set(expected_fields), payload)
            for key, value in expected_fields.items():
                self.assertEqual(item["fields"][key], value, payload)

    def assert_child_observation(
        self,
        root: Path,
        manifest_row: dict[str, object],
        observation: dict[str, object],
        *,
        executable_search_path: str | None = None,
    ) -> None:
        self.assertEqual(set(observation), OBSERVATION_FIELDS)
        self.assertEqual(observation["schema"], OBSERVATION_SCHEMA)
        command = manifest_row["command"]
        self.assertEqual(observation["command_id"], command["id"])
        self.assertIsInstance(observation["process_argv"], list)
        self.assertIsInstance(observation["normalized_process_argv"], list)
        self.assertEqual(
            observation["normalized_process_argv"],
            _expected_fixture_argv(
                root, command, executable_search_path=executable_search_path
            ),
        )
        self.assertIsInstance(observation["executable"], dict)
        executable = observation["executable"]
        self.assertEqual(set(executable), EXECUTABLE_IDENTITY_FIELDS)
        expected_executable = _resolve_executable_identity(
            command["argv"][0], executable_search_path
        )
        self.assertEqual(executable, expected_executable)
        process_argv = observation["process_argv"]
        self.assertIsInstance(observation["interpreter"], str)
        self.assertTrue(observation["interpreter"])
        self.assertTrue(process_argv)
        self.assertTrue(all(isinstance(value, str) for value in process_argv))
        self.assertEqual(process_argv[0], observation["interpreter"])
        self.assertEqual(observation["cwd"], str(root.resolve()))
        artifact_path = _resolve_manifest_path(root, manifest_row["artifact"]["path"])
        self.assertEqual(observation["artifact_path"], artifact_path)
        artifact_sha256 = _sha256_resolved_path(root, manifest_row["artifact"]["path"])
        self.assertEqual(observation["artifact_sha256"], artifact_sha256)
        self.assertIsInstance(observation["nonce"], str)
        self.assertTrue(observation["nonce"])
        self.assertEqual(
            observation["challenge"],
            _observation_challenge(
                command["id"],
                observation["nonce"],
                observation["cwd"],
                observation["artifact_path"],
                artifact_sha256,
            ),
        )

    def test_checked_in_production_manifest_tracks_the_v2_canonical_gate(self) -> None:
        self.assertTrue(PRODUCTION_MANIFEST.is_file())
        production = tomllib.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        canonical = tomllib.loads(valid_manifest())
        if production.get("schema") == LEGACY_SCHEMA:
            self.assertEqual(production["schema"], LEGACY_SCHEMA)
            self.assertNotEqual(production, canonical)
            return

        self.assertEqual(production.get("schema"), SCHEMA)
        self.assertEqual(production, canonical)
        result = self.run_production_checker()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_required_symlink_capability_is_available(self) -> None:
        capability = _probe_symlink_capability()
        self.assertTrue(
            capability["available"],
            json.dumps(
                {
                    "event": "symlink-capability-unavailable",
                    "required": True,
                    "capability": capability,
                },
                sort_keys=True,
            ),
        )

    def test_cli_schema_probe_rejects_non_contract_evidence(self) -> None:
        for tool, expected in (
            ("checker", CHECKER_CLI_CONTRACT),
            ("runner", RUNNER_CLI_CONTRACT),
        ):
            wrong_schema = {**expected, "schema": "wrong"}
            adversarial_sources = (
                ("comment", "# --contract-schema\n"),
                (
                    "unused-constant",
                    f"CONTRACT_SCHEMA = {expected!r}\n",
                ),
                ("file-existence", ""),
                ("wrong-json", 'print("not-json")\n'),
                (
                    "stderr",
                    "import sys\n"
                    f"print({json.dumps(json.dumps(expected))})\n"
                    "sys.stderr.write(\"not-json\")\n",
                ),
                (
                    "extra-stdout",
                    f"print({json.dumps(json.dumps(expected))})\n"
                    "print(\"extra\")\n",
                ),
                (
                    "wrong-schema",
                    f"print({json.dumps(json.dumps(wrong_schema))!r})\n",
                ),
            )
            for case, source in adversarial_sources:
                with self.subTest(tool=tool, case=case):
                    with tempfile.TemporaryDirectory() as temporary:
                        probe = Path(temporary) / "fake-tool.py"
                        probe.write_text(source, encoding="utf-8")
                        self.assertFalse(_probe_cli_contract(probe, expected))

    def test_cli_schema_probe_accepts_exact_contract(self) -> None:
        for tool, expected in (
            ("checker", CHECKER_CLI_CONTRACT),
            ("runner", RUNNER_CLI_CONTRACT),
        ):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as temporary:
                probe = Path(temporary) / "fake-tool.py"
                probe.write_text(
                    f"print({json.dumps(json.dumps(expected))})\n",
                    encoding="utf-8",
                )
                self.assertTrue(_probe_cli_contract(probe, expected))

    def test_requested_executable_identity_uses_current_path_for_python_and_cargo(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for requested in ("python3", "cargo"):
                executable = bin_dir / requested
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
            search_path = str(bin_dir)
            for requested in ("python3", "cargo"):
                with self.subTest(requested=requested):
                    expected_path = (bin_dir / requested).resolve()
                    self.assertEqual(
                        _resolve_executable_identity(requested, search_path),
                        {
                            "requested": requested,
                            "resolved": str(expected_path),
                            "sha256": _sha256_bytes(expected_path.read_bytes()),
                        },
                    )

    def test_checker_cli_schema_probe_is_required(self) -> None:
        _require_v2_checker()

    def test_runner_cli_schema_probe_is_required(self) -> None:
        _require_v2_runner()

    def test_checker_accepts_canonical_cargo_command_with_path_shim(self) -> None:
        production_rows = {
            row["id"]: row for row in tomllib.loads(valid_manifest())["obligations"]
        }
        production_command = production_rows["p0-control-plane"]["command"]
        self.assertEqual(production_command["kind"], "cargo-test")
        self.assertEqual(production_command["argv"][0], "cargo")
        self.assertNotIn("bin/cargo", production_command["argv"])
        self.assertNotIn("python3", production_command["argv"])
        fixture_rows = {
            row["id"]: row
            for row in tomllib.loads(_cargo_path_shim_manifest())["obligations"]
        }
        self.assertEqual(
            fixture_rows["p0-control-plane"]["command"], production_command
        )
        result = self.run_checker(
            _cargo_path_shim_manifest(),
            files=marker_files(_cargo_path_shim_manifest()),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_v2_checker_rejects_legacy_production_manifest_until_migration(self) -> None:
        self.assertTrue(PRODUCTION_MANIFEST.is_file())
        production = tomllib.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        canonical = tomllib.loads(valid_manifest())
        if production.get("schema") == LEGACY_SCHEMA:
            self.assertEqual(production["schema"], LEGACY_SCHEMA)
            result = self.run_production_checker("--diagnostics-json")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assert_result_diagnostic(
                result, "E_SCHEMA_VERSION", fields={"actual": LEGACY_SCHEMA}
            )
            return

        self.assertEqual(production.get("schema"), SCHEMA)
        self.assertEqual(production, canonical)
        result = self.run_production_checker()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_atomic_v2_migration_removes_legacy_surface(self) -> None:
        if _legacy_tooling_is_current():
            raise RedExpectedFailure(MIGRATION_CAUSE)

        self.assertTrue(V2_RED_SUITE.is_file())
        event_violations = _post_migration_red_event_violations(
            RED_EXPECTED_FAILURES, self._testMethodName
        )
        self.assertEqual(
            event_violations,
            [],
            f"temporary migration RED event retained: {event_violations}",
        )
        sentinel_violations = _post_migration_sentinel_violations(
            V2_RED_SUITE.read_text(encoding="utf-8")
        )
        self.assertEqual(
            sentinel_violations,
            [],
            f"temporary RED migration sentinels retained: {sentinel_violations}",
        )

        _require_v2_checker()
        _require_v2_runner()

        self.assertTrue(PRODUCTION_MANIFEST.is_file())
        production = tomllib.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(production.get("schema"), SCHEMA)
        self.assertEqual(production, tomllib.loads(valid_manifest()))

        inventory = _storage_tooling_inventory(ROOT)
        self.assertEqual(inventory, [], f"legacy tooling path/token violations: {inventory}")
        allowlist_drift = _storage_tooling_allowlist_drift(ROOT)
        self.assertEqual(
            allowlist_drift,
            [],
            f"intentional inventory allowlist drift: {allowlist_drift}",
        )

        self.assertFalse(V1_TEST_SUITE.exists())

        self.assertTrue(LEGACY_V1_MANIFEST_FIXTURE.is_file())
        legacy_fixture = tomllib.loads(
            LEGACY_V1_MANIFEST_FIXTURE.read_text(encoding="utf-8")
        )
        self.assertEqual(legacy_fixture, {"schema": LEGACY_SCHEMA})

        v2_manifests = []
        for path in sorted((ROOT / "scripts").rglob("*.toml")):
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
            if parsed.get("schema") == SCHEMA:
                v2_manifests.append(path)
        self.assertEqual(v2_manifests, [PRODUCTION_MANIFEST])

    def test_post_migration_inventory_rejects_renamed_legacy_tooling(self) -> None:
        cases = (
            (
                "scripts/renamed_checker.py",
                f'SCHEMA = "{LEGACY_SCHEMA}"\n',
                LEGACY_SCHEMA,
            ),
            (
                "scripts/renamed_suite.py",
                "# storage-ownership-contracts retired suite\n"
                "class CheckerTests(unittest.TestCase):\n    pass\n",
                "class CheckerTests(unittest.TestCase)",
            ),
            (
                "scripts/hidden_compatibility_shim.py",
                "# storage-ownership-contracts compatibility shim\n"
                'parser.add_argument("--compatibility-mode", action="store_true")\n',
                "--compatibility-mode",
            ),
            (
                "scripts/moved_table_parser.py",
                "# storage-ownership-contracts table parser\n"
                'TOP_LEVEL_KEYS = frozenset({"fixtures", "fixture_suites"})\n',
                "TOP_LEVEL_KEYS = frozenset",
            ),
            (
                "scripts/moved_manifest.toml",
                f'schema = "{LEGACY_SCHEMA}"\n[[fixtures]]\nid = "old"\n',
                "[[fixtures]]",
            ),
            (
                V1_TEST_SUITE_RELATIVE,
                "# retired storage suite path\n",
                V1_TEST_SUITE_RELATIVE,
            ),
        )
        for relative_path, source, token in cases:
            with self.subTest(path=relative_path, token=token):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    _write_files(root, {relative_path: source})
                    violations = _storage_tooling_inventory(root)
                    self.assertIn((relative_path, token), violations)

    def test_post_migration_inventory_requires_canonical_checker_path(self) -> None:
        for case in ("empty-tree", "unrelated-script"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                if case == "unrelated-script":
                    _write_files(root, {"scripts/unrelated.py": "value = 1\n"})
                self.assertIn(
                    (CHECKER_RELATIVE, "<missing-canonical-v2-checker>"),
                    _storage_tooling_inventory(root),
                )

    def test_post_migration_inventory_accepts_clean_v2_only_tooling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(
                root,
                {
                    "scripts/check-storage-ownership-contracts.py": (
                        f'SCHEMA = "{SCHEMA}"\n'
                        "\ndef _load_manifest(path):\n"
                        "    return {\"schema\": path}\n"
                    ),
                    "scripts/run-storage-ownership-contracts.py": (
                        f'SCHEMA = "{SCHEMA}"\n'
                    ),
                    "scripts/storage-ownership-contracts.toml": (
                        f'schema = "{SCHEMA}"\n'
                    ),
                    LEGACY_FIXTURE_RELATIVE: f'schema = "{LEGACY_SCHEMA}"\n',
                    "scripts/test-storage-ownership-contracts-v2.py": (
                        "# clean v2-only test tooling\n"
                    ),
                },
            )
            self.assertEqual(_storage_tooling_inventory(root), [])

    def test_post_migration_inventory_ignores_unrelated_generic_legacy_vocabulary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(
                root,
                {
                    "scripts/check-storage-ownership-contracts.py": (
                        f'SCHEMA = "{SCHEMA}"\n'
                    ),
                    "scripts/legacy/v1_fixture_report.py": (
                        'report = {"fixtures": ["sample"], '
                        '"fixture_suites": ["unit"], '
                        '"source_scans": ["docs"], '
                        '"source_inventory": ["assets"]}\n'
                        'flags = ("--legacy", "--compatibility-mode", "--v1")\n'
                        "v1_compat = allow_legacy = compatibility_mode = False\n"
                    ),
                    "scripts/unrelated-fixtures.toml": (
                        'schema = "example.v1"\n'
                        '[[fixtures]]\nname = "sample"\n'
                        '[[fixture_suites]]\nname = "unit"\n'
                        '[[source_scans]]\nname = "docs"\n'
                        '[[source_inventory]]\nname = "assets"\n'
                    ),
                },
            )
            self.assertEqual(_storage_tooling_inventory(root), [])

    def test_post_migration_sentinel_inventory_detects_each_retained_target(
        self,
    ) -> None:
        hashes = (
            "7694da2a07fb702c" + "dc0e2003eeff6b2610d1b8714cd19f78a04b07e4c9082fcf",
            "91ab78217adbb74f" + "8f6bf55a48ec6bb0c6c7eea17b9c51251dcdc092627dc718",
            "e4dbf32d274f7671" + "430a7a1e474016337b60fcab555087e2d111d093acccbdfe",
            "fed8c80e0e5b8969" + "f18a46f729644bad267adeb8a137499638d3a4926ed1b2ec",
        )
        targets = (
            *hashes,
            "LEGACY_V1_" + "QUARTET_SHA256",
            "_legacy_tooling_" + "is_current",
            "MIGRATION_" + "CAUSE",
            "v2-atomic-migration-" + "not-landed",
        )
        for target in targets:
            with self.subTest(target=target):
                self.assertEqual(
                    _post_migration_sentinel_violations(f"prefix\n{target}\nsuffix\n"),
                    [target],
                )
        event_name = "test_atomic_v2_migration_" + "removes_legacy_surface"
        self.assertEqual(
            _post_migration_red_event_violations(
                {event_name: {"cause": "temporary"}}, event_name
            ),
            [event_name],
        )
        self.assertEqual(_post_migration_red_event_violations({}, event_name), [])

    def test_storage_tooling_inventory_allowlist_has_no_repository_drift(self) -> None:
        self.assertEqual(_storage_tooling_allowlist_drift(ROOT), [])

    def test_storage_tooling_inventory_allowlist_rejects_extra_v2_red_occurrence(
        self,
    ) -> None:
        entry = next(
            entry
            for entry in STORAGE_TOOLING_INVENTORY_ALLOWLIST
            if entry.kind == "content" and entry.relative_path == V2_RED_SUITE_RELATIVE
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(
                root,
                {
                    V2_RED_SUITE_RELATIVE: (
                        V2_RED_SUITE.read_text(encoding="utf-8")
                        + f"\n{entry.token}\n"
                    ),
                    LEGACY_FIXTURE_RELATIVE: LEGACY_V1_MANIFEST_FIXTURE.read_text(
                        encoding="utf-8"
                    ),
                    CHECKER_RELATIVE: CHECKER.read_text(encoding="utf-8"),
                },
            )
            drift = _storage_tooling_allowlist_drift(root)
            self.assertIn(
                (V2_RED_SUITE_RELATIVE, entry.token),
                {(path, token) for path, token, _purpose in drift},
            )

    def test_result_diagnostic_rejects_wrong_schema_and_array_shape(self) -> None:
        for payload in (
            {"schema": "wrong", "diagnostics": [{"code": "E_X", "fields": {}}]},
            {
                "schema": "tenferro.storage-ownership-diagnostics.v1",
                "diagnostics": {"code": "E_X", "fields": {}},
            },
        ):
            with self.subTest(payload=payload), self.assertRaises(AssertionError):
                self.assert_result_diagnostic(
                    subprocess.CompletedProcess([], 1, stdout=json.dumps(payload), stderr=""),
                    "E_X",
                )

    def test_result_diagnostic_requires_obligation_identity_and_message(self) -> None:
        expected_fields = {
            "obligation_id": "p0-control-plane",
            "field": "artifact_sha256",
            "expected": "a" * 64,
            "actual": "b" * 64,
        }
        payload = {
            "schema": "tenferro.storage-ownership-diagnostics.v1",
            "diagnostics": [
                {
                    "code": "E_RECEIPT_DIGEST",
                    "fields": dict(expected_fields),
                    "message": "artifact digest changed",
                }
            ],
        }
        for case in ("missing-obligation", "wrong-obligation", "empty-message"):
            with self.subTest(case=case):
                candidate = json.loads(json.dumps(payload))
                if case == "missing-obligation":
                    del candidate["diagnostics"][0]["fields"]["obligation_id"]
                elif case == "wrong-obligation":
                    candidate["diagnostics"][0]["fields"]["obligation_id"] = (
                        "p1-ledger"
                    )
                else:
                    candidate["diagnostics"][0]["message"] = "   "
                with self.assertRaises(AssertionError):
                    self.assert_result_diagnostic(
                        subprocess.CompletedProcess(
                            [], 1, stdout=json.dumps(candidate), stderr=""
                        ),
                        "E_RECEIPT_DIGEST",
                        fields=expected_fields,
                    )

    def test_receipt_shape_diagnostic_requires_exact_fields(self) -> None:
        expected_fields = {
            "field": "base_commit",
            "expected": "present",
            "actual": "missing",
        }
        payload = {
            "schema": "tenferro.storage-ownership-diagnostics.v1",
            "diagnostics": [
                {
                    "code": "E_RECEIPT_SHAPE",
                    "fields": dict(expected_fields),
                    "message": "receipt field is missing",
                }
            ],
        }
        valid = subprocess.CompletedProcess(
            [], 1, stdout=json.dumps(payload), stderr=""
        )
        self.assert_result_diagnostic(
            valid, "E_RECEIPT_SHAPE", fields=expected_fields
        )
        for case in ("missing-field", "wrong-field"):
            with self.subTest(case=case):
                candidate = json.loads(json.dumps(payload))
                if case == "missing-field":
                    del candidate["diagnostics"][0]["fields"]["actual"]
                else:
                    candidate["diagnostics"][0]["fields"]["field"] = "terminal"
                with self.assertRaises(AssertionError):
                    self.assert_result_diagnostic(
                        subprocess.CompletedProcess(
                            [], 1, stdout=json.dumps(candidate), stderr=""
                        ),
                        "E_RECEIPT_SHAPE",
                        fields=expected_fields,
                    )

    def test_nominal_v2_manifest_is_green(self) -> None:
        result = self.run_checker(valid_manifest())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_v1_is_rejected_without_compatibility_mode(self) -> None:
        self.assert_checker_error(
            valid_manifest(schema="tenferro.storage-ownership-contracts.v1"),
            "E_SCHEMA_VERSION",
            fields={"actual": "tenferro.storage-ownership-contracts.v1"},
        )

    def test_legacy_v1_fixture_and_source_tables_are_rejected(self) -> None:
        _require_v2_checker()
        self.assertTrue(LEGACY_V1_MANIFEST_FIXTURE.is_file())
        legacy = LEGACY_V1_MANIFEST_FIXTURE.read_text(encoding="utf-8")
        parsed = tomllib.loads(legacy)
        self.assertEqual(parsed, {"schema": LEGACY_SCHEMA})
        self.assert_checker_error(
            legacy,
            "E_SCHEMA_VERSION",
            fields={"actual": LEGACY_SCHEMA},
        )

    def test_one_tagged_obligation_table_replaces_parallel_status_tables(self) -> None:
        malformed = _replace_in_obligation(
            valid_manifest(),
            "p0-control-plane",
            'state = { kind = "active" }',
            'status = "active"',
        )
        self.assert_checker_error(
            malformed, "E_OBLIGATION_TAGGED_STATE", fields={"obligation_id": "p0-control-plane"}
        )
        self.assert_checker_error(
            valid_manifest() + '\n[[obligations.active]]\nid = "legacy-active"\n',
            "E_SCHEMA_PARALLEL_TABLE", fields={"table": "obligations.active"},
        )

    def test_every_canonical_unit_has_required_obligations(self) -> None:
        parsed = tomllib.loads(valid_manifest())
        covered = {row["unit"] for row in parsed["obligations"]}
        self.assertEqual(covered, {unit for unit, _, _ in UNITS})
        missing_p0 = "\n\n".join(
            block
            for block in valid_manifest().strip().split("\n\n")
            if 'id = "p0-control-plane"' not in block
        ) + "\n"
        self.assert_checker_error(
            missing_p0, "E_UNIT_OBLIGATION_MISSING", fields={"unit": "P0"}
        )

    def test_lifecycle_compile_and_runtime_proofs_are_canonical_obligations(self) -> None:
        rows = {row["id"]: row for row in tomllib.loads(valid_manifest())["obligations"]}
        expected = {
            "p4-production-borrow-contract": ("P4", {"G1", "G4"}, "compile-contract"),
            "p3-auto-trait-contract": ("P3", {"G1", "G4"}, "compile-contract"),
            "p4-provider-release-lifecycle": ("P4", {"G1", "G3"}, "provider-test"),
        }
        for obligation_id, (unit, gates, artifact_kind) in expected.items():
            with self.subTest(obligation_id=obligation_id):
                row = rows[obligation_id]
                self.assertEqual(row["unit"], unit)
                self.assertEqual(set(row["gates"]), gates)
                self.assertEqual(row["artifact"]["kind"], artifact_kind)
                self.assertEqual(row["state"]["kind"], "deferred")
                self.assertEqual(row["command"]["artifact_id"], row["artifact"]["id"])
                self.assertNotIn("test-storage-ownership-contracts-v2.py", row["artifact"]["path"])

    def test_production_borrow_proof_is_not_a_self_referential_fixture(self) -> None:
        rows = {row["id"]: row for row in tomllib.loads(valid_manifest())["obligations"]}
        row = rows["p4-production-borrow-contract"]
        self.assertEqual(row["artifact"]["path"], "crates/tenferro-tensor/tests/storage_borrow_contract.rs")
        self.assertEqual(
            row["command"]["argv"],
            ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_borrow_contract"],
        )
        self.assertEqual(row["command"]["artifact_id"], "artifact-production-borrow-contract")
        self.assertEqual(row["state"]["kind"], "deferred")

    def test_canonical_future_lifecycle_proof_commands_execute(self) -> None:
        rows = {row["id"]: row for row in tomllib.loads(valid_manifest())["obligations"]}
        deferred_ids = tuple(row[0] for row in DEFERRED_OBLIGATIONS)
        deferred_id_set = frozenset(deferred_ids)
        self.assertEqual(
            tuple(row_id for row_id in rows if row_id in deferred_id_set),
            deferred_ids,
        )
        for canonical in DEFERRED_OBLIGATIONS:
            (
                obligation_id,
                unit,
                gates,
                artifact_id,
                artifact_path,
                artifact_kind,
                command_id,
                command_kind,
                command_argv,
                command_path_args,
            ) = canonical
            with self.subTest(obligation_id=obligation_id):
                row = rows[obligation_id]
                self.assertEqual(row["unit"], unit)
                self.assertEqual(row["gates"], list(gates))
                self.assertEqual(
                    row["artifact"],
                    {
                        "id": artifact_id,
                        "kind": artifact_kind,
                        "path": artifact_path,
                    },
                )
                self.assertEqual(
                    row["state"],
                    {
                        "kind": "deferred",
                        "activation_unit": unit,
                        "promotion": {"mode": "activate-in-place"},
                    },
                )
                self.assertEqual(
                    row["command"],
                    {
                        "id": command_id,
                        "kind": command_kind,
                        "argv": list(command_argv),
                        "cwd": ".",
                        "path_args": list(command_path_args),
                        "artifact_id": artifact_id,
                    },
                )
                artifact = ROOT / artifact_path
                if not artifact.is_file():
                    raise RedExpectedFailure(FUTURE_ARTIFACT_CAUSE)
                result = subprocess.run(
                    list(command_argv),
                    cwd=ROOT / row["command"]["cwd"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_canonical_graph_keeps_p0_p1_roots_and_p2_only_depends_on_p1(self) -> None:
        parsed = _parse_registry(valid_manifest())
        self.assertEqual(parsed["edges"].count(("P1", "P2")), 1)
        self.assertNotIn(("P0", "P2"), parsed["edges"])
        self.assertEqual(CUTOVER["prerequisites"], ("P0", "P5"))
        self.assertEqual(CUTOVER["members"], ("P3", "P9"))

        wrong_root = valid_manifest(edges=EDGES + (("P0", "P2"),))
        self.assert_checker_error(
            wrong_root, "E_GRAPH_P2_PREREQUISITE", fields={"unit": "P2"}
        )
        wrong_cutover = _replace_once(
            valid_manifest(),
            'prerequisites = ["P0", "P5"]',
            'prerequisites = ["P1", "P5"]',
        )
        self.assert_checker_error(
            wrong_cutover, "E_COHORT_DEFINITION", fields={"cohort_id": "cutover"}
        )

    def test_graph_rejects_duplicate_and_unknown_target_links(self) -> None:
        duplicate = valid_manifest(edges=EDGES + (("P1", "P2"),))
        self.assert_checker_error(
            duplicate, "E_GRAPH_DUPLICATE_EDGE", fields={"from": "P1", "to": "P2"}
        )
        unknown = valid_manifest(edges=EDGES + (("P2", "P99"),))
        self.assert_checker_error(
            unknown, "E_GRAPH_UNKNOWN_UNIT", fields={"unit": "P99"}
        )

    def test_cutover_is_atomic_and_partial_activation_is_rejected(self) -> None:
        partial = valid_manifest(
            promote=CUTOVER_PARTIAL_OBLIGATIONS
        )
        files = repository_files()
        for row in DEFERRED_OBLIGATIONS:
            if row[0] in CUTOVER_PARTIAL_OBLIGATIONS:
                files[row[4]] = "fn promoted_contract() {}\n"
        self.assert_checker_error(
            partial,
            "E_COHORT_PARTIAL_PROMOTION",
            fields={"cohort_id": "cutover"},
            files=files,
        )

    def test_stale_p3_p4_ownership_rows_are_not_a_second_authority(self) -> None:
        stale = valid_manifest() + textwrap.dedent(
            '''\
            [[registry.ownerships]]
            id = "p3-ad-retention"
            gate = "G7"
            phase = "P3"

            [[registry.ownerships]]
            id = "p4-ad-runtime"
            gate = "G7"
            phase = "P4"
            '''
        )
        self.assert_checker_error(
            stale, "E_OBSOLETE_OWNERSHIP_TABLE", fields={"table": "registry.ownerships"}
        )

    def test_synthetic_terminal_artifacts_and_terminal_flags_are_rejected(self) -> None:
        synthetic = _replace_once(
            valid_manifest(),
            'kind = "manifest", path = "scripts/storage-ownership-contracts.toml"',
            'kind = "synthetic-terminal", path = ".ledger-terminal"',
        )
        self.assert_checker_error(
            synthetic, "E_ARTIFACT_SYNTHETIC_TERMINAL", fields={"artifact_id": "artifact-ledger"}
        )
        self.assert_checker_error(
            valid_manifest() + '\nterminal = true\n',
            "E_TERMINAL_DECLARED", fields={"field": "terminal"},
        )

    def test_terminal_state_is_derived_from_obligations_and_receipts(self) -> None:
        terminal = valid_manifest()
        self.assertNotIn("terminal", terminal)
        result = self.run_checker(terminal, extra_args=("--summary-json",))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertFalse(summary["terminal"])

    def test_artifact_paths_are_unique_repository_relative_and_real(self) -> None:
        duplicate = _replace_once(
            valid_manifest(),
            'id = "artifact-api-parity", kind = "rust-test", path = "crates/tenferro-tensor/tests/storage_api_parity.rs"',
            'id = "artifact-api-parity", kind = "rust-test", path = "scripts/storage-ownership-contracts.toml"',
        )
        self.assert_checker_error(
            duplicate, "E_ARTIFACT_DUPLICATE_TARGET", fields={"artifact_id": "artifact-api-parity"}
        )
        escape = valid_manifest(active_override={"p1-ledger": "../outside.toml"})
        self.assert_checker_error(
            escape, "E_PATH_ESCAPE", fields={"obligation_id": "p1-ledger"}
        )
        missing = valid_manifest(active_override={"p1-ledger": "scripts/missing.toml"})
        self.assert_checker_error(
            missing, "E_ARTIFACT_MISSING", fields={"artifact_id": "artifact-ledger"}
        )

    def test_real_symlink_escape_is_rejected(self) -> None:
        _require_v2_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = repository_files()
            _write_files(root, files)
            outside = root.parent / f"ledger-outside-{root.name}.rs"
            link = root / "scripts" / "escaped.toml"
            try:
                outside.write_text("fn outside() {}\n", encoding="utf-8")
                _create_required_symlink(
                    link, outside, operation="repository-relative-escape"
                )
                manifest = valid_manifest(active_override={"p1-ledger": "scripts/escaped.toml"})
                manifest_path = root / "ledger.toml"
                manifest_path.write_text(manifest, encoding="utf-8")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(CHECKER),
                        "--root",
                        str(root),
                        "--manifest",
                        "ledger.toml",
                        "--diagnostics-json",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            finally:
                if link.is_symlink() or link.exists():
                    link.unlink()
                if outside.is_symlink() or outside.exists():
                    outside.unlink()
        self.assert_result_diagnostic(
            result, "E_PATH_SYMLINK_ESCAPE", fields={"obligation_id": "p1-ledger"}
        )

    def test_deferred_artifact_cannot_be_promoted_by_existing_file_alone(self) -> None:
        files = repository_files()
        deferred_path = next(row[4] for row in DEFERRED_OBLIGATIONS if row[0] == "p3-host-owner")
        files[deferred_path] = "fn future_contract() {}\n"
        self.assert_checker_error(
            valid_manifest(),
            "E_DEFERRED_ARTIFACT_EXISTS",
            fields={"obligation_id": "p3-host-owner"},
            files=files,
        )

    def test_command_allowlist_is_typed_and_fail_closed(self) -> None:
        shell = _replace_once(
            valid_manifest(),
            'kind = "python-test", argv = ["python3", "scripts/check-storage-ownership-contracts.py"]',
            'kind = "shell", argv = ["sh", "-c", "echo unsafe"]',
        )
        self.assert_checker_error(
            shell,
            "E_COMMAND_KIND",
            fields={"command_id": "cmd-ledger", "kind": "shell"},
        )
        empty = _replace_once(
            valid_manifest(),
            'argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_api_parity"]',
            "argv = []",
        )
        self.assert_checker_error(empty, "E_COMMAND_ARGV", fields={"command_id": "cmd-api-parity"})
        path_escape = _replace_once(
            valid_manifest(),
            'path_args = ["scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"]',
            'path_args = ["../outside.toml"]',
        )
        self.assert_checker_error(
            path_escape, "E_COMMAND_PATH_ESCAPE", fields={"command_id": "cmd-ledger"}
        )

    def test_command_argv_exact_allowlist_is_enforced(self) -> None:
        exercised: list[tuple[str, int]] = []
        for command_id, index in CANONICAL_COMMAND_ARGV_COORDINATES:
            with self.subTest(command_id=command_id, index=index):
                expected = CANONICAL_COMMAND_ARGV[command_id][index]
                actual = f"{expected}-mutated-{index}"
                mutated, expected = _replace_command_argv_in_all_occurrences(
                    valid_manifest(), command_id, index=index, actual=actual
                )
                exercised.append((command_id, index))
                self.assert_checker_error(
                    mutated,
                    "E_COMMAND_ARGV_BINDING",
                    fields={
                        "command_id": command_id,
                        "index": index,
                        "expected": expected,
                        "actual": actual,
                    },
                )
        canonical = set(CANONICAL_COMMAND_ARGV_COORDINATES)
        self.assertEqual(set(exercised), canonical)
        self.assertEqual(Counter(exercised), Counter(canonical))

    def test_command_argv_length_is_enforced(self) -> None:
        exercised: list[tuple[str, str]] = []
        for command_id in CANONICAL_COMMAND_IDS:
            for case in COMMAND_ARGV_LENGTH_CASES:
                with self.subTest(command_id=command_id, case=case):
                    mutated, expected, actual = (
                        _replace_command_argv_length_in_all_occurrences(
                            valid_manifest(), command_id, case=case
                        )
                    )
                    exercised.append((command_id, case))
                    self.assert_checker_error(
                        mutated,
                        "E_COMMAND_ARGV_LENGTH",
                        fields={
                            "command_id": command_id,
                            "expected": expected,
                            "actual": actual,
                        },
                    )
        canonical = {
            (command_id, case)
            for command_id in CANONICAL_COMMAND_IDS
            for case in COMMAND_ARGV_LENGTH_CASES
        }
        self.assertEqual(set(exercised), canonical)
        self.assertEqual(Counter(exercised), Counter(canonical))
        for case in COMMAND_ARGV_LENGTH_CASES:
            self.assertEqual(
                {command_id for command_id, actual_case in exercised if actual_case == case},
                set(CANONICAL_COMMAND_IDS),
            )

    def test_command_cwd_confinement_rejects_absolute_and_parent_escape(self) -> None:
        for cwd in ("/tmp/ledger-command-outside", "../ledger-command-outside"):
            with self.subTest(cwd=cwd):
                mutated = _replace_in_obligation(
                    valid_manifest(),
                    "p1-ledger",
                    'cwd = "."',
                    f"cwd = {_quote(cwd)}",
                )
                self.assert_checker_error(
                    mutated,
                    "E_COMMAND_CWD_ESCAPE",
                    fields={"command_id": "cmd-ledger", "cwd": cwd},
                )

    def test_command_argv_path_escape_ignores_path_args_metadata(self) -> None:
        argv = 'argv = ["python3", "scripts/check-storage-ownership-contracts.py"]'
        path_args = (
            'path_args = ["scripts/check-storage-ownership-contracts.py", '
            '"scripts/storage-ownership-contracts.toml"]'
        )
        for argument in ("/tmp/ledger-command-outside.py", "../ledger-command-outside.py"):
            with self.subTest(argument=argument):
                mutated = _replace_in_obligation(
                    valid_manifest(),
                    "p1-ledger",
                    argv,
                    f'argv = ["python3", {_quote(argument)}]',
                )
                # The path_args declaration deliberately lies: argv is scanned
                # independently and cannot be made safe by omitting the value.
                mutated = _replace_in_obligation(
                    mutated,
                    "p1-ledger",
                    path_args,
                    "path_args = []",
                )
                self.assert_checker_error(
                    mutated,
                    "E_COMMAND_ARGV_PATH_ESCAPE",
                    fields={
                        "command_id": "cmd-ledger",
                        "index": 1,
                        "argument": argument,
                    },
                )

    def test_command_symlink_confinement_rejects_cwd_and_argv_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(root, repository_files())
            links = root / "links"
            links.mkdir(parents=True, exist_ok=True)
            outside_cwd = root.parent / f"ledger-command-outside-cwd-{root.name}"
            outside_path = root.parent / f"ledger-command-outside-{root.name}.py"
            cwd_link = links / "outside-cwd"
            argv_link = links / "outside-command.py"
            outside_cwd.mkdir()
            outside_path.write_text("raise SystemExit(99)\n", encoding="utf-8")
            try:
                _create_required_symlink(
                    cwd_link, outside_cwd, operation="command-cwd-symlink-escape"
                )
                _create_required_symlink(
                    argv_link, outside_path, operation="command-argv-symlink-escape"
                )
                cwd_manifest = _replace_in_obligation(
                    valid_manifest(),
                    "p1-ledger",
                    'cwd = "."',
                    'cwd = "links/outside-cwd"',
                )
                argv_manifest = _replace_in_obligation(
                    valid_manifest(),
                    "p1-ledger",
                    'argv = ["python3", "scripts/check-storage-ownership-contracts.py"]',
                    'argv = ["python3", "links/outside-command.py"]',
                )
                argv_manifest = _replace_in_obligation(
                    argv_manifest,
                    "p1-ledger",
                    'path_args = ["scripts/check-storage-ownership-contracts.py", '
                    '"scripts/storage-ownership-contracts.toml"]',
                    "path_args = []",
                )
                cases = (
                    (
                        "cwd-symlink",
                        cwd_manifest,
                        "E_COMMAND_CWD_SYMLINK_ESCAPE",
                        {"command_id": "cmd-ledger", "cwd": "links/outside-cwd"},
                    ),
                    (
                        "argv-symlink",
                        argv_manifest,
                        "E_COMMAND_ARGV_SYMLINK_ESCAPE",
                        {
                            "command_id": "cmd-ledger",
                            "index": 1,
                            "argument": "links/outside-command.py",
                        },
                    ),
                )
                for case, manifest, code, fields in cases:
                    with self.subTest(case=case):
                        result = self.run_checker_at_root(
                            root, manifest, extra_args=("--diagnostics-json",)
                        )
                        self.assert_result_diagnostic(result, code, fields=fields)
            finally:
                for link in (cwd_link, argv_link):
                    if link.is_symlink() or link.exists():
                        link.unlink()
                if outside_path.is_symlink() or outside_path.exists():
                    outside_path.unlink()
                if outside_cwd.is_symlink() or outside_cwd.exists():
                    outside_cwd.rmdir()

    def test_command_must_bind_to_exact_artifact_and_target_links(self) -> None:
        wrong_id = _replace_once(
            valid_manifest(),
            'artifact_id = "artifact-ledger"',
            'artifact_id = "artifact-contract-document"',
        )
        self.assert_checker_error(
            wrong_id, "E_COMMAND_ARTIFACT_BINDING", fields={"command_id": "cmd-ledger"}
        )
        wrong_target = _replace_once(
            valid_manifest(),
            'path_args = ["scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"]',
            'path_args = ["scripts/check-storage-design-docs.py"]',
        )
        self.assert_checker_error(
            wrong_target, "E_COMMAND_TARGET_BINDING", fields={"command_id": "cmd-ledger"}
        )
        duplicate_command = _replace_once(
            valid_manifest(),
            'id = "cmd-api-parity"',
            'id = "cmd-ledger"',
        )
        self.assert_checker_error(
            duplicate_command, "E_COMMAND_ID_CONFLICT", fields={"command_id": "cmd-ledger"}
        )

    def test_source_and_fixture_tables_cannot_reappear_as_parallel_authority(self) -> None:
        for table in ("fixtures", "fixture_suites", "source_scans", "source_inventory", "ownerships"):
            with self.subTest(table=table):
                self.assert_checker_error(
                    valid_manifest() + f'\n[[{table}]]\nid = "legacy"\n',
                    "E_SCHEMA_UNKNOWN_TABLE",
                    fields={"table": table},
                )

    def test_promotion_preserves_immutable_identity_and_binds_receipt_to_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit, _, _, receipt_path, _ = self.emit_runner_receipt(
                root,
                promote=CUTOVER_CANDIDATE_OBLIGATIONS,
                message="atomic cutover candidate",
            )
            result = self.run_repository_checker(
                root, base_commit, receipt_path, "--summary-json"
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cutover_requires_non_vacuous_p0_and_p5_receipt_proof(self) -> None:
        for unit, obligation_id, missing in (
            ("P0", "p0-control-plane", True),
            ("P5", "p5-allocation-group", False),
        ):
            with self.subTest(unit=unit), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_commit, _, _, receipt_path, receipt = self.emit_runner_receipt(
                    root,
                    promote=CUTOVER_CANDIDATE_OBLIGATIONS,
                    message="cutover prerequisite proof candidate",
                )
                if missing:
                    _remove_receipt_execution(receipt, obligation_id)
                else:
                    _receipt_execution(receipt, obligation_id)["exit_code"] = 9
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                result = self.run_repository_checker(
                    root, base_commit, receipt_path, "--diagnostics-json"
                )
                self.assert_result_diagnostic(
                    result,
                    "E_COHORT_PREREQUISITE_INCOMPLETE",
                    fields={"unit": unit, "obligation_id": obligation_id},
                )

    def test_matching_fake_commit_ids_cannot_replace_git_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual_base, actual_candidate, _, receipt_path, receipt = self.emit_runner_receipt(root)
            self.assertEqual(receipt["base_commit"], actual_base)
            self.assertEqual(receipt["candidate_commit"], actual_candidate)
            fake = "f" * 40
            receipt["base_commit"] = fake
            receipt["candidate_commit"] = fake
            for execution in _receipt_executions(receipt):
                execution["candidate_commit"] = fake
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            result = self.run_repository_checker(
                root, fake, receipt_path, "--diagnostics-json"
            )
        self.assert_result_diagnostic(result, "E_RECEIPT_COMMIT", fields={"actual_head": actual_candidate})

    def test_runner_receipt_digests_match_independent_sha256_calculations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit, _, candidate, _, receipt = self.emit_runner_receipt(root)
            self.assertEqual(
                receipt["base_manifest_sha256"],
                _sha256_bytes(_git_show_bytes(root, base_commit, "scripts/storage-ownership-contracts.toml")),
            )
            self.assertEqual(
                receipt["candidate_manifest_sha256"],
                _sha256_bytes((root / "scripts/storage-ownership-contracts.toml").read_bytes()),
            )
            rows = {row["id"]: row for row in tomllib.loads(candidate)["obligations"]}
            for execution in _receipt_executions(receipt):
                row = rows[execution["obligation_id"]]
                self.assertEqual(
                    execution["artifact_sha256"],
                    _sha256_resolved_path(root, row["artifact"]["path"]),
                )
                self.assertEqual(
                    execution["command_sha256"],
                    _sha256_command(row["command"]),
                )

    def test_receipt_digests_bind_exact_manifest_artifact_and_command(self) -> None:
        for digest_kind in ("manifest", "artifact", "command"):
            with self.subTest(digest_kind=digest_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_commit, _, candidate, receipt_path, receipt = self.emit_runner_receipt(root)
                if digest_kind == "manifest":
                    manifest_path = root / "scripts/storage-ownership-contracts.toml"
                    manifest_path.write_bytes(
                        manifest_path.read_bytes() + b"\npost-receipt manifest mutation\n"
                    )
                    diagnostic_code = "E_RECEIPT_MANIFEST_DIGEST"
                    expected_fields = {
                        "field": "candidate_manifest_sha256",
                        "expected": _sha256_bytes(manifest_path.read_bytes()),
                        "actual": receipt["candidate_manifest_sha256"],
                    }
                elif digest_kind == "artifact":
                    row = next(
                        row for row in tomllib.loads(candidate)["obligations"]
                        if row["id"] == "p0-control-plane"
                    )
                    execution = _receipt_execution(receipt, "p0-control-plane")
                    artifact = root / row["artifact"]["path"]
                    execution["artifact_sha256"] = _sha256_bytes(
                        artifact.resolve().read_bytes() + b"\npost-receipt mutation\n"
                    )
                    diagnostic_code = "E_RECEIPT_DIGEST"
                    expected_fields = {
                        "obligation_id": "p0-control-plane",
                        "field": "artifact_sha256",
                        "expected": _sha256_resolved_path(root, row["artifact"]["path"]),
                        "actual": execution["artifact_sha256"],
                    }
                else:
                    row = next(
                        row for row in tomllib.loads(candidate)["obligations"]
                        if row["id"] == "p0-control-plane"
                    )
                    command_bytes = json.dumps(
                        row["command"], sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                    execution = _receipt_execution(receipt, "p0-control-plane")
                    execution["command_sha256"] = _sha256_bytes(
                        command_bytes + b"post-receipt mutation"
                    )
                    diagnostic_code = "E_RECEIPT_DIGEST"
                    expected_fields = {
                        "obligation_id": "p0-control-plane",
                        "field": "command_sha256",
                        "expected": _sha256_command(row["command"]),
                        "actual": execution["command_sha256"],
                    }
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                result = self.run_repository_checker(
                    root, base_commit, receipt_path, "--diagnostics-json"
                )
                self.assert_result_diagnostic(
                    result, diagnostic_code, fields=expected_fields
                )

    def test_post_receipt_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit, _, candidate, receipt_path, receipt = self.emit_runner_receipt(root)
            row = next(
                row for row in tomllib.loads(candidate)["obligations"]
                if row["id"] == "p0-control-plane"
            )
            artifact = root / row["artifact"]["path"]
            execution = _receipt_execution(receipt, "p0-control-plane")
            recorded = execution["artifact_sha256"]
            artifact.write_bytes(artifact.read_bytes() + b"\npost-receipt artifact mutation\n")
            current = _sha256_resolved_path(root, row["artifact"]["path"])
            result = self.run_repository_checker(
                root, base_commit, receipt_path, "--diagnostics-json"
            )
            self.assert_result_diagnostic(
                result,
                "E_RECEIPT_DIGEST",
                fields={
                    "obligation_id": "p0-control-plane",
                    "field": "artifact_sha256",
                    "expected": current,
                    "actual": recorded,
                },
            )

    def test_post_receipt_base_manifest_mutation_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit, _, _, receipt_path, receipt = self.emit_runner_receipt(root)
            base_bytes = _git_show_bytes(
                root, base_commit, "scripts/storage-ownership-contracts.toml"
            )
            # A Git object is immutable; model a post-receipt mutation of the
            # base-manifest bytes in the recorded digest independently.
            receipt["base_manifest_sha256"] = _sha256_bytes(
                base_bytes + b"\npost-receipt base-manifest mutation\n"
            )
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            result = self.run_repository_checker(
                root, base_commit, receipt_path, "--diagnostics-json"
            )
        self.assert_result_diagnostic(
            result,
            "E_RECEIPT_MANIFEST_DIGEST",
            fields={
                "field": "base_manifest_sha256",
                "expected": _sha256_bytes(base_bytes),
                "actual": receipt["base_manifest_sha256"],
            },
        )

    def test_post_receipt_artifact_symlink_retarget_with_identical_external_bytes_is_rejected(
        self,
    ) -> None:
        override = {"p0-control-plane": "artifacts/control.py"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = valid_manifest(marker=True, active_override=override)
            base_commit = _init_git_repository(root, base, files=marker_files())
            artifact = root / "artifacts/control.py"
            target_a = root / "artifacts/target-a.py"
            external = root.parent / f"ledger-artifact-retarget-{root.name}.py"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            identical = b"identical artifact bytes\n"
            target_a.write_bytes(identical)
            _create_required_symlink(
                artifact, Path("target-a.py"), operation="post-receipt-initial"
            )
            base_commit = _commit(root, "base internal artifact symlink")
            candidate = valid_manifest(marker=True, active_override=override)
            (root / "scripts/storage-ownership-contracts.toml").write_text(
                candidate, encoding="utf-8"
            )
            (root / "candidate-note").write_text("candidate\n", encoding="utf-8")
            _commit(root, "candidate with internal artifact symlink")
            receipt_path = root / "receipt.json"
            runner = self.run_repository_runner(root, base_commit, receipt_path)
            self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
            try:
                expected_path = str(target_a.resolve())
                external.write_bytes(identical)
                artifact.unlink()
                _create_required_symlink(
                    artifact,
                    external,
                    operation="post-receipt-artifact-retarget",
                )
                self.assertEqual(artifact.resolve().read_bytes(), identical)
                self.assertEqual(str(artifact.resolve()), str(external.resolve()))
                result = self.run_repository_checker(
                    root, base_commit, receipt_path, "--diagnostics-json"
                )
                self.assert_result_diagnostic(
                    result,
                    "E_RECEIPT_PATH_IDENTITY",
                    fields={
                        "obligation_id": "p0-control-plane",
                        "field": "artifact_path",
                        "expected": expected_path,
                        "actual": str(external.resolve()),
                    },
                )
            finally:
                if artifact.is_symlink() or artifact.exists():
                    artifact.unlink()
                if external.is_symlink() or external.exists():
                    external.unlink()

    def test_post_receipt_cwd_symlink_retarget_is_rejected(self) -> None:
        manifest = _replace_in_obligation(
            valid_manifest(marker=True),
            "p0-control-plane",
            'cwd = "."',
            'cwd = "links/cwd"',
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit = _init_git_repository(root, manifest, files=marker_files())
            command_bytes = (root / "markers/cmd-control-plane.py").read_bytes()
            artifact_bytes = (
                root / "crates/tenferro-runtime/tests/execution_engine_identity.rs"
            ).read_bytes()
            cwd_link = root / "links/cwd"
            cwd_link.parent.mkdir(parents=True, exist_ok=True)
            _create_required_symlink(
                cwd_link, Path(".."), operation="post-receipt-cwd-initial"
            )
            base_commit = _commit(root, "base internal cwd symlink")
            (root / "scripts/storage-ownership-contracts.toml").write_text(
                manifest, encoding="utf-8"
            )
            (root / "candidate-note").write_text("candidate\n", encoding="utf-8")
            _commit(root, "candidate with internal cwd symlink")
            receipt_path = root / "receipt.json"
            runner = self.run_repository_runner(root, base_commit, receipt_path)
            self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
            target = root / "cwd-after-receipt"
            target.mkdir()
            self.assertEqual(
                (root / "markers/cmd-control-plane.py").read_bytes(), command_bytes
            )
            self.assertEqual(
                (root / "crates/tenferro-runtime/tests/execution_engine_identity.rs").read_bytes(),
                artifact_bytes,
            )
            cwd_link.unlink()
            _create_required_symlink(
                cwd_link,
                Path("../cwd-after-receipt"),
                operation="post-receipt-cwd-retarget",
            )
            result = self.run_repository_checker(
                root, base_commit, receipt_path, "--diagnostics-json"
            )
            self.assert_result_diagnostic(
                result,
                "E_RECEIPT_PATH_IDENTITY",
                fields={
                    "obligation_id": "p0-control-plane",
                    "field": "cwd",
                    "expected": str(root.resolve()),
                    "actual": str(target.resolve()),
                },
            )

    def test_post_receipt_command_path_symlink_retarget_is_rejected(self) -> None:
        _require_v2_runner()
        command_path = "links/cmd-control-plane.py"
        manifest = valid_manifest(marker=True)
        manifest = _replace_in_obligation(
            manifest,
            "p0-control-plane",
            'argv = ["python3", "markers/cmd-control-plane.py", '
            '"crates/tenferro-runtime/tests/execution_engine_identity.rs"]',
            f'argv = ["python3", "{command_path}", '
            '"crates/tenferro-runtime/tests/execution_engine_identity.rs"]',
        )
        manifest = _replace_in_obligation(
            manifest,
            "p0-control-plane",
            'path_args = ["markers/cmd-control-plane.py", '
            '"crates/tenferro-runtime/tests/execution_engine_identity.rs"]',
            f'path_args = ["{command_path}", '
            '"crates/tenferro-runtime/tests/execution_engine_identity.rs"]',
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit = _init_git_repository(root, manifest, files=marker_files())
            link = root / command_path
            link.parent.mkdir(parents=True, exist_ok=True)
            command_bytes = (root / "markers/cmd-control-plane.py").read_bytes()
            identical_target = root / "links/cmd-control-plane-identical.py"
            identical_target.write_bytes(command_bytes)
            identical_target.chmod(0o755)
            _create_required_symlink(
                link,
                Path("../markers/cmd-control-plane.py"),
                operation="post-receipt-command-initial",
            )
            base_commit = _commit(root, "base internal command symlink")
            (root / "scripts/storage-ownership-contracts.toml").write_text(
                manifest, encoding="utf-8"
            )
            (root / "candidate-note").write_text("candidate\n", encoding="utf-8")
            _commit(root, "candidate with internal command symlink")
            receipt_path = root / "receipt.json"
            runner = self.run_repository_runner(root, base_commit, receipt_path)
            self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
            expected_path = str((root / "markers/cmd-control-plane.py").resolve())
            link.unlink()
            _create_required_symlink(
                link,
                Path("cmd-control-plane-identical.py"),
                operation="post-receipt-command-in-repo-retarget",
            )
            self.assertTrue(identical_target.is_file())
            self.assertTrue(os.access(identical_target, os.X_OK))
            self.assertEqual(identical_target.read_bytes(), command_bytes)
            self.assertEqual(link.resolve().read_bytes(), command_bytes)
            result = self.run_repository_checker(
                root, base_commit, receipt_path, "--diagnostics-json"
            )
            self.assert_result_diagnostic(
                result,
                "E_RECEIPT_PATH_IDENTITY",
                fields={
                    "obligation_id": "p0-control-plane",
                    "field": "argv[1].resolved_path",
                    "expected": expected_path,
                    "actual": str(identical_target.resolve()),
                },
            )

    def test_promotion_rejects_artifact_or_command_identity_change(self) -> None:
        for case in ("artifact", "command"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                _require_v2_checker()
                root = Path(temporary)
                base = valid_manifest()
                base_commit = _init_git_repository(root, base, files=repository_files())
                candidate = valid_manifest(
                    deferred_override=(
                        {
                            "p3-host-owner": (
                                "crates/tenferro-tensor/tests/ui/storage/fail/changed.rs"
                            )
                        }
                        if case == "artifact"
                        else None
                    ),
                    promote=CUTOVER_CANDIDATE_OBLIGATIONS,
                )
                if case == "command":
                    candidate = _replace_in_obligation(
                        candidate,
                        "p3-host-owner",
                        '"storage_compile_contract"]',
                        '"forged_storage_compile_contract"]',
                    )
                _materialize_active_artifacts(root, candidate)
                (root / "scripts/storage-ownership-contracts.toml").write_text(
                    candidate, encoding="utf-8"
                )
                _commit(root, "changed immutable identity")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(CHECKER),
                        "--root",
                        str(root),
                        "--manifest",
                        "scripts/storage-ownership-contracts.toml",
                        "--base-commit",
                        base_commit,
                        "--diagnostics-json",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assert_result_diagnostic(
                    result,
                    "E_PROMOTION_IDENTITY",
                    fields={"obligation_id": "p3-host-owner"},
                )

    def test_runner_emits_exact_candidate_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit, candidate_commit, candidate, _, receipt = self.emit_runner_receipt(root)
            self.assertEqual(set(receipt), RECEIPT_FIELDS)
            log = (root / "runner.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(receipt["base_commit"], base_commit)
            self.assertEqual(receipt["candidate_commit"], candidate_commit)
            self.assertEqual(
                receipt["base_manifest_sha256"],
                _sha256_bytes(
                    _git_show_bytes(root, base_commit, "scripts/storage-ownership-contracts.toml")
                ),
            )
            self.assertEqual(
                receipt["candidate_manifest_sha256"],
                _sha256_bytes((root / "scripts/storage-ownership-contracts.toml").read_bytes()),
            )
            manifest_rows = {
                row["id"]: row for row in tomllib.loads(candidate)["obligations"]
            }
            active_rows = [
                row for row in manifest_rows.values() if row["state"]["kind"] == "active"
            ]
            executions = _receipt_executions(receipt)
            self.assertEqual(len(executions), len(active_rows))
            self.assertEqual(
                Counter(row["obligation_id"] for row in executions),
                Counter(row["id"] for row in active_rows),
            )
            observations = _child_observations(root)
            self.assertEqual(len(observations), len(active_rows))
            self.assertEqual(
                len({observation["nonce"] for _, observation in observations}),
                len(observations),
            )
            self.assertEqual(
                len({observation["challenge"] for _, observation in observations}),
                len(observations),
            )
            observations_by_key = {}
            for observation_path, observation in observations:
                key = (observation["command_id"], observation["artifact_path"])
                self.assertNotIn(key, observations_by_key)
                observations_by_key[key] = (observation_path, observation)
            active_observation_keys = {
                (
                    row["command"]["id"],
                    _resolve_manifest_path(root, row["artifact"]["path"]),
                )
                for row in active_rows
            }
            self.assertEqual(set(observations_by_key), active_observation_keys)
            for execution in executions:
                obligation_id = execution["obligation_id"]
                self.assertIn(obligation_id, manifest_rows)
                manifest_row = manifest_rows[obligation_id]
                observation_key = (
                    manifest_row["command"]["id"],
                    _resolve_manifest_path(root, manifest_row["artifact"]["path"]),
                )
                self.assertIn(observation_key, observations_by_key)
                _, observation = observations_by_key[observation_key]
                self.assert_child_observation(root, manifest_row, observation)
                self.assertEqual(set(execution), RECEIPT_EXECUTION_FIELDS)
                self.assertEqual(obligation_id, manifest_row["id"])
                self.assertEqual(execution["artifact_id"], manifest_row["artifact"]["id"])
                self.assertEqual(execution["command_id"], manifest_row["command"]["id"])
                self.assertEqual(execution["candidate_commit"], candidate_commit)
                self.assertEqual(execution["exit_code"], 0)
                self.assertEqual(
                    execution["artifact_sha256"],
                    _sha256_resolved_path(root, manifest_row["artifact"]["path"]),
                )
                self.assertEqual(
                    execution["command_sha256"], _sha256_command(manifest_row["command"])
                )
                self.assertEqual(execution["argv"], observation["normalized_process_argv"])
                self.assertEqual(execution["cwd"], observation["cwd"])
                self.assertEqual(execution["artifact_path"], observation["artifact_path"])
                self.assertEqual(execution["executable"], observation["executable"])
                self.assertEqual(execution["observation_nonce"], observation["nonce"])
                self.assertEqual(
                    execution["observation_challenge"], observation["challenge"]
                )
            for row in BASE_ACTIVE_OBLIGATIONS:
                self.assertEqual(log.count(row[6]), 1)
            observed_command_ids = {observation["command_id"] for _, observation in observations}
            for row in DEFERRED_OBLIGATIONS:
                self.assertNotIn(row[6], log)
                self.assertNotIn(row[6], observed_command_ids)

    def test_receipt_envelope_missing_and_extra_fields_are_rejected(self) -> None:
        cases = tuple(
            ("missing-field", field, "present", "missing")
            for field in sorted(RECEIPT_FIELDS)
        ) + (("extra-field", "terminal", "absent", "present"),)
        for case, field, expected, actual in cases:
            with self.subTest(
                case=case, field=field
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_commit, _, _, receipt_path, receipt = self.emit_runner_receipt(
                    root
                )
                if case == "missing-field":
                    del receipt[field]
                else:
                    receipt[field] = True
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                result = self.run_repository_checker(
                    root, base_commit, receipt_path, "--diagnostics-json"
                )
                self.assert_result_diagnostic(
                    result,
                    "E_RECEIPT_SHAPE",
                    fields={
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    },
                )

    def test_runner_executes_canonical_cargo_command_with_path_shim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _cargo_path_shim_manifest()
            files = marker_files(manifest)
            files["bin/cargo"] = _cargo_path_shim_source()
            runner_env = dict(os.environ)
            runner_env["PATH"] = os.pathsep.join(
                (str(root / "bin"), runner_env.get("PATH", ""))
            )
            _, _, candidate, _, receipt = self.emit_runner_receipt(
                root,
                manifest_override=manifest,
                files_override=files,
                runner_env=runner_env,
                executable_paths=("bin/cargo",),
            )
            rows = {row["id"]: row for row in tomllib.loads(candidate)["obligations"]}
            cargo_row = rows["p0-control-plane"]
            self.assertEqual(cargo_row["command"]["kind"], "cargo-test")
            self.assertEqual(cargo_row["command"]["argv"][0], "cargo")
            self.assertNotIn("bin/cargo", cargo_row["command"]["argv"])
            observations = _child_observations(root)
            cargo_observation = next(
                observation
                for _, observation in observations
                if observation["command_id"] == cargo_row["command"]["id"]
            )
            self.assert_child_observation(
                root,
                cargo_row,
                cargo_observation,
                executable_search_path=runner_env["PATH"],
            )
            self.assertEqual(cargo_observation["executable"]["requested"], "cargo")
            self.assertEqual(
                cargo_observation["executable"]["resolved"],
                str((root / "bin/cargo").resolve()),
            )
            execution = _receipt_execution(receipt, "p0-control-plane")
            self.assertEqual(execution["argv"], cargo_observation["normalized_process_argv"])
            self.assertEqual(execution["cwd"], cargo_observation["cwd"])

    def test_receipt_execution_identity_mutations_are_rejected(self) -> None:
        mutations = {
            "artifact_id": "artifact-contract-document",
            "command_id": "cmd-contract-document",
            "candidate_commit": "f" * 40,
            "argv": ["forged-executable", "forged-argument"],
            "cwd": "/tmp/forged-cwd",
            "artifact_path": "/tmp/forged-artifact",
            "executable": {
                "requested": "python3",
                "resolved": "/tmp/forged-python",
                "sha256": "0" * 64,
            },
            "observation_nonce": "forged-observation-nonce",
            "observation_challenge": "0" * 64,
        }
        for field, actual in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_commit, candidate_commit, candidate, receipt_path, receipt = (
                    self.emit_runner_receipt(root)
                )
                execution = _receipt_execution(receipt, "p0-control-plane")
                expected = execution[field]
                execution[field] = actual
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                result = self.run_repository_checker(
                    root, base_commit, receipt_path, "--diagnostics-json"
                )
                self.assert_result_diagnostic(
                    result,
                    "E_RECEIPT_EXECUTION_BINDING",
                    fields={
                        "obligation_id": "p0-control-plane",
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    },
                )

    def test_child_observation_mutations_are_rejected(self) -> None:
        for case in ("missing", "duplicate", "forged", "swapped"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_commit, _, candidate, receipt_path, receipt = self.emit_runner_receipt(
                    root
                )
                rows = {row["id"]: row for row in tomllib.loads(candidate)["obligations"]}
                p0_row = rows["p0-control-plane"]
                p1_row = rows["p1-ledger"]
                observations = {
                    observation["command_id"]: (path, observation)
                    for path, observation in _child_observations(root)
                }
                p0_path, p0_observation = observations[p0_row["command"]["id"]]
                p1_path, p1_observation = observations[p1_row["command"]["id"]]
                if case == "missing":
                    p0_path.unlink()
                    expected_fields = {
                        "obligation_id": p0_row["id"],
                        "field": "observation_count",
                        "expected": 1,
                        "actual": 0,
                    }
                elif case == "duplicate":
                    p0_path.with_name(f"{p0_path.stem}-duplicate.json").write_text(
                        json.dumps(p0_observation), encoding="utf-8"
                    )
                    expected_fields = {
                        "obligation_id": p0_row["id"],
                        "field": "observation_count",
                        "expected": 1,
                        "actual": 2,
                    }
                elif case == "forged":
                    forged = dict(p0_observation)
                    forged["artifact_sha256"] = "0" * 64
                    p0_path.write_text(json.dumps(forged), encoding="utf-8")
                    expected_fields = {
                        "obligation_id": p0_row["id"],
                        "field": "artifact_sha256",
                        "expected": p0_observation["artifact_sha256"],
                        "actual": "0" * 64,
                    }
                else:
                    p0_path.write_text(json.dumps(p1_observation), encoding="utf-8")
                    p1_path.write_text(json.dumps(p0_observation), encoding="utf-8")
                    expected_fields = {
                        "obligation_id": p0_row["id"],
                        "field": "command_id",
                        "expected": p0_row["command"]["id"],
                        "actual": p1_row["command"]["id"],
                    }
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                result = self.run_repository_checker(
                    root, base_commit, receipt_path, "--diagnostics-json"
                )
                self.assert_result_diagnostic(
                    result,
                    "E_RECEIPT_OBSERVATION_BINDING",
                    fields=expected_fields,
                )

    def test_runner_is_fail_closed_on_command_failure(self) -> None:
        _require_v2_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = marker_files()
            files["markers/cmd-ledger.py"] = "raise SystemExit(23)\n"
            manifest = valid_manifest(marker=True)
            base_commit = _init_git_repository(root, manifest, files=files)
            (root / "candidate-note").write_text("candidate\n", encoding="utf-8")
            _commit(root, "candidate")
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--root",
                    str(root),
                    "--manifest",
                    "scripts/storage-ownership-contracts.toml",
                    "--base-commit",
                    base_commit,
                    "--receipt-out",
                    str(root / "receipt.json"),
                    "--diagnostics-json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            log_path = root / "runner.log"
            log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        self.assert_result_diagnostic(
            result,
            "E_COMMAND_FAILED",
            fields={"command_id": "cmd-ledger", "exit_code": 23},
        )
        self.assertNotIn("cmd-owner-compile", log)

    def test_terminal_true_requires_zero_deferred_and_complete_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit, _, _, receipt_path, receipt = self.emit_runner_receipt(
                root,
                promote=frozenset(row[0] for row in DEFERRED_OBLIGATIONS),
                message="terminal candidate",
            )
            terminal_execution_ids = {
                row["obligation_id"] for row in _receipt_executions(receipt)
            }
            self.assertTrue(
                {"p3-auto-trait-contract", "p4-provider-release-lifecycle"}
                <= terminal_execution_ids
            )
            result = self.run_repository_checker(
                root, base_commit, receipt_path, "--summary-json"
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["terminal"])

            _remove_receipt_execution(receipt, "p13-closure")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            failed = self.run_repository_checker(
                root, base_commit, receipt_path, "--diagnostics-json"
            )
        self.assert_result_diagnostic(
            failed, "E_RECEIPT_INCOMPLETE", fields={"obligation_id": "p13-closure"}
        )

    def test_json_summary_has_no_boolean_terminal_input(self) -> None:
        self.assertNotIn("terminal =", valid_manifest())
        self.assertNotIn("done =", valid_manifest())
        self.assertNotIn("status =", valid_manifest())

    def test_red_event_matching_rejects_duplicate_events(self) -> None:
        expected = [
            {
                "test": "sample",
                "kind": "failure",
                "params": None,
                "exception_type": "RedExpectedFailure",
                "cause": "test-fixture",
            }
        ]
        observed = [
            {
                "test": "sample",
                "kind": "failure",
                "params": None,
                "exception_type": "RedExpectedFailure",
                "cause": "test-fixture",
            },
            {
                "test": "sample",
                "kind": "failure",
                "params": None,
                "exception_type": "RedExpectedFailure",
                "cause": "test-fixture",
            },
        ]
        _, unexpected, missing, count_matches = _compare_red_events(expected, observed)
        self.assertFalse(count_matches)
        self.assertEqual(len(unexpected), 1)
        self.assertEqual(missing, [])

    def test_red_event_matching_rejects_unrelated_exception_and_wrong_cause(self) -> None:
        expected = [
            {
                "test": "sample",
                "kind": "failure",
                "params": None,
                "exception_type": "RedExpectedFailure",
                "cause": "v2-checker-not-implemented",
            }
        ]
        for observed in (
            {
                "test": "sample",
                "kind": "failure",
                "params": None,
                "exception_type": "AssertionError",
                "cause": None,
            },
            {
                "test": "sample",
                "kind": "error",
                "params": None,
                "exception_type": "RuntimeError",
                "cause": None,
            },
            {
                "test": "sample",
                "kind": "failure",
                "params": None,
                "exception_type": "RedExpectedFailure",
                "cause": "wrong-cause",
            },
        ):
            with self.subTest(observed=observed):
                _, unexpected, missing, count_matches = _compare_red_events(
                    expected, [observed]
                )
                self.assertFalse(count_matches)
                self.assertEqual(len(unexpected), 1)
                self.assertEqual(len(missing), 1)

    def test_red_event_matching_rejects_skipped_subtest(self) -> None:
        expected = [
            {
                "test": "sample",
                "kind": "subtest-failure",
                "params": {"case": "required"},
                "exception_type": "RedExpectedFailure",
                "cause": "future-production-proof-artifact-not-landed",
            }
        ]
        observed = [
            {
                "test": "sample",
                "kind": "skip",
                "params": {"case": "required"},
                "exception_type": None,
                "cause": None,
                "reason": "capability unavailable",
            }
        ]
        _, unexpected, missing, count_matches = _compare_red_events(expected, observed)
        self.assertFalse(count_matches)
        self.assertEqual(len(unexpected), 1)
        self.assertEqual(len(missing), 1)

    def test_red_result_preserves_subtest_failure_error_and_skip_metadata(self) -> None:
        import io

        expected_failure = RedExpectedFailure("test-fixture")
        expected_error = RedExpectedError("test-fixture")
        self.assertIsInstance(expected_failure, AssertionError)
        self.assertIsInstance(expected_error, RuntimeError)
        self.assertEqual(expected_failure.cause, "test-fixture")
        self.assertEqual(expected_error.cause, "test-fixture")

        class Probe(unittest.TestCase):
            def test_events(self) -> None:
                with self.subTest(case="failure"):
                    self.fail("unrelated assertion")
                with self.subTest(case="error"):
                    raise RuntimeError("unrelated runtime error")
                with self.subTest(case="skip"):
                    self.skipTest("capability unavailable")

        result = _RedResult(stream=io.StringIO(), descriptions=False, verbosity=0)
        unittest.defaultTestLoader.loadTestsFromTestCase(Probe).run(result)
        self.assertEqual(
            [event["kind"] for event in result.events],
            ["subtest-failure", "subtest-error", "skip"],
        )
        self.assertEqual(
            [event["params"] for event in result.events],
            [
                {"case": "failure"},
                {"case": "error"},
                {"case": "skip"},
            ],
        )
        self.assertEqual(result.events[0]["exception_type"], "AssertionError")
        self.assertIsNone(result.events[0]["cause"])
        self.assertEqual(result.events[1]["exception_type"], "RuntimeError")
        self.assertIsNone(result.events[1]["cause"])
        self.assertIsNone(result.events[2]["exception_type"])
        self.assertIsNone(result.events[2]["cause"])
        self.assertEqual(result.events[2]["reason"], "capability unavailable")


def _parse_registry(manifest: str) -> dict[str, list[tuple[str, str]]]:
    """Parse only the graph fields needed by the RED self-checks."""
    edges: list[tuple[str, str]] = []
    lines = manifest.splitlines()
    in_edge = False
    source: str | None = None
    for line in lines:
        if line.strip() == "[[registry.edges]]":
            in_edge = True
            source = None
        elif in_edge and line.startswith("from = "):
            source = json.loads(line.split("=", 1)[1].strip())
        elif in_edge and line.startswith("to = "):
            target = json.loads(line.split("=", 1)[1].strip())
            if source is None:
                raise AssertionError("graph edge target appeared before source")
            edges.append((source, target))
            in_edge = False
    return {"edges": edges}


def _subtest_parameters(subtest: unittest.case.TestCase) -> dict[str, object] | None:
    params = getattr(subtest, "params", None)
    if params is None:
        return None
    return {str(key): value for key, value in params.items()}


def _exception_metadata(err: object) -> tuple[str | None, str | None]:
    if not isinstance(err, tuple) or len(err) < 2:
        return None, None
    exception_type = err[0]
    exception = err[1]
    type_name = getattr(exception_type, "__name__", type(exception).__name__)
    cause = getattr(exception, "cause", None)
    if cause is not None and not isinstance(cause, str):
        cause = str(cause)
    return str(type_name), cause


def _is_assertion_failure(err: object) -> bool:
    if not isinstance(err, tuple) or not err:
        return False
    exception_type = err[0]
    return isinstance(exception_type, type) and issubclass(
        exception_type, AssertionError
    )


class _RedResult(unittest.TextTestResult):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[dict[str, object]] = []

    def _record(
        self,
        test: unittest.case.TestCase,
        kind: str,
        params: object,
        *,
        err: object | None = None,
        reason: str | None = None,
    ) -> None:
        exception_type, cause = _exception_metadata(err)
        event: dict[str, object] = {
            "test": test.id().rsplit(".", 1)[-1],
            "kind": kind,
            "params": params,
            "exception_type": exception_type,
            "cause": cause,
        }
        if reason is not None:
            event["reason"] = reason
        self.events.append(event)

    def addFailure(self, test: unittest.case.TestCase, err: object) -> None:
        self._record(test, "failure", _subtest_parameters(test), err=err)
        super().addFailure(test, err)

    def addError(self, test: unittest.case.TestCase, err: object) -> None:
        self._record(test, "error", _subtest_parameters(test), err=err)
        super().addError(test, err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        self._record(
            test,
            "skip",
            _subtest_parameters(test),
            reason=reason,
        )
        super().addSkip(test, reason)

    def addSubTest(
        self,
        test: unittest.case.TestCase,
        subtest: unittest.case.TestCase,
        err: object | None,
    ) -> None:
        if err is not None:
            kind = "subtest-failure" if _is_assertion_failure(err) else "subtest-error"
            self._record(
                test,
                kind,
                _subtest_parameters(subtest),
                err=err,
            )
        super().addSubTest(test, subtest, err)


def _expected_red_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for test, specification in RED_EXPECTED_FAILURES.items():
        subtests = specification.get("subtests")
        is_subtest = subtests is not None
        if subtests is None:
            subtests = [None]
        cause = str(specification["cause"])
        base_kind, exception_type = RED_CAUSE_EVENT_SHAPES[cause]
        kind = (
            f"subtest-{base_kind}"
            if is_subtest
            else base_kind
        )
        for params in subtests:
            events.append(
                {
                    "test": test,
                    "kind": kind,
                    "params": params,
                    "exception_type": exception_type,
                    "cause": cause,
                }
            )
    return events


def _red_event_key(event: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(event["test"]),
        str(event["kind"]),
        json.dumps(event["params"], sort_keys=True, separators=(",", ":")),
        json.dumps(event.get("exception_type"), sort_keys=True),
        json.dumps(event.get("cause"), sort_keys=True),
    )


def _compare_red_events(
    expected: list[dict[str, object]], observed: list[dict[str, object]]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    bool,
]:
    remaining = Counter(_red_event_key(event) for event in expected)
    unexpected: list[dict[str, object]] = []
    observed_report: list[dict[str, object]] = []
    for event in observed:
        key = _red_event_key(event)
        if remaining[key] > 0:
            remaining[key] -= 1
            classification = "expected"
        else:
            unexpected.append(event)
            classification = "unexpected"
        observed_report.append(
            {
                **event,
                "classification": classification,
            }
        )

    missing: list[dict[str, object]] = []
    for event in expected:
        key = _red_event_key(event)
        if remaining[key] > 0:
            missing.append(event)
            remaining[key] -= 1
    expected_counts = Counter(_red_event_key(event) for event in expected)
    observed_counts = Counter(_red_event_key(event) for event in observed)
    return observed_report, unexpected, missing, observed_counts == expected_counts


def _run_red_suite() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StorageOwnershipV2RedTests)
    runner = unittest.TextTestRunner(
        stream=sys.stderr,
        verbosity=1,
        resultclass=_RedResult,
    )
    result = runner.run(suite)
    expected = _expected_red_events()
    observed = result.events
    observed_report, unexpected, missing, count_matches = _compare_red_events(
        expected, observed
    )
    report = {
        "schema": "tenferro.storage-ownership-red-report.v1",
        "tests_run": result.testsRun,
        "expected_failure_count": len(expected),
        "observed_failure_count": len(observed),
        "expected_event_count": len(expected),
        "observed_event_count": len(observed),
        "event_count_matches": count_matches,
        "expected_failures": expected,
        "observed_failures": observed_report,
        "unexpected_failures": unexpected,
        "missing_expected_failures": missing,
        "skipped": [test.id() for test, _ in result.skipped],
    }
    print(json.dumps(report, sort_keys=True))
    if unexpected or missing or not count_matches:
        return 2
    return 1 if observed else 0


if __name__ == "__main__":
    sys.exit(_run_red_suite())
