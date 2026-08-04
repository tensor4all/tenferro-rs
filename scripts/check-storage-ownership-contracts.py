#!/usr/bin/env python3
"""Validate the schema-v2 storage-ownership contract ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib


SCHEMA = "tenferro.storage-ownership-contracts.v2"
RECEIPT_SCHEMA = "tenferro.storage-ownership-receipt.v1"
DIAGNOSTICS_SCHEMA = "tenferro.storage-ownership-diagnostics.v1"
CLI_SCHEMA = "tenferro.storage-ownership-cli-contract.v1"
CONTRACT_PROBE = "--contract-schema"
MANIFEST_RELATIVE_DEFAULT = "scripts/storage-ownership-contracts.toml"
ACTIVE_STATE = "active"
DEFERRED_STATE = "deferred"

CLI_CONTRACT = {
    "schema": CLI_SCHEMA,
    "tool": "check-storage-ownership-contracts",
    "role": "checker",
    "manifest_schema": SCHEMA,
    "probe": CONTRACT_PROBE,
    "options": [
        "--root",
        "--manifest",
        "--base-commit",
        "--receipt",
        "--summary-json",
        "--diagnostics-json",
    ],
}

RECEIPT_FIELDS = frozenset({"schema", "base_commit", "candidate_commit", "executions"})
EXECUTION_FIELDS = frozenset(
    {"obligation_id", "argv", "cwd", "artifact_path", "exit_code"}
)

DIAGNOSTIC_FIELDS = {
    "E_MANIFEST_INPUT": frozenset({"actual"}),
    "E_SCHEMA_VERSION": frozenset({"actual"}),
    "E_SCHEMA_PARALLEL_TABLE": frozenset({"table"}),
    "E_SCHEMA_UNKNOWN_TABLE": frozenset({"table"}),
    "E_OBLIGATION_TAGGED_STATE": frozenset({"obligation_id"}),
    "E_UNIT_OBLIGATION_MISSING": frozenset({"unit"}),
    "E_GRAPH_P2_PREREQUISITE": frozenset({"unit"}),
    "E_GRAPH_PREREQUISITE_INCOMPLETE": frozenset(
        {"source_unit", "target_unit", "obligation_id"}
    ),
    "E_GRAPH_DUPLICATE_EDGE": frozenset({"from", "to"}),
    "E_GRAPH_UNKNOWN_UNIT": frozenset({"unit"}),
    "E_COHORT_DEFINITION": frozenset({"cohort_id"}),
    "E_COHORT_PARTIAL_PROMOTION": frozenset({"cohort_id"}),
    "E_COHORT_PREREQUISITE_INCOMPLETE": frozenset({"unit", "obligation_id"}),
    "E_OBSOLETE_OWNERSHIP_TABLE": frozenset({"table"}),
    "E_ARTIFACT_SYNTHETIC_TERMINAL": frozenset({"artifact_id"}),
    "E_ARTIFACT_DUPLICATE_TARGET": frozenset({"artifact_id"}),
    "E_ARTIFACT_ID_CONFLICT": frozenset({"artifact_id"}),
    "E_ARTIFACT_MISSING": frozenset({"artifact_id"}),
    "E_DEFERRED_ARTIFACT_EXISTS": frozenset({"obligation_id"}),
    "E_PATH_ESCAPE": frozenset({"obligation_id"}),
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
    "E_COMMAND_ARTIFACT_BINDING": frozenset({"command_id"}),
    "E_COMMAND_TARGET_BINDING": frozenset({"command_id"}),
    "E_COMMAND_ID_CONFLICT": frozenset({"command_id"}),
    "E_COMMAND_FAILED": frozenset({"command_id", "exit_code"}),
    "E_PROMOTION_IDENTITY": frozenset({"obligation_id"}),
    "E_PROMOTION_REGISTRY": frozenset({"component"}),
    "E_RECEIPT_COMMIT": frozenset({"actual_head"}),
    "E_RECEIPT_SHAPE": frozenset({"field", "expected", "actual"}),
    "E_RECEIPT_EXECUTION_BINDING": frozenset(
        {"obligation_id", "field", "expected", "actual"}
    ),
    "E_RECEIPT_INCOMPLETE": frozenset({"obligation_id"}),
    "E_RECEIPT_TRACKING": frozenset({"path", "status"}),
    "E_TERMINAL_DECLARED": frozenset({"field"}),
}


def _policy(
    kind: str,
    argv: tuple[str, ...],
    path_args: tuple[str, ...],
    artifact_id: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    return kind, argv, path_args, artifact_id


# This is executable command policy: lifecycle ownership, state, and graph
# membership remain solely in the tagged manifest rows.
COMMAND_POLICY = {
    "cmd-control-plane": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-runtime", "--test", "integration"),
        (),
        "artifact-control-plane",
    ),
    "cmd-ledger": _policy(
        "python-test",
        ("python3", "scripts/check-storage-ownership-contracts.py"),
        ("scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"),
        "artifact-ledger",
    ),
    "cmd-contract-document": _policy(
        "doc-check",
        ("python3", "scripts/check-storage-design-docs.py"),
        ("scripts/check-storage-design-docs.py", "docs/design/storage-ownership-contracts.md"),
        "artifact-contract-document",
    ),
    "cmd-api-parity": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_api_parity"),
        (),
        "artifact-api-parity",
    ),
    "cmd-element-access-baseline": _policy(
        "benchmark-check",
        ("python3", "scripts/verify-storage-element-access-baseline.py", "--report", "docs/testing/storage-element-access-baseline.json"),
        ("scripts/verify-storage-element-access-baseline.py", "docs/testing/storage-element-access-baseline.json"),
        "artifact-element-access-baseline",
    ),
    "cmd-root-claims": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_root_claims"),
        (),
        "artifact-root-claims",
    ),
    "cmd-production-borrow-contract": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_borrow_contract"),
        (),
        "artifact-production-borrow-contract",
    ),
    "cmd-owner-compile": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"),
        (),
        "artifact-owner-compile",
    ),
    "cmd-static-rank-preservation": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_static_rank"),
        (),
        "artifact-static-rank-preservation",
    ),
    "cmd-as-view-zero-allocation": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_as_view_allocation"),
        (),
        "artifact-as-view-zero-allocation",
    ),
    "cmd-storage-auto-traits": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_auto_traits"),
        (),
        "artifact-storage-auto-traits",
    ),
    "cmd-prepared-validation-boundary": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_prepared_validation"),
        (),
        "artifact-prepared-validation-boundary",
    ),
    "cmd-provider-event-retirement": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_provider_event_retirement"),
        (),
        "artifact-provider-event-retirement",
    ),
    "cmd-traversal-resolution-counts": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_traversal_resolution"),
        (),
        "artifact-traversal-resolution-counts",
    ),
    "cmd-prepared-access-api": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_prepared_access"),
        (),
        "artifact-prepared-access-api",
    ),
    "cmd-allocation-group": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_allocation_group"),
        (),
        "artifact-allocation-group",
    ),
    "cmd-submit-compile": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"),
        (),
        "artifact-submit-compile",
    ),
    "cmd-reinterpret": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_reinterpret"),
        (),
        "artifact-reinterpret",
    ),
    "cmd-reinterpret-rank-policy": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_reinterpret_rank"),
        (),
        "artifact-reinterpret-rank-policy",
    ),
    "cmd-cuda-provider": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-cuda", "--test", "storage_provider"),
        (),
        "artifact-cuda-provider",
    ),
    "cmd-webgpu-metal-provider": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-wgpu", "--test", "storage_provider"),
        (),
        "artifact-webgpu-metal-provider",
    ),
    "cmd-api-normalization": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_public_api"),
        (),
        "artifact-api-normalization",
    ),
    "cmd-element-hot-path-structure": _policy(
        "python-test",
        ("python3", "scripts/check-storage-element-hot-path.py"),
        ("scripts/check-storage-element-hot-path.py",),
        "artifact-element-hot-path-structure",
    ),
    "cmd-storage-traversal-performance": _policy(
        "benchmark-check",
        ("python3", "scripts/verify-storage-traversal-performance.py", "--baseline-obligation", "p1-element-access-baseline", "--baseline-report", "docs/testing/storage-element-access-baseline.json", "--report", "docs/testing/storage-traversal-performance.md"),
        ("scripts/verify-storage-traversal-performance.py", "docs/testing/storage-element-access-baseline.json", "docs/testing/storage-traversal-performance.md"),
        "artifact-storage-traversal-performance",
    ),
    "cmd-static-rank-codegen": _policy(
        "codegen-check",
        ("python3", "scripts/check-storage-static-rank-codegen.py", "--report", "docs/testing/storage-static-rank-codegen.md"),
        ("scripts/check-storage-static-rank-codegen.py", "docs/testing/storage-static-rank-codegen.md"),
        "artifact-static-rank-codegen",
    ),
    "cmd-contract-freeze": _policy(
        "doc-check",
        ("python3", "scripts/check-storage-contract-freeze.py", "--report", "docs/design/storage-contract-freeze.md"),
        ("scripts/check-storage-contract-freeze.py", "docs/design/storage-contract-freeze.md"),
        "artifact-contract-freeze",
    ),
    "cmd-hardware-matrix": _policy(
        "doc-check",
        ("python3", "scripts/check-storage-hardware-matrix.py", "--report", "docs/testing/storage-hardware-matrix.md"),
        ("scripts/check-storage-hardware-matrix.py", "docs/testing/storage-hardware-matrix.md"),
        "artifact-hardware-matrix",
    ),
    "cmd-storage-guide": _policy(
        "doc-check",
        ("python3", "scripts/check-storage-docs.py", "--include-rendered"),
        ("scripts/check-storage-docs.py", "docs/storage-ownership.md"),
        "artifact-storage-guide",
    ),
    "cmd-element-access-guide": _policy(
        "doc-check",
        ("python3", "scripts/check-storage-element-access-docs.py", "docs/guides/views-and-slicing.md"),
        ("scripts/check-storage-element-access-docs.py", "docs/guides/views-and-slicing.md"),
        "artifact-element-access-guide",
    ),
    "cmd-element-access-examples": _policy(
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tutorial-code", "--release", "tutorial_binaries_run_successfully", "--", "--exact"),
        (),
        "artifact-element-access-examples",
    ),
    "cmd-storage-closure": _policy(
        "source-contract",
        ("python3", "scripts/check-storage-redesign-closure.py", "--report", "docs/worklogs/storage-redesign-closure.md"),
        ("scripts/check-storage-redesign-closure.py", "docs/worklogs/storage-redesign-closure.md"),
        "artifact-storage-closure",
    ),
}
ALLOWED_KINDS = frozenset(item[0] for item in COMMAND_POLICY.values())


class LedgerFailure(Exception):
    """A typed, machine-readable contract failure."""

    def __init__(self, code: str, fields: dict[str, object], message: str) -> None:
        super().__init__(message)
        if code not in DIAGNOSTIC_FIELDS or set(fields) != DIAGNOSTIC_FIELDS[code]:
            raise ValueError(f"invalid diagnostic shape for {code}")
        self.code = code
        self.fields = fields
        self.message = message


def _fail(code: str, fields: dict[str, object], message: str) -> None:
    raise LedgerFailure(code, fields, message)


def _diagnostic(error: LedgerFailure) -> dict[str, object]:
    return {"code": error.code, "fields": error.fields, "message": error.message}


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "git command failed")
    return result.stdout.strip()


def _canonical_commit(root: Path, revision: str | None) -> str | None:
    if revision is None:
        return None
    if not isinstance(revision, str) or not revision:
        raise ValueError("base commit revision must be a non-empty string")
    return _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")


def _git_bytes(root: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.decode(errors="replace").strip() or "Git object is unavailable")
    return result.stdout


def _git_is_ancestor(root: Path, base: str, head: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, head],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _load_toml_bytes(contents: bytes) -> dict[str, object]:
    try:
        data = tomllib.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot parse manifest: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("manifest must be a TOML table")
    return data


def _load_receipt(path: Path) -> dict[str, object]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load receipt '{path}': {error}") from error
    if not isinstance(receipt, dict):
        _fail("E_RECEIPT_SHAPE", {"field": "envelope", "expected": "object", "actual": type(receipt).__name__}, "receipt must be an object")
    missing = sorted(RECEIPT_FIELDS - set(receipt))
    if missing:
        _fail("E_RECEIPT_SHAPE", {"field": missing[0], "expected": "present", "actual": "missing"}, "receipt field is missing")
    extra = sorted(set(receipt) - RECEIPT_FIELDS)
    if extra:
        _fail("E_RECEIPT_SHAPE", {"field": extra[0], "expected": "absent", "actual": "present"}, "receipt contains an unexpected field")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        _fail("E_RECEIPT_SHAPE", {"field": "schema", "expected": RECEIPT_SCHEMA, "actual": str(receipt.get("schema"))}, "receipt schema is invalid")
    if not isinstance(receipt.get("candidate_commit"), str) or not receipt["candidate_commit"]:
        _fail("E_RECEIPT_SHAPE", {"field": "candidate_commit", "expected": "string", "actual": type(receipt.get("candidate_commit")).__name__}, "candidate commit must be a string")
    if receipt.get("base_commit") is not None and not isinstance(receipt.get("base_commit"), str):
        _fail("E_RECEIPT_SHAPE", {"field": "base_commit", "expected": "string or null", "actual": type(receipt.get("base_commit")).__name__}, "base commit must be a string or null")
    if not isinstance(receipt.get("executions"), list):
        _fail("E_RECEIPT_SHAPE", {"field": "executions", "expected": "array", "actual": type(receipt.get("executions")).__name__}, "receipt executions must be an array")
    return receipt


def _confined_path(root: Path, value: str, code: str, fields: dict[str, object]) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        _fail(code, fields, f"path '{value}' is outside the repository")
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(code, fields, f"path '{value}' resolves outside the repository")
    return resolved


def _path_like(value: str) -> bool:
    path = Path(value)
    return (
        path.is_absolute()
        or value.startswith(".")
        or "/" in value
        or "\\" in value
        or path.suffix in {".py", ".toml", ".rs", ".json", ".md"}
    )


def _validate_command(root: Path, row: dict[str, object]) -> Path:
    obligation_id = str(row.get("id"))
    command = row.get("command")
    artifact = row.get("artifact")
    if not isinstance(command, dict) or not isinstance(artifact, dict):
        raise ValueError(f"obligation '{obligation_id}' must contain command and artifact tables")
    command_id = command.get("id")
    if not isinstance(command_id, str) or command_id not in COMMAND_POLICY:
        _fail("E_COMMAND_ID_CONFLICT", {"command_id": str(command_id)}, "command ID is not registered")
    expected_kind, expected_argv, expected_path_args, expected_artifact_id = COMMAND_POLICY[command_id]
    kind = command.get("kind")
    argv = command.get("argv")
    path_args = command.get("path_args")
    cwd = command.get("cwd")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        _fail("E_COMMAND_ARGV", {"command_id": command_id}, "command argv must be a non-empty string array")
    if not isinstance(path_args, list) or not all(isinstance(item, str) for item in path_args):
        _fail("E_COMMAND_TARGET_BINDING", {"command_id": command_id}, "command path_args must be a string array")
    if not isinstance(cwd, str) or not cwd:
        _fail("E_COMMAND_CWD_ESCAPE", {"command_id": command_id, "cwd": str(cwd)}, "command cwd is invalid")
    cwd_path = _confined_path(root, cwd, "E_COMMAND_CWD_ESCAPE", {"command_id": command_id, "cwd": cwd})
    for index, argument in enumerate(argv[1:], start=1):
        if _path_like(argument):
            _confined_path(root, argument, "E_COMMAND_ARGV_PATH_ESCAPE", {"command_id": command_id, "index": index, "argument": argument})
    for argument in path_args:
        _confined_path(root, argument, "E_COMMAND_PATH_ESCAPE", {"command_id": command_id})
    if kind not in ALLOWED_KINDS or kind != expected_kind:
        _fail("E_COMMAND_KIND", {"command_id": command_id, "kind": str(kind)}, "command kind is not canonical")
    if len(argv) != len(expected_argv):
        _fail("E_COMMAND_ARGV_LENGTH", {"command_id": command_id, "expected": len(expected_argv), "actual": len(argv)}, "command argv length is not canonical")
    for index, (expected, actual) in enumerate(zip(expected_argv, argv)):
        if actual != expected:
            _fail("E_COMMAND_ARGV_BINDING", {"command_id": command_id, "index": index, "expected": expected, "actual": actual}, "command argv element is not canonical")
    artifact_id = artifact.get("id")
    if artifact_id != expected_artifact_id or command.get("artifact_id") != artifact_id:
        _fail("E_COMMAND_ARTIFACT_BINDING", {"command_id": command_id}, "command artifact binding is not canonical")
    if path_args != list(expected_path_args):
        _fail("E_COMMAND_TARGET_BINDING", {"command_id": command_id}, "command path target set is not canonical")
    return cwd_path


def _validate_registry(data: dict[str, object]) -> dict[str, object]:
    registry = data.get("registry")
    if not isinstance(registry, dict):
        raise ValueError("registry must be a table")
    if "ownerships" in registry:
        _fail("E_OBSOLETE_OWNERSHIP_TABLE", {"table": "registry.ownerships"}, "parallel ownership table is not permitted")
    for key in registry:
        if key not in {"revision", "gates", "units", "edges", "cohorts"}:
            _fail("E_SCHEMA_UNKNOWN_TABLE", {"table": f"registry.{key}"}, "registry contains an unknown table")
    revision = registry.get("revision", 1)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        _fail(
            "E_PROMOTION_REGISTRY",
            {"component": "revision"},
            "registry revision must be a positive integer",
        )
    values = [registry.get(key) for key in ("gates", "units", "edges", "cohorts")]
    if not all(isinstance(value, list) for value in values):
        raise ValueError("registry tables must be arrays")
    unit_ids = set()
    for row in registry["units"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("registry unit is invalid")
        if row["id"] in unit_ids:
            raise ValueError(f"duplicate registry unit '{row['id']}'")
        unit_ids.add(row["id"])
    gate_ids = set()
    for row in registry["gates"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("registry gate is invalid")
        if row["id"] in gate_ids:
            raise ValueError(f"duplicate registry gate '{row['id']}'")
        gate_ids.add(row["id"])
    edges = []
    for row in registry["edges"]:
        if not isinstance(row, dict):
            raise ValueError("registry edge is invalid")
        source, target = row.get("from"), row.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("registry edge endpoints are invalid")
        pair = (source, target)
        if pair in edges:
            _fail("E_GRAPH_DUPLICATE_EDGE", {"from": source, "to": target}, "registry contains a duplicate edge")
        if source not in unit_ids:
            _fail("E_GRAPH_UNKNOWN_UNIT", {"unit": source}, "edge references an unknown unit")
        if target not in unit_ids:
            _fail("E_GRAPH_UNKNOWN_UNIT", {"unit": target}, "edge references an unknown unit")
        edges.append(pair)
    if [source for source, target in edges if target == "P2"] != ["P1"]:
        _fail("E_GRAPH_P2_PREREQUISITE", {"unit": "P2"}, "P2 must depend directly on P1")
    if len(registry["cohorts"]) != 1 or not isinstance(registry["cohorts"][0], dict):
        _fail("E_COHORT_DEFINITION", {"cohort_id": "cutover"}, "cutover cohort is missing")
    cohort = registry["cohorts"][0]
    if cohort != {"id": "cutover", "prerequisites": ["P0", "P5"], "members": ["P3", "P9"]}:
        _fail("E_COHORT_DEFINITION", {"cohort_id": "cutover"}, "cutover cohort is not canonical")
    return {
        "revision": revision,
        "units": unit_ids,
        "gates": gate_ids,
        "edges": edges,
        "cohort": cohort,
    }


def _validate_manifest(root: Path, data: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if data.get("schema") != SCHEMA:
        _fail("E_SCHEMA_VERSION", {"actual": str(data.get("schema"))}, "manifest schema is not v2")
    if "terminal" in data:
        _fail("E_TERMINAL_DECLARED", {"field": "terminal"}, "terminal state is derived")
    for key in data:
        if key not in {"schema", "registry", "obligations"}:
            _fail("E_SCHEMA_UNKNOWN_TABLE", {"table": str(key)}, "manifest contains an unknown table")
    obligations_value = data.get("obligations")
    if isinstance(obligations_value, dict):
        child = next(iter(obligations_value), "active")
        _fail("E_SCHEMA_PARALLEL_TABLE", {"table": f"obligations.{child}"}, "obligations must be one array")
    if not isinstance(obligations_value, list):
        raise ValueError("obligations must be an array")
    registry = _validate_registry(data)
    rows = []
    obligation_ids = set()
    command_ids = set()
    artifact_ids = set()
    artifact_paths = set()
    for raw in obligations_value:
        if not isinstance(raw, dict):
            raise ValueError("obligation must be a table")
        if "active" in raw:
            _fail("E_SCHEMA_PARALLEL_TABLE", {"table": "obligations.active"}, "obligations must be one array")
        if "deferred" in raw:
            _fail("E_SCHEMA_PARALLEL_TABLE", {"table": "obligations.deferred"}, "obligations must be one array")
        if "terminal" in raw:
            _fail("E_TERMINAL_DECLARED", {"field": "terminal"}, "terminal state is derived")
        obligation_id = raw.get("id")
        if not isinstance(obligation_id, str) or not obligation_id:
            raise ValueError("obligation ID is invalid")
        if obligation_id in obligation_ids:
            raise ValueError(f"duplicate obligation '{obligation_id}'")
        obligation_ids.add(obligation_id)
        state = raw.get("state")
        if not isinstance(state, dict) or "status" in raw or not isinstance(state.get("kind"), str):
            _fail("E_OBLIGATION_TAGGED_STATE", {"obligation_id": obligation_id}, "obligation must use a tagged state")
        state_kind = state["kind"]
        if state_kind not in {ACTIVE_STATE, DEFERRED_STATE}:
            _fail("E_OBLIGATION_TAGGED_STATE", {"obligation_id": obligation_id}, "obligation state is invalid")
        if state_kind == ACTIVE_STATE and state != {"kind": ACTIVE_STATE}:
            _fail("E_OBLIGATION_TAGGED_STATE", {"obligation_id": obligation_id}, "active state is not canonical")
        unit = raw.get("unit")
        if unit not in registry["units"]:
            raise ValueError(f"obligation '{obligation_id}' references an unknown unit")
        gates = raw.get("gates")
        if not isinstance(gates, list) or not gates or not all(isinstance(gate, str) and gate in registry["gates"] for gate in gates):
            raise ValueError(f"obligation '{obligation_id}' has invalid gates")
        artifact = raw.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"obligation '{obligation_id}' artifact is invalid")
        artifact_id = artifact.get("id")
        artifact_kind = artifact.get("kind")
        artifact_path = artifact.get("path")
        if not all(isinstance(value, str) and value for value in (artifact_id, artifact_kind, artifact_path)):
            raise ValueError(f"obligation '{obligation_id}' artifact is invalid")
        if artifact_id in artifact_ids:
            _fail("E_ARTIFACT_ID_CONFLICT", {"artifact_id": artifact_id}, "artifact ID is duplicated")
        if artifact_path in artifact_paths:
            _fail("E_ARTIFACT_DUPLICATE_TARGET", {"artifact_id": artifact_id}, "artifact path is duplicated")
        artifact_ids.add(artifact_id)
        artifact_paths.add(artifact_path)
        if artifact_kind == "synthetic-terminal":
            _fail("E_ARTIFACT_SYNTHETIC_TERMINAL", {"artifact_id": artifact_id}, "synthetic terminal artifacts are not permitted")
        resolved_artifact = _confined_path(root, artifact_path, "E_PATH_ESCAPE", {"obligation_id": obligation_id})
        if state_kind == DEFERRED_STATE:
            if state != {"kind": DEFERRED_STATE, "activation_unit": unit, "promotion": {"mode": "activate-in-place"}}:
                _fail("E_OBLIGATION_TAGGED_STATE", {"obligation_id": obligation_id}, "deferred state is not canonical")
            if resolved_artifact.exists():
                _fail("E_DEFERRED_ARTIFACT_EXISTS", {"obligation_id": obligation_id}, "deferred artifact must not exist")
        elif not resolved_artifact.is_file():
            _fail("E_ARTIFACT_MISSING", {"artifact_id": artifact_id}, "active artifact is missing")
        command = raw.get("command")
        if not isinstance(command, dict):
            raise ValueError(f"obligation '{obligation_id}' command is invalid")
        command_id = command.get("id")
        if command_id in command_ids:
            _fail("E_COMMAND_ID_CONFLICT", {"command_id": str(command_id)}, "command ID is duplicated")
        command_ids.add(command_id)
        _validate_command(root, raw)
        rows.append(raw)
    for unit in sorted(registry["units"]):
        if not any(row.get("unit") == unit for row in rows):
            _fail("E_UNIT_OBLIGATION_MISSING", {"unit": unit}, "registered unit has no obligation")
    _validate_graph_prerequisites(rows, registry["edges"])
    _validate_cohort(rows, registry["cohort"])
    return rows, registry


def _validate_graph_prerequisites(
    rows: list[dict[str, object]], edges: list[tuple[str, str]]
) -> None:
    for source, target in edges:
        if not any(
            row["unit"] == target and row["state"]["kind"] == ACTIVE_STATE
            for row in rows
        ):
            continue
        for row in rows:
            if row["unit"] == source and row["state"]["kind"] != ACTIVE_STATE:
                _fail(
                    "E_GRAPH_PREREQUISITE_INCOMPLETE",
                    {
                        "source_unit": source,
                        "target_unit": target,
                        "obligation_id": row["id"],
                    },
                    "active target unit has an incomplete direct prerequisite",
                )


def _validate_cohort(rows: list[dict[str, object]], cohort: dict[str, object]) -> None:
    members = set(cohort["members"])
    member_rows = [row for row in rows if row["unit"] in members]
    active = [row for row in member_rows if row["state"]["kind"] == ACTIVE_STATE]
    deferred = [row for row in member_rows if row["state"]["kind"] == DEFERRED_STATE]
    if active and deferred:
        _fail("E_COHORT_PARTIAL_PROMOTION", {"cohort_id": "cutover"}, "cutover members must promote together")
    if active:
        for unit in cohort["prerequisites"]:
            if not any(row["unit"] == unit and row["state"]["kind"] == ACTIVE_STATE for row in rows):
                prerequisite = next((row["id"] for row in rows if row["unit"] == unit), unit)
                _fail("E_COHORT_PREREQUISITE_INCOMPLETE", {"unit": unit, "obligation_id": prerequisite}, "cutover prerequisite is not active")


def _manifest_relative(root: Path, value: str) -> tuple[str, Path]:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError("manifest path must be repository-relative")
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("manifest path resolves outside repository") from error
    return path.as_posix(), resolved


def _promotion_check(root: Path, data: dict[str, object], *, base_commit: str | None, manifest_relative: str) -> None:
    if base_commit is None:
        return
    head = _git(root, "rev-parse", "HEAD")
    if not _git_is_ancestor(root, base_commit, head):
        _fail("E_RECEIPT_COMMIT", {"actual_head": head}, "base commit is not an ancestor of HEAD")
    base_data = _load_toml_bytes(_git_bytes(root, base_commit, manifest_relative))
    base_registry = base_data.get("registry")
    candidate_registry = data.get("registry")
    if not isinstance(base_registry, dict) or not isinstance(candidate_registry, dict):
        _fail(
            "E_PROMOTION_REGISTRY",
            {"component": "registry"},
            "promotion changed the canonical registry",
        )
    base_revision = base_registry.get("revision", 1)
    candidate_revision = candidate_registry.get("revision", 1)
    if not isinstance(base_revision, int) or isinstance(base_revision, bool):
        base_revision = 1
    if not isinstance(candidate_revision, int) or isinstance(candidate_revision, bool):
        candidate_revision = -1
    base_topology = {key: value for key, value in base_registry.items() if key != "revision"}
    candidate_topology = {
        key: value for key, value in candidate_registry.items() if key != "revision"
    }
    if base_topology != candidate_topology:
        components = ("units", "edges", "gates", "cohorts")
        component = next(
            (
                key
                for key in components
                if base_registry.get(key) != candidate_registry.get(key)
            ),
            "registry",
        )
        _fail(
            "E_PROMOTION_REGISTRY",
            {"component": component},
            "promotion changed the canonical registry",
        )
    revision_changed = candidate_revision != base_revision
    if revision_changed and candidate_revision != base_revision + 1:
        _fail(
            "E_PROMOTION_REGISTRY",
            {"component": "revision"},
            "contract revision must advance by exactly one",
        )
    base_rows = base_data.get("obligations")
    candidate_rows = data.get("obligations")
    if not isinstance(base_rows, list) or not isinstance(candidate_rows, list):
        raise ValueError("base and candidate manifests must contain obligation arrays")
    base_by_id = {row.get("id"): row for row in base_rows if isinstance(row, dict)}
    candidate_by_id = {row.get("id"): row for row in candidate_rows if isinstance(row, dict)}
    if set(base_by_id) != set(candidate_by_id):
        differing = sorted((set(base_by_id) ^ set(candidate_by_id)), key=str)[0]
        _fail("E_PROMOTION_IDENTITY", {"obligation_id": str(differing)}, "promotion changed obligation membership")
    immutable = ("id", "unit", "gates", "artifact", "command")
    for obligation_id, candidate in candidate_by_id.items():
        base = base_by_id[obligation_id]
        base_state = base.get("state")
        candidate_state = candidate.get("state")
        if not isinstance(base_state, dict) or not isinstance(candidate_state, dict):
            continue
        if base_state.get("kind") == ACTIVE_STATE and candidate_state.get("kind") != ACTIVE_STATE:
            _fail("E_PROMOTION_IDENTITY", {"obligation_id": str(obligation_id)}, "an active obligation cannot be deferred")
        identity_changed = any(base.get(key) != candidate.get(key) for key in immutable)
        if revision_changed:
            if base_state != candidate_state:
                _fail(
                    "E_PROMOTION_IDENTITY",
                    {"obligation_id": str(obligation_id)},
                    "contract revision cannot change obligation state",
                )
            if base_state.get("kind") == ACTIVE_STATE and identity_changed:
                _fail(
                    "E_PROMOTION_IDENTITY",
                    {"obligation_id": str(obligation_id)},
                    "contract revision changed active obligation identity",
                )
        elif identity_changed:
            _fail("E_PROMOTION_IDENTITY", {"obligation_id": str(obligation_id)}, "candidate changed immutable obligation identity")


def _tracked_tree_clean(
    root: Path,
    rows: list[dict[str, object]] | None = None,
    manifest_relative: str | None = None,
) -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode == 1:
        _fail("E_RECEIPT_TRACKING", {"path": "<tracked-tree>", "status": "modified"}, "tracked files differ from candidate HEAD")
    if result.returncode != 0:
        raise ValueError(result.stderr.decode(errors="replace").strip() or "cannot inspect tracked tree")
    if rows is None or manifest_relative is None:
        return
    required_paths = {manifest_relative}
    for row in rows:
        if row["state"]["kind"] != ACTIVE_STATE:
            continue
        required_paths.add(row["artifact"]["path"])
        required_paths.update(row["command"]["path_args"])
    for relative in sorted(path for path in required_paths if path != "."):
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0:
            _fail("E_RECEIPT_TRACKING", {"path": relative, "status": "untracked"}, "receipt identity requires tracked inputs")


def _validate_receipt(
    root: Path,
    rows: list[dict[str, object]],
    receipt: dict[str, object],
    *,
    base_commit: str | None,
    manifest_relative: str,
) -> bool:
    head = _git(root, "rev-parse", "HEAD")
    if receipt["candidate_commit"] != head:
        _fail("E_RECEIPT_COMMIT", {"actual_head": head}, "receipt candidate commit is not HEAD")
    receipt_base = receipt["base_commit"]
    if base_commit != receipt_base:
        _fail("E_RECEIPT_COMMIT", {"actual_head": head}, "receipt base commit does not match the requested base")
    if receipt_base is not None and not _git_is_ancestor(root, receipt_base, head):
        _fail("E_RECEIPT_COMMIT", {"actual_head": head}, "receipt base commit is not an ancestor of HEAD")
    _tracked_tree_clean(root, rows, manifest_relative)
    executions = receipt["executions"]
    execution_by_id: dict[str, dict[str, object]] = {}
    for execution in executions:
        if not isinstance(execution, dict):
            _fail("E_RECEIPT_SHAPE", {"field": "execution", "expected": "object", "actual": type(execution).__name__}, "receipt execution must be an object")
        missing = sorted(EXECUTION_FIELDS - set(execution))
        if missing:
            _fail("E_RECEIPT_SHAPE", {"field": missing[0], "expected": "present", "actual": "missing"}, "receipt execution field is missing")
        extra = sorted(set(execution) - EXECUTION_FIELDS)
        if extra:
            _fail("E_RECEIPT_SHAPE", {"field": extra[0], "expected": "absent", "actual": "present"}, "receipt execution contains an unexpected field")
        obligation_id = execution["obligation_id"]
        if not isinstance(obligation_id, str) or obligation_id in execution_by_id:
            _fail("E_RECEIPT_SHAPE", {"field": "obligation_id", "expected": "unique string", "actual": str(obligation_id)}, "receipt execution identity is invalid")
        if not isinstance(execution["argv"], list) or not all(isinstance(value, str) for value in execution["argv"]):
            _fail("E_RECEIPT_SHAPE", {"field": "argv", "expected": "string array", "actual": type(execution["argv"]).__name__}, "receipt argv is invalid")
        if not all(isinstance(execution[field], str) for field in ("cwd", "artifact_path")):
            _fail("E_RECEIPT_SHAPE", {"field": "cwd", "expected": "string", "actual": "non-string"}, "receipt path identity is invalid")
        if not isinstance(execution["exit_code"], int) or isinstance(execution["exit_code"], bool):
            _fail("E_RECEIPT_SHAPE", {"field": "exit_code", "expected": "integer", "actual": type(execution["exit_code"]).__name__}, "receipt exit status is invalid")
        execution_by_id[obligation_id] = execution
    active_rows = sorted((row for row in rows if row["state"]["kind"] == ACTIVE_STATE), key=lambda row: row["id"])
    active_ids = {row["id"] for row in active_rows}
    for row in active_rows:
        if row["id"] not in execution_by_id:
            _fail("E_RECEIPT_INCOMPLETE", {"obligation_id": row["id"]}, "receipt lacks an active execution")
    extra_ids = set(execution_by_id) - active_ids
    if extra_ids:
        _fail("E_RECEIPT_SHAPE", {"field": "obligation_id", "expected": "active obligation", "actual": sorted(extra_ids)[0]}, "receipt contains a deferred execution")
    for row in active_rows:
        obligation_id = row["id"]
        command = row["command"]
        artifact = row["artifact"]
        execution = execution_by_id[obligation_id]
        expected = {
            "obligation_id": obligation_id,
            "argv": command["argv"],
            "cwd": command["cwd"],
            "artifact_path": artifact["path"],
        }
        for field, value in expected.items():
            if execution[field] != value:
                _fail("E_RECEIPT_EXECUTION_BINDING", {"obligation_id": obligation_id, "field": field, "expected": value, "actual": execution[field]}, "receipt execution does not match the candidate manifest")
        if execution["exit_code"] != 0:
            _fail("E_COMMAND_FAILED", {"command_id": command["id"], "exit_code": execution["exit_code"]}, "receipt records a failed command")
    return not any(row["state"]["kind"] == DEFERRED_STATE for row in rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default=MANIFEST_RELATIVE_DEFAULT)
    parser.add_argument("--base-commit")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--summary-json", action="store_true")
    parser.add_argument("--diagnostics-json", action="store_true")
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> int:
    root = arguments.root.resolve()
    if not root.is_dir():
        raise ValueError(f"repository root '{root}' is not a directory")
    manifest_relative, manifest_path = _manifest_relative(root, arguments.manifest)
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read manifest '{manifest_relative}': {error}") from error
    receipt = _load_receipt(arguments.receipt.resolve()) if arguments.receipt is not None else None
    effective_base = _canonical_commit(root, arguments.base_commit)
    if receipt is not None and effective_base is None:
        effective_base = _canonical_commit(root, receipt["base_commit"])
    data = _load_toml_bytes(manifest_bytes)
    _promotion_check(root, data, base_commit=effective_base, manifest_relative=manifest_relative)
    rows, _ = _validate_manifest(root, data)
    terminal = False
    if receipt is not None:
        terminal = _validate_receipt(
            root,
            rows,
            receipt,
            base_commit=effective_base,
            manifest_relative=manifest_relative,
        )
    if arguments.summary_json:
        print(json.dumps({"terminal": terminal}, sort_keys=True))
    else:
        print("storage ownership contract ledger: OK")
    return 0


def main() -> int:
    if sys.argv[1:] == [CONTRACT_PROBE]:
        print(json.dumps(CLI_CONTRACT, separators=(",", ":")))
        return 0
    try:
        return _run(_parse_args())
    except LedgerFailure as error:
        if "--diagnostics-json" in sys.argv[1:]:
            print(json.dumps({"schema": DIAGNOSTICS_SCHEMA, "diagnostics": [_diagnostic(error)]}, sort_keys=True))
        else:
            print(f"error: {error.message}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        if "--diagnostics-json" in sys.argv[1:]:
            diagnostic = LedgerFailure("E_MANIFEST_INPUT", {"actual": str(error)}, "unable to validate storage ownership manifest")
            print(json.dumps({"schema": DIAGNOSTICS_SCHEMA, "diagnostics": [_diagnostic(diagnostic)]}, sort_keys=True))
        else:
            print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
