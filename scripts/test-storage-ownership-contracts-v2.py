#!/usr/bin/env python3
"""Executable RED specification for the v2 storage-ownership ledger.

This file is a contract test, not the checker or runner.  The v2 checker and
runner are intentionally absent at this checkpoint.  The tests therefore
describe the required green behavior and are expected to fail until the
implementation phase lands.  Keeping the tests executable prevents the
ledger contract from becoming prose that can silently drift.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check-storage-ownership-contracts.py"
RUNNER = ROOT / "scripts" / "run-storage-ownership-contracts.py"
PRODUCTION_MANIFEST = ROOT / "scripts" / "storage-ownership-contracts.toml"
DESIGN_CONTRACT = ROOT / "docs" / "design" / "storage-ownership-contracts.md"

SCHEMA = "tenferro.storage-ownership-contracts.v2"
GATES = tuple(f"G{number}" for number in range(1, 8))

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
        "p1-borrow-dispatch-proof",
        "P1",
        ("G1", "G4"),
        "artifact-borrow-dispatch-proof",
        "scripts/test-storage-ownership-contracts-v2.py",
        "python-test",
        "cmd-borrow-dispatch-proof",
        "python-test",
        (
            "python3",
            "scripts/test-storage-ownership-contracts-v2.py",
            "StorageOwnershipV2RedTests.test_private_dispatch_borrow_shape_compiles",
        ),
        ("scripts/test-storage-ownership-contracts-v2.py",),
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
CUTOVER_CANDIDATE_OBLIGATIONS = frozenset(
    row[0] for row in DEFERRED_OBLIGATIONS if row[1] in {"P3", "P4", "P5", "P9"}
)
CUTOVER_PARTIAL_OBLIGATIONS = frozenset(
    row[0] for row in DEFERRED_OBLIGATIONS if row[1] in {"P4", "P5", "P9"}
)


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
                marker=marker,
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
                marker=marker,
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


def marker_files() -> dict[str, str]:
    files = repository_files()
    for row in ALL_OBLIGATIONS:
        command_id = row[6]
        files[f"markers/{command_id}.py"] = textwrap.dedent(
            f'''\
            import sys
            from pathlib import Path
            Path(sys.argv[1]).read_bytes()
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


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout.strip()


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

    def run_production_checker(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(ROOT),
                "--manifest",
                "scripts/storage-ownership-contracts.toml",
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
    ) -> subprocess.CompletedProcess[str]:
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
    ) -> tuple[str, str, str, Path, dict[str, object]]:
        base = valid_manifest(marker=True)
        base_commit = _init_git_repository(root, base, files=marker_files())
        candidate = valid_manifest(marker=True, promote=promote)
        _materialize_active_artifacts(root, candidate)
        (root / "scripts/storage-ownership-contracts.toml").write_text(candidate, encoding="utf-8")
        (root / "candidate-note").write_text(f"{message}\n", encoding="utf-8")
        candidate_commit = _commit(root, message)
        receipt_path = root / "receipt.json"
        runner = self.run_repository_runner(root, base_commit, receipt_path)
        self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertIsInstance(receipt, dict)
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
        self.assertEqual(payload.get("schema"), "tenferro.storage-ownership-diagnostics.v1")
        diagnostics = payload.get("diagnostics")
        self.assertIsInstance(diagnostics, list)
        self.assertTrue(diagnostics, payload)
        for diagnostic in diagnostics:
            self.assertIsInstance(diagnostic, dict)
            self.assertIsInstance(diagnostic.get("code"), str)
            self.assertIsInstance(diagnostic.get("fields"), dict)
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
        matching = [item for item in payload["diagnostics"] if item["code"] == code]
        self.assertTrue(matching, payload)
        for key, value in (fields or {}).items():
            self.assertTrue(
                any(item.get("fields", {}).get(key) == value for item in matching),
                payload,
            )

    def assert_result_diagnostic(
        self,
        result: subprocess.CompletedProcess[str],
        code: str,
        *,
        fields: dict[str, object] | None = None,
    ) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.parse_diagnostic_payload(result)
        matching = [item for item in payload["diagnostics"] if item["code"] == code]
        self.assertTrue(matching, payload)
        for key, value in (fields or {}).items():
            self.assertTrue(any(item.get("fields", {}).get(key) == value for item in matching), payload)

    def test_checked_in_production_manifest_is_the_gate_input(self) -> None:
        self.assertTrue(PRODUCTION_MANIFEST.is_file())
        production = tomllib.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(production.get("schema"), SCHEMA)
        result = self.run_production_checker()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_nominal_v2_manifest_is_green(self) -> None:
        result = self.run_checker(valid_manifest())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_v1_is_rejected_without_compatibility_mode(self) -> None:
        self.assert_checker_error(
            valid_manifest(schema="tenferro.storage-ownership-contracts.v1"),
            "E_SCHEMA_VERSION",
            fields={"actual": "tenferro.storage-ownership-contracts.v1"},
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
            "E_SCHEMA_PARALLEL_TABLE",
            fields={"table": "obligations.active"},
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

    def test_canonical_future_lifecycle_proof_commands_execute(self) -> None:
        rows = {row["id"]: row for row in tomllib.loads(valid_manifest())["obligations"]}
        for obligation_id in (
            "p3-auto-trait-contract",
            "p4-provider-release-lifecycle",
        ):
            with self.subTest(obligation_id=obligation_id):
                row = rows[obligation_id]
                artifact = ROOT / row["artifact"]["path"]
                self.assertTrue(artifact.is_file(), f"future proof artifact is absent: {artifact}")
                command = row["command"]
                result = subprocess.run(
                    command["argv"],
                    cwd=ROOT / command["cwd"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lease_thread_transfer_doc_signatures_are_supplemental(self) -> None:
        design = DESIGN_CONTRACT.read_text(encoding="utf-8")
        for signature in (
            "unsafe impl Send for BackendRawLease {}",
            "_not_sync: PhantomData<Cell<()>>",
            "_thread_bound: PhantomData<Rc<()>>",
            "assert_send::<UseLease>();",
            "let Some(parts) = self.pending.take() else { return Ok(()) };",
            "catch_unwind(AssertUnwindSafe",
            "QuarantineReason::ProviderReleasePanic",
        ):
            with self.subTest(signature=signature):
                self.assertIn(signature, design)
        self.assertNotIn("unsafe impl Sync for BackendRawLease", design)
        self.assertNotIn("unsafe impl Send for UseLease", design)

    def test_private_dispatch_borrow_shape_compiles(self) -> None:
        design = DESIGN_CONTRACT.read_text(encoding="utf-8")
        self.assertIn("fn dispatch_host_write<'a>", design)
        self.assertIn("fn dispatch_device_write<'a>", design)
        self.assertNotIn("fn backend_write_request", design)
        source = r'''
use std::{marker::PhantomData, sync::Arc};

struct Error;
struct Endpoint;
struct Raw;
struct Claim;
#[derive(Clone)] struct Span;
struct Provider;

impl Provider {
    fn write<'a>(&self, _request: &Request<'a>, _endpoint: Option<Endpoint>)
        -> Result<Raw, Error> { Ok(Raw) }
}

struct State { provider: Provider }
#[derive(Clone)] struct Pin { state: Arc<State> }
struct Request<'a> { _pin: &'a Pin, _claim: &'a mut Claim, _span: &'a Span }

impl State {
    fn dispatch_host_write<'a>(
        &'a self,
        pin: &'a Pin,
        claim: &'a mut Claim,
        span: &'a Span,
    ) -> Result<Raw, Error> {
        let request = Request { _pin: pin, _claim: claim, _span: span };
        self.provider.write(&request, None)
    }

    fn dispatch_device_write<'a>(
        &'a self,
        pin: &'a Pin,
        claim: &'a mut Claim,
        span: &'a Span,
        endpoint: Endpoint,
    ) -> Result<Raw, Error> {
        let request = Request { _pin: pin, _claim: claim, _span: span };
        self.provider.write(&request, Some(endpoint))
    }
}

struct Owner { pin: Pin, claim: Claim }
struct Capability<'a> { owner: &'a mut Owner }
struct ResolvedWrite<'a> { capability: Capability<'a>, span: Span }
struct Guard<'a> { _raw: Raw, _borrow: PhantomData<&'a mut [u8]> }
struct Lease { _pin: Pin, _span: Span, _raw: Raw }
struct Binding<'a> { _resolved: ResolvedWrite<'a>, _lease: Lease }

impl<'a> ResolvedWrite<'a> {
    fn acquire_host_write(&mut self) -> Result<Guard<'_>, Error> {
        let owner = &mut *self.capability.owner;
        let (pin, claim) = (&owner.pin, &mut owner.claim);
        let state = &*pin.state;
        let raw = state.dispatch_host_write(pin, claim, &self.span)?;
        Ok(Guard { _raw: raw, _borrow: PhantomData })
    }

    fn acquire_device_write(self, endpoint: Endpoint)
        -> Result<Binding<'a>, (Self, Error)> {
        let this = self;
        let admission = {
            let owner = &mut *this.capability.owner;
            let (pin, claim) = (&owner.pin, &mut owner.claim);
            let state = &*pin.state;
            state.dispatch_device_write(pin, claim, &this.span, endpoint)
        };
        match admission {
            Ok(raw) => Ok(Binding {
                _lease: Lease {
                    _pin: this.capability.owner.pin.clone(),
                    _span: this.span.clone(),
                    _raw: raw,
                },
                _resolved: this,
            }),
            Err(error) => Err((this, error)),
        }
    }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "private_dispatch_borrow_shape.rs"
            output_path = root / "private_dispatch_borrow_shape.rmeta"
            source_path.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [
                    "rustc",
                    "--edition=2021",
                    "--crate-type=lib",
                    "--emit=metadata",
                    "-o",
                    str(output_path),
                    str(source_path),
                ],
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
            "E_TERMINAL_DECLARED",
            fields={"field": "terminal"},
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = repository_files()
            _write_files(root, files)
            outside = root.parent / f"ledger-outside-{root.name}.rs"
            outside.write_text("fn outside() {}\n", encoding="utf-8")
            link = root / "scripts" / "escaped.toml"
            link.symlink_to(outside)
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
        self.assert_checker_error(shell, "E_COMMAND_KIND", fields={"kind": "shell"})
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

    def test_receipt_digests_bind_exact_manifest_artifact_and_command(self) -> None:
        for digest_kind in ("manifest", "artifact", "command"):
            with self.subTest(digest_kind=digest_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_commit, _, _, receipt_path, receipt = self.emit_runner_receipt(root)
                if digest_kind == "manifest":
                    receipt["candidate_manifest_sha256"] = "0" * 64
                elif digest_kind == "artifact":
                    _receipt_execution(receipt, "p0-control-plane")["artifact_sha256"] = "0" * 64
                else:
                    _receipt_execution(receipt, "p0-control-plane")["command_sha256"] = "0" * 64
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                result = self.run_repository_checker(
                    root, base_commit, receipt_path, "--diagnostics-json"
                )
                self.assert_result_diagnostic(
                    result, "E_RECEIPT_DIGEST", fields={"digest_kind": digest_kind}
                )

    def test_promotion_rejects_artifact_or_command_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = valid_manifest()
            base_commit = _init_git_repository(root, base, files=repository_files())
            candidate = valid_manifest(
                deferred_override={"p3-host-owner": "crates/tenferro-tensor/tests/ui/storage/fail/changed.rs"},
                promote=CUTOVER_CANDIDATE_OBLIGATIONS,
            )
            _materialize_active_artifacts(root, candidate)
            changed = root / "crates/tenferro-tensor/tests/ui/storage/fail/changed.rs"
            changed.parent.mkdir(parents=True, exist_ok=True)
            changed.write_text("fn changed_contract() {}\n", encoding="utf-8")
            (root / "scripts/storage-ownership-contracts.toml").write_text(candidate, encoding="utf-8")
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
            result, "E_PROMOTION_IDENTITY", fields={"obligation_id": "p3-host-owner"}
        )

    def test_runner_emits_exact_candidate_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_commit, candidate_commit, _, _, receipt = self.emit_runner_receipt(root)
            log = (root / "runner.log").read_text(encoding="utf-8").splitlines()
        self.assertEqual(receipt["base_commit"], base_commit)
        self.assertEqual(receipt["candidate_commit"], candidate_commit)
        self.assertEqual(
            {row["obligation_id"] for row in receipt["executions"]},
            {row[0] for row in BASE_ACTIVE_OBLIGATIONS},
        )
        for row in BASE_ACTIVE_OBLIGATIONS:
            self.assertEqual(log.count(row[6]), 1)
        for row in DEFERRED_OBLIGATIONS:
            self.assertNotIn(row[6], log)

    def test_runner_is_fail_closed_on_command_failure(self) -> None:
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
        self.assert_result_diagnostic(result, "E_COMMAND_FAILED", fields={"command_id": "cmd-ledger"})
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


if __name__ == "__main__":
    unittest.main()
