#!/usr/bin/env python3
"""Executable contract tests for the Phase-1 storage ownership ledger."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check-storage-ownership-contracts.py"
RUNNER = ROOT / "scripts/run-storage-ownership-contracts.py"
DOC_CHECKER = ROOT / "scripts/check-storage-design-docs.py"
MANIFEST = ROOT / "scripts/storage-ownership-contracts.toml"
V1_FIXTURE = ROOT / "scripts/fixtures/storage-ownership-contracts-v1.toml"

SCHEMA = "tenferro.storage-ownership-contracts.v2"
RECEIPT_SCHEMA = "tenferro.storage-ownership-receipt.v1"
DIAGNOSTICS_SCHEMA = "tenferro.storage-ownership-diagnostics.v1"

ACTIVE_IDS = frozenset(
    {
        "p0-control-plane",
        "p1-ledger",
        "p1-contract-document",
        "p1-api-parity",
        "p1-element-access-baseline",
        "p2-root-claims",
        "p4-production-borrow-contract",
        "p4-access-retirement",
        "p4-provider-release-lifecycle",
        "p4-traversal-resolution-counts",
        "p4-prepared-access-api",
        "p5-allocation-group",
        "p3-host-owner",
        "p3-static-rank-preservation",
        "p3-as-view-zero-allocation",
        "p3-auto-trait-contract",
        "p9-submission",
        "p6-reinterpret",
        "p6-reinterpret-rank-policy",
        "p7-cuda",
    }
)
DEFERRED_CORRECTIONS: dict[str, str] = {}

CHECKER_CLI = {
    "schema": "tenferro.storage-ownership-cli-contract.v1",
    "tool": "check-storage-ownership-contracts",
    "role": "checker",
    "manifest_schema": SCHEMA,
    "probe": "--contract-schema",
    "options": [
        "--root",
        "--manifest",
        "--base-commit",
        "--receipt",
        "--summary-json",
        "--diagnostics-json",
    ],
}
RUNNER_CLI = {
    **CHECKER_CLI,
    "tool": "run-storage-ownership-contracts",
    "role": "runner",
    "options": [
        "--root",
        "--manifest",
        "--base-commit",
        "--receipt-out",
        "--diagnostics-json",
    ],
}

RECEIPT_FIELDS = frozenset({"schema", "base_commit", "candidate_commit", "executions"})
EXECUTION_FIELDS = frozenset(
    {
        "obligation_id",
        "exit_code",
        "argv",
        "cwd",
        "artifact_path",
    }
)


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def _manifest_rows(manifest: str | None = None) -> list[dict[str, object]]:
    data = tomllib.loads(_manifest_text() if manifest is None else manifest)
    rows = data.get("obligations")
    if not isinstance(rows, list):
        raise AssertionError("canonical manifest must contain an obligation array")
    return rows


def _active_rows(manifest: str | None = None) -> list[dict[str, object]]:
    return [
        row for row in _manifest_rows(manifest) if row["state"]["kind"] == "active"
    ]


def _deferred_rows(manifest: str | None = None) -> list[dict[str, object]]:
    return [
        row for row in _manifest_rows(manifest) if row["state"]["kind"] == "deferred"
    ]


def _replace_once(manifest: str, old: str, new: str) -> str:
    if manifest.count(old) != 1:
        raise AssertionError(f"expected one occurrence of {old!r}")
    return manifest.replace(old, new, 1)


def _replace_row_state(manifest: str, obligation_id: str, new_state: str) -> str:
    marker = f'id = "{obligation_id}"'
    start = manifest.index(marker)
    end = manifest.find("\n[[obligations]]", start + len(marker))
    if end < 0:
        end = len(manifest)
    section = manifest[start:end]
    state_start = section.index("state = ")
    state_end = section.index("\n", state_start)
    replacement = section[:state_start] + f"state = {new_state}" + section[state_end:]
    return manifest[:start] + replacement + manifest[end:]


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def _fixture_files(manifest: str | None = None) -> dict[str, str]:
    files: dict[str, str] = {}
    for row in _active_rows(manifest):
        artifact = row["artifact"]
        files[artifact["path"]] = "fixture artifact\n"
        for argument in row["command"]["path_args"]:
            files.setdefault(argument, "fixture command target\n")
    return files


def _run_checker(
    manifest: str,
    *,
    files: dict[str, str] | None = None,
    root: Path | None = None,
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    owned_root = root is None
    temporary = tempfile.TemporaryDirectory() if owned_root else None
    try:
        target = Path(temporary.name) if temporary is not None else root
        assert target is not None
        if files is None:
            try:
                files = _fixture_files(manifest)
            except (AssertionError, KeyError, TypeError):
                files = {}
        _write_files(target, files)
        manifest_path = target / "scripts/storage-ownership-contracts.toml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest, encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(target),
                "--manifest",
                "scripts/storage-ownership-contracts.toml",
                *extra,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def _diagnostics(result: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(f"tool did not emit JSON diagnostics: {result.stdout!r}") from error
    if set(payload) != {"schema", "diagnostics"}:
        raise AssertionError(f"unexpected diagnostic envelope: {payload!r}")
    if payload["schema"] != DIAGNOSTICS_SCHEMA:
        raise AssertionError(f"unexpected diagnostic schema: {payload!r}")
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, list) or not diagnostics:
        raise AssertionError(f"diagnostics must be a non-empty array: {payload!r}")
    for diagnostic in diagnostics:
        if set(diagnostic) != {"code", "fields", "message"}:
            raise AssertionError(f"diagnostic shape is not exact: {diagnostic!r}")
        if not isinstance(diagnostic["message"], str) or not diagnostic["message"].strip():
            raise AssertionError(f"diagnostic message is empty: {diagnostic!r}")
    return diagnostics


def _assert_error(
    testcase: unittest.TestCase,
    result: subprocess.CompletedProcess[str],
    code: str,
    fields: dict[str, object] | None = None,
) -> None:
    testcase.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
    diagnostics = _diagnostics(result)
    testcase.assertEqual({item["code"] for item in diagnostics}, {code}, diagnostics)
    if fields is not None:
        testcase.assertEqual(diagnostics[0]["fields"], fields, diagnostics)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _commit(root: Path, message: str, *, paths: list[str] | None = None) -> str:
    _git(root, "add", *(paths or ["."]))
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _git_repository(
    manifest: str,
    files: dict[str, str],
) -> tuple[tempfile.TemporaryDirectory[str], Path, str, str]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    _write_files(root, files)
    manifest_path = root / "scripts/storage-ownership-contracts.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "ledger-tests@example.invalid")
    _git(root, "config", "user.name", "ledger-tests")
    base = _commit(root, "base")
    (root / "candidate-note").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(root, "candidate", paths=["candidate-note"])
    return temporary, root, base, candidate


def _runner_files() -> dict[str, str]:
    files = _fixture_files()
    files.update({
        "scripts/check-storage-ownership-contracts.py": CHECKER.read_text(encoding="utf-8"),
        "scripts/check-storage-design-docs.py": DOC_CHECKER.read_text(encoding="utf-8"),
        # This test exercises runner argv/receipt binding; the baseline verifier
        # itself has a dedicated 8-case test suite.
        "scripts/verify-storage-element-access-baseline.py": (
            "raise SystemExit(0)\n"
        ),
        "docs/design/storage-ownership-contracts.md": (
            ROOT / "docs/design/storage-ownership-contracts.md"
        ).read_text(encoding="utf-8"),
        "crates/tenferro-tensor/tests/storage_api_parity.rs": (
            ROOT / "crates/tenferro-tensor/tests/storage_api_parity.rs"
        ).read_text(encoding="utf-8"),
    })
    return files


def _runner(
    root: Path,
    base_commit: str,
    receipt: Path,
    *,
    environment: dict[str, str],
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
            str(receipt),
            *( ("--diagnostics-json",) if diagnostics else () ),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _checker_with_receipt(
    root: Path, base_commit: str, receipt: Path, *, diagnostics: bool = False
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
            str(receipt),
            "--summary-json",
            *( ("--diagnostics-json",) if diagnostics else () ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class StorageOwnershipV2Tests(unittest.TestCase):
    def test_cli_contracts_are_exact(self) -> None:
        for tool, expected in ((CHECKER, CHECKER_CLI), (RUNNER, RUNNER_CLI)):
            result = subprocess.run(
                [sys.executable, str(tool), "--contract-schema"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertEqual(json.loads(result.stdout), expected)

    def test_design_document_checker_is_real_and_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, str(DOC_CHECKER)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("storage-design-docs-ok", result.stdout)

    def test_production_manifest_has_current_active_rows(self) -> None:
        data = tomllib.loads(_manifest_text())
        self.assertEqual(data["schema"], SCHEMA)
        rows = data["obligations"]
        self.assertIsInstance(rows, list)
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        active = {row["id"] for row in rows if row["state"]["kind"] == "active"}
        self.assertEqual(active, ACTIVE_IDS)
        for obligation_id, activation_unit in DEFERRED_CORRECTIONS.items():
            row = next(row for row in rows if row["id"] == obligation_id)
            self.assertEqual(row["state"]["kind"], "deferred")
            self.assertEqual(row["state"]["activation_unit"], activation_unit)
        self.assertEqual(
            len(rows),
            len([row for row in rows if row["state"]["kind"] == "active"])
            + len([row for row in rows if row["state"]["kind"] == "deferred"]),
        )
        self.assertEqual(
            tomllib.loads(_manifest_text()),
            tomllib.loads(MANIFEST.read_text(encoding="utf-8")),
        )
        traversal = next(
            row for row in rows if row["id"] == "p10-storage-traversal-performance"
        )
        self.assertFalse(any("receipt" in argument for argument in traversal["command"]["argv"]))
        self.assertNotIn(
            ".storage-ownership-receipts/p1-element-access-baseline.json",
            traversal["command"]["path_args"],
        )

    def test_checked_in_manifest_passes_checker_and_is_nonterminal(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(ROOT),
                "--manifest",
                "scripts/storage-ownership-contracts.toml",
                "--summary-json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), {"terminal": False})

    def test_schema_only_v1_fixture_is_rejected(self) -> None:
        self.assertEqual(
            tomllib.loads(V1_FIXTURE.read_text(encoding="utf-8")),
            {"schema": "tenferro.storage-ownership-contracts.v1"},
        )
        result = _run_checker(
            'schema = "tenferro.storage-ownership-contracts.v1"',
            extra=("--diagnostics-json",),
        )
        _assert_error(self, result, "E_SCHEMA_VERSION", {"actual": "tenferro.storage-ownership-contracts.v1"})

    def test_obligations_use_one_tagged_state_table(self) -> None:
        status = _replace_row_state(_manifest_text(), "p1-ledger", '{ status = "active" }')
        result = _run_checker(status, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_OBLIGATION_TAGGED_STATE", {"obligation_id": "p1-ledger"})

        parallel = _manifest_text() + '\n[[obligations.active]]\nid = "extra"\n'
        result = _run_checker(parallel, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_SCHEMA_PARALLEL_TABLE", {"table": "obligations.active"})

        terminal = _manifest_text() + "\nterminal = true\n"
        result = _run_checker(terminal, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_TERMINAL_DECLARED", {"field": "terminal"})

    def test_registry_graph_and_cohort_are_structural(self) -> None:
        data = tomllib.loads(_manifest_text())
        registry = data["registry"]
        units = {row["id"] for row in registry["units"]}
        self.assertTrue(units)
        self.assertEqual(
            [(edge["from"], edge["to"]) for edge in registry["edges"] if edge["to"] == "P2"],
            [("P1", "P2")],
        )
        self.assertEqual(registry["cohorts"], [{
            "id": "cutover",
            "prerequisites": ["P0", "P5"],
            "members": ["P3", "P9"],
        }])

        duplicate = _replace_once(
            _manifest_text(),
            'from = "P1"\nto = "P2"',
            'from = "P1"\nto = "P2"\n\n[[registry.edges]]\nfrom = "P1"\nto = "P2"',
        )
        result = _run_checker(duplicate, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_GRAPH_DUPLICATE_EDGE", {"from": "P1", "to": "P2"})

    def test_active_unit_requires_all_direct_source_obligations_active(self) -> None:
        manifest = _replace_row_state(
            _replace_row_state(
                _manifest_text(),
                "p2-root-claims",
                '{ kind = "deferred", activation_unit = "P2", promotion = { mode = "activate-in-place" } }',
            ),
            "p4-production-borrow-contract",
            '{ kind = "active" }',
        )
        result = _run_checker(manifest, extra=("--diagnostics-json",))
        _assert_error(
            self,
            result,
            "E_GRAPH_PREREQUISITE_INCOMPLETE",
            {
                "source_unit": "P2",
                "target_unit": "P4",
                "obligation_id": "p2-root-claims",
            },
        )

    def test_active_and_deferred_artifacts_have_honest_filesystem_state(self) -> None:
        deferred = _deferred_rows()
        self.assertTrue(deferred)
        for row in deferred:
            self.assertFalse((ROOT / row["artifact"]["path"]).exists(), row["id"])

        row = deferred[0]
        files = _fixture_files()
        files[row["artifact"]["path"]] = "not a fabricated production artifact\n"
        result = _run_checker(
            _manifest_text(), files=files, extra=("--diagnostics-json",)
        )
        _assert_error(self, result, "E_DEFERRED_ARTIFACT_EXISTS", {"obligation_id": row["id"]})

    def test_path_confinement_precedes_command_identity(self) -> None:
        escaped_artifact = _replace_once(
            _manifest_text(),
            'path = "docs/design/storage-ownership-contracts.md"',
            'path = "../outside-storage-contracts.md"',
        )
        result = _run_checker(escaped_artifact, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_PATH_ESCAPE", {"obligation_id": "p1-contract-document"})

        escaped_cwd = _replace_once(
            _manifest_text(),
            'cwd = ".", path_args = ["scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"]',
            'cwd = "../outside", path_args = ["scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"]',
        )
        result = _run_checker(escaped_cwd, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_COMMAND_CWD_ESCAPE", {"command_id": "cmd-ledger", "cwd": "../outside"})

        escaped_argv = _replace_once(
            _manifest_text(),
            'argv = ["python3", "scripts/check-storage-ownership-contracts.py"]',
            'argv = ["python3", "../outside.py"]',
        )
        result = _run_checker(escaped_argv, extra=("--diagnostics-json",))
        _assert_error(
            self,
            result,
            "E_COMMAND_ARGV_PATH_ESCAPE",
            {"command_id": "cmd-ledger", "index": 1, "argument": "../outside.py"},
        )

    def test_command_allowlist_and_typed_argv_are_exact(self) -> None:
        mutated = _replace_once(
            _manifest_text(),
            'argv = ["python3", "scripts/check-storage-ownership-contracts.py"]',
            'argv = ["python3-mutated", "scripts/check-storage-ownership-contracts.py"]',
        )
        result = _run_checker(mutated, extra=("--diagnostics-json",))
        _assert_error(
            self,
            result,
            "E_COMMAND_ARGV_BINDING",
            {"command_id": "cmd-ledger", "index": 0, "expected": "python3", "actual": "python3-mutated"},
        )

        kind = _replace_once(
            _manifest_text(),
            'kind = "python-test", argv = ["python3", "scripts/check-storage-ownership-contracts.py"]',
            'kind = "shell", argv = ["python3", "scripts/check-storage-ownership-contracts.py"]',
        )
        result = _run_checker(kind, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_COMMAND_KIND", {"command_id": "cmd-ledger", "kind": "shell"})

    def test_artifact_identity_and_terminal_declarations_are_structural(self) -> None:
        synthetic = _replace_once(
            _manifest_text(),
            'kind = "manifest", path = "scripts/storage-ownership-contracts.toml"',
            'kind = "synthetic-terminal", path = "scripts/storage-ownership-contracts.toml"',
        )
        result = _run_checker(synthetic, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_ARTIFACT_SYNTHETIC_TERMINAL", {"artifact_id": "artifact-ledger"})

        duplicate = _replace_once(
            _manifest_text(),
            'path = "crates/tenferro-tensor/tests/storage_api_parity.rs"',
            'path = "scripts/storage-ownership-contracts.toml"',
        )
        result = _run_checker(duplicate, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_ARTIFACT_DUPLICATE_TARGET", {"artifact_id": "artifact-api-parity"})

    def test_promotion_changes_only_state_and_preserves_identity(self) -> None:
        base_manifest = _manifest_text()
        promoted = _replace_row_state(
            base_manifest,
            "p0-control-plane",
            '{ kind = "active" }',
        )
        files = _fixture_files(promoted)
        p0 = next(row for row in _manifest_rows(promoted) if row["id"] == "p0-control-plane")
        files[p0["artifact"]["path"]] = "candidate artifact\n"
        temporary, root, base, _ = _git_repository(base_manifest, _fixture_files(base_manifest))
        try:
            _write_files(root, files)
            (root / "scripts/storage-ownership-contracts.toml").write_text(promoted, encoding="utf-8")
            candidate = _commit(root, "promote p0")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    "scripts/storage-ownership-contracts.toml",
                    "--base-commit",
                    base,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(candidate, _git(root, "rev-parse", "HEAD"))

            changed = _replace_once(
                promoted,
                'path = "crates/tenferro-runtime/tests/integration/execution_engine_identity.rs"',
                'path = "crates/tenferro-runtime/tests/integration/execution_engine_identity-renamed.rs"',
            )
            (root / "scripts/storage-ownership-contracts.toml").write_text(changed, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    "scripts/storage-ownership-contracts.toml",
                    "--base-commit",
                    base,
                    "--diagnostics-json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            _assert_error(self, result, "E_PROMOTION_IDENTITY", {"obligation_id": "p0-control-plane"})
        finally:
            temporary.cleanup()

    def test_promotion_preserves_entire_registry_value(self) -> None:
        base_manifest = _manifest_text()
        mutations = (
            (
                "units",
                'name = "cuda"',
                'name = "cuda-renamed"',
            ),
            (
                "gates",
                'title = "G7 storage ownership gate"',
                'title = "G7 renamed storage ownership gate"',
            ),
            (
                "edges",
                'from = "P7"\nto = "P10"',
                'from = "P7"\nto = "P11"',
            ),
            (
                "cohorts",
                'prerequisites = ["P0", "P5"]',
                'prerequisites = ["P5", "P0"]',
            ),
        )
        for component, old, new in mutations:
            with self.subTest(component=component):
                candidate_manifest = _replace_once(base_manifest, old, new)
                temporary, root, base, _ = _git_repository(
                    base_manifest, _fixture_files(base_manifest)
                )
                try:
                    manifest_path = root / "scripts/storage-ownership-contracts.toml"
                    manifest_path.write_text(candidate_manifest, encoding="utf-8")
                    _commit(root, f"mutate registry {component}")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(CHECKER),
                            "--root",
                            str(root),
                            "--manifest",
                            "scripts/storage-ownership-contracts.toml",
                            "--base-commit",
                            base,
                            "--diagnostics-json",
                        ],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    _assert_error(
                        self,
                        result,
                        "E_PROMOTION_REGISTRY",
                        {"component": component},
                    )
                finally:
                    temporary.cleanup()

    def test_contract_revision_may_change_only_deferred_identity(self) -> None:
        base_manifest = _manifest_text().replace("revision = 4", "revision = 3", 1)
        revised = base_manifest.replace("revision = 3", "revision = 4", 1)
        revised = _replace_once(
            revised,
            'gates = ["G1", "G3", "G5"]\nartifact = { id = "artifact-webgpu-metal-provider"',
            'gates = ["G3"]\nartifact = { id = "artifact-webgpu-metal-provider"',
        )
        temporary, root, base, _ = _git_repository(
            base_manifest, _fixture_files(base_manifest)
        )
        try:
            (root / "scripts/storage-ownership-contracts.toml").write_text(
                revised, encoding="utf-8"
            )
            _commit(root, "revise deferred contract")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    "scripts/storage-ownership-contracts.toml",
                    "--base-commit",
                    base,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            changed_active = _replace_once(
                revised,
                'gates = ["G1", "G3", "G5"]\nartifact = { id = "artifact-ledger"',
                'gates = ["G1"]\nartifact = { id = "artifact-ledger"',
            )
            (root / "scripts/storage-ownership-contracts.toml").write_text(
                changed_active, encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    "scripts/storage-ownership-contracts.toml",
                    "--base-commit",
                    base,
                    "--diagnostics-json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            _assert_error(
                self,
                result,
                "E_PROMOTION_IDENTITY",
                {"obligation_id": "p1-ledger"},
            )
        finally:
            temporary.cleanup()

    def test_contract_revision_must_be_single_step_and_cannot_promote(self) -> None:
        base_manifest = _manifest_text().replace("revision = 4", "revision = 3", 1)
        for revision in (3, 5):
            with self.subTest(revision=revision):
                candidate = base_manifest.replace(
                    "revision = 3", f"revision = {revision}", 1
                )
                candidate = _replace_once(
                    candidate,
                    'gates = ["G1", "G3", "G5"]\nartifact = { id = "artifact-cuda-provider"',
                    'gates = ["G3"]\nartifact = { id = "artifact-cuda-provider"',
                )
                temporary, root, base, _ = _git_repository(
                    base_manifest, _fixture_files(base_manifest)
                )
                try:
                    (root / "scripts/storage-ownership-contracts.toml").write_text(
                        candidate, encoding="utf-8"
                    )
                    _commit(root, "invalid contract revision")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(CHECKER),
                            "--root",
                            str(root),
                            "--manifest",
                            "scripts/storage-ownership-contracts.toml",
                            "--base-commit",
                            base,
                            "--diagnostics-json",
                        ],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    _assert_error(
                        self,
                        result,
                        "E_PROMOTION_REGISTRY" if revision == 5 else "E_PROMOTION_IDENTITY",
                        {"component": "revision"} if revision == 5 else {"obligation_id": "p7-cuda"},
                    )
                finally:
                    temporary.cleanup()

        promoted = _manifest_text()
        base_manifest = _replace_row_state(
            promoted.replace("revision = 4", "revision = 3", 1),
            "p0-control-plane",
            '{ kind = "deferred", activation_unit = "P0", promotion = { mode = "activate-in-place" } }',
        )
        files = _fixture_files(promoted)
        row = next(row for row in _manifest_rows(promoted) if row["id"] == "p0-control-plane")
        files[row["artifact"]["path"]] = "candidate artifact\n"
        temporary, root, base, _ = _git_repository(
            base_manifest, _fixture_files(base_manifest)
        )
        try:
            _write_files(root, files)
            (root / "scripts/storage-ownership-contracts.toml").write_text(
                promoted, encoding="utf-8"
            )
            _commit(root, "invalid mixed revision and promotion")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    "scripts/storage-ownership-contracts.toml",
                    "--base-commit",
                    base,
                    "--diagnostics-json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            _assert_error(
                self,
                result,
                "E_PROMOTION_IDENTITY",
                {"obligation_id": "p0-control-plane"},
            )
        finally:
            temporary.cleanup()

    def test_partial_cutover_cohort_is_rejected(self) -> None:
        manifest = _replace_row_state(
            _manifest_text(),
            "p9-submission",
            '{ kind = "deferred", activation_unit = "P9", promotion = { mode = "activate-in-place" } }',
        )
        manifest = _replace_row_state(
            manifest,
            "p6-reinterpret",
            '{ kind = "deferred", activation_unit = "P6", promotion = { mode = "activate-in-place" } }',
        )
        manifest = _replace_row_state(
            manifest,
            "p6-reinterpret-rank-policy",
            '{ kind = "deferred", activation_unit = "P6", promotion = { mode = "activate-in-place" } }',
        )
        manifest = _replace_row_state(
            manifest,
            "p7-cuda",
            '{ kind = "deferred", activation_unit = "P7", promotion = { mode = "activate-in-place" } }',
        )
        files = _fixture_files(manifest)
        result = _run_checker(manifest, files=files, extra=("--diagnostics-json",))
        _assert_error(self, result, "E_COHORT_PARTIAL_PROMOTION", {"cohort_id": "cutover"})

    def test_runner_executes_active_argv_and_emits_small_receipt(self) -> None:
        temporary, root, base, candidate = _git_repository(_manifest_text(), _runner_files())
        try:
            bin_dir = root / "bin"
            bin_dir.mkdir()
            cargo = bin_dir / "cargo"
            cargo.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > cargo-argv\nexit 0\n', encoding="utf-8")
            cargo.chmod(0o755)
            environment = dict(__import__("os").environ)
            environment["PATH"] = str(bin_dir) + ":" + environment.get("PATH", "")
            receipt_path = root / "receipt.json"
            result = _runner(root, base, receipt_path, environment=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(receipt_path.is_file())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(set(receipt), RECEIPT_FIELDS)
            self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
            self.assertEqual(receipt["base_commit"], base)
            self.assertEqual(receipt["candidate_commit"], candidate)
            executions = receipt["executions"]
            self.assertEqual({item["obligation_id"] for item in executions}, ACTIVE_IDS)
            rows = {row["id"]: row for row in _manifest_rows()}
            self.assertEqual(
                [item["obligation_id"] for item in executions],
                sorted(ACTIVE_IDS),
            )
            for execution in executions:
                self.assertEqual(set(execution), EXECUTION_FIELDS)
                row = rows[execution["obligation_id"]]
                self.assertEqual(execution["argv"], row["command"]["argv"])
                self.assertEqual(execution["cwd"], row["command"]["cwd"])
                self.assertEqual(execution["artifact_path"], row["artifact"]["path"])
                self.assertEqual(execution["exit_code"], 0)
            cargo_argv = (root / "cargo-argv").read_text(encoding="utf-8").splitlines()
            last_id = sorted(ACTIVE_IDS)[-1]
            self.assertEqual(cargo_argv, rows[last_id]["command"]["argv"][1:])
            checked = _checker_with_receipt(root, base, receipt_path)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(json.loads(checked.stdout), {"terminal": False})
        finally:
            temporary.cleanup()

    def test_runner_reports_exit_status_and_writes_no_receipt_on_failure(self) -> None:
        temporary, root, base, _ = _git_repository(_manifest_text(), _runner_files())
        try:
            bin_dir = root / "bin"
            bin_dir.mkdir()
            cargo = bin_dir / "cargo"
            cargo.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
            cargo.chmod(0o755)
            environment = dict(__import__("os").environ)
            environment["PATH"] = str(bin_dir) + ":" + environment.get("PATH", "")
            receipt_path = root / "receipt.json"
            result = _runner(root, base, receipt_path, environment=environment, diagnostics=True)
            _assert_error(
                self,
                result,
                "E_COMMAND_FAILED",
                {"command_id": "cmd-control-plane", "exit_code": 17},
            )
            self.assertFalse(receipt_path.exists())
        finally:
            temporary.cleanup()

    def test_base_revision_alias_is_canonicalized_in_receipt(self) -> None:
        temporary, root, base, _ = _git_repository(_manifest_text(), _runner_files())
        try:
            bin_dir = root / "bin"
            bin_dir.mkdir()
            cargo = bin_dir / "cargo"
            cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            cargo.chmod(0o755)
            environment = dict(__import__("os").environ)
            environment["PATH"] = str(bin_dir) + ":" + environment.get("PATH", "")
            receipt_path = root / "receipt.json"
            result = _runner(root, "HEAD~1", receipt_path, environment=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["base_commit"], base)
            self.assertNotEqual(receipt["base_commit"], "HEAD~1")
            checked = _checker_with_receipt(root, "HEAD~1", receipt_path)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
        finally:
            temporary.cleanup()

    def test_receipt_identity_and_full_tracked_tree_cleanliness(self) -> None:
        temporary, root, base, _ = _git_repository(_manifest_text(), _runner_files())
        try:
            bin_dir = root / "bin"
            bin_dir.mkdir()
            cargo = bin_dir / "cargo"
            cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            cargo.chmod(0o755)
            environment = dict(__import__("os").environ)
            environment["PATH"] = str(bin_dir) + ":" + environment.get("PATH", "")
            receipt_path = root / "receipt.json"
            result = _runner(root, base, receipt_path, environment=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            receipt["candidate_commit"] = "not-head"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            checked = _checker_with_receipt(root, base, receipt_path, diagnostics=True)
            _assert_error(self, checked, "E_RECEIPT_COMMIT", {"actual_head": _git(root, "rev-parse", "HEAD")})

            receipt["candidate_commit"] = _git(root, "rev-parse", "HEAD")
            document_execution = next(
                item for item in receipt["executions"]
                if item["obligation_id"] == "p1-contract-document"
            )
            document_execution["argv"] = ["python3", "different.py"]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            checked = _checker_with_receipt(root, base, receipt_path, diagnostics=True)
            _assert_error(
                self,
                checked,
                "E_RECEIPT_EXECUTION_BINDING",
                {
                    "obligation_id": "p1-contract-document",
                    "field": "argv",
                    "expected": ["python3", "scripts/check-storage-design-docs.py"],
                    "actual": ["python3", "different.py"],
                },
            )

            document_execution["argv"] = ["python3", "scripts/check-storage-design-docs.py"]
            (root / "user-note").write_text("unrelated\n", encoding="utf-8")
            (root / "target").mkdir()
            (root / "target" / "unrelated-build-output").write_text("target\n", encoding="utf-8")
            (root / "logs").mkdir()
            (root / "logs" / "runner.log").write_text("log\n", encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            checked = _checker_with_receipt(root, base, receipt_path)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

            design = root / "docs/design/storage-ownership-contracts.md"
            design.write_text(design.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")
            checked = _checker_with_receipt(root, base, receipt_path, diagnostics=True)
            _assert_error(
                self,
                checked,
                "E_RECEIPT_TRACKING",
                {"path": "<tracked-tree>", "status": "modified"},
            )
        finally:
            temporary.cleanup()

    def test_relevant_untracked_artifact_is_rejected(self) -> None:
        temporary, root, base, _ = _git_repository(_manifest_text(), _runner_files())
        try:
            artifact_path = "docs/design/storage-ownership-contracts.md"
            _git(root, "rm", "--cached", "--", artifact_path)
            _git(root, "commit", "-m", "remove tracked artifact from candidate")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            cargo = bin_dir / "cargo"
            cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            cargo.chmod(0o755)
            environment = dict(__import__("os").environ)
            environment["PATH"] = str(bin_dir) + ":" + environment.get("PATH", "")
            receipt_path = root / "receipt.json"
            result = _runner(root, base, receipt_path, environment=environment, diagnostics=True)
            _assert_error(
                self,
                result,
                "E_RECEIPT_TRACKING",
                {"path": artifact_path, "status": "untracked"},
            )
            self.assertFalse(receipt_path.exists())
        finally:
            temporary.cleanup()

    def test_symlink_path_escape_is_rejected_before_execution(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        try:
            files = _fixture_files()
            _write_files(root, files)
            manifest = _replace_once(
                _manifest_text(),
                'path = "docs/design/storage-ownership-contracts.md"',
                'path = "docs/design/design-link.md"',
            )
            manifest_path = root / "scripts/storage-ownership-contracts.toml"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(manifest, encoding="utf-8")
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            outside_file = outside / "design.md"
            outside_file.write_text("outside\n", encoding="utf-8")
            link = root / "docs/design/design-link.md"
            link.symlink_to(outside_file)
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--manifest",
                    "scripts/storage-ownership-contracts.toml",
                    "--diagnostics-json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            _assert_error(
                self,
                result,
                "E_PATH_ESCAPE",
                {"obligation_id": "p1-contract-document"},
            )
        finally:
            outside = root.parent / f"{root.name}-outside"
            if outside.exists():
                for path in outside.iterdir():
                    path.unlink()
                outside.rmdir()
            temporary.cleanup()

    def test_no_v1_authority_or_removed_protocol_remains(self) -> None:
        self.assertFalse((ROOT / "scripts/test-check-storage-ownership-contracts.py").exists())
        runner_source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("shell=True", runner_source)
        self.assertNotIn("os.system", runner_source)

        design = (ROOT / "docs/design/storage-ownership-contracts.md").read_text(
            encoding="utf-8"
        )
        for removed_requirement in (
            "UseLease",
            "single typed provider bridge",
            "generational descriptors",
            "quarantine poisoning",
            "artifact digest",
            "manifest digest",
        ):
            self.assertNotIn(removed_requirement, design)

        manifest = _manifest_text()
        for obligation_id in (
            "p13-freeze",
            "p11-hardware",
            "p12-documentation",
            "p13-closure",
        ):
            start = manifest.index(f'id = "{obligation_id}"')
            end = manifest.find("\n[[obligations]]", start)
            row = manifest[start:] if end < 0 else manifest[start:end]
            self.assertNotIn("scripts/check-storage-design-docs.py", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
