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
import hashlib
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
ACTIVE_OBLIGATIONS = (
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
        "p4-access-retirement",
        "P4",
        ("G1",),
        "artifact-corruption-map",
        "crates/tenferro-tensor/src/storage/tests/corruption_map.rs",
        "corruption-test",
        "cmd-corruption-map",
        "cargo-test",
        ("cargo", "test", "-p", "tenferro-tensor", "--lib", "storage::tests::corruption_map"),
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
    marker: bool,
) -> str:
    if marker:
        argv = ("python3", f"markers/{command_id}.py")
        path_args = (f"markers/{command_id}.py",)
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
        {_command(command_id=command_id, kind=command_kind, argv=argv, path_args=path_args, artifact_id=artifact_id, marker=marker)}
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
    for row in ACTIVE_OBLIGATIONS:
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
    for row in ACTIVE_OBLIGATIONS:
        files[row[4]] = "fn active_contract() {}\n"
    return files


def marker_files() -> dict[str, str]:
    files = repository_files()
    for row in (*ACTIVE_OBLIGATIONS, *DEFERRED_OBLIGATIONS):
        command_id = row[6]
        files[f"markers/{command_id}.py"] = textwrap.dedent(
            f'''\
            from pathlib import Path
            Path("runner.log").open("a", encoding="utf-8").write("{command_id}\\n")
            '''
        )
    return files


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt_promotions(root: Path, manifest: str, obligation_ids: tuple[str, ...]) -> list[dict[str, str]]:
    parsed = tomllib.loads(manifest)
    rows = {row["id"]: row for row in parsed["obligations"]}
    promotions: list[dict[str, str]] = []
    for obligation_id in obligation_ids:
        row = rows[obligation_id]
        command = row["command"]
        command_bytes = json.dumps(command, sort_keys=True, separators=(",", ":")).encode("utf-8")
        promotions.append(
            {
                "obligation_id": obligation_id,
                "artifact_id": row["artifact"]["id"],
                "artifact_sha256": _sha256((root / row["artifact"]["path"]).read_bytes()),
                "command_id": command["id"],
                "command_sha256": _sha256(command_bytes),
                "state": "active",
            }
        )
    return promotions


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
                    str(manifest_path),
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
                str(PRODUCTION_MANIFEST),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_runner(
        self,
        manifest: str,
        *,
        files: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(root, files or marker_files())
            manifest_path = root / "ledger.toml"
            manifest_path.write_text(manifest, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
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
            log_path = root / "runner.log"
            return result, log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    def assert_checker_error(self, manifest: str, needle: str, *, files: dict[str, str] | None = None) -> None:
        result = self.run_checker(manifest, files=files)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)

    def test_checked_in_production_manifest_is_the_gate_input(self) -> None:
        self.assertTrue(PRODUCTION_MANIFEST.is_file())
        production = tomllib.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(production.get("schema"), SCHEMA)
        result = self.run_production_checker()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nominal_v2_manifest_is_green(self) -> None:
        result = self.run_checker(valid_manifest())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_v1_is_rejected_without_compatibility_mode(self) -> None:
        self.assert_checker_error(
            valid_manifest(schema="tenferro.storage-ownership-contracts.v1"),
            "manifest schema must be 'tenferro.storage-ownership-contracts.v2'",
        )

    def test_one_tagged_obligation_table_replaces_parallel_status_tables(self) -> None:
        malformed = valid_manifest().replace(
            'state = { kind = "active" }',
            'status = "active"',
            1,
        )
        self.assert_checker_error(malformed, "obligation 'p1-ledger' must declare tagged state")
        self.assert_checker_error(
            valid_manifest() + '\n[[obligations.active]]\nid = "legacy-active"\n',
            "manifest must use one canonical obligations table",
        )

    def test_canonical_graph_keeps_p0_p1_roots_and_p2_only_depends_on_p1(self) -> None:
        parsed = _parse_registry(valid_manifest())
        self.assertEqual(parsed["edges"].count(("P1", "P2")), 1)
        self.assertNotIn(("P0", "P2"), parsed["edges"])
        self.assertEqual(CUTOVER["prerequisites"], ("P0", "P5"))
        self.assertEqual(CUTOVER["members"], ("P3", "P9"))

        wrong_root = valid_manifest(edges=EDGES + (("P0", "P2"),))
        self.assert_checker_error(wrong_root, "P2 must have exactly one prerequisite: P1")
        wrong_cutover = valid_manifest().replace(
            'prerequisites = ["P0", "P5"]',
            'prerequisites = ["P1", "P5"]',
            1,
        )
        self.assert_checker_error(wrong_cutover, "cutover prerequisites must be exactly P0 and P5")

    def test_graph_rejects_duplicate_and_unknown_target_links(self) -> None:
        duplicate = valid_manifest(edges=EDGES + (("P1", "P2"),))
        self.assert_checker_error(duplicate, "duplicate graph edge P1 -> P2")
        unknown = valid_manifest(edges=EDGES + (("P2", "P99"),))
        self.assert_checker_error(unknown, "graph edge target 'P99' is not a registered unit")

    def test_cutover_is_atomic_and_partial_activation_is_rejected(self) -> None:
        partial = valid_manifest(
            promote=frozenset({"p9-submission"})
        )
        files = repository_files()
        promoted_path = DEFERRED_OBLIGATIONS[2][4]
        files[promoted_path] = "fn promoted_contract() {}\n"
        self.assert_checker_error(
            partial,
            "cohort 'cutover' must activate all members atomically",
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
        self.assert_checker_error(stale, "registry.ownerships is obsolete; obligations refer to unit IDs")

    def test_synthetic_terminal_artifacts_and_terminal_flags_are_rejected(self) -> None:
        synthetic = valid_manifest().replace(
            'kind = "manifest", path = "scripts/storage-ownership-contracts.toml"',
            'kind = "synthetic-terminal", path = ".ledger-terminal"',
            1,
        )
        self.assert_checker_error(synthetic, "synthetic terminal artifacts are not allowed")
        self.assert_checker_error(
            valid_manifest() + '\nterminal = true\n',
            "terminal status is derived and cannot be declared",
        )

    def test_terminal_state_is_derived_from_obligations_and_receipts(self) -> None:
        terminal = valid_manifest()
        self.assertNotIn("terminal", terminal)
        result = self.run_checker(terminal, extra_args=("--json",))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertFalse(summary["terminal"])

    def test_artifact_paths_are_unique_repository_relative_and_real(self) -> None:
        duplicate = valid_manifest().replace(
            'id = "artifact-api-parity", kind = "rust-test", path = "crates/tenferro-tensor/tests/storage_api_parity.rs"',
            'id = "artifact-api-parity", kind = "rust-test", path = "scripts/storage-ownership-contracts.toml"',
            1,
        )
        self.assert_checker_error(duplicate, "duplicate artifact target")
        escape = valid_manifest(active_override={"p1-ledger": "../outside.toml"})
        self.assert_checker_error(escape, "artifact path must remain inside the repository")
        missing = valid_manifest(active_override={"p1-ledger": "scripts/missing.toml"})
        self.assert_checker_error(missing, "active artifact path does not exist")

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
                [sys.executable, str(CHECKER), "--root", str(root), "--manifest", str(manifest_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            outside.unlink()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact path resolves through a symlink outside the repository", result.stderr)

    def test_deferred_artifact_cannot_be_promoted_by_existing_file_alone(self) -> None:
        files = repository_files()
        deferred_path = DEFERRED_OBLIGATIONS[0][4]
        files[deferred_path] = "fn future_contract() {}\n"
        self.assert_checker_error(
            valid_manifest(),
            "deferred artifact exists but has not been promoted in the candidate manifest",
            files=files,
        )

    def test_command_allowlist_is_typed_and_fail_closed(self) -> None:
        shell = valid_manifest().replace(
            'kind = "python-test", argv = ["python3", "scripts/check-storage-ownership-contracts.py"]',
            'kind = "shell", argv = ["sh", "-c", "echo unsafe"]',
            1,
        )
        self.assert_checker_error(shell, "command kind 'shell' is not allow-listed")
        empty = valid_manifest().replace(
            'argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_api_parity"]',
            "argv = []",
            1,
        )
        self.assert_checker_error(empty, "command argv must be a non-empty array")
        path_escape = valid_manifest().replace(
            'path_args = ["scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"]',
            'path_args = ["../outside.toml"]',
            1,
        )
        self.assert_checker_error(path_escape, "command path argument must remain inside the repository")

    def test_command_must_bind_to_exact_artifact_and_target_links(self) -> None:
        wrong_id = valid_manifest().replace(
            'artifact_id = "artifact-ledger"',
            'artifact_id = "artifact-contract-document"',
            1,
        )
        self.assert_checker_error(wrong_id, "command 'cmd-ledger' is not bound to its obligation artifact")
        wrong_target = valid_manifest().replace(
            'path_args = ["scripts/check-storage-ownership-contracts.py", "scripts/storage-ownership-contracts.toml"]',
            'path_args = ["scripts/check-storage-design-docs.py"]',
            1,
        )
        self.assert_checker_error(wrong_target, "command target links do not match artifact binding")
        duplicate_command = valid_manifest().replace(
            'id = "cmd-api-parity"',
            'id = "cmd-ledger"',
            1,
        )
        self.assert_checker_error(duplicate_command, "duplicate command identity has different artifact binding")

    def test_source_and_fixture_tables_cannot_reappear_as_parallel_authority(self) -> None:
        for table in ("fixtures", "fixture_suites", "source_scans", "source_inventory", "ownerships"):
            with self.subTest(table=table):
                self.assert_checker_error(
                    valid_manifest() + f'\n[[{table}]]\nid = "legacy"\n',
                    f"manifest table '{table}' is not part of v2",
                )

    def test_promotion_preserves_immutable_identity_and_binds_receipt_to_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(root, repository_files())
            base_path = root / "base.toml"
            candidate_path = root / "candidate.toml"
            receipt_path = root / "receipt.json"
            base = valid_manifest()
            candidate = valid_manifest(
                promote=frozenset({"p3-host-owner", "p9-submission"})
            )
            for row in DEFERRED_OBLIGATIONS:
                if row[0] in {"p3-host-owner", "p9-submission"}:
                    path = root / row[4]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("fn promoted_contract() {}\n", encoding="utf-8")
            base_path.write_text(base, encoding="utf-8")
            candidate_path.write_text(candidate, encoding="utf-8")
            promotions = _receipt_promotions(
                root, candidate, ("p3-host-owner", "p9-submission")
            )
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema": "tenferro.storage-ownership-receipt.v1",
                        "base_commit": "base-sha",
                        "candidate_commit": "candidate-sha",
                        "base_manifest_sha256": _sha256(base.encode("utf-8")),
                        "candidate_manifest_sha256": _sha256(candidate.encode("utf-8")),
                        "promotions": promotions,
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--base-manifest",
                    str(base_path),
                    "--manifest",
                    str(candidate_path),
                    "--base-commit",
                    "base-sha",
                    "--candidate-commit",
                    "candidate-sha",
                    "--receipt",
                    str(receipt_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_promotion_rejects_artifact_or_command_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_files(root, repository_files())
            base_path = root / "base.toml"
            candidate_path = root / "candidate.toml"
            base_path.write_text(valid_manifest(), encoding="utf-8")
            candidate_path.write_text(
                valid_manifest(
                    deferred_override={"p3-host-owner": "crates/tenferro-tensor/tests/ui/storage/fail/changed.rs"}
                ),
                encoding="utf-8",
            )
            changed = root / "crates/tenferro-tensor/tests/ui/storage/fail/changed.rs"
            changed.parent.mkdir(parents=True, exist_ok=True)
            changed.write_text("fn changed_contract() {}\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--root",
                    str(root),
                    "--base-manifest",
                    str(base_path),
                    "--manifest",
                    str(candidate_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("promotion changes immutable artifact or command identity", result.stderr)

    def test_runner_executes_each_active_command_once_and_never_deferred(self) -> None:
        result, log = self.run_runner(valid_manifest(marker=True))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        entries = log.splitlines()
        self.assertEqual(entries.count("cmd-ledger"), 1)
        self.assertEqual(entries.count("cmd-contract-document"), 1)
        self.assertEqual(entries.count("cmd-api-parity"), 1)
        self.assertNotIn("cmd-owner-compile", entries)
        self.assertNotIn("cmd-corruption-map", entries)

    def test_runner_is_fail_closed_on_command_failure(self) -> None:
        files = marker_files()
        files["markers/cmd-ledger.py"] = 'raise SystemExit(23)\n'
        result, log = self.run_runner(valid_manifest(marker=True), files=files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cmd-ledger", result.stderr)
        self.assertNotIn("cmd-owner-compile", log)

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
