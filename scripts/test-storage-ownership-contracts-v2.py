#!/usr/bin/env python3
"""RED tests for the fail-closed v2 storage ownership contract ledger.

These tests intentionally describe the replacement schema before the checker
implementation exists.  They are kept separate from the superseded v1 tests
so the first RED run is unambiguous; the v1 test module is removed when v2 is
implemented.
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
INVENTORY_TOOL_MANIFEST = ROOT / "tools" / "storage-contract-inventory" / "Cargo.toml"

GATES = tuple(f"G{index}" for index in range(1, 8))

PHASES = (
    ("P0", 1556),
    ("P1", 1557),
    ("P2", 1558),
    ("P3", 1559),
    ("P4", 1560),
    ("P5", 1561),
    ("P6", 1562),
    ("P7", 1563),
    ("P8", 1564),
    ("P9", 1565),
    ("P10", 1566),
    ("P11", 1568),
    ("P12", 1569),
    ("P13", 1567),
)

# This is deliberately not one row per gate.  Gates are cross-cutting, while
# ownership records are the stable IDs consumed by fixture/source/matrix
# obligations.  The checker must validate this exact canonical registry and
# derive the phase issue from PHASES rather than trusting duplicated row data.
OWNERSHIPS = (
    ("p0-repository-contract", "G5", "P0"),
    ("p1-ledger-schema", "G1", "P1"),
    ("p1-ad-retention", "G7", "P1"),
    ("p1-source-ledger", "G5", "P1"),
    ("p1-api-baseline", "G4", "P1"),
    ("p1-contract-documentation", "G6", "P1"),
    ("p1-documentation-index", "G6", "P1"),
    ("p2-allocation-core", "G1", "P2"),
    ("p2-group-skeleton", "G2", "P2"),
    ("p2-provider-identity", "G5", "P2"),
    ("p3-host-ownership", "G4", "P3"),
    ("p3-ad-retention", "G7", "P3"),
    ("p4-access-retirement", "G1", "P4"),
    ("p4-provider-reclaim", "G5", "P4"),
    ("p4-ad-runtime", "G7", "P4"),
    ("p5-allocation-group", "G2", "P5"),
    ("p5-capability-distribution", "G4", "P5"),
    ("p6-reinterpret", "G2", "P6"),
    ("p6-provider-identity", "G5", "P6"),
    ("p7-cuda-storage", "G5", "P7"),
    ("p7-provider-ad", "G7", "P7"),
    ("p8-webgpu-metal", "G5", "P8"),
    ("p8-provider-ad", "G7", "P8"),
    ("p9-submission", "G3", "P9"),
    ("p9-ad-retention", "G7", "P9"),
    ("p10-api-normalization", "G5", "P10"),
    ("p10-runtime-api", "G3", "P10"),
    ("p11-hardware", "G5", "P11"),
    ("p11-ad-provider", "G7", "P11"),
    ("p12-documentation", "G6", "P12"),
    ("p12-ad-docs", "G7", "P12"),
    ("p13-closure", "G6", "P13"),
    ("p13-closure-ownership", "G7", "P13"),
)

CURRENT_ACTIVE_OWNERSHIPS = frozenset(
    {
        "p1-ledger-schema",
        "p1-source-ledger",
        "p1-api-baseline",
        "p1-ad-retention",
        "p1-contract-documentation",
        "p1-documentation-index",
    }
)


def _registry() -> str:
    gates = "\n".join(
        f'[[registry.gates]]\nid = "{gate}"\ntitle = "{gate} cross-cutting gate"'
        for gate in GATES
    )
    phases = "\n".join(
        f'[[registry.phases]]\nid = "{phase}"\nissue = {issue}'
        for phase, issue in PHASES
    )
    ownerships = "\n".join(
        f'[[registry.ownerships]]\nid = "{name}"\ngate = "{gate}"\nphase = "{phase}"'
        for name, gate, phase in OWNERSHIPS
    )
    return "\n".join(("[registry]", gates, phases, ownerships))


def _commands(
    *,
    command_argv: str = 'argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"]',
) -> str:
    return textwrap.dedent(
        f'''\
        [[commands]]
        id = "compile-contract"
        policy = "cargo-test"
        {command_argv}
        cwd = "."
        path_args = []

        [[commands]]
        id = "source-inventory"
        policy = "cargo-run-tool"
        argv = ["cargo", "run", "--manifest-path", "tools/storage-contract-inventory/Cargo.toml", "--", "--root", ".", "--manifest", "scripts/storage-ownership-contracts.toml"]
        cwd = "."
        path_args = ["tools/storage-contract-inventory/Cargo.toml", "scripts/storage-ownership-contracts.toml"]

        [[commands]]
        id = "matrix-parity"
        policy = "cargo-test"
        argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_api_parity"]
        cwd = "."
        path_args = []

        [[commands]]
        id = "future-property"
        policy = "cargo-test"
        argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "future_storage_contract"]
        cwd = "."
        path_args = []

        [[commands]]
        id = "design-docs"
        policy = "doc-snippets"
        argv = ["python3", "scripts/check-storage-design-docs.py"]
        cwd = "."
        path_args = ["scripts/check-storage-design-docs.py"]
        '''
    )


def _fixture_path(index: int, ownership_id: str) -> str:
    return f"fixtures/{index}-{ownership_id}.rs"


def _trybuild_payload(index: int, ownership_id: str) -> str:
    path = _fixture_path(index, ownership_id)
    return (
        "payload = { kind = \"trybuild-suite\", root = \"fixtures\", "
        f"glob = \"{index}-{ownership_id}.rs\", fixture_ids = [\"fixture-{ownership_id}\"], "
        f"fixture_paths = [\"{path}\"] }}"
    )


def _source_inventory_payload() -> str:
    return textwrap.dedent(
        '''\
        payload = { kind = "source-inventory", root = "src", glob = "legacy.rs", records = [
          { id = "source-raw-handle", category = "raw-handle-extraction", evidence_class = "lexical-drift", selector = { kind = "lexical-drift", value = "RAW_HANDLE" }, remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" } },
          { id = "source-old-function", category = "temporary-migration-adapter", evidence_class = "rust-structural", selector = { kind = "rust-item", item_kind = "function", name = "old" }, remediation = { kind = "narrow", owner_ownership_id = "p3-host-ownership", target_ownership_id = "p3-ad-retention", verification_command_id = "future-property" } }
        ] }
        '''
    ).strip()


def _matrix_payload() -> str:
    return textwrap.dedent(
        '''\
        payload = { kind = "api-matrix", path = "contracts/storage_api_parity.rs", convergence_target_ownership_id = "p5-capability-distribution", faces = ["typed_owner", "typed_view", "typed_view_mut", "erased_owner", "erased_view", "erased_view_mut"], families = ["layout_metadata", "dtype", "placement_device", "host_read"], applicability = { typed_owner = { layout_metadata = "required", dtype = "not-applicable", placement_device = "required", host_read = "required" }, typed_view = { layout_metadata = "required", dtype = "not-applicable", placement_device = "required", host_read = "required" }, typed_view_mut = { layout_metadata = "required", dtype = "not-applicable", placement_device = "required", host_read = "required" }, erased_owner = { layout_metadata = "required", dtype = "required", placement_device = "required", host_read = "required" }, erased_view = { layout_metadata = "required", dtype = "required", placement_device = "required", host_read = "required" }, erased_view_mut = { layout_metadata = "required", dtype = "required", placement_device = "required", host_read = "required" } } }
        '''
    ).strip()


def _documentation_payload(path: str) -> str:
    if path.endswith("index.md"):
        snippet_ids = ["storage-design-index"]
    else:
        snippet_ids = ["storage-ownership-contracts-gates", "storage-ownership-contracts-transitions"]
    return (
        f'payload = {{ kind = "doc-snippets", path = "{path}", '
        f'snippet_ids = {json.dumps(snippet_ids)} }}'
    )


def _active_obligations(*, terminal: bool = False) -> str:
    rows = []
    for index, (ownership_id, _gate, _phase) in enumerate(OWNERSHIPS):
        if not terminal and ownership_id not in CURRENT_ACTIVE_OWNERSHIPS:
            continue
        if ownership_id == "p1-source-ledger":
            evidence = 'evidence = { kind = "source-inventory" }'
            command_id = "source-inventory"
            payload = _source_inventory_payload()
            rationale = "Active source inventory records authority-risk seams as drift evidence, not semantic safety proof."
        elif ownership_id == "p1-api-baseline":
            evidence = 'evidence = { kind = "semantic" }'
            command_id = "matrix-parity"
            payload = _matrix_payload()
            rationale = "The API matrix is a direct semantic compile contract for the currently applicable surfaces."
        elif ownership_id == "p1-ad-retention":
            evidence = 'evidence = { kind = "documentation" }'
            command_id = "design-docs"
            payload = _documentation_payload("docs/design/storage-ownership-contracts.md")
            rationale = "The merged Phase 1 AD retention contract is verified by reproducible documentation snippets."
        elif ownership_id == "p1-contract-documentation":
            evidence = 'evidence = { kind = "documentation" }'
            command_id = "design-docs"
            payload = _documentation_payload("docs/design/storage-ownership-contracts.md")
            rationale = "The canonical design document is an active Phase 1 deliverable verified by its snippets."
        elif ownership_id == "p1-documentation-index":
            evidence = 'evidence = { kind = "documentation" }'
            command_id = "design-docs"
            payload = _documentation_payload("docs/design/index.md")
            rationale = "The design index is an active Phase 1 documentation deliverable verified by its snippets."
        else:
            evidence = 'evidence = { kind = "semantic" }'
            command_id = "compile-contract"
            payload = _trybuild_payload(index, ownership_id)
            rationale = "The exact active trybuild artifact is the decisive semantic obligation for this realized ownership."
        rows.append(
            textwrap.dedent(
                f'''\
                [[obligations.active]]
                id = "active-{ownership_id}"
                ownership_id = "{ownership_id}"
                command_id = "{command_id}"
                {evidence}
                {payload}
                rationale = "{rationale}"
                '''
            )
        )
    return "\n".join(rows)


def _deferred_obligations(*, terminal: bool = False) -> str:
    if terminal:
        return ""
    rows = []
    for index, (ownership_id, _gate, phase) in enumerate(OWNERSHIPS):
        if ownership_id in CURRENT_ACTIVE_OWNERSHIPS:
            continue
        rows.append(
            textwrap.dedent(
                f'''\
                [[obligations.deferred]]
                id = "deferred-{ownership_id}"
                ownership_id = "{ownership_id}"
                owner_phase = "{phase}"
                command_id = "future-property"
                payload = {{ kind = "future-artifact", artifacts = ["future/{index}-{ownership_id}.rs"] }}
                promotion = {{ artifact = "future/{index}-{ownership_id}.rs", condition = "{phase} promotes the artifact and executes the registered future command in the same commit." }}
                rationale = "This future obligation is planned coverage, not current semantic proof."
                '''
            )
        )
    return "\n".join(rows)


def valid_manifest(
    *,
    include: set[str] | None = None,
    schema: str = "tenferro.storage-ownership-contracts.v2",
    terminal: bool = False,
    commands: str | None = None,
) -> str:
    sections = {
        "registry": _registry(),
        "commands": _commands() if commands is None else commands,
        "obligations": "\n".join(
            (
                "[obligations]",
                "deferred = []" if terminal else "",
                _active_obligations(terminal=terminal),
                _deferred_obligations(terminal=terminal),
            )
        ),
    }
    selected = set(sections) if include is None else include
    return "\n\n".join(
        part for name, part in (("schema", f'schema = "{schema}"'), *sections.items()) if name == "schema" or name in selected
    ) + "\n"


def repository_files() -> dict[str, str]:
    files = {
        "src/legacy.rs": "pub fn old() { RAW_HANDLE; }\n",
    }
    for index, (ownership_id, _gate, _phase) in enumerate(OWNERSHIPS):
        files[f"fixtures/{index}-{ownership_id}.rs"] = "fn compile_fixture() {}\n"
    files["contracts/storage_api_parity.rs"] = "fn parity_fixture() {}\n"
    files["docs/design/storage-ownership-contracts.md"] = "# Storage ownership contracts\n\n<!-- snippet: storage-ownership-contracts-gates -->\n- G1\n- G7\n<!-- end snippet -->\n\n<!-- snippet: storage-ownership-contracts-transitions -->\n```rust\nfn documented_contract() {}\n```\n<!-- end snippet -->\n"
    files["docs/design/index.md"] = "# Design index\n\n<!-- snippet: storage-design-index -->\n- storage ownership contracts\n<!-- end snippet -->\n"
    files["scripts/check-storage-design-docs.py"] = "raise SystemExit(0)\n"
    files["tools/storage-contract-inventory/Cargo.toml"] = "[package]\nname = \"storage-contract-inventory\"\nversion = \"0.1.0\"\n"
    files["scripts/storage-ownership-contracts.toml"] = "schema = \"tenferro.storage-ownership-contracts.v2\"\n"
    return files


def marker_commands() -> str:
    rows = []
    for command_id in ("compile-contract", "source-inventory", "matrix-parity", "future-property", "design-docs"):
        rows.append(
            textwrap.dedent(
                f'''\
                [[commands]]
                id = "{command_id}"
                policy = "python-script"
                argv = ["python3", "scripts/markers/{command_id}.py"]
                cwd = "."
                path_args = ["scripts/markers/{command_id}.py"]
                '''
            )
        )
    return "\n".join(rows)


def marker_files(*, failing: str | None = None) -> dict[str, str]:
    files = repository_files()
    for command_id in ("compile-contract", "source-inventory", "matrix-parity", "future-property", "design-docs"):
        exit_clause = "raise SystemExit(23)" if command_id == failing else ""
        files[f"scripts/markers/{command_id}.py"] = textwrap.dedent(
            f'''\
            from pathlib import Path

            Path("runner.log").open("a", encoding="utf-8").write("{command_id}\\n")
            {exit_clause}
            '''
        )
    return files


class CheckerV2Tests(unittest.TestCase):
    def run_checker(
        self,
        manifest: str,
        *,
        files: dict[str, str] | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, contents in (files or repository_files()).items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
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

    def run_runner(
        self,
        manifest: str,
        *,
        files: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, contents in (files or marker_files()).items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
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
            log = (root / "runner.log").read_text(encoding="utf-8") if (root / "runner.log").exists() else ""
            return result, log

    def run_inventory(
        self,
        manifest: str,
        *,
        files: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, contents in (files or repository_files()).items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            manifest_path = root / "ledger.toml"
            manifest_path.write_text(manifest, encoding="utf-8")
            return subprocess.run(
                [
                    "cargo",
                    "run",
                    "--manifest-path",
                    str(INVENTORY_TOOL_MANIFEST),
                    "--",
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def assert_error(self, manifest: str, needle: str) -> None:
        result = self.run_checker(manifest)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)

    def test_nominal_v2_manifest_is_green(self) -> None:
        result = self.run_checker(valid_manifest())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_v1_manifest_is_not_parsed_as_a_compatibility_mode(self) -> None:
        self.assert_error(
            valid_manifest(schema="tenferro.storage-ownership-contracts.v1"),
            "manifest schema must be 'tenferro.storage-ownership-contracts.v2'",
        )

    def test_schema_only_manifest_is_rejected(self) -> None:
        self.assert_error(
            'schema = "tenferro.storage-ownership-contracts.v2"\n',
            "manifest section 'registry' must be present and non-empty",
        )

    def test_every_required_section_is_non_empty(self) -> None:
        for section in (
            *valid_manifest_sections(),
        ):
            with self.subTest(section=section):
                self.assert_error(
                    valid_manifest(include=set(valid_manifest_sections()) - {section}),
                    f"manifest section '{section}' must be present and non-empty",
                )

    def test_multiple_ownerships_per_gate_are_valid(self) -> None:
        result = self.run_checker(valid_manifest())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(sum(gate == "G5" for _id, gate, _phase in OWNERSHIPS), 2)

    def test_arbitrary_gate_phase_pairing_is_rejected(self) -> None:
        manifest = valid_manifest().replace(
            'id = "p2-group-skeleton"\ngate = "G2"\nphase = "P2"',
            'id = "p2-group-skeleton"\ngate = "G1"\nphase = "P2"',
        )
        self.assert_error(manifest, "ownership 'p2-group-skeleton' has non-canonical gate/phase pairing")

    def test_unknown_contract_relationship_is_rejected(self) -> None:
        self.assert_error(
            valid_manifest().replace(
                'id = "active-p1-ledger-schema"\nownership_id = "p1-ledger-schema"',
                'id = "active-p1-ledger-schema"\nownership_id = "not-registered"',
                1,
            ),
            "obligation 'active-p1-ledger-schema' references unknown ownership 'not-registered'",
        )

    def test_duplicate_registry_ids_are_rejected(self) -> None:
        self.assert_error(
            valid_manifest() + '\n[[registry.gates]]\nid = "G1"\ntitle = "duplicate"\n',
            "duplicate registry gate id 'G1'",
        )

    def test_registry_requires_every_canonical_phase_and_exact_issue_mapping(self) -> None:
        missing = valid_manifest().replace(
            '[[registry.phases]]\nid = "P13"\nissue = 1567\n',
            "",
        )
        self.assert_error(missing, "registry phases must contain exactly canonical phase IDs")

        wrong_issue = valid_manifest().replace(
            '[[registry.phases]]\nid = "P2"\nissue = 1558',
            '[[registry.phases]]\nid = "P2"\nissue = 1557',
        )
        self.assert_error(wrong_issue, "phase 'P2' must map to issue #1558")

        unknown_phase = valid_manifest().replace(
            '[[registry.phases]]\nid = "P13"\nissue = 1567',
            '[[registry.phases]]\nid = "P14"\nissue = 1570',
        )
        self.assert_error(unknown_phase, "unknown canonical phase 'P14'")

        duplicate_phase = valid_manifest() + '\n[[registry.phases]]\nid = "P2"\nissue = 1558\n'
        self.assert_error(duplicate_phase, "duplicate registry phase id 'P2'")

    def test_registry_rejects_unknown_gate_and_duplicate_ownership_identity(self) -> None:
        unknown_gate = valid_manifest().replace(
            'id = "p2-group-skeleton"\ngate = "G2"\nphase = "P2"',
            'id = "p2-group-skeleton"\ngate = "G9"\nphase = "P2"',
        )
        self.assert_error(unknown_gate, "ownership 'p2-group-skeleton' references unknown gate 'G9'")

        duplicate_id = valid_manifest() + '\n[[registry.ownerships]]\nid = "p2-group-skeleton"\ngate = "G2"\nphase = "P2"\n'
        self.assert_error(duplicate_id, "duplicate registry ownership id 'p2-group-skeleton'")

    def test_distinct_stable_ownerships_may_share_gate_and_phase(self) -> None:
        result = self.run_checker(valid_manifest())
        self.assertEqual(result.returncode, 0, result.stderr)
        same_tuple = [
            ownership_id
            for ownership_id, gate, phase in OWNERSHIPS
            if gate == "G6" and phase == "P1"
        ]
        self.assertEqual(set(same_tuple), {"p1-contract-documentation", "p1-documentation-index"})

    def test_registry_requires_coverage_for_all_gates_and_phases(self) -> None:
        without_gate = valid_manifest().replace(
            'id = "p12-documentation"\ngate = "G6"\nphase = "P12"',
            "",
        ).replace(
            'id = "p13-closure"\ngate = "G6"\nphase = "P13"',
            "",
        )
        self.assert_error(without_gate, "registry must retain canonical gate coverage for 'G6'")

        without_phase = valid_manifest().replace(
            'id = "p13-closure"\ngate = "G6"\nphase = "P13"',
            "",
        ).replace(
            'id = "p13-closure-ownership"\ngate = "G7"\nphase = "P13"',
            "",
        )
        self.assert_error(without_phase, "registry must retain canonical phase coverage for 'P13'")

    def test_active_semantic_proof_cannot_be_claimed_by_inventory_command(self) -> None:
        manifest = valid_manifest().replace(
            'id = "active-p1-ledger-schema"\nownership_id = "p1-ledger-schema"\ncommand_id = "compile-contract"',
            'id = "active-p1-ledger-schema"\nownership_id = "p1-ledger-schema"\ncommand_id = "source-inventory"',
        )
        self.assert_error(
            manifest,
            "semantic obligation 'active-p1-ledger-schema' command 'source-inventory' cannot verify payload kind 'trybuild-suite'",
        )

    def test_phase1_baselines_do_not_activate_future_implementation_ownerships(self) -> None:
        parsed = tomllib.loads(valid_manifest())
        active_ids = {row["ownership_id"] for row in parsed["obligations"]["active"]}
        deferred_ids = {row["ownership_id"] for row in parsed["obligations"]["deferred"]}
        self.assertIn("p1-source-ledger", active_ids)
        self.assertIn("p1-api-baseline", active_ids)
        self.assertIn("p3-host-ownership", deferred_ids)
        self.assertIn("p5-capability-distribution", deferred_ids)
        self.assertNotIn("p3-host-ownership", active_ids)
        self.assertNotIn("p5-capability-distribution", active_ids)

        result = self.run_checker(valid_manifest(), extra_args=("--json",))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertNotIn("p3-host-ownership", summary["coverage"]["active"]["ownerships"])
        self.assertNotIn("p5-capability-distribution", summary["coverage"]["active"]["ownerships"])
        self.assertIn("p3-host-ownership", summary["coverage"]["deferred"]["ownerships"])
        self.assertIn("p5-capability-distribution", summary["coverage"]["deferred"]["ownerships"])

    def test_each_ownership_has_exactly_one_obligation_variant(self) -> None:
        duplicate_active = valid_manifest() + textwrap.dedent(
            '''\
            [[obligations.active]]
            id = "active-duplicate-p1"
            ownership_id = "p1-ledger-schema"
            command_id = "compile-contract"
            evidence = { kind = "semantic" }
            payload = { kind = "trybuild-suite", root = "fixtures", glob = "1-p1-ledger-schema.rs", fixture_ids = ["fixture-duplicate"], fixture_paths = ["fixtures/1-p1-ledger-schema.rs"] }
            rationale = "duplicate ownership test"
            '''
        )
        self.assert_error(
            duplicate_active,
            "ownership 'p1-ledger-schema' must be covered by exactly one obligation variant",
        )

        active_and_deferred = valid_manifest().replace(
            'ownership_id = "p0-repository-contract"',
            'ownership_id = "p1-ledger-schema"',
            1,
        )
        self.assert_error(
            active_and_deferred,
            "ownership 'p1-ledger-schema' must be covered by exactly one obligation variant",
        )

        deferred_evidence = valid_manifest().replace(
            'command_id = "future-property"\npayload = { kind = "future-artifact"',
            'command_id = "future-property"\nevidence = { kind = "semantic" }\npayload = { kind = "future-artifact"',
            1,
        )
        self.assert_error(
            deferred_evidence,
            "deferred obligation 'deferred-p0-repository-contract' has unknown field 'evidence'",
        )

    def test_source_scans_and_inventory_are_removed_as_duplicate_models(self) -> None:
        for table, expected in (
            ("source_scans", "source_scans"),
            ("source_inventory", "source_inventory"),
            ("source_contracts", "source_contracts"),
            ("matrix_contracts", "matrix_contracts"),
            ("semantic_obligations", "semantic_obligations"),
            ("fixture_suites", "fixture_suites"),
            ("fixtures", "fixtures"),
        ):
            with self.subTest(table=table):
                self.assert_error(
                    valid_manifest() + f"\n[[{table}]]\nid = \"legacy\"\n",
                    f"manifest has unknown top-level field '{expected}'",
                )

    def test_lifecycle_does_not_accept_status_or_parallel_path_fields(self) -> None:
        self.assert_error(
            valid_manifest() + '\nstatus = "deferred"\n',
            "manifest has unknown top-level field 'status'",
        )
        self.assert_error(
            valid_manifest() + '\n[[obligations.active]]\nid = "bad"\nownership_id = "p1-ledger-schema"\ncommand_id = "compile-contract"\nstatus = "active"\npayload = { kind = "trybuild-suite", root = "fixtures", glob = "1-p1-ledger-schema.rs", fixture_ids = ["fixture-p1-ledger-schema"], fixture_paths = ["fixtures/1-p1-ledger-schema.rs"] }\nevidence = { kind = "semantic" }\nrationale = "bad"\n',
            "active obligation 'bad' has unknown field 'status'",
        )
        self.assert_error(
            valid_manifest() + '\n[[fixtures.active]]\nid = "bad"\n',
            "manifest has unknown top-level field 'fixtures'",
        )

    def test_terminal_manifest_has_structural_empty_deferred_collection(self) -> None:
        parsed = tomllib.loads(valid_manifest(terminal=True))
        self.assertEqual(parsed["obligations"]["deferred"], [])
        self.assertTrue(parsed["obligations"]["active"])
        self.assertNotIn("deferred", parsed["obligations"]["active"][-1])
        self.assertNotIn("fixtures", parsed)
        self.assertNotIn("semantic_obligations", parsed)
        result = self.run_checker(valid_manifest(terminal=True))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_deferred_obligation_requires_distinct_promotion_record(self) -> None:
        manifest = valid_manifest().replace(
            'promotion = { artifact = "future/0-p0-repository-contract.rs", condition = "P0 promotes the artifact and executes the registered future command in the same commit." }',
            'promotion = { artifact = "future/other.rs", condition = "future" }',
        )
        self.assert_error(manifest, "deferred obligation 'deferred-p0-repository-contract' promotion artifact must equal payload artifact")

    def test_obligation_artifacts_are_unique_and_repository_contained(self) -> None:
        duplicate_active = valid_manifest(terminal=True).replace(
            'fixture_paths = ["fixtures/1-p1-ledger-schema.rs"]',
            'fixture_paths = ["fixtures/0-p0-repository-contract.rs"]',
            1,
        )
        self.assert_error(duplicate_active, "duplicate active artifact 'fixtures/0-p0-repository-contract.rs'")

        duplicate_deferred = valid_manifest() + textwrap.dedent(
            '''\
            [[obligations.deferred]]
            id = "deferred-duplicate-artifact"
            ownership_id = "p12-documentation"
            owner_phase = "P12"
            command_id = "future-property"
            payload = { kind = "future-artifact", artifacts = ["future/0-p0-repository-contract.rs"] }
            promotion = { artifact = "future/0-p0-repository-contract.rs", condition = "duplicate test" }
            rationale = "duplicate artifact test"
            '''
        )
        self.assert_error(duplicate_deferred, "duplicate deferred artifact 'future/0-p0-repository-contract.rs'")

        active_future_collision = valid_manifest().replace(
            'artifacts = ["future/0-p0-repository-contract.rs"]',
            'artifacts = ["fixtures/1-p1-ledger-schema.rs"]',
            1,
        )
        self.assert_error(
            active_future_collision,
            "deferred obligation 'deferred-p0-repository-contract' artifact collides with active artifact",
        )

        self.assert_error(
            valid_manifest().replace(
                'fixture_paths = ["fixtures/1-p1-ledger-schema.rs"]',
                'fixture_paths = ["../outside.rs"]',
                1,
            ),
            "active obligation 'active-p1-ledger-schema' artifact must remain inside the repository",
        )
        self.assert_error(
            valid_manifest().replace(
                'artifacts = ["future/0-p0-repository-contract.rs"]',
                'artifacts = ["../outside.rs"]',
                1,
            ),
            "deferred obligation 'deferred-p0-repository-contract' artifact must remain inside the repository",
        )

    def test_trybuild_discovery_has_exact_suite_ownership(self) -> None:
        parsed = tomllib.loads(valid_manifest(terminal=True))
        trybuild = [
            row for row in parsed["obligations"]["active"] if row["payload"]["kind"] == "trybuild-suite"
        ]
        self.assertTrue(trybuild)
        for row in trybuild:
            payload = row["payload"]
            self.assertEqual(len(payload["fixture_ids"]), len(payload["fixture_paths"]))
            self.assertEqual(len(payload["fixture_ids"]), 1)

        duplicate_fixture_id = valid_manifest(terminal=True).replace(
            'fixture_ids = ["fixture-p1-ledger-schema"]',
            'fixture_ids = ["fixture-p0-repository-contract"]',
            1,
        )
        self.assert_error(duplicate_fixture_id, "fixture ID 'fixture-p0-repository-contract' has duplicate active ownership")

        empty_suite = valid_manifest().replace(
            'glob = "1-p1-ledger-schema.rs"',
            'glob = "missing/*.rs"',
            1,
        )
        self.assert_error(empty_suite, "trybuild suite 'active-p1-ledger-schema' discovers no fixtures")

        orphan_files = repository_files()
        orphan_files["fixtures/orphan.rs"] = "fn orphan() {}\n"
        orphan_manifest = valid_manifest().replace(
            'glob = "1-p1-ledger-schema.rs"',
            'glob = "**/*.rs"',
            1,
        )
        result = self.run_checker(orphan_manifest, files=orphan_files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("discovered fixture 'fixtures/orphan.rs' has no active ownership", result.stderr)

        deferred_files = repository_files()
        deferred_files["fixtures/future/p0-repository-contract.rs"] = "fn future_fixture() {}\n"
        deferred_manifest = valid_manifest().replace(
            'artifacts = ["future/0-p0-repository-contract.rs"]',
            'artifacts = ["fixtures/future/p0-repository-contract.rs"]',
            1,
        ).replace(
            'glob = "1-p1-ledger-schema.rs"',
            'glob = "**/*.rs"',
            1,
        )
        result = self.run_checker(deferred_manifest, files=deferred_files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deferred artifact enters active trybuild discovery", result.stderr)

    def test_symlink_escape_is_rejected_for_source_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = repository_files()
            for relative, contents in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            outside = root.parent / f"storage-ledger-outside-{root.name}.rs"
            outside.write_text("fn outside() {}\n", encoding="utf-8")
            link = root / "src" / "escape.rs"
            link.symlink_to(outside)
            manifest = valid_manifest().replace(
                'scope = { root = "src", glob = "legacy.rs" }',
                'scope = { root = "src", glob = "escape.rs" }',
                1,
            )
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
        self.assertIn("source scope resolves outside the repository", result.stderr)

    def test_existing_deferred_artifact_cannot_bypass_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = repository_files()
            files["future/0-p0-repository-contract.rs"] = "fn future_fixture() {}\n"
            for relative, contents in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            manifest_path = root / "ledger.toml"
            manifest_path.write_text(valid_manifest(), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(CHECKER), "--root", str(root), "--manifest", str(manifest_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deferred obligation artifact", result.stderr)

    def test_command_strings_are_not_accepted(self) -> None:
        self.assert_error(
            valid_manifest() + '\ncommand = "cargo test"\n',
            "manifest has unknown top-level field 'command'",
        )

    def test_command_registry_requires_structured_argv(self) -> None:
        self.assert_error(
            valid_manifest().replace(
                'argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"]',
                'argv = "cargo test -p tenferro-tensor"',
                1,
            ),
            "command 'compile-contract' argv must be a non-empty array of strings",
        )

    def test_command_registry_rejects_duplicate_unknown_and_empty_shapes(self) -> None:
        duplicate = valid_manifest() + '\n[[commands]]\nid = "compile-contract"\npolicy = "cargo-test"\nargv = ["cargo", "test"]\ncwd = "."\npath_args = []\n'
        self.assert_error(duplicate, "duplicate command id 'compile-contract'")

        self.assert_error(
            valid_manifest().replace('policy = "cargo-test"', 'policy = "not-a-command-policy"', 1),
            "command 'compile-contract' has unknown policy 'not-a-command-policy'",
        )

        self.assert_error(
            valid_manifest().replace(
                'argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"]',
                'argv = []',
                1,
            ),
            "command 'compile-contract' argv must be a non-empty array of strings",
        )
        self.assert_error(
            valid_manifest().replace('path_args = []', 'path_args = [""]', 1),
            "command 'compile-contract' path_args must not contain empty arguments",
        )

    def test_command_policy_rejects_shell_and_path_escape(self) -> None:
        for argv, expected in (
            ('argv = ["sh", "-c", "echo unsafe"]', "command 'compile-contract' executable 'sh' is not allowed"),
            ('argv = ["cargo", "test", "--manifest-path", "../Cargo.toml"]', "repository-relative"),
            ('argv = ["python3", "-c", "print(1)"]', "command 'compile-contract' policy requires a repository script target"),
        ):
            with self.subTest(argv=argv):
                self.assert_error(valid_manifest().replace('argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"]', argv, 1), expected)

        self.assert_error(
            valid_manifest().replace('cwd = "."', 'cwd = ".."', 1),
            "command 'compile-contract' cwd must remain inside the repository",
        )
        self.assert_error(
            valid_manifest().replace(
                'path_args = ["tools/storage-contract-inventory/Cargo.toml", "scripts/storage-ownership-contracts.toml"]',
                'path_args = ["tools/storage-contract-inventory/Cargo.toml"]',
                1,
            ),
            "command 'source-inventory' path_args must exactly match repository path arguments",
        )

    def test_active_obligation_runner_executes_each_command_once(self) -> None:
        result, log = self.run_runner(valid_manifest(commands=marker_commands()))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        entries = log.splitlines()
        self.assertEqual(entries.count("compile-contract"), 1)
        self.assertEqual(entries.count("source-inventory"), 1)
        self.assertEqual(entries.count("matrix-parity"), 1)
        self.assertEqual(entries.count("design-docs"), 1)
        self.assertNotIn("future-property", entries)

    def test_active_registered_command_cannot_be_green_without_execution(self) -> None:
        result, log = self.run_runner(valid_manifest(commands=marker_commands()))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("compile-contract", log)
        self.assertIn("source-inventory", log)
        self.assertIn("matrix-parity", log)
        self.assertIn("design-docs", log)

    def test_active_obligation_nonzero_exit_fails_and_deferred_is_not_run(self) -> None:
        result, log = self.run_runner(
            valid_manifest(commands=marker_commands()),
            files=marker_files(failing="compile-contract"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compile-contract", result.stderr)
        self.assertNotIn("future-property", log)

    def test_duplicate_shared_command_ids_are_deduplicated(self) -> None:
        manifest = valid_manifest(commands=marker_commands()).replace(
            'command_id = "matrix-parity"',
            'command_id = "compile-contract"',
        )
        result, log = self.run_runner(manifest)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(log.splitlines().count("compile-contract"), 1)

    def test_unknown_command_and_unknown_obligation_command_fail(self) -> None:
        self.assert_error(
            valid_manifest().replace('command_id = "compile-contract"', 'command_id = "missing-command"', 1),
            "references unknown command 'missing-command'",
        )

    def test_source_contract_selector_is_a_tagged_sum(self) -> None:
        self.assert_error(
            valid_manifest().replace(
                'selector = { kind = "lexical-drift", value = "RAW_HANDLE" }',
                'selector = { kind = "lexical-drift", value = "RAW_HANDLE", symbol = "old" }',
            ),
            "source record 'source-raw-handle' selector has unknown field 'symbol'",
        )

        self.assert_error(
            valid_manifest().replace(
                'selector = { kind = "lexical-drift", value = "RAW_HANDLE" }',
                'selector = { kind = "regex", value = "RAW_HANDLE" }',
            ),
            "source record 'source-raw-handle' has unknown selector kind 'regex'",
        )
        self.assert_error(
            valid_manifest().replace(
                'selector = { kind = "lexical-drift", value = "RAW_HANDLE" }',
                'selector = { kind = "lexical-drift" }',
            ),
            "source record 'source-raw-handle' lexical selector requires value",
        )
        self.assert_error(
            valid_manifest().replace(
                'selector = { kind = "rust-item", item_kind = "function", name = "old" }',
                'selector = { kind = "rust-item", name = "old" }',
            ),
            "source record 'source-old-function' rust-item selector requires item_kind",
        )

    def test_structural_inventory_ignores_comments_docstrings_formatting_and_aliases(self) -> None:
        files = repository_files()
        files["src/legacy.rs"] = textwrap.dedent(
            '''\
            /// RAW_HANDLE old helper mention must not be an AST item match.
            // RAW_HANDLE and `pub fn old_alias()` are lexical drift only.
            fn helper() {}
            pub use helper as old_alias;
            pub
            fn old() {
                let _ = helper;
                let _marker = "RAW_HANDLE";
            }
            '''
        )
        result = self.run_inventory(valid_manifest(), files=files)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        source = summary["obligations"]["active"]["source-inventory"]
        self.assertEqual(source["source-old-function"]["evidence_class"], "rust-structural")
        self.assertEqual(source["source-old-function"]["structural_match_count"], 1)
        self.assertEqual(source["source-raw-handle"]["evidence_class"], "lexical-drift")
        self.assertTrue(source["source-raw-handle"]["lexical_drift"])

    def test_source_risk_is_distinct_from_drift_evidence_class(self) -> None:
        parsed = tomllib.loads(valid_manifest())
        source_obligation = next(
            row for row in parsed["obligations"]["active"] if row["payload"]["kind"] == "source-inventory"
        )
        records = source_obligation["payload"]["records"]
        self.assertEqual(records[0]["category"], "raw-handle-extraction")
        self.assertEqual(records[0]["evidence_class"], "lexical-drift")
        self.assertEqual(records[1]["category"], "temporary-migration-adapter")
        self.assertEqual(records[1]["evidence_class"], "rust-structural")
        self.assertNotIn("semantic evidence", valid_manifest())

        invalid_risk = valid_manifest().replace(
            'category = "raw-handle-extraction"',
            'category = "lexical-drift"',
            1,
        )
        self.assert_error(invalid_risk, "source record 'source-raw-handle' has unknown SourceRisk 'lexical-drift'")
        invalid_evidence = valid_manifest().replace(
            'evidence_class = "rust-structural"',
            'evidence_class = "semantic-proof"',
            1,
        )
        self.assert_error(invalid_evidence, "source record 'source-old-function' has unknown EvidenceClass 'semantic-proof'")

    def test_source_remediation_sum_state_has_no_parallel_cross_product(self) -> None:
        self.assert_error(
            valid_manifest().replace(
                'remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
                'remediation = { kind = "remove", verification_command_id = "future-property" }',
            ),
            "remove remediation requires owner_ownership_id",
        )
        self.assert_error(
            valid_manifest().replace(
                'remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
                'remediation = { kind = "remove", target_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
            ),
            "remove remediation has unknown field 'target_ownership_id'",
        )
        self.assert_error(
            valid_manifest().replace(
                'remediation = { kind = "narrow", owner_ownership_id = "p3-host-ownership", target_ownership_id = "p3-ad-retention", verification_command_id = "future-property" }',
                'remediation = { kind = "narrow", target_ownership_id = "p3-ad-retention", verification_command_id = "future-property" }',
            ),
            "narrow remediation requires owner_ownership_id",
        )
        self.assert_error(
            valid_manifest().replace(
                'remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
                'remediation = { kind = "replace", owner_ownership_id = "p3-host-ownership", target_ownership_id = "p3-host-ownership" }',
            ),
            "replace remediation requires verification_command_id",
        )
        self.assert_error(
            valid_manifest().replace(
                'remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
                'remediation = { kind = "copy", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
            ),
            "source record 'source-raw-handle' has unknown remediation kind 'copy'",
        )

    def test_replacement_requires_target_contract_and_verification_command(self) -> None:
        manifest = valid_manifest().replace(
            'remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
            'remediation = { kind = "replace", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
        )
        self.assert_error(manifest, "replace remediation requires target_ownership_id")

    def test_removal_is_not_encoded_as_a_free_form_disposition(self) -> None:
        manifest = valid_manifest().replace(
            'remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
            'remediation = { disposition = "remove" }',
        )
        self.assert_error(manifest, "source record 'source-raw-handle' remediation has unknown field 'disposition'")

        outer_field = valid_manifest().replace(
            'remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
            'remediation_target_ownership_id = "p3-host-ownership", remediation = { kind = "remove", owner_ownership_id = "p3-host-ownership", verification_command_id = "future-property" }',
        )
        self.assert_error(
            outer_field,
            "source record 'source-raw-handle' has unknown field 'remediation_target_ownership_id'",
        )

    def test_api_matrix_is_explicit_and_has_no_formatting_string_assertion(self) -> None:
        parsed = tomllib.loads(valid_manifest())
        matrix_obligation = next(
            row for row in parsed["obligations"]["active"] if row["payload"]["kind"] == "api-matrix"
        )
        matrix = matrix_obligation["payload"]
        self.assertEqual(set(matrix["faces"]), {"typed_owner", "typed_view", "typed_view_mut", "erased_owner", "erased_view", "erased_view_mut"})
        self.assertEqual(set(matrix["families"]), {"layout_metadata", "dtype", "placement_device", "host_read"})
        self.assertEqual(set(matrix["applicability"]), set(matrix["faces"]))
        self.assertEqual(set(matrix["applicability"]["typed_owner"]), set(matrix["families"]))
        self.assertIn("not-applicable", matrix["applicability"]["typed_owner"].values())
        self.assertNotIn("layout_summary().contains", valid_manifest())

        invalid_status = valid_manifest().replace(
            'dtype = "not-applicable"',
            'dtype = "maybe"',
            1,
        )
        self.assert_error(invalid_status, "API matrix applicability must be required, not-applicable, or deferred")
        missing_face = valid_manifest().replace(
            'erased_view_mut = { layout_metadata = "required", dtype = "required", placement_device = "required", host_read = "required" }, ',
            "",
            1,
        )
        self.assert_error(missing_face, "API matrix applicability must cover every declared face")

    def test_doc_snippet_payload_is_bound_to_real_unique_snippets_and_command(self) -> None:
        parsed = tomllib.loads(valid_manifest())
        docs = [
            row for row in parsed["obligations"]["active"] if row["payload"]["kind"] == "doc-snippets"
        ]
        self.assertEqual(len(docs), 3)
        for row in docs:
            self.assertEqual(row["command_id"], "design-docs")
            self.assertEqual(len(row["payload"]["snippet_ids"]), len(set(row["payload"]["snippet_ids"])))
            self.assertTrue(row["payload"]["path"].startswith("docs/design/"))

        missing_snippet = valid_manifest().replace(
            "storage-ownership-contracts-transitions",
            "missing-storage-snippet",
            1,
        )
        self.assert_error(
            missing_snippet,
            "doc-snippets obligation 'active-p1-ad-retention' references missing snippet 'missing-storage-snippet'",
        )
        duplicate_snippet = valid_manifest().replace(
            '["storage-ownership-contracts-gates", "storage-ownership-contracts-transitions"]',
            '["storage-ownership-contracts-gates", "storage-ownership-contracts-gates"]',
            1,
        )
        self.assert_error(
            duplicate_snippet,
            "doc-snippets obligation 'active-p1-ad-retention' contains duplicate snippet ID",
        )
        wrong_path = valid_manifest().replace(
            'path = "docs/design/storage-ownership-contracts.md"',
            'path = "docs/design/index.md"',
            1,
        )
        self.assert_error(
            wrong_path,
            "doc-snippets obligation 'active-p1-ad-retention' snippet 'storage-ownership-contracts-gates' is not bound to target path",
        )
        wrong_command = valid_manifest().replace(
            'id = "active-p1-ad-retention"\nownership_id = "p1-ad-retention"\ncommand_id = "design-docs"',
            'id = "active-p1-ad-retention"\nownership_id = "p1-ad-retention"\ncommand_id = "source-inventory"',
        )
        self.assert_error(
            wrong_command,
            "doc-snippets obligation 'active-p1-ad-retention' requires command policy 'doc-snippets'",
        )

    def test_json_summary_derives_coverage_and_counts(self) -> None:
        result = self.run_checker(valid_manifest(), extra_args=("--json",))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["schema"], "tenferro.storage-ownership-contracts.v2")
        active_ownerships = set(CURRENT_ACTIVE_OWNERSHIPS)
        deferred_ownerships = {ownership_id for ownership_id, _gate, _phase in OWNERSHIPS} - active_ownerships
        self.assertEqual(set(summary["coverage"]["registry"]["gates"]), {f"G{i}" for i in range(1, 8)})
        self.assertEqual(set(summary["coverage"]["registry"]["phases"]), {phase for phase, _issue in PHASES})
        self.assertEqual(set(summary["coverage"]["active"]["ownerships"]), active_ownerships)
        self.assertEqual(set(summary["coverage"]["deferred"]["ownerships"]), deferred_ownerships)
        self.assertEqual(summary["counts"]["active_obligations"], len(active_ownerships))
        self.assertEqual(summary["counts"]["deferred_obligations"], len(deferred_ownerships))

        terminal = self.run_checker(valid_manifest(terminal=True), extra_args=("--json",))
        self.assertEqual(terminal.returncode, 0, terminal.stderr)
        terminal_summary = json.loads(terminal.stdout)
        self.assertEqual(terminal_summary["counts"]["deferred_obligations"], 0)
        self.assertEqual(
            set(terminal_summary["coverage"]["active"]["ownerships"]),
            {ownership_id for ownership_id, _gate, _phase in OWNERSHIPS},
        )
        self.assertEqual(terminal_summary["coverage"]["deferred"]["ownerships"], [])


def valid_manifest_sections() -> tuple[str, ...]:
    return (
        "registry",
        "commands",
        "obligations",
    )


if __name__ == "__main__":
    unittest.main()
