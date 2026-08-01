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
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check-storage-ownership-contracts.py"

CONTRACTS = (
    ("g1-span-access", "G1", "P4"),
    ("g2-allocation-group", "G2", "P5"),
    ("g3-submission", "G3", "P9"),
    ("g4-capability", "G4", "P3"),
    ("g5-cuda-provider", "G5", "P7"),
    ("g6-documentation", "G6", "P12"),
    ("g7-ad-retention", "G7", "P9"),
)


def _registry() -> str:
    gates = "\n".join(
        f'[[registry.gates]]\nid = "{gate}"\ntitle = "{gate} contract"'
        for _, gate, _ in CONTRACTS
    )
    phases = "\n".join(
        f'[[registry.phases]]\nid = "P{number}"\nissue = {issue}'
        for number, issue in (
            (0, 1556),
            (1, 1557),
            (2, 1558),
            (3, 1559),
            (4, 1560),
            (5, 1561),
            (6, 1562),
            (7, 1563),
            (8, 1564),
            (9, 1565),
            (10, 1566),
            (11, 1568),
            (12, 1569),
            (13, 1567),
        )
    )
    contracts = "\n".join(
        f'[[registry.contracts]]\nid = "{name}"\ngate = "{gate}"\nphase = "{phase}"'
        for name, gate, phase in CONTRACTS
    )
    return "\n".join(("[registry]", gates, phases, contracts))


def _commands(*, command_argv: str = 'argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"]') -> str:
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
        policy = "cargo-test"
        argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"]
        cwd = "."
        path_args = []
        '''
    )


def _fixture_suites() -> str:
    return textwrap.dedent(
        '''\
        [[fixture_suites]]
        id = "storage-pass-suite"
        contract = "g4-capability"
        kind = "trybuild-pass"
        root = "fixtures"
        glob = "**/*.rs"
        rationale = "The active compile fixtures are owned by the v2 harness."
        '''
    )


def _active_fixtures() -> str:
    rows = []
    for index, (name, _gate, _phase) in enumerate(CONTRACTS[:-1]):
        rows.append(
            textwrap.dedent(
                f'''\
                [[active_fixtures]]
                id = "active-{name}"
                contract = "{name}"
                kind = "trybuild-pass"
                path = "fixtures/{index}-{name}.rs"
                command_id = "compile-contract"
                rationale = "Active v2 fixture."
                '''
            )
        )
    return "\n".join(rows)


def _deferred_fixtures() -> str:
    return textwrap.dedent(
        '''\
        [[deferred_fixtures]]
        id = "deferred-g7-ad-retention"
        contract = "g7-ad-retention"
        kind = "property"
        future_path = "future/ad_retention.rs"
        command_id = "compile-contract"
        promotion = { artifact = "future/ad_retention.rs", condition = "Phase 9 implementation promotes this row in the same commit as the fixture." }
        rationale = "The AD retention fixture is not compiled before its owning phase."
        '''
    )


def _source_contracts() -> str:
    return textwrap.dedent(
        '''\
        [[source_contracts]]
        id = "source-raw-handle"
        contract = "g4-capability"
        scope = { root = "src", glob = "legacy.rs" }
        selector = { kind = "lexical-drift", value = "RAW_HANDLE" }
        category = "raw-handle-extraction"
        remediation = { kind = "remove", owner_contract = "g6-documentation", verification_command_id = "source-inventory" }
        rationale = "The lexical marker is only supplemental drift evidence."
        '''
    )


def valid_manifest(
    *,
    include: set[str] | None = None,
    schema: str = "tenferro.storage-ownership-contracts.v2",
) -> str:
    sections = {
        "registry": _registry(),
        "commands": _commands(),
        "fixture_suites": _fixture_suites(),
        "active_fixtures": _active_fixtures(),
        "deferred_fixtures": _deferred_fixtures(),
        "source_contracts": _source_contracts(),
    }
    selected = set(sections) if include is None else include
    return "\n\n".join(
        part for name, part in (("schema", f'schema = "{schema}"'), *sections.items()) if name == "schema" or name in selected
    ) + "\n"


def repository_files() -> dict[str, str]:
    files = {
        "src/legacy.rs": "pub fn old() { RAW_HANDLE; }\n",
    }
    for index, (name, _gate, _phase) in enumerate(CONTRACTS[:-1]):
        files[f"fixtures/{index}-{name}.rs"] = "fn compile_fixture() {}\n"
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
            "registry",
            "commands",
            "fixture_suites",
            "active_fixtures",
            "deferred_fixtures",
            "source_contracts",
        ):
            with self.subTest(section=section):
                self.assert_error(
                    valid_manifest(include=set(valid_manifest_sections()) - {section}),
                    f"manifest section '{section}' must be present and non-empty",
                )

    def test_gate_coverage_is_derived_from_authoritative_registry(self) -> None:
        manifest = valid_manifest().replace(
            'id = "g6-documentation"\ngate = "G6"',
            'id = "g6-documentation"\ngate = "G5"',
        )
        self.assert_error(manifest, "registry contract 'g6-documentation' duplicates gate 'G5'")

    def test_unknown_contract_relationship_is_rejected(self) -> None:
        self.assert_error(
            valid_manifest().replace('contract = "g1-span-access"', 'contract = "not-registered"', 1),
            "fixture 'active-g1-span-access' references unknown registry contract 'not-registered'",
        )

    def test_duplicate_registry_ids_are_rejected(self) -> None:
        self.assert_error(
            valid_manifest() + '\n[[registry.gates]]\nid = "G1"\ntitle = "duplicate"\n',
            "duplicate registry gate id 'G1'",
        )

    def test_source_scans_and_inventory_are_removed_as_duplicate_models(self) -> None:
        self.assert_error(
            valid_manifest() + '\n[[source_scans]]\nid = "legacy"\n',
            "manifest has unknown top-level field 'source_scans'",
        )

    def test_lifecycle_does_not_accept_status_or_parallel_path_fields(self) -> None:
        self.assert_error(
            valid_manifest() + '\nstatus = "deferred"\n',
            "manifest has unknown top-level field 'status'",
        )
        self.assert_error(
            valid_manifest() + '\n[[active_fixtures]]\nid = "bad"\ncontract = "g1-span-access"\nkind = "trybuild-pass"\npath = "fixtures/0-g1-span-access.rs"\nfuture_path = "future/bad.rs"\ncommand_id = "compile-contract"\nrationale = "bad"\n',
            "active fixture 'bad' has unknown field 'future_path'",
        )

    def test_deferred_fixture_requires_distinct_promotion_record(self) -> None:
        manifest = valid_manifest().replace(
            'promotion = { artifact = "future/ad_retention.rs", condition = "Phase 9 implementation promotes this row in the same commit as the fixture." }',
            'promotion = { artifact = "future/other.rs", condition = "future" }',
        )
        self.assert_error(manifest, "deferred fixture 'deferred-g7-ad-retention' promotion artifact must equal future_path")

    def test_existing_deferred_artifact_cannot_bypass_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = repository_files()
            files["future/ad_retention.rs"] = "fn future_fixture() {}\n"
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
        self.assertIn("future_path", result.stderr)

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

    def test_command_policy_rejects_shell_and_path_escape(self) -> None:
        for argv, expected in (
            ('argv = ["sh", "-c", "echo unsafe"]', "command 'compile-contract' executable 'sh' is not allowed"),
            ('argv = ["cargo", "test", "--manifest-path", "../Cargo.toml"]', "repository-relative"),
            ('argv = ["python3", "-c", "print(1)"]', "command 'compile-contract' policy requires a repository script target"),
        ):
            with self.subTest(argv=argv):
                self.assert_error(valid_manifest().replace('argv = ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract"]', argv, 1), expected)

    def test_unknown_command_and_unknown_obligation_command_fail(self) -> None:
        self.assert_error(
            valid_manifest().replace('command_id = "compile-contract"', 'command_id = "missing-command"', 1),
            "references unknown command 'missing-command'",
        )

    def test_source_contract_selector_is_a_tagged_sum(self) -> None:
        self.assert_error(
            valid_manifest().replace(
                'selector = { kind = "lexical-drift", value = "RAW_HANDLE" }',
                'needle = "RAW_HANDLE"\nsymbol = "old"',
            ),
            "source contract 'source-raw-handle' has unknown field 'needle'",
        )

    def test_replacement_requires_target_contract_and_verification_command(self) -> None:
        manifest = valid_manifest().replace(
            'remediation = { kind = "remove", owner_contract = "g6-documentation", verification_command_id = "source-inventory" }',
            'remediation = { kind = "replace", verification_command_id = "source-inventory" }',
        )
        self.assert_error(manifest, "replacement remediation must declare target_contract")

    def test_removal_is_not_encoded_as_a_free_form_disposition(self) -> None:
        manifest = valid_manifest().replace(
            'remediation = { kind = "remove", owner_contract = "g6-documentation", verification_command_id = "source-inventory" }',
            'remediation = { disposition = "remove" }',
        )
        self.assert_error(manifest, "source contract 'source-raw-handle' remediation has unknown field 'disposition'")

    def test_json_summary_derives_coverage_and_counts(self) -> None:
        result = self.run_checker(valid_manifest(), extra_args=("--json",))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["schema"], "tenferro.storage-ownership-contracts.v2")
        self.assertEqual(set(summary["coverage"]["gates"]), {f"G{i}" for i in range(1, 8)})
        self.assertEqual(summary["counts"]["active_fixtures"], 6)
        self.assertEqual(summary["counts"]["deferred_fixtures"], 1)


def valid_manifest_sections() -> tuple[str, ...]:
    return (
        "registry",
        "commands",
        "fixture_suites",
        "active_fixtures",
        "deferred_fixtures",
        "source_contracts",
    )


if __name__ == "__main__":
    unittest.main()
