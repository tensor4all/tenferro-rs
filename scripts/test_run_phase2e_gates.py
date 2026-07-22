#!/usr/bin/env python3
"""Contract tests for Phase 2E dispatch and characterization gates."""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol
from scripts import run_phase2e_gates as gates


def row(key: str, owner: str) -> dict[str, object]:
    counts, mode = gates.expected_row_contract(key)
    return {
        "key": key,
        "owner": owner,
        "surface": key.rsplit("/", 1)[-1],
        "budget": 2,
        "mode": mode,
        "counts": counts,
        "observed_cpus": [0],
        "hardware_skip": None,
        "numerical_passed": True,
        "typed_error_recovered": True,
        "unwind_recovered": True,
        "post_recovery_passed": True,
    }


def artifacts() -> tuple[dict[str, object], dict[str, object]]:
    cpu_keys, ad_keys = gates.canonical_keys()
    cpu = {
        "owner": "cpu",
        "canonical_vectors": [
            [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1],
        ],
        "characterization": [row(key, "cpu") for key in sorted(cpu_keys)],
    }
    ad = {
        "owner": "ad",
        "session_entries": [1, 1, 1, 1, 1],
        "characterization": [row(key, "ad") for key in sorted(ad_keys)],
    }
    return cpu, ad


class InventoryTests(unittest.TestCase):
    def test_exact_owner_partitions_compose_to_47(self) -> None:
        cpu, ad = artifacts()
        composed = gates.compose_characterization(cpu, ad)
        self.assertEqual(composed["row_count"], 47)
        self.assertEqual(len(composed["rows"]), 47)

    def test_duplicate_missing_and_wrong_owner_are_rejected(self) -> None:
        for mutation in ("duplicate", "missing", "owner"):
            cpu, ad = artifacts()
            rows = cpu["characterization"]
            assert isinstance(rows, list)
            if mutation == "duplicate":
                rows[-1] = dict(rows[0])
            elif mutation == "missing":
                rows.pop()
            else:
                rows[0]["owner"] = "ad"
            with self.assertRaises(protocol.ProtocolError, msg=mutation):
                gates.compose_characterization(cpu, ad)

    def test_hardware_skip_is_typed(self) -> None:
        cpu, _ad = artifacts()
        rows = cpu["characterization"]
        assert isinstance(rows, list)
        rows[0]["hardware_skip"] = "not-typed"
        with self.assertRaises(protocol.ProtocolError):
            gates.validate_partition(cpu, "cpu")
        rows[0]["hardware_skip"] = {
            "kind": "InsufficientAllowedCpus", "required": 4, "available": 2
        }
        gates.validate_partition(cpu, "cpu")

    def test_success_row_requires_actual_cpu_observations(self) -> None:
        cpu, _ad = artifacts()
        rows = cpu["characterization"]
        assert isinstance(rows, list)
        rows[0]["observed_cpus"] = []
        with self.assertRaises(protocol.ProtocolError):
            gates.validate_partition(cpu, "cpu")

    def test_source_item_scans_the_complete_balanced_body(self) -> None:
        source = "fn hot() { if true { dispatch(); } TypeId::of::<u8>(); }\nfn cold() {}"
        self.assertIn("TypeId", gates.source_item(source, "fn hot()"))
        self.assertNotIn("cold", gates.source_item(source, "fn hot()"))
        with self.assertRaises(protocol.ProtocolError):
            gates.source_item(source, "fn missing()")


class ProvenanceTests(unittest.TestCase):
    def test_exact_locked_feature_build_commands_and_deadlines(self) -> None:
        for package in ("tenferro-cpu", "tenferro-ad"):
            command = build.DISPATCH_TEST_COMMANDS[package]
            self.assertEqual(command[:4], ("cargo", "test", "--locked", "--no-run"))
            self.assertIn("--no-default-features", command)
            self.assertEqual(command[command.index("--features") + 1], "cpu-faer")
            self.assertEqual(command[-1], "--message-format=json")
        self.assertEqual(gates.TEST_DEADLINE_SECONDS, 120)
        self.assertEqual(gates.BENCH_ROW_DEADLINE_SECONDS, 30)
        self.assertEqual(gates.TERMINATION_GRACE_SECONDS, 5)

    def test_manifest_binds_candidate_executable_digest_and_exact_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = pathlib.Path(temporary) / "test-bin"
            executable.write_bytes(b"binary")
            digest = hashlib.sha256(b"binary").hexdigest()
            manifest = {
                "validity_state": "COMPLETE", "candidate": "a" * 40,
                "package": "tenferro-cpu", "source_sha256": "s",
                "lock_sha256": "l", "feature_graph_sha256": "f",
                "argv": list(build.DISPATCH_TEST_COMMANDS["tenferro-cpu"]),
                "environment": {"PATH": "/controlled"},
                "requested_features": ["cpu-faer"], "no_default_features": True,
                "target": "x86_64-unknown-linux-gnu",
                "feature_query_argv": list(build.feature_query_command(
                    "x86_64-unknown-linux-gnu", package="tenferro-cpu",
                    requested_features=("cpu-faer",), no_default_features=True,
                )),
                "toolchain": {"cargo": "cargo 1.90"},
                "executable": str(executable), "executable_sha256": digest,
            }
            self.assertEqual(
                gates.validate_test_build_manifest(
                    manifest, package="tenferro-cpu", candidate="a" * 40
                ), executable.resolve()
            )
            manifest["argv"] = ["cargo", "test"]
            with self.assertRaises(protocol.ProtocolError):
                gates.validate_test_build_manifest(
                    manifest, package="tenferro-cpu", candidate="a" * 40
                )

    def test_direct_run_uses_hashed_executable_filter_and_nocapture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "test-bin"
            executable.write_bytes(b"binary")
            completed = __import__("subprocess").CompletedProcess([], 0, "", "")
            with mock.patch.object(gates, "run_bounded", return_value=completed) as run:
                gates.run_test_executable(
                    executable, gates.CPU_FILTER, repository=root,
                    evidence_root=root / "evidence", environment={"PATH": "/controlled"},
                )
            argv = run.call_args.args[0]
            self.assertEqual(argv, (str(executable.resolve()), gates.CPU_FILTER, "--nocapture"))
            self.assertEqual(run.call_args.kwargs["deadline"], 120)
            self.assertEqual(
                run.call_args.kwargs["environment"][gates.EVIDENCE_ENVIRONMENT_KEY],
                str((root / "evidence").resolve()),
            )

    def test_bench_row_uses_direct_binary_and_30_second_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "bench-bin"
            executable.write_bytes(b"binary")
            completed = __import__("subprocess").CompletedProcess([], 0, "", "")
            with mock.patch.object(gates, "run_bounded", return_value=completed) as run:
                gates.run_bench_row(
                    executable, "managed-exact/budget-2/D-N", repository=root,
                    environment={"PATH": "/controlled"}, criterion_home=root / "criterion",
                )
            self.assertEqual(
                run.call_args.args[0],
                (str(executable.resolve()), "managed-exact/budget-2/D-N", "--noplot"),
            )
            self.assertEqual(run.call_args.kwargs["deadline"], 30)

    def test_atomic_json_and_digest_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "nested/manifest.json"
            gates.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})
            self.assertEqual(gates.sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
