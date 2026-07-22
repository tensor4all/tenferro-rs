#!/usr/bin/env python3
"""Contract tests for Phase 2E dispatch and characterization gates."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol
from scripts import run_phase2e_gates as gates


def row(key: str, owner: str) -> dict[str, object]:
    counts, mode = gates.expected_row_contract(key)
    value = {
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
    if owner == "ad":
        value.pop("counts")
        value.pop("mode")
        value.update({
            "session_entry": 1,
            "downstream_vector": "borrowed-add" if key.endswith("/E-N") else "borrowed-dot",
            "actual_install": 1,
            "actual_submit": 0,
            "actual_provider": 0 if key.endswith("/E-N") else 1,
        })
    elif key.endswith("/U-O"):
        value.update({
            "typed_error_kind": "Scheduling",
            "typed_error_source": "CPU domain executor scheduling failed: CPU domain CpuDomainId(9) does not support Outer mode",
            "observed_cpus": [],
        })
    return value


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
    def test_characterization_benches_measure_exact_workloads_only(self) -> None:
        cpu = pathlib.Path("crates/tenferro-cpu/benches/numa_execution.rs").read_text()
        task7 = cpu.split("fn bench_phase2e_rows", 1)[1]
        self.assertNotIn("native.mul", task7)
        self.assertNotIn("run_session_workload(&mut dot", task7)
        grouped_setup, grouped_loop = task7.split(
            'c.bench_function(&format!("phase2e/{ownership}/budget-{budget}/G-O")', 1
        )
        self.assertIn("GroupedGemmConfig::new", grouped_setup)
        self.assertNotIn("GroupedGemmConfig::new", grouped_loop.split("});", 1)[0])

    def test_cli_builds_candidate_artifacts_before_running_evidence(self) -> None:
        source = __import__("inspect").getsource(gates._run_main)
        self.assertIn("build_dispatch_and_characterization_artifacts", source)
        for option in (
            "--common-lock", "--scratch-root", "--path", "--home", "--cargo-home"
        ):
            self.assertIn(option, source)

    def test_normative_manifest_contract_is_exact_not_nonempty(self) -> None:
        source = __import__("inspect").getsource(gates.validate_test_build_manifest)
        self.assertNotIn("if not manifest.get(field)", source)
        for field in (
            "protocol_sha256", "candidate_tree_sha256", "source_inventory",
            "common_lock_sha256", "feature_graph_sha256", "executable_sha256",
        ):
            self.assertIn(field, source)

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
            root = pathlib.Path(temporary)
            repository = pathlib.Path.cwd()
            candidate = subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=repository,
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            tree = subprocess.run(
                ("git", "ls-tree", "-r", "-z", "--full-tree", candidate),
                cwd=repository, check=True, capture_output=True, text=True,
            ).stdout
            tree_digest = hashlib.sha256(tree.encode()).hexdigest()
            executable = root / "test-bin"
            executable.write_bytes(b"binary")
            digest = hashlib.sha256(b"binary").hexdigest()
            tool = pathlib.Path(sys.executable).resolve()
            tool_identity = {"path": str(tool), "sha256": gates.sha256_file(tool)}
            path = root / "bin"
            path.mkdir()
            environment = protocol.cargo_environment(
                path=str(path), home=str(root / "home"),
                cargo_home=str(root / "cargo-home"), target_dir=str(root / "target"),
            )
            graph = "feature graph\n"
            lock = repository / "Cargo.lock"
            manifest = {
                "validity_state": "COMPLETE", "candidate": candidate,
                "protocol_version": protocol.PROTOCOL_VERSION,
                "protocol_sha256": gates.sha256_file(repository / "scripts/phase2e_protocol.py"),
                "package": "tenferro-cpu", "source_sha256": tree_digest,
                "candidate_tree_sha256": tree_digest,
                "source_inventory": {
                    relative: gates.sha256_file(repository / relative)
                    for relative in build.TASK7_SOURCE_PATHS
                },
                "lock_sha256": gates.sha256_file(lock),
                "common_lock_sha256": gates.sha256_file(lock),
                "feature_graph": graph,
                "feature_graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
                "argv": list(build.DISPATCH_TEST_COMMANDS["tenferro-cpu"]),
                "environment": environment,
                "requested_features": ["cpu-faer"], "no_default_features": True,
                "target": "x86_64-unknown-linux-gnu",
                "feature_query_argv": list(build.feature_query_command(
                    "x86_64-unknown-linux-gnu", package="tenferro-cpu",
                    requested_features=("cpu-faer",), no_default_features=True,
                )),
                "toolchain": {
                    "git": tool_identity, "cargo": tool_identity,
                    "rustc": {**tool_identity, "version": "rustc test"},
                },
                "executable": str(executable), "executable_sha256": digest,
            }
            self.assertEqual(
                gates.validate_test_build_manifest(
                    manifest, package="tenferro-cpu", candidate=candidate,
                    repository=repository, common_lock=lock,
                ), executable.resolve()
            )
            manifest["argv"] = ["cargo", "test"]
            with self.assertRaises(protocol.ProtocolError):
                gates.validate_test_build_manifest(
                    manifest, package="tenferro-cpu", candidate=candidate,
                    repository=repository, common_lock=lock,
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
                (
                    str(executable.resolve()), "managed-exact/budget-2/D-N",
                    "--bench", "--noplot",
                ),
            )
            self.assertEqual(run.call_args.kwargs["deadline"], 30)

    def test_normative_bench_row_copies_logs_estimate_and_ci(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "bench-bin"
            executable.write_bytes(b"binary")
            scratch = root / "criterion"
            estimate = scratch / "criterion-id/new/estimates.json"
            estimate.parent.mkdir(parents=True)
            estimate.write_text(json.dumps({
                "mean": {
                    "confidence_interval": {
                        "confidence_level": 0.95,
                        "lower_bound": 10.0,
                        "upper_bound": 14.0,
                    },
                    "point_estimate": 12.0,
                }
            }))
            completed = subprocess.CompletedProcess([], 0, "stdout\n", "stderr\n")
            with mock.patch.object(gates, "run_bench_row", return_value=completed):
                record = gates.capture_bench_row(
                    executable, "managed-exact/budget-2/D-N", repository=root,
                    environment={"PATH": "/controlled"}, criterion_home=scratch,
                    evidence_root=root / "evidence",
                )
            self.assertEqual(record["latency_ns"], {
                "point_estimate": 12.0, "lower_bound": 10.0,
                "upper_bound": 14.0, "confidence_level": 0.95,
            })
            for name in ("stdout", "stderr", "criterion_estimates"):
                artifact = pathlib.Path(record["artifacts"][name]["path"])
                self.assertTrue(artifact.is_file())
                self.assertEqual(
                    record["artifacts"][name]["sha256"], gates.sha256_file(artifact)
                )

    def test_atomic_json_and_digest_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "nested/manifest.json"
            gates.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})
            self.assertEqual(gates.sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
