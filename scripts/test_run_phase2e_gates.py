#!/usr/bin/env python3
"""Contract tests for Phase 2E dispatch and characterization gates."""

from __future__ import annotations

import hashlib
import json
import os
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
    budget = int(key.split("/budget-", 1)[1].split("/", 1)[0])
    value = {
        "key": key,
        "owner": owner,
        "surface": key.rsplit("/", 1)[-1],
        "budget": budget,
        "mode": mode,
        "counts": counts,
        "observed_cpus": [0],
        "hardware_skip": None,
        "numerical_passed": True,
        "typed_error_recovered": True,
        "unwind_recovered": True,
        "post_recovery_passed": True,
        "recovery": {
            "fresh_reset": True,
            "counts": counts,
            "mode": mode,
            "observed_cpus": [] if key.endswith("/U-O") else [0],
            "numerical_passed": True,
            "subset_passed": True,
        },
    }
    if owner == "ad":
        value.pop("counts")
        value.pop("mode")
        value.update({
            "session_entry": 1,
            "session_entry_cpus": [0],
            "placement_audit": [[worker, 0] for worker in range(budget)],
            "declared_cpus": [] if key.startswith("external-advisory/") else [0],
            "downstream_vector": "borrowed-add" if key.endswith("/E-N") else "borrowed-dot",
            "actual_install": 1,
            "actual_submit": 0,
            "actual_provider": 0 if key.endswith("/E-N") else 1,
            "operation_workers": [[0, 0]] if key.endswith("/E-N") else [],
        })
        value["recovery"] = {
            "fresh_reset": True,
            "session_entry": 1,
            "actual_install": 1,
            "actual_submit": 0,
            "actual_provider": 0 if key.endswith("/E-N") else 1,
            "operation_workers": [[0, 0]] if key.endswith("/E-N") else [],
            "observed_cpus": [0],
            "numerical_passed": True,
            "subset_passed": True,
        }
    elif key.endswith("/U-O"):
        value.update({
            "typed_error_kind": "Scheduling",
            "typed_error_source": "CPU domain executor scheduling failed: CPU domain CpuDomainId(9) does not support Outer mode",
            "observed_cpus": [],
        })
    return value


class ActualOperationWorkerEvidenceTests(unittest.TestCase):
    def test_managed_eager_native_rejects_missing_operation_worker_pairs(self) -> None:
        cpu, ad = artifacts()
        target = next(
            item for item in ad["characterization"]
            if item["key"] == "managed-exact/budget-4/E-N"
        )
        target["operation_workers"] = []
        with self.assertRaisesRegex(protocol.ProtocolError, "operation worker"):
            gates.compose_characterization(cpu, ad)


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
        "cross_socket_locality": {
            "usable_numa_nodes": 1,
            "hardware_skip": {
                "kind": "InsufficientNumaNodes", "required": 2, "available": 1,
            },
            "probes": [],
        },
    }
    ad = {
        "owner": "ad",
        "session_entries": [1, 1, 1, 1, 1],
        "characterization": [row(key, "ad") for key in sorted(ad_keys)],
    }
    return cpu, ad


class InventoryTests(unittest.TestCase):
    def test_asymmetric_numa_capacities_are_bound_per_placement(self) -> None:
        cpu, ad = artifacts()
        composed = gates.compose_characterization(cpu, ad)
        capacity = {
            "process_allowed_cpus": [0, 1, 2, 3],
            "process_allowed_capacity": 4,
            "managed_node_cpus": [0],
            "managed_node_capacity": 1,
            "usable_numa_nodes": 1,
        }
        gates.attach_hardware_validity(composed, capacity_provenance=capacity)
        managed = next(
            item for item in composed["rows"]
            if item["key"] == "managed-exact/budget-2/D-N"
        )
        external = next(
            item for item in composed["rows"]
            if item["key"] == "external-exact/budget-2/D-N"
        )
        self.assertEqual(managed["placement_capacity"], 1)
        self.assertEqual(managed["affinity_hardware_skip"]["available"], 1)
        self.assertEqual(external["placement_capacity"], 4)
        self.assertIsNone(external["affinity_hardware_skip"])
        self.assertEqual(composed["capacity_provenance"], capacity)

    def test_real_hardware_skips_are_typed_but_correctness_never_skips(self) -> None:
        cpu, ad = artifacts()
        composed = gates.compose_characterization(cpu, ad)
        gates.attach_hardware_validity(composed, capacity_provenance={
            "process_allowed_cpus": [0], "process_allowed_capacity": 1,
            "managed_node_cpus": [0], "managed_node_capacity": 1,
            "usable_numa_nodes": 1,
        })
        for item in composed["rows"]:
            self.assertIsNone(item["hardware_skip"])
            if item["surface"] in {"U-O", "U-I"} or item["budget"] == 1:
                self.assertIsNone(item["affinity_hardware_skip"])
            else:
                self.assertEqual(item["affinity_hardware_skip"], {
                    "kind": "InsufficientAllowedCpus",
                    "required": item["budget"],
                    "available": 1,
                })
        self.assertEqual(composed["cross_socket_locality"]["hardware_skip"], {
            "kind": "InsufficientNumaNodes", "required": 2, "available": 1,
        })

    def test_exact_owner_partitions_compose_to_47(self) -> None:
        cpu, ad = artifacts()
        composed = gates.compose_characterization(cpu, ad)
        self.assertEqual(composed["row_count"], 47)
        self.assertEqual(len(composed["rows"]), 47)
        ad_rows = [item for item in composed["rows"] if item["owner"] == "ad"]
        self.assertTrue(all("observed_cpu_source" not in item for item in ad_rows))
        self.assertTrue(all(item["downstream_mode_source"].endswith(("/D-N", "/D-D")) for item in ad_rows))

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

    def test_source_contract_binds_every_hot_item_independently(self) -> None:
        inventory = gates.validate_source_contract(pathlib.Path.cwd())
        self.assertEqual(len(inventory), len(gates.SOURCE_HOT_ITEMS))
        self.assertEqual(
            [(item["path"], item["signature"]) for item in inventory],
            list(gates.SOURCE_HOT_ITEMS),
        )
        self.assertTrue(all(set(item) == {
            "path", "signature", "source_sha256", "item_sha256"
        } for item in inventory))
        self.assertEqual(
            len({(item["path"], item["signature"]) for item in inventory}),
            len(gates.SOURCE_HOT_ITEMS),
        )


class ProvenanceTests(unittest.TestCase):
    def test_main_never_owns_failure_artifacts_through_rejected_root_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            scratch = root / "scratch"
            scratch.mkdir()
            arguments = [
                "--repository", str(pathlib.Path.cwd()),
                "--candidate", "a" * 40,
                "--common-lock", str(pathlib.Path.cwd() / "Cargo.lock"),
                "--scratch-root", str(scratch),
                "--path", "/usr/bin",
                "--home", str(root),
                "--cargo-home", str(root),
            ]
            for kind in ("final", "ancestor"):
                target = root / f"{kind}-target"
                target.mkdir()
                if kind == "final":
                    evidence = root / "final-link"
                    evidence.symlink_to(target, target_is_directory=True)
                else:
                    ancestor = root / "ancestor-link"
                    ancestor.symlink_to(target, target_is_directory=True)
                    evidence = ancestor / "evidence"
                    evidence.mkdir()
                    target = evidence.resolve()
                locked = target / build.LOCK_PATHS["common"]
                locked.parent.mkdir(parents=True)
                locked.write_bytes(b"attacker-controlled lock\n")
                before = {
                    path.relative_to(target): path.read_bytes()
                    for path in target.rglob("*") if path.is_file()
                }
                with (
                    self.subTest(kind=kind),
                    mock.patch.object(gates, "validate_candidate_worktree"),
                    self.assertRaisesRegex(
                        protocol.ProtocolError, "symbolic link"
                    ),
                ):
                    gates.main([
                        "--evidence-root", str(evidence), *arguments,
                    ])
                after = {
                    path.relative_to(target): path.read_bytes()
                    for path in target.rglob("*") if path.is_file()
                }
                self.assertEqual(after, before)

    def test_runner_rejects_symlink_evidence_root_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            real = root / "real-evidence"
            real.mkdir()
            alias = root / "evidence-link"
            alias.symlink_to(real, target_is_directory=True)
            scratch = root / "scratch"
            scratch.mkdir()
            with (
                mock.patch.object(gates, "validate_candidate_worktree"),
                mock.patch.object(
                    gates, "validate_external_scratch_root", return_value=scratch
                ),
                mock.patch.object(gates, "validate_source_contract", return_value=[]),
                mock.patch.object(
                    build, "build_dispatch_and_characterization_artifacts"
                ) as build_artifacts,
                self.assertRaises(protocol.ProtocolError),
            ):
                gates._run_main(
                    [
                        "--repository", str(pathlib.Path.cwd()),
                        "--evidence-root", str(alias),
                        "--candidate", "a" * 40,
                        "--common-lock", str(pathlib.Path.cwd() / "Cargo.lock"),
                        "--scratch-root", str(scratch),
                        "--path", "/usr/bin",
                        "--home", str(root),
                        "--cargo-home", str(root),
                    ]
                )
            build_artifacts.assert_not_called()

    def test_normative_reads_inventory_and_writes_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(protocol.ProtocolError):
                gates._read_json(link)
            with self.assertRaises(protocol.ProtocolError):
                gates._write_new_bytes(link, b"replacement")
            with self.assertRaises(protocol.ProtocolError):
                gates.atomic_write_json(link, {"replacement": True})
            special = root / "special"
            os.mkfifo(special)
            with self.assertRaises(protocol.ProtocolError):
                gates.normative_regular_files(root)

    def test_active_root_identity_rejects_swapped_normative_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            root = base / "evidence"
            identity = protocol.prepare_empty_root_identity(root)
            outside = base / "outside"
            outside.mkdir()
            nested = root / "nested"
            nested.mkdir()
            nested.rmdir()
            nested.symlink_to(outside, target_is_directory=True)
            previous = gates._ACTIVE_ROOT_IDENTITY
            gates._ACTIVE_ROOT_IDENTITY = identity
            try:
                with self.assertRaisesRegex(protocol.ProtocolError, "symbolic link"):
                    gates._write_new_bytes(nested / "manifest.json", b"{}\n")
            finally:
                gates._ACTIVE_ROOT_IDENTITY = previous
                identity.close()
            self.assertEqual(list(outside.iterdir()), [])

    def test_rust_evidence_writers_are_exclusive_and_non_following(self) -> None:
        paths = (
            "crates/tenferro-cpu/src/tests/phase2e.rs",
            "crates/tenferro-ad/src/eager/tests/phase2e.rs",
            "crates/tenferro-cpu/benches/numa_execution.rs",
            "crates/tenferro-ad/benches/phase2e_characterization.rs",
        )
        for relative in paths:
            source = pathlib.Path(relative).read_text()
            self.assertIn("create_new(true)", source, relative)
            self.assertIn("write_all", source, relative)
            self.assertNotIn("std::fs::write(", source, relative)

    def test_phase2e_rust_targets_have_non_linux_compile_gates(self) -> None:
        repository = pathlib.Path(__file__).resolve().parents[1]
        for relative in (
            "crates/tenferro-cpu/src/tests.rs",
            "crates/tenferro-ad/src/eager/tests.rs",
        ):
            source = (repository / relative).read_text()
            self.assertIn(
                '#[cfg(any(target_os = "linux", target_os = "android"))]\nmod phase2e;',
                source,
                relative,
            )
        for relative in (
            "crates/tenferro-cpu/benches/numa_execution.rs",
            "crates/tenferro-ad/benches/phase2e_characterization.rs",
        ):
            source = (repository / relative).read_text()
            self.assertIn(
                '#[cfg(any(target_os = "linux", target_os = "android"))]\nfn current_cpu()',
                source,
                relative,
            )
            self.assertIn('#[cfg(not(any(target_os = "linux", target_os = "android")))]', source)
            self.assertIn("fn main() {}", source, relative)

    def test_operation_observer_is_custom_cfg_only_not_a_cargo_feature(self) -> None:
        cpu_manifest = pathlib.Path("crates/tenferro-cpu/Cargo.toml").read_text()
        ad_manifest = pathlib.Path("crates/tenferro-ad/Cargo.toml").read_text()
        self.assertNotIn("phase2e-observe", cpu_manifest)
        self.assertNotIn("phase2e-observe", ad_manifest)
        for relative in (
            "crates/tenferro-cpu/src/lib.rs",
            "crates/tenferro-cpu/src/affinity.rs",
            "crates/tenferro-cpu/src/elementwise.rs",
            "crates/tenferro-ad/src/eager/tests/phase2e.rs",
        ):
            source = pathlib.Path(relative).read_text()
            self.assertNotIn('feature = "phase2e-observe"', source)
            self.assertIn("tenferro_phase2e_operation_observe", source)

    def test_characterization_benches_measure_exact_workloads_only(self) -> None:
        cpu = pathlib.Path("crates/tenferro-cpu/benches/numa_execution.rs").read_text()
        task7 = cpu.split("fn bench_phase2e_rows", 1)[1]
        self.assertNotIn("native.mul", task7)
        self.assertNotIn("run_session_workload(&mut dot", task7)
        grouped_setup, grouped_loop = task7.split(
            'c.bench_function(&format!("phase2e/{grouped_key}")', 1
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
            self.assertEqual(
                command[command.index("--features") + 1],
                "cpu-faer",
            )
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
            environment = build.dispatch_cargo_environment(
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
                "requested_features": ["cpu-faer"],
                "compiler_configuration": {
                    "observer_cfg": "tenferro_phase2e_operation_observe",
                    "rustflags": build.DISPATCH_RUSTFLAGS,
                },
                "no_default_features": True,
                "target": "x86_64-unknown-linux-gnu",
                "feature_query_argv": list(build.feature_query_command(
                    "x86_64-unknown-linux-gnu", package="tenferro-cpu",
                    requested_features=("cpu-faer",),
                    no_default_features=True,
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
            manifest["argv"] = list(build.DISPATCH_TEST_COMMANDS["tenferro-cpu"])
            manifest["environment"]["RUSTFLAGS"] += " -Ctarget-cpu=native"
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
            sealed = protocol.runtime_environment(path="/controlled", home="/empty-home")
            with mock.patch.object(gates, "run_bounded", return_value=completed) as run:
                gates.run_test_executable(
                    executable, gates.CPU_FILTER, repository=root,
                    evidence_root=root / "evidence", environment=sealed,
                )
            argv = run.call_args.args[0]
            self.assertEqual(argv, (str(executable.resolve()), gates.CPU_FILTER, "--nocapture"))
            self.assertEqual(run.call_args.kwargs["deadline"], 120)
            self.assertEqual(
                run.call_args.kwargs["environment"][gates.EVIDENCE_ENVIRONMENT_KEY],
                str((root / "evidence").resolve()),
            )
            expected = dict(sealed)
            expected[gates.EVIDENCE_ENVIRONMENT_KEY] = str((root / "evidence").resolve())
            self.assertEqual(run.call_args.kwargs["environment"], expected)
            self.assertNotIn("CARGO_HOME", run.call_args.kwargs["environment"])

    def test_bench_row_uses_direct_binary_and_30_second_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "bench-bin"
            executable.write_bytes(b"binary")
            (root / "criterion").mkdir()
            completed = __import__("subprocess").CompletedProcess([], 0, "", "")
            sealed = protocol.runtime_environment(path="/controlled", home="/empty-home")
            original = dict(sealed)
            with (
                mock.patch.object(gates, "run_bounded", return_value=completed) as run,
                mock.patch.object(
                    protocol,
                    "runtime_environment",
                    wraps=protocol.runtime_environment,
                ) as construct_environment,
            ):
                gates.run_bench_row(
                    executable, "managed-exact/budget-2/D-N", repository=root,
                    environment=sealed, criterion_home=root / "criterion",
                )
            construct_environment.assert_called_once_with(
                path="/controlled",
                home="/empty-home",
                criterion_home=str((root / "criterion").resolve()),
                affinity_row="managed-exact/budget-2/D-N",
                affinity_file=str((root / "criterion" / "affinity.json").resolve()),
            )
            self.assertEqual(sealed, original)
            self.assertEqual(
                run.call_args.args[0],
                (
                    str(executable.resolve()), "managed-exact/budget-2/D-N",
                    "--bench", "--noplot",
                ),
            )
            self.assertEqual(run.call_args.kwargs["deadline"], 30)
            self.assertEqual(
                run.call_args.kwargs["environment"],
                protocol.runtime_environment(
                    path="/controlled", home="/empty-home",
                    criterion_home=str((root / "criterion").resolve()),
                    affinity_row="managed-exact/budget-2/D-N",
                    affinity_file=str(
                        (root / "criterion" / "affinity.json").resolve()
                    ),
                ),
            )

    def test_runtime_helpers_reject_build_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "binary"
            executable.write_bytes(b"binary")
            controlled_bin = root / "bin"
            controlled_bin.mkdir()
            cargo_home = root / "cargo-home"
            cargo_home.mkdir()
            target = root / "target"
            target.mkdir()
            build_environment = protocol.cargo_environment(
                path=str(controlled_bin), home=str(root / "empty-home"),
                cargo_home=str(cargo_home), target_dir=str(target),
            )
            with self.assertRaises(protocol.ProtocolError):
                gates.run_test_executable(
                    executable, gates.CPU_FILTER, repository=root,
                    evidence_root=root / "evidence", environment=build_environment,
                )
            with self.assertRaises(protocol.ProtocolError):
                gates.run_bench_row(
                    executable, "managed-exact/budget-2/D-N", repository=root,
                    environment=build_environment, criterion_home=root / "criterion",
                )

    def test_scratch_root_must_be_external_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            repository = root / "repository"
            repository.mkdir()
            evidence = root / "evidence"
            external = root / "external-scratch"
            external.mkdir()
            gates.validate_external_scratch_root(repository, evidence, external)
            for invalid in (
                repository / "scratch", repository,
                evidence / "scratch", evidence,
                root,
            ):
                invalid.mkdir(parents=True, exist_ok=True)
                with self.assertRaises(protocol.ProtocolError, msg=str(invalid)):
                    gates.validate_external_scratch_root(repository, evidence, invalid)

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
            (scratch / "affinity.json").write_text(json.dumps({
                "key": "managed-exact/budget-2/D-N",
                "ownership": "managed-exact",
                "guarantee": "ExactDeclared",
                "budget": 2,
                "worker_count": 2,
                "declared_cpus": [0, 1],
                "observations": [[0, 0], [1, 1]],
            }))
            completed = subprocess.CompletedProcess([], 0, "stdout\n", "stderr\n")
            with mock.patch.object(gates, "run_bench_row", return_value=completed):
                record = gates.capture_bench_row(
                    executable, "managed-exact/budget-2/D-N", repository=root,
                    environment={"PATH": "/controlled"}, criterion_home=scratch,
                    evidence_root=root / "evidence", placement_capacity=2,
                )
            self.assertEqual(record["latency_ns"], {
                "point_estimate": 12.0, "lower_bound": 10.0,
                "upper_bound": 14.0, "confidence_level": 0.95,
            })
            for name in ("stdout", "stderr", "criterion_estimates", "fixture_affinity"):
                artifact = pathlib.Path(record["artifacts"][name]["path"])
                self.assertTrue(artifact.is_file())
                self.assertEqual(
                    record["artifacts"][name]["sha256"], gates.sha256_file(artifact)
                )

    def test_exact_bench_row_requires_budget_distinct_cpu_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "bench-bin"
            executable.write_bytes(b"binary")
            scratch = root / "criterion"
            estimate = scratch / "criterion-id/new/estimates.json"
            estimate.parent.mkdir(parents=True)
            estimate.write_text(json.dumps({"mean": {"confidence_interval": {
                "confidence_level": 0.95, "lower_bound": 10.0, "upper_bound": 14.0,
            }, "point_estimate": 12.0}}))
            (scratch / "affinity.json").write_text(json.dumps({
                "key": "managed-exact/budget-2/D-N",
                "ownership": "managed-exact",
                "guarantee": "ExactDeclared",
                "budget": 2,
                "worker_count": 2,
                "declared_cpus": [0, 1],
                "placement_capacity": 2,
                "observations": [[0, 0], [1, 0]],
            }))
            completed = subprocess.CompletedProcess([], 0, "", "")
            with mock.patch.object(gates, "run_bench_row", return_value=completed):
                with self.assertRaisesRegex(protocol.ProtocolError, "placement"):
                    gates.capture_bench_row(
                        executable, "managed-exact/budget-2/D-N", repository=root,
                        environment={"PATH": "/controlled"}, criterion_home=scratch,
                        evidence_root=root / "evidence", placement_capacity=2,
                    )
    def test_latency_hardware_skip_does_not_launch_or_create_fake_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            executable = root / "bench-bin"
            executable.write_bytes(b"binary")
            skip = {
                "kind": "InsufficientAllowedCpus", "required": 4, "available": 2,
            }
            with mock.patch.object(gates, "run_bench_row") as run:
                record = gates.capture_bench_row(
                    executable, "managed-exact/budget-4/D-N", repository=root,
                    environment=protocol.runtime_environment(path="/bin", home="/tmp/home"),
                    criterion_home=root / "criterion", evidence_root=root / "evidence",
                    placement_capacity=1, hardware_skip=skip,
                )
            run.assert_not_called()
            self.assertEqual(record, {
                "row_id": "managed-exact__budget-4__D-N",
                "key": "managed-exact/budget-4/D-N",
                "hardware_skip": skip,
                "placement_capacity": 1,
                "latency_ns": None,
                "artifacts": {},
            })

    def test_atomic_json_and_digest_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "nested/manifest.json"
            gates.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})
            self.assertEqual(gates.sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
