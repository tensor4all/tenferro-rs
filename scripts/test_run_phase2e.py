#!/usr/bin/env python3
"""Contract tests for the Phase 2E outer orchestrator."""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from unittest import mock


from scripts import phase2e_protocol as protocol
from scripts import run_phase2e as orchestrator
from scripts import run_phase2e_gates as gates
from scripts import test_phase2e_build as build_test_fixtures
from scripts import test_run_phase2e_gates as gate_test_fixtures


class OuterOrchestratorTests(unittest.TestCase):
    REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
    CANDIDATE = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY,
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    def test_global_index_paths_are_fixed_repository_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            self.assertEqual(
                orchestrator.campaign_index_paths(repository),
                (
                    repository / orchestrator.INDEX_PATH,
                    repository / orchestrator.INDEX_LOCK_PATH,
                ),
            )
        parser = orchestrator.build_parser()
        for command in ("start", "rerun-invalid-lane", "continue", "record-index"):
            option_strings = {
                option
                for action in parser._subparsers._group_actions[0]
                .choices[command]
                ._actions
                for option in action.option_strings
            }
            self.assertNotIn("--index", option_strings)
            self.assertNotIn("--index-lock", option_strings)

    def test_first_index_create_compare_and_reserve_share_one_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            (repository / "docs" / "worklogs").mkdir(parents=True)
            root = repository / "docs" / "worklogs" / "root"
            lock_held = False
            root_lock_held = False
            events = []

            @contextmanager
            def observed_lock(path):
                nonlocal lock_held, root_lock_held
                is_index = path == repository / orchestrator.INDEX_LOCK_PATH
                if is_index:
                    self.assertFalse(lock_held)
                    lock_held = True
                    events.append("index-lock-enter")
                else:
                    self.assertEqual(path, root / ".orchestrator.lock")
                    self.assertTrue(lock_held)
                    self.assertFalse(root_lock_held)
                    root_lock_held = True
                    events.append("root-lock-enter")
                try:
                    yield 1
                finally:
                    if is_index:
                        self.assertFalse(root_lock_held)
                        events.append("index-lock-exit")
                        lock_held = False
                    else:
                        events.append("root-lock-exit")
                        root_lock_held = False

            def compare(_repository, index_path, *, allow_absent):
                self.assertTrue(lock_held)
                self.assertEqual(index_path, repository / orchestrator.INDEX_PATH)
                self.assertTrue(allow_absent)
                events.append("remote-compare")

            @contextmanager
            def observed_root_lock(_descriptor, name):
                nonlocal root_lock_held
                self.assertEqual(name, ".orchestrator.lock")
                self.assertTrue(lock_held)
                root_lock_held = True
                events.append("root-lock-enter")
                try:
                    yield 2
                finally:
                    events.append("root-lock-exit")
                    root_lock_held = False

            with mock.patch.object(
                orchestrator, "exclusive_lock", side_effect=observed_lock
            ), mock.patch.object(
                orchestrator, "exclusive_lock_at", side_effect=observed_root_lock
            ), mock.patch.object(
                orchestrator, "require_remote_index", side_effect=compare
            ), mock.patch.object(
                orchestrator, "seal_abandoned_root"
            ):
                code = orchestrator.initialize_campaign(
                    repository=repository,
                    root=root,
                    reservation_id="r1",
                    candidate_sha=self.CANDIDATE,
                    candidate_tree_sha256="c" * 64,
                    experiment_identity_digest="d" * 64,
                    campaign_identity_digest="e" * 64,
                )
            self.assertEqual(code, 0)
            self.assertEqual(
                events,
                [
                    "index-lock-enter",
                    "remote-compare",
                    "root-lock-enter",
                    "root-lock-exit",
                    "index-lock-exit",
                ],
            )
            index = orchestrator._read_index(repository / orchestrator.INDEX_PATH)
            self.assertEqual(orchestrator.index_state(index), "ACTIVE")

    def test_lock_rejects_symlink_ancestor_leaf_and_special_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory).resolve()
            real = base / "real"
            real.mkdir(mode=0o700)
            alias = base / "alias"
            alias.symlink_to(real, target_is_directory=True)
            target = real / "target"
            target.write_text("untouched", encoding="utf-8")
            leaf = real / "leaf.lock"
            leaf.symlink_to(target)
            fifo = real / "fifo.lock"
            os.mkfifo(fifo)
            for path in (alias / "new.lock", leaf, fifo):
                with self.subTest(path=path), self.assertRaises(protocol.ProtocolError):
                    with orchestrator.exclusive_lock(path):
                        self.fail("untrusted lock acquired")
            self.assertEqual(target.read_text(encoding="utf-8"), "untouched")
            self.assertFalse((real / "new.lock").exists())

    def test_index_symlink_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            index_path = repository / orchestrator.INDEX_PATH
            index_path.parent.mkdir(parents=True)
            target = repository / "foreign-index.json"
            protocol.atomic_write_json(target, orchestrator.new_campaign_index())
            original = target.read_bytes()
            index_path.symlink_to(target)
            with mock.patch.object(orchestrator, "require_remote_index"):
                with self.assertRaises(protocol.ProtocolError):
                    orchestrator.initialize_campaign(
                        repository=repository,
                        root=repository / "docs" / "worklogs" / "root",
                        reservation_id="r1",
                        candidate_sha=self.CANDIDATE,
                        candidate_tree_sha256="c" * 64,
                        experiment_identity_digest="d" * 64,
                        campaign_identity_digest="e" * 64,
                    )
            self.assertTrue(index_path.is_symlink())
            self.assertEqual(target.read_bytes(), original)

    def test_root_replacement_self_seals_held_inode_and_leaves_target_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            (repository / "docs" / "worklogs").mkdir(parents=True)
            root = repository / "docs" / "worklogs" / "root"
            moved = repository / "docs" / "worklogs" / "moved-root"
            target = repository / "foreign"
            target.mkdir(mode=0o700)
            marker = target / "marker"
            marker.write_text("untouched", encoding="utf-8")

            def replace(path):
                path.rename(moved)
                path.symlink_to(target, target_is_directory=True)
                raise OSError("replacement")

            with mock.patch.object(orchestrator, "require_remote_index"):
                code = orchestrator.initialize_campaign(
                    repository=repository,
                    root=root,
                    reservation_id="r1",
                    candidate_sha=self.CANDIDATE,
                    candidate_tree_sha256="c" * 64,
                    experiment_identity_digest="d" * 64,
                    campaign_identity_digest="e" * 64,
                    initializer=replace,
                )
            self.assertEqual(code, 5)
            self.assertTrue((moved / orchestrator.ABANDONMENT_SEAL).is_file())
            self.assertEqual(marker.read_text(encoding="utf-8"), "untouched")
            self.assertFalse((target / orchestrator.ABANDONMENT_SEAL).exists())

    def make_stage_context(self, base: pathlib.Path) -> tuple[pathlib.Path, dict]:
        repository = base / "repository"
        evidence = repository / "docs" / "worklogs" / "evidence"
        scratch = base / "scratch"
        repository.mkdir()
        evidence.parent.mkdir(parents=True)
        scratch.mkdir()
        context = {
            "version": 1,
            "repository": str(repository.resolve()),
            "evidence_root": str(evidence.resolve()),
            "scratch_parent": str(scratch.resolve()),
            "candidate_sha": self.CANDIDATE,
            "candidate_tree_sha256": "c" * 64,
            "reservation_id": "reservation-1",
            "experiment_identity_digest": "b" * 64,
            "command_contract_digest": "0" * 64,
            "path": "/bin",
            "home": str((base / "home").resolve()),
            "cargo_home": str((base / "cargo-home").resolve()),
            "index": str((base / "index.json").resolve()),
            "index_lock": str((base / "index.lock").resolve()),
        }
        context["command_contract_digest"] = orchestrator.stage_context_contract_digest(
            context
        )
        path = base / "context.json"
        protocol.atomic_write_json(path, context)
        return path, context

    def test_stage_context_rejects_digest_tamper_and_reused_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            path, context = self.make_stage_context(base)
            orchestrator.load_stage_context(path)
            context["command_contract_digest"] = "d" * 64
            protocol.atomic_write_json(path, context)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.load_stage_context(path)
            context["command_contract_digest"] = orchestrator.stage_context_contract_digest(
                context
            )
            pathlib.Path(context["scratch_parent"], "foreign").write_text("x")
            protocol.atomic_write_json(path, context)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.load_stage_context(path)

    def test_candidate_provenance_rejects_stale_worktree_head(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                orchestrator, "_git", side_effect=("f" * 40, "")
            ):
                with self.assertRaises(protocol.ProtocolError):
                    orchestrator.validate_candidate_provenance(
                        pathlib.Path(directory),
                        self.CANDIDATE,
                        "c" * 64,
                        "b" * 64,
                        {},
                    )

    def test_stage_context_digest_binds_runtime_and_root_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path, context = self.make_stage_context(pathlib.Path(directory))
            original = context["command_contract_digest"]
            context["path"] = "/usr/bin"
            self.assertNotEqual(
                orchestrator.stage_context_contract_digest(context), original
            )
            protocol.atomic_write_json(path, context)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.load_stage_context(path)

    def test_worker_rejects_replaced_self_rehashed_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path, context = self.make_stage_context(pathlib.Path(directory))
            original_sha = protocol.sha256_file(path)
            context["path"] = "/usr/bin"
            context["command_contract_digest"] = orchestrator.stage_context_contract_digest(
                context
            )
            protocol.atomic_write_json(path, context)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.execute_stage_worker(
                    "dispatch-gates", path, original_sha
                )
    def test_private_stage_worker_dispatches_exact_registered_handler(self):
        self.assertEqual(set(orchestrator.STAGE_HANDLERS), set(orchestrator.STAGE_ORDER))
        with tempfile.TemporaryDirectory() as directory:
            path, _context = self.make_stage_context(pathlib.Path(directory))
            called = []
            handlers = {
                stage: (lambda _context, stage=stage: called.append(stage) or 0)
                for stage in orchestrator.STAGE_ORDER
            }
            with mock.patch.object(orchestrator, "STAGE_HANDLERS", handlers), mock.patch.object(
                orchestrator, "validate_worker_binding"
            ):
                code = orchestrator.main(
                    [
                        "_stage-worker",
                        "--stage",
                        orchestrator.STAGE_ORDER[0],
                        "--context",
                        str(path.resolve()),
                        "--context-sha256",
                        protocol.sha256_file(path),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(called, [orchestrator.STAGE_ORDER[0]])

    def test_private_stage_worker_invokes_real_registered_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            path, context = self.make_stage_context(base)
            with mock.patch(
                "scripts.run_phase2e_gates.run_dispatch_gate_stage"
            ) as stage, mock.patch.object(orchestrator, "validate_worker_binding"):
                self.assertEqual(orchestrator.main(
                    [
                        "_stage-worker",
                        "--stage",
                        "dispatch-gates",
                        "--context",
                        str(path.resolve()),
                        "--context-sha256",
                        protocol.sha256_file(path),
                    ]
                ), 0)
            stage.assert_called_once()

    def test_timing_handlers_use_the_two_canonical_baseline_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            _path, context = self.make_stage_context(pathlib.Path(directory))
            pathlib.Path(context["evidence_root"]).mkdir(parents=True)
            calls = []
            with mock.patch.object(orchestrator, "_next_attempt", return_value=1), mock.patch(
                "scripts.run_phase1_eager_campaign.main",
                side_effect=lambda argv: calls.append(argv) or 0,
            ):
                self.assertEqual(orchestrator._timing(context, "direct-current-main"), 0)
                self.assertEqual(orchestrator._timing(context, "common-lock-normalized"), 0)
            root = pathlib.Path(context["evidence_root"])
            self.assertEqual(
                pathlib.Path(calls[0][calls[0].index("--baseline-build-manifest") + 1]),
                root / orchestrator.build.BUILD_MANIFEST_PATHS["direct-current-main-baseline"],
            )
            self.assertEqual(
                pathlib.Path(calls[1][calls[1].index("--baseline-build-manifest") + 1]),
                root / orchestrator.build.BUILD_MANIFEST_PATHS["common-lock-normalized-baseline"],
            )
    def test_stage_contract_rejects_shell_and_foreign_executable(self):
        for argv in (("/bin/sh", "-c", "true"), ("/usr/bin/true",)):
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_stage_argv(
                    orchestrator.STAGE_ORDER[0],
                    argv,
                    pathlib.Path("/repo/context.json"),
                    "c" * 64,
                )

    def test_progress_journal_never_aliases_terminal_aggregate(self):
        self.assertNotEqual(
            orchestrator.PROGRESS_MANIFEST, orchestrator.AGGREGATE_MANIFEST
        )

    def test_minimal_synthetic_pass_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root)
            (root / "builds" / "locks" / "common.Cargo.lock").unlink(missing_ok=True)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_root(root)

    def test_seal_rejects_ledger_candidate_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root)
            ledger_path = root / "evidence-ledger.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["candidate_sha"] = "f" * 40
            protocol.atomic_write_json(ledger_path, ledger)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.seal_root(
                    root,
                    candidate_sha=self.CANDIDATE,
                    reservation_id="reservation-1",
                    experiment_identity_digest="b" * 64,
                )

    def test_seal_rejects_empty_placeholder_build_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root, seal=False)
            protocol.atomic_write_json(
                root / orchestrator.build.BUILD_MANIFEST_PATHS["candidate"], {}
            )
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_semantic_root(root)

    def test_direct_and_module_entrypoints_expose_every_help_surface(self):
        repository = pathlib.Path(__file__).resolve().parent.parent
        direct = str(repository / "scripts" / "run_phase2e.py")
        commands = (
            None,
            "start",
            "rerun-invalid-lane",
            "continue",
            "validate",
            "record-index",
            "record-preserved",
            "compare-experiment-identity",
        )
        for prefix in (
            (sys.executable, direct),
            (sys.executable, "-m", "scripts.run_phase2e"),
        ):
            for command in commands:
                argv = [*prefix, *(() if command is None else (command,)), "--help"]
                result = subprocess.run(
                    argv,
                    cwd=repository,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, (argv, result.stderr))
                self.assertIn("usage:", result.stdout)

    def make_complete_root(
        self, root: pathlib.Path, *, seal: bool = True,
        normalized_timing_result: str = "PASS",
    ) -> None:
        ledger = protocol.new_ledger(self.CANDIDATE)
        for stage in protocol.STAGE_NAMES:
            for lane in protocol.LANE_NAMES:
                attempt = 1
                artifact = str((root / "attempts" / stage / lane / "1").absolute())
                ledger = protocol.open_attempt(
                    ledger,
                    stage,
                    lane,
                    attempt,
                    artifact_root=artifact if stage == "allocation" else None,
                )
                if stage == "allocation":
                    ledger = protocol.bind_attempt_artifact(
                        ledger,
                        stage,
                        lane,
                        attempt,
                        artifact_root=artifact,
                        artifact_device=1,
                        artifact_inode=1,
                    )
                result = (
                    normalized_timing_result
                    if stage == "timing" and lane == "common-lock-normalized"
                    else "PASS"
                )
                ledger = protocol.close_attempt(ledger, stage, lane, attempt, result)
        protocol.atomic_write_json(root / "evidence-ledger.json", ledger)
        manifest_factory = build_test_fixtures.ManifestTests()
        external_fixture = tempfile.TemporaryDirectory()
        self.addCleanup(external_fixture.cleanup)
        for relative in orchestrator.build.LOCK_PATHS.values():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture lock\n")
        for role, relative in orchestrator.build.BUILD_MANIFEST_PATHS.items():
            fixture_root = pathlib.Path(external_fixture.name) / role
            fixture_root.mkdir(parents=True)
            target = fixture_root / "target"
            target.mkdir()
            executable = target / "bench"
            executable.write_bytes(b"fixture executable")
            executable.chmod(0o700)
            lock_key = (
                "direct" if role == "direct-current-main-baseline" else "common"
            )
            manifest = manifest_factory.manifest(
                role,
                executable,
                target,
                lock_sha256=protocol.sha256_file(
                    root / orchestrator.build.LOCK_PATHS[lock_key]
                ),
            )
            manifest["head"] = self.CANDIDATE
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            protocol.atomic_write_json(root / relative, manifest)
        self.make_gate_collector(root)
        for relative in orchestrator.required_root_paths(ledger):
            if relative == orchestrator.PROGRESS_MANIFEST or relative.startswith(
                "children/"
            ) or relative.startswith(("dispatch-gates/", "characterization/")):
                continue
            path = root / relative
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            protocol.atomic_write_json(path, {})
        scratch_fixture = tempfile.TemporaryDirectory()
        self.addCleanup(scratch_fixture.cleanup)
        probe_patch = mock.patch.object(
            orchestrator.build,
            "validate_allocation_probe_set",
            return_value={role: {} for role in orchestrator.build.PROBE_BUILD_MANIFEST_PATHS},
        )
        allocation_patch = mock.patch(
            "scripts.run_phase2e_allocation_campaign.validate_completed_attempt",
            return_value=0,
        )
        timing_patch = mock.patch(
            "scripts.run_phase1_eager_campaign.validate_retained_attempt",
            return_value=0,
        )
        for patcher in (probe_patch, allocation_patch, timing_patch):
            patcher.start()
            self.addCleanup(patcher.stop)
        identity = {
            "candidate_sha": self.CANDIDATE,
            "candidate_tree_sha256": "c" * 64,
            "reservation_id": "reservation-1",
            "experiment_identity_digest": "b" * 64,
            "command_contract_digest": orchestrator.command_contract_digest(),
            "context_sha256": "e" * 64,
            "repository": str(self.REPOSITORY),
            "evidence_root": str(root.resolve()),
            "scratch_parent": scratch_fixture.name,
            "path": "/bin", "home": "/tmp", "cargo_home": "/tmp",
            "index": str(root / "index.json"),
            "index_lock": str(root / "index.lock"),
            "context_path": "/context.json",
        }
        with mock.patch.object(orchestrator, "seal_root"):
            orchestrator.run_fixed_stages(
                root,
                protocol.runtime_environment(path="/bin", home="/tmp"),
                lambda _stage, _environment: 0,
                identity=identity,
            )
        if seal:
            orchestrator.seal_root(
                root,
                candidate_sha=self.CANDIDATE,
                reservation_id="reservation-1",
                experiment_identity_digest="b" * 64,
            )

    def make_gate_collector(self, root: pathlib.Path) -> None:
        gate_root = root / "gate-collector"
        common = gate_root / orchestrator.build.LOCK_PATHS["common"]
        common.parent.mkdir(parents=True)
        common.write_bytes((root / orchestrator.build.LOCK_PATHS["common"]).read_bytes())
        cpu, ad = gate_test_fixtures.artifacts()
        for short, artifact in (("cpu", cpu), ("ad", ad)):
            path = gate_root / "dispatch-gates" / f"{short}-evidence.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            protocol.atomic_write_json(path, artifact)
            for stream in ("stdout", "stderr"):
                (path.parent / f"{short}-{stream}.log").write_bytes(b"")
        dispatch_builds = {}
        for package in ("tenferro-cpu", "tenferro-ad"):
            relative = orchestrator.build.DISPATCH_BUILD_MANIFEST_PATHS[package]
            path = gate_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            protocol.atomic_write_json(path, {"executable_sha256": "7" * 64})
            dispatch_builds[package] = path
        bench_builds = {}
        for owner, relative in orchestrator.build.CHARACTERIZATION_BUILD_MANIFEST_PATHS.items():
            path = gate_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            protocol.atomic_write_json(path, {"executable_sha256": "8" * 64})
            bench_builds[owner] = path
        source_inventory = gates.validate_source_contract(self.REPOSITORY)
        tree = subprocess.run(
            ("git", "ls-tree", "-r", "-z", "--full-tree", self.CANDIDATE),
            cwd=self.REPOSITORY, check=True, capture_output=True, text=True,
        ).stdout
        tree_digest = hashlib.sha256(tree.encode()).hexdigest()
        protocol_digest = protocol.sha256_file(
            self.REPOSITORY / "scripts/phase2e_protocol.py"
        )
        composed = gates.compose_characterization(cpu, ad)
        gates.attach_hardware_validity(
            composed,
            capacity_provenance={
                "process_allowed_cpus": [0], "process_allowed_capacity": 1,
                "managed_node_cpus": [0], "managed_node_capacity": 1,
                "usable_numa_nodes": 1,
            },
        )
        latency_rows = []
        for row in composed["rows"]:
            if row["surface"] in {"U-O", "U-I"}:
                continue
            row_id = row["key"].replace("/", "__")
            record = {
                "key": row["key"], "row_id": row_id,
                "hardware_skip": row["affinity_hardware_skip"],
                "placement_capacity": row["placement_capacity"],
                "artifacts": {}, "latency_ns": None,
            }
            if row["affinity_hardware_skip"] is None:
                row_root = gate_root / "characterization" / "rows" / row_id
                row_root.mkdir(parents=True)
                estimates = {
                    "mean": {
                        "point_estimate": 1.0,
                        "confidence_interval": {
                            "lower_bound": 0.9, "upper_bound": 1.1,
                            "confidence_level": 0.95,
                        },
                    }
                }
                files = {
                    "stdout": ("stdout.log", b""),
                    "stderr": ("stderr.log", b""),
                    "criterion_estimates": (
                        "estimates.json",
                        (json.dumps(estimates, sort_keys=True, separators=(",", ":")) + "\n").encode(),
                    ),
                    "fixture_affinity": ("affinity.json", b"{}\n"),
                }
                for name, (filename, payload) in files.items():
                    path = row_root / filename
                    path.write_bytes(payload)
                    record["artifacts"][name] = {
                        "path": str(path.resolve()),
                        "sha256": protocol.sha256_file(path),
                    }
                record["latency_ns"] = {
                    "point_estimate": 1.0, "lower_bound": 0.9,
                    "upper_bound": 1.1, "confidence_level": 0.95,
                }
            latency_rows.append(record)
        common_fields = {
            "validity_state": "PASS", "candidate": self.CANDIDATE,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": protocol_digest,
            "candidate_tree_sha256": tree_digest,
            "source_inventory": source_inventory,
            "common_lock_sha256": protocol.sha256_file(common),
        }
        dispatch_records = {}
        for package, short in (("tenferro-cpu", "cpu"), ("tenferro-ad", "ad")):
            artifact = gate_root / "dispatch-gates" / f"{short}-evidence.json"
            dispatch_records[short] = {
                "artifact": str(artifact.resolve()),
                "sha256": protocol.sha256_file(artifact),
                "stdout": {
                    "path": str((artifact.parent / f"{short}-stdout.log").resolve()),
                    "sha256": protocol.sha256_file(artifact.parent / f"{short}-stdout.log"),
                },
                "stderr": {
                    "path": str((artifact.parent / f"{short}-stderr.log").resolve()),
                    "sha256": protocol.sha256_file(artifact.parent / f"{short}-stderr.log"),
                },
                "build_manifest": {
                    "path": str(dispatch_builds[package].resolve()),
                    "sha256": protocol.sha256_file(dispatch_builds[package]),
                },
                "executable_sha256": "7" * 64,
            }
        protocol.atomic_write_json(
            gate_root / "dispatch-gates/manifest.json",
            {**common_fields, "row_count": 47, **dispatch_records},
        )
        composed.update(
            {
                **common_fields,
                "bench_executable_sha256": {"cpu": "8" * 64, "ad": "8" * 64},
                "latency_row_count": 45,
                "latency_rows": latency_rows,
                "bench_build_manifests": {
                    owner: {
                        "path": str(path.resolve()),
                        "sha256": protocol.sha256_file(path),
                    }
                    for owner, path in bench_builds.items()
                },
            }
        )
        protocol.atomic_write_json(
            gate_root / "characterization/manifest.json", composed
        )

    def test_full_fake_stage_sequence_seals_and_validates_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root, seal=False)
            orchestrator.seal_root(
                root,
                candidate_sha=self.CANDIDATE,
                reservation_id="reservation-1",
                experiment_identity_digest="b" * 64,
            )
            self.assertTrue((root / orchestrator.AGGREGATE_MANIFEST).is_file())
            self.assertEqual(orchestrator.validate_root(root), "PASS")

    def test_aggregate_worker_maps_normalized_timing_fail_to_exit_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(
                root, seal=False, normalized_timing_result="FAIL"
            )
            self.assertEqual(
                orchestrator._aggregate_validation({"evidence_root": str(root)}),
                3,
            )

    def test_normalized_timing_fail_seals_as_valid_fail_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(
                root, seal=False, normalized_timing_result="FAIL"
            )
            orchestrator.seal_root(
                root,
                candidate_sha=self.CANDIDATE,
                reservation_id="reservation-1",
                experiment_identity_digest="b" * 64,
            )
            self.assertEqual(orchestrator.validate_root(root), "FAIL")

    def test_parent_seals_and_returns_terminal_aggregate_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            identity = {
                "candidate_sha": self.CANDIDATE,
                "candidate_tree_sha256": "c" * 64,
                "reservation_id": "reservation-1",
                "experiment_identity_digest": "b" * 64,
                "command_contract_digest": "d" * 64,
                "context_sha256": "e" * 64,
            }
            with mock.patch.object(orchestrator, "seal_root") as seal:
                result = orchestrator.run_fixed_stages(
                    root,
                    protocol.runtime_environment(path="/bin", home="/tmp"),
                    lambda stage, _environment: (
                        3 if stage == "aggregate-validation" else 0
                    ),
                    identity=identity,
                )
            self.assertEqual(result, 3)
            seal.assert_called_once()

    def test_stage_order_is_frozen(self):
        self.assertEqual(
            orchestrator.STAGE_ORDER,
            (
                "timing-builds",
                "probe-builds",
                "allocation/direct-current-main",
                "allocation/common-lock-normalized",
                "dispatch-builds",
                "dispatch-gates",
                "characterization-builds",
                "characterization",
                "timing/direct-current-main",
                "timing/common-lock-normalized",
                "aggregate-validation",
            ),
        )

    def test_root_pass_requires_every_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root)
            self.assertEqual(orchestrator.validate_root(root), "PASS")
            manifest_path = root / "gate-collector" / "characterization" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["gating_result"] = "FAIL"
            protocol.atomic_write_json(manifest_path, manifest)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_root(root)

    def test_root_level_gate_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root, seal=False)
            shutil.rmtree(root / "gate-collector")
            for component in ("dispatch-gates", "characterization"):
                path = root / component / "manifest.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                protocol.atomic_write_json(
                    path, {"candidate": self.CANDIDATE, "gating_result": "PASS"}
                )
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_semantic_root(root)

    def test_semantic_root_validates_retained_inconclusive_timing_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root, seal=False)
            ledger_path = root / "evidence-ledger.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            attempt = next(
                item
                for item in ledger["attempts"]
                if item["stage"] == "timing"
                and item["lane"] == "common-lock-normalized"
            )
            attempt.update(
                state="INCONCLUSIVE",
                validity_state="INCONCLUSIVE",
                statistical_result=None,
            )
            lane = next(
                lane
                for stage in ledger["stages"]
                if stage["name"] == "timing"
                for lane in stage["lanes"]
                if lane["name"] == "common-lock-normalized"
            )
            lane.update(state="RETRYABLE", result=None)
            protocol.validate_ledger(ledger)
            protocol.atomic_write_json(ledger_path, ledger)
            progress = json.loads(
                (root / orchestrator.PROGRESS_MANIFEST).read_text(encoding="utf-8")
            )
            with mock.patch.object(
                orchestrator, "validate_progress", return_value=progress
            ), mock.patch(
                "scripts.run_phase1_eager_campaign.validate_retained_attempt",
                return_value=2,
            ) as validate:
                orchestrator.validate_semantic_root(root)
            self.assertTrue(
                any(
                    call.kwargs == {
                        "comparison_kind": "common-lock-normalized",
                        "attempt_id": 1,
                    }
                    for call in validate.call_args_list
                )
            )

    def test_manifest_hashes_every_normative_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root)
            manifest = json.loads(
                (root / orchestrator.AGGREGATE_MANIFEST).read_text(encoding="utf-8")
            )
            expected = {
                "evidence-ledger.json",
                "gate-collector/dispatch-gates/manifest.json",
                "gate-collector/characterization/manifest.json",
            }
            self.assertTrue(expected.issubset(manifest["inventory"]))
            (root / "gate-collector" / "dispatch-gates" / "manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_root(root)

    def test_experiment_identity_ignores_only_worklogs(self):
        contract = {"protocol": 2, "thresholds": {"ratio": 1.03}}
        inventory = (
            ("100644", "src/lib.rs", "a" * 40),
            ("100644", "docs/worklogs/note.md", "b" * 40),
        )
        changed_note = (inventory[0], ("100644", "docs/worklogs/note.md", "c" * 40))
        self.assertEqual(
            orchestrator.experiment_identity_digest(inventory, contract),
            orchestrator.experiment_identity_digest(changed_note, contract),
        )
        changed_source = (("100755", "src/lib.rs", "a" * 40), inventory[1])
        self.assertNotEqual(
            orchestrator.experiment_identity_digest(inventory, contract),
            orchestrator.experiment_identity_digest(changed_source, contract),
        )

    def test_index_lifecycle_blocks_until_preserved(self):
        index = orchestrator.new_campaign_index()
        active = orchestrator.record_active(
            index,
            reservation_id="r1",
            candidate_sha=self.CANDIDATE,
            candidate_tree_sha256="c" * 64,
            root="docs/worklogs/root-1",
            experiment_identity_digest="d" * 64,
            campaign_identity_digest="e" * 64,
        )
        with self.assertRaises(protocol.ProtocolError):
            orchestrator.record_active(
                active,
                reservation_id="r2",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root="docs/worklogs/root-2",
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
        pending = orchestrator.record_terminal(
            active,
            reservation_id="r1",
            status="PASS",
            root_digest="f" * 64,
        )
        self.assertEqual(orchestrator.index_state(pending), "PENDING_PRESERVATION")
        preserved = orchestrator.record_preserved_event(
            pending,
            reservation_id="r1",
            preservation_commit="1" * 40,
            issue_url=(
                "https://github.com/tensor4all/tenferro-rs/issues/1436"
                "#issuecomment-1"
            ),
        )
        self.assertEqual(orchestrator.index_state(preserved), "PRESERVED")
        self.assertEqual(preserved["current_evidence_root"], "docs/worklogs/root-1")

    def test_only_preserved_validity_inconclusive_allows_retry(self):
        index = orchestrator.new_campaign_index()
        active = orchestrator.record_active(
            index,
            reservation_id="r1",
            candidate_sha=self.CANDIDATE,
            candidate_tree_sha256="c" * 64,
            root="docs/worklogs/root-1",
            experiment_identity_digest="d" * 64,
            campaign_identity_digest="e" * 64,
        )
        pending = orchestrator.record_terminal(
            active,
            reservation_id="r1",
            status="VALIDITY_INCONCLUSIVE",
            root_digest="f" * 64,
        )
        preserved = orchestrator.record_preserved_event(
            pending,
            reservation_id="r1",
            preservation_commit="1" * 40,
            issue_url=(
                "https://github.com/tensor4all/tenferro-rs/issues/1436"
                "#issuecomment-1"
            ),
        )
        retry = orchestrator.record_active(
            preserved,
            reservation_id="r2",
            candidate_sha=self.CANDIDATE,
            candidate_tree_sha256="c" * 64,
            root="docs/worklogs/root-2",
            experiment_identity_digest="d" * 64,
            campaign_identity_digest="e" * 64,
        )
        self.assertEqual(orchestrator.index_state(retry), "ACTIVE")

    def test_preserved_pass_closes_identity_and_owns_current_root(self):
        index = orchestrator.new_campaign_index()
        active = orchestrator.record_active(
            index,
            reservation_id="r1",
            candidate_sha=self.CANDIDATE,
            candidate_tree_sha256="c" * 64,
            root="docs/worklogs/root-1",
            experiment_identity_digest="d" * 64,
            campaign_identity_digest="e" * 64,
        )
        pending = orchestrator.record_terminal(
            active, reservation_id="r1", status="PASS", root_digest="f" * 64
        )
        preserved = orchestrator.record_preserved_event(
            pending,
            reservation_id="r1",
            preservation_commit="1" * 40,
            issue_url=(
                "https://github.com/tensor4all/tenferro-rs/issues/1436"
                "#issuecomment-1"
            ),
        )
        forged = {**preserved, "current_evidence_root": "docs/worklogs/forged"}
        with self.assertRaises(protocol.ProtocolError):
            orchestrator.index_state(forged)
        with self.assertRaises(protocol.ProtocolError):
            orchestrator.record_active(
                preserved,
                reservation_id="r2",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root="docs/worklogs/root-2",
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )

    def test_abandonment_seal_rejects_symlink_and_hashes_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "partial.log").write_text("partial", encoding="utf-8")
            (root / "unique.tmp").write_text("tmp", encoding="utf-8")
            seal = orchestrator.seal_abandoned_root(root)
            self.assertEqual(
                set(seal["inventory"]), {"partial.log", "unique.tmp"}
            )
            self.assertEqual(orchestrator.validate_root(root), "ABANDONED")
            os.symlink("partial.log", root / "bad-link")
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.seal_abandoned_root(root)

    def test_index_updates_are_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            index_path = repository / orchestrator.INDEX_PATH
            index_path.parent.mkdir(parents=True)
            protocol.atomic_write_json(index_path, orchestrator.new_campaign_index())
            failures = []

            def update(number: int) -> None:
                try:
                    orchestrator.mutate_index(
                        repository,
                        lambda value: {
                            **value,
                            "audit": sorted([*value.get("audit", []), number]),
                        },
                    )
                except BaseException as error:
                    failures.append(error)

            threads = [
                threading.Thread(target=update, args=(number,))
                for number in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertFalse(failures)
            actual = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(actual["audit"], list(range(8)))

    def test_gitignore_has_only_the_exact_normative_lock(self):
        repository = pathlib.Path(__file__).resolve().parent.parent
        matches = [
            line
            for line in (repository / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if "phase2e-index.lock" in line
        ]
        self.assertEqual(matches, ["docs/worklogs/.phase2e-index.lock"])

    def test_fixed_stage_runner_stops_and_checkpoints_after_each_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            calls = []

            def runner(stage, environment):
                calls.append((stage, dict(environment)))
                return 3 if stage == STAGE_FAILURE else 0

            STAGE_FAILURE = orchestrator.STAGE_ORDER[3]
            environment = protocol.runtime_environment(path="/bin", home="/tmp")
            code = orchestrator.run_fixed_stages(
                root, environment, runner
            )
            self.assertEqual(code, 3)
            self.assertEqual(
                [stage for stage, _ in calls],
                list(orchestrator.STAGE_ORDER[:4]),
            )
            progress = json.loads(
                (root / orchestrator.PROGRESS_MANIFEST).read_text(encoding="utf-8")
            )
            self.assertEqual(len(progress["children"]), 4)
            self.assertTrue(all(environment == calls[0][1] for _, environment in calls))

    def test_rerun_retains_invalid_attempt_and_runs_only_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            environment = protocol.runtime_environment(path="/bin", home="/tmp")
            failure = orchestrator.STAGE_ORDER[1]
            orchestrator.run_fixed_stages(
                root,
                environment,
                lambda stage, _environment: 2 if stage == failure else 0,
            )
            calls = []
            code = orchestrator.rerun_invalid_stage(
                root,
                environment,
                lambda stage, _environment: calls.append(stage) or 0,
            )
            self.assertEqual(code, 0)
            self.assertEqual(calls, [failure])
            progress = orchestrator.validate_progress(root)
            self.assertEqual(
                [child["stage"] for child in progress["children"]],
                [orchestrator.STAGE_ORDER[0], failure, failure],
            )

    def test_rerun_rejects_mutated_durable_child_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            environment = protocol.runtime_environment(path="/bin", home="/tmp")
            orchestrator.run_fixed_stages(
                root, environment, lambda _stage, _environment: 2
            )
            record_path = root / "children" / "01-timing-builds.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["exit_code"] = 0
            protocol.atomic_write_json(record_path, record)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.rerun_invalid_stage(
                    root, environment, lambda _stage, _environment: 0
                )

    def test_resume_identity_mismatch_leaves_evidence_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            _path, context = self.make_stage_context(base)
            root = pathlib.Path(context["evidence_root"])
            root.mkdir(parents=True)
            protocol.atomic_write_json(root / "owned.json", {"value": 1})
            before = protocol.regular_file_inventory(root)
            context_sha = "e" * 64
            active = {
                **{name: context[name] for name in (
                    "candidate_sha", "candidate_tree_sha256",
                    "experiment_identity_digest", "command_contract_digest",
                )},
                "reservation_id": context["reservation_id"],
                "context_sha256": context_sha,
            }
            progress = {**context, "context_sha256": context_sha}
            changed = {**context, "candidate_sha": "f" * 40}
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_resume_identity(
                    active, changed, context_sha, progress
                )
            self.assertEqual(protocol.regular_file_inventory(root), before)

    def test_initialization_failure_self_seals_and_records_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory).resolve()
            (base / "docs" / "worklogs").mkdir(parents=True)
            index_path = base / orchestrator.INDEX_PATH
            root = base / "docs" / "worklogs" / "root"
            def fail(_root):
                raise OSError("boom")

            with mock.patch.object(orchestrator, "require_remote_index"):
                code = orchestrator.initialize_campaign(
                    repository=base,
                    root=root,
                    reservation_id="r1",
                    candidate_sha=self.CANDIDATE,
                    candidate_tree_sha256="c" * 64,
                    experiment_identity_digest="d" * 64,
                    campaign_identity_digest="e" * 64,
                    initializer=fail,
                )
            self.assertEqual(code, 5)
            self.assertTrue((root / orchestrator.ABANDONMENT_SEAL).is_file())
            self.assertEqual(
                orchestrator.index_state(orchestrator._read_index(index_path)),
                "PENDING_PRESERVATION",
            )

    def test_git_inventory_requires_exact_root_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root)
            expected = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            orchestrator.validate_git_blob_inventory(root, expected)
            missing = dict(expected)
            missing.pop(next(iter(missing)))
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_git_blob_inventory(root, missing)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_git_blob_inventory(
                    root, {**expected, "extra": b"foreign"}
                )

    def test_preservation_comment_binds_commit_root_candidate_and_status(self):
        text = " ".join(
            ["1" * 40, "docs/worklogs/root-1", self.CANDIDATE, "PASS"]
        )
        orchestrator.validate_preservation_comment(
            "https://github.com/tensor4all/tenferro-rs/issues/1436#issuecomment-9",
            text,
            preservation_commit="1" * 40,
            root="docs/worklogs/root-1",
            candidate_sha=self.CANDIDATE,
            status="PASS",
        )
        with self.assertRaises(protocol.ProtocolError):
            orchestrator.validate_preservation_comment(
                "https://github.com/tensor4all/tenferro-rs/issues/1435#issuecomment-9",
                text,
                preservation_commit="1" * 40,
                root="docs/worklogs/root-1",
                candidate_sha=self.CANDIDATE,
                status="PASS",
            )


if __name__ == "__main__":
    unittest.main()
