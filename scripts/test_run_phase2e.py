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

    @staticmethod
    def write_process_journal(root: pathlib.Path, *, extra: dict | None = None) -> None:
        entry = {
            "ordinal": 1,
            "stage": orchestrator.STAGE_ORDER[0],
            "argv": ["worker"],
            "pid": 999999,
            "pgid": 999999,
            "start_ticks": 1,
            "state": "EXITED",
            "exit_code": 0,
            "signals": [],
            "reaped": True,
        }
        entry.update(extra or {})
        protocol.atomic_write_json(
            root / orchestrator.PROCESS_JOURNAL,
            {"version": 1, "entries": [entry]},
        )

    def init_git_repository(self, base: pathlib.Path) -> pathlib.Path:
        repository = base / "repository"
        repository.mkdir()
        subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
        subprocess.run(
            ("git", "config", "user.email", "phase2e@example.invalid"),
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ("git", "config", "user.name", "Phase 2E Test"),
            cwd=repository,
            check=True,
        )
        return repository

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

            def compare(_repository, index_path, *, allow_absent, transaction):
                self.assertTrue(lock_held)
                self.assertEqual(index_path, repository / orchestrator.INDEX_PATH)
                self.assertTrue(allow_absent)
                self.assertIsInstance(
                    transaction, orchestrator.CampaignIndexTransaction
                )
                events.append("remote-compare")

            @contextmanager
            def observed_root_lock(_descriptor, name):
                nonlocal lock_held, root_lock_held
                if name == pathlib.Path(orchestrator.INDEX_LOCK_PATH).name:
                    self.assertFalse(lock_held)
                    lock_held = True
                    events.append("index-lock-enter")
                    try:
                        yield 1
                    finally:
                        self.assertFalse(root_lock_held)
                        events.append("index-lock-exit")
                        lock_held = False
                else:
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

    def test_index_transaction_rejects_worklog_parent_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            worklogs = repository / "docs" / "worklogs"
            worklogs.mkdir(parents=True)
            displaced = repository / "docs" / "displaced-worklogs"
            replacement = repository / "docs" / "worklogs"

            def replace_parent(*_args, **_kwargs):
                worklogs.rename(displaced)
                replacement.mkdir()

            with mock.patch.object(
                orchestrator, "require_remote_index", side_effect=replace_parent
            ):
                with self.assertRaises(protocol.ProtocolError):
                    orchestrator.initialize_campaign(
                        repository=repository,
                        root=worklogs / "root",
                        reservation_id="r1",
                        candidate_sha=self.CANDIDATE,
                        candidate_tree_sha256="c" * 64,
                        experiment_identity_digest="d" * 64,
                        campaign_identity_digest="e" * 64,
                    )
            self.assertFalse((replacement / "phase2e-index.json").exists())
            self.assertFalse((replacement / "root").exists())

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
            path, context = self.make_stage_context(pathlib.Path(directory))
            pathlib.Path(context["evidence_root"]).mkdir(mode=0o700)
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
            pathlib.Path(context["evidence_root"]).mkdir(mode=0o700)
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

    def test_terminal_event_carries_complete_active_and_root_identity(self):
        active = orchestrator.record_active(
            orchestrator.new_campaign_index(),
            reservation_id="r1",
            candidate_sha=self.CANDIDATE,
            candidate_tree_sha256="c" * 64,
            root="/canonical/root",
            experiment_identity_digest="d" * 64,
            campaign_identity_digest="e" * 64,
            command_digest="f" * 64,
            context_sha256="1" * 64,
        )
        pending = orchestrator.record_terminal(
            active,
            reservation_id="r1",
            status="PASS",
            root_digest="2" * 64,
            ledger_sha256="3" * 64,
        )
        terminal = pending["events"][-1]
        self.assertEqual(
            terminal,
            {
                "ordinal": 2,
                "event": "TERMINAL",
                "reservation_id": "r1",
                "status": "PASS",
                "root_digest": "2" * 64,
                "ledger_sha256": "3" * 64,
                "candidate_sha": self.CANDIDATE,
                "candidate_tree_sha256": "c" * 64,
                "root": "/canonical/root",
                "experiment_identity_digest": "d" * 64,
                "campaign_identity_digest": "e" * 64,
                "command_contract_digest": "f" * 64,
                "context_sha256": "1" * 64,
            },
        )

    def test_record_index_rejects_foreign_root_before_sealing(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            worklogs = repository / "docs" / "worklogs"
            worklogs.mkdir(parents=True)
            active_root = worklogs / "active"
            foreign_root = worklogs / "foreign"
            active_root.mkdir(mode=0o700)
            foreign_root.mkdir(mode=0o700)
            index = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="r1",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root=str(active_root),
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
            protocol.atomic_write_json(repository / orchestrator.INDEX_PATH, index)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.record_index_root(
                    repository=repository,
                    root=foreign_root,
                    reservation_id="r1",
                    abandoned=True,
                )
            self.assertFalse((foreign_root / orchestrator.ABANDONMENT_SEAL).exists())
            self.assertEqual(
                orchestrator.index_state(
                    orchestrator._read_index(repository / orchestrator.INDEX_PATH)
                ),
                "ACTIVE",
            )

    def test_record_index_exact_replay_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            worklogs = repository / "docs" / "worklogs"
            worklogs.mkdir(parents=True)
            root = worklogs / "active"
            root.mkdir(mode=0o700)
            self.write_process_journal(root)
            index = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="r1",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root=str(root),
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
            protocol.atomic_write_json(repository / orchestrator.INDEX_PATH, index)
            with mock.patch.object(orchestrator.os, "killpg", side_effect=ProcessLookupError):
                first = orchestrator.record_index_root(
                    repository=repository,
                    root=root,
                    reservation_id="r1",
                    abandoned=True,
                )
                second = orchestrator.record_index_root(
                    repository=repository,
                    root=root,
                    reservation_id="r1",
                    abandoned=True,
                )
            seal = json.loads(
                (root / orchestrator.ABANDONMENT_SEAL).read_text(encoding="utf-8")
            )
            self.assertEqual(seal["abandonment_kind"], "UNCAUGHT_INTERRUPTION")
            self.assertEqual(second, first)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.record_index_root(
                    repository=repository,
                    root=root,
                    reservation_id="changed",
                    abandoned=True,
                )

    def test_abandonment_requires_complete_process_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            root = repository / "docs" / "worklogs" / "root"
            root.mkdir(parents=True, mode=0o700)
            index = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="r1",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root=str(root),
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
            protocol.atomic_write_json(repository / orchestrator.INDEX_PATH, index)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.record_index_root(
                    repository=repository,
                    root=root,
                    reservation_id="r1",
                    abandoned=True,
                )
            self.write_process_journal(root, extra={"foreign": True})
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.record_index_root(
                    repository=repository,
                    root=root,
                    reservation_id="r1",
                    abandoned=True,
                )

    def test_record_index_rejects_self_consistent_root_for_foreign_active(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            root = repository / "docs" / "worklogs" / "root"
            root.mkdir(parents=True)
            self.make_complete_root(root)
            index = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="reservation-1",
                candidate_sha="f" * 40,
                candidate_tree_sha256="c" * 64,
                root=str(root),
                experiment_identity_digest="b" * 64,
                campaign_identity_digest="e" * 64,
                command_digest=orchestrator.command_contract_digest(),
                context_sha256="e" * 64,
            )
            protocol.atomic_write_json(repository / orchestrator.INDEX_PATH, index)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.record_index_root(
                    repository=repository,
                    root=root,
                    reservation_id="reservation-1",
                )

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

    def test_parallel_start_has_one_authoritative_active_and_no_root_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            (repository / "docs" / "worklogs").mkdir(parents=True)
            entered = threading.Event()
            release = threading.Event()
            results = []

            def start(number):
                root = repository / "docs" / "worklogs" / f"root-{number}"

                def initialize(_root):
                    entered.set()
                    release.wait(timeout=5)

                try:
                    result = orchestrator.initialize_campaign(
                        repository=repository,
                        root=root,
                        reservation_id=f"r{number}",
                        candidate_sha=self.CANDIDATE,
                        candidate_tree_sha256="c" * 64,
                        experiment_identity_digest=f"{number}" * 64,
                        campaign_identity_digest="e" * 64,
                        initializer=initialize,
                    )
                    results.append((number, result))
                except BaseException as error:
                    results.append((number, error))

            with mock.patch.object(orchestrator, "require_remote_index"):
                first = threading.Thread(target=start, args=(1,))
                second = threading.Thread(target=start, args=(2,))
                first.start()
                self.assertTrue(entered.wait(timeout=5))
                second.start()
                release.set()
                first.join(timeout=5)
                second.join(timeout=5)
            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(sum(result == 0 for _, result in results), 1)
            self.assertEqual(
                sum(isinstance(result, protocol.ProtocolError) for _, result in results),
                1,
            )
            index = orchestrator._read_index(repository / orchestrator.INDEX_PATH)
            self.assertEqual(orchestrator.index_state(index), "ACTIVE")
            winner = next(number for number, result in results if result == 0)
            loser = 1 if winner == 2 else 2
            self.assertFalse(
                (repository / "docs" / "worklogs" / f"root-{loser}").exists()
            )

    def test_parallel_record_index_has_one_transition_and_identical_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            root = repository / "docs" / "worklogs" / "root"
            root.mkdir(parents=True, mode=0o700)
            self.write_process_journal(root)
            active = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="r1",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root=str(root),
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
            protocol.atomic_write_json(repository / orchestrator.INDEX_PATH, active)
            results = []

            def record():
                try:
                    results.append(
                        orchestrator.record_index_root(
                            repository=repository,
                            root=root,
                            reservation_id="r1",
                            abandoned=True,
                        )
                    )
                except BaseException as error:
                    results.append(error)

            with mock.patch.object(orchestrator.os, "killpg", side_effect=ProcessLookupError):
                threads = [threading.Thread(target=record) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertTrue(all(type(result) is dict for result in results))
            self.assertEqual(results[0], results[1])
            self.assertEqual(len(results[0]["events"]), 2)

    def test_active_operation_and_record_follow_index_then_root_without_deadlock(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            root = repository / "docs" / "worklogs" / "root"
            root.mkdir(parents=True, mode=0o700)
            self.write_process_journal(root)
            active = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="r1",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root=str(root),
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
            protocol.atomic_write_json(repository / orchestrator.INDEX_PATH, active)
            entered = threading.Event()
            release = threading.Event()

            def hold_active():
                with orchestrator.active_campaign_lock(
                    repository, root, reservation_id="r1"
                ):
                    entered.set()
                    release.wait(timeout=5)

            holder = threading.Thread(target=hold_active)
            holder.start()
            self.assertTrue(entered.wait(timeout=5))
            with mock.patch.object(orchestrator.os, "killpg", side_effect=ProcessLookupError):
                recorder = threading.Thread(
                    target=lambda: orchestrator.record_index_root(
                        repository=repository,
                        root=root,
                        reservation_id="r1",
                        abandoned=True,
                    )
                )
                recorder.start()
                self.assertTrue(recorder.is_alive())
                release.set()
                holder.join(timeout=5)
                recorder.join(timeout=5)
            self.assertFalse(holder.is_alive() or recorder.is_alive())
            self.assertEqual(
                orchestrator.index_state(
                    orchestrator._read_index(repository / orchestrator.INDEX_PATH)
                ),
                "PENDING_PRESERVATION",
            )

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

    def test_subprocess_is_journaled_before_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )

            class Process:
                pid = 4242
                returncode = 0

                def wait(self, timeout=None):
                    journal = json.loads(
                        (root / orchestrator.PROCESS_JOURNAL).read_text(encoding="utf-8")
                    )
                    self.assert_running = journal["entries"][-1]["state"]
                    return 0

            process = Process()
            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: process,
                process_identity=lambda pid: {"pid": pid, "start_ticks": 99},
                process_group=lambda pid: pid,
            )
            self.assertEqual(
                runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"}), 0
            )
            self.assertEqual(process.assert_running, "RUNNING")
            journal = orchestrator.validate_process_journal(root, require_entries=True)
            self.assertEqual(journal["entries"][-1]["state"], "EXITED")
            self.assertTrue(journal["entries"][-1]["reaped"])

    def test_post_popen_identity_failure_terminates_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )
            signals = []

            class Process:
                pid = 4242
                waits = 0

                def wait(self, timeout=None):
                    self.waits += 1
                    return -15

            process = Process()
            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: process,
                process_identity=lambda _pid: (_ for _ in ()).throw(
                    protocol.ProtocolError("identity failed")
                ),
                kill_process_group=lambda pgid, sig: signals.append((pgid, sig)),
            )
            with self.assertRaises(protocol.ProtocolError):
                runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            self.assertEqual(signals, [(4242, orchestrator.signal.SIGTERM)])
            self.assertEqual(process.waits, 1)

    def test_post_popen_identity_failure_reaps_when_term_loses_process_race(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )

            class Process:
                pid = 4242
                waits = 0

                def wait(self, timeout=None):
                    self.waits += 1
                    return -15

            process = Process()
            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: process,
                process_identity=lambda _pid: (_ for _ in ()).throw(
                    protocol.ProtocolError("identity failed")
                ),
                kill_process_group=lambda _pgid, _sig: (_ for _ in ()).throw(
                    ProcessLookupError("already exited")
                ),
            )
            with self.assertRaisesRegex(protocol.ProtocolError, "identity failed"):
                runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            self.assertEqual(process.waits, 1)

    def test_post_popen_signal_error_preserves_exact_primary_after_reap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )
            primary = protocol.ProtocolError("exact identity failure")

            class Process:
                pid = 4242
                waits = 0

                def wait(self, timeout=None):
                    self.waits += 1
                    return -15

            process = Process()
            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: process,
                process_identity=lambda _pid: (_ for _ in ()).throw(primary),
                kill_process_group=lambda _pgid, _sig: (_ for _ in ()).throw(
                    PermissionError("TERM denied")
                ),
            )
            with self.assertRaises(protocol.ProtocolError) as caught:
                runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            self.assertIs(caught.exception, primary)
            self.assertEqual(str(caught.exception), "exact identity failure")
            self.assertEqual(process.waits, 1)
            self.assertTrue(
                any(
                    "TERM denied" in note
                    for note in getattr(caught.exception, "__notes__", [])
                )
            )

    def test_post_popen_getpgid_failure_terminates_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )
            signals = []

            class Process:
                pid = 4242
                waits = 0

                def wait(self, timeout=None):
                    self.waits += 1
                    return -15

            process = Process()
            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: process,
                process_identity=lambda pid: {"pid": pid, "start_ticks": 99},
                process_group=lambda _pid: (_ for _ in ()).throw(OSError("getpgid")),
                kill_process_group=lambda pgid, sig: signals.append((pgid, sig)),
            )
            with self.assertRaises(OSError):
                runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            self.assertEqual(signals, [(4242, orchestrator.signal.SIGTERM)])
            self.assertEqual(process.waits, 1)

    def test_post_popen_journal_failure_terminates_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )
            signals = []

            class Process:
                pid = 4242
                waits = 0

                def wait(self, timeout=None):
                    self.waits += 1
                    return -15

            process = Process()
            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: process,
                process_identity=lambda pid: {"pid": pid, "start_ticks": 99},
                process_group=lambda pid: pid,
                kill_process_group=lambda pgid, sig: signals.append((pgid, sig)),
            )
            with mock.patch.object(
                orchestrator,
                "_start_process_journal_entry",
                side_effect=protocol.ProtocolError("journal failed"),
            ):
                with self.assertRaises(protocol.ProtocolError):
                    runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            self.assertEqual(signals, [(4242, orchestrator.signal.SIGTERM)])
            self.assertEqual(process.waits, 1)

    def test_runner_root_replacement_writes_nothing_to_replacement_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory).resolve()
            root = base / "root"
            root.mkdir(mode=0o700)
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )
            held = protocol.PreparedRootIdentity(root)
            displaced = base / "displaced"
            signals = []

            class Process:
                pid = 4242
                waits = 0

                def wait(self, timeout=None):
                    self.waits += 1
                    return -15

            process = Process()

            def replace(_pid):
                root.rename(displaced)
                root.mkdir(mode=0o700)
                return {"pid": 4242, "start_ticks": 99}

            try:
                runner = orchestrator._subprocess_stage_runner(
                    pathlib.Path("/context.json"),
                    "e" * 64,
                    self.REPOSITORY,
                    root=root,
                    root_identity=held,
                    process_factory=lambda *args, **kwargs: process,
                    process_identity=replace,
                    process_group=lambda pid: pid,
                    kill_process_group=lambda pgid, sig: signals.append((pgid, sig)),
                )
                with self.assertRaises(protocol.ProtocolError):
                    runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            finally:
                held.close()
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(signals, [(4242, orchestrator.signal.SIGTERM)])
            self.assertEqual(process.waits, 1)

    def test_parent_checkpoint_rejects_root_replacement_without_writing_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory).resolve()
            root = base / "root"
            root.mkdir(mode=0o700)
            displaced = base / "displaced"
            held = protocol.PreparedRootIdentity(root)

            def replace(_stage, _environment):
                root.rename(displaced)
                root.mkdir(mode=0o700)
                return 0

            try:
                with self.assertRaises(protocol.ProtocolError):
                    orchestrator.run_fixed_stages(
                        root,
                        protocol.runtime_environment(path="/bin", home="/tmp"),
                        replace,
                        root_identity=held,
                        _locked=True,
                    )
            finally:
                held.close()
            self.assertEqual(list(root.iterdir()), [])
            self.assertFalse((root / orchestrator.PROGRESS_MANIFEST).exists())

    def test_stage_worker_context_rejects_root_replacement_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory).resolve()
            context_path, context = self.make_stage_context(base)
            root = pathlib.Path(context["evidence_root"])
            root.mkdir(mode=0o700)
            displaced = root.with_name("displaced")
            stage = orchestrator.STAGE_ORDER[0]

            def replace_then_write(guarded_context):
                root.rename(displaced)
                root.mkdir(mode=0o700)
                destination = pathlib.Path(guarded_context["evidence_root"])
                destination.joinpath("injected").write_text("bad", encoding="utf-8")
                return 0

            with mock.patch.object(
                orchestrator, "validate_worker_binding"
            ), mock.patch.dict(
                orchestrator.STAGE_HANDLERS, {stage: replace_then_write}
            ):
                with self.assertRaises(protocol.ProtocolError):
                    orchestrator.execute_stage_worker(
                        stage, context_path, protocol.sha256_file(context_path)
                    )
            self.assertEqual(list(root.iterdir()), [])

    def test_subprocess_timeout_terminates_kills_reaps_and_journals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )
            signals = []

            class Process:
                pid = 4242
                returncode = None
                waits = 0

                def wait(self, timeout=None):
                    self.waits += 1
                    if self.waits < 3:
                        raise subprocess.TimeoutExpired(("worker",), timeout)
                    self.returncode = -9
                    return self.returncode

            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: Process(),
                process_identity=lambda pid: {"pid": pid, "start_ticks": 99},
                process_group=lambda pid: pid,
                kill_process_group=lambda pgid, signal: signals.append((pgid, signal)),
                deadline_seconds=0.01,
                termination_grace_seconds=0.01,
            )
            with self.assertRaises(protocol.ProtocolError):
                runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            self.assertEqual(
                signals,
                [(4242, orchestrator.signal.SIGTERM), (4242, orchestrator.signal.SIGKILL)],
            )
            journal = orchestrator.validate_process_journal(root, require_entries=True)
            entry = journal["entries"][-1]
            self.assertEqual(entry["state"], "TERMINATED")
            self.assertEqual(entry["signals"], ["TERM", "KILL"])
            self.assertTrue(entry["reaped"])

    def test_keyboard_interrupt_terminates_reaps_and_is_re_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )

            class Process:
                pid = 4242
                calls = 0

                def wait(self, timeout=None):
                    self.calls += 1
                    if self.calls == 1:
                        raise KeyboardInterrupt()
                    return -15

            runner = orchestrator._subprocess_stage_runner(
                pathlib.Path("/context.json"),
                "e" * 64,
                self.REPOSITORY,
                root=root,
                process_factory=lambda *args, **kwargs: Process(),
                process_identity=lambda pid: {"pid": pid, "start_ticks": 99},
                process_group=lambda pid: pid,
                kill_process_group=lambda _pgid, _signal: None,
            )
            with self.assertRaises(KeyboardInterrupt):
                runner(orchestrator.STAGE_ORDER[0], {"PATH": "/bin"})
            entry = orchestrator.validate_process_journal(
                root, require_entries=True
            )["entries"][-1]
            self.assertEqual(entry["state"], "TERMINATED")
            self.assertTrue(entry["reaped"])

    def test_abandonment_cli_has_no_boolean_or_process_group_bypass(self):
        parser = orchestrator.build_parser()
        options = {
            option
            for action in parser._subparsers._group_actions[0]
            .choices["record-index"]
            ._actions
            for option in action.option_strings
        }
        self.assertNotIn("--confirm-no-live-processes", options)
        self.assertNotIn("--process-group", options)

    def test_spawn_journal_becomes_normative_root_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root)
            self.write_process_journal(root)
            ledger = json.loads(
                (root / "evidence-ledger.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                orchestrator.PROCESS_JOURNAL,
                orchestrator._canonical_root_paths(root, ledger),
            )

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
            seal = json.loads(
                (root / orchestrator.ABANDONMENT_SEAL).read_text(encoding="utf-8")
            )
            self.assertEqual(seal["abandonment_kind"], "INITIALIZATION_FAILURE")
            self.assertEqual(
                orchestrator.index_state(orchestrator._read_index(index_path)),
                "PENDING_PRESERVATION",
            )

    def test_initialization_durably_creates_empty_process_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            (repository / "docs" / "worklogs").mkdir(parents=True)
            root = repository / "docs" / "worklogs" / "root"
            with mock.patch.object(orchestrator, "require_remote_index"):
                self.assertEqual(
                    orchestrator.initialize_campaign(
                        repository=repository,
                        root=root,
                        reservation_id="r1",
                        candidate_sha=self.CANDIDATE,
                        candidate_tree_sha256="c" * 64,
                        experiment_identity_digest="d" * 64,
                        campaign_identity_digest="e" * 64,
                    ),
                    0,
                )
            self.assertEqual(
                orchestrator.validate_process_journal(root),
                {"version": 1, "entries": []},
            )

    def test_manual_abandonment_accepts_durable_empty_process_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            root = repository / "docs" / "worklogs" / "root"
            root.mkdir(parents=True, mode=0o700)
            protocol.atomic_write_json(
                root / orchestrator.PROCESS_JOURNAL,
                {"version": 1, "entries": []},
            )
            active = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="r1",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root=str(root),
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
            protocol.atomic_write_json(repository / orchestrator.INDEX_PATH, active)
            updated = orchestrator.record_index_root(
                repository=repository,
                root=root,
                reservation_id="r1",
                abandoned=True,
            )
            self.assertEqual(orchestrator.index_state(updated), "PENDING_PRESERVATION")

    def test_committed_active_write_failure_self_seals_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory).resolve()
            (repository / "docs" / "worklogs").mkdir(parents=True)
            root = repository / "docs" / "worklogs" / "root"
            real_write = orchestrator.CampaignIndexTransaction.write

            def fail_after_active(transaction, payload):
                real_write(transaction, payload)
                events = payload.get("events", [])
                if events and events[-1].get("event") == "ACTIVE":
                    raise protocol.AtomicWriteDurabilityError(
                        "ACTIVE committed but parent fsync failed"
                    )

            with mock.patch.object(orchestrator, "require_remote_index"), mock.patch.object(
                orchestrator.CampaignIndexTransaction,
                "write",
                autospec=True,
                side_effect=fail_after_active,
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
            self.assertEqual(code, 5)
            index = orchestrator._read_index(repository / orchestrator.INDEX_PATH)
            self.assertEqual(orchestrator.index_state(index), "PENDING_PRESERVATION")
            self.assertEqual(index["events"][-1]["status"], "ABANDONED")

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

    def test_git_selector_reconstructs_abandoned_ignored_inventory_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.init_git_repository(pathlib.Path(directory))
            root = repository / "docs" / "worklogs" / "root"
            root.mkdir(parents=True, mode=0o700)
            (root / "Cargo.lock").write_text("lock\n", encoding="utf-8")
            (root / "partial.log").write_text("partial\n", encoding="utf-8")
            seal = orchestrator.seal_abandoned_root(root)
            subprocess.run(
                ("git", "add", "-f", "--", root.relative_to(repository)),
                cwd=repository,
                check=True,
            )
            terminal = {
                "status": "ABANDONED",
                "root_digest": protocol.sha256_json(seal),
                "ledger_sha256": "0" * 64,
            }
            orchestrator.validate_git_selector(
                repository, root, selector=":", terminal_event=terminal
            )
            (root / "alias").symlink_to("Cargo.lock")
            subprocess.run(
                ("git", "add", "-f", "--", root.relative_to(repository)),
                cwd=repository,
                check=True,
            )
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_git_selector(repository, root, selector=":")

    def test_preservation_objects_require_exact_index_worklog_and_root(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.init_git_repository(pathlib.Path(directory))
            worklogs = repository / "docs" / "worklogs"
            root = worklogs / "root"
            root.mkdir(parents=True, mode=0o700)
            (root / "partial.log").write_text("partial\n", encoding="utf-8")
            (root / ".orchestrator.lock").touch(mode=0o600)
            seal = orchestrator.seal_abandoned_root(root)
            index_path = repository / orchestrator.INDEX_PATH
            protocol.atomic_write_json(index_path, orchestrator.new_campaign_index())
            worklog = worklogs / "campaign.md"
            worklog.write_text("campaign\n", encoding="utf-8")
            subprocess.run(
                (
                    "git",
                    "add",
                    "-f",
                    "--",
                    root.relative_to(repository),
                    index_path.relative_to(repository),
                    worklog.relative_to(repository),
                ),
                cwd=repository,
                check=True,
            )
            orchestrator.validate_preservation_objects(
                repository,
                selector=":",
                root=root,
                index_path=index_path,
                worklog=worklog,
                terminal_event={
                    "status": "ABANDONED",
                    "root_digest": protocol.sha256_json(seal),
                    "ledger_sha256": "0" * 64,
                },
            )
            subprocess.run(
                ("git", "rm", "--cached", "-q", worklog.relative_to(repository)),
                cwd=repository,
                check=True,
            )
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_preservation_objects(
                    repository,
                    selector=":",
                    root=root,
                    index_path=index_path,
                    worklog=worklog,
                    terminal_event={
                        "status": "ABANDONED",
                        "root_digest": protocol.sha256_json(seal),
                        "ledger_sha256": "0" * 64,
                    },
                )

    def test_comment_proof_rejects_cross_issue_or_fabricated_metadata(self):
        url = (
            "https://github.com/tensor4all/tenferro-rs/issues/1436"
            "#issuecomment-99"
        )
        proof = {
            "id": 99,
            "html_url": url,
            "issue_url": (
                "https://api.github.com/repos/tensor4all/tenferro-rs/issues/1436"
            ),
            "body": " ".join(
                ["1" * 40, "docs/worklogs/root", self.CANDIDATE, "PASS"]
            ),
        }
        orchestrator.validate_preservation_comment_proof(
            url,
            proof,
            preservation_commit="1" * 40,
            root="docs/worklogs/root",
            candidate_sha=self.CANDIDATE,
            status="PASS",
        )
        with self.assertRaises(protocol.ProtocolError):
            orchestrator.validate_preservation_comment_proof(
                url,
                {**proof, "issue_url": proof["issue_url"].replace("1436", "1435")},
                preservation_commit="1" * 40,
                root="docs/worklogs/root",
                candidate_sha=self.CANDIDATE,
                status="PASS",
            )

    def test_origin_url_allows_only_canonical_tensor4all_tenferro_forms(self):
        for url in (
            "https://github.com/tensor4all/tenferro-rs.git",
            "git@github.com:tensor4all/tenferro-rs.git",
            "ssh://git@github.com/tensor4all/tenferro-rs.git",
        ):
            orchestrator.validate_canonical_origin_url(url)
        for url in (
            "https://github.com/tensor4all/tenferro-rs",
            "https://github.com/other/tenferro-rs.git",
            "git@github.com:tensor4all/tenferro-rs",
            "file:///tmp/tenferro-rs.git",
        ):
            with self.subTest(url=url), self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_canonical_origin_url(url)

    def test_record_preserved_uses_commit_objects_and_structured_comment_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.init_git_repository(pathlib.Path(directory)).resolve()
            worklogs = repository / "docs" / "worklogs"
            root = worklogs / "root"
            root.mkdir(parents=True, mode=0o700)
            (root / "partial.log").write_text("partial\n", encoding="utf-8")
            (root / ".orchestrator.lock").touch(mode=0o600)
            seal = orchestrator.seal_abandoned_root(root)
            active = orchestrator.record_active(
                orchestrator.new_campaign_index(),
                reservation_id="r1",
                candidate_sha=self.CANDIDATE,
                candidate_tree_sha256="c" * 64,
                root=str(root),
                experiment_identity_digest="d" * 64,
                campaign_identity_digest="e" * 64,
            )
            pending = orchestrator.record_terminal(
                active,
                reservation_id="r1",
                status="ABANDONED",
                root_digest=protocol.sha256_json(seal),
                ledger_sha256="0" * 64,
            )
            index_path = repository / orchestrator.INDEX_PATH
            protocol.atomic_write_json(index_path, pending)
            worklog = worklogs / "campaign.md"
            worklog.write_text("campaign\n", encoding="utf-8")
            subprocess.run(
                (
                    "git",
                    "add",
                    "-f",
                    "--",
                    root.relative_to(repository),
                    index_path.relative_to(repository),
                    worklog.relative_to(repository),
                ),
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ("git", "commit", "-q", "-m", "preserve"),
                cwd=repository,
                check=True,
            )
            commit = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=repository,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            url = (
                "https://github.com/tensor4all/tenferro-rs/issues/1436"
                "#issuecomment-99"
            )
            proof = {
                "id": 99,
                "html_url": url,
                "issue_url": (
                    "https://api.github.com/repos/tensor4all/tenferro-rs/issues/1436"
                ),
                "body": " ".join(
                    [commit, str(root), self.CANDIDATE, "ABANDONED"]
                ),
            }
            remote_calls = []
            updated = orchestrator.record_preserved(
                repository=repository,
                root=root,
                reservation_id="r1",
                preservation_commit=commit,
                issue_url=url,
                worklog=worklog,
                remote_validator=lambda repo, sha: remote_calls.append((repo, sha)),
                comment_fetcher=lambda _url: proof,
            )
            self.assertEqual(remote_calls, [(repository, commit)])
            self.assertEqual(orchestrator.index_state(updated), "PRESERVED")
            self.assertEqual(
                orchestrator._read_index(index_path),
                updated,
            )
            results = []

            def replay():
                try:
                    results.append(
                        orchestrator.record_preserved(
                            repository=repository,
                            root=root,
                            reservation_id="r1",
                            preservation_commit=commit,
                            issue_url=url,
                            worklog=worklog,
                            remote_validator=lambda _repo, _sha: None,
                            comment_fetcher=lambda _url: proof,
                        )
                    )
                except BaseException as error:
                    results.append(error)

            threads = [threading.Thread(target=replay) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertTrue(all(result == updated for result in results))

            race_results = []
            barrier = threading.Barrier(2)
            next_root = worklogs / "root-2"

            def replay_during_start():
                barrier.wait(timeout=5)
                try:
                    race_results.append(
                        orchestrator.record_preserved(
                            repository=repository,
                            root=root,
                            reservation_id="r1",
                            preservation_commit=commit,
                            issue_url=url,
                            worklog=worklog,
                            remote_validator=lambda _repo, _sha: None,
                            comment_fetcher=lambda _url: proof,
                        )
                    )
                except BaseException as error:
                    race_results.append(error)

            def start_next():
                barrier.wait(timeout=5)
                try:
                    with mock.patch.object(orchestrator, "require_remote_index"):
                        race_results.append(
                            orchestrator.initialize_campaign(
                                repository=repository,
                                root=next_root,
                                reservation_id="r2",
                                candidate_sha=self.CANDIDATE,
                                candidate_tree_sha256="c" * 64,
                                experiment_identity_digest="d" * 64,
                                campaign_identity_digest="e" * 64,
                            )
                        )
                except BaseException as error:
                    race_results.append(error)

            racers = [
                threading.Thread(target=replay_during_start),
                threading.Thread(target=start_next),
            ]
            for racer in racers:
                racer.start()
            for racer in racers:
                racer.join(timeout=5)
            self.assertFalse(any(racer.is_alive() for racer in racers))
            raced_index = orchestrator._read_index(index_path)
            self.assertIn(orchestrator.index_state(raced_index), {"PRESERVED", "ACTIVE"})
            self.assertLessEqual(
                sum(event["event"] == "ACTIVE" for event in raced_index["events"]),
                2,
            )

    def test_preserved_exact_replay_is_idempotent_and_changed_replay_rejected(self):
        active = orchestrator.record_active(
            orchestrator.new_campaign_index(),
            reservation_id="r1",
            candidate_sha=self.CANDIDATE,
            candidate_tree_sha256="c" * 64,
            root="docs/worklogs/root",
            experiment_identity_digest="d" * 64,
            campaign_identity_digest="e" * 64,
        )
        pending = orchestrator.record_terminal(
            active,
            reservation_id="r1",
            status="ABANDONED",
            root_digest="f" * 64,
        )
        url = (
            "https://github.com/tensor4all/tenferro-rs/issues/1436"
            "#issuecomment-99"
        )
        preserved = orchestrator.record_preserved_event(
            pending,
            reservation_id="r1",
            preservation_commit="1" * 40,
            issue_url=url,
        )
        self.assertEqual(
            orchestrator.record_preserved_event(
                preserved,
                reservation_id="r1",
                preservation_commit="1" * 40,
                issue_url=url,
            ),
            preserved,
        )
        with self.assertRaises(protocol.ProtocolError):
            orchestrator.record_preserved_event(
                preserved,
                reservation_id="r1",
                preservation_commit="2" * 40,
                issue_url=url,
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
