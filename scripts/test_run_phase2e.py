#!/usr/bin/env python3
"""Contract tests for the Phase 2E outer orchestrator."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock


from scripts import phase2e_protocol as protocol
from scripts import run_phase2e as orchestrator
from scripts import test_phase2e_build as build_test_fixtures


class OuterOrchestratorTests(unittest.TestCase):
    CANDIDATE = "a" * 40

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
        for directory in ("dispatch-gates", "characterization"):
            (root / directory).mkdir()
            protocol.atomic_write_json(
                root / directory / "manifest.json",
                {"candidate": self.CANDIDATE, "gating_result": "PASS"},
            )
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
        for relative in orchestrator.required_root_paths(ledger):
            if relative == orchestrator.PROGRESS_MANIFEST or relative.startswith(
                "children/"
            ):
                continue
            path = root / relative
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            protocol.atomic_write_json(path, {})
        orchestrator.run_fixed_stages(
            root,
            protocol.runtime_environment(path="/bin", home="/tmp"),
            lambda _stage, _environment: 0,
        )
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
            "scripts.run_phase1_eager_campaign.validate_completed_attempt",
            return_value=0,
        )
        for patcher in (probe_patch, allocation_patch, timing_patch):
            patcher.start()
            self.addCleanup(patcher.stop)
        if seal:
            orchestrator.seal_root(
                root,
                candidate_sha=self.CANDIDATE,
                reservation_id="reservation-1",
                experiment_identity_digest="b" * 64,
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
            manifest_path = root / "characterization" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["gating_result"] = "FAIL"
            protocol.atomic_write_json(manifest_path, manifest)
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_root(root)

    def test_manifest_hashes_every_normative_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_complete_root(root)
            manifest = json.loads(
                (root / orchestrator.AGGREGATE_MANIFEST).read_text(encoding="utf-8")
            )
            expected = {
                "evidence-ledger.json",
                "dispatch-gates/manifest.json",
                "characterization/manifest.json",
            }
            self.assertTrue(expected.issubset(manifest["inventory"]))
            (root / "dispatch-gates" / "manifest.json").write_text(
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
            index_path = pathlib.Path(directory) / "index.json"
            protocol.atomic_write_json(index_path, orchestrator.new_campaign_index())
            failures = []

            def update(number: int) -> None:
                try:
                    orchestrator.mutate_index(
                        index_path,
                        lambda value: {
                            **value,
                            "audit": sorted([*value.get("audit", []), number]),
                        },
                        lock_path=pathlib.Path(directory) / "index.lock",
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
            base = pathlib.Path(directory)
            index_path = base / "index.json"
            lock_path = base / "index.lock"
            root = base / "root"
            protocol.atomic_write_json(index_path, orchestrator.new_campaign_index())
            def fail(_root):
                raise OSError("boom")

            code = orchestrator.initialize_campaign(
                index_path=index_path,
                index_lock=lock_path,
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
