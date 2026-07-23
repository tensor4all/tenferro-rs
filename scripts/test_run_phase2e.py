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


from scripts import phase2e_protocol as protocol
from scripts import run_phase2e as orchestrator


class OuterOrchestratorTests(unittest.TestCase):
    CANDIDATE = "a" * 40

    def test_stage_contract_rejects_shell_and_foreign_executable(self):
        for argv in (("/bin/sh", "-c", "true"), ("/usr/bin/true",)):
            with self.assertRaises(protocol.ProtocolError):
                orchestrator.validate_stage_argv(
                    orchestrator.STAGE_ORDER[0],
                    argv,
                    pathlib.Path("/repo/context.json"),
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

    def make_complete_root(self, root: pathlib.Path, *, seal: bool = True) -> None:
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
                ledger = protocol.close_attempt(
                    ledger, stage, lane, attempt, "PASS"
                )
        protocol.atomic_write_json(root / "evidence-ledger.json", ledger)
        for directory in ("dispatch-gates", "characterization"):
            (root / directory).mkdir()
            protocol.atomic_write_json(
                root / directory / "manifest.json",
                {"candidate": self.CANDIDATE, "gating_result": "PASS"},
            )
        for relative in orchestrator.required_root_paths(ledger):
            path = root / relative
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            protocol.atomic_write_json(path, {})
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
            environment = protocol.runtime_environment(path="/bin", home="/tmp")
            self.assertEqual(
                orchestrator.run_fixed_stages(
                    root, environment, lambda _stage, _environment: 0
                ),
                0,
            )
            orchestrator.seal_root(
                root,
                candidate_sha=self.CANDIDATE,
                reservation_id="reservation-1",
                experiment_identity_digest="b" * 64,
            )
            self.assertEqual(orchestrator.validate_root(root), "PASS")

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
