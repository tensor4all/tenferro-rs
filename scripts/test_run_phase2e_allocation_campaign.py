#!/usr/bin/env python3
"""Contract tests for the atomic Phase 2E allocation campaign."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol


SCRIPT = pathlib.Path(__file__).with_name("run_phase2e_allocation_campaign.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("phase2e_allocation_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def record(case: str, *, count: int = 7, allocated_bytes: int = 64) -> str:
    return json.dumps(
        {
            "allocated_bytes": allocated_bytes,
            "allocation_count": count,
            "allocation_failures": 0,
            "case": case,
            "checksum": 1.25,
            "counter_overflow": False,
            "repetitions": 4096,
        },
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


class FakeCommandRunner:
    def __init__(
        self,
        *,
        candidate_count: int = 7,
        candidate_bytes: int = 64,
        baseline_count: int = 7,
        baseline_bytes: int = 64,
        invalid_at: int | None = None,
        malformed_at: int | None = None,
    ) -> None:
        self.candidate_count = candidate_count
        self.candidate_bytes = candidate_bytes
        self.baseline_count = baseline_count
        self.baseline_bytes = baseline_bytes
        self.invalid_at = invalid_at
        self.malformed_at = malformed_at
        self.calls = []
        self.identities = []

    def __call__(self, argv, *, cwd, environment, deadline_seconds, **_kwargs):
        identity = pathlib.Path(argv[0]).name
        if identity.isdigit() and pathlib.Path(argv[0]).parent == pathlib.Path("/proc/self/fd"):
            identity = pathlib.Path(os.readlink(argv[0])).name
            payload = pathlib.Path(argv[0]).read_bytes()
            identity = "candidate-probe" if payload == b"candidate" else "baseline-probe"
        self.calls.append((tuple(argv), pathlib.Path(cwd), dict(environment), deadline_seconds))
        self.identities.append(identity)
        ordinal = len(self.calls)
        case = argv[1]
        if ordinal == self.malformed_at:
            stdout = "{}\n"
        else:
            candidate = identity == "candidate-probe"
            stdout = record(
                case,
                count=self.candidate_count if candidate else self.baseline_count,
                allocated_bytes=self.candidate_bytes if candidate else self.baseline_bytes,
            )
        return build.CommandResult(
            argv=tuple(argv),
            cwd=str(cwd),
            environment=dict(sorted(environment.items())),
            deadline_seconds=deadline_seconds,
            returncode=1 if ordinal == self.invalid_at else 0,
            stdout=stdout,
            stderr="",
            validity_state="COMPLETE",
            failure_reason=None,
            terminated=False,
            killed=False,
        )


def fixture(root: pathlib.Path, lane: str = "direct-current-main"):
    tool_dir = root / "tools"
    tool_dir.mkdir()
    home = root / "home"
    home.mkdir()
    environment = protocol.runtime_environment(
        path=str(tool_dir.resolve()), home=str(home.resolve())
    )
    inventory = list(protocol.CANONICAL_CASES)
    roles = (
        "direct-current-main-baseline",
        "common-lock-normalized-baseline",
        "candidate",
    )
    probes = {}
    tenferro = {}
    for role in roles:
        binary_name = {
            "direct-current-main-baseline": "direct-baseline-probe",
            "common-lock-normalized-baseline": "common-baseline-probe",
            "candidate": "candidate-probe",
        }[role]
        binary = root / role / binary_name
        binary.parent.mkdir()
        binary.write_bytes(b"candidate" if role == "candidate" else b"baseline")
        binary.chmod(0o755)
        lock_name = "direct-probe" if role == "direct-current-main-baseline" else "common-probe"
        probes[role] = {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "role": role,
            "head": {
                "direct-current-main-baseline": "d" * 40,
                "common-lock-normalized-baseline": "e" * 40,
                "candidate": "c" * 40,
            }[role],
            "target": "x86_64-unknown-linux-gnu",
            "profile": "bench",
            "validity_state": "COMPLETE",
            "generated_root": str(binary.parent.resolve()),
            "target_dir": str(binary.parent.parent.resolve()),
            "executable": str(binary.resolve()),
            "executable_sha256": protocol.sha256_file(binary),
            "lock_name": lock_name,
            "lock_sha256": ("7" if lock_name == "direct-probe" else "8") * 64,
            "cargo_config_chain": [],
            "config_chain_sha256": protocol.sha256_json([]),
            "resolved_features": "phase2e-allocation-probe v0.0.0\n",
            "resolved_features_sha256": build.sha256_bytes(
                b"phase2e-allocation-probe v0.0.0\n"
            ),
            "template_sha256": "1" * 64,
            "source_sha256": {"src/main.rs": "2" * 64, "src/tests.rs": "3" * 64},
            "generated_manifest_sha256": "4" * 64,
            "generated_source_sha256": {
                "src/main.rs": "2" * 64,
                "src/tests.rs": "3" * 64,
            },
            "case_inventory": inventory,
            "repetitions": 4096,
            "build_commands": [
                {"name": "build", "argv": ["cargo", "build"], "deadline_seconds": 1800},
                {"name": "list-cases", "argv": [str(binary)], "deadline_seconds": 30},
            ],
            "build_environment": environment,
            "environment": environment,
            "toolchain_sha256": protocol.sha256_json({}),
            "tenferro_build_manifest_sha256": "0" * 64,
        }
        tenferro[role] = {
            "role": role,
            "head": probes[role]["head"],
            "validity_state": "COMPLETE",
            "lock_sha256": probes[role]["lock_sha256"],
            "target": probes[role]["target"],
            "worktree": str((root / role).resolve()),
        }
        probes[role]["tenferro_build_manifest_sha256"] = protocol.sha256_json(
            tenferro[role]
        )
    ledger = root / "ledger.json"
    ledger_payload = protocol.new_ledger("c" * 40)
    if lane == "common-lock-normalized":
        ledger_payload = protocol.open_attempt(
            ledger_payload, "allocation", "direct-current-main", 1
        )
        ledger_payload = protocol.close_attempt(
            ledger_payload,
            "allocation",
            "direct-current-main",
            1,
            "PASS",
        )
    ledger.write_text(json.dumps(ledger_payload) + "\n")
    artifact = root / "attempt"
    args = argparse.Namespace(
        comparison_kind=lane,
        ledger=ledger,
        attempt_id=1,
        artifact_root=artifact,
        working_directory=root.resolve(),
    )
    return args, probes, tenferro


class AllocationRecordTests(unittest.TestCase):
    def test_record_parser_accepts_only_exact_framing_schema_and_types(self) -> None:
        runner = load_runner()
        case = next(iter(protocol.CANONICAL_CASES))
        parsed = runner.parse_probe_record(record(case), case, 0)
        self.assertEqual(parsed["repetitions"], 4096)
        invalid = [
            record(case) + "\n",
            record(case).rstrip("\n"),
            record(case).replace('"allocation_count":7', '"allocation_count":true'),
            record(case).replace('"checksum":1.25', '"checksum":1'),
            record(case).replace('"checksum":1.25', '"checksum":NaN'),
            record(case).replace(f'"case":"{case}"', '"case":"foreign"'),
            record(case).replace('"repetitions":4096', '"repetitions":4095'),
            record(case).replace('"allocation_failures":0', '"allocation_failures":1'),
            record(case).replace('"counter_overflow":false', '"counter_overflow":true'),
            record(case).replace("{", '{"extra":0,', 1),
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(protocol.ProtocolError):
                    runner.parse_probe_record(payload, case, 0)
        for returncode in (1, 2, -9):
            with self.subTest(returncode=returncode):
                with self.assertRaises(protocol.ProtocolError):
                    runner.parse_probe_record(record(case), case, returncode)

    def test_record_parser_rejects_duplicate_known_keys(self) -> None:
        runner = load_runner()
        case = next(iter(protocol.CANONICAL_CASES))
        original = record(case)
        for key, changed in (
            ("allocated_bytes", "65"),
            ("allocation_count", "8"),
            ("allocation_failures", "1"),
            ("case", '"foreign"'),
            ("checksum", "2.5"),
            ("counter_overflow", "true"),
            ("repetitions", "4095"),
        ):
            marker = f'"{key}":'
            start = original.index(marker)
            value_start = start + len(marker)
            value_end = original.find(",", value_start)
            if value_end < 0:
                value_end = original.find("}", value_start)
            value = original[value_start:value_end]
            for duplicate in (value, changed):
                payload = original[:value_end] + f',"{key}":{duplicate}' + original[value_end:]
                with self.subTest(key=key, duplicate=duplicate):
                    with self.assertRaises(protocol.ProtocolError):
                        runner.parse_probe_record(payload, case, 0)


class AllocationCampaignTests(unittest.TestCase):
    def test_complete_campaign_launches_exact_fresh_fixed_matrix_and_passes(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
            result = runner._run_comparison(
                args,
                probe_manifests=probes,
                tenferro_manifests=tenferro,
                command_runner=commands,
            )
            self.assertEqual(result, 0)
            self.assertEqual(len(commands.calls), 2 * 3 * 28)
            expected = []
            for case in protocol.CANONICAL_CASES:
                expected.extend(
                    [
                        ("baseline-probe", case),
                        ("candidate-probe", case),
                        ("candidate-probe", case),
                        ("baseline-probe", case),
                        ("baseline-probe", case),
                        ("candidate-probe", case),
                    ]
                )
            observed = [
                (identity, call[0][1])
                for identity, call in zip(commands.identities, commands.calls, strict=True)
            ]
            self.assertEqual(observed, expected)
            self.assertTrue(all(call[3] == 30 for call in commands.calls))
            for identity, (argv, cwd, environment, _deadline) in zip(
                commands.identities, commands.calls, strict=True
            ):
                role = (
                    "candidate"
                    if identity == "candidate-probe"
                    else "direct-current-main-baseline"
                )
                self.assertEqual(cwd, root.resolve())
                self.assertEqual(environment, probes[role]["environment"])
            manifest = json.loads((args.artifact_root / "allocation.json").read_text())
            self.assertEqual((manifest["validity_state"], manifest["gate"]), ("COMPLETE", "PASS"))
            self.assertEqual(manifest["launch_count"], 168)
            self.assertEqual(manifest["protocol_sha256"], runner.PROTOCOL_SHA256)
            self.assertEqual(len(manifest["observations"]), 168)
            self.assertEqual(len({item["launch_index"] for item in manifest["observations"]}), 168)

    def test_both_count_and_bytes_must_be_nonregressing_for_every_observation(self) -> None:
        runner = load_runner()
        for count, allocated_bytes, expected_exit in ((8, 64, 3), (7, 65, 3), (7, 64, 0)):
            with self.subTest(count=count, allocated_bytes=allocated_bytes):
                with tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    args, probes, tenferro = fixture(root)
                    commands = FakeCommandRunner(
                        candidate_count=count, candidate_bytes=allocated_bytes
                    )
                    result = runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    )
                    self.assertEqual(result, expected_exit)
                    self.assertEqual(len(commands.calls), 168)

    def test_normalized_lane_uses_common_baseline_and_its_own_full_matrix(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root, "common-lock-normalized")
            commands = FakeCommandRunner()
            self.assertEqual(
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                ),
                0,
            )
            self.assertEqual(len(commands.calls), 168)
            manifest = json.loads((args.artifact_root / "allocation.json").read_text())
            self.assertIn("common-lock-normalized-baseline", manifest["probe_builds"])
            self.assertNotIn("direct-current-main-baseline", manifest["probe_builds"])

    def test_first_invalid_process_stops_whole_comparison_and_closes_retryable(self) -> None:
        runner = load_runner()
        for invalid_at, malformed_at in ((5, None), (None, 9)):
            with self.subTest(invalid_at=invalid_at, malformed_at=malformed_at):
                with tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    args, probes, tenferro = fixture(root)
                    commands = FakeCommandRunner(
                        invalid_at=invalid_at, malformed_at=malformed_at
                    )
                    result = runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    )
                    self.assertEqual(result, 2)
                    self.assertEqual(len(commands.calls), invalid_at or malformed_at)
                    manifest = json.loads((args.artifact_root / "allocation.json").read_text())
                    self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
                    self.assertIsNone(manifest["gate"])
                    self.assertEqual(manifest["launch_count"], invalid_at or malformed_at)
                    self.assertEqual(len(manifest["observations"]), invalid_at or malformed_at)
                    self.assertIsNone(manifest["observations"][-1]["record"])
                    ledger = json.loads(args.ledger.read_text())
                    self.assertIsNone(ledger["active_attempt_id"])
                    self.assertEqual(ledger["attempts"][-1]["validity_state"], "INCONCLUSIVE")

    def test_running_manifest_exists_after_ledger_registration_before_launch(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)

            class InspectingRunner(FakeCommandRunner):
                def __call__(self, argv, **kwargs):
                    ledger = json.loads(args.ledger.read_text())
                    manifest = json.loads((args.artifact_root / "allocation.json").read_text())
                    self.test.assertEqual(ledger["active_attempt_id"], 1)
                    self.test.assertEqual(manifest["validity_state"], "RUNNING")
                    return super().__call__(argv, **kwargs)

            commands = InspectingRunner(invalid_at=1)
            commands.test = self
            self.assertEqual(
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                ),
                2,
            )

    def test_probe_manifest_is_bound_to_exact_tenferro_build_and_inventory(self) -> None:
        runner = load_runner()
        for mutation in ("tenferro", "inventory", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                if mutation == "tenferro":
                    tenferro["candidate"]["lock_sha256"] = "9" * 64
                elif mutation == "inventory":
                    probes["candidate"]["case_inventory"].reverse()
                else:
                    probes["candidate"]["source_sha256"]["src/main.rs"] = "9" * 64
                with self.assertRaises(protocol.ProtocolError):
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=FakeCommandRunner(),
                    )

    def test_timeout_and_within_binary_inconsistency_are_whole_attempt_invalidity(self) -> None:
        runner = load_runner()

        class InvalidatingRunner(FakeCommandRunner):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode

            def __call__(self, argv, **kwargs):
                result = super().__call__(argv, **kwargs)
                if self.mode == "timeout" and len(self.calls) == 4:
                    return build.CommandResult(
                        **{
                            **result.__dict__,
                            "returncode": None,
                            "validity_state": "INCONCLUSIVE",
                            "failure_reason": "deadline-exceeded",
                            "terminated": True,
                        }
                    )
                if (
                    self.mode == "inconsistent"
                    and len(self.calls) == 6
                    and self.identities[-1] == "candidate-probe"
                ):
                    return build.CommandResult(
                        **{**result.__dict__, "stdout": record(argv[1], count=8)}
                    )
                return result

        for mode, expected_launches in (("timeout", 4), ("inconsistent", 6)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                commands = InvalidatingRunner(mode)
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    ),
                    2,
                )
                self.assertEqual(len(commands.calls), expected_launches)

    def test_logical_binary_replacement_after_pin_cannot_change_executed_bytes(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            candidate = pathlib.Path(probes["candidate"]["executable"])

            class ReplacingRunner(FakeCommandRunner):
                def __call__(self, argv, **kwargs):
                    if not self.calls:
                        replacement = candidate.with_name("replacement")
                        replacement.write_bytes(b"foreign")
                        replacement.chmod(0o755)
                        os.replace(replacement, candidate)
                    return super().__call__(argv, **kwargs)

            commands = ReplacingRunner(candidate_count=6, candidate_bytes=63)
            self.assertEqual(
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                ),
                0,
            )
            self.assertEqual(len(commands.calls), 168)
            self.assertTrue(
                all(
                    pathlib.Path(argv[0]).parent == pathlib.Path("/proc/self/fd")
                    for argv, *_ in commands.calls
                )
            )
            manifest = json.loads((args.artifact_root / "allocation.json").read_text())
            identity = manifest["executable_identities"]["candidate"]
            self.assertNotEqual(identity["source_inode"], candidate.stat().st_ino)
            self.assertEqual(identity["sha256"], probes["candidate"]["executable_sha256"])

    def test_same_inode_mutation_after_pin_is_rejected_before_that_role_launch(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            candidate = pathlib.Path(probes["candidate"]["executable"])

            class MutatingRunner(FakeCommandRunner):
                def __call__(self, argv, **kwargs):
                    result = super().__call__(argv, **kwargs)
                    if len(self.calls) == 1:
                        candidate.write_bytes(b"mutated-in-place")
                    return result

            commands = MutatingRunner()
            self.assertEqual(
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                ),
                2,
            )
            self.assertEqual(len(commands.calls), 1)

    def test_root_close_after_complete_cannot_change_pass_exit(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            original_close = runner.PinnedDirectory.close
            calls = []

            def failing_close(handle):
                calls.append(handle.logical_path)
                original_close(handle)
                raise OSError("injected root close failure")

            with mock.patch.object(runner.PinnedDirectory, "close", failing_close):
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=FakeCommandRunner(
                            candidate_count=6, candidate_bytes=63
                        ),
                    ),
                    0,
                )
            self.assertEqual(calls, [args.artifact_root.resolve()])

    def test_committed_finalization_recovers_without_rerunning_or_double_close(self) -> None:
        runner = load_runner()
        for failure_point in ("ledger-close", "publish"):
            with self.subTest(
                failure_point=failure_point
            ), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
                original_publish = runner._publish_staged_allocation

                def failing_writer(path, payload):
                    if (
                        failure_point == "ledger-close"
                        and pathlib.Path(path) == args.ledger
                        and payload.get("active_attempt_id") is None
                    ):
                        raise OSError("injected ledger close failure")
                    protocol.atomic_write_json(path, payload)

                def failing_publish(_root):
                    raise OSError("injected publish failure")

                if failure_point == "publish":
                    runner._publish_staged_allocation = failing_publish
                try:
                    with self.assertRaises(OSError):
                        runner._run_comparison(
                            args,
                            probe_manifests=probes,
                            tenferro_manifests=tenferro,
                            command_runner=commands,
                            atomic_writer=failing_writer,
                        )
                finally:
                    runner._publish_staged_allocation = original_publish
                self.assertEqual(len(commands.calls), 168)
                recovery_commands = FakeCommandRunner()
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=recovery_commands,
                    ),
                    0,
                )
                self.assertEqual(recovery_commands.calls, [])
                self.assertEqual(
                    set(path.name for path in args.artifact_root.iterdir()),
                    {"allocation.json"},
                )
                ledger = json.loads(args.ledger.read_text())
                self.assertIsNone(ledger["active_attempt_id"])
                self.assertEqual(len(ledger["attempts"]), 1)

    def test_public_persisted_campaign_runs_exact_matrix(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            args.probe_manifest_root = root / "probe-manifests"
            args.tenferro_manifest_root = root / "tenferro-manifests"
            args.repository = root.resolve()
            for role, relative in build.BUILD_MANIFEST_PATHS.items():
                path = args.tenferro_manifest_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(tenferro[role]) + "\n")
            commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
            with mock.patch.object(runner.build, "validate_build_manifest"), mock.patch.object(
                runner.build, "validate_allocation_probe_set", return_value=probes
            ) as probe_validator:
                self.assertEqual(runner.run_campaign(args, command_runner=commands), 0)
            self.assertEqual(len(commands.calls), 168)
            probe_validator.assert_called_once_with(
                args.probe_manifest_root, tenferro, repository=args.repository
            )

    def test_cli_exit_and_stream_contract_is_exact(self) -> None:
        runner = load_runner()
        argv = [
            "--comparison-kind",
            "direct-current-main",
            "--ledger",
            "/ledger",
            "--attempt-id",
            "1",
            "--artifact-root",
            "/artifact",
            "--working-directory",
            "/work",
            "--probe-manifest-root",
            "/probes",
            "--tenferro-manifest-root",
            "/builds",
            "--repository",
            "/repository",
        ]
        for result in (0, 2, 3):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.subTest(result=result), mock.patch.object(
                runner, "run_campaign", return_value=result
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(runner.main(argv), result)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
        failure = protocol.ProtocolError("typed failure")
        stderr = io.StringIO()
        with mock.patch.object(
            runner, "run_campaign", side_effect=failure
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(runner.main(argv), 2)
        self.assertEqual(
            stderr.getvalue(),
            "phase2e allocation campaign error: typed failure\n",
        )

    def test_control_exception_is_finalized_but_propagates_same_object(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            interruption = KeyboardInterrupt("stop")

            class InterruptingRunner(FakeCommandRunner):
                def __call__(self, argv, **kwargs):
                    if len(self.calls) == 2:
                        raise interruption
                    return super().__call__(argv, **kwargs)

            commands = InterruptingRunner()
            with self.assertRaises(KeyboardInterrupt) as caught:
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                )
            self.assertIs(caught.exception, interruption)
            manifest = json.loads((args.artifact_root / "allocation.json").read_text())
            self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
            ledger = json.loads(args.ledger.read_text())
            self.assertIsNone(ledger["active_attempt_id"])

    def test_import_safe_strict_cli(self) -> None:
        runner = load_runner()
        self.assertTrue(callable(runner.main))
        with self.assertRaises(SystemExit):
            runner.parse_args([])


if __name__ == "__main__":
    unittest.main()
