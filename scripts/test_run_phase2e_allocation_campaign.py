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
import shutil
import subprocess
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
        self.inherited_descriptors = []
        self.snapshot_inodes = []

    def __call__(
        self,
        argv,
        *,
        cwd,
        environment,
        deadline_seconds,
        inherited_descriptors,
        **_kwargs,
    ):
        self.test.assertEqual(len(inherited_descriptors), 1)
        inherited = inherited_descriptors[0]
        self.test.assertEqual(argv[0], f"/proc/self/fd/{inherited}")
        identity = pathlib.Path(argv[0]).name
        if identity.isdigit() and pathlib.Path(argv[0]).parent == pathlib.Path("/proc/self/fd"):
            payload = pathlib.Path(argv[0]).read_bytes()
            identity = {
                b"direct-baseline": "direct-current-main-baseline",
                b"common-baseline": "common-lock-normalized-baseline",
                b"candidate": "candidate",
            }[payload]
        self.calls.append((tuple(argv), pathlib.Path(cwd), dict(environment), deadline_seconds))
        self.identities.append(identity)
        self.inherited_descriptors.append(tuple(inherited_descriptors))
        self.snapshot_inodes.append(os.fstat(inherited).st_ino)
        ordinal = len(self.calls)
        case = argv[1]
        if ordinal == self.malformed_at:
            stdout = "{}\n"
        else:
            candidate = identity == "candidate"
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
            inherited_descriptors=tuple(inherited_descriptors),
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
        binary.write_bytes(
            {
                "direct-current-main-baseline": b"direct-baseline",
                "common-lock-normalized-baseline": b"common-baseline",
                "candidate": b"candidate",
            }[role]
        )
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
    def setUp(self) -> None:
        FakeCommandRunner.test = self

    def persisted_fixture(self, root: pathlib.Path):
        runner = load_runner()
        repository = SCRIPT.parent.parent.resolve()
        tool_paths = []
        for name in ("git", "cargo", "rustc"):
            if name == "git":
                executable = pathlib.Path(shutil.which("git") or "").resolve()
            else:
                located = subprocess.run(
                    ["rustup", "which", name],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                executable = pathlib.Path(located).resolve()
            if executable.parent not in tool_paths:
                tool_paths.append(executable.parent)
        tools = build.resolve_toolchain(os.pathsep.join(map(str, tool_paths)))
        cargo_version = subprocess.run(
            [tools.cargo.path, "--version", "--verbose"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        rustc_version = subprocess.run(
            [tools.rustc.path, "--version", "--verbose"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        target = build._rustc_host(rustc_version, "persisted integration")
        toolchain = build._toolchain_manifest(tools, cargo_version, rustc_version)
        scratch = root / "scratch"
        evidence = root / "evidence"
        evidence.mkdir()
        home = root / "home"
        cargo_home = root / "cargo-home"
        home.mkdir()
        cargo_home.mkdir()
        tracked_template = build._read_regular_bytes(
            repository / build.ALLOCATION_PROBE_SOURCE_ROOT / build.ALLOCATION_PROBE_TEMPLATE
        )
        tracked_sources = {
            relative: build._read_regular_bytes(
                repository / build.ALLOCATION_PROBE_SOURCE_ROOT / relative
            )
            for relative in build.ALLOCATION_PROBE_SOURCES
        }
        tenferro = {}
        probes = {}
        role_locks = {}
        for index, role in enumerate(build.BUILD_MANIFEST_PATHS, start=1):
            worktree = scratch / role
            for crate in ("tenferro-ad", "tenferro-cpu", "tenferro-tensor"):
                crate_root = worktree / "crates" / crate
                (crate_root / "src").mkdir(parents=True)
                features = (
                    "\n[features]\ncpu-faer = []\n"
                    if crate != "tenferro-tensor"
                    else ""
                )
                dependency_version = 1 if index == 1 else 2
                (crate_root / "Cargo.toml").write_text(
                    f'[package]\nname = "{crate}"\n'
                    f'version = "0.0.{dependency_version}"\n'
                    f'edition = "2021"\n{features}'
                )
                (crate_root / "src/lib.rs").write_text("")
            timing_target = scratch / "targets" / role
            timing_binary = timing_target / "release/deps/timing"
            timing_binary.parent.mkdir(parents=True)
            shutil.copyfile("/bin/true", timing_binary)
            timing_binary.chmod(0o755)
            build_environment = protocol.cargo_environment(
                path=tools.path,
                home=str(home.resolve()),
                cargo_home=str(cargo_home.resolve()),
                target_dir=str(timing_target.resolve()),
            )
            config_chain = []
            head = ("abcdef"[index] * 40)
            tenferro[role] = {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "toolchain": toolchain,
                "target": target,
                "profile": "bench",
                "requested_features": list(build.REQUESTED_FEATURES),
                "provider": "Faer",
                "benchmark_sha256": "1" * 64,
                "benchmark_stanza_sha256": "2" * 64,
                "command_template": list(build.BENCH_COMMAND),
                "config_chain_sha256": protocol.sha256_json(config_chain),
                "role": role,
                "head": head,
                "tracked_tree_sha256": "3" * 64,
                "resolved_features_sha256": "4" * 64,
                "lock_sha256": "5" * 64,
                "worktree": str(worktree.resolve()),
                "target_dir": str(timing_target.resolve()),
                "executable": str(timing_binary.resolve()),
                "executable_sha256": protocol.sha256_file(timing_binary),
                "validity_state": "COMPLETE",
                "source_delta": list(build._ROLE_SOURCE_DELTAS[role]),
                "commands": [
                    command.to_manifest()
                    for command in build.build_command_plan(target, tools.cargo)
                ],
                "environment": build_environment,
                "cargo_config_chain": config_chain,
            }

            generated = scratch / "allocation-probes" / role
            (generated / "src").mkdir(parents=True)
            (generated / "Cargo.toml").write_bytes(
                build._render_allocation_probe_manifest(tracked_template, worktree.resolve())
            )
            for relative, payload in tracked_sources.items():
                (generated / relative).write_bytes(payload)
            probe_target = scratch / "allocation-probe-targets" / role
            probe_binary = probe_target / "release" / build.ALLOCATION_PROBE_BINARY
            probe_binary.parent.mkdir(parents=True)
            count = 6 if role == "candidate" else 7
            probe_binary.write_text(
                "#!/bin/sh\n"
                f'# role={role}\n'
                "printf '{\"allocated_bytes\":64,\"allocation_count\":"
                f"{count}"
                ",\"allocation_failures\":0,\"case\":\"%s\","
                "\"checksum\":1.25,\"counter_overflow\":false,"
                "\"repetitions\":4096}\\n' \"$1\"\n"
            )
            probe_binary.chmod(0o755)
            probe_environment = protocol.cargo_environment(
                path=tools.path,
                home=str(home.resolve()),
                cargo_home=str(cargo_home.resolve()),
                target_dir=str(probe_target.resolve()),
            )
            subprocess.run(
                [
                    tools.cargo.path,
                    "generate-lockfile",
                    "--manifest-path",
                    generated / "Cargo.toml",
                ],
                cwd=generated,
                env=probe_environment,
                capture_output=True,
                text=True,
                check=True,
            )
            feature_step = build.allocation_probe_build_only_command_plan(
                generated / "Cargo.toml",
                probe_binary,
                str(tools.cargo.path),
                target,
            )[0]
            feature_graph = subprocess.run(
                feature_step.argv,
                cwd=generated,
                env=probe_environment,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            lock_bytes = (generated / "Cargo.lock").read_bytes()
            role_locks[role] = lock_bytes
            source_sha256 = {
                str(path): build.sha256_bytes(payload)
                for path, payload in tracked_sources.items()
            }
            runtime_environment = protocol.runtime_environment(
                path=tools.path, home=str(home.resolve())
            )
            probes[role] = {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "role": role,
                "head": head,
                "target": target,
                "profile": "bench",
                "validity_state": "COMPLETE",
                "generated_root": str(generated.resolve()),
                "target_dir": str(probe_target.resolve()),
                "executable": str(probe_binary.resolve()),
                "executable_sha256": protocol.sha256_file(probe_binary),
                "lock_name": "direct-probe" if index == 1 else "common-probe",
                "lock_sha256": build.sha256_bytes(lock_bytes),
                "cargo_config_chain": [],
                "config_chain_sha256": protocol.sha256_json([]),
                "resolved_features": feature_graph,
                "resolved_features_sha256": build.sha256_bytes(feature_graph.encode()),
                "template_sha256": build.sha256_bytes(tracked_template),
                "source_sha256": source_sha256,
                "generated_manifest_sha256": protocol.sha256_file(generated / "Cargo.toml"),
                "generated_source_sha256": source_sha256,
                "case_inventory": list(protocol.CANONICAL_CASES),
                "repetitions": 4096,
                "build_commands": [
                    command.to_manifest()
                    for command in build.allocation_probe_build_only_command_plan(
                        generated / "Cargo.toml",
                        probe_binary,
                        str(tools.cargo.path),
                        target,
                    )
                ],
                "build_environment": probe_environment,
                "environment": runtime_environment,
                "toolchain_sha256": protocol.sha256_json(toolchain),
                "tenferro_build_manifest_sha256": protocol.sha256_json(tenferro[role]),
            }

        direct_lock = role_locks["direct-current-main-baseline"]
        common_lock = role_locks["candidate"]
        for name, payload in (("direct-probe", direct_lock), ("common-probe", common_lock)):
            lock_path = evidence / build.LOCK_PATHS[name]
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_bytes(payload)
        common_generated = pathlib.Path(
            probes["common-lock-normalized-baseline"]["generated_root"]
        ) / "Cargo.lock"
        common_generated.write_bytes(common_lock)
        probes["common-lock-normalized-baseline"]["lock_sha256"] = build.sha256_bytes(
            common_lock
        )
        for role in build.BUILD_MANIFEST_PATHS:
            probes[role]["tenferro_build_manifest_sha256"] = protocol.sha256_json(
                tenferro[role]
            )
            for mapping, relative in (
                (tenferro, build.BUILD_MANIFEST_PATHS[role]),
                (probes, build.PROBE_BUILD_MANIFEST_PATHS[role]),
            ):
                path = evidence / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(mapping[role]) + "\n")
        ledger = evidence / "ledger.json"
        ledger.write_text(
            json.dumps(protocol.new_ledger(tenferro["candidate"]["head"])) + "\n"
        )
        args = argparse.Namespace(
            comparison_kind="direct-current-main",
            ledger=ledger,
            attempt_id=1,
            artifact_root=root / "attempt",
            working_directory=root.resolve(),
            probe_manifest_root=evidence,
            tenferro_manifest_root=evidence,
            repository=repository,
        )
        return runner, args

    def test_real_bounded_launch_inherits_only_sealed_snapshot_descriptor(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            logical = root / "probe"
            shutil.copyfile("/bin/sh", logical)
            logical.chmod(0o755)
            digest = protocol.sha256_file(logical)
            pinned = runner.PinnedExecutable.open(logical.resolve(), digest)
            unrelated = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
            arbitrary_regular = os.open("/bin/true", os.O_RDONLY | os.O_CLOEXEC)
            try:
                shutil.copyfile("/bin/false", logical)
                logical.chmod(0o755)
                command = (
                    str(pinned.launch_path),
                    "-c",
                    (
                        f"test -e /proc/self/fd/{pinned.descriptor} && "
                        f"test ! -e /proc/self/fd/{unrelated} && printf sealed"
                    ),
                )
                result = build.run_bounded_command(
                    command,
                    cwd=root,
                    environment={"PATH": "/usr/bin:/bin", "HOME": str(root)},
                    deadline_seconds=5,
                    inherited_descriptors=(pinned.descriptor,),
                )
                self.assertEqual(result.validity_state, "COMPLETE")
                self.assertEqual(result.stdout, "sealed")
                self.assertEqual(result.stderr, "")
                with self.assertRaises(protocol.ProtocolError):
                    build.run_bounded_command(
                        command,
                        cwd=root,
                        environment={"PATH": "/usr/bin:/bin", "HOME": str(root)},
                        deadline_seconds=5,
                        inherited_descriptors=(unrelated,),
                    )
                with self.assertRaises(protocol.ProtocolError):
                    build.run_bounded_command(
                        (f"/proc/self/fd/{arbitrary_regular}",),
                        cwd=root,
                        environment={"PATH": "/usr/bin:/bin", "HOME": str(root)},
                        deadline_seconds=5,
                        inherited_descriptors=(arbitrary_regular,),
                    )
            finally:
                os.close(unrelated)
                os.close(arbitrary_regular)
                pinned.close()

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
                        ("direct-current-main-baseline", case),
                        ("candidate", case),
                        ("candidate", case),
                        ("direct-current-main-baseline", case),
                        ("direct-current-main-baseline", case),
                        ("candidate", case),
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
                    if identity == "candidate"
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
            for identity, inode in zip(
                commands.identities, commands.snapshot_inodes, strict=True
            ):
                self.assertEqual(
                    inode,
                    manifest["executable_identities"][identity]["snapshot_inode"],
                )

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
            self.assertEqual(commands.identities.count("candidate"), 84)
            self.assertEqual(
                commands.identities.count("common-lock-normalized-baseline"), 84
            )
            baseline_identity = manifest["executable_identities"][
                "common-lock-normalized-baseline"
            ]
            direct_inode = pathlib.Path(
                probes["direct-current-main-baseline"]["executable"]
            ).stat().st_ino
            self.assertNotEqual(baseline_identity["source_inode"], direct_inode)
            for identity, inode in zip(
                commands.identities, commands.snapshot_inodes, strict=True
            ):
                self.assertEqual(
                    inode,
                    manifest["executable_identities"][identity]["snapshot_inode"],
                )

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
                    and self.identities[-1] == "candidate"
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
                    with self.assertRaises(protocol.ProtocolError):
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

    def test_stage_and_marker_atomic_write_failure_matrix_is_recoverable(self) -> None:
        runner = load_runner()
        for name, committed, first_exit in (
            (runner.FINALIZATION_STAGE, False, 2),
            (runner.FINALIZATION_STAGE, True, None),
            (runner.FINALIZATION_MARKER, False, None),
            (runner.FINALIZATION_MARKER, True, None),
        ):
            with self.subTest(
                name=name, committed=committed
            ), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
                original = runner.protocol.atomic_write_json_at
                injected = False

                def fail_once(directory_fd, selected_name, payload):
                    nonlocal injected
                    if selected_name == name and not injected:
                        injected = True
                        if committed:
                            original(directory_fd, selected_name, payload)
                        raise OSError(
                            f"injected {'post' if committed else 'pre'}-commit failure"
                        )
                    return original(directory_fd, selected_name, payload)

                with mock.patch.object(
                    runner.protocol, "atomic_write_json_at", side_effect=fail_once
                ):
                    if first_exit is None:
                        with self.assertRaises(protocol.ProtocolError):
                            runner._run_comparison(
                                args,
                                probe_manifests=probes,
                                tenferro_manifests=tenferro,
                                command_runner=commands,
                            )
                    else:
                        self.assertEqual(
                            runner._run_comparison(
                                args,
                                probe_manifests=probes,
                                tenferro_manifests=tenferro,
                                command_runner=commands,
                            ),
                            first_exit,
                        )
                self.assertTrue(injected)
                self.assertEqual(len(commands.calls), 168)

                recovery = FakeCommandRunner()
                expected_recovery = 2 if name == runner.FINALIZATION_STAGE and not committed else 0
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=recovery,
                    ),
                    expected_recovery,
                )
                self.assertEqual(recovery.calls, [])
                terminal = json.loads((args.artifact_root / "allocation.json").read_text())
                self.assertEqual(terminal["launch_count"], 168)
                self.assertEqual(
                    terminal["validity_state"],
                    "INCONCLUSIVE" if expected_recovery == 2 else "COMPLETE",
                )
                ledger = json.loads(args.ledger.read_text())
                self.assertIsNone(ledger["active_attempt_id"])
                self.assertEqual(len(ledger["attempts"]), 1)

    def test_cleanup_failure_during_recovery_is_idempotent(self) -> None:
        runner = load_runner()
        for committed in (False, True):
            with self.subTest(
                committed=committed
            ), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)

                def fail_close(path, payload):
                    if (
                        pathlib.Path(path) == args.ledger
                        and payload.get("active_attempt_id") is None
                    ):
                        raise OSError("leave committed finalization partial")
                    protocol.atomic_write_json(path, payload)

                with self.assertRaises(protocol.ProtocolError):
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                        atomic_writer=fail_close,
                    )
                self.assertEqual(len(commands.calls), 168)

                original_cleanup = runner._cleanup_finalization

                def fail_cleanup(handle):
                    if committed:
                        original_cleanup(handle)
                    raise OSError("injected recovery cleanup failure")

                recovery = FakeCommandRunner()
                with mock.patch.object(
                    runner, "_cleanup_finalization", side_effect=fail_cleanup
                ), self.assertRaises(protocol.ProtocolError):
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=recovery,
                    )
                self.assertEqual(recovery.calls, [])
                final_recovery = FakeCommandRunner()
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=final_recovery,
                    ),
                    0,
                )
                self.assertEqual(final_recovery.calls, [])

    def test_finalization_control_exception_identity_is_preserved_with_recovery(self) -> None:
        runner = load_runner()
        for name, expected_exit in (
            (runner.FINALIZATION_STAGE, 2),
            (runner.FINALIZATION_MARKER, 0),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
                original = runner.protocol.atomic_write_json_at
                interruption = KeyboardInterrupt(f"interrupt {name}")
                injected = False

                def interrupt_once(directory_fd, selected_name, payload):
                    nonlocal injected
                    if selected_name == name and not injected:
                        injected = True
                        raise interruption
                    return original(directory_fd, selected_name, payload)

                with mock.patch.object(
                    runner.protocol,
                    "atomic_write_json_at",
                    side_effect=interrupt_once,
                ), self.assertRaises(KeyboardInterrupt) as raised:
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    )
                self.assertIs(raised.exception, interruption)
                self.assertEqual(len(commands.calls), 168)
                recovery = FakeCommandRunner()
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=recovery,
                    ),
                    expected_exit,
                )
                self.assertEqual(recovery.calls, [])

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

    def test_public_persisted_validation_and_real_pinned_launch_are_unmocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner, args = self.persisted_fixture(pathlib.Path(temporary))
            self.assertEqual(runner.run_campaign(args), 0)
            terminal = json.loads((args.artifact_root / "allocation.json").read_text())
            self.assertEqual(terminal["launch_count"], 168)
            self.assertEqual((terminal["validity_state"], terminal["gate"]), ("COMPLETE", "PASS"))
            self.assertEqual(
                {
                    observation["role"]
                    for observation in terminal["observations"]
                    if observation["role"] != "candidate"
                },
                {"direct-current-main-baseline"},
            )
            argv = [
                "--comparison-kind",
                args.comparison_kind,
                "--ledger",
                str(args.ledger),
                "--attempt-id",
                str(args.attempt_id),
                "--artifact-root",
                str(args.artifact_root),
                "--working-directory",
                str(args.working_directory),
                "--probe-manifest-root",
                str(args.probe_manifest_root),
                "--tenferro-manifest-root",
                str(args.tenferro_manifest_root),
                "--repository",
                str(args.repository),
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(runner.main(argv), 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")

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
