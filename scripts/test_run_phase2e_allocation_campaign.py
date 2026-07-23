#!/usr/bin/env python3
"""Contract tests for the atomic Phase 2E allocation campaign."""

from __future__ import annotations

import argparse
import copy
import contextlib
import fcntl
import importlib.util
import io
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol
from scripts import run_phase2e as outer_orchestrator


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


class ProcessCommandRunner:
    def __init__(
        self,
        launch_log: pathlib.Path,
        *,
        ready: pathlib.Path | None = None,
        release: pathlib.Path | None = None,
        crash_on_first: bool = False,
    ) -> None:
        self.launch_log = launch_log
        self.ready = ready
        self.release = release
        self.crash_on_first = crash_on_first
        self.calls = 0

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
        self.calls += 1
        descriptor = inherited_descriptors[0]
        payload = pathlib.Path(argv[0]).read_bytes()
        with self.launch_log.open("ab") as stream:
            stream.write(b"launch\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.calls == 1 and self.crash_on_first:
            os._exit(17)
        if self.calls == 1 and self.ready is not None:
            self.ready.write_bytes(b"ready")
            while self.release is not None and not self.release.exists():
                time.sleep(0.01)
        role = {
            b"direct-baseline": "direct-current-main-baseline",
            b"common-baseline": "common-lock-normalized-baseline",
            b"candidate": "candidate",
        }[payload]
        candidate = role == "candidate"
        return build.CommandResult(
            argv=tuple(argv),
            cwd=str(cwd),
            environment=dict(sorted(environment.items())),
            deadline_seconds=deadline_seconds,
            returncode=0,
            stdout=record(
                argv[1],
                count=6 if candidate else 7,
                allocated_bytes=63 if candidate else 64,
            ),
            stderr="",
            validity_state="COMPLETE",
            failure_reason=None,
            terminated=False,
            killed=False,
            inherited_descriptors=(descriptor,),
        )


def run_campaign_process(
    args,
    probes,
    tenferro,
    launch_log,
    result_queue,
    *,
    ready=None,
    release=None,
    crash_on_first=False,
) -> None:
    runner = load_runner()
    try:
        result = runner._run_comparison(
            args,
            probe_manifests=probes,
            tenferro_manifests=tenferro,
            command_runner=ProcessCommandRunner(
                launch_log,
                ready=ready,
                release=release,
                crash_on_first=crash_on_first,
            ),
        )
    except BaseException as error:
        result_queue.put((type(error).__name__, str(error)))
    else:
        result_queue.put(("exit", result))


def wait_for_ready(path: pathlib.Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not path.exists():
        raise AssertionError(f"child readiness marker was not published: {path}")


def release_and_reap_processes(
    processes: list[multiprocessing.Process],
    release: pathlib.Path,
    *,
    join_timeout_seconds: float = 2.0,
) -> None:
    failure = None

    def retain_failure(context, error):
        nonlocal failure
        if failure is None:
            failure = error
        else:
            build._record_suppressed_failure(failure, context, error)

    try:
        release.write_bytes(b"release")
    except BaseException as error:
        retain_failure("child release marker", error)
    for process in processes:
        try:
            process.join(join_timeout_seconds)
        except BaseException as error:
            retain_failure("initial child join", error)
        try:
            alive = process.is_alive()
        except BaseException as error:
            retain_failure("child liveness after initial join", error)
            alive = True
        if alive:
            try:
                process.terminate()
            except BaseException as error:
                retain_failure("child terminate", error)
            try:
                process.join(join_timeout_seconds)
            except BaseException as error:
                retain_failure("child join after terminate", error)
            try:
                alive = process.is_alive()
            except BaseException as error:
                retain_failure("child liveness after terminate", error)
                alive = True
            if alive:
                try:
                    process.kill()
                except BaseException as error:
                    retain_failure("child kill", error)
                try:
                    process.join(join_timeout_seconds)
                except BaseException as error:
                    retain_failure("final child join", error)
                try:
                    alive = process.is_alive()
                except BaseException as error:
                    retain_failure("final child liveness", error)
                    alive = True
        if alive:
            retain_failure(
                "child process cleanup",
                AssertionError(f"child process survived cleanup: {process.pid}"),
            )
    if failure is not None:
        raise failure


def wait_for_release_without_ready(release: pathlib.Path) -> None:
    while not release.exists():
        time.sleep(0.01)


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
    ledger = root / "evidence-ledger.json"
    ledger_payload = protocol.new_ledger("c" * 40)
    if lane == "common-lock-normalized":
        ledger_payload = protocol.open_attempt(
            ledger_payload,
            "allocation",
            "direct-current-main",
            1,
            artifact_root=str((root / "direct-attempt").resolve()),
        )
        ledger_payload = protocol.close_attempt(
            ledger_payload,
            "allocation",
            "direct-current-main",
            1,
            "PASS",
        )
    protocol.atomic_write_json(ledger, ledger_payload)
    (root / ".orchestrator.lock").write_bytes(b"")
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
                protocol.atomic_write_json(path, mapping[role])
        ledger = evidence / "evidence-ledger.json"
        protocol.atomic_write_json(
            ledger, protocol.new_ledger(tenferro["candidate"]["head"])
        )
        (evidence / ".orchestrator.lock").write_bytes(b"")
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
        for invalid_at, malformed_at in ((5, None), (168, None), (None, 9)):
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

    def test_public_recovery_preserves_durable_failed_probe_tail(self) -> None:
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
                protocol.atomic_write_json(path, tenferro[role])

            commands = FakeCommandRunner(invalid_at=1)
            original_atomic_write = runner.protocol.atomic_write_json_at
            injected = False

            def fail_stage_precommit(directory_fd, name, payload):
                nonlocal injected
                if name == runner.FINALIZATION_STAGE and not injected:
                    injected = True
                    raise OSError("injected failed-tail stage pre-commit failure")
                return original_atomic_write(directory_fd, name, payload)

            with mock.patch.object(
                runner.build, "validate_build_manifest"
            ), mock.patch.object(
                runner.build, "validate_allocation_probe_set", return_value=probes
            ), mock.patch.object(
                runner.protocol,
                "atomic_write_json_at",
                side_effect=fail_stage_precommit,
            ), self.assertRaises(protocol.ProtocolError):
                runner.run_campaign(args, command_runner=commands)
            self.assertTrue(injected)
            self.assertEqual(len(commands.calls), 1)

            running = json.loads(
                (args.artifact_root / "allocation.json").read_text()
            )
            self.assertEqual(running["validity_state"], "RUNNING")
            self.assertEqual(running["launch_count"], 1)
            self.assertEqual(len(running["observations"]), 1)
            failed_observation = running["observations"][0]
            self.assertIsNone(failed_observation["record"])
            self.assertEqual(failed_observation["launch_index"], 1)
            self.assertTrue(failed_observation["invalid_reason"])

            recovery = FakeCommandRunner()
            with mock.patch.object(
                runner.build, "validate_build_manifest"
            ), mock.patch.object(
                runner.build, "validate_allocation_probe_set", return_value=probes
            ):
                self.assertEqual(
                    runner.run_campaign(args, command_runner=recovery),
                    2,
                )
            self.assertEqual(recovery.calls, [])
            terminal = json.loads(
                (args.artifact_root / "allocation.json").read_text()
            )
            self.assertEqual(terminal["launch_count"], 1)
            self.assertEqual(terminal["observations"], [failed_observation])
            self.assertEqual(
                terminal["invalid_reason"], failed_observation["invalid_reason"]
            )

    def test_ordinary_failed_probe_checkpoint_failure_is_secondary(self) -> None:
        runner = load_runner()
        for checkpoint_error in (OSError("failed observation checkpoint"),):
            with self.subTest(
                checkpoint_error=type(checkpoint_error).__name__
            ), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                original_atomic_write = runner.protocol.atomic_write_json_at
                original_record = runner.build._record_suppressed_failure
                recorded = []

                def fail_failed_observation(directory_fd, name, payload):
                    if (
                        name == "allocation.json"
                        and payload.get("validity_state") == "RUNNING"
                        and payload.get("launch_count") == 1
                        and payload.get("observations")
                        and payload["observations"][-1].get("record") is None
                    ):
                        raise checkpoint_error
                    return original_atomic_write(directory_fd, name, payload)

                def record(primary, context, secondary):
                    recorded.append((primary, context, secondary))
                    original_record(primary, context, secondary)

                with mock.patch.object(
                    runner.protocol,
                    "atomic_write_json_at",
                    side_effect=fail_failed_observation,
                ), mock.patch.object(
                    runner.build,
                    "_record_suppressed_failure",
                    side_effect=record,
                ):
                    self.assertEqual(
                        runner._run_comparison(
                            args,
                            probe_manifests=probes,
                            tenferro_manifests=tenferro,
                            command_runner=FakeCommandRunner(invalid_at=1),
                        ),
                        2,
                    )
                matching = [
                    item
                    for item in recorded
                    if item[1] == "failed allocation observation checkpoint"
                ]
                self.assertEqual(len(matching), 1)
                primary, _context, secondary = matching[0]
                self.assertIsInstance(primary, protocol.ProtocolError)
                self.assertIs(secondary, checkpoint_error)

    def test_failed_probe_checkpoint_control_is_primary_and_recoverable(self) -> None:
        runner = load_runner()
        for exception_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(
                exception_type=exception_type.__name__
            ), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                interruption = exception_type("interrupt failed observation checkpoint")
                injected_traceback = None
                original_atomic_write = runner.protocol.atomic_write_json_at

                def interrupt_failed_observation(directory_fd, name, payload):
                    nonlocal injected_traceback
                    if (
                        name == "allocation.json"
                        and payload.get("validity_state") == "RUNNING"
                        and payload.get("launch_count") == 1
                        and payload.get("observations")
                        and payload["observations"][-1].get("record") is None
                    ):
                        try:
                            raise interruption
                        except BaseException as caught:
                            injected_traceback = caught.__traceback__
                            raise
                    return original_atomic_write(directory_fd, name, payload)

                commands = FakeCommandRunner(invalid_at=1)
                caught = None
                caught_traceback = None
                try:
                    with mock.patch.object(
                        runner.protocol,
                        "atomic_write_json_at",
                        side_effect=interrupt_failed_observation,
                    ):
                        runner._run_comparison(
                            args,
                            probe_manifests=probes,
                            tenferro_manifests=tenferro,
                            command_runner=commands,
                        )
                except BaseException as error:
                    caught = error
                    caught_traceback = error.__traceback__
                self.assertIs(caught, interruption)
                self.assertIsNotNone(caught_traceback)
                while caught_traceback.tb_next is not None:
                    caught_traceback = caught_traceback.tb_next
                self.assertIs(caught_traceback, injected_traceback)
                self.assertEqual(len(commands.calls), 1)

                terminal = json.loads(
                    (args.artifact_root / "allocation.json").read_text()
                )
                self.assertEqual(terminal["validity_state"], "INCONCLUSIVE")
                self.assertEqual(terminal["launch_count"], 1)
                self.assertIsNone(terminal["observations"][-1]["record"])
                recovery = FakeCommandRunner()
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=recovery,
                    ),
                    2,
                )
                self.assertEqual(recovery.calls, [])

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

        for mode, expected_launches in (("timeout", 4), ("inconsistent", 168)):
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
                if mode == "inconsistent":
                    terminal = json.loads(
                        (args.artifact_root / "allocation.json").read_text()
                    )
                    self.assertEqual(
                        terminal["invalid_reason"],
                        "allocation observations are inconsistent within candidate "
                        f"for {next(iter(protocol.CANONICAL_CASES))}",
                    )

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

    def test_root_close_after_complete_propagates_typed_failure(self) -> None:
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

            with mock.patch.object(
                runner.PinnedDirectory, "close", failing_close
            ), self.assertRaisesRegex(
                protocol.ProtocolError, "cannot close allocation pinned resource"
            ):
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=FakeCommandRunner(
                        candidate_count=6, candidate_bytes=63
                    ),
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
            (runner.FINALIZATION_STAGE, False, None),
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
                expected_recovery = 0
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
                    "COMPLETE",
                )
                ledger = json.loads(args.ledger.read_text())
                self.assertIsNone(ledger["active_attempt_id"])
                self.assertEqual(len(ledger["attempts"]), 1)

    def test_fabricated_closed_pass_without_observations_is_rejected(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            ledger = json.loads(args.ledger.read_text())
            args.artifact_root.mkdir()
            ledger = protocol.open_attempt(
                ledger,
                "allocation",
                args.comparison_kind,
                args.attempt_id,
                artifact_root=str(args.artifact_root.resolve()),
            )
            metadata = args.artifact_root.stat()
            ledger = protocol.bind_attempt_artifact(
                ledger,
                "allocation",
                args.comparison_kind,
                args.attempt_id,
                artifact_root=str(args.artifact_root.resolve()),
                artifact_device=metadata.st_dev,
                artifact_inode=metadata.st_ino,
            )
            ledger = protocol.close_attempt(
                ledger,
                "allocation",
                args.comparison_kind,
                args.attempt_id,
                "PASS",
            )
            protocol.atomic_write_json(args.ledger, ledger)
            protocol.atomic_write_json(
                args.artifact_root / "allocation.json",
                {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "protocol_sha256": runner.PROTOCOL_SHA256,
                    "comparison_kind": args.comparison_kind,
                    "attempt_id": args.attempt_id,
                    "expected_launch_count": 168,
                    "validity_state": "COMPLETE",
                    "gate": "PASS",
                    "tenferro_builds": {
                        "candidate": {"head": tenferro["candidate"]["head"]}
                    },
                },
            )
            commands = FakeCommandRunner()
            with self.assertRaises(protocol.ProtocolError):
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                )
            self.assertEqual(commands.calls, [])

    def test_public_recovery_rejects_full_success_downgraded_to_inconclusive(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
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

            terminal_path = args.artifact_root / "allocation.json"
            terminal = json.loads(terminal_path.read_text())
            terminal["validity_state"] = "INCONCLUSIVE"
            terminal["gate"] = None
            terminal["invalid_reason"] = "forged full-success downgrade"
            protocol.atomic_write_json(terminal_path, terminal)

            ledger = json.loads(args.ledger.read_text())
            attempt = ledger["attempts"][0]
            attempt["state"] = "INCONCLUSIVE"
            attempt["validity_state"] = "INCONCLUSIVE"
            attempt["statistical_result"] = None
            direct = ledger["stages"][0]["lanes"][0]
            direct["state"] = "RETRYABLE"
            direct["result"] = None
            normalized = ledger["stages"][0]["lanes"][1]
            normalized["state"] = "BLOCKED"
            normalized["result"] = None
            protocol.validate_ledger(ledger)
            protocol.atomic_write_json(args.ledger, ledger)
            forged_ledger = args.ledger.read_bytes()

            args.probe_manifest_root = root / "probe-manifests"
            args.tenferro_manifest_root = root / "tenferro-manifests"
            args.repository = root.resolve()
            for role, relative in build.BUILD_MANIFEST_PATHS.items():
                path = args.tenferro_manifest_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                protocol.atomic_write_json(path, tenferro[role])
            recovery = FakeCommandRunner()
            with mock.patch.object(runner.build, "validate_build_manifest"), mock.patch.object(
                runner.build, "validate_allocation_probe_set", return_value=probes
            ), self.assertRaises(protocol.ProtocolError):
                runner.run_campaign(args, command_runner=recovery)
            self.assertEqual(recovery.calls, [])
            self.assertEqual(args.ledger.read_bytes(), forged_ledger)

    def test_reserved_initialization_crash_closes_without_touching_unproven_root(self) -> None:
        runner = load_runner()
        for root_state in ("absent", "empty", "running"):
            with self.subTest(root_state=root_state), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                ledger = json.loads(args.ledger.read_text())
                ledger = protocol.open_attempt(
                    ledger,
                    "allocation",
                    args.comparison_kind,
                    args.attempt_id,
                    artifact_root=str(args.artifact_root.resolve()),
                )
                protocol.atomic_write_json(args.ledger, ledger)
                if root_state != "absent":
                    args.artifact_root.mkdir()
                if root_state == "running":
                    (args.artifact_root / "allocation.json").write_text("untrusted\n")
                before = (
                    None
                    if root_state == "absent"
                    else {
                        path.name: path.read_bytes()
                        for path in args.artifact_root.iterdir()
                    }
                )
                commands = FakeCommandRunner()
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    ),
                    2,
                )
                self.assertEqual(commands.calls, [])
                closed = json.loads(args.ledger.read_text())
                self.assertIsNone(closed["active_attempt_id"])
                self.assertEqual(closed["attempts"][0]["state"], "INCONCLUSIVE")
                if before is None:
                    self.assertFalse(args.artifact_root.exists())
                else:
                    self.assertEqual(
                        {
                            path.name: path.read_bytes()
                            for path in args.artifact_root.iterdir()
                        },
                        before,
                    )

    def test_initialization_atomic_write_matrix_is_recoverable(self) -> None:
        runner = load_runner()
        for phase in ("reservation", "manifest", "binding"):
            for committed in (False, True):
                with self.subTest(
                    phase=phase, committed=committed
                ), tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    args, probes, tenferro = fixture(root)
                    commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
                    original_path_write = protocol.atomic_write_json
                    original_root_write = runner.protocol.atomic_write_json_at
                    injected = False

                    def fail_path_write(path, payload):
                        nonlocal injected
                        selected = None
                        if pathlib.Path(path) == args.ledger and payload.get(
                            "active_attempt_id"
                        ) == args.attempt_id:
                            attempt = payload["attempts"][-1]
                            selected = (
                                "reservation"
                                if attempt["artifact_state"] == "RESERVED"
                                else "binding"
                            )
                        if selected == phase and not injected:
                            injected = True
                            if committed:
                                original_path_write(path, payload)
                            raise OSError(f"injected {phase} path write")
                        return original_path_write(path, payload)

                    def fail_root_write(directory_fd, name, payload):
                        nonlocal injected
                        if (
                            phase == "manifest"
                            and name == "allocation.json"
                            and not injected
                        ):
                            injected = True
                            if committed:
                                original_root_write(directory_fd, name, payload)
                            raise OSError("injected manifest write")
                        return original_root_write(directory_fd, name, payload)

                    with mock.patch.object(
                        runner.protocol,
                        "atomic_write_json_at",
                        side_effect=fail_root_write,
                    ):
                        if committed:
                            self.assertEqual(
                                runner._run_comparison(
                                    args,
                                    probe_manifests=probes,
                                    tenferro_manifests=tenferro,
                                    command_runner=commands,
                                    atomic_writer=fail_path_write,
                                ),
                                0,
                            )
                            self.assertEqual(len(commands.calls), 168)
                        else:
                            with self.assertRaises(protocol.ProtocolError):
                                runner._run_comparison(
                                    args,
                                    probe_manifests=probes,
                                    tenferro_manifests=tenferro,
                                    command_runner=commands,
                                    atomic_writer=fail_path_write,
                                )
                            self.assertEqual(commands.calls, [])
                    self.assertTrue(injected)

                    if not committed:
                        recovery = FakeCommandRunner(
                            candidate_count=6, candidate_bytes=63
                        )
                        expected = 0 if phase == "reservation" else 2
                        self.assertEqual(
                            runner._run_comparison(
                                args,
                                probe_manifests=probes,
                                tenferro_manifests=tenferro,
                                command_runner=recovery,
                            ),
                            expected,
                        )
                        self.assertEqual(
                            len(recovery.calls), 168 if expected == 0 else 0
                        )
                    ledger = json.loads(args.ledger.read_text())
                    self.assertIsNone(ledger["active_attempt_id"])

    def test_initialization_persists_bound_before_running_manifest(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            events = []
            original_path_write = protocol.atomic_write_json
            original_root_write = runner.protocol.atomic_write_json_at

            def record_path_write(path, payload):
                if pathlib.Path(path) == args.ledger and payload.get(
                    "active_attempt_id"
                ) == args.attempt_id:
                    artifact_state = payload["attempts"][-1]["artifact_state"]
                    if artifact_state == "BOUND":
                        self.assertTrue(args.artifact_root.is_dir())
                        self.assertFalse(
                            (args.artifact_root / "allocation.json").exists()
                        )
                    events.append(artifact_state)
                return original_path_write(path, payload)

            def record_root_write(directory_fd, name, payload):
                if name == "allocation.json" and payload.get("validity_state") == "RUNNING":
                    persisted = json.loads(args.ledger.read_text())
                    self.assertEqual(
                        persisted["attempts"][-1]["artifact_state"], "BOUND"
                    )
                    events.append("RUNNING")
                return original_root_write(directory_fd, name, payload)

            with mock.patch.object(
                runner.protocol, "atomic_write_json_at", side_effect=record_root_write
            ):
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=FakeCommandRunner(invalid_at=1),
                        atomic_writer=record_path_write,
                    ),
                    2,
                )
            self.assertEqual(events[:3], ["RESERVED", "BOUND", "RUNNING"])

    def test_bound_before_running_control_crashes_recover_without_launch(self) -> None:
        runner = load_runner()
        for phase, committed in (
            ("binding", True),
            ("manifest", False),
            ("manifest", True),
        ):
            with self.subTest(
                phase=phase, committed=committed
            ), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                commands = FakeCommandRunner()
                original_path_write = protocol.atomic_write_json
                original_root_write = runner.protocol.atomic_write_json_at
                interruption = KeyboardInterrupt(f"interrupt {phase}")
                injected = False

                def interrupt_path_write(path, payload):
                    nonlocal injected
                    attempt = payload.get("attempts", [{}])[-1]
                    selected = (
                        pathlib.Path(path) == args.ledger
                        and payload.get("active_attempt_id") == args.attempt_id
                        and attempt.get("artifact_state") == "BOUND"
                    )
                    if phase == "binding" and selected and not injected:
                        injected = True
                        original_path_write(path, payload)
                        raise interruption
                    return original_path_write(path, payload)

                def interrupt_root_write(directory_fd, name, payload):
                    nonlocal injected
                    selected = (
                        phase == "manifest"
                        and name == "allocation.json"
                        and payload.get("validity_state") == "RUNNING"
                        and payload.get("launch_count") == 0
                    )
                    if selected and not injected:
                        injected = True
                        if committed:
                            original_root_write(directory_fd, name, payload)
                        raise interruption
                    return original_root_write(directory_fd, name, payload)

                with mock.patch.object(
                    runner.protocol,
                    "atomic_write_json_at",
                    side_effect=interrupt_root_write,
                ), self.assertRaises(KeyboardInterrupt) as caught:
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                        atomic_writer=interrupt_path_write,
                    )
                self.assertIs(caught.exception, interruption)
                self.assertTrue(injected)
                self.assertEqual(commands.calls, [])

                recovery = FakeCommandRunner()
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=recovery,
                    ),
                    2,
                )
                self.assertEqual(recovery.calls, [])
                terminal = json.loads(
                    (args.artifact_root / "allocation.json").read_text()
                )
                self.assertEqual(
                    (
                        terminal["validity_state"],
                        terminal["gate"],
                        terminal["invalid_reason"],
                    ),
                    (
                        "INCONCLUSIVE",
                        None,
                        "allocation execution interrupted before launch 1",
                    ),
                )
                ledger = json.loads(args.ledger.read_text())
                self.assertIsNone(ledger["active_attempt_id"])
                self.assertEqual(ledger["attempts"][-1]["state"], "INCONCLUSIVE")

    def test_missing_or_nonregular_orchestrator_lock_rejects_before_launch(self) -> None:
        runner = load_runner()
        for corruption in ("missing", "symlink", "directory"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                lock = args.ledger.parent / ".orchestrator.lock"
                lock.unlink()
                if corruption == "symlink":
                    lock.symlink_to(args.ledger)
                elif corruption == "directory":
                    lock.mkdir()
                commands = FakeCommandRunner()
                with self.assertRaises(protocol.ProtocolError):
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    )
                self.assertEqual(commands.calls, [])
                self.assertFalse(args.artifact_root.exists())

    def test_outer_initialized_root_enters_allocation_lock_acquisition(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary).resolve()
            artifacts = repository / "docs" / "worklogs" / "artifacts"
            artifacts.mkdir(parents=True)
            root = artifacts / "outer-initialized-root"
            with mock.patch.object(outer_orchestrator, "require_remote_index"):
                code = outer_orchestrator.initialize_campaign(
                    repository=repository,
                    root=root,
                    reservation_id="a" * 64,
                    candidate_sha="c" * 40,
                    candidate_tree_sha256="b" * 64,
                    experiment_identity_digest="d" * 64,
                    campaign_identity_digest="e" * 64,
                )
            self.assertEqual(code, 0)

            lock = runner.EvidenceLock.acquire(root / "evidence-ledger.json")
            try:
                self.assertEqual(lock.path, root / ".orchestrator.lock")
                lock.assert_root_identity()
            finally:
                lock.close()

    def test_inherited_capability_rejects_missing_forged_and_wrong_descriptors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            args, _probes, _tenferro = fixture(root)
            (root / protocol.ORCHESTRATOR_LOCK_NAME).chmod(0o600)
            root_descriptor = outer_orchestrator._open_directory_descriptor(root)
            outer_orchestrator.fcntl.flock(
                root_descriptor, outer_orchestrator.fcntl.LOCK_EX
            )
            parent_identity = outer_orchestrator._linux_process_identity(os.getpid())

            def seal(payload):
                descriptor = os.memfd_create(
                    "phase2e-test-forged-capability",
                    os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
                )
                encoded = protocol._canonical_json_bytes(payload)
                os.write(descriptor, encoded)
                os.fchmod(descriptor, 0o400)
                fcntl.fcntl(
                    descriptor,
                    fcntl.F_ADD_SEALS,
                    protocol.CAMPAIGN_LOCK_CAPABILITY_SEALS,
                )
                return descriptor

            source = """
import os
import pathlib
from scripts import phase2e_protocol as protocol
from scripts import run_phase2e as outer

root = pathlib.Path(os.environ["PHASE2E_TEST_ROOT"])
identity = protocol.PreparedRootIdentity(root)
try:
    try:
        capability = protocol.InheritedCampaignLockCapability.discover(
            identity,
            stage="allocation/direct-current-main",
            context_sha256="a" * 64,
            reservation_id="reservation-1",
            parent_identity=outer._linux_process_identity(os.getppid()),
        )
    except protocol.ProtocolError:
        identity.revalidate()
        raise SystemExit(0)
    else:
        capability.close()
        raise SystemExit(9)
finally:
    identity.close()
"""
            try:
                with outer_orchestrator.exclusive_lock_at(
                    root_descriptor, protocol.ORCHESTRATOR_LOCK_NAME
                ) as lock_descriptor:
                    valid = protocol._campaign_lock_record_payload(
                        stage="allocation/direct-current-main",
                        context_sha256="a" * 64,
                        reservation_id="reservation-1",
                        root_descriptor=root_descriptor,
                        lock_descriptor=lock_descriptor,
                        parent_identity=parent_identity,
                    )
                    cases = [("missing-record", (root_descriptor, lock_descriptor))]
                    wrong_lock = os.open(
                        root / protocol.ORCHESTRATOR_LOCK_NAME,
                        os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                    )
                    wrong_root = os.open(
                        root,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                    )
                    records = []
                    try:
                        for name, field, descriptor in (
                            ("wrong-lock-ofd", "lock_descriptor", wrong_lock),
                            ("wrong-root-ofd", "root_descriptor", wrong_root),
                        ):
                            payload = dict(valid)
                            payload[field] = descriptor
                            record = seal(payload)
                            records.append(record)
                            cases.append(
                                (
                                    name,
                                    (
                                        root_descriptor
                                        if field != "root_descriptor"
                                        else wrong_root,
                                        lock_descriptor
                                        if field != "lock_descriptor"
                                        else wrong_lock,
                                        record,
                                    ),
                                )
                            )
                        wrong_parent_payload = dict(valid)
                        wrong_parent_payload["parent_start_ticks"] += 1
                        wrong_parent_record = seal(wrong_parent_payload)
                        records.append(wrong_parent_record)
                        cases.append(
                            (
                                "wrong-parent",
                                (
                                    root_descriptor,
                                    lock_descriptor,
                                    wrong_parent_record,
                                ),
                            )
                        )
                        environment = dict(os.environ)
                        environment["PHASE2E_TEST_ROOT"] = str(root)
                        for name, inherited in cases:
                            with self.subTest(case=name):
                                completed = subprocess.run(
                                    [os.sys.executable, "-c", source],
                                    cwd=pathlib.Path(__file__).resolve().parent.parent,
                                    env=environment,
                                    pass_fds=inherited,
                                    capture_output=True,
                                    text=True,
                                    timeout=2,
                                    check=False,
                                )
                                self.assertEqual(
                                    completed.returncode,
                                    0,
                                    msg=(
                                        f"case={name} stdout={completed.stdout!r} "
                                        f"stderr={completed.stderr!r}"
                                    ),
                                )
                    finally:
                        for descriptor in records:
                            os.close(descriptor)
                        os.close(wrong_root)
                        os.close(wrong_lock)
            finally:
                outer_orchestrator.fcntl.flock(
                    root_descriptor, outer_orchestrator.fcntl.LOCK_UN
                )
                os.close(root_descriptor)

    def test_outer_owned_lock_capability_enters_allocation_without_self_deadlock(
        self,
    ) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            args, _probes, _tenferro = fixture(root)
            (root / protocol.ORCHESTRATOR_LOCK_NAME).chmod(0o600)
            root_descriptor = outer_orchestrator._open_directory_descriptor(root)
            record = None
            try:
                outer_orchestrator.fcntl.flock(
                    root_descriptor, outer_orchestrator.fcntl.LOCK_EX
                )
                with outer_orchestrator.exclusive_lock_at(
                    root_descriptor, protocol.ORCHESTRATOR_LOCK_NAME
                ) as lock_descriptor:
                    inherited = [root_descriptor, lock_descriptor]
                    if hasattr(protocol, "SealedCampaignLockRecord"):
                        record = protocol.SealedCampaignLockRecord.create(
                            stage="allocation/direct-current-main",
                            context_sha256="a" * 64,
                            reservation_id="reservation-1",
                            root_descriptor=root_descriptor,
                            lock_descriptor=lock_descriptor,
                            parent_identity=outer_orchestrator._linux_process_identity(
                                os.getpid()
                            ),
                        )
                        inherited.append(record.descriptor)
                    source = """
import json
import os
import pathlib
import subprocess
import sys
from scripts import phase2e_protocol as protocol
from scripts import run_phase2e as outer
from scripts import run_phase2e_allocation_campaign as allocation

ledger = pathlib.Path(os.environ["PHASE2E_TEST_LEDGER"])
if hasattr(protocol, "InheritedCampaignLockCapability"):
    identity = protocol.PreparedRootIdentity(ledger.parent)
    capability = protocol.InheritedCampaignLockCapability.discover(
        identity,
        stage="allocation/direct-current-main",
        context_sha256="a" * 64,
        reservation_id="reservation-1",
        parent_identity=outer._linux_process_identity(os.getppid()),
    )
    try:
        inherited_identities = [
            [os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino]
            for descriptor in (
                capability.root_descriptor,
                capability.lock_descriptor,
                capability.record_descriptor,
            )
        ]
        leakage_probe = '''
import json
import os
import sys
expected = {tuple(item) for item in json.loads(sys.argv[1])}
for name in os.listdir("/proc/self/fd"):
    try:
        metadata = os.fstat(int(name))
    except (OSError, ValueError):
        continue
    if (metadata.st_dev, metadata.st_ino) in expected:
        raise SystemExit(7)
'''
        leaked = subprocess.run(
            [sys.executable, "-c", leakage_probe, json.dumps(inherited_identities)],
            check=False,
        )
        if leaked.returncode:
            raise SystemExit(leaked.returncode)
        lock = allocation.EvidenceLock.acquire(
            ledger,
            inherited_capability=capability,
            stage="allocation/direct-current-main",
            context_sha256="a" * 64,
            reservation_id="reservation-1",
        )
        lock.close()
    finally:
        capability.close()
        identity.close()
else:
    lock = allocation.EvidenceLock.acquire(ledger)
    lock.close()
"""
                    environment = dict(os.environ)
                    environment["PHASE2E_TEST_LEDGER"] = str(args.ledger)
                    completed = subprocess.run(
                        [os.sys.executable, "-c", source],
                        cwd=pathlib.Path(__file__).resolve().parent.parent,
                        env=environment,
                        pass_fds=tuple(inherited),
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
                    )
            finally:
                if record is not None:
                    record.close()
                outer_orchestrator.fcntl.flock(
                    root_descriptor, outer_orchestrator.fcntl.LOCK_UN
                )
                os.close(root_descriptor)

    def test_standalone_allocation_waits_for_outer_lock_then_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            args, _probes, _tenferro = fixture(root)
            (root / protocol.ORCHESTRATOR_LOCK_NAME).chmod(0o600)
            root_descriptor = outer_orchestrator._open_directory_descriptor(root)
            outer_orchestrator.fcntl.flock(
                root_descriptor, outer_orchestrator.fcntl.LOCK_EX
            )
            environment = dict(os.environ)
            environment["PHASE2E_TEST_LEDGER"] = str(args.ledger)
            source = """
import os
import pathlib
from scripts import run_phase2e_allocation_campaign as allocation
lock = allocation.EvidenceLock.acquire(
    pathlib.Path(os.environ["PHASE2E_TEST_LEDGER"])
)
lock.close()
"""
            process = None
            try:
                with outer_orchestrator.exclusive_lock_at(
                    root_descriptor, protocol.ORCHESTRATOR_LOCK_NAME
                ):
                    process = subprocess.Popen(
                        [os.sys.executable, "-c", source],
                        cwd=pathlib.Path(__file__).resolve().parent.parent,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    time.sleep(0.1)
                    self.assertIsNone(process.poll())
                self.assertIsNone(process.poll())
                outer_orchestrator.fcntl.flock(
                    root_descriptor, outer_orchestrator.fcntl.LOCK_UN
                )
                stdout, stderr = process.communicate(timeout=2)
                self.assertEqual(
                    process.returncode,
                    0,
                    msg=f"stdout={stdout!r}\nstderr={stderr!r}",
                )
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.communicate(timeout=2)
                try:
                    outer_orchestrator.fcntl.flock(
                        root_descriptor, outer_orchestrator.fcntl.LOCK_UN
                    )
                finally:
                    os.close(root_descriptor)

    def test_inherited_capability_rejects_root_and_lock_path_replacement(
        self,
    ) -> None:
        source = """
import os
import pathlib
from scripts import phase2e_protocol as protocol
from scripts import run_phase2e as outer

identity = protocol.PreparedRootIdentity(
    pathlib.Path(os.environ["PHASE2E_TEST_ROOT"])
)
try:
    try:
        capability = protocol.InheritedCampaignLockCapability.discover(
            identity,
            stage="allocation/direct-current-main",
            context_sha256="a" * 64,
            reservation_id="reservation-1",
            parent_identity=outer._linux_process_identity(os.getppid()),
        )
    except protocol.ProtocolError:
        raise SystemExit(0)
    else:
        capability.close()
        raise SystemExit(9)
finally:
    identity.close()
"""
        for replacement in ("root", "lock"):
            with self.subTest(
                replacement=replacement
            ), tempfile.TemporaryDirectory() as temporary:
                base = pathlib.Path(temporary).resolve()
                root = base / "evidence"
                root.mkdir(mode=0o700)
                fixture(root)
                lock_path = root / protocol.ORCHESTRATOR_LOCK_NAME
                lock_path.chmod(0o600)
                root_descriptor = outer_orchestrator._open_directory_descriptor(root)
                lock_descriptor = os.open(
                    lock_path,
                    os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                )
                outer_orchestrator.fcntl.flock(
                    root_descriptor, outer_orchestrator.fcntl.LOCK_EX
                )
                outer_orchestrator.fcntl.flock(
                    lock_descriptor, outer_orchestrator.fcntl.LOCK_EX
                )
                record = protocol.SealedCampaignLockRecord.create(
                    stage="allocation/direct-current-main",
                    context_sha256="a" * 64,
                    reservation_id="reservation-1",
                    root_descriptor=root_descriptor,
                    lock_descriptor=lock_descriptor,
                    parent_identity=outer_orchestrator._linux_process_identity(
                        os.getpid()
                    ),
                )
                displaced = base / f"retained-{replacement}"
                try:
                    if replacement == "lock":
                        lock_path.rename(displaced)
                        lock_path.touch(mode=0o600)
                        lock_path.chmod(0o600)
                    else:
                        root.rename(displaced)
                        root.mkdir(mode=0o700)
                    environment = dict(os.environ)
                    environment["PHASE2E_TEST_ROOT"] = str(root)
                    completed = subprocess.run(
                        [os.sys.executable, "-c", source],
                        cwd=pathlib.Path(__file__).resolve().parent.parent,
                        env=environment,
                        pass_fds=(
                            root_descriptor,
                            lock_descriptor,
                            record.descriptor,
                        ),
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
                    )
                finally:
                    if replacement == "lock":
                        lock_path.unlink()
                        displaced.rename(lock_path)
                    else:
                        root.rmdir()
                        displaced.rename(root)
                    record.close()
                    outer_orchestrator.fcntl.flock(
                        lock_descriptor, outer_orchestrator.fcntl.LOCK_UN
                    )
                    os.close(lock_descriptor)
                    outer_orchestrator.fcntl.flock(
                        root_descriptor, outer_orchestrator.fcntl.LOCK_UN
                    )
                    os.close(root_descriptor)

    def test_outer_and_allocation_consume_shared_lock_name(self) -> None:
        runner = load_runner()
        self.assertEqual(protocol.ORCHESTRATOR_LOCK_NAME, ".orchestrator.lock")
        self.assertEqual(
            outer_orchestrator.ORCHESTRATOR_LOCK_NAME,
            protocol.ORCHESTRATOR_LOCK_NAME,
        )
        self.assertEqual(runner.ORCHESTRATOR_LOCK, protocol.ORCHESTRATOR_LOCK_NAME)

    def test_orchestrator_lock_control_exception_closes_descriptor_once(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, _probes, _tenferro = fixture(root)
            interruption = KeyboardInterrupt("stop lock acquisition")
            real_close = runner.os.close
            closed = []

            def interrupt_flock(_descriptor, _operation):
                raise interruption

            def tracking_close(descriptor):
                closed.append(descriptor)
                return real_close(descriptor)

            with mock.patch.object(
                runner.fcntl, "flock", side_effect=interrupt_flock
            ), mock.patch.object(runner.os, "close", side_effect=tracking_close):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    runner.EvidenceLock.acquire(args.ledger)
            self.assertIs(caught.exception, interruption)
            self.assertEqual(len(closed), 1)

            lock = runner.EvidenceLock.acquire(args.ledger)
            descriptor = lock.descriptor
            root_descriptor = lock.root_descriptor
            lock.close()
            lock.close()
            with self.assertRaises(OSError):
                os.fstat(descriptor)
            with self.assertRaises(OSError):
                os.fstat(root_descriptor)

    def test_orchestrator_lock_close_releases_both_descriptors_once(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, _probes, _tenferro = fixture(root)
            lock = runner.EvidenceLock.acquire(args.ledger)
            descriptor = lock.descriptor
            root_descriptor = lock.root_descriptor
            real_close = runner.os.close
            closed = []

            def fail_root_close(owned):
                closed.append(owned)
                real_close(owned)
                if owned == root_descriptor:
                    raise OSError("injected outer evidence root close failure")

            with mock.patch.object(runner.os, "close", side_effect=fail_root_close):
                with self.assertRaisesRegex(
                    OSError, "outer evidence root close failure"
                ):
                    lock.close()
                lock.close()
            self.assertEqual(closed, [descriptor, root_descriptor])
            for owned in (descriptor, root_descriptor):
                with self.assertRaises(OSError):
                    os.fstat(owned)

    def test_orchestrator_lock_requires_canonical_colocated_ledger(self) -> None:
        runner = load_runner()
        for corruption in ("wrong-name", "ledger-symlink", "parent-symlink"):
            with self.subTest(
                corruption=corruption
            ), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, _probes, _tenferro = fixture(root)
                ledger = args.ledger
                if corruption == "wrong-name":
                    selected = ledger.with_name("foreign-ledger.json")
                    selected.write_bytes(ledger.read_bytes())
                elif corruption == "ledger-symlink":
                    foreign = root / "foreign-ledger.json"
                    foreign.write_bytes(ledger.read_bytes())
                    ledger.unlink()
                    ledger.symlink_to(foreign)
                    selected = ledger
                else:
                    alias = root / "evidence-alias"
                    alias.symlink_to(ledger.parent, target_is_directory=True)
                    selected = alias / ledger.name
                with self.assertRaises(protocol.ProtocolError):
                    runner.EvidenceLock.acquire(selected)

    def test_primary_error_suppresses_lock_release_failure(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            interruption = KeyboardInterrupt("primary campaign interruption")
            close_error = OSError("secondary lock release failure")
            original_close = runner.EvidenceLock.close
            original_record = runner.build._record_suppressed_failure
            recorded = []

            def interrupt_locked(*_args, **_kwargs):
                raise interruption

            def fail_after_close(lock):
                original_close(lock)
                raise close_error

            def record(primary, context, secondary):
                recorded.append((primary, context, secondary))
                original_record(primary, context, secondary)

            caught = None
            with mock.patch.object(
                runner, "_run_comparison_locked", side_effect=interrupt_locked
            ), mock.patch.object(
                runner.EvidenceLock, "close", fail_after_close
            ), mock.patch.object(
                runner.build, "_record_suppressed_failure", side_effect=record
            ):
                try:
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                    )
                except BaseException as error:
                    caught = error
            self.assertIs(caught, interruption)
            self.assertEqual(
                recorded,
                [(interruption, "orchestrator lock release", close_error)],
            )

    def test_lock_close_rejects_replaced_outer_root_path_and_closes_fds(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            outer = parent / "evidence"
            outer.mkdir()
            args, _probes, _tenferro = fixture(outer)
            lock = runner.EvidenceLock.acquire(args.ledger)
            descriptors = (lock.descriptor, lock.root_descriptor)
            os.rename(outer, parent / "retained-evidence")
            outer.mkdir()
            with self.assertRaisesRegex(
                protocol.ProtocolError,
                "outer evidence root pathname identity changed",
            ):
                lock.close()
            for descriptor in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    def test_control_primary_suppresses_real_outer_root_identity_failure(self) -> None:
        runner = load_runner()
        for exception_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(
                exception_type=exception_type.__name__
            ), tempfile.TemporaryDirectory() as temporary:
                parent = pathlib.Path(temporary)
                outer = parent / "evidence"
                outer.mkdir()
                args, probes, tenferro = fixture(outer)
                moved = parent / "retained-evidence"
                interruption = exception_type("interrupt after outer-root replacement")
                injected_traceback = None

                def replace_and_interrupt(*_args, **_kwargs):
                    nonlocal injected_traceback
                    os.rename(outer, moved)
                    outer.mkdir()
                    (outer / "evidence-ledger.json").write_bytes(
                        (moved / "evidence-ledger.json").read_bytes()
                    )
                    (outer / ".orchestrator.lock").write_bytes(b"")
                    try:
                        raise interruption
                    except BaseException as caught:
                        injected_traceback = caught.__traceback__
                        raise

                caught = None
                caught_traceback = None
                try:
                    with mock.patch.object(
                        runner,
                        "_run_comparison_locked",
                        side_effect=replace_and_interrupt,
                    ):
                        runner._run_comparison(
                            args,
                            probe_manifests=probes,
                            tenferro_manifests=tenferro,
                        )
                except BaseException as error:
                    caught = error
                    caught_traceback = error.__traceback__
                self.assertIs(caught, interruption)
                self.assertIsNotNone(caught_traceback)
                while caught_traceback.tb_next is not None:
                    caught_traceback = caught_traceback.tb_next
                self.assertIs(caught_traceback, injected_traceback)

    def test_two_processes_serialize_one_attempt_and_bind_one_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            second_args = copy.copy(args)
            second_args.artifact_root = root / "foreign-attempt"
            launch_log = root / "launches.log"
            ready = root / "first-ready"
            release = root / "release-first"
            context = multiprocessing.get_context("fork")
            results = context.Queue()
            first = context.Process(
                target=run_campaign_process,
                args=(args, probes, tenferro, launch_log, results),
                kwargs={"ready": ready, "release": release},
            )
            second = context.Process(
                target=run_campaign_process,
                args=(second_args, probes, tenferro, launch_log, results),
            )
            started = []
            first.start()
            started.append(first)
            try:
                wait_for_ready(ready, 10)
                lock_path = args.ledger.parent / ".orchestrator.lock"
                original_lock_identity = lock_path.stat().st_ino
                replacement = args.ledger.parent / ".replacement-orchestrator.lock"
                replacement.write_bytes(b"")
                os.replace(replacement, lock_path)
                self.assertNotEqual(lock_path.stat().st_ino, original_lock_identity)
                second.start()
                started.append(second)
                time.sleep(0.2)
                self.assertTrue(second.is_alive())
                self.assertEqual(launch_log.read_text().splitlines(), ["launch"])
            finally:
                release_and_reap_processes(
                    started, release, join_timeout_seconds=20
                )
            self.assertEqual((first.exitcode, second.exitcode), (0, 0))
            outcomes = {results.get(timeout=2), results.get(timeout=2)}
            self.assertIn(("exit", 0), outcomes)
            self.assertTrue(
                any(
                    kind == "ProtocolError" and "different artifact root" in detail
                    for kind, detail in outcomes
                )
            )
            self.assertEqual(len(launch_log.read_text().splitlines()), 168)
            self.assertFalse(second_args.artifact_root.exists())
            ledger = json.loads(args.ledger.read_text())
            self.assertEqual(len(ledger["attempts"]), 1)
            self.assertEqual(
                ledger["attempts"][0]["artifact_root"],
                str(args.artifact_root.resolve()),
            )

    def test_outer_root_path_replacement_cannot_close_same_reservation(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            outer = parent / "evidence"
            outer.mkdir()
            args, probes, tenferro = fixture(outer)
            moved = parent / "retained-evidence"

            class ReplacingRunner(FakeCommandRunner):
                replacement_ledger = None

                def __call__(self, *call_args, **call_kwargs):
                    result = super().__call__(*call_args, **call_kwargs)
                    if len(self.calls) == 1:
                        os.rename(outer, moved)
                        outer.mkdir()
                        self.replacement_ledger = (
                            moved / "evidence-ledger.json"
                        ).read_bytes()
                        (outer / "evidence-ledger.json").write_bytes(
                            self.replacement_ledger
                        )
                        (outer / ".orchestrator.lock").write_bytes(b"")
                    return result

            commands = ReplacingRunner(candidate_count=6, candidate_bytes=63)
            with self.assertRaisesRegex(
                protocol.ProtocolError,
                "outer evidence root pathname identity changed",
            ):
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                )
            self.assertEqual(len(commands.calls), 1)
            self.assertEqual(
                (outer / "evidence-ledger.json").read_bytes(),
                commands.replacement_ledger,
            )
            replacement = json.loads(
                (outer / "evidence-ledger.json").read_text()
            )
            self.assertEqual(replacement["active_attempt_id"], 1)
            self.assertEqual(replacement["attempts"][-1]["state"], "RUNNING")
            self.assertFalse((outer / "attempt").exists())

    def test_recovery_rejects_outer_root_path_replacement(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            outer = parent / "evidence"
            outer.mkdir()
            args, probes, tenferro = fixture(outer)
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
            moved = parent / "retained-evidence"
            original_cleanup = runner._cleanup_finalization
            replacement_ledger = None

            def replace_after_cleanup(handle):
                nonlocal replacement_ledger
                original_cleanup(handle)
                os.rename(outer, moved)
                outer.mkdir()
                replacement_ledger = (
                    moved / "evidence-ledger.json"
                ).read_bytes()
                (outer / "evidence-ledger.json").write_bytes(replacement_ledger)
                (outer / ".orchestrator.lock").write_bytes(b"")

            recovery = FakeCommandRunner()
            with mock.patch.object(
                runner, "_cleanup_finalization", side_effect=replace_after_cleanup
            ), self.assertRaisesRegex(
                protocol.ProtocolError,
                "outer evidence root pathname identity changed",
            ):
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=recovery,
                )
            self.assertEqual(recovery.calls, [])
            self.assertEqual(
                (outer / "evidence-ledger.json").read_bytes(), replacement_ledger
            )

    def test_failed_ready_assertion_reaps_child_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            ready = root / "never-ready"
            release = root / "release"
            context = multiprocessing.get_context("fork")
            process = context.Process(
                target=wait_for_release_without_ready,
                args=(release,),
            )
            process.start()
            try:
                with self.assertRaisesRegex(
                    AssertionError, "readiness marker was not published"
                ):
                    wait_for_ready(ready, 0.05)
            finally:
                release_and_reap_processes(
                    [process], release, join_timeout_seconds=0.2
                )
            self.assertFalse(process.is_alive())
            self.assertIsNotNone(process.exitcode)

    def test_reaper_attempts_every_escalation_after_control_failures(self) -> None:
        class FailingProcess:
            pid = 12345

            def __init__(self, *, join_failure=None, terminate_failure=None):
                self.join_failure = join_failure
                self.terminate_failure = terminate_failure
                self.join_calls = 0
                self.alive = True
                self.calls = []

            def join(self, timeout):
                self.join_calls += 1
                self.calls.append(("join", timeout))
                if self.join_calls == 1 and self.join_failure is not None:
                    raise self.join_failure
                if self.join_calls == 3:
                    self.alive = False

            def is_alive(self):
                self.calls.append(("is_alive",))
                return self.alive

            def terminate(self):
                self.calls.append(("terminate",))
                if self.terminate_failure is not None:
                    raise self.terminate_failure

            def kill(self):
                self.calls.append(("kill",))

        for phase in ("join", "terminate"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                primary = KeyboardInterrupt(f"interrupt child {phase}")
                secondary = SystemExit(f"interrupt after child {phase}")
                process = FailingProcess(
                    join_failure=primary if phase == "join" else None,
                    terminate_failure=(
                        secondary if phase == "join" else primary
                    ),
                )
                sibling = FailingProcess()
                caught = None
                try:
                    release_and_reap_processes(
                        [process, sibling],
                        pathlib.Path(temporary) / "release",
                        join_timeout_seconds=0.25,
                    )
                except BaseException as error:
                    caught = error
                self.assertIs(caught, primary)
                self.assertFalse(process.alive)
                self.assertFalse(sibling.alive)
                self.assertEqual(
                    process.calls,
                    [
                        ("join", 0.25),
                        ("is_alive",),
                        ("terminate",),
                        ("join", 0.25),
                        ("is_alive",),
                        ("kill",),
                        ("join", 0.25),
                        ("is_alive",),
                    ],
                )

    def test_process_crash_releases_lock_and_bound_root_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            launch_log = root / "crash-launches.log"
            context = multiprocessing.get_context("fork")
            results = context.Queue()
            process = context.Process(
                target=run_campaign_process,
                args=(args, probes, tenferro, launch_log, results),
                kwargs={"crash_on_first": True},
            )
            process.start()
            process.join(20)
            self.assertEqual(process.exitcode, 17)
            self.assertEqual(launch_log.read_text().splitlines(), ["launch"])
            recovery = FakeCommandRunner()
            self.assertEqual(
                load_runner()._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=recovery,
                ),
                2,
            )
            self.assertEqual(recovery.calls, [])
            terminal = json.loads(
                (args.artifact_root / "allocation.json").read_text()
            )
            self.assertEqual(terminal["validity_state"], "INCONCLUSIVE")
            self.assertEqual(
                terminal["invalid_reason"],
                "allocation execution interrupted before launch 1",
            )
            ledger = json.loads(args.ledger.read_text())
            self.assertIsNone(ledger["active_attempt_id"])

    def test_terminal_mutation_matrix_is_rejected_without_launches(self) -> None:
        runner = load_runner()

        def mutations(terminal):
            cases = []

            def changed(name, edit):
                payload = copy.deepcopy(terminal)
                edit(payload)
                cases.append((name, payload))

            changed("drop", lambda value: value["observations"].pop())
            changed(
                "extra",
                lambda value: value["observations"].append(
                    copy.deepcopy(value["observations"][-1])
                ),
            )
            changed(
                "reorder",
                lambda value: value["observations"].__setitem__(
                    slice(0, 2), reversed(value["observations"][:2])
                ),
            )
            changed(
                "duplicate-launch",
                lambda value: value["observations"][1].__setitem__(
                    "launch_index", value["observations"][0]["launch_index"]
                ),
            )
            changed("role", lambda value: value["observations"][0].__setitem__("role", "candidate"))
            changed("case", lambda value: value["observations"][0].__setitem__("case", "foreign"))
            changed("order", lambda value: value["observations"][0].__setitem__("order", "B/A"))
            changed("position", lambda value: value["observations"][0].__setitem__("position", 2))
            changed(
                "observation",
                lambda value: value["observations"][0].__setitem__("observation", 2),
            )
            changed(
                "count",
                lambda value: value["observations"][0]["record"].__setitem__(
                    "allocation_count", 999
                ),
            )
            changed(
                "bytes",
                lambda value: value["observations"][0]["record"].__setitem__(
                    "allocated_bytes", 999
                ),
            )
            changed("gate", lambda value: value.__setitem__("gate", "FAIL"))
            changed("launch-count", lambda value: value.__setitem__("launch_count", 167))
            changed(
                "build",
                lambda value: value["tenferro_builds"]["candidate"].__setitem__(
                    "head", "0" * 40
                ),
            )
            changed(
                "lock",
                lambda value: value["role_locks"].__setitem__("candidate", "0" * 64),
            )
            changed(
                "executable",
                lambda value: value["executable_identities"]["candidate"].__setitem__(
                    "sha256", "0" * 64
                ),
            )
            changed("top-extra", lambda value: value.__setitem__("extra", None))
            return cases

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
            self.assertEqual(
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=commands,
                ),
                0,
            )
            terminal = json.loads((args.artifact_root / "allocation.json").read_text())
            for name, mutated in mutations(terminal):
                with self.subTest(name=name):
                    protocol.atomic_write_json(
                        args.artifact_root / "allocation.json", mutated
                    )
                    recovery = FakeCommandRunner()
                    with self.assertRaises(protocol.ProtocolError):
                        runner._run_comparison(
                            args,
                            probe_manifests=probes,
                            tenferro_manifests=tenferro,
                            command_runner=recovery,
                        )
                    self.assertEqual(recovery.calls, [])
            protocol.atomic_write_json(args.artifact_root / "allocation.json", terminal)

    def test_persisted_terminal_rejects_duplicate_nonfinite_and_noncanonical_json(self) -> None:
        runner = load_runner()
        for corruption in ("duplicate", "nonfinite", "noncanonical"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
                self.assertEqual(
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    ),
                    0,
                )
                path = args.artifact_root / "allocation.json"
                content = path.read_text()
                if corruption == "duplicate":
                    content = content.replace("{", '{"gate":"PASS",', 1)
                elif corruption == "nonfinite":
                    content = content.replace("{", '{"poison":NaN,', 1)
                else:
                    content = json.dumps(json.loads(content)) + "\n"
                path.write_text(content)
                with self.assertRaises(protocol.ProtocolError):
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=FakeCommandRunner(),
                    )

    def test_probe_repetitions_require_exact_integer_type(self) -> None:
        runner = load_runner()
        for invalid in (4096.0, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                args, probes, tenferro = fixture(root)
                probes["candidate"]["repetitions"] = invalid
                commands = FakeCommandRunner()
                with self.assertRaises(protocol.ProtocolError):
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=commands,
                    )
                self.assertEqual(commands.calls, [])

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

    def test_normal_outcome_propagates_orchestrator_lock_close_failure(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            original_close = runner.EvidenceLock.close

            def fail_after_close(lock):
                original_close(lock)
                raise OSError("injected orchestrator lock close failure")

            with mock.patch.object(
                runner.EvidenceLock, "close", fail_after_close
            ), self.assertRaisesRegex(
                protocol.ProtocolError, "cannot release orchestrator lock"
            ):
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=FakeCommandRunner(
                        candidate_count=6, candidate_bytes=63
                    ),
                )

    def test_normal_outcome_propagates_pinned_resource_close_failure(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            original_close = runner.PinnedExecutable.close

            def fail_after_close(executable):
                original_close(executable)
                raise OSError("injected pinned executable close failure")

            with mock.patch.object(
                runner.PinnedExecutable, "close", fail_after_close
            ), self.assertRaisesRegex(
                protocol.ProtocolError, "cannot close allocation pinned resource"
            ):
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=FakeCommandRunner(
                        candidate_count=6, candidate_bytes=63
                    ),
                )

    def test_normal_recovery_propagates_root_close_failure(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
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
            original_close = runner.PinnedDirectory.close

            def fail_after_close(handle):
                original_close(handle)
                raise OSError("injected recovery root close failure")

            recovery = FakeCommandRunner()
            with mock.patch.object(
                runner.PinnedDirectory, "close", fail_after_close
            ), self.assertRaisesRegex(
                protocol.ProtocolError, "cannot close allocation recovery root"
            ):
                runner._run_comparison(
                    args,
                    probe_manifests=probes,
                    tenferro_manifests=tenferro,
                    command_runner=recovery,
                )
            self.assertEqual(recovery.calls, [])

    def test_normal_outcome_preserves_lock_close_control_exception(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args, probes, tenferro = fixture(root)
            original_close = runner.EvidenceLock.close
            interruption = KeyboardInterrupt("interrupt lock close")

            def interrupt_after_close(lock):
                original_close(lock)
                raise interruption

            caught = None
            with mock.patch.object(
                runner.EvidenceLock, "close", interrupt_after_close
            ):
                try:
                    runner._run_comparison(
                        args,
                        probe_manifests=probes,
                        tenferro_manifests=tenferro,
                        command_runner=FakeCommandRunner(
                            candidate_count=6, candidate_bytes=63
                        ),
                    )
                except BaseException as error:
                    caught = error
            self.assertIs(caught, interruption)

    def test_finalization_control_exception_identity_is_preserved_with_recovery(self) -> None:
        runner = load_runner()
        for name, expected_exit in (
            (runner.FINALIZATION_STAGE, 0),
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

    def test_recovery_marker_control_exception_preserves_identity_and_traceback(self) -> None:
        runner = load_runner()
        for exception_type in (KeyboardInterrupt, SystemExit):
            for committed in (False, True):
                with self.subTest(
                    exception_type=exception_type.__name__, committed=committed
                ), tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    args, probes, tenferro = fixture(root)
                    commands = FakeCommandRunner(candidate_count=6, candidate_bytes=63)
                    original_atomic_write = runner.protocol.atomic_write_json_at

                    def leave_marker_absent(directory_fd, selected_name, payload):
                        if selected_name == runner.FINALIZATION_MARKER:
                            raise OSError("leave recovery marker absent")
                        return original_atomic_write(directory_fd, selected_name, payload)

                    with mock.patch.object(
                        runner.protocol,
                        "atomic_write_json_at",
                        side_effect=leave_marker_absent,
                    ), self.assertRaises(protocol.ProtocolError):
                        runner._run_comparison(
                            args,
                            probe_manifests=probes,
                            tenferro_manifests=tenferro,
                            command_runner=commands,
                        )
                    self.assertEqual(len(commands.calls), 168)

                    interruption = exception_type("interrupt recovery marker")
                    injected_traceback = None

                    def interrupt_marker(directory_fd, selected_name, payload):
                        nonlocal injected_traceback
                        if selected_name == runner.FINALIZATION_MARKER:
                            if committed:
                                original_atomic_write(
                                    directory_fd, selected_name, payload
                                )
                            try:
                                raise interruption
                            except BaseException as caught:
                                injected_traceback = caught.__traceback__
                                raise
                        return original_atomic_write(
                            directory_fd, selected_name, payload
                        )

                    original_close = runner.PinnedDirectory.close

                    def failing_close(handle):
                        original_close(handle)
                        raise OSError("secondary recovery close failure")

                    recovery = FakeCommandRunner()
                    caught = None
                    caught_traceback = None
                    try:
                        with mock.patch.object(
                            runner.protocol,
                            "atomic_write_json_at",
                            side_effect=interrupt_marker,
                        ), mock.patch.object(
                            runner.PinnedDirectory, "close", failing_close
                        ):
                            runner._run_comparison(
                                args,
                                probe_manifests=probes,
                                tenferro_manifests=tenferro,
                                command_runner=recovery,
                            )
                    except BaseException as error:
                        caught = error
                        caught_traceback = error.__traceback__
                    self.assertIs(caught, interruption)
                    self.assertIsNotNone(caught_traceback)
                    while caught_traceback.tb_next is not None:
                        caught_traceback = caught_traceback.tb_next
                    self.assertIs(caught_traceback, injected_traceback)
                    self.assertEqual(recovery.calls, [])
                    self.assertEqual(len(commands.calls), 168)

                    partial_files = {
                        path.name for path in args.artifact_root.iterdir()
                    }
                    self.assertIn(runner.FINALIZATION_STAGE, partial_files)
                    self.assertEqual(
                        runner.FINALIZATION_MARKER in partial_files, committed
                    )
                    persisted = json.loads(
                        (args.artifact_root / "allocation.json").read_text()
                    )
                    self.assertEqual(persisted["validity_state"], "RUNNING")
                    ledger = json.loads(args.ledger.read_text())
                    self.assertEqual(ledger["active_attempt_id"], args.attempt_id)

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
                    terminal = json.loads(
                        (args.artifact_root / "allocation.json").read_text()
                    )
                    self.assertEqual(
                        (terminal["validity_state"], terminal["gate"]),
                        ("COMPLETE", "PASS"),
                    )
                    self.assertEqual(terminal["launch_count"], 168)

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
                protocol.atomic_write_json(path, tenferro[role])
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

    def test_read_only_completed_attempt_api_delegates_to_owning_validators(self):
        runner = load_runner()
        terminal = {"validity_state": "COMPLETE"}
        with mock.patch.object(runner, "_read_json", return_value=terminal), mock.patch.object(
            runner, "_validate_terminal_allocation", return_value=3
        ) as validate, mock.patch.object(
            runner, "_require_closed_allocation_attempt"
        ) as closed:
            result = runner.validate_completed_attempt(
                pathlib.Path("/evidence/attempt"), {},
                comparison_kind="common-lock-normalized", attempt_id=2,
                probe_manifests={}, tenferro_manifests={},
            )
        self.assertEqual(result, 3)
        self.assertIs(validate.call_args.kwargs["validate_live_sources"], True)
        closed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
