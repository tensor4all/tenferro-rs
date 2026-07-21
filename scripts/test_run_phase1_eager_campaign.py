#!/usr/bin/env python3
"""Whole-campaign contract tests for the atomic Phase 2E timing runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol


SCRIPT = pathlib.Path(__file__).with_name("run_phase1_eager_campaign.py")
SPEC = importlib.util.spec_from_file_location("phase2e_timing_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_build_manifests(
    root: pathlib.Path, comparison_kind: str
) -> tuple[pathlib.Path, pathlib.Path]:
    tool_dir = root / "tools"
    tool_dir.mkdir(parents=True)
    for name in ("git", "cargo", "rustc"):
        tool = tool_dir / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
    tools = build.resolve_toolchain(str(tool_dir.resolve()))
    target = "x86_64-unknown-linux-gnu"
    config_chain = [{"path": ".cargo/config.toml", "sha256": "1" * 64}]
    paths = []
    baseline_role = {
        "direct-current-main": "direct-current-main-baseline",
        "common-lock-normalized": "common-lock-normalized-baseline",
    }[comparison_kind]
    for identity, role in (("baseline", baseline_role), ("candidate", "candidate")):
        target_dir = (root / f"target-{identity}").resolve()
        executable = target_dir / "release/deps/eager_dispatch_baseline"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(f"{identity} executable".encode())
        executable.chmod(0o755)
        environment = protocol.cargo_environment(
            path=tools.path,
            home=str((root / "home").resolve()),
            cargo_home=str((root / "cargo-home").resolve()),
            target_dir=str(target_dir),
        )
        toolchain = {
            name: {
                "path": str(getattr(tools, name).path),
                "sha256": getattr(tools, name).sha256,
                **({} if name == "git" else {"version": f"{name} 1.90.0"}),
            }
            for name in ("git", "cargo", "rustc")
        }
        common_lock = comparison_kind == "common-lock-normalized"
        lock_sha = "8" * 64 if common_lock or identity == "candidate" else "7" * 64
        payload = {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "toolchain": toolchain,
            "target": target,
            "profile": "bench",
            "requested_features": list(build.REQUESTED_FEATURES),
            "provider": "Faer",
            "benchmark_sha256": "2" * 64,
            "benchmark_stanza_sha256": "3" * 64,
            "command_template": list(build.BENCH_COMMAND),
            "config_chain_sha256": protocol.sha256_json(config_chain),
            "role": role,
            "head": ("c" if identity == "candidate" else "d") * 40,
            "tracked_tree_sha256": "5" * 64,
            "resolved_features_sha256": "6" * 64,
            "lock_sha256": lock_sha,
            "worktree": str((root / f"worktree-{identity}").resolve()),
            "target_dir": str(target_dir),
            "executable": str(executable),
            "executable_sha256": protocol.sha256_file(executable),
            "validity_state": "COMPLETE",
            "source_delta": list(build._ROLE_SOURCE_DELTAS[role]),
            "commands": [
                command.to_manifest()
                for command in build.build_command_plan(target, tools.cargo)
            ],
            "environment": environment,
            "cargo_config_chain": config_chain,
        }
        path = (root / f"{identity}-build.json").resolve()
        write_json(path, payload)
        paths.append(path)
    return paths[0], paths[1]


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeProcess:
    def __init__(self, factory, ordinal, argv, stdout, stderr, environment) -> None:
        self.factory = factory
        self.ordinal = ordinal
        self.argv = argv
        self.stdout = stdout
        self.stderr = stderr
        self.environment = environment
        self.pid = 10_000 + ordinal
        self.returncode = None
        self.polls = 0
        self.timed_out = factory.timeout_at == ordinal
        self.survives_term = factory.survive_term_at == ordinal
        stdout.write(f"stdout launch {ordinal}\n")
        stderr.write(f"stderr launch {ordinal}\n")
        stdout.flush()
        stderr.flush()
        factory._write_criterion(argv, environment)

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.timed_out:
            return None
        self.polls += 1
        if self.polls == 1:
            return None
        self.returncode = 1 if self.factory.invalid_at == self.ordinal else 0
        return self.returncode

    def wait(self, timeout=None):
        if (
            self.factory.wait_exception is not None
            and not self.factory.wait_exception_raised
        ):
            self.factory.wait_exception_raised = True
            raise self.factory.wait_exception
        if self.returncode is not None:
            return self.returncode
        if self.survives_term and not self.factory.killed.get(self.pid):
            raise subprocess.TimeoutExpired(self.argv, timeout)
        self.returncode = -9 if self.factory.killed.get(self.pid) else -15
        return self.returncode


class FakeProcessFactory:
    def __init__(
        self,
        *,
        invalid_at=None,
        timeout_at=None,
        survive_term_at=None,
        statistical_result="PASS",
        raise_at=None,
        raise_exception=None,
        target_interval=None,
        wait_exception=None,
        signal_exception=None,
        swap_executable_at=None,
        mutate_executable_at=None,
        executable_path=None,
        swap_artifact_root_at=None,
        artifact_root=None,
        swap_criterion_root_at=None,
        criterion_root=None,
    ) -> None:
        self.invalid_at = invalid_at
        self.timeout_at = timeout_at
        self.survive_term_at = survive_term_at
        self.statistical_result = statistical_result
        self.raise_at = raise_at
        self.raise_exception = raise_exception
        self.target_interval = target_interval
        self.wait_exception = wait_exception
        self.signal_exception = signal_exception
        self.swap_executable_at = swap_executable_at
        self.mutate_executable_at = mutate_executable_at
        self.executable_path = executable_path
        self.swap_artifact_root_at = swap_artifact_root_at
        self.artifact_root = artifact_root
        self.swap_criterion_root_at = swap_criterion_root_at
        self.criterion_root = criterion_root
        self.detached_roots = {}
        self.wait_exception_raised = False
        self.signal_exception_raised = False
        self.launch_count = 0
        self.launches = []
        self.processes = {}
        self.killed = {}

    def __call__(self, argv, **kwargs):
        self.launch_count += 1
        if self.raise_at == self.launch_count:
            raise self.raise_exception or RuntimeError("injected launch failure")
        if self.swap_executable_at == self.launch_count:
            replacement = self.executable_path.with_name("replacement-benchmark")
            replacement.write_bytes(b"replacement executable")
            replacement.chmod(0o755)
            os.replace(replacement, self.executable_path)
        if self.mutate_executable_at == self.launch_count:
            self.executable_path.write_bytes(b"mutated executable")
        if self.swap_artifact_root_at == self.launch_count:
            self._swap_root("artifact", self.artifact_root)
        if self.swap_criterion_root_at == self.launch_count:
            self._swap_root("criterion", self.criterion_root)
        process = FakeProcess(
            self,
            self.launch_count,
            list(argv),
            kwargs["stdout"],
            kwargs["stderr"],
            dict(kwargs["env"]),
        )
        self.launches.append({"argv": list(argv), **kwargs})
        self.processes[process.pid] = process
        return process

    def _swap_root(self, name, path):
        detached = path.with_name(f"{path.name}-detached")
        outside = path.with_name(f"{path.name}-outside")
        path.rename(detached)
        outside.mkdir()
        path.symlink_to(outside, target_is_directory=True)
        self.detached_roots[name] = (detached, outside)

    def signal_group(self, pid, requested_signal):
        process = self.processes[pid]
        if self.signal_exception is not None and not self.signal_exception_raised:
            self.signal_exception_raised = True
            raise self.signal_exception
        if requested_signal == signal.SIGTERM and not process.survives_term:
            process.returncode = -15
        if requested_signal == signal.SIGKILL:
            self.killed[pid] = True
            process.returncode = -9

    def _write_estimate(self, path: pathlib.Path, lower: float, upper: float) -> None:
        write_json(
            path,
            {
                "mean": {
                    "confidence_interval": {
                        "lower_bound": lower,
                        "upper_bound": upper,
                    },
                    "point_estimate": (lower + upper) / 2.0,
                }
            },
        )

    def _write_criterion(self, argv, environment) -> None:
        benchmark = argv[argv.index("--bench") + 1]
        option = "--save-baseline" if "--save-baseline" in argv else "--baseline"
        name = argv[argv.index(option) + 1]
        directory = runner.criterion_directory(
            pathlib.Path(environment["CRITERION_HOME"]), benchmark
        )
        if option == "--save-baseline":
            self._write_estimate(directory / name / "estimates.json", -0.01, 0.01)
            return
        self._write_estimate(directory / "new/estimates.json", -0.01, 0.01)
        is_sentinel = "sentinel" in name
        if not is_sentinel and self.target_interval is not None:
            lower, upper = self.target_interval
        elif is_sentinel or self.statistical_result == "PASS":
            lower, upper = -0.01, 0.02
        elif self.statistical_result == "INCONCLUSIVE":
            lower, upper = -0.01, 0.06
        else:
            pair = int(name.split("-p", 1)[1].split("-", 1)[0])
            lower, upper = ((-0.08, -0.06) if pair == 2 else (0.06, 0.08))
        self._write_estimate(directory / "change/estimates.json", lower, upper)


class AtomicCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temporary.name)
        self.clock = FakeClock()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self, *, comparison_kind="direct-current-main"):
        build_root = self.base / f"build-{len(list(self.base.glob('build-*')))}"
        baseline, candidate = make_build_manifests(build_root, comparison_kind)
        ledger = self.base / f"ledger-{len(list(self.base.glob('ledger-*')))}.json"
        value = protocol.new_ledger("c" * 40)
        if comparison_kind == "common-lock-normalized":
            value = protocol.open_attempt(value, "timing", "direct-current-main", 1)
            value = protocol.close_attempt(
                value, "timing", "direct-current-main", 1, "PASS"
            )
        protocol.atomic_write_json(ledger, value)
        ordinal = len(list(self.base.glob("artifact-*")))
        return argparse.Namespace(
            comparison_kind=comparison_kind,
            baseline_build_manifest=baseline,
            candidate_build_manifest=candidate,
            ledger=ledger,
            attempt_id=1,
            artifact_root=self.base / f"artifact-{ordinal}",
            criterion_root=self.base / f"criterion-{ordinal}",
            working_directory=SCRIPT.parent.parent,
            cpu=3,
            normalized_load_limit=0.25,
        )

    def execute(self, args, factory, **kwargs):
        load_provider = kwargs.pop("load_provider", lambda: 0.8)
        allowed_cpu_provider = kwargs.pop(
            "allowed_cpu_provider", lambda: {3, 4, 5, 6}
        )
        baseline = json.loads(args.baseline_build_manifest.read_text())
        candidate = json.loads(args.candidate_build_manifest.read_text())
        validated_builds = {
            baseline["role"]: baseline,
            candidate["role"]: candidate,
        }
        return runner._run_campaign(
            args,
            validated_builds=validated_builds,
            process_factory=factory,
            allowed_cpu_provider=allowed_cpu_provider,
            signal_process_group=factory.signal_group,
            monotonic=self.clock.monotonic,
            sleep=self.clock.sleep,
            affinity_provider=lambda _pid: {3},
            load_provider=load_provider,
            build_process_provider=lambda: [],
            **kwargs,
        )

    def test_valid_campaign_launches_exact_matrix_and_registers_classifier_outputs(self):
        args = self.arguments()
        fake = FakeProcessFactory()

        code = self.execute(args, fake)

        self.assertEqual(code, 0)
        self.assertEqual(fake.launch_count, 28 * 3 * 4)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "COMPLETE")
        self.assertEqual(manifest["statistical_result"], "PASS")
        self.assertEqual(set(manifest["cases"]), set(protocol.CANONICAL_CASES))
        self.assertEqual(
            set(manifest["classification_artifacts"]),
            {"classification.json", "summary.md"},
        )
        self.assertTrue((args.artifact_root / "classification.json").is_file())
        self.assertTrue((args.artifact_root / "summary.md").is_file())
        self.assertFalse(
            any("_rejected" in path.parts for path in args.artifact_root.rglob("*"))
        )

    def test_target_estimate_is_preserved_before_same_benchmark_sentinel(self):
        args = self.arguments()
        fake = FakeProcessFactory(target_interval=(0.06, 0.08))

        code = self.execute(args, fake)

        self.assertEqual(code, 3)
        estimate = runner.classification.read_change(
            args.artifact_root / "lazy_neg_1/pair1/change-estimates.json"
        )
        self.assertEqual(estimate[:2], (0.06, 0.08))

    def test_first_invalid_process_closes_whole_campaign_without_retry(self):
        args = self.arguments()
        fake = FakeProcessFactory(invalid_at=17)

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        self.assertEqual(fake.launch_count, 17)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
        self.assertIsNone(manifest["statistical_result"])
        self.assertEqual(manifest["invalid"]["role"], "sentinel_before")
        self.assertNotIn("_rejected", json.dumps(manifest))

    def test_fresh_roots_and_running_resume_are_rejected_before_launch(self):
        for root_kind in ("artifact", "criterion"):
            with self.subTest(root_kind=root_kind):
                args = self.arguments()
                root = getattr(args, f"{root_kind}_root")
                root.mkdir(parents=True)
                stale = root / (
                    "campaign.json" if root_kind == "artifact" else "stale"
                )
                stale.write_text("{}")
                fake = FakeProcessFactory()
                self.assertEqual(self.execute(args, fake), 1)
                self.assertEqual(fake.launch_count, 0)

    def test_identical_or_nested_roots_are_rejected(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                args = self.arguments()
                args.criterion_root = (
                    args.artifact_root
                    if not nested
                    else args.artifact_root / "criterion"
                )
                fake = FakeProcessFactory()
                self.assertEqual(self.execute(args, fake), 1)
                self.assertEqual(fake.launch_count, 0)

    def test_symlink_root_is_rejected_instead_of_followed(self):
        args = self.arguments()
        target = self.base / "artifact-target"
        target.mkdir()
        args.artifact_root.symlink_to(target, target_is_directory=True)
        fake = FakeProcessFactory()

        self.assertEqual(self.execute(args, fake), 1)
        self.assertEqual(fake.launch_count, 0)

    def test_invalid_cpu_is_rejected_before_ledger_registration(self):
        args = self.arguments()
        original = args.ledger.read_bytes()
        fake = FakeProcessFactory()

        self.assertEqual(
            self.execute(args, fake, allowed_cpu_provider=lambda: {4, 5}), 1
        )

        self.assertEqual(args.ledger.read_bytes(), original)
        self.assertEqual(fake.launch_count, 0)

    def test_startup_keyboard_interrupt_is_preserved_without_fd_leak(self):
        args = self.arguments()
        original = args.ledger.read_bytes()
        fake = FakeProcessFactory()
        interruption = KeyboardInterrupt("startup interrupted")
        before = len(os.listdir("/proc/self/fd"))

        with self.assertRaises(KeyboardInterrupt) as raised:
            self.execute(
                args,
                fake,
                allowed_cpu_provider=lambda: (_ for _ in ()).throw(interruption),
            )

        self.assertIs(raised.exception, interruption)
        self.assertEqual(args.ledger.read_bytes(), original)
        self.assertEqual(fake.launch_count, 0)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_pinned_atomic_json_preserves_primary_base_exception(self):
        root = self.base / "pinned-root"
        root.mkdir()
        handle = runner.PinnedDirectory(root)
        interruption = KeyboardInterrupt("atomic write interrupted")
        real_close = os.close
        close_calls = 0

        def close_parent_then_fail(descriptor):
            nonlocal close_calls
            close_calls += 1
            real_close(descriptor)
            if close_calls == 1:
                raise OSError("parent close failed")

        try:
            with mock.patch.object(
                runner.protocol,
                "atomic_write_json_at",
                side_effect=interruption,
            ), mock.patch.object(
                runner.os,
                "close",
                side_effect=close_parent_then_fail,
            ):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    handle.atomic_json("campaign.json", {})
            self.assertIs(raised.exception, interruption)
        finally:
            handle.close()

    def test_ledger_candidate_mismatch_is_rejected_before_mutation(self):
        args = self.arguments()
        mismatched = protocol.new_ledger("d" * 40)
        protocol.atomic_write_json(args.ledger, mismatched)
        original = args.ledger.read_bytes()
        fake = FakeProcessFactory()

        self.assertEqual(self.execute(args, fake), 1)

        self.assertEqual(args.ledger.read_bytes(), original)
        self.assertEqual(fake.launch_count, 0)

    def test_quiet_wait_is_bounded_to_300_one_second_polls(self):
        args = self.arguments()
        fake = FakeProcessFactory()

        code = self.execute(args, fake, load_provider=lambda: 8.0)

        self.assertEqual(code, 2)
        self.assertEqual(fake.launch_count, 0)
        self.assertEqual(self.clock.monotonic(), 1_300.0)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertIn("quiet-host-timeout", manifest["invalid"]["reason"])

    def test_process_timeout_terminates_then_kills_survivors(self):
        args = self.arguments()
        fake = FakeProcessFactory(timeout_at=1, survive_term_at=1)

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        self.assertEqual(fake.launch_count, 1)
        process = fake.processes[10_001]
        self.assertEqual(process.returncode, -9)
        self.assertTrue(fake.killed[process.pid])

    def test_process_timeout_kills_group_even_after_leader_term_succeeds(self):
        args = self.arguments()
        fake = FakeProcessFactory(timeout_at=1)

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        process = fake.processes[10_001]
        self.assertEqual(process.returncode, -9)
        self.assertTrue(fake.killed[process.pid])

    @unittest.skipUnless(
        os.name == "posix" and sys.platform.startswith("linux"),
        "requires Linux process groups and /proc",
    )
    def test_real_cleanup_kills_descendant_after_leader_exits(self):
        script = """
import os
import signal
import sys

leader_group = os.getpgrp()
reaper = os.fork()
if reaper == 0:
    os.setpgid(0, 0)
    for descriptor in (1, 2):
        try:
            os.close(descriptor)
        except OSError:
            pass
    child = os.fork()
    if child == 0:
        os.setpgid(0, leader_group)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open(sys.argv[1], "w", encoding="utf-8") as identity:
            identity.write(f"{os.getpid()} {os.getpgrp()}\\n")
            identity.flush()
            os.fsync(identity.fileno())
        while True:
            signal.pause()
    os.waitpid(child, 0)
    os._exit(0)

while True:
    signal.pause()
"""
        identity_path = self.base / "descendant.identity"
        identity_path.touch()
        process = subprocess.Popen(
            (sys.executable, "-c", script, str(identity_path)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        descendant = None
        try:
            deadline = time.monotonic() + 5
            last_identity = ""
            while time.monotonic() < deadline:
                try:
                    last_identity = identity_path.read_text()
                    fields = last_identity.split()
                    if len(fields) == 2:
                        descendant, process_group = map(int, fields)
                        break
                except (FileNotFoundError, ValueError):
                    pass
                if process.poll() is not None:
                    self.fail("leader exited before publishing identity")
                time.sleep(0.01)
            else:
                self.fail(
                    "descendant did not publish complete identity: "
                    f"{last_identity!r}"
                )
            self.assertEqual(process_group, process.pid)

            terminated, killed, failures = runner._terminate_group(
                process, os.killpg
            )

            self.assertTrue(terminated)
            self.assertTrue(killed)
            self.assertEqual(failures, [])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if not pathlib.Path(f"/proc/{descendant}").exists():
                    break
                time.sleep(0.01)
            else:
                self.fail("descendant process survived group cleanup")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            if descendant is not None:
                try:
                    os.kill(descendant, signal.SIGKILL)
                except OSError:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def test_timeout_wait_keyboard_interrupt_is_cleaned_and_propagated(self):
        args = self.arguments()
        interruption = KeyboardInterrupt("wait cancelled")
        fake = FakeProcessFactory(
            timeout_at=1,
            survive_term_at=1,
            wait_exception=interruption,
        )

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.execute(args, fake)

        self.assertIs(caught.exception, interruption)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")

    def test_timeout_signal_base_exception_is_cleaned_and_propagated(self):
        args = self.arguments()
        interruption = SystemExit("signal cancelled")
        fake = FakeProcessFactory(
            timeout_at=1,
            survive_term_at=1,
            signal_exception=interruption,
        )

        with self.assertRaises(SystemExit) as caught:
            self.execute(args, fake)

        self.assertIs(caught.exception, interruption)
        self.assertTrue(fake.killed[fake.processes[10_001].pid])

    def test_every_launch_records_complete_sealed_environment_and_four_roles(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        self.assertEqual(self.execute(args, fake), 0)

        expected = protocol.runtime_environment(
            path=fake.launches[0]["env"]["PATH"],
            home=fake.launches[0]["env"]["HOME"],
            criterion_home=fake.launches[0]["env"]["CRITERION_HOME"],
        )
        self.assertTrue(expected["CRITERION_HOME"].startswith("/proc/self/fd/"))
        self.assertEqual(
            {tuple(sorted(item["env"].items())) for item in fake.launches},
            {tuple(sorted(expected.items()))},
        )
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        roles = []
        for case in manifest["cases"].values():
            for pair in case["pairs"].values():
                validity = json.loads(
                    (args.artifact_root / pair["validity_path"]).read_text()
                )
                for run in validity["runs"]:
                    roles.append(run["role"])
                    self.assertEqual(
                        run["environment_sha256"],
                        protocol.sha256_json(run["environment"]),
                    )
                    self.assertEqual(run["environment"], expected)
                    self.assertEqual(run["argv"][0], run["executable"]["launch_path"])
                    self.assertEqual(
                        run["criterion_binding"], manifest["criterion_binding"]
                    )
                    self.assertEqual(
                        run["environment"]["CRITERION_HOME"],
                        run["criterion_binding"]["actual_home"],
                    )
        self.assertEqual(set(roles), set(protocol.RUN_ROLES))
        self.assertEqual(len(roles), 336)
        first_log = next(args.artifact_root.rglob("sentinel_before.stdout.log"))
        preamble = json.loads(first_log.read_text().splitlines()[0])
        self.assertEqual(preamble["environment"], expected)
        self.assertEqual(
            preamble["environment_sha256"], protocol.sha256_json(expected)
        )

    def test_runtime_environment_uses_authoritative_manifest_payload(self):
        args = self.arguments()
        candidate = json.loads(args.candidate_build_manifest.read_text())
        authoritative_environment = dict(candidate["environment"])
        candidate["environment"] = authoritative_environment
        tampered = json.loads(args.candidate_build_manifest.read_text())
        tampered["environment"]["HOME"] = "/attacker-controlled"
        write_json(args.candidate_build_manifest, tampered)

        environment = runner._runtime_environment(candidate, pathlib.Path("/proc/fd"))

        self.assertEqual(environment["HOME"], authoritative_environment["HOME"])

    def test_launch_exception_finalizes_readable_terminal_manifest(self):
        args = self.arguments()
        fake = FakeProcessFactory(raise_at=17)

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
        self.assertIn("injected launch failure", json.dumps(manifest))
        actual_prefix = {
            path.relative_to(args.artifact_root).as_posix()
            for path in args.artifact_root.rglob("*")
            if path.is_file() and path.name != "campaign.json"
        }
        self.assertEqual(set(manifest["prefix_inventory"]), actual_prefix)
        self.assertEqual(
            manifest["prefix_inventory"], manifest["artifact_inventory"]
        )

    def test_base_exception_is_finalized_then_re_raised_unchanged(self):
        args = self.arguments()
        interruption = KeyboardInterrupt("injected cancellation")
        fake = FakeProcessFactory(raise_at=1, raise_exception=interruption)
        before_fds = len(os.listdir("/proc/self/fd"))

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.execute(args, fake)

        self.assertIs(caught.exception, interruption)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
        self.assertIn("KeyboardInterrupt", manifest["invalid"]["reason"])
        self.assertEqual(len(os.listdir("/proc/self/fd")), before_fds)

    def test_terminal_write_failure_does_not_replace_base_exception(self):
        args = self.arguments()
        interruption = KeyboardInterrupt("original cancellation")
        fake = FakeProcessFactory(raise_at=1, raise_exception=interruption)

        def fail_terminal(path, payload):
            if (
                path.name == "campaign.json"
                and payload.get("validity_state") == "INCONCLUSIVE"
            ):
                raise protocol.AtomicWriteError("terminal failed", committed=False)

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.execute(args, fake, campaign_write_observer=fail_terminal)

        self.assertIs(caught.exception, interruption)

    def test_prefix_inventory_failure_does_not_replace_base_exception(self):
        args = self.arguments()
        interruption = KeyboardInterrupt("original cancellation")
        fake = FakeProcessFactory(raise_at=1, raise_exception=interruption)

        with mock.patch.object(
            runner.PinnedDirectory,
            "inventory",
            side_effect=OSError("inventory failed"),
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.execute(args, fake)

        self.assertIs(caught.exception, interruption)

    def test_criterion_estimate_copy_is_byte_exact_and_rejects_symlink(self):
        source_root = self.base / "criterion-copy-source"
        destination_root = self.base / "criterion-copy-destination"
        source_root.mkdir()
        destination_root.mkdir()
        source = source_root / "estimate.json"
        payload = b'{"mean":{"point_estimate":0.125}}\n'
        source.write_bytes(payload)
        source_handle = runner.PinnedDirectory(source_root)
        destination_handle = runner.PinnedDirectory(destination_root)
        try:
            runner._copy_regular_at(
                source_handle,
                "estimate.json",
                destination_handle.descriptor,
                "copied.json",
            )

            self.assertEqual((destination_root / "copied.json").read_bytes(), payload)
            (source_root / "estimate-link.json").symlink_to(source)
            with self.assertRaises(protocol.ProtocolError):
                runner._copy_regular_at(
                    source_handle,
                    "estimate-link.json",
                    destination_handle.descriptor,
                    "must-not-exist.json",
                )
        finally:
            destination_handle.close()
            source_handle.close()

    def test_executable_path_swap_cannot_redirect_launch(self):
        args = self.arguments()
        candidate = json.loads(args.candidate_build_manifest.read_text())
        executable = pathlib.Path(candidate["executable"])
        original_inode = executable.stat().st_ino
        fake = FakeProcessFactory(
            swap_executable_at=1,
            executable_path=executable,
        )

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        self.assertEqual(fake.launch_count, 1)
        launch = fake.launches[0]
        self.assertTrue(launch["argv"][0].startswith("/proc/self/fd/"))
        self.assertIn(int(launch["argv"][0].rsplit("/", 1)[1]), launch["pass_fds"])
        self.assertNotEqual(executable.stat().st_ino, original_inode)

    def test_executable_same_inode_mutation_during_run_is_invalid(self):
        args = self.arguments()
        candidate = json.loads(args.candidate_build_manifest.read_text())
        fake = FakeProcessFactory(
            mutate_executable_at=1,
            executable_path=pathlib.Path(candidate["executable"]),
        )

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        self.assertEqual(fake.launch_count, 1)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertIn("executable", manifest["invalid"]["reason"])

    @unittest.skipUnless(
        os.name == "posix" and sys.platform.startswith("linux"),
        "requires Linux /proc/self/fd executable launch",
    )
    def test_real_pinned_executable_ignores_path_replacement(self):
        executable = self.base / "pinned-script"
        executable.write_text("#!/bin/sh\nprintf 'old\\n'\n")
        executable.chmod(0o755)
        digest = protocol.sha256_file(executable)
        pinned = runner.PinnedExecutable.open(executable, digest)
        try:
            replacement = self.base / "replacement-script"
            replacement.write_text("#!/bin/sh\nprintf 'new\\n'\n")
            replacement.chmod(0o755)
            os.replace(replacement, executable)

            completed = subprocess.run(
                (str(pinned.launch_path),),
                pass_fds=(pinned.descriptor,),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "old\n")
            with self.assertRaisesRegex(protocol.ProtocolError, "identity changed"):
                pinned.validate()
        finally:
            pinned.close()

    @unittest.skipUnless(
        os.name == "posix"
        and sys.platform.startswith("linux")
        and hasattr(os, "memfd_create"),
        "requires Linux sealed memfd execution",
    )
    def test_sealed_executable_cannot_run_transient_source_mutation(self):
        executable = self.base / "sealed-source"
        side_effect = self.base / "malicious-ran"
        original = b"#!/bin/sh\nprintf 'safe\\n'\n"
        malicious = (
            "#!/bin/sh\n"
            f"touch {side_effect}\n"
            "printf 'malicious\\n'\n"
        ).encode()
        executable.write_bytes(original)
        executable.chmod(0o755)
        pinned = runner.PinnedExecutable.open(
            executable, protocol.sha256_file(executable)
        )
        try:
            executable.write_bytes(malicious)
            completed = subprocess.run(
                (str(pinned.launch_path),),
                pass_fds=(pinned.descriptor,),
                check=False,
                capture_output=True,
                text=True,
            )
            executable.write_bytes(original)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "safe\n")
            self.assertFalse(side_effect.exists())
            self.assertEqual(
                runner.fcntl.fcntl(pinned.descriptor, runner.fcntl.F_GET_SEALS),
                runner.EXECUTABLE_SEALS,
            )
            pinned.validate()
        finally:
            pinned.close()

    def test_sealed_executable_setup_preserves_base_exception_without_fd_leak(self):
        executable = self.base / "sealed-cancel-source"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        interruption = KeyboardInterrupt("sealing cancelled")
        before = len(os.listdir("/proc/self/fd"))

        with mock.patch.object(
            runner.fcntl,
            "fcntl",
            side_effect=interruption,
        ):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.PinnedExecutable.open(
                    executable, protocol.sha256_file(executable)
                )

        self.assertIs(raised.exception, interruption)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    @unittest.skipUnless(
        os.name == "posix" and sys.platform.startswith("linux"),
        "requires Linux wait/proc process monitoring",
    )
    def test_real_normal_exit_has_valid_endpoint_monitor_sample(self):
        executable = self.base / "normal-exit-script"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        pinned = runner.PinnedExecutable.open(
            executable, protocol.sha256_file(executable)
        )
        artifact_root = self.base / "real-monitor-artifacts"
        artifact_root.mkdir()
        artifact = runner.PinnedDirectory(artifact_root)
        pair = "case/pair1"
        pair_fd = artifact.open_directory(pair, create=True)
        os.close(pair_fd)
        selected_cpu = min(os.sched_getaffinity(0))
        try:
            record, reason = runner._process_record(
                command=(str(pinned.launch_path),),
                environment=dict(os.environ),
                cwd=self.base,
                artifact_root=artifact,
                pair_relative=pair,
                role="sentinel_before",
                identity="candidate",
                binary_sha=pinned.digest,
                executable=pinned,
                selected_cpu=selected_cpu,
                allowed_count=len(os.sched_getaffinity(0)),
                process_factory=subprocess.Popen,
                signal_process_group=os.killpg,
                monotonic=time.monotonic,
                sleep=time.sleep,
                affinity_provider=os.sched_getaffinity,
                load_provider=lambda: 0.0,
                build_process_provider=lambda: [],
                inherited_descriptors=(),
                root_validators=(artifact.validate_link,),
            )

            self.assertIsNone(reason)
            self.assertEqual(record["exit_status"], 0)
            self.assertEqual(record["monitor_samples"][0]["phase"], "start")
            self.assertEqual(record["monitor_samples"][-1]["phase"], "end")
            self.assertEqual(
                record["monitor_samples"][-1]["observed_affinity"],
                str(selected_cpu),
            )
        finally:
            artifact.close()
            pinned.close()

    @unittest.skipUnless(
        os.name == "posix" and sys.platform.startswith("linux"),
        "requires Linux process groups and /proc",
    )
    def test_real_normal_exit_kills_and_invalidates_surviving_descendant(self):
        executable = self.base / "normal-exit-with-child.py"
        executable.write_text(
            """#!/usr/bin/env python3
import os
import signal
import sys
import time

child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    for descriptor in (1, 2):
        os.close(descriptor)
    with open(sys.argv[1], "w", encoding="utf-8") as identity:
        identity.write(str(os.getpid()))
        identity.flush()
        os.fsync(identity.fileno())
    while True:
        signal.pause()
while not os.path.exists(sys.argv[1]):
    time.sleep(0.001)
"""
        )
        executable.chmod(0o755)
        identity = self.base / "normal-child.pid"
        pinned = runner.PinnedExecutable.open(
            executable, protocol.sha256_file(executable)
        )
        artifact_root = self.base / "normal-child-artifacts"
        artifact_root.mkdir()
        artifact = runner.PinnedDirectory(artifact_root)
        pair = "case/pair1"
        os.close(artifact.open_directory(pair, create=True))
        selected_cpu = min(os.sched_getaffinity(0))
        descendant = None
        try:
            cleanup_started = time.monotonic()
            record, reason = runner._process_record(
                command=(str(pinned.launch_path), str(identity)),
                environment=dict(os.environ),
                cwd=self.base,
                artifact_root=artifact,
                pair_relative=pair,
                role="sentinel_before",
                identity="candidate",
                binary_sha=pinned.digest,
                executable=pinned,
                selected_cpu=selected_cpu,
                allowed_count=len(os.sched_getaffinity(0)),
                process_factory=subprocess.Popen,
                signal_process_group=os.killpg,
                monotonic=time.monotonic,
                sleep=time.sleep,
                affinity_provider=os.sched_getaffinity,
                load_provider=lambda: 0.0,
                build_process_provider=lambda: [],
                inherited_descriptors=(),
                root_validators=(artifact.validate_link,),
            )
            cleanup_elapsed = time.monotonic() - cleanup_started
            descendant = int(identity.read_text())

            self.assertIn("benchmark-process-survivor", reason)
            self.assertTrue(record["process_group_cleanup"]["survivor_observed"])
            self.assertGreaterEqual(
                cleanup_elapsed, runner.TERMINATION_GRACE_SECONDS
            )
            deadline = time.monotonic() + 5
            while pathlib.Path(f"/proc/{descendant}").exists():
                if time.monotonic() >= deadline:
                    self.fail("normal-exit descendant survived cleanup")
                time.sleep(0.01)
        finally:
            if descendant is not None:
                try:
                    os.kill(descendant, signal.SIGKILL)
                except OSError:
                    pass
            artifact.close()
            pinned.close()

    def test_artifact_root_swap_cannot_redirect_campaign_writes(self):
        args = self.arguments()
        fake = FakeProcessFactory(
            swap_artifact_root_at=1,
            artifact_root=args.artifact_root.resolve(),
        )

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        detached, outside = fake.detached_roots["artifact"]
        self.assertEqual(list(outside.iterdir()), [])
        manifest = json.loads((detached / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")

    def test_fresh_root_is_pinned_before_emptiness_validation(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        replacement = args.artifact_root.with_name("fresh-root-replacement")

        def swap_after_pin(label, handle):
            if label != "artifact":
                return
            detached = handle.logical_path.with_name("fresh-root-detached")
            handle.logical_path.rename(detached)
            replacement.mkdir()
            handle.logical_path.symlink_to(replacement, target_is_directory=True)

        code = self.execute(args, fake, root_pin_observer=swap_after_pin)

        self.assertEqual(code, 1)
        self.assertEqual(fake.launch_count, 0)
        self.assertEqual(list(replacement.iterdir()), [])

    def test_fresh_root_parent_swap_cannot_redirect_creation(self):
        parent = self.base / "fresh-parent"
        parent.mkdir()
        root = parent / "artifact"
        detached = self.base / "fresh-parent-detached"
        replacement = self.base / "fresh-parent-replacement"
        swapped = False

        def swap_parent(opened_path, _descriptor):
            nonlocal swapped
            if opened_path == parent and not swapped:
                swapped = True
                parent.rename(detached)
                replacement.mkdir()
                parent.symlink_to(replacement, target_is_directory=True)

        with self.assertRaisesRegex(protocol.ProtocolError, "identity changed"):
            runner.PinnedDirectory.create_fresh(
                root, component_observer=swap_parent
            )

        self.assertTrue((detached / "artifact").is_dir())
        self.assertEqual(list(replacement.iterdir()), [])

    def test_finalization_reachable_state_matrix_is_exact(self):
        expected = {
            ("RUNNING", False, False, False, True): "NO_RESUME",
            ("RUNNING", True, True, False, True): "RUNNING",
            ("RUNNING", True, True, True, True): "RUNNING",
            ("RUNNING", True, True, False, False): "RUNNING",
            ("RUNNING", True, True, True, False): "RUNNING",
            ("COMPLETE", True, True, False, False): "COMPLETE",
            ("COMPLETE", False, True, False, False): "COMPLETE",
            ("COMPLETE", False, False, False, False): "COMPLETE",
        }
        for validity in ("RUNNING", "COMPLETE"):
            for marker in (False, True):
                for stage in (False, True):
                    for publish in (False, True):
                        for ledger_active in (False, True):
                            key = (
                                validity,
                                marker,
                                stage,
                                publish,
                                ledger_active,
                            )
                            with self.subTest(state=key):
                                if key in expected:
                                    self.assertEqual(
                                        runner._finalization_state_kind(*key),
                                        expected[key],
                                    )
                                else:
                                    with self.assertRaises(protocol.ProtocolError):
                                        runner._finalization_state_kind(*key)

    def test_component_close_failure_preserves_primary_and_closes_child(self):
        root = self.base / "component-close-root"
        nested = root / "nested"
        nested.mkdir(parents=True)
        before = len(os.listdir("/proc/self/fd"))
        interruption = KeyboardInterrupt("old descriptor close interrupted")
        real_close = os.close
        calls = 0

        def close_then_interrupt(descriptor):
            nonlocal calls
            calls += 1
            real_close(descriptor)
            if calls == 1:
                raise interruption

        with mock.patch.object(runner.os, "close", side_effect=close_then_interrupt):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.PinnedDirectory.create_fresh(root / "new")
        self.assertIs(raised.exception, interruption)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            calls = 0
            with mock.patch.object(
                runner.classification.os,
                "close",
                side_effect=close_then_interrupt,
            ):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    runner.classification._open_retained_directory_path(
                        pathlib.Path(f"/proc/self/fd/{root_fd}/nested")
                    )
            self.assertIs(raised.exception, interruption)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before + 1)
        finally:
            os.close(root_fd)

    def test_criterion_root_swap_cannot_redirect_benchmark_scratch(self):
        args = self.arguments()
        fake = FakeProcessFactory(
            swap_criterion_root_at=1,
            criterion_root=args.criterion_root.resolve(),
        )

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        detached, outside = fake.detached_roots["criterion"]
        self.assertEqual(list(outside.iterdir()), [])
        self.assertTrue(any(detached.rglob("estimates.json")))

    def test_root_swap_during_final_child_cannot_produce_pass(self):
        args = self.arguments()
        fake = FakeProcessFactory(
            swap_artifact_root_at=28 * 3 * 4,
            artifact_root=args.artifact_root.resolve(),
        )

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        self.assertEqual(fake.launch_count, 28 * 3 * 4)
        detached, outside = fake.detached_roots["artifact"]
        self.assertEqual(list(outside.iterdir()), [])
        manifest = json.loads((detached / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")

    def test_terminal_atomic_record_failure_returns_one(self):
        args = self.arguments()
        fake = FakeProcessFactory(invalid_at=1)

        def fail_terminal(path, payload):
            if (
                path.name == "campaign.json"
                and payload.get("validity_state") == "INCONCLUSIVE"
            ):
                raise protocol.AtomicWriteError("terminal write failed", committed=False)

        self.assertEqual(
            self.execute(args, fake, campaign_write_observer=fail_terminal), 1
        )

    def test_running_checkpoint_failure_is_finalized_inconclusive(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        running_writes = 0

        def fail_second_running_checkpoint(path, payload):
            nonlocal running_writes
            if (
                path.name == "campaign.json"
                and payload.get("validity_state") == "RUNNING"
            ):
                running_writes += 1
                if running_writes == 2:
                    raise protocol.AtomicWriteError(
                        "checkpoint write failed", committed=False
                    )

        self.assertEqual(
            self.execute(
                args,
                fake,
                campaign_write_observer=fail_second_running_checkpoint,
            ),
            2,
        )
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
        self.assertIn("checkpoint write failed", manifest["invalid"]["reason"])

    def test_only_final_complete_write_failure_leaves_running_manifest(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        complete_writes = 0

        def fail_only_complete(path, payload):
            nonlocal complete_writes
            if (
                path.name == "campaign.json"
                and payload.get("validity_state") == "COMPLETE"
            ):
                complete_writes += 1
                raise protocol.AtomicWriteError(
                    "final terminal write failed", committed=False
                )

        self.assertEqual(
            self.execute(args, fake, campaign_write_observer=fail_only_complete),
            1,
        )
        self.assertEqual(complete_writes, 1)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "RUNNING")
        ledger = json.loads(args.ledger.read_text())
        self.assertIsNone(ledger["active_attempt_id"])

    def test_finalization_failures_recover_without_remeasurement(self):
        for phase in ("prepared", "ledger_closed", "published", "directory_synced"):
            with self.subTest(phase=phase):
                args = self.arguments()
                first = FakeProcessFactory()
                injected = False

                def interrupt(selected_phase):
                    nonlocal injected
                    if selected_phase == phase and not injected:
                        injected = True
                        raise OSError(f"injected finalization failure: {phase}")

                self.assertEqual(
                    self.execute(args, first, finalization_observer=interrupt),
                    1,
                )
                self.assertEqual(first.launch_count, 28 * 3 * 4)

                recovery = FakeProcessFactory()
                self.assertEqual(self.execute(args, recovery), 0)
                self.assertEqual(recovery.launch_count, 0)
                manifest = json.loads(
                    (args.artifact_root / "campaign.json").read_text()
                )
                self.assertEqual(manifest["validity_state"], "COMPLETE")
                ledger = json.loads(args.ledger.read_text())
                self.assertIsNone(ledger["active_attempt_id"])

    def test_published_recovery_revalidates_all_classification_artifacts(self):
        args = self.arguments()
        first = FakeProcessFactory()

        def interrupt(phase):
            if phase == "published":
                raise OSError("stop after publish")

        self.assertEqual(
            self.execute(args, first, finalization_observer=interrupt), 1
        )
        classification_path = args.artifact_root / "classification.json"
        classification_path.write_text("{}\n")

        recovery = FakeProcessFactory()
        self.assertEqual(self.execute(args, recovery), 1)
        self.assertEqual(recovery.launch_count, 0)
        self.assertTrue(
            (args.artifact_root / runner.FINALIZATION_MARKER).is_file()
        )

    def test_published_recovery_rejects_missing_stage_as_unreachable(self):
        args = self.arguments()
        first = FakeProcessFactory()

        def interrupt(phase):
            if phase == "published":
                raise OSError("stop after publish")

        self.assertEqual(
            self.execute(args, first, finalization_observer=interrupt), 1
        )
        (args.artifact_root / runner.FINALIZATION_STAGE).unlink()

        recovery = FakeProcessFactory()
        self.assertEqual(self.execute(args, recovery), 1)
        self.assertEqual(recovery.launch_count, 0)
        self.assertTrue(
            (args.artifact_root / runner.FINALIZATION_MARKER).is_file()
        )

    def test_publish_partial_must_be_the_stage_hard_link(self):
        args = self.arguments()
        first = FakeProcessFactory()
        real_replace = os.replace

        def fail_publish(source, destination, *positional, **keywords):
            if source == runner.FINALIZATION_PUBLISH:
                raise OSError("stop with publish link")
            return real_replace(source, destination, *positional, **keywords)

        with mock.patch.object(runner.os, "replace", side_effect=fail_publish):
            self.assertEqual(self.execute(args, first), 1)
        stage = args.artifact_root / runner.FINALIZATION_STAGE
        publish = args.artifact_root / runner.FINALIZATION_PUBLISH
        content = publish.read_bytes()
        publish.unlink()
        publish.write_bytes(content)
        self.assertNotEqual(stage.stat().st_ino, publish.stat().st_ino)

        recovery = FakeProcessFactory()
        self.assertEqual(self.execute(args, recovery), 1)
        self.assertEqual(recovery.launch_count, 0)
        self.assertTrue(publish.is_file())

    def test_committed_marker_durability_failure_recovers_without_measurement(self):
        args = self.arguments()
        first = FakeProcessFactory()
        original = protocol.atomic_write_json_at
        injected = False

        def committed_marker_failure(directory_fd, name, payload):
            nonlocal injected
            result = original(directory_fd, name, payload)
            if name == runner.FINALIZATION_MARKER and not injected:
                injected = True
                raise protocol.AtomicWriteDurabilityError(
                    "marker installed but parent fsync failed"
                )
            return result

        with mock.patch.object(
            runner.protocol,
            "atomic_write_json_at",
            side_effect=committed_marker_failure,
        ):
            self.assertEqual(self.execute(args, first), 1)

        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        ledger = json.loads(args.ledger.read_text())
        self.assertEqual(manifest["validity_state"], "RUNNING")
        self.assertEqual(ledger["active_attempt_id"], args.attempt_id)
        self.assertTrue((args.artifact_root / runner.FINALIZATION_STAGE).is_file())
        self.assertTrue((args.artifact_root / runner.FINALIZATION_MARKER).is_file())

        recovery = FakeProcessFactory()
        self.assertEqual(self.execute(args, recovery), 0)
        self.assertEqual(recovery.launch_count, 0)

    def test_raw_marker_post_replace_interruption_preserves_transaction(self):
        for interruption in (
            KeyboardInterrupt("marker fsync interrupted"),
            SystemExit("marker fsync exited"),
        ):
            with self.subTest(interruption=type(interruption).__name__):
                args = self.arguments()
                first = FakeProcessFactory()
                original = protocol.atomic_write_json_at
                injected = False

                def interrupt_after_marker(directory_fd, name, payload):
                    nonlocal injected
                    result = original(directory_fd, name, payload)
                    if name == runner.FINALIZATION_MARKER and not injected:
                        injected = True
                        raise interruption
                    return result

                with mock.patch.object(
                    runner.protocol,
                    "atomic_write_json_at",
                    side_effect=interrupt_after_marker,
                ):
                    with self.assertRaises(type(interruption)) as raised:
                        self.execute(args, first)
                self.assertIs(raised.exception, interruption)
                manifest = json.loads(
                    (args.artifact_root / "campaign.json").read_text()
                )
                ledger = json.loads(args.ledger.read_text())
                self.assertEqual(manifest["validity_state"], "RUNNING")
                self.assertEqual(ledger["active_attempt_id"], args.attempt_id)
                self.assertTrue(
                    (args.artifact_root / runner.FINALIZATION_STAGE).is_file()
                )
                self.assertTrue(
                    (args.artifact_root / runner.FINALIZATION_MARKER).is_file()
                )

                recovery = FakeProcessFactory()
                self.assertEqual(self.execute(args, recovery), 0)
                self.assertEqual(recovery.launch_count, 0)

    def test_recovery_rejects_symlinked_parent_component(self):
        args = self.arguments()
        parent = self.base / "recovery-parent"
        parent.mkdir()
        args.artifact_root = parent / "artifact"
        first = FakeProcessFactory()

        def interrupt(phase):
            if phase == "prepared":
                raise OSError("stop with active finalization")

        self.assertEqual(
            self.execute(args, first, finalization_observer=interrupt), 1
        )
        detached = self.base / "recovery-parent-detached"
        parent.rename(detached)
        parent.symlink_to(detached, target_is_directory=True)

        recovery = FakeProcessFactory()
        self.assertEqual(self.execute(args, recovery), 1)
        self.assertEqual(recovery.launch_count, 0)

    def test_missing_finalization_file_still_fsyncs_root(self):
        root_path = self.base / "unlink-fsync-root"
        root_path.mkdir()
        root = runner.PinnedDirectory(root_path)
        calls = []
        try:
            with mock.patch.object(
                runner.os,
                "unlink",
                side_effect=FileNotFoundError("already absent"),
            ), mock.patch.object(
                runner.os,
                "fsync",
                side_effect=lambda descriptor: calls.append(descriptor),
            ):
                runner._unlink_finalization_file(root, "absent")
            self.assertEqual(calls, [root.descriptor])
        finally:
            root.close()

    def test_completed_campaign_rerun_is_idempotent_without_launch(self):
        args = self.arguments()
        first = FakeProcessFactory()
        self.assertEqual(self.execute(args, first), 0)

        rerun = FakeProcessFactory()
        self.assertEqual(self.execute(args, rerun), 0)
        self.assertEqual(rerun.launch_count, 0)

    def test_mismatched_finalization_partial_is_rejected_without_launch(self):
        args = self.arguments()
        first = FakeProcessFactory()

        def interrupt(phase):
            if phase == "prepared":
                raise OSError("stop after finalization prepare")

        self.assertEqual(
            self.execute(args, first, finalization_observer=interrupt), 1
        )
        stage = args.artifact_root / runner.FINALIZATION_STAGE
        payload = json.loads(stage.read_text())
        payload["completed_at"] = "tampered"
        write_json(stage, payload)

        recovery = FakeProcessFactory()
        self.assertEqual(self.execute(args, recovery), 1)
        self.assertEqual(recovery.launch_count, 0)

    def test_finalization_rename_fsync_and_cleanup_failures_are_recoverable(self):
        for operation in (
            "rename",
            "dir_fsync",
            "cleanup",
            "marker_unlink_fsync",
        ):
            with self.subTest(operation=operation):
                args = self.arguments()
                first = FakeProcessFactory()
                real_replace = os.replace
                real_fsync = os.fsync
                real_unlink = os.unlink
                injected = False
                marker_seen = False

                def replace_once(source, destination, *positional, **keywords):
                    nonlocal injected
                    if operation == "rename" and source == runner.FINALIZATION_PUBLISH:
                        injected = True
                        raise OSError("injected final campaign rename failure")
                    return real_replace(source, destination, *positional, **keywords)

                def unlink_once(path, *positional, **keywords):
                    nonlocal injected
                    if (
                        operation == "cleanup"
                        and path == runner.FINALIZATION_MARKER
                        and not injected
                    ):
                        injected = True
                        raise OSError("injected final marker cleanup failure")
                    return real_unlink(path, *positional, **keywords)

                def fsync_once(descriptor):
                    nonlocal injected, marker_seen
                    entries = set()
                    if operation in {"dir_fsync", "marker_unlink_fsync"}:
                        try:
                            entries = set(os.listdir(descriptor))
                        except OSError:
                            pass
                        marker_seen = marker_seen or (
                            runner.FINALIZATION_MARKER in entries
                        )
                    if operation == "dir_fsync" and not injected:
                        if (
                            runner.FINALIZATION_MARKER in entries
                            and runner.FINALIZATION_STAGE in entries
                            and runner.FINALIZATION_PUBLISH not in entries
                        ):
                            injected = True
                            raise OSError("injected final directory fsync failure")
                    if (
                        operation == "marker_unlink_fsync"
                        and marker_seen
                        and not injected
                    ):
                        if (
                            runner.FINALIZATION_MARKER not in entries
                            and runner.FINALIZATION_STAGE in entries
                            and "campaign.json" in entries
                        ):
                            injected = True
                            raise OSError("injected marker unlink fsync failure")
                    return real_fsync(descriptor)

                with mock.patch.object(
                    runner.os, "replace", side_effect=replace_once
                ), mock.patch.object(
                    runner.os, "fsync", side_effect=fsync_once
                ), mock.patch.object(runner.os, "unlink", side_effect=unlink_once):
                    self.assertEqual(self.execute(args, first), 1)
                self.assertTrue(injected)

                recovery = FakeProcessFactory()
                self.assertEqual(self.execute(args, recovery), 0)
                self.assertEqual(recovery.launch_count, 0)

    def test_complete_manifest_is_written_once_after_ledger_close(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        original = protocol.atomic_write_json
        events = []

        def record_ledger_close(path, payload):
            if path == args.ledger and payload.get("active_attempt_id") is None:
                events.append("ledger-close")

            return original(path, payload)

        def record_campaign_complete(path, payload):
            if (
                path.name == "campaign.json"
                and payload.get("validity_state") == "COMPLETE"
            ):
                events.append("campaign-complete")
                self.assertEqual(
                    set(payload["classification_artifacts"]),
                    {"classification.json", "summary.md"},
                )

        self.assertEqual(
            self.execute(
                args,
                fake,
                atomic_writer=record_ledger_close,
                campaign_write_observer=record_campaign_complete,
            ),
            0,
        )
        self.assertEqual(events, ["ledger-close", "campaign-complete"])

    def test_terminal_ledger_close_atomic_failure_returns_one(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        original = protocol.atomic_write_json

        def fail_ledger_close(path, payload):
            if (
                path == args.ledger
                and payload.get("attempts")
                and payload.get("active_attempt_id") is None
            ):
                raise protocol.AtomicWriteError(
                    "terminal ledger close failed", committed=False
                )
            return original(path, payload)

        self.assertEqual(self.execute(args, fake, atomic_writer=fail_ledger_close), 1)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "RUNNING")
        recovery = FakeProcessFactory()
        self.assertEqual(self.execute(args, recovery), 0)
        self.assertEqual(recovery.launch_count, 0)

    def test_exact_exit_mapping_for_statistical_results(self):
        for result, expected in (("PASS", 0), ("FAIL", 3), ("INCONCLUSIVE", 4)):
            with self.subTest(result=result):
                args = self.arguments()
                fake = FakeProcessFactory(statistical_result=result)
                self.assertEqual(self.execute(args, fake), expected)
                manifest = json.loads(
                    (args.artifact_root / "campaign.json").read_text()
                )
                self.assertEqual(manifest["statistical_result"], result)

    def test_cli_has_no_pair_retry_option_and_requires_atomic_inputs(self):
        parser = runner.build_argument_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        self.assertNotIn("--max-attempts", options)
        for required in (
            "--comparison-kind",
            "--baseline-build-manifest",
            "--candidate-build-manifest",
            "--repository",
            "--build-evidence-root",
            "--build-scratch-root",
            "--candidate-commit",
            "--controlled-path",
            "--controlled-home",
            "--controlled-cargo-home",
            "--ledger",
            "--artifact-root",
            "--criterion-root",
        ):
            self.assertIn(required, options)

    def test_public_runner_has_no_validated_build_bypass(self):
        public = inspect.signature(runner.run_campaign)
        private = inspect.signature(runner._run_campaign)

        self.assertNotIn("validated_builds", public.parameters)
        self.assertIn("validated_builds", private.parameters)

    def test_public_runner_rejects_fabricated_builds_before_ledger_mutation(self):
        args = self.arguments()
        candidate = json.loads(args.candidate_build_manifest.read_text())
        environment = candidate["environment"]
        args.repository = SCRIPT.parent.parent.resolve()
        args.build_evidence_root = args.baseline_build_manifest.parent
        args.build_scratch_root = self.base / "missing-authoritative-scratch"
        args.candidate_commit = "c" * 40
        args.controlled_path = environment["PATH"]
        args.controlled_home = pathlib.Path(environment["HOME"])
        args.controlled_cargo_home = pathlib.Path(environment["CARGO_HOME"])
        original = args.ledger.read_bytes()
        fake = FakeProcessFactory()

        code = runner.run_campaign(args, process_factory=fake)

        self.assertEqual(code, 1)
        self.assertEqual(args.ledger.read_bytes(), original)
        self.assertEqual(fake.launch_count, 0)


if __name__ == "__main__":
    unittest.main()
