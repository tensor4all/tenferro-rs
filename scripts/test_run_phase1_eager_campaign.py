#!/usr/bin/env python3
"""Whole-campaign contract tests for the atomic Phase 2E timing runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import signal
import subprocess
import tempfile
import unittest

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
    ) -> None:
        self.invalid_at = invalid_at
        self.timeout_at = timeout_at
        self.survive_term_at = survive_term_at
        self.statistical_result = statistical_result
        self.raise_at = raise_at
        self.raise_exception = raise_exception
        self.launch_count = 0
        self.launches = []
        self.processes = {}
        self.killed = {}

    def __call__(self, argv, **kwargs):
        self.launch_count += 1
        if self.raise_at == self.launch_count:
            raise self.raise_exception or RuntimeError("injected launch failure")
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

    def signal_group(self, pid, requested_signal):
        process = self.processes[pid]
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
        if is_sentinel or self.statistical_result == "PASS":
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
        return runner.run_campaign(
            args,
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

    def test_process_timeout_does_not_kill_after_term_succeeds(self):
        args = self.arguments()
        fake = FakeProcessFactory(timeout_at=1)

        code = self.execute(args, fake)

        self.assertEqual(code, 2)
        process = fake.processes[10_001]
        self.assertEqual(process.returncode, -15)
        self.assertNotIn(process.pid, fake.killed)

    def test_every_launch_records_complete_sealed_environment_and_four_roles(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        self.assertEqual(self.execute(args, fake), 0)

        expected = protocol.runtime_environment(
            path=fake.launches[0]["env"]["PATH"],
            home=fake.launches[0]["env"]["HOME"],
            criterion_home=str(args.criterion_root.resolve()),
        )
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
                roles.extend(run["role"] for run in validity["runs"])
        self.assertEqual(set(roles), set(protocol.RUN_ROLES))
        self.assertEqual(len(roles), 336)
        first_log = next(args.artifact_root.rglob("sentinel_before.stdout.log"))
        preamble = json.loads(first_log.read_text().splitlines()[0])
        self.assertEqual(preamble["environment"], expected)
        self.assertEqual(
            preamble["environment_sha256"], protocol.sha256_json(expected)
        )

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

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.execute(args, fake)

        self.assertIs(caught.exception, interruption)
        manifest = json.loads((args.artifact_root / "campaign.json").read_text())
        self.assertEqual(manifest["validity_state"], "INCONCLUSIVE")
        self.assertIn("KeyboardInterrupt", manifest["invalid"]["reason"])

    def test_criterion_estimate_copy_is_byte_exact_and_rejects_symlink(self):
        source = self.base / "estimate.json"
        destination = self.base / "copied.json"
        payload = b'{"mean":{"point_estimate":0.125}}\n'
        source.write_bytes(payload)

        runner._copy_regular(source, destination)

        self.assertEqual(destination.read_bytes(), payload)
        link = self.base / "estimate-link.json"
        link.symlink_to(source)
        with self.assertRaises(protocol.ProtocolError):
            runner._copy_regular(link, self.base / "must-not-exist.json")

    def test_terminal_atomic_record_failure_returns_one(self):
        args = self.arguments()
        fake = FakeProcessFactory(invalid_at=1)
        original = protocol.atomic_write_json

        def fail_terminal(path, payload):
            if (
                path.name == "campaign.json"
                and payload.get("validity_state") == "INCONCLUSIVE"
            ):
                raise protocol.AtomicWriteError("terminal write failed", committed=False)
            return original(path, payload)

        self.assertEqual(self.execute(args, fake, atomic_writer=fail_terminal), 1)

    def test_complete_registration_atomic_failure_returns_one(self):
        args = self.arguments()
        fake = FakeProcessFactory()
        original = protocol.atomic_write_json
        complete_writes = 0

        def fail_registered_complete(path, payload):
            nonlocal complete_writes
            if (
                path.name == "campaign.json"
                and payload.get("validity_state") == "COMPLETE"
            ):
                complete_writes += 1
                if complete_writes == 2:
                    raise protocol.AtomicWriteError(
                        "registered terminal write failed", committed=False
                    )
            return original(path, payload)

        self.assertEqual(
            self.execute(args, fake, atomic_writer=fail_registered_complete), 1
        )

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
        self.assertEqual(manifest["validity_state"], "COMPLETE")

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
            "--ledger",
            "--artifact-root",
            "--criterion-root",
        ):
            self.assertIn(required, options)


if __name__ == "__main__":
    unittest.main()
