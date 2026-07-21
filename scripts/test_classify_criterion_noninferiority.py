#!/usr/bin/env python3
"""Contract tests for the terminal protocol-v2 Criterion classifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol


SCRIPT = pathlib.Path(__file__).with_name("classify_criterion_noninferiority.py")
SPEC = importlib.util.spec_from_file_location("criterion_classifier", SCRIPT)
classifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(classifier)


CRITERION_SETTINGS = {
    "warm_up_seconds": 2,
    "measurement_seconds": 5,
    "sample_size": 100,
    "confidence_level": 0.95,
}


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_estimate(
    path: pathlib.Path,
    lower: float = -0.01,
    upper: float = 0.02,
    point: float = 0.005,
) -> None:
    write_json(
        path,
        {
            "mean": {
                "confidence_interval": {
                    "lower_bound": lower,
                    "upper_bound": upper,
                },
                "point_estimate": point,
            }
        },
    )


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(path: pathlib.Path) -> dict[str, str]:
    return {"sha256": sha256(path)}


def monitor_samples(selected_cpu: int = 3, load: float = 0.10) -> list[dict]:
    return [
        {
            "sequence": sequence,
            "phase": "start" if sequence == 0 else "end" if sequence == 7 else "periodic",
            "monotonic_seconds": float(sequence),
            "observed_affinity": str(selected_cpu),
            "normalized_load": load,
            "cargo_processes": [],
            "rustc_processes": [],
        }
        for sequence in range(8)
    ]


def make_run(
    role: str,
    binary: str,
    binary_sha256: str,
    *,
    selected_cpu: int = 3,
    load: float = 0.10,
) -> dict:
    return {
        "role": role,
        "binary": binary,
        "binary_sha256": binary_sha256,
        "validity_state": "COMPLETE",
        "exit_status": 0,
        "stdout_artifact": f"{role}.stdout.log",
        "stderr_artifact": f"{role}.stderr.log",
        "process_started_monotonic": 0.0,
        "process_ended_monotonic": 7.0,
        "monitor_samples": monitor_samples(selected_cpu, load),
    }


def make_build_manifests(root: pathlib.Path) -> dict[str, dict[str, str]]:
    tool_dir = root / "tools"
    tool_dir.mkdir()
    for name in ("git", "cargo", "rustc"):
        path = tool_dir / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    tools = build.resolve_toolchain(str(tool_dir))
    target = "x86_64-unknown-linux-gnu"
    config_chain = [{"path": ".cargo/config.toml", "sha256": "1" * 64}]
    records = {}
    for identity in ("baseline", "candidate"):
        role = "direct-current-main-baseline" if identity == "baseline" else "candidate"
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
            "head": ("d" if identity == "baseline" else "c") * 40,
            "tracked_tree_sha256": "5" * 64,
            "resolved_features_sha256": "6" * 64,
            "lock_sha256": ("7" if identity == "baseline" else "8") * 64,
            "worktree": str((root / f"worktree-{identity}").resolve()),
            "target_dir": str(target_dir),
            "executable": str(executable),
            "executable_sha256": protocol.sha256_file(executable),
            "validity_state": "COMPLETE",
            "source_delta": ["frozen-benchmark-harness"] if identity == "baseline" else [],
            "commands": [
                command.to_manifest()
                for command in build.build_command_plan(target, tools.cargo)
            ],
            "environment": environment,
            "cargo_config_chain": config_chain,
        }
        path = (root / f"{identity}-build.json").resolve()
        write_json(path, payload)
        records[identity] = {
            "path": str(path),
            "sha256": sha256(path),
            "role": role,
            "executable_sha256": payload["executable_sha256"],
        }
    return records


def make_campaign(root: pathlib.Path, build_root: pathlib.Path) -> dict:
    build_manifests = make_build_manifests(build_root)
    cases: dict[str, dict] = {}
    inventory: dict[str, dict[str, str]] = {}
    for case, benchmark in protocol.CANONICAL_CASES.items():
        pair_entries = {}
        for pair, order in enumerate(protocol.PAIR_ORDERS, start=1):
            identities = (
                ("candidate", "baseline", "candidate", "candidate")
                if order == "A/B"
                else ("candidate", "candidate", "baseline", "candidate")
            )
            pair_dir = root / case / f"pair{pair}"
            pair_dir.mkdir(parents=True)
            change = pair_dir / "change-estimates.json"
            sentinel = pair_dir / "sentinel-change-estimates.json"
            write_estimate(change)
            write_estimate(sentinel)

            runs = [
                make_run(
                    role,
                    binary,
                    build_manifests[binary]["executable_sha256"],
                )
                for role, binary in zip(protocol.RUN_ROLES, identities)
            ]
            local_artifacts: dict[str, dict[str, str]] = {}
            for run in runs:
                for stream_name in ("stdout_artifact", "stderr_artifact"):
                    name = run[stream_name]
                    path = pair_dir / name
                    path.write_text(f"{case}/pair{pair}/{name}\n", encoding="utf-8")
                    local_artifacts[name] = artifact_record(path)

            monitor_path = pair_dir / "monitor-samples.json"
            write_json(
                monitor_path,
                {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "case": case,
                    "pair": pair,
                    "runs": {
                        run["role"]: copy.deepcopy(run["monitor_samples"])
                        for run in runs
                    },
                },
            )
            local_artifacts.update(
                {
                    "change-estimates.json": artifact_record(change),
                    "sentinel-change-estimates.json": artifact_record(sentinel),
                    "monitor-samples.json": artifact_record(monitor_path),
                }
            )
            validity = {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "case": case,
                "pair": pair,
                "order": order,
                "selected_cpu": 3,
                "allowed_cpu_count": 8,
                "validity_state": "COMPLETE",
                "runs": runs,
                "artifacts": local_artifacts,
            }
            validity_path = pair_dir / "validity.json"
            write_json(validity_path, validity)

            for name, record in local_artifacts.items():
                relative = (pair_dir / name).relative_to(root).as_posix()
                inventory[relative] = copy.deepcopy(record)
            validity_relative = validity_path.relative_to(root).as_posix()
            inventory[validity_relative] = artifact_record(validity_path)
            pair_entries[str(pair)] = {
                "order": order,
                "validity_path": validity_relative,
                "validity_sha256": sha256(validity_path),
            }
        cases[case] = {
            "benchmark": benchmark,
            "statistical_result": "PASS",
            "pairs": pair_entries,
        }

    campaign = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": sha256(pathlib.Path(protocol.__file__)),
        "classifier_sha256": sha256(SCRIPT),
        "candidate_sha": "c" * 40,
        "comparison_kind": "direct-current-main",
        "build_manifests": build_manifests,
        "selected_cpu": 3,
        "allowed_cpus": "0-7",
        "allowed_cpu_count": 8,
        "normalized_load_limit": 0.25,
        "thread_environment": dict(protocol.THREAD_ENV),
        "orders": list(protocol.PAIR_ORDERS),
        "criterion": CRITERION_SETTINGS,
        "validity_state": "COMPLETE",
        "statistical_result": "PASS",
        "completed_at": "2026-07-20T00:00:00+00:00",
        "cases": cases,
        "artifact_inventory": inventory,
        "classification_artifacts": None,
    }
    write_json(root / "campaign.json", campaign)
    return campaign


class ClassificationBoundaryTests(unittest.TestCase):
    def test_huge_cpu_range_is_typed_without_materializing(self) -> None:
        program = """
import resource
resource.setrlimit(resource.RLIMIT_AS, (64 * 1024 * 1024, 64 * 1024 * 1024))
from scripts import classify_criterion_noninferiority as classifier
from scripts import phase2e_protocol as protocol
try:
    classifier.parse_cpu_inventory("0-20000000")
except protocol.ProtocolError:
    raise SystemExit(0)
except BaseException as error:
    print(type(error).__name__)
    raise SystemExit(2)
raise SystemExit(3)
"""
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=SCRIPT.parent.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_cpu_inventory_requires_canonical_unique_ranges(self) -> None:
        self.assertEqual(classifier.parse_cpu_inventory("0,2-4,7"), {0, 2, 3, 4, 7})
        for inventory in ("1,1", "0-2,2-4", "0,1", "2,0", "3-3"):
            with self.subTest(inventory=inventory):
                with self.assertRaisesRegex(protocol.ProtocolError, "CPU inventory"):
                    classifier.parse_cpu_inventory(inventory)

    def test_regular_snapshot_allows_unrelated_ancestor_sibling_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "evidence"
            evidence.mkdir()
            path = evidence / "artifact.json"
            path.write_text('{"value":"original"}\n')
            sibling = root / "unrelated-sibling"
            original_read = os.read
            changed = False

            def racing_read(descriptor, size):
                nonlocal changed
                content = original_read(descriptor, size)
                if content and not changed:
                    changed = True
                    sibling.write_text("unrelated\n")
                    sibling.unlink()
                return content

            with mock.patch.object(classifier.os, "read", side_effect=racing_read):
                content, _digest = classifier._read_regular_snapshot(path)

            self.assertEqual(content, b'{"value":"original"}\n')

    def test_read_regular_at_preserves_read_base_exception_over_close_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "artifact"
            path.write_bytes(b"payload")
            directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            original_close = os.close
            before_fds = len(os.listdir("/proc/self/fd"))

            def close_then_fail(descriptor):
                original_close(descriptor)
                raise OSError("injected close failure")

            try:
                with mock.patch.object(
                    classifier.os, "read", side_effect=KeyboardInterrupt("read stop")
                ), mock.patch.object(
                    classifier.os, "close", side_effect=close_then_fail
                ):
                    with self.assertRaisesRegex(KeyboardInterrupt, "read stop"):
                        classifier._read_regular_at(directory_fd, path.name)
                self.assertEqual(len(os.listdir("/proc/self/fd")), before_fds)
            finally:
                original_close(directory_fd)

    def test_stage_write_preserves_base_exception_over_close_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            original_close = os.close
            before_fds = len(os.listdir("/proc/self/fd"))

            def close_then_fail(descriptor):
                original_close(descriptor)
                raise OSError("injected close failure")

            try:
                with mock.patch.object(
                    classifier.os, "write", side_effect=KeyboardInterrupt("stop")
                ), mock.patch.object(
                    classifier.os, "close", side_effect=close_then_fail
                ):
                    with self.assertRaisesRegex(KeyboardInterrupt, "stop"):
                        classifier._stage_file_at(directory_fd, "stable", b"payload")
                self.assertEqual(os.listdir(temporary), [])
                self.assertEqual(len(os.listdir("/proc/self/fd")), before_fds)
            finally:
                original_close(directory_fd)

    def test_classify_case_requires_exactly_three_intervals(self) -> None:
        interval = {"lower": -0.01, "upper": 0.02}
        for count in (0, 1, 2, 4):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "exactly three"):
                    classifier.classify_case([interval] * count)

    def test_regular_snapshot_rejects_swap_between_open_and_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "artifact.json"
            replacement = root / "replacement.json"
            path.write_text('{"value":"original"}\n')
            replacement.write_text('{"value":"replacement"}\n')
            original_read = os.read
            swapped = False

            def racing_read(descriptor, size):
                nonlocal swapped
                content = original_read(descriptor, size)
                if content and not swapped:
                    swapped = True
                    os.replace(replacement, path)
                return content

            self.assertTrue(hasattr(classifier, "_read_regular_snapshot"))
            with mock.patch.object(classifier.os, "read", side_effect=racing_read):
                with self.assertRaisesRegex(protocol.ProtocolError, "changed"):
                    classifier._read_regular_snapshot(path)

    def test_regular_snapshot_rejects_parent_component_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            parent = root / "evidence"
            replacement_parent = root / "replacement"
            parent.mkdir()
            replacement_parent.mkdir()
            path = parent / "artifact.json"
            path.write_text('{"value":"original"}\n')
            (replacement_parent / "artifact.json").write_text(
                '{"value":"replacement"}\n'
            )
            original_read = os.read
            swapped = False

            def racing_read(descriptor, size):
                nonlocal swapped
                content = original_read(descriptor, size)
                if content and not swapped:
                    swapped = True
                    parent.rename(root / "detached")
                    replacement_parent.rename(parent)
                return content

            with mock.patch.object(classifier.os, "read", side_effect=racing_read):
                with self.assertRaisesRegex(protocol.ProtocolError, "changed"):
                    classifier._read_regular_snapshot(path)

    def test_script_entrypoint_can_import_protocol_module(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=SCRIPT.parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_pass_requires_every_upper_endpoint_at_or_below_threshold(self):
        intervals = [
            {"lower": -0.01, "upper": 0.05},
            {"lower": 0.00, "upper": 0.049},
            {"lower": 0.01, "upper": 0.05},
        ]
        self.assertEqual(classifier.classify_case(intervals), "PASS")

    def test_fail_requires_two_lower_endpoints_strictly_above_threshold(self):
        intervals = [
            {"lower": 0.050001, "upper": 0.06},
            {"lower": 0.050001, "upper": 0.07},
            {"lower": -0.01, "upper": 0.01},
        ]
        self.assertEqual(classifier.classify_case(intervals), "FAIL")

    def test_other_interval_combinations_are_inconclusive(self):
        intervals = [
            {"lower": 0.05, "upper": 0.06},
            {"lower": 0.050001, "upper": 0.07},
            {"lower": -0.01, "upper": 0.01},
        ]
        self.assertEqual(classifier.classify_case(intervals), "INCONCLUSIVE")

    def test_campaign_result_uses_fail_then_inconclusive_precedence(self):
        self.assertEqual(
            classifier.campaign_result({"a": "PASS", "b": "FAIL"}), "FAIL"
        )
        self.assertEqual(
            classifier.campaign_result({"a": "PASS", "b": "INCONCLUSIVE"}),
            "INCONCLUSIVE",
        )
        self.assertEqual(classifier.campaign_result({"a": "PASS"}), "PASS")

    def test_b_over_a_interval_is_inverted_to_candidate_over_baseline(self):
        lower, upper, point = classifier.invert_interval(-0.10, -0.06, -0.08)
        self.assertAlmostEqual(lower, 1.0 / 0.94 - 1.0)
        self.assertAlmostEqual(upper, 1.0 / 0.90 - 1.0)
        self.assertAlmostEqual(point, 1.0 / 0.92 - 1.0)

    def test_sentinel_drift_band_boundaries_are_frozen(self):
        self.assertFalse(classifier.sentinel_breached(0.05, 0.08))
        self.assertFalse(classifier.sentinel_breached(-0.08, -0.05))
        self.assertTrue(classifier.sentinel_breached(0.050001, 0.08))
        self.assertTrue(classifier.sentinel_breached(-0.08, -0.050001))

    def test_estimate_booleans_huge_integers_and_nonfinite_values_are_typed(self):
        invalid_values = (True, 10**1000, float("nan"), float("inf"))
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "change.json"
            for value in invalid_values:
                with self.subTest(value=repr(value)):
                    write_estimate(path, lower=value, upper=value, point=value)
                    with self.assertRaises(protocol.ProtocolError):
                        classifier.read_change(path)


class ProtocolV2CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temporary.name)
        self.root = self.base / "timing"
        self.root.mkdir()
        self.build_root = self.base / "builds"
        self.build_root.mkdir()
        self.campaign = make_campaign(self.root, self.build_root)
        self.campaign_path = self.root / "campaign.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_campaign(self) -> None:
        write_json(self.campaign_path, self.campaign)

    def rewrite_validity(self, validity_path: pathlib.Path, validity: dict) -> None:
        write_json(validity_path, validity)
        relative = validity_path.relative_to(self.root).as_posix()
        record = artifact_record(validity_path)
        self.campaign["artifact_inventory"][relative] = record
        case = validity["case"]
        pair = str(validity["pair"])
        self.campaign["cases"][case]["pairs"][pair]["validity_sha256"] = record[
            "sha256"
        ]

    def rewrite_build(self, identity: str, payload: dict) -> None:
        record = self.campaign["build_manifests"][identity]
        path = pathlib.Path(record["path"])
        write_json(path, payload)
        record["sha256"] = sha256(path)

    def remove_unregistered_outputs(self) -> None:
        for name in ("classification.json", "summary.md"):
            path = self.root / name
            if path.exists():
                path.unlink()

    def test_complete_campaign_is_recomputed_rendered_and_hashed(self) -> None:
        result = classifier.classify_campaign(self.campaign_path, self.root)

        self.assertEqual(result["statistical_result"], "PASS")
        self.assertEqual(len(result["cases"]), 28)
        classification_path = self.root / "classification.json"
        summary_path = self.root / "summary.md"
        self.assertTrue(classification_path.is_file())
        self.assertTrue(summary_path.is_file())
        self.assertEqual(
            json.loads(classification_path.read_text()),
            {
                key: value
                for key, value in result.items()
                if key != "output_artifacts"
            },
        )
        self.assertIn("campaign=PASS", summary_path.read_text())
        self.assertEqual(
            set(result["output_artifacts"]),
            {"classification.json", "summary.md"},
        )
        for name, record in result["output_artifacts"].items():
            self.assertEqual(record["path"], name)
            self.assertEqual(record["sha256"], sha256(self.root / name))

    def test_terminal_view_classifies_while_disk_manifest_remains_running(self) -> None:
        terminal = copy.deepcopy(self.campaign)
        for record in self.campaign["cases"].values():
            record["statistical_result"] = None
        self.campaign["validity_state"] = "RUNNING"
        self.campaign["statistical_result"] = None
        self.campaign["completed_at"] = ""
        self.write_campaign()

        result = classifier.classify_terminal_view(
            self.campaign_path, terminal, self.root
        )

        persisted = json.loads(self.campaign_path.read_text())
        self.assertEqual(persisted["validity_state"], "RUNNING")
        self.assertEqual(result["statistical_result"], "PASS")
        self.assertTrue((self.root / "classification.json").is_file())

    def test_terminal_view_rejects_changes_outside_allowed_final_fields(self) -> None:
        terminal = copy.deepcopy(self.campaign)
        terminal["selected_cpu"] = 4
        for record in self.campaign["cases"].values():
            record["statistical_result"] = None
        self.campaign["validity_state"] = "RUNNING"
        self.campaign["statistical_result"] = None
        self.campaign["completed_at"] = ""
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "terminal view"):
            classifier.classify_terminal_view(
                self.campaign_path, terminal, self.root
            )

    def test_terminal_view_root_fd_survives_logical_root_swap(self) -> None:
        terminal = copy.deepcopy(self.campaign)
        for record in self.campaign["cases"].values():
            record["statistical_result"] = None
        self.campaign["validity_state"] = "RUNNING"
        self.campaign["statistical_result"] = None
        self.campaign["completed_at"] = ""
        self.write_campaign()
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        detached = self.base / "timing-detached"
        outside = self.base / "timing-outside"
        try:
            self.root.rename(detached)
            outside.mkdir()
            self.root.symlink_to(outside, target_is_directory=True)

            result = classifier.classify_terminal_view(
                self.campaign_path,
                terminal,
                self.root,
                root_descriptor=descriptor,
            )

            self.assertEqual(result["statistical_result"], "PASS")
            self.assertTrue((detached / "classification.json").is_file())
            self.assertEqual(list(outside.iterdir()), [])
        finally:
            os.close(descriptor)

    def test_protocol_one_is_rejected(self) -> None:
        self.campaign["protocol_version"] = 1
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "protocol version"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_duplicate_top_level_campaign_key_is_rejected(self) -> None:
        raw = self.campaign_path.read_text()
        self.campaign_path.write_text(
            raw.replace("{", '{\n  "protocol_version": 2,', 1)
        )

        with self.assertRaisesRegex(protocol.ProtocolError, "duplicate JSON key"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_duplicate_nested_estimate_key_is_rejected(self) -> None:
        estimate = self.root / "lazy_neg_1/pair1/change-estimates.json"
        estimate.write_text(
            '{"mean":{"confidence_interval":{"lower_bound":-0.01,'
            '"lower_bound":-0.01,"upper_bound":0.02},"point_estimate":0.005}}\n'
        )
        estimate_record = artifact_record(estimate)
        relative = estimate.relative_to(self.root).as_posix()
        self.campaign["artifact_inventory"][relative] = estimate_record
        validity_path = estimate.parent / "validity.json"
        validity = json.loads(validity_path.read_text())
        validity["artifacts"][estimate.name] = estimate_record
        self.rewrite_validity(validity_path, validity)
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "duplicate JSON key"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_duplicate_build_manifest_key_is_rejected(self) -> None:
        record = self.campaign["build_manifests"]["candidate"]
        path = pathlib.Path(record["path"])
        raw = path.read_text().replace(
            '"profile": "bench",', '"profile": "bench",\n  "profile": "bench",', 1
        )
        path.write_text(raw)
        record["sha256"] = sha256(path)
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "duplicate JSON key"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_duplicate_transaction_commit_key_is_rejected(self) -> None:
        with mock.patch.object(
            classifier, "_publish_stage", side_effect=OSError("stop before publish")
        ):
            with self.assertRaises(OSError):
                classifier.classify_campaign(self.campaign_path, self.root)
        commit = self.root / ".classification-transaction/commit.json"
        raw = commit.read_text().replace(
            '"version": 1', '"version": 1,\n  "version": 1', 1
        )
        commit.write_text(raw)

        with self.assertRaisesRegex(protocol.ProtocolError, "duplicate JSON key"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_running_campaign_is_rejected(self) -> None:
        self.campaign["validity_state"] = "RUNNING"
        self.campaign["statistical_result"] = None
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "COMPLETE"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_validity_inconclusive_campaign_is_rejected(self) -> None:
        self.campaign["validity_state"] = "INCONCLUSIVE"
        self.campaign["statistical_result"] = None
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "COMPLETE"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_null_terminal_statistical_result_is_rejected(self) -> None:
        self.campaign["statistical_result"] = None
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "statistical result"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_huge_normalized_load_limit_is_a_typed_rejection(self) -> None:
        self.campaign["normalized_load_limit"] = 10**1000
        self.write_campaign()

        with self.assertRaises(protocol.ProtocolError):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_partial_case_inventory_is_rejected(self) -> None:
        self.campaign["cases"].pop("lazy_neg_1")
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "case inventory"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_unhashed_estimate_is_rejected(self) -> None:
        estimate = "lazy_neg_1/pair1/change-estimates.json"
        self.campaign["artifact_inventory"][estimate] = {}
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "SHA-256"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_mismatched_classifier_digest_is_rejected(self) -> None:
        self.campaign["classifier_sha256"] = "0" * 64
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "classifier digest"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_mismatched_protocol_digest_is_rejected(self) -> None:
        self.campaign["protocol_sha256"] = "0" * 64
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "protocol digest"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_mismatched_build_manifest_digest_is_rejected(self) -> None:
        self.campaign["build_manifests"]["candidate"]["sha256"] = "0" * 64
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "build manifest digest"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_build_manifest_requires_every_strict_provenance_field(self) -> None:
        path = pathlib.Path(self.campaign["build_manifests"]["candidate"]["path"])
        pristine = json.loads(path.read_text())
        fields = (
            "lock_sha256",
            "environment",
            "toolchain",
            "source_delta",
            "requested_features",
            "profile",
            "validity_state",
            "executable",
        )
        for field in fields:
            with self.subTest(field=field):
                payload = copy.deepcopy(pristine)
                payload.pop(field)
                self.rewrite_build("candidate", payload)
                self.write_campaign()
                try:
                    with self.assertRaisesRegex(protocol.ProtocolError, "build manifest"):
                        classifier.classify_campaign(self.campaign_path, self.root)
                finally:
                    self.remove_unregistered_outputs()

    def test_build_executable_is_rehashed_from_retained_manifest(self) -> None:
        manifest_path = pathlib.Path(
            self.campaign["build_manifests"]["candidate"]["path"]
        )
        manifest = json.loads(manifest_path.read_text())
        pathlib.Path(manifest["executable"]).write_bytes(b"replacement executable")

        with self.assertRaisesRegex(protocol.ProtocolError, "executable digest"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_build_pair_invariant_difference_is_rejected(self) -> None:
        record = self.campaign["build_manifests"]["candidate"]
        payload = json.loads(pathlib.Path(record["path"]).read_text())
        payload["benchmark_sha256"] = "9" * 64
        self.rewrite_build("candidate", payload)
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "invariant field"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_declared_case_result_must_match_independent_recalculation(self) -> None:
        self.campaign["cases"]["lazy_neg_1"]["statistical_result"] = "FAIL"
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "declared case result"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_declared_campaign_result_must_match_independent_recalculation(self) -> None:
        self.campaign["statistical_result"] = "FAIL"
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "declared campaign result"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_monitor_samples_require_start_and_end(self) -> None:
        validity_path = self.root / "lazy_neg_1/pair1/validity.json"
        validity = json.loads(validity_path.read_text())
        validity["runs"][0]["monitor_samples"].pop()
        monitor_path = validity_path.parent / "monitor-samples.json"
        monitor = json.loads(monitor_path.read_text())
        monitor["runs"][protocol.RUN_ROLES[0]].pop()
        write_json(monitor_path, monitor)
        validity["artifacts"]["monitor-samples.json"] = artifact_record(
            monitor_path
        )
        monitor_relative = monitor_path.relative_to(self.root).as_posix()
        self.campaign["artifact_inventory"][monitor_relative] = artifact_record(
            monitor_path
        )
        self.rewrite_validity(validity_path, validity)
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "monitor samples"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_two_samples_do_not_cover_a_seven_second_process(self) -> None:
        validity_path = self.root / "lazy_neg_1/pair1/validity.json"
        validity = json.loads(validity_path.read_text())
        samples = validity["runs"][0]["monitor_samples"]
        validity["runs"][0]["monitor_samples"] = [samples[0], samples[-1]]
        validity["runs"][0]["monitor_samples"][1]["sequence"] = 1
        monitor_path = validity_path.parent / "monitor-samples.json"
        monitor = json.loads(monitor_path.read_text())
        monitor["runs"][protocol.RUN_ROLES[0]] = copy.deepcopy(
            validity["runs"][0]["monitor_samples"]
        )
        write_json(monitor_path, monitor)
        monitor_record = artifact_record(monitor_path)
        validity["artifacts"]["monitor-samples.json"] = monitor_record
        self.campaign["artifact_inventory"][
            monitor_path.relative_to(self.root).as_posix()
        ] = monitor_record
        self.rewrite_validity(validity_path, validity)
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "cadence"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_malformed_monitor_sample_is_a_typed_rejection(self) -> None:
        validity_path = self.root / "lazy_neg_1/pair1/validity.json"
        validity = json.loads(validity_path.read_text())
        validity["runs"][0]["monitor_samples"][0] = None
        monitor_path = validity_path.parent / "monitor-samples.json"
        monitor = json.loads(monitor_path.read_text())
        monitor["runs"][protocol.RUN_ROLES[0]][0] = None
        write_json(monitor_path, monitor)
        monitor_record = artifact_record(monitor_path)
        validity["artifacts"]["monitor-samples.json"] = monitor_record
        monitor_relative = monitor_path.relative_to(self.root).as_posix()
        self.campaign["artifact_inventory"][monitor_relative] = monitor_record
        self.rewrite_validity(validity_path, validity)
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "monitor sample"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_symlink_campaign_manifest_is_rejected(self) -> None:
        linked_root = self.base / "linked"
        linked_root.mkdir()
        linked_campaign = linked_root / "campaign.json"
        linked_campaign.symlink_to(self.campaign_path)

        with self.assertRaisesRegex(protocol.ProtocolError, "regular file|canonical"):
            classifier.classify_campaign(linked_campaign, self.root)

    def test_unexpected_normative_file_is_rejected(self) -> None:
        (self.root / "unexpected.log").write_text("not inventoried\n")
        with self.assertRaisesRegex(protocol.ProtocolError, "artifact inventory"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_unexpected_empty_directories_are_rejected(self) -> None:
        unexpected = (
            self.root / "unknown-empty",
            self.root / "lazy_neg_1/_rejected",
            self.root / "lazy_neg_1/pair1/_rejected",
        )
        for directory in unexpected:
            with self.subTest(directory=directory.relative_to(self.root)):
                directory.mkdir(parents=True)
                try:
                    with self.assertRaisesRegex(
                        protocol.ProtocolError, "directory inventory"
                    ):
                        classifier.classify_campaign(self.campaign_path, self.root)
                finally:
                    self.remove_unregistered_outputs()
                    directory.rmdir()

    def test_partial_registered_output_digest_set_is_rejected(self) -> None:
        self.campaign["classification_artifacts"] = {
            "classification.json": {
                "path": str((self.root / "classification.json").resolve()),
                "sha256": "0" * 64,
            }
        }
        self.write_campaign()
        with self.assertRaisesRegex(protocol.ProtocolError, "classification artifacts"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_registered_outputs_are_revalidated_without_rewriting(self) -> None:
        first = classifier.classify_campaign(self.campaign_path, self.root)
        self.campaign["classification_artifacts"] = copy.deepcopy(
            first["output_artifacts"]
        )
        for record in first["output_artifacts"].values():
            path = self.root / record["path"]
            relative = path.relative_to(self.root).as_posix()
            self.campaign["artifact_inventory"][relative] = {
                "sha256": record["sha256"]
            }
        self.write_campaign()
        mtimes = {
            name: (self.root / record["path"]).stat().st_mtime_ns
            for name, record in first["output_artifacts"].items()
        }

        second = classifier.classify_campaign(self.campaign_path, self.root)

        self.assertEqual(second, first)
        self.assertEqual(
            mtimes,
            {
                name: (self.root / record["path"]).stat().st_mtime_ns
                for name, record in second["output_artifacts"].items()
            },
        )

    def test_registered_output_digest_mismatch_is_rejected(self) -> None:
        result = classifier.classify_campaign(self.campaign_path, self.root)
        self.campaign["classification_artifacts"] = copy.deepcopy(
            result["output_artifacts"]
        )
        for record in result["output_artifacts"].values():
            path = self.root / record["path"]
            relative = path.relative_to(self.root).as_posix()
            self.campaign["artifact_inventory"][relative] = {
                "sha256": record["sha256"]
            }
        self.campaign["classification_artifacts"]["summary.md"]["sha256"] = (
            "0" * 64
        )
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "output digest"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_registered_outputs_survive_evidence_root_relocation(self) -> None:
        result = classifier.classify_campaign(self.campaign_path, self.root)
        self.campaign["classification_artifacts"] = copy.deepcopy(
            result["output_artifacts"]
        )
        for record in result["output_artifacts"].values():
            path = self.root / record["path"]
            self.campaign["artifact_inventory"][record["path"]] = artifact_record(path)
        self.write_campaign()
        relocated = self.base / "relocated"
        shutil.copytree(self.root, relocated)

        second = classifier.classify_campaign(relocated / "campaign.json", relocated)

        self.assertEqual(second, result)

    def test_output_transaction_recovers_after_second_final_install_failure(self) -> None:
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-install failure")
            return original(*args, **kwargs)

        original = classifier._publish_stage
        with mock.patch.object(
            classifier, "_publish_stage", side_effect=fail_second
        ):
            with self.assertRaises(OSError):
                classifier.classify_campaign(self.campaign_path, self.root)

        transaction = self.root / ".classification-transaction"
        self.assertTrue((transaction / "commit.json").is_file())
        result = classifier.classify_campaign(self.campaign_path, self.root)
        self.assertEqual(result["statistical_result"], "PASS")
        self.assertFalse(transaction.exists())

    def test_output_transaction_handles_short_staging_writes(self) -> None:
        original_write = os.write
        short_writes = 0

        def short_write(descriptor, content):
            nonlocal short_writes
            short_writes += 1
            limited = content[: max(1, len(content) // 3)]
            return original_write(descriptor, limited)

        with mock.patch.object(classifier.os, "write", side_effect=short_write):
            result = classifier.classify_campaign(self.campaign_path, self.root)

        self.assertEqual(result["statistical_result"], "PASS")
        self.assertGreater(short_writes, 3)
        self.assertTrue((self.root / "classification.json").is_file())
        self.assertTrue((self.root / "summary.md").is_file())

    def test_stage_path_swap_is_rejected_without_deleting_unowned_final(
        self,
    ) -> None:
        original_link = os.link
        swapped = False

        def swap_before_link(source, destination, *args, **kwargs):
            nonlocal swapped
            if source == "staged-summary.md" and destination == "summary.md":
                swapped = True
                transaction = self.root / ".classification-transaction"
                (transaction / source).rename(transaction / "detached-stage")
                (transaction / source).write_bytes(b"attacker output\n")
            return original_link(source, destination, *args, **kwargs)

        with mock.patch.object(classifier.os, "link", side_effect=swap_before_link):
            with self.assertRaisesRegex(protocol.ProtocolError, "staged|publication"):
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertTrue(swapped)
        self.assertEqual((self.root / "summary.md").read_bytes(), b"attacker output\n")
        self.assertFalse((self.root / "classification.json").exists())

    def test_final_replacement_during_publish_is_not_deleted_as_owned(self) -> None:
        original_stat = os.stat
        replaced = False
        replacement = b"attacker replacement\n"

        def replace_before_post_link_stat(path, *args, **kwargs):
            nonlocal replaced
            if path == "summary.md" and kwargs.get("dir_fd") is not None and not replaced:
                replaced = True
                final = self.root / "summary.md"
                final.unlink()
                final.write_bytes(replacement)
            return original_stat(path, *args, **kwargs)

        with mock.patch.object(
            classifier.os, "stat", side_effect=replace_before_post_link_stat
        ):
            with self.assertRaisesRegex(protocol.ProtocolError, "changed|publication"):
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertTrue(replaced)
        self.assertEqual((self.root / "summary.md").read_bytes(), replacement)

    def test_publication_cancellation_survives_cleanup_failure(self) -> None:
        original_verify = classifier._verify_regular_link
        calls = 0
        cancellation = KeyboardInterrupt("cancel publication")

        def cancel_second_verification(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise cancellation
            return original_verify(*args, **kwargs)

        with mock.patch.object(
            classifier, "_verify_regular_link", side_effect=cancel_second_verification
        ), mock.patch.object(
            classifier,
            "_remove_created_final",
            side_effect=OSError("cleanup failed"),
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertIs(caught.exception, cancellation)
        self.assertTrue((self.root / "summary.md").exists())

    def test_allowed_cpu_count_is_bounded_before_inventory_expansion(self) -> None:
        self.campaign["allowed_cpu_count"] = 20_000_001
        self.campaign["allowed_cpus"] = "0-20000000"
        self.write_campaign()

        with self.assertRaisesRegex(protocol.ProtocolError, "CPU identity"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_pending_stage_unlink_failure_is_recoverable(self) -> None:
        original_unlink = os.unlink
        failed = False

        def fail_pending_unlink(path, *args, **kwargs):
            nonlocal failed
            if path == "pending-staged-classification.json" and not failed:
                failed = True
                raise OSError("injected pending unlink failure")
            return original_unlink(path, *args, **kwargs)

        with mock.patch.object(classifier.os, "unlink", side_effect=fail_pending_unlink):
            with self.assertRaisesRegex(OSError, "pending unlink"):
                classifier.classify_campaign(self.campaign_path, self.root)

        pending = (
            self.root
            / ".classification-transaction"
            / "pending-staged-classification.json"
        )
        self.assertTrue(pending.is_file())
        result = classifier.classify_campaign(self.campaign_path, self.root)
        self.assertEqual(result["statistical_result"], "PASS")
        self.assertFalse((self.root / ".classification-transaction").exists())

    def test_mismatched_owned_pending_stage_is_rejected(self) -> None:
        transaction = self.root / ".classification-transaction"
        transaction.mkdir()
        (transaction / "pending-staged-classification.json").write_bytes(b"wrong\n")

        with self.assertRaisesRegex(protocol.ProtocolError, "pending staged hash differs"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_alternate_pending_stage_name_is_rejected(self) -> None:
        transaction = self.root / ".classification-transaction"
        transaction.mkdir()
        (transaction / "pending-staged-unknown.json").write_bytes(b"payload\n")

        with self.assertRaisesRegex(protocol.ProtocolError, "inventory|unknown partials"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_staging_fsync_failure_never_exposes_a_partial_final_and_retries(self) -> None:
        original_fsync = classifier._fsync_fd
        calls = 0

        def fail_first_stage(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected staged-file fsync failure")
            return original_fsync(descriptor)

        with mock.patch.object(classifier, "_fsync_fd", side_effect=fail_first_stage):
            with self.assertRaisesRegex(OSError, "staged-file fsync"):
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertFalse((self.root / "classification.json").exists())
        self.assertFalse((self.root / "summary.md").exists())
        classifier.classify_campaign(self.campaign_path, self.root)
        self.assertTrue((self.root / "classification.json").is_file())
        self.assertTrue((self.root / "summary.md").is_file())

    def test_crash_after_first_final_link_is_recoverable(self) -> None:
        original = classifier._publish_stage
        calls = 0

        def crash_after_link(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            if calls == 1:
                raise OSError("injected crash after first link")
            return result

        with mock.patch.object(classifier, "_publish_stage", side_effect=crash_after_link):
            with self.assertRaisesRegex(OSError, "after first link"):
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertTrue((self.root / "summary.md").is_file())
        self.assertFalse((self.root / "classification.json").exists())
        classifier.classify_campaign(self.campaign_path, self.root)
        self.assertTrue((self.root / "classification.json").is_file())

    def test_crash_after_transaction_rmdir_is_idempotent(self) -> None:
        original = classifier._fsync_fd
        injected = False

        def fail_after_rmdir(descriptor):
            nonlocal injected
            result = original(descriptor)
            transaction = self.root / ".classification-transaction"
            if (
                not injected
                and not transaction.exists()
                and (self.root / "classification.json").exists()
            ):
                injected = True
                raise OSError("injected post-rmdir fsync failure")
            return result

        with mock.patch.object(classifier, "_fsync_fd", side_effect=fail_after_rmdir):
            with self.assertRaisesRegex(OSError, "post-rmdir"):
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertFalse((self.root / ".classification-transaction").exists())
        classifier.classify_campaign(self.campaign_path, self.root)

    def test_output_directory_swap_never_redirects_publication(self) -> None:
        original = classifier._publish_stage
        detached = self.base / "detached-output"
        replacement = self.root
        swapped = False
        before_fds = len(os.listdir("/proc/self/fd"))

        def swap_output(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                swapped = True
                replacement.rename(detached)
                replacement.mkdir()
            return original(*args, **kwargs)

        with mock.patch.object(classifier, "_publish_stage", side_effect=swap_output):
            with self.assertRaisesRegex(protocol.ProtocolError, "directory changed"):
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertEqual(list(replacement.iterdir()), [])
        self.assertEqual(len(os.listdir("/proc/self/fd")), before_fds)

    def test_transaction_directory_swap_is_detected_without_cleanup_redirect(self) -> None:
        original = classifier._publish_stage
        transaction = self.root / ".classification-transaction"
        detached = self.root / ".detached-transaction"
        swapped = False

        def swap_transaction(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                swapped = True
                transaction.rename(detached)
                transaction.mkdir()
                (transaction / "replacement-sentinel").write_text("keep\n")
            return original(*args, **kwargs)

        with mock.patch.object(classifier, "_publish_stage", side_effect=swap_transaction):
            with self.assertRaisesRegex(protocol.ProtocolError, "transaction directory changed"):
                classifier.classify_campaign(self.campaign_path, self.root)

        self.assertEqual(
            (transaction / "replacement-sentinel").read_text(), "keep\n"
        )

    def test_complete_unregistered_output_pair_is_immediately_idempotent(self) -> None:
        first = classifier.classify_campaign(self.campaign_path, self.root)

        second = classifier.classify_campaign(self.campaign_path, self.root)

        self.assertEqual(second, first)
        self.assertFalse((self.root / ".classification-transaction").exists())

    def test_unowned_single_final_is_not_treated_as_committed(self) -> None:
        classifier.classify_campaign(self.campaign_path, self.root)
        (self.root / "classification.json").unlink()

        with self.assertRaisesRegex(protocol.ProtocolError, "without transaction"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_unknown_output_transaction_partial_is_rejected(self) -> None:
        transaction = self.root / ".classification-transaction"
        transaction.mkdir()
        (transaction / "unknown.tmp").write_text("unowned\n")

        with self.assertRaisesRegex(protocol.ProtocolError, "transaction"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_output_transaction_recovers_from_interrupted_cleanup(self) -> None:
        def interrupted_cleanup(*_args):
            transaction = self.root / ".classification-transaction"
            (transaction / "staged-classification.json").unlink()
            raise OSError("injected cleanup interruption")

        with mock.patch.object(
            classifier,
            "_cleanup_output_transaction",
            side_effect=interrupted_cleanup,
            create=True,
        ):
            with self.assertRaises(OSError):
                classifier.classify_campaign(self.campaign_path, self.root)

        transaction = self.root / ".classification-transaction"
        self.assertTrue(transaction.is_dir())
        self.assertTrue((self.root / "classification.json").is_file())
        self.assertTrue((self.root / "summary.md").is_file())
        classifier.classify_campaign(self.campaign_path, self.root)
        self.assertFalse(transaction.exists())

    def test_owned_existing_final_with_wrong_hash_is_rejected(self) -> None:
        transaction = self.root / ".classification-transaction"
        transaction.mkdir()
        (self.root / "classification.json").write_text("wrong\n")

        with self.assertRaisesRegex(protocol.ProtocolError, "final output hash differs"):
            classifier.classify_campaign(self.campaign_path, self.root)

    def test_raw_criterion_root_has_no_legacy_classifier_bypass(self) -> None:
        legacy = self.base / "legacy"
        pair_root = legacy / "one-case"
        for pair in (1, 2, 3):
            write_estimate(pair_root / f"pair{pair}" / "change-estimates.json")

        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(legacy), "--legacy-artifacts"],
            cwd=SCRIPT.parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("legacy", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
