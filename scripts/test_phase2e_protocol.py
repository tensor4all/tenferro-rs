#!/usr/bin/env python3
"""Tests for the Phase 2E protocol-v2 evidence primitives."""

import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import phase2e_protocol as protocol


EXPECTED_CASES = {
    "lazy_neg_1": "eager_dispatch_baseline/lazy/neg_f64/1",
    "lazy_neg_8": "eager_dispatch_baseline/lazy/neg_f64/8",
    "lazy_neg_64": "eager_dispatch_baseline/lazy/neg_f64/64",
    "lazy_add_1": "eager_dispatch_baseline/lazy/add_f64/1",
    "lazy_add_8": "eager_dispatch_baseline/lazy/add_f64/8",
    "lazy_add_64": "eager_dispatch_baseline/lazy/add_f64/64",
    "lazy_reduce_1": "eager_dispatch_baseline/lazy/reduce_sum_f64/1",
    "lazy_reduce_8": "eager_dispatch_baseline/lazy/reduce_sum_f64/8",
    "lazy_reduce_64": "eager_dispatch_baseline/lazy/reduce_sum_f64/64",
    "lazy_slice_1": "eager_dispatch_baseline/lazy/slice_f64/1",
    "lazy_slice_8": "eager_dispatch_baseline/lazy/slice_f64/8",
    "lazy_slice_64": "eager_dispatch_baseline/lazy/slice_f64/64",
    "lazy_dot_1": "eager_dispatch_baseline/lazy/dot_general_f64/1",
    "lazy_dot_2": "eager_dispatch_baseline/lazy/dot_general_f64/2",
    "materialized_neg_1": "eager_dispatch_baseline/materialized/neg_f64/1",
    "materialized_neg_8": "eager_dispatch_baseline/materialized/neg_f64/8",
    "materialized_neg_64": "eager_dispatch_baseline/materialized/neg_f64/64",
    "materialized_add_1": "eager_dispatch_baseline/materialized/add_f64/1",
    "materialized_add_8": "eager_dispatch_baseline/materialized/add_f64/8",
    "materialized_add_64": "eager_dispatch_baseline/materialized/add_f64/64",
    "materialized_reduce_1": (
        "eager_dispatch_baseline/materialized/reduce_sum_f64/1"
    ),
    "materialized_reduce_8": (
        "eager_dispatch_baseline/materialized/reduce_sum_f64/8"
    ),
    "materialized_reduce_64": (
        "eager_dispatch_baseline/materialized/reduce_sum_f64/64"
    ),
    "materialized_slice_1": "eager_dispatch_baseline/materialized/slice_f64/1",
    "materialized_slice_8": "eager_dispatch_baseline/materialized/slice_f64/8",
    "materialized_slice_64": "eager_dispatch_baseline/materialized/slice_f64/64",
    "materialized_dot_1": (
        "eager_dispatch_baseline/materialized/dot_general_f64/1"
    ),
    "materialized_dot_2": (
        "eager_dispatch_baseline/materialized/dot_general_f64/2"
    ),
}


class ProtocolConstantsTests(unittest.TestCase):
    def test_inventory_and_orders_are_frozen(self) -> None:
        self.assertEqual(protocol.PROTOCOL_VERSION, 2)
        self.assertEqual(dict(protocol.CANONICAL_CASES), EXPECTED_CASES)
        self.assertEqual(len(protocol.CANONICAL_CASES), 28)
        self.assertEqual(protocol.PAIR_ORDERS, ("A/B", "B/A", "A/B"))
        self.assertEqual(
            protocol.RUN_ROLES,
            ("sentinel_before", "first_target", "second_target", "sentinel_after"),
        )

    def test_inventory_and_thread_environment_are_immutable(self) -> None:
        with self.assertRaises(TypeError):
            protocol.CANONICAL_CASES["extra"] = "unexpected"
        with self.assertRaises(TypeError):
            protocol.THREAD_ENV["OMP_NUM_THREADS"] = "2"


class EmptyRootTests(unittest.TestCase):
    def test_prepare_empty_root_accepts_nonexistent_and_empty_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            missing = parent / "nested" / "evidence"
            self.assertEqual(protocol.prepare_empty_root(missing), missing)
            self.assertTrue(missing.is_dir())
            self.assertEqual(protocol.prepare_empty_root(missing), missing)

    def test_prepare_empty_root_rejects_every_prepopulated_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            populated = parent / "populated"
            populated.mkdir()
            (populated / ".partial").write_text("evidence", encoding="utf-8")
            with self.assertRaises(protocol.ProtocolError):
                protocol.prepare_empty_root(populated)

            regular_file = parent / "regular-file"
            regular_file.write_text("not a root", encoding="utf-8")
            with self.assertRaises(protocol.ProtocolError):
                protocol.prepare_empty_root(regular_file)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_runtime_environment_is_an_exact_allowlist(self) -> None:
        expected = {
            "PATH": "/usr/bin",
            "HOME": "/tmp/empty-home",
            "LC_ALL": "C",
            "TZ": "UTC",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
        }
        with mock.patch.dict(
            os.environ,
            {
                "LD_PRELOAD": "/tmp/inject.so",
                "GLIBC_TUNABLES": "glibc.cpu.hwcaps=-AVX2",
                "RAYON_RS_NUM_CPUS": "99",
                "UNDECLARED_AMBIENT": "must-not-leak",
            },
            clear=True,
        ):
            actual = protocol.runtime_environment(
                path="/usr/bin", home="/tmp/empty-home"
            )
        self.assertEqual(actual, expected)

    def test_runtime_environment_adds_only_explicit_criterion_home(self) -> None:
        without_criterion = protocol.runtime_environment(path="/bin", home="/tmp/h")
        self.assertNotIn("CRITERION_HOME", without_criterion)

        with_criterion = protocol.runtime_environment(
            path="/bin", home="/tmp/h", criterion_home="/tmp/criterion"
        )
        self.assertEqual(with_criterion["CRITERION_HOME"], "/tmp/criterion")
        self.assertEqual(set(with_criterion), set(without_criterion) | {"CRITERION_HOME"})

    def test_runtime_environment_adds_paired_affinity_parameters(self) -> None:
        environment = protocol.runtime_environment(
            path="/bin",
            home="/tmp/h",
            criterion_home="/tmp/criterion",
            affinity_row="managed-exact/budget-2/D-N",
            affinity_file="/tmp/criterion/affinity.json",
        )
        self.assertEqual(
            environment["TENFERRO_PHASE2E_AFFINITY_ROW"],
            "managed-exact/budget-2/D-N",
        )
        self.assertEqual(
            environment["TENFERRO_PHASE2E_AFFINITY_FILE"],
            "/tmp/criterion/affinity.json",
        )

    def test_runtime_environment_rejects_unpaired_or_invalid_affinity_parameters(self) -> None:
        invalid = (
            {"affinity_row": "managed-exact/budget-2/D-N"},
            {"affinity_file": "/tmp/criterion/affinity.json"},
            {
                "affinity_row": "not-a-canonical-row",
                "affinity_file": "/tmp/criterion/affinity.json",
            },
            {
                "affinity_row": "managed-exact/budget-2/U-O",
                "affinity_file": "/tmp/criterion/affinity.json",
            },
            {
                "affinity_row": "managed-exact/budget-2/D-N",
                "affinity_file": "relative/affinity.json",
            },
            {
                "affinity_row": "managed-exact/budget-2/D-N",
                "affinity_file": "/tmp/outside/affinity.json",
            },
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(
                protocol.ProtocolError
            ):
                protocol.runtime_environment(
                    path="/bin",
                    home="/tmp/h",
                    criterion_home="/tmp/criterion",
                    **parameters,
                )


class AtomicJsonAndHashTests(unittest.TestCase):
    def test_atomic_json_at_remains_bound_to_held_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            root = base / "root"
            root.mkdir()
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            detached = base / "detached"
            outside = base / "outside"
            try:
                root.rename(detached)
                outside.mkdir()
                root.symlink_to(outside, target_is_directory=True)

                protocol.atomic_write_json_at(
                    descriptor, "campaign.json", {"state": "RUNNING"}
                )

                self.assertEqual(
                    json.loads((detached / "campaign.json").read_text()),
                    {"state": "RUNNING"},
                )
                self.assertEqual(list(outside.iterdir()), [])
                with self.assertRaises(protocol.ProtocolError):
                    protocol.atomic_write_json_at(
                        descriptor, "escape/campaign.json", {}
                    )
            finally:
                os.close(descriptor)

    def test_atomic_json_at_handles_short_writes_and_fsyncs_file_and_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            real_write = os.write
            real_fsync = os.fsync
            fsynced = []

            def short_write(fd, payload):
                return real_write(fd, bytes(payload[:3]))

            def tracking_fsync(fd):
                fsynced.append(fd)
                return real_fsync(fd)

            try:
                with mock.patch.object(
                    protocol.os, "write", side_effect=short_write
                ), mock.patch.object(
                    protocol.os, "fsync", side_effect=tracking_fsync
                ):
                    protocol.atomic_write_json_at(
                        descriptor, "campaign.json", {"answer": 42}
                    )
                self.assertEqual(
                    json.loads((root / "campaign.json").read_text()),
                    {"answer": 42},
                )
                self.assertGreaterEqual(len(fsynced), 2)
                self.assertEqual(fsynced[-1], descriptor)
            finally:
                os.close(descriptor)

    def test_atomic_json_at_preserves_base_exception_and_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "campaign.json"
            target.write_text('{"old":true}\n')
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            before_fds = len(os.listdir("/proc/self/fd"))
            interruption = KeyboardInterrupt("write cancelled")
            try:
                with mock.patch.object(
                    protocol.os, "write", side_effect=interruption
                ):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        protocol.atomic_write_json_at(
                            descriptor, "campaign.json", {"answer": 42}
                        )
                self.assertIs(caught.exception, interruption)
                self.assertEqual(target.read_text(), '{"old":true}\n')
                self.assertEqual(
                    len(list(root.glob(".campaign.json.write-*.tmp"))), 1
                )
                self.assertEqual(len(os.listdir("/proc/self/fd")), before_fds)
            finally:
                os.close(descriptor)

    def test_atomic_json_at_replace_failure_is_precommit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "campaign.json"
            target.write_text('{"old":true}\n')
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with mock.patch.object(
                    protocol.os,
                    "replace",
                    side_effect=OSError("replace failed"),
                ):
                    with self.assertRaises(protocol.AtomicWriteError) as caught:
                        protocol.atomic_write_json_at(
                            descriptor, "campaign.json", {"answer": 42}
                        )
                self.assertFalse(caught.exception.committed)
                self.assertEqual(target.read_text(), '{"old":true}\n')
                self.assertEqual(
                    len(list(root.glob(".campaign.json.write-*.tmp"))), 1
                )
            finally:
                os.close(descriptor)

    def test_atomic_json_is_deterministic_sorted_and_collision_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "manifest.json"
            legacy = root / "manifest.json.tmp"
            colliding = root / ".manifest.json.write-interrupted.tmp"
            legacy_bytes = b"legacy partial\x00"
            colliding_bytes = b"unique interrupted partial\x00"
            legacy.write_bytes(legacy_bytes)
            colliding.write_bytes(colliding_bytes)

            protocol.atomic_write_json(target, {"z": 1, "a": 2})
            first = target.read_bytes()
            protocol.atomic_write_json(target, {"a": 2, "z": 1})

            self.assertEqual(first, b'{\n  "a": 2,\n  "z": 1\n}\n')
            self.assertEqual(target.read_bytes(), first)
            self.assertEqual(legacy.read_bytes(), legacy_bytes)
            self.assertEqual(colliding.read_bytes(), colliding_bytes)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                [
                    ".manifest.json.write-interrupted.tmp",
                    "manifest.json",
                    "manifest.json.tmp",
                ],
            )

    def test_atomic_json_preserves_new_partial_temporary_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "manifest.json"
            old_bytes = b'{"old":true}\n'
            target.write_bytes(old_bytes)
            descriptors, tracked = tracking_open()
            with (
                mock.patch.object(protocol.os, "open", tracked),
                mock.patch.object(
                    protocol.os,
                    "replace",
                    side_effect=OSError("injected replace failure"),
                ),
            ):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.atomic_write_json(target, {"answer": 42})

            assert_atomic_error(self, caught.exception, "AtomicWriteError", False)
            assert_descriptors_closed(self, descriptors)
            self.assertEqual(target.read_bytes(), old_bytes)
            partials = list(root.glob(".manifest.json.write-*.tmp"))
            self.assertEqual(len(partials), 1)
            self.assertEqual(partials[0].read_bytes(), b'{\n  "answer": 42\n}\n')
            self.assertNotEqual(partials[0].name, "manifest.json.tmp")

    def test_atomic_json_directory_open_failure_is_precommit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "manifest.json"
            old_bytes = b'{"old":true}\n'
            target.write_bytes(old_bytes)
            descriptors, tracked = tracking_open()

            def fail_parent_open(path, flags, *args, **kwargs):
                if pathlib.Path(path) == root and flags & os.O_DIRECTORY:
                    raise OSError("injected directory open failure")
                return tracked(path, flags, *args, **kwargs)

            with mock.patch.object(protocol.os, "open", fail_parent_open):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.atomic_write_json(target, {"answer": 42})

            assert_atomic_error(self, caught.exception, "AtomicWriteError", False)
            assert_descriptors_closed(self, descriptors)
            self.assertEqual(target.read_bytes(), old_bytes)
            self.assertEqual(list(root.glob(".manifest.json.write-*.tmp")), [])

    def test_atomic_json_write_failure_retains_partial_and_closes_fds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "manifest.json"
            old_bytes = b'{"old":true}\n'
            target.write_bytes(old_bytes)
            descriptors, tracked = tracking_open()
            real_fdopen = os.fdopen

            def failing_fdopen(descriptor, *args, **kwargs):
                descriptors.append(descriptor)
                return FailingWriteStream(
                    real_fdopen(descriptor, *args, **kwargs)
                )

            with (
                mock.patch.object(protocol.os, "open", tracked),
                mock.patch.object(protocol.os, "fdopen", failing_fdopen),
            ):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.atomic_write_json(target, {"answer": 42})

            assert_atomic_error(self, caught.exception, "AtomicWriteError", False)
            assert_descriptors_closed(self, descriptors)
            self.assertEqual(target.read_bytes(), old_bytes)
            self.assertEqual(len(list(root.glob(".manifest.json.write-*.tmp"))), 1)

    def test_atomic_json_file_fsync_failure_is_precommit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "manifest.json"
            old_bytes = b'{"old":true}\n'
            target.write_bytes(old_bytes)
            descriptors, tracked = tracking_open()

            with (
                mock.patch.object(protocol.os, "open", tracked),
                mock.patch.object(
                    protocol.os,
                    "fsync",
                    side_effect=OSError("injected file fsync failure"),
                ),
            ):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.atomic_write_json(target, {"answer": 42})

            assert_atomic_error(self, caught.exception, "AtomicWriteError", False)
            assert_descriptors_closed(self, descriptors)
            self.assertEqual(target.read_bytes(), old_bytes)
            self.assertEqual(len(list(root.glob(".manifest.json.write-*.tmp"))), 1)

    def test_atomic_json_directory_fsync_failure_reports_committed_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "manifest.json"
            target.write_bytes(b'{"old":true}\n')
            new_bytes = b'{\n  "answer": 42\n}\n'
            descriptors, tracked = tracking_open()
            real_fsync = os.fsync
            calls = 0

            def fail_second_fsync(descriptor):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected directory fsync failure")
                return real_fsync(descriptor)

            with (
                mock.patch.object(protocol.os, "open", tracked),
                mock.patch.object(protocol.os, "fsync", fail_second_fsync),
            ):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.atomic_write_json(target, {"answer": 42})

            assert_atomic_error(
                self, caught.exception, "AtomicWriteDurabilityError", True
            )
            assert_descriptors_closed(self, descriptors)
            self.assertEqual(target.read_bytes(), new_bytes)
            self.assertEqual(list(root.glob(".manifest.json.write-*.tmp")), [])

    def test_sha256_helpers_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "payload.bin"
            payload = b"phase-2e\x00evidence\n"
            path.write_bytes(payload)
            self.assertEqual(
                protocol.sha256_file(path), hashlib.sha256(payload).hexdigest()
            )

        first = {"nested": {"z": 1, "a": 2}, "items": [3, 2, 1]}
        second = {"items": [3, 2, 1], "nested": {"a": 2, "z": 1}}
        self.assertEqual(protocol.sha256_json(first), protocol.sha256_json(second))
        self.assertEqual(
            protocol.sha256_json(first),
            hashlib.sha256(
                (json.dumps(first, sort_keys=True, indent=2) + "\n").encode("utf-8")
            ).hexdigest(),
        )

    def test_sha256_file_rejects_symlinks_and_non_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            regular = root / "regular.bin"
            regular.write_bytes(b"evidence")
            symlink = root / "symlink.bin"
            symlink.symlink_to(regular)

            with self.assertRaises(protocol.ProtocolError):
                protocol.sha256_file(symlink)
            with self.assertRaises(protocol.ProtocolError):
                protocol.sha256_file(pathlib.Path("/dev/null"))

    def test_sha256_file_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fifo = pathlib.Path(temporary) / "evidence.fifo"
            os.mkfifo(fifo)
            script = """
import pathlib
import sys
from scripts import phase2e_protocol

try:
    phase2e_protocol.sha256_file(pathlib.Path(sys.argv[1]))
except phase2e_protocol.ProtocolError:
    print("PROTOCOL_ERROR")
else:
    print("UNEXPECTED_SUCCESS")
    raise SystemExit(2)
"""
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", script, str(fifo)],
                    cwd=pathlib.Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                self.fail("sha256_file blocked while opening a FIFO")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "PROTOCOL_ERROR")

    def test_sha256_file_closes_descriptor_when_validation_or_read_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "payload.bin"
            path.write_bytes(b"phase-2e")

            for phase in ("fstat", "read"):
                with self.subTest(phase=phase):
                    descriptors, tracked = tracking_open()
                    patches = [mock.patch.object(protocol.os, "open", tracked)]
                    if phase == "fstat":
                        patches.append(
                            mock.patch.object(
                                protocol.os,
                                "fstat",
                                side_effect=OSError("injected fstat failure"),
                            )
                        )
                    else:
                        patches.append(
                            mock.patch.object(
                                protocol.os,
                                "read",
                                side_effect=OSError("injected read failure"),
                            )
                        )
                    with patches[0], patches[1]:
                        with self.assertRaises(protocol.ProtocolError):
                            protocol.sha256_file(path)
                    self.assertTrue(descriptors)
                    assert_descriptors_closed(self, descriptors)

    def test_sha256_file_closes_fd_and_preserves_base_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "payload.bin"
            path.write_bytes(b"phase-2e")
            descriptors, tracked = tracking_open()
            real_close = os.close
            interruption = KeyboardInterrupt("injected cancellation")

            def close_then_fail(descriptor):
                real_close(descriptor)
                raise OSError("injected close failure")

            leaked = []
            try:
                with (
                    mock.patch.object(protocol.os, "open", tracked),
                    mock.patch.object(protocol.os, "read", side_effect=interruption),
                    mock.patch.object(protocol.os, "close", close_then_fail),
                ):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        protocol.sha256_file(path)
                self.assertIs(caught.exception, interruption)
                self.assertTrue(descriptors)
                for descriptor in set(descriptors):
                    try:
                        os.fstat(descriptor)
                    except OSError:
                        continue
                    leaked.append(descriptor)
            finally:
                for descriptor in leaked:
                    real_close(descriptor)
            self.assertEqual(leaked, [], "sha256_file leaked an evidence descriptor")


class ManifestValidationTests(unittest.TestCase):
    def test_strict_manifest_fields_accept_exact_schema(self) -> None:
        manifest = {"protocol_version": 2, "candidate_sha": "a" * 40}
        schema = {"protocol_version": int, "candidate_sha": str}
        protocol.validate_manifest_fields(manifest, schema, context="build manifest")

    def test_strict_manifest_fields_reject_missing_extra_and_wrong_fields(self) -> None:
        schema = {"protocol_version": int, "candidate_sha": str}
        invalid = (
            {"protocol_version": 2},
            {
                "protocol_version": 2,
                "candidate_sha": "a" * 40,
                "ambient": "unexpected",
            },
            {"protocol_version": "2", "candidate_sha": "a" * 40},
            {"protocol_version": True, "candidate_sha": "a" * 40},
        )
        for manifest in invalid:
            with self.subTest(manifest=manifest):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.validate_manifest_fields(manifest, schema)


def stage(ledger: dict, name: str) -> dict:
    return next(item for item in ledger["stages"] if item["name"] == name)


def lane(ledger: dict, stage_name: str, lane_name: str) -> dict:
    return next(
        item for item in stage(ledger, stage_name)["lanes"] if item["name"] == lane_name
    )


def validate_ledger(ledger: dict) -> None:
    validator = getattr(protocol, "validate_ledger", None)
    assert validator is not None, "phase2e_protocol must expose validate_ledger"
    return validator(ledger)


def tracking_open() -> tuple[list[int], object]:
    real_open = os.open
    descriptors: list[int] = []

    def tracked(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    return descriptors, tracked


def assert_descriptors_closed(test: unittest.TestCase, descriptors: list[int]) -> None:
    for descriptor in set(descriptors):
        with test.subTest(descriptor=descriptor):
            with test.assertRaises(OSError):
                os.fstat(descriptor)


def assert_atomic_error(
    test: unittest.TestCase,
    error: BaseException,
    expected_type_name: str,
    committed: bool,
) -> None:
    test.assertEqual(type(error).__name__, expected_type_name)
    test.assertIs(getattr(error, "committed", None), committed)


class FailingWriteStream:
    def __init__(self, stream) -> None:
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False

    def write(self, _payload):
        raise OSError("injected write failure")

    def flush(self) -> None:
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()

    def close(self) -> None:
        self.stream.close()


def retry_then_normalized_pass(stage_name: str) -> dict:
    ledger = protocol.new_ledger("a" * 40)
    assert "next_transition_ordinal" in ledger, (
        "ledger must persist the next global transition ordinal"
    )
    ownership = (
        {"artifact_root": "/evidence/direct-attempt-1"}
        if stage_name == "allocation"
        else {}
    )
    ledger = protocol.open_attempt(
        ledger, stage_name, "direct-current-main", 1, **ownership
    )
    ledger = protocol.close_attempt(
        ledger,
        stage_name,
        "direct-current-main",
        1,
        None,
        validity_state="INCONCLUSIVE",
    )
    ownership = (
        {"artifact_root": "/evidence/direct-attempt-2"}
        if stage_name == "allocation"
        else {}
    )
    ledger = protocol.open_attempt(
        ledger, stage_name, "direct-current-main", 2, **ownership
    )
    ledger = protocol.close_attempt(
        ledger, stage_name, "direct-current-main", 2, "PASS"
    )
    ownership = (
        {"artifact_root": "/evidence/normalized-attempt-1"}
        if stage_name == "allocation"
        else {}
    )
    ledger = protocol.open_attempt(
        ledger, stage_name, "common-lock-normalized", 1, **ownership
    )
    return protocol.close_attempt(
        ledger, stage_name, "common-lock-normalized", 1, "PASS"
    )


class LedgerTests(unittest.TestCase):
    def test_new_ledger_has_fixed_ordered_stages_and_lanes(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        self.assertEqual(ledger["protocol_version"], 2)
        self.assertEqual(ledger["candidate_sha"], "a" * 40)
        self.assertEqual(
            [item["name"] for item in ledger["stages"]], ["allocation", "timing"]
        )
        for stage_record in ledger["stages"]:
            self.assertEqual(
                [item["name"] for item in stage_record["lanes"]],
                ["direct-current-main", "common-lock-normalized"],
            )
        self.assertEqual(ledger["attempts"], [])
        self.assertIn("next_transition_ordinal", ledger)
        self.assertEqual(ledger["next_transition_ordinal"], 1)

    def test_attempt_ids_are_positive_sequential_and_registered_once(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        for invalid_id in (0, -1, 2):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.open_attempt(
                        ledger, "timing", "direct-current-main", invalid_id
                    )

        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(ledger, "allocation", "direct-current-main", 2)

        ledger = protocol.close_attempt(
            ledger,
            "timing",
            "direct-current-main",
            1,
            None,
            validity_state="INCONCLUSIVE",
        )
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 2)
        self.assertEqual([item["attempt_id"] for item in ledger["attempts"]], [1, 2])
        self.assertEqual(lane(ledger, "timing", "direct-current-main")["attempt_ids"], [1, 2])
        self.assertIn("next_transition_ordinal", ledger)
        self.assertEqual(ledger["next_transition_ordinal"], 4)

        ledger = protocol.close_attempt(
            ledger, "timing", "direct-current-main", 2, "PASS"
        )
        try:
            ledger = protocol.open_attempt(
                ledger, "timing", "common-lock-normalized", 1
            )
        except protocol.ProtocolError as error:
            self.fail(f"normalized lane-local attempt 1 was rejected: {error}")
        self.assertEqual(
            [item["attempt_id"] for item in ledger["attempts"]], [1, 2, 1]
        )
        self.assertEqual(
            lane(ledger, "timing", "common-lock-normalized")["attempt_ids"], [1]
        )
        self.assertEqual(ledger["next_transition_ordinal"], 6)

    def test_attempt_artifact_ownership_schema_is_exact(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        timing = protocol.open_attempt(
            ledger, "timing", "direct-current-main", 1
        )
        self.assertEqual(
            {
                name: timing["attempts"][0][name]
                for name in (
                    "artifact_root",
                    "artifact_device",
                    "artifact_inode",
                    "artifact_state",
                )
            },
            {
                "artifact_root": None,
                "artifact_device": None,
                "artifact_inode": None,
                "artifact_state": "NOT_APPLICABLE",
            },
        )

        ledger = protocol.new_ledger("a" * 40)
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(
                ledger, "allocation", "direct-current-main", 1
            )
        reserved = protocol.open_attempt(
            ledger,
            "allocation",
            "direct-current-main",
            1,
            artifact_root="/evidence/allocation-1",
        )
        attempt = reserved["attempts"][0]
        self.assertEqual(attempt["artifact_root"], "/evidence/allocation-1")
        self.assertEqual(attempt["artifact_state"], "RESERVED")
        self.assertIsNone(attempt["artifact_device"])
        self.assertIsNone(attempt["artifact_inode"])

        bound = protocol.bind_attempt_artifact(
            reserved,
            "allocation",
            "direct-current-main",
            1,
            artifact_root="/evidence/allocation-1",
            artifact_device=12,
            artifact_inode=34,
        )
        attempt = bound["attempts"][0]
        self.assertEqual(
            (attempt["artifact_state"], attempt["artifact_device"], attempt["artifact_inode"]),
            ("BOUND", 12, 34),
        )
        for name, invalid in (
            ("artifact_device", True),
            ("artifact_inode", -1),
            ("artifact_inode", 1 << 64),
            ("artifact_root", "relative"),
            ("artifact_state", "FOREIGN"),
        ):
            with self.subTest(name=name):
                mutated = copy.deepcopy(bound)
                mutated["attempts"][0][name] = invalid
                with self.assertRaises(protocol.ProtocolError):
                    protocol.validate_ledger(mutated)

    def test_only_validity_inconclusive_permits_a_whole_lane_retry(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(
            ledger,
            "allocation",
            "direct-current-main",
            1,
            artifact_root="/evidence/direct-attempt-1",
        )
        ledger = protocol.close_attempt(
            ledger,
            "allocation",
            "direct-current-main",
            1,
            None,
            validity_state="INCONCLUSIVE",
        )
        self.assertEqual(
            lane(ledger, "allocation", "direct-current-main")["state"], "RETRYABLE"
        )
        ledger = protocol.open_attempt(
            ledger,
            "allocation",
            "direct-current-main",
            2,
            artifact_root="/evidence/direct-attempt-2",
        )
        ledger = protocol.close_attempt(
            ledger, "allocation", "direct-current-main", 2, "PASS"
        )
        self.assertEqual(
            lane(ledger, "allocation", "direct-current-main")["state"], "COMPLETE"
        )
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(ledger, "allocation", "direct-current-main", 3)

    def test_complete_pass_fail_and_statistical_inconclusive_close_lane(self) -> None:
        for result in ("PASS", "FAIL", "INCONCLUSIVE"):
            with self.subTest(result=result):
                ledger = protocol.new_ledger("a" * 40)
                ledger = protocol.open_attempt(
                    ledger, "timing", "direct-current-main", 1
                )
                ledger = protocol.close_attempt(
                    ledger, "timing", "direct-current-main", 1, result
                )
                attempt = ledger["attempts"][0]
                self.assertEqual(attempt["validity_state"], "COMPLETE")
                self.assertEqual(attempt["statistical_result"], result)
                self.assertEqual(
                    lane(ledger, "timing", "direct-current-main")["state"],
                    "COMPLETE",
                )
                with self.assertRaises(protocol.ProtocolError):
                    protocol.open_attempt(
                        ledger, "timing", "direct-current-main", 2
                    )

    def test_direct_pass_is_required_before_normalized_lane_opens(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(
                ledger, "timing", "common-lock-normalized", 1
            )
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        ledger = protocol.close_attempt(
            ledger, "timing", "direct-current-main", 1, "PASS"
        )
        try:
            ledger = protocol.open_attempt(
                ledger, "timing", "common-lock-normalized", 1
            )
        except protocol.ProtocolError as error:
            self.fail(f"normalized lane-local attempt 1 was rejected: {error}")
        self.assertEqual(
            lane(ledger, "timing", "common-lock-normalized")["state"], "RUNNING"
        )

    def test_invalid_ledger_transitions_are_rejected_without_mutation(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        snapshot = json.dumps(ledger, sort_keys=True)
        invalid_closes = (
            ("allocation", "direct-current-main", 1, "PASS", {}),
            ("timing", "direct-current-main", 2, "PASS", {}),
            ("timing", "direct-current-main", 1, "UNKNOWN", {}),
            (
                "timing",
                "direct-current-main",
                1,
                "PASS",
                {"validity_state": "INCONCLUSIVE"},
            ),
        )
        for stage_name, lane_name, attempt_id, result, options in invalid_closes:
            with self.subTest(
                stage=stage_name, lane=lane_name, attempt_id=attempt_id, result=result
            ):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.close_attempt(
                        ledger,
                        stage_name,
                        lane_name,
                        attempt_id,
                        result,
                        **options,
                    )
                self.assertEqual(json.dumps(ledger, sort_keys=True), snapshot)

        ledger = protocol.close_attempt(
            ledger, "timing", "direct-current-main", 1, "FAIL"
        )
        with self.assertRaises(protocol.ProtocolError):
            protocol.close_attempt(
                ledger, "timing", "direct-current-main", 1, "FAIL"
            )

    def test_malformed_lane_state_is_rejected_before_a_transition(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        lane(ledger, "timing", "direct-current-main")["result"] = "PASS"
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(ledger, "timing", "direct-current-main", 1)

    def test_completed_fail_history_cannot_be_rewritten_as_retryable(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        ledger = protocol.close_attempt(
            ledger, "timing", "direct-current-main", 1, "FAIL"
        )
        summary = lane(ledger, "timing", "direct-current-main")
        summary["state"] = "RETRYABLE"
        summary["result"] = None

        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(ledger, "timing", "direct-current-main", 2)

    def test_statistical_inconclusive_history_cannot_be_rewritten_as_retryable(
        self,
    ) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        ledger = protocol.close_attempt(
            ledger, "timing", "direct-current-main", 1, "INCONCLUSIVE"
        )
        summary = lane(ledger, "timing", "direct-current-main")
        summary["state"] = "RETRYABLE"
        summary["result"] = None

        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(ledger, "timing", "direct-current-main", 2)

    def test_validity_inconclusive_history_requires_retryable_empty_summary(
        self,
    ) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        ledger = protocol.close_attempt(
            ledger,
            "timing",
            "direct-current-main",
            1,
            None,
            validity_state="INCONCLUSIVE",
        )

        tampered = copy.deepcopy(ledger)
        lane(tampered, "timing", "direct-current-main")["state"] = "READY"
        with self.assertRaises(protocol.ProtocolError):
            protocol.open_attempt(tampered, "timing", "direct-current-main", 2)

        retried = protocol.open_attempt(
            ledger, "timing", "direct-current-main", 2
        )
        self.assertEqual(
            lane(retried, "timing", "direct-current-main")["state"], "RUNNING"
        )

    def test_pass_history_must_match_terminal_lane_state_and_result(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        ledger = protocol.close_attempt(
            ledger, "timing", "direct-current-main", 1, "PASS"
        )
        try:
            ledger = protocol.open_attempt(
                ledger, "timing", "common-lock-normalized", 1
            )
        except protocol.ProtocolError as error:
            self.fail(f"normalized lane-local attempt 1 was rejected: {error}")
        ledger = protocol.close_attempt(
            ledger, "timing", "common-lock-normalized", 1, "PASS"
        )

        wrong_result = copy.deepcopy(ledger)
        lane(wrong_result, "timing", "common-lock-normalized")["result"] = "FAIL"
        with self.assertRaises(protocol.ProtocolError):
            validate_ledger(wrong_result)

        wrong_state = copy.deepcopy(ledger)
        summary = lane(wrong_state, "timing", "common-lock-normalized")
        summary["state"] = "RETRYABLE"
        summary["result"] = None
        with self.assertRaises(protocol.ProtocolError):
            validate_ledger(wrong_state)

    def test_legitimate_retry_history_has_lane_local_ids_and_global_ordinals(
        self,
    ) -> None:
        for stage_name in ("allocation", "timing"):
            with self.subTest(stage=stage_name):
                ledger = retry_then_normalized_pass(stage_name)
                validate_ledger(ledger)
                self.assertEqual(
                    lane(ledger, stage_name, "direct-current-main")["attempt_ids"],
                    [1, 2],
                )
                self.assertEqual(
                    lane(ledger, stage_name, "common-lock-normalized")[
                        "attempt_ids"
                    ],
                    [1],
                )
                self.assertEqual(
                    [
                        (attempt["open_ordinal"], attempt["close_ordinal"])
                        for attempt in ledger["attempts"]
                    ],
                    [(1, 2), (3, 4), (5, 6)],
                )
                self.assertEqual(ledger["next_transition_ordinal"], 7)

    def test_normalized_open_must_follow_direct_pass_close_in_both_stages(
        self,
    ) -> None:
        for stage_name in ("allocation", "timing"):
            with self.subTest(stage=stage_name):
                ledger = retry_then_normalized_pass(stage_name)
                direct_pass = ledger["attempts"][1]
                normalized = ledger["attempts"][2]
                direct_pass["close_ordinal"], normalized["open_ordinal"] = (
                    normalized["open_ordinal"],
                    direct_pass["close_ordinal"],
                )
                with self.assertRaises(protocol.ProtocolError):
                    validate_ledger(ledger)

    def test_transition_ordinals_reject_duplicates_gaps_and_reversal(self) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger = protocol.open_attempt(ledger, "timing", "direct-current-main", 1)
        ledger = protocol.close_attempt(
            ledger, "timing", "direct-current-main", 1, "PASS"
        )

        duplicate = copy.deepcopy(ledger)
        duplicate["attempts"][0]["close_ordinal"] = 1
        with self.assertRaises(protocol.ProtocolError):
            validate_ledger(duplicate)

        gap = copy.deepcopy(ledger)
        gap["attempts"][0]["close_ordinal"] = 3
        gap["next_transition_ordinal"] = 4
        with self.assertRaises(protocol.ProtocolError):
            validate_ledger(gap)

        reversed_order = copy.deepcopy(ledger)
        reversed_order["attempts"][0]["open_ordinal"] = 2
        reversed_order["attempts"][0]["close_ordinal"] = 1
        with self.assertRaises(protocol.ProtocolError):
            validate_ledger(reversed_order)

    def test_huge_transition_counter_is_rejected_without_counter_sized_work(
        self,
    ) -> None:
        ledger = protocol.new_ledger("a" * 40)
        ledger["next_transition_ordinal"] = 10**100
        real_range = range

        def bounded_range(*arguments):
            stop = arguments[0] if len(arguments) == 1 else arguments[1]
            if stop > 1_000:
                raise AssertionError("validator attempted counter-sized work")
            return real_range(*arguments)

        with mock.patch.object(protocol, "range", bounded_range, create=True):
            with self.assertRaises(protocol.ProtocolError):
                validate_ledger(ledger)

    def test_retry_open_must_follow_prior_validity_inconclusive_close(self) -> None:
        ledger = retry_then_normalized_pass("timing")
        first_direct, second_direct = ledger["attempts"][:2]
        first_direct["close_ordinal"], second_direct["open_ordinal"] = (
            second_direct["open_ordinal"],
            first_direct["close_ordinal"],
        )
        with self.assertRaises(protocol.ProtocolError):
            validate_ledger(ledger)


if __name__ == "__main__":
    unittest.main()
