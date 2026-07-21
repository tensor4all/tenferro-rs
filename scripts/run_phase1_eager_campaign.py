#!/usr/bin/env python3
"""Run one indivisible protocol-v2 Phase 2E timing campaign."""

from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import platform
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

try:
    from scripts import classify_criterion_noninferiority as classification
    from scripts import phase2e_build as build
    from scripts import phase2e_protocol as protocol
except ModuleNotFoundError:
    import classify_criterion_noninferiority as classification
    import phase2e_build as build
    import phase2e_protocol as protocol


CANONICAL_CASES = protocol.CANONICAL_CASES
PAIR_ORDERS = protocol.PAIR_ORDERS
RUN_ROLES = protocol.RUN_ROLES
THREAD_ENVIRONMENT = dict(protocol.THREAD_ENV)
SENTINEL_BENCHMARK = CANONICAL_CASES["lazy_neg_1"]
QUIET_DEADLINE_SECONDS = 300
QUIET_POLL_SECONDS = 1
PROCESS_DEADLINE_SECONDS = 30
TERMINATION_GRACE_SECONDS = 5
EXIT_BY_RESULT = {
    ("COMPLETE", "PASS"): 0,
    ("INCONCLUSIVE", None): 2,
    ("COMPLETE", "FAIL"): 3,
    ("COMPLETE", "INCONCLUSIVE"): 4,
}
EXECUTABLE_SEALS = (
    fcntl.F_SEAL_WRITE
    | fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_SEAL
)
FINALIZATION_STAGE = ".campaign-final.json"
FINALIZATION_MARKER = ".campaign-finalization.json"
FINALIZATION_PUBLISH = ".campaign-publish.json"
FINALIZATION_FILES = {
    FINALIZATION_STAGE,
    FINALIZATION_MARKER,
    FINALIZATION_PUBLISH,
}
MARKER_EXACT = "EXACT"
MARKER_ABSENT = "ABSENT"
MARKER_MISMATCH = "MISMATCH"
MARKER_UNKNOWN_IO = "UNKNOWN_IO"


class PinnedExecutable:
    """A retained source identity plus an immutable executable snapshot."""

    def __init__(
        self,
        logical_path: pathlib.Path,
        source_descriptor: int,
        descriptor: int,
        digest: str,
    ):
        self.logical_path = logical_path
        self.source_descriptor = source_descriptor
        self.descriptor = descriptor
        metadata = os.fstat(source_descriptor)
        self.device = metadata.st_dev
        self.inode = metadata.st_ino
        snapshot = os.fstat(descriptor)
        self.snapshot_device = snapshot.st_dev
        self.snapshot_inode = snapshot.st_ino
        self.digest = digest

    @classmethod
    def open(cls, path: pathlib.Path, expected_digest: str) -> "PinnedExecutable":
        logical = pathlib.Path(path)
        source_descriptor = os.open(
            logical,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        descriptor: int | None = None
        failure: BaseException | None = None
        result = None
        try:
            metadata = os.fstat(source_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise protocol.ProtocolError(
                    f"benchmark executable is not regular: {logical}"
                )
            if metadata.st_mode & 0o111 == 0:
                raise protocol.ProtocolError(
                    f"benchmark executable is not executable: {logical}"
                )
            digest = _sha256_open_file(source_descriptor)
            if digest != expected_digest:
                raise protocol.ProtocolError(
                    f"benchmark executable digest differs: {logical}"
                )
            descriptor = os.memfd_create(
                "phase2e-eager-executable",
                os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
            )
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short executable snapshot write")
                    view = view[written:]
            os.fchmod(descriptor, 0o500)
            if _sha256_open_file(descriptor) != expected_digest:
                raise protocol.ProtocolError("executable snapshot digest differs")
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, EXECUTABLE_SEALS)
            if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != EXECUTABLE_SEALS:
                raise protocol.ProtocolError("executable snapshot seals differ")
            if _sha256_open_file(source_descriptor) != expected_digest:
                raise protocol.ProtocolError(
                    f"benchmark executable changed while sealing: {logical}"
                )
            result = cls(
                logical, source_descriptor, descriptor, expected_digest
            )
            result.validate()
        except BaseException as error:
            failure = error
        if failure is not None:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            except BaseException:
                pass
            try:
                os.close(source_descriptor)
            except BaseException:
                pass
            raise failure
        if result is None:
            raise RuntimeError("executable pin completed without a result")
        return result

    @property
    def launch_path(self) -> pathlib.Path:
        path = pathlib.Path(f"/proc/self/fd/{self.descriptor}")
        if not sys.platform.startswith("linux") or not path.exists():
            raise protocol.ProtocolError(
                "pinned executable launch requires Linux /proc/self/fd"
            )
        return path

    def validate(self) -> None:
        opened = os.fstat(self.source_descriptor)
        current = os.stat(self.logical_path, follow_symlinks=False)
        snapshot = os.fstat(self.descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (self.device, self.inode)
            or (current.st_dev, current.st_ino) != (self.device, self.inode)
            or _sha256_open_file(self.source_descriptor) != self.digest
            or (snapshot.st_dev, snapshot.st_ino)
            != (self.snapshot_device, self.snapshot_inode)
            or fcntl.fcntl(self.descriptor, fcntl.F_GET_SEALS) != EXECUTABLE_SEALS
            or _sha256_open_file(self.descriptor) != self.digest
        ):
            raise protocol.ProtocolError(
                f"benchmark executable identity changed: {self.logical_path}"
            )

    def close(self) -> None:
        failure: BaseException | None = None
        for descriptor in (self.descriptor, self.source_descriptor):
            try:
                os.close(descriptor)
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure


def _sha256_open_file(descriptor: int) -> str:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after):
        raise protocol.ProtocolError("open executable changed while hashing")
    return digest.hexdigest()


def _advance_directory_descriptor(parent: int, child: int) -> int:
    """Transfer directory ownership without leaking on an interrupted close."""
    try:
        os.close(parent)
    except BaseException as error:
        try:
            os.close(child)
        except BaseException:
            pass
        raise error
    return child


def _open_absolute_directory(
    logical_path: pathlib.Path,
    *,
    component_observer: Callable[[pathlib.Path, int], None] | None = None,
) -> int:
    """Open an absolute directory one no-follow component at a time."""
    logical = pathlib.Path(os.path.abspath(logical_path))
    descriptor = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        opened_path = pathlib.Path("/")
        for component in logical.parts[1:]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            descriptor = _advance_directory_descriptor(descriptor, child)
            opened_path /= component
            if component_observer is not None:
                component_observer(opened_path, descriptor)
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        raise


class PinnedDirectory:
    """A no-follow directory identity retained across pathname replacement."""

    def __init__(self, logical_path: pathlib.Path):
        self.logical_path = pathlib.Path(os.path.abspath(logical_path))
        self.descriptor = _open_absolute_directory(self.logical_path)
        try:
            metadata = os.fstat(self.descriptor)
            self.identity = (metadata.st_dev, metadata.st_ino)
            self.validate_link()
            proc_path = pathlib.Path(f"/proc/self/fd/{self.descriptor}")
            if not sys.platform.startswith("linux") or not proc_path.exists():
                raise protocol.ProtocolError(
                    "pinned campaign roots require Linux /proc/self/fd"
                )
            self.proc_path = proc_path
        except BaseException:
            try:
                os.close(self.descriptor)
            except BaseException:
                pass
            raise

    @classmethod
    def create_fresh(
        cls,
        logical_path: pathlib.Path,
        *,
        pinned_observer: Callable[["PinnedDirectory"], None] | None = None,
        component_observer: Callable[[pathlib.Path, int], None] | None = None,
    ) -> "PinnedDirectory":
        """Create/open and prove an empty root through one retained descriptor."""
        logical = pathlib.Path(os.path.abspath(logical_path))
        parent_path = logical.parent
        parent = _open_absolute_directory(
            parent_path, component_observer=component_observer
        )
        descriptor: int | None = None
        failure: BaseException | None = None
        try:
            try:
                os.mkdir(logical.name, mode=0o700, dir_fd=parent)
                os.fsync(parent)
            except FileExistsError:
                pass
            descriptor = os.open(
                logical.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except BaseException as error:
            failure = error
        parent_to_close = parent
        parent = -1
        try:
            os.close(parent_to_close)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            if descriptor is not None:
                descriptor_to_close = descriptor
                descriptor = None
                try:
                    os.close(descriptor_to_close)
                except BaseException:
                    pass
            raise failure
        assert descriptor is not None
        result = cls.__new__(cls)
        result.logical_path = logical
        result.descriptor = descriptor
        try:
            metadata = os.fstat(descriptor)
            result.identity = (metadata.st_dev, metadata.st_ino)
            result.proc_path = pathlib.Path(f"/proc/self/fd/{descriptor}")
            if not sys.platform.startswith("linux") or not result.proc_path.exists():
                raise protocol.ProtocolError(
                    "pinned campaign roots require Linux /proc/self/fd"
                )
            if pinned_observer is not None:
                pinned_observer(result)
            result.validate_link()
            if os.listdir(descriptor):
                raise protocol.ProtocolError(
                    f"campaign root must be empty: {logical}"
                )
            result.validate_link()
            return result
        except BaseException:
            try:
                os.close(descriptor)
            except BaseException:
                pass
            raise

    def validate_link(self) -> None:
        try:
            current = os.stat(self.logical_path, follow_symlinks=False)
            opened = os.fstat(self.descriptor)
        except OSError as error:
            raise protocol.ProtocolError(
                f"campaign root identity changed: {self.logical_path}: {error}"
            ) from error
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != self.identity
            or (opened.st_dev, opened.st_ino) != self.identity
        ):
            raise protocol.ProtocolError(
                f"campaign root identity changed: {self.logical_path}"
            )

    @staticmethod
    def _parts(relative: str) -> tuple[str, ...]:
        path = pathlib.PurePosixPath(relative)
        if (
            not relative
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise protocol.ProtocolError(f"invalid root-relative path: {relative}")
        return path.parts

    def open_directory(self, relative: str, *, create: bool = False) -> int:
        descriptor = os.dup(self.descriptor)
        try:
            for component in self._parts(relative):
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                        os.fsync(descriptor)
                    except FileExistsError:
                        pass
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                descriptor = _advance_directory_descriptor(descriptor, child)
            return descriptor
        except BaseException:
            try:
                os.close(descriptor)
            except BaseException:
                pass
            raise

    def _open_parent(self, relative: str, *, create: bool = False) -> tuple[int, str]:
        parts = self._parts(relative)
        if len(parts) == 1:
            return os.dup(self.descriptor), parts[0]
        parent = self.open_directory("/".join(parts[:-1]), create=create)
        return parent, parts[-1]

    def open_file(
        self, relative: str, flags: int, mode: int = 0o600
    ) -> int:
        parent, name = self._open_parent(relative)
        descriptor: int | None = None
        failure: BaseException | None = None
        try:
            try:
                descriptor = os.open(
                    name,
                    flags | os.O_CLOEXEC | os.O_NOFOLLOW,
                    mode,
                    dir_fd=parent,
                )
            except FileNotFoundError:
                raise
            except OSError as error:
                raise protocol.ProtocolError(
                    f"cannot open root artifact {relative}: {error}"
                ) from error
        except BaseException as error:
            failure = error
        try:
            os.close(parent)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
            raise failure
        assert descriptor is not None
        return descriptor

    def atomic_json(self, relative: str, payload: Any) -> None:
        parent, name = self._open_parent(relative)
        failure: BaseException | None = None
        try:
            protocol.atomic_write_json_at(parent, name, payload)
        except BaseException as error:
            failure = error
        try:
            os.close(parent)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    def read_regular(self, relative: str) -> bytes:
        descriptor = self.open_file(relative, os.O_RDONLY | os.O_NONBLOCK)
        failure: BaseException | None = None
        content = b""
        try:
            before = os.fstat(descriptor)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            identity = lambda item: (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
            )
            if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after):
                raise protocol.ProtocolError(
                    f"root artifact changed while reading: {relative}"
                )
            content = b"".join(chunks)
        except BaseException as error:
            failure = error
        try:
            os.close(descriptor)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure
        return content

    def inventory(self) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        directories: set[str] = set()

        def visit(directory_fd: int, prefix: str) -> None:
            for name in sorted(os.listdir(directory_fd)):
                relative = f"{prefix}/{name}" if prefix else name
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    directories.add(relative)
                    child = os.open(
                        name,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        visit(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(metadata.st_mode):
                    files.add(relative)
                else:
                    raise protocol.ProtocolError(
                        f"invalid root artifact type: {relative}"
                    )

        visit(self.descriptor, "")
        return files, directories

    def close(self) -> None:
        os.close(self.descriptor)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def format_cpu_list(cpus) -> str:
    ordered = sorted(cpus)
    if not ordered:
        return ""
    ranges = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def criterion_directory(criterion_root: pathlib.Path, benchmark: str) -> pathlib.Path:
    components = benchmark.split("/")
    if len(components) != 4 or any(not component for component in components):
        raise protocol.ProtocolError(f"unexpected benchmark identifier: {benchmark}")
    group = f"{components[0]}_{components[1]}"
    return criterion_root / group / components[2] / components[3]


def criterion_relative(benchmark: str, suffix: str) -> str:
    return criterion_directory(pathlib.Path(""), benchmark).joinpath(suffix).as_posix()


def run_identities(order: str) -> tuple[str, str, str, str]:
    if order == "A/B":
        return "candidate", "baseline", "candidate", "candidate"
    if order == "B/A":
        return "candidate", "candidate", "baseline", "candidate"
    raise protocol.ProtocolError(f"unsupported pair order: {order}")


def benchmark_command(
    binary: pathlib.Path,
    benchmark: str,
    comparison_option: str,
    comparison_name: str,
) -> tuple[str, ...]:
    return (
        str(binary),
        "--bench",
        benchmark,
        comparison_option,
        comparison_name,
        "--noplot",
    )


def exact_build_processes() -> list[dict[str, Any]]:
    processes = []
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        return processes
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name in {"cargo", "rustc"}:
            processes.append({"pid": int(entry.name), "name": name})
    return sorted(processes, key=lambda record: record["pid"])


def _normalized_load(load_provider: Callable[[], float], allowed_count: int) -> float:
    return float(load_provider()) / allowed_count


def _is_within(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_roots(
    artifact_root: pathlib.Path,
    criterion_root: pathlib.Path,
    *,
    root_pin_observer: Callable[[str, PinnedDirectory], None] | None = None,
) -> tuple[PinnedDirectory, PinnedDirectory]:
    artifact_input = pathlib.Path(os.path.abspath(artifact_root))
    criterion_input = pathlib.Path(os.path.abspath(criterion_root))
    artifact = artifact_input.resolve(strict=False)
    criterion = criterion_input.resolve(strict=False)
    if artifact == criterion or _is_within(artifact, criterion) or _is_within(
        criterion, artifact
    ):
        raise protocol.ProtocolError("artifact and Criterion roots must be disjoint")
    artifact_handle = PinnedDirectory.create_fresh(
        artifact_input,
        pinned_observer=(
            None
            if root_pin_observer is None
            else lambda handle: root_pin_observer("artifact", handle)
        ),
    )
    try:
        criterion_handle = PinnedDirectory.create_fresh(
            criterion_input,
            pinned_observer=(
                None
                if root_pin_observer is None
                else lambda handle: root_pin_observer("criterion", handle)
            ),
        )
    except BaseException:
        artifact_handle.close()
        raise
    return artifact_handle, criterion_handle


def _read_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    value = classification.read_json(path)
    if type(value) is not dict:
        raise protocol.ProtocolError(f"{context} must be a JSON object")
    return value


def _build_inputs(
    args,
    validated_builds: Mapping[str, Mapping[str, Any]],
    authoritative_paths: Mapping[str, pathlib.Path] | None,
) -> tuple[
    dict[str, Any],
    dict[str, pathlib.Path],
    dict[str, str],
    dict[str, Mapping[str, Any]],
]:
    paths = {
        "baseline": pathlib.Path(args.baseline_build_manifest).resolve(strict=True),
        "candidate": pathlib.Path(args.candidate_build_manifest).resolve(strict=True),
    }
    manifests = {
        identity: _read_json(path, f"{identity} build manifest")
        for identity, path in paths.items()
    }
    expected_baseline_role = {
        "direct-current-main": "direct-current-main-baseline",
        "common-lock-normalized": "common-lock-normalized-baseline",
    }[args.comparison_kind]
    expected = {
        "baseline": validated_builds.get(expected_baseline_role),
        "candidate": validated_builds.get("candidate"),
    }
    if any(expected[identity] is None for identity in expected):
        raise protocol.ProtocolError("validated build set is incomplete")
    if any(manifests[identity] != expected[identity] for identity in manifests):
        raise protocol.ProtocolError(
            "retained build manifest differs from authoritative validation"
        )
    if authoritative_paths is not None:
        required_paths = {
            "baseline": pathlib.Path(authoritative_paths[expected_baseline_role]),
            "candidate": pathlib.Path(authoritative_paths["candidate"]),
        }
        if any(paths[name] != required_paths[name].resolve() for name in paths):
            raise protocol.ProtocolError(
                "build manifest path differs from authoritative evidence root"
            )
    build.validate_pair(args.comparison_kind, manifests["baseline"], manifests["candidate"])
    binaries = {
        identity: pathlib.Path(manifest["executable"]).resolve(strict=True)
        for identity, manifest in manifests.items()
    }
    binary_shas = {
        identity: protocol.sha256_file(binary) for identity, binary in binaries.items()
    }
    for identity, manifest in manifests.items():
        if binary_shas[identity] != manifest["executable_sha256"]:
            raise protocol.ProtocolError(f"{identity} executable digest changed")
    records = {
        identity: {
            "path": str(paths[identity]),
            "sha256": protocol.sha256_file(paths[identity]),
            "role": manifests[identity]["role"],
            "executable_sha256": binary_shas[identity],
        }
        for identity in ("baseline", "candidate")
    }
    return records, binaries, binary_shas, manifests


def _runtime_environment(
    candidate_manifest: Mapping[str, Any], criterion_root: pathlib.Path
) -> dict[str, str]:
    environment = candidate_manifest["environment"]
    return protocol.runtime_environment(
        path=environment["PATH"],
        home=environment["HOME"],
        criterion_home=str(criterion_root),
    )


def _pin_executables(
    binaries: Mapping[str, pathlib.Path], binary_shas: Mapping[str, str]
) -> dict[str, PinnedExecutable]:
    pinned: dict[str, PinnedExecutable] = {}
    try:
        for identity in ("baseline", "candidate"):
            pinned[identity] = PinnedExecutable.open(
                binaries[identity], binary_shas[identity]
            )
    except BaseException:
        for executable in pinned.values():
            try:
                executable.close()
            except BaseException:
                pass
        raise
    return pinned


def _sample_host(
    *,
    pid: int,
    phase: str,
    sequence: int,
    allowed_count: int,
    affinity_provider: Callable[[int], set[int]],
    load_provider: Callable[[], float],
    build_process_provider: Callable[[], list[dict[str, Any]]],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    try:
        affinity = format_cpu_list(affinity_provider(pid))
    except (ProcessLookupError, PermissionError, OSError):
        affinity = _proc_allowed_cpu_list(pid)
    processes = build_process_provider()
    return {
        "sequence": sequence,
        "phase": phase,
        "monotonic_seconds": float(monotonic()),
        "observed_affinity": affinity,
        "normalized_load": _normalized_load(load_provider, allowed_count),
        "cargo_processes": [record for record in processes if record.get("name") == "cargo"],
        "rustc_processes": [record for record in processes if record.get("name") == "rustc"],
    }


def _proc_allowed_cpu_list(pid: int) -> str:
    """Read the affinity of an exited-but-unreaped Linux child."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _exited_without_reaping(process) -> bool:
    """Observe a real child exit while retaining its proc endpoint metadata."""
    if not isinstance(process, subprocess.Popen):
        return process.poll() is not None
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            return os.waitid(os.P_PID, process.pid, flags) is not None
        except InterruptedError:
            continue


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _best_effort_signal(
    pid: int, requested_signal: signal.Signals, signal_process_group
) -> bool:
    try:
        signal_process_group(pid, requested_signal)
    except ProcessLookupError:
        return True
    except Exception:
        return False
    return True


def _terminate_group(process, signal_process_group) -> tuple[bool, bool, list[str]]:
    failures = []
    primary: BaseException | None = None
    try:
        terminated = _best_effort_signal(
            process.pid, signal.SIGTERM, signal_process_group
        )
    except BaseException as error:
        primary = error
        terminated = False
    if not terminated:
        failures.append("term-signal-failed")
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    except Exception as error:
        failures.append(f"term-wait-failed:{type(error).__name__}")
    except BaseException as error:
        if primary is None:
            primary = error
    try:
        killed = _best_effort_signal(
            process.pid, signal.SIGKILL, signal_process_group
        )
    except BaseException as error:
        if primary is None:
            primary = error
        killed = False
    if not killed:
        failures.append("kill-signal-failed")
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        failures.append("kill-wait-timeout")
    except Exception as error:
        failures.append(f"kill-wait-failed:{type(error).__name__}")
    except BaseException as error:
        if primary is None:
            primary = error
    if primary is not None:
        raise primary
    return terminated, killed, failures


def _cleanup_reaped_process_group(
    process_group: int,
    signal_process_group,
    monotonic,
    sleep,
) -> tuple[bool, bool, list[str]]:
    """Give descendants the fixed TERM grace, then kill the whole group."""
    failures: list[str] = []
    terminated = _best_effort_signal(
        process_group, signal.SIGTERM, signal_process_group
    )
    if not terminated:
        failures.append("term-signal-failed")
    deadline = float(monotonic()) + TERMINATION_GRACE_SECONDS
    while float(monotonic()) < deadline:
        sleep(min(0.1, deadline - float(monotonic())))
    killed = _best_effort_signal(
        process_group, signal.SIGKILL, signal_process_group
    )
    if not killed:
        failures.append("kill-signal-failed")
    disappearance_deadline = float(monotonic()) + TERMINATION_GRACE_SECONDS
    while _process_group_exists(process_group):
        now = float(monotonic())
        if now >= disappearance_deadline:
            failures.append("kill-group-survived")
            break
        sleep(min(0.01, disappearance_deadline - now))
    return terminated, killed, failures


def _process_record(
    *,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    cwd: pathlib.Path,
    artifact_root: PinnedDirectory,
    pair_relative: str,
    role: str,
    identity: str,
    binary_sha: str,
    executable: PinnedExecutable,
    selected_cpu: int,
    allowed_count: int,
    process_factory,
    signal_process_group,
    monotonic,
    sleep,
    affinity_provider,
    load_provider,
    build_process_provider,
    inherited_descriptors: tuple[int, ...],
    root_validators: tuple[Callable[[], None], ...],
    criterion_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    if criterion_binding is None:
        actual_home = environment.get("CRITERION_HOME", "")
        criterion_binding = {
            "logical_path": actual_home,
            "actual_home": actual_home,
            "device": 0,
            "inode": 0,
        }
    stdout_name = f"{role}.stdout.log"
    stderr_name = f"{role}.stderr.log"
    stdout_relative = f"{pair_relative}/{stdout_name}"
    stderr_relative = f"{pair_relative}/{stderr_name}"
    preamble = {
        "argv": list(command),
        "environment": dict(sorted(environment.items())),
        "environment_sha256": protocol.sha256_json(dict(sorted(environment.items()))),
        "executable": {
            "logical_path": str(executable.logical_path),
            "source_device": executable.device,
            "source_inode": executable.inode,
            "snapshot_device": executable.snapshot_device,
            "snapshot_inode": executable.snapshot_inode,
            "snapshot_sha256": executable.digest,
            "launch_path": str(executable.launch_path),
        },
        "criterion_binding": dict(criterion_binding),
    }
    process = None
    samples = []
    timed_out = False
    cleanup_failures: list[str] = []
    cleanup_terminated = False
    cleanup_killed = False
    survivor_observed = False
    stdout_descriptor = artifact_root.open_file(
        stdout_relative, os.O_WRONLY | os.O_CREAT | os.O_EXCL
    )
    try:
        stderr_descriptor = artifact_root.open_file(
            stderr_relative, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
    except BaseException:
        os.close(stdout_descriptor)
        raise
    with os.fdopen(stdout_descriptor, "w", encoding="utf-8") as stdout, os.fdopen(
        stderr_descriptor, "w", encoding="utf-8"
    ) as stderr:
        stdout.write(json.dumps(preamble, sort_keys=True) + "\n")
        stdout.flush()
        try:
            executable.validate()
            for validate_root in root_validators:
                validate_root()
            process = process_factory(
                list(command),
                cwd=str(cwd),
                env=dict(sorted(environment.items())),
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
                pass_fds=tuple(
                    sorted({executable.descriptor, *inherited_descriptors})
                ),
                preexec_fn=lambda: os.sched_setaffinity(0, {selected_cpu}),
            )
            started = float(monotonic())
            samples.append(
                _sample_host(
                    pid=process.pid,
                    phase="start",
                    sequence=0,
                    allowed_count=allowed_count,
                    affinity_provider=affinity_provider,
                    load_provider=load_provider,
                    build_process_provider=build_process_provider,
                    monotonic=monotonic,
                )
            )
            executable.validate()
            for validate_root in root_validators:
                validate_root()
            deadline = started + PROCESS_DEADLINE_SECONDS
            while True:
                if _exited_without_reaping(process):
                    break
                now = float(monotonic())
                if now >= deadline:
                    timed_out = True
                    cleanup_terminated, cleanup_killed, cleanup_failures = _terminate_group(
                        process, signal_process_group
                    )
                    status = process.returncode
                    break
                sleep(min(1.0, deadline - now))
                if not _exited_without_reaping(process):
                    if float(monotonic()) >= deadline:
                        timed_out = True
                        cleanup_terminated, cleanup_killed, cleanup_failures = _terminate_group(
                            process, signal_process_group
                        )
                        status = process.returncode
                        break
                    samples.append(
                        _sample_host(
                            pid=process.pid,
                            phase="periodic",
                            sequence=len(samples),
                            allowed_count=allowed_count,
                            affinity_provider=affinity_provider,
                            load_provider=load_provider,
                            build_process_provider=build_process_provider,
                            monotonic=monotonic,
                        )
                    )
            ended = max(float(monotonic()), started + 1e-9)
            samples.append(
                _sample_host(
                    pid=process.pid,
                    phase="end",
                    sequence=len(samples),
                    allowed_count=allowed_count,
                    affinity_provider=affinity_provider,
                    load_provider=load_provider,
                    build_process_provider=build_process_provider,
                    monotonic=lambda: ended,
                )
            )
            if not timed_out:
                status = process.wait()
                if isinstance(process, subprocess.Popen) and _process_group_exists(
                    process.pid
                ):
                    survivor_observed = True
                    cleanup_failures.append("normal-exit-group-survivor")
                    cleanup_terminated, cleanup_killed, survivor_failures = (
                        _cleanup_reaped_process_group(
                            process.pid,
                            signal_process_group,
                            monotonic,
                            sleep,
                        )
                    )
                    cleanup_failures.extend(survivor_failures)
            executable.validate()
            for validate_root in root_validators:
                validate_root()
        except BaseException:
            if process is not None:
                try:
                    _terminate_group(process, signal_process_group)
                except BaseException:
                    pass
            raise

    reason = None
    if timed_out:
        reason = "benchmark-process-timeout"
        if cleanup_failures:
            reason += ":" + "+".join(cleanup_failures)
    elif cleanup_failures:
        reason = "benchmark-process-survivor:" + "+".join(cleanup_failures)
    elif status != 0:
        reason = f"benchmark-process-exit:{status}"
    elif any(
        sample["observed_affinity"] != str(selected_cpu)
        or sample["normalized_load"] > 0.25
        or sample["cargo_processes"]
        or sample["rustc_processes"]
        for sample in samples
    ):
        reason = "benchmark-monitor-invalid"
    return (
        {
            "role": role,
            "binary": identity,
            "binary_sha256": binary_sha,
            "validity_state": "COMPLETE" if reason is None else "INCONCLUSIVE",
            "exit_status": int(status if status is not None else -1),
            "stdout_artifact": stdout_name,
            "stderr_artifact": stderr_name,
            "process_started_monotonic": started,
            "process_ended_monotonic": ended,
            "monitor_samples": samples,
            "argv": list(command),
            "environment": dict(sorted(environment.items())),
            "environment_sha256": protocol.sha256_json(
                dict(sorted(environment.items()))
            ),
            "executable": copy.deepcopy(preamble["executable"]),
            "criterion_binding": dict(criterion_binding),
            "process_group_cleanup": {
                "survivor_observed": survivor_observed,
                "term_signal_sent": cleanup_terminated,
                "kill_signal_sent": cleanup_killed,
                "failures": cleanup_failures,
            },
        },
        reason,
    )


def _copy_regular_at(
    source_root: PinnedDirectory,
    source_relative: str,
    destination_directory: int,
    destination_name: str,
) -> None:
    content = source_root.read_regular(source_relative)
    descriptor = os.open(
        destination_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=destination_directory,
    )
    failure: BaseException | None = None
    try:
        view = memoryview(content)
        while view:
            try:
                written = os.write(descriptor, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError("short Criterion estimate copy")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException as error:
        failure = error
    try:
        os.close(descriptor)
    except BaseException as error:
        if failure is None:
            failure = error
    if failure is not None:
        raise failure
    os.fsync(destination_directory)


def _artifact_record(root: PinnedDirectory, relative: str) -> dict[str, str]:
    return {"sha256": hashlib.sha256(root.read_regular(relative)).hexdigest()}


def _record_artifact(
    campaign: dict[str, Any], root: PinnedDirectory, relative: str
) -> dict[str, str]:
    record = _artifact_record(root, relative)
    if relative in campaign["artifact_inventory"]:
        raise protocol.ProtocolError(f"artifact registered more than once: {relative}")
    campaign["artifact_inventory"][relative] = record
    return record


def _synchronize_prefix_inventory(
    campaign: dict[str, Any], artifact_root: PinnedDirectory
) -> None:
    """Make an invalid campaign's inventory match every durable prefix file."""
    discovered: dict[str, dict[str, str]] = {}
    files, _directories = artifact_root.inventory()
    for relative in sorted(files):
        if relative == "campaign.json":
            continue
        discovered[relative] = _artifact_record(artifact_root, relative)
    campaign["artifact_inventory"] = discovered


def _quiet_host(
    *,
    allowed_count: int,
    load_limit: float,
    monotonic,
    sleep,
    load_provider,
    build_process_provider,
) -> str | None:
    started = float(monotonic())
    while True:
        load = _normalized_load(load_provider, allowed_count)
        processes = build_process_provider()
        if load <= load_limit and not processes:
            return None
        now = float(monotonic())
        if now - started >= QUIET_DEADLINE_SECONDS:
            return "quiet-host-timeout"
        sleep(min(float(QUIET_POLL_SECONDS), QUIET_DEADLINE_SECONDS - (now - started)))


def _pair_specs(case: str, benchmark: str, pair: int, order: str):
    identities = run_identities(order)
    target_name = f"phase2e-target-{case}-p{pair}"
    sentinel_name = f"phase2e-sentinel-{case}-p{pair}"
    return (
        (RUN_ROLES[0], identities[0], SENTINEL_BENCHMARK, "--save-baseline", sentinel_name),
        (RUN_ROLES[1], identities[1], benchmark, "--save-baseline", target_name),
        (RUN_ROLES[2], identities[2], benchmark, "--baseline", target_name),
        (RUN_ROLES[3], identities[3], SENTINEL_BENCHMARK, "--baseline", sentinel_name),
    )


def _write_monitor_artifact(
    pair_directory: int, case: str, pair: int, runs: list[dict[str, Any]]
) -> str:
    name = "monitor-samples.json"
    protocol.atomic_write_json_at(
        pair_directory,
        name,
        {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "case": case,
            "pair": pair,
            "runs": {run["role"]: copy.deepcopy(run["monitor_samples"]) for run in runs},
        },
    )
    return name


def _invalid_campaign(
    campaign: dict[str, Any], *, case: str, pair: int, role: str, reason: str
) -> dict[str, Any]:
    terminal = copy.deepcopy(campaign)
    terminal["validity_state"] = "INCONCLUSIVE"
    terminal["statistical_result"] = None
    terminal["completed_at"] = utc_now()
    terminal["invalid"] = {
        "case": case,
        "pair": pair,
        "role": role,
        "reason": reason,
    }
    terminal["prefix_inventory"] = copy.deepcopy(terminal["artifact_inventory"])
    return terminal


def _initial_campaign(
    args,
    *,
    build_records,
    selected_cpu: int,
    allowed_cpus: set[int],
    candidate_sha: str,
    criterion_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.sha256_file(pathlib.Path(protocol.__file__)),
        "classifier_sha256": protocol.sha256_file(pathlib.Path(classification.__file__)),
        "candidate_sha": candidate_sha,
        "comparison_kind": args.comparison_kind,
        "build_manifests": build_records,
        "selected_cpu": selected_cpu,
        "allowed_cpus": format_cpu_list(allowed_cpus),
        "allowed_cpu_count": len(allowed_cpus),
        "normalized_load_limit": args.normalized_load_limit,
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "orders": list(PAIR_ORDERS),
        "criterion": dict(classification.CRITERION_SETTINGS),
        "criterion_binding": dict(criterion_binding),
        "validity_state": "RUNNING",
        "statistical_result": None,
        "completed_at": "",
        "cases": {
            case: {"benchmark": benchmark, "statistical_result": None, "pairs": {}}
            for case, benchmark in CANONICAL_CASES.items()
        },
        "artifact_inventory": {},
        "classification_artifacts": None,
    }


def _declare_results(campaign: dict[str, Any], root: PinnedDirectory) -> str:
    results = {}
    for case in sorted(CANONICAL_CASES):
        intervals = []
        for pair, order in enumerate(PAIR_ORDERS, start=1):
            relative = f"{case}/pair{pair}/change-estimates.json"
            estimate = classification._read_change_content(
                root.read_regular(relative), relative
            )
            intervals.append(
                classification.invert_interval(*estimate) if order == "B/A" else {
                    "lower": estimate[0], "upper": estimate[1], "point": estimate[2]
                }
            )
        normalized = [
            value
            if isinstance(value, Mapping)
            else {"lower": value[0], "upper": value[1], "point": value[2]}
            for value in intervals
        ]
        result = classification.classify_case(normalized)
        campaign["cases"][case]["statistical_result"] = result
        results[case] = result
    return classification.campaign_result(results)


def _close_ledger(
    ledger_path: pathlib.Path,
    ledger: dict[str, Any],
    args,
    result: str | None,
    validity_state: str,
    atomic_writer,
) -> None:
    closed = protocol.close_attempt(
        ledger,
        "timing",
        args.comparison_kind,
        args.attempt_id,
        result,
        validity_state=validity_state,
    )
    atomic_writer(ledger_path, closed)


def _finalization_marker(terminal: Mapping[str, Any], args) -> dict[str, Any]:
    return {
        "version": 1,
        "candidate_sha": terminal["candidate_sha"],
        "comparison_kind": args.comparison_kind,
        "attempt_id": args.attempt_id,
        "campaign_sha256": protocol.sha256_json(terminal),
        "statistical_result": terminal["statistical_result"],
    }


def _read_root_json(root: PinnedDirectory, relative: str) -> dict[str, Any]:
    try:
        value = json.loads(root.read_regular(relative))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise protocol.ProtocolError(f"invalid finalization JSON: {relative}") from error
    if type(value) is not dict:
        raise protocol.ProtocolError(f"finalization JSON is not an object: {relative}")
    return value


def _validate_finalization_marker(marker: Mapping[str, Any], args) -> None:
    protocol.validate_manifest_fields(
        marker,
        {
            "version": int,
            "candidate_sha": str,
            "comparison_kind": str,
            "attempt_id": int,
            "campaign_sha256": str,
            "statistical_result": str,
        },
        context="campaign finalization marker",
    )
    if (
        marker["version"] != 1
        or marker["comparison_kind"] != args.comparison_kind
        or marker["attempt_id"] != args.attempt_id
        or marker["statistical_result"] not in {"PASS", "FAIL", "INCONCLUSIVE"}
    ):
        raise protocol.ProtocolError("campaign finalization marker identity differs")


def _probe_committed_marker(
    root: PinnedDirectory, marker: Mapping[str, Any]
) -> str:
    """Classify marker state without turning uncertain I/O into absence."""
    try:
        content = root.read_regular(FINALIZATION_MARKER)
    except FileNotFoundError:
        return MARKER_ABSENT
    except BaseException:
        return MARKER_UNKNOWN_IO
    try:
        if content == protocol._canonical_json_bytes(marker):
            return MARKER_EXACT
        return MARKER_MISMATCH
    except BaseException:
        return MARKER_UNKNOWN_IO


def _ledger_attempt(ledger: Mapping[str, Any], args) -> Mapping[str, Any]:
    matches = [
        attempt
        for attempt in ledger["attempts"]
        if attempt["stage"] == "timing"
        and attempt["lane"] == args.comparison_kind
        and attempt["attempt_id"] == args.attempt_id
    ]
    if len(matches) != 1:
        raise protocol.ProtocolError("finalization ledger attempt is not unique")
    return matches[0]


def _require_closed_attempt(
    ledger: Mapping[str, Any], args, statistical_result: str
) -> None:
    attempt = _ledger_attempt(ledger, args)
    if (
        ledger["active_attempt_id"] is not None
        or attempt["state"] != "COMPLETE"
        or attempt["validity_state"] != "COMPLETE"
        or attempt["statistical_result"] != statistical_result
    ):
        raise protocol.ProtocolError("finalization ledger terminal state differs")


def _finalization_state_kind(
    validity_state: str,
    marker_exists: bool,
    stage_exists: bool,
    publish_exists: bool,
    ledger_active: bool,
) -> str:
    """Classify only crash states reachable by the finalization protocol."""
    state = (
        validity_state,
        marker_exists,
        stage_exists,
        publish_exists,
        ledger_active,
    )
    if state == ("RUNNING", False, False, False, True):
        return "NO_RESUME"
    if state == ("RUNNING", False, True, False, True):
        return "NO_RESUME"
    if (
        validity_state == "RUNNING"
        and marker_exists
        and stage_exists
    ):
        return "RUNNING"
    if (
        validity_state == "COMPLETE"
        and not ledger_active
        and not publish_exists
        and ((marker_exists and stage_exists) or not marker_exists)
    ):
        return "COMPLETE"
    raise protocol.ProtocolError("unreachable campaign finalization state")


def _require_same_regular_inode(
    root: PinnedDirectory, first: str, second: str
) -> None:
    try:
        first_metadata = os.stat(
            first, dir_fd=root.descriptor, follow_symlinks=False
        )
        second_metadata = os.stat(
            second, dir_fd=root.descriptor, follow_symlinks=False
        )
    except OSError as error:
        raise protocol.ProtocolError(
            "cannot authenticate finalization publish partial"
        ) from error
    if (
        not stat.S_ISREG(first_metadata.st_mode)
        or not stat.S_ISREG(second_metadata.st_mode)
        or (first_metadata.st_dev, first_metadata.st_ino)
        != (second_metadata.st_dev, second_metadata.st_ino)
    ):
        raise protocol.ProtocolError(
            "finalization publish partial is not the stage hard link"
        )


def _publish_staged_campaign(root: PinnedDirectory) -> None:
    try:
        os.link(
            FINALIZATION_STAGE,
            FINALIZATION_PUBLISH,
            src_dir_fd=root.descriptor,
            dst_dir_fd=root.descriptor,
            follow_symlinks=False,
        )
        os.fsync(root.descriptor)
    except FileExistsError:
        pass
    _require_same_regular_inode(root, FINALIZATION_STAGE, FINALIZATION_PUBLISH)
    os.replace(
        FINALIZATION_PUBLISH,
        "campaign.json",
        src_dir_fd=root.descriptor,
        dst_dir_fd=root.descriptor,
    )


def _unlink_finalization_file(root: PinnedDirectory, name: str) -> None:
    try:
        os.unlink(name, dir_fd=root.descriptor)
    except FileNotFoundError:
        pass
    os.fsync(root.descriptor)


def _validate_completed_campaign(
    root: PinnedDirectory,
    args,
    ledger: Mapping[str, Any],
    *,
    marker: Mapping[str, Any] | None,
    files: set[str],
) -> tuple[dict[str, Any], int]:
    campaign = _read_root_json(root, "campaign.json")
    statistical_result = campaign.get("statistical_result")
    if (
        campaign.get("validity_state") != "COMPLETE"
        or statistical_result not in {"PASS", "FAIL", "INCONCLUSIVE"}
        or campaign.get("comparison_kind") != args.comparison_kind
        or campaign.get("candidate_sha") != ledger["candidate_sha"]
    ):
        raise protocol.ProtocolError("completed recovery campaign identity differs")
    campaign_digest = protocol.sha256_json(campaign)
    if marker is not None:
        if (
            marker["campaign_sha256"] != campaign_digest
            or marker["candidate_sha"] != campaign["candidate_sha"]
            or marker["statistical_result"] != statistical_result
        ):
            raise protocol.ProtocolError("completed recovery marker differs")
    for name in (FINALIZATION_STAGE, FINALIZATION_PUBLISH):
        if name in files and protocol.sha256_json(
            _read_root_json(root, name)
        ) != campaign_digest:
            raise protocol.ProtocolError(f"completed recovery partial differs: {name}")
    if FINALIZATION_STAGE in files:
        _require_same_regular_inode(root, "campaign.json", FINALIZATION_STAGE)
    classification.classify_campaign_retained(
        root.logical_path / "campaign.json",
        root.logical_path,
        root_descriptor=root.descriptor,
        ignored_root_files=set(FINALIZATION_FILES),
    )
    root.validate_link()
    _require_closed_attempt(ledger, args, statistical_result)
    return campaign, EXIT_BY_RESULT[("COMPLETE", statistical_result)]


def _recover_finalization(args, atomic_writer) -> int | None:
    artifact_path = pathlib.Path(os.path.abspath(args.artifact_root))
    try:
        metadata = os.lstat(artifact_path)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        return None
    root = PinnedDirectory(artifact_path)
    try:
        files, _directories = root.inventory()
        if "campaign.json" not in files:
            if files.intersection(FINALIZATION_FILES):
                raise protocol.ProtocolError(
                    "finalization partial exists without campaign.json"
                )
            return None
        ledger_path = pathlib.Path(args.ledger).resolve(strict=True)
        ledger = _read_json(ledger_path, "evidence ledger")
        protocol.validate_ledger(ledger)
        marker = None
        if FINALIZATION_MARKER in files:
            marker = _read_root_json(root, FINALIZATION_MARKER)
            _validate_finalization_marker(marker, args)
            if ledger["candidate_sha"] != marker["candidate_sha"]:
                raise protocol.ProtocolError("finalization ledger candidate differs")

        persisted = _read_root_json(root, "campaign.json")
        active_attempt = ledger["active_attempt_id"]
        if active_attempt not in (None, args.attempt_id):
            raise protocol.ProtocolError(
                "finalization ledger has a different active attempt"
            )
        state_kind = _finalization_state_kind(
            persisted.get("validity_state"),
            FINALIZATION_MARKER in files,
            FINALIZATION_STAGE in files,
            FINALIZATION_PUBLISH in files,
            active_attempt == args.attempt_id,
        )
        if state_kind == "NO_RESUME":
            return None
        if state_kind == "COMPLETE":
            _campaign, exit_code = _validate_completed_campaign(
                root, args, ledger, marker=marker, files=files
            )
            for name in (
                FINALIZATION_MARKER,
                FINALIZATION_PUBLISH,
                FINALIZATION_STAGE,
            ):
                _unlink_finalization_file(root, name)
            root.validate_link()
            return exit_code

        assert state_kind == "RUNNING"
        assert marker is not None
        terminal = _read_root_json(root, FINALIZATION_STAGE)
        if protocol.sha256_json(terminal) != marker["campaign_sha256"]:
            raise protocol.ProtocolError("finalization campaign digest differs")
        if (
            terminal.get("candidate_sha") != marker["candidate_sha"]
            or terminal.get("comparison_kind") != args.comparison_kind
            or terminal.get("validity_state") != "COMPLETE"
            or terminal.get("statistical_result") != marker["statistical_result"]
        ):
            raise protocol.ProtocolError("finalization campaign identity differs")
        classification.classify_terminal_view(
            artifact_path / "campaign.json",
            terminal,
            artifact_path,
            root_descriptor=root.descriptor,
        )
        root.validate_link()

        attempt = _ledger_attempt(ledger, args)
        if attempt["state"] == "RUNNING" and ledger["active_attempt_id"] == args.attempt_id:
            _close_ledger(
                ledger_path,
                ledger,
                args,
                marker["statistical_result"],
                "COMPLETE",
                atomic_writer,
            )
            ledger = _read_json(ledger_path, "evidence ledger")
            protocol.validate_ledger(ledger)
        else:
            _require_closed_attempt(ledger, args, marker["statistical_result"])

        _publish_staged_campaign(root)
        os.fsync(root.descriptor)
        root.validate_link()
        files, _directories = root.inventory()
        _campaign, exit_code = _validate_completed_campaign(
            root, args, ledger, marker=marker, files=files
        )
        _unlink_finalization_file(root, FINALIZATION_MARKER)
        _unlink_finalization_file(root, FINALIZATION_PUBLISH)
        _unlink_finalization_file(root, FINALIZATION_STAGE)
        root.validate_link()
        return exit_code
    finally:
        root.close()


def _run_campaign(
    args,
    *,
    validated_builds: Mapping[str, Mapping[str, Any]],
    authoritative_manifest_paths: Mapping[str, pathlib.Path] | None = None,
    process_factory=subprocess.Popen,
    signal_process_group=os.killpg,
    monotonic=time.monotonic,
    sleep=time.sleep,
    affinity_provider=os.sched_getaffinity,
    allowed_cpu_provider=lambda: set(os.sched_getaffinity(0)),
    load_provider=lambda: os.getloadavg()[0],
    build_process_provider=exact_build_processes,
    atomic_writer=protocol.atomic_write_json,
    campaign_write_observer: Callable[[pathlib.Path, Mapping[str, Any]], None]
    | None = None,
    root_pin_observer: Callable[[str, PinnedDirectory], None] | None = None,
    finalization_observer: Callable[[str], None] | None = None,
) -> int:
    manifest_path = None
    campaign = None
    ledger = None
    ledger_closed = False
    pinned_executables: dict[str, PinnedExecutable] = {}
    artifact_handle: PinnedDirectory | None = None
    criterion_handle: PinnedDirectory | None = None
    pair_descriptor: int | None = None
    finalization_started = False
    current = {"case": "<startup>", "pair": 0, "role": "<none>"}
    try:
        recovered = _recover_finalization(args, atomic_writer)
        if recovered is not None:
            return recovered
        if args.comparison_kind not in protocol.LANE_NAMES:
            raise protocol.ProtocolError("invalid comparison kind")
        if args.normalized_load_limit != 0.25:
            raise protocol.ProtocolError("normalized load limit must be exactly 0.25")
        artifact_handle, criterion_handle = _prepare_roots(
            args.artifact_root,
            args.criterion_root,
            root_pin_observer=root_pin_observer,
        )
        artifact_root = artifact_handle.logical_path
        criterion_root = criterion_handle.logical_path
        args.artifact_root = artifact_root
        args.criterion_root = criterion_root
        build_records, binaries, binary_shas, build_manifests = _build_inputs(
            args, validated_builds, authoritative_manifest_paths
        )
        pinned_executables = _pin_executables(binaries, binary_shas)
        for identity, executable in pinned_executables.items():
            build_records[identity].update(
                {
                    "executable_path": str(executable.logical_path),
                    "executable_device": executable.device,
                    "executable_inode": executable.inode,
                    "snapshot_sha256": executable.digest,
                    "snapshot_device": executable.snapshot_device,
                    "snapshot_inode": executable.snapshot_inode,
                }
            )
        for path in (
            pathlib.Path(args.baseline_build_manifest).resolve(),
            pathlib.Path(args.candidate_build_manifest).resolve(),
            pathlib.Path(args.ledger).resolve(),
        ):
            if _is_within(path, artifact_root) or _is_within(path, criterion_root):
                raise protocol.ProtocolError("read-only campaign input is inside a fresh root")
        ledger_path = pathlib.Path(args.ledger).resolve(strict=True)
        ledger = _read_json(ledger_path, "evidence ledger")
        allowed_cpus = set(allowed_cpu_provider())
        if not allowed_cpus:
            raise protocol.ProtocolError("process has no allowed CPUs")
        selected_cpu = min(allowed_cpus) if args.cpu is None else args.cpu
        if selected_cpu not in allowed_cpus:
            raise protocol.ProtocolError("selected CPU is not process-allowed")
        criterion_binding = {
            "logical_path": str(criterion_root),
            "actual_home": str(criterion_handle.proc_path),
            "device": criterion_handle.identity[0],
            "inode": criterion_handle.identity[1],
        }
        environment = _runtime_environment(
            build_manifests["candidate"], criterion_handle.proc_path
        )
        campaign = _initial_campaign(
            args,
            build_records=build_records,
            selected_cpu=selected_cpu,
            allowed_cpus=allowed_cpus,
            candidate_sha=build_manifests["candidate"]["head"],
            criterion_binding=criterion_binding,
        )
        if ledger.get("candidate_sha") != campaign["candidate_sha"]:
            raise protocol.ProtocolError(
                "evidence ledger candidate differs from candidate build"
            )
        opened = protocol.open_attempt(
            ledger, "timing", args.comparison_kind, args.attempt_id
        )
        atomic_writer(ledger_path, opened)
        ledger = opened
        manifest_path = artifact_root / "campaign.json"

        def write_campaign(payload: Mapping[str, Any]) -> None:
            if campaign_write_observer is not None:
                campaign_write_observer(manifest_path, payload)
            artifact_handle.atomic_json("campaign.json", payload)

        write_campaign(campaign)

        for case in sorted(CANONICAL_CASES):
            benchmark = CANONICAL_CASES[case]
            for pair, order in enumerate(PAIR_ORDERS, start=1):
                current = {"case": case, "pair": pair, "role": "quiet_wait"}
                quiet_error = _quiet_host(
                    allowed_count=len(allowed_cpus),
                    load_limit=args.normalized_load_limit,
                    monotonic=monotonic,
                    sleep=sleep,
                    load_provider=load_provider,
                    build_process_provider=build_process_provider,
                )
                if quiet_error is not None:
                    _synchronize_prefix_inventory(campaign, artifact_handle)
                    terminal = _invalid_campaign(campaign, reason=quiet_error, **current)
                    write_campaign(terminal)
                    _close_ledger(
                        pathlib.Path(args.ledger), ledger, args, None, "INCONCLUSIVE", atomic_writer
                    )
                    return 2

                pair_relative = f"{case}/pair{pair}"
                pair_descriptor = artifact_handle.open_directory(
                    pair_relative, create=True
                )
                runs = []
                local_artifacts = {}
                invalid_reason = None
                for role, identity, run_benchmark, option, name in _pair_specs(
                    case, benchmark, pair, order
                ):
                    current = {"case": case, "pair": pair, "role": role}
                    record, invalid_reason = _process_record(
                        command=benchmark_command(
                            pinned_executables[identity].launch_path,
                            run_benchmark,
                            option,
                            name,
                        ),
                        environment=environment,
                        cwd=pathlib.Path(args.working_directory),
                        artifact_root=artifact_handle,
                        pair_relative=pair_relative,
                        role=role,
                        identity=identity,
                        binary_sha=binary_shas[identity],
                        executable=pinned_executables[identity],
                        selected_cpu=selected_cpu,
                        allowed_count=len(allowed_cpus),
                        process_factory=process_factory,
                        signal_process_group=signal_process_group,
                        monotonic=monotonic,
                        sleep=sleep,
                        affinity_provider=affinity_provider,
                        load_provider=load_provider,
                        build_process_provider=build_process_provider,
                        inherited_descriptors=(criterion_handle.descriptor,),
                        root_validators=(
                            artifact_handle.validate_link,
                            criterion_handle.validate_link,
                        ),
                        criterion_binding=criterion_binding,
                    )
                    runs.append(record)
                    for name in (
                        f"{role}.stdout.log",
                        f"{role}.stderr.log",
                    ):
                        local_artifacts[name] = _record_artifact(
                            campaign, artifact_handle, f"{pair_relative}/{name}"
                        )
                    campaign["cases"][case]["active_pair"] = {
                        "pair": pair,
                        "order": order,
                        "runs": copy.deepcopy(runs),
                    }
                    write_campaign(campaign)
                    if invalid_reason is not None:
                        break
                    if role == "second_target":
                        _copy_regular_at(
                            criterion_handle,
                            criterion_relative(benchmark, "change/estimates.json"),
                            pair_descriptor,
                            "change-estimates.json",
                        )
                        local_artifacts["change-estimates.json"] = _record_artifact(
                            campaign,
                            artifact_handle,
                            f"{pair_relative}/change-estimates.json",
                        )
                        write_campaign(campaign)
                    elif role == "sentinel_after":
                        _copy_regular_at(
                            criterion_handle,
                            criterion_relative(
                                SENTINEL_BENCHMARK, "change/estimates.json"
                            ),
                            pair_descriptor,
                            "sentinel-change-estimates.json",
                        )
                        local_artifacts[
                            "sentinel-change-estimates.json"
                        ] = _record_artifact(
                            campaign,
                            artifact_handle,
                            f"{pair_relative}/sentinel-change-estimates.json",
                        )
                        write_campaign(campaign)

                if invalid_reason is None:
                    sentinel = classification._read_change_content(
                        artifact_handle.read_regular(
                            f"{pair_relative}/sentinel-change-estimates.json"
                        ),
                        f"{pair_relative}/sentinel-change-estimates.json",
                    )
                    if classification.sentinel_breached(sentinel[0], sentinel[1]):
                        invalid_reason = "sentinel interval breaches drift band"

                monitor_name = _write_monitor_artifact(
                    pair_descriptor, case, pair, runs
                )
                local_artifacts[monitor_name] = _record_artifact(
                    campaign,
                    artifact_handle,
                    f"{pair_relative}/{monitor_name}",
                )
                validity = {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "case": case,
                    "pair": pair,
                    "order": order,
                    "selected_cpu": selected_cpu,
                    "allowed_cpu_count": len(allowed_cpus),
                    "validity_state": "COMPLETE" if invalid_reason is None else "INCONCLUSIVE",
                    "runs": runs,
                    "artifacts": local_artifacts,
                }
                if invalid_reason is not None:
                    validity["reason"] = invalid_reason
                protocol.atomic_write_json_at(
                    pair_descriptor, "validity.json", validity
                )
                validity_relative = f"{pair_relative}/validity.json"
                validity_record = _record_artifact(
                    campaign, artifact_handle, validity_relative
                )
                campaign["cases"][case].pop("active_pair", None)
                if invalid_reason is not None:
                    _synchronize_prefix_inventory(campaign, artifact_handle)
                    terminal = _invalid_campaign(
                        campaign, reason=invalid_reason, **current
                    )
                    write_campaign(terminal)
                    _close_ledger(
                        pathlib.Path(args.ledger), ledger, args, None, "INCONCLUSIVE", atomic_writer
                    )
                    return 2
                campaign["cases"][case]["pairs"][str(pair)] = {
                    "order": order,
                    "validity_path": validity_relative,
                    "validity_sha256": validity_record["sha256"],
                }
                os.close(pair_descriptor)
                pair_descriptor = None
                write_campaign(campaign)

        terminal = copy.deepcopy(campaign)
        terminal["statistical_result"] = _declare_results(terminal, artifact_handle)
        terminal["validity_state"] = "COMPLETE"
        terminal["completed_at"] = utc_now()
        classified = classification.classify_terminal_view(
            manifest_path,
            terminal,
            artifact_root,
            root_descriptor=artifact_handle.descriptor,
        )
        terminal["classification_artifacts"] = classified["output_artifacts"]
        for record in classified["output_artifacts"].values():
            terminal["artifact_inventory"][record["path"]] = {
                "sha256": record["sha256"]
            }
        verified = classification.classify_terminal_view(
            manifest_path,
            terminal,
            artifact_root,
            root_descriptor=artifact_handle.descriptor,
        )
        if verified["statistical_result"] != terminal["statistical_result"]:
            raise protocol.ProtocolError("classifier result differs after registration")
        protocol.atomic_write_json_at(
            artifact_handle.descriptor, FINALIZATION_STAGE, terminal
        )
        marker = _finalization_marker(terminal, args)
        try:
            protocol.atomic_write_json_at(
                artifact_handle.descriptor, FINALIZATION_MARKER, marker
            )
        except BaseException as error:
            marker_probe = _probe_committed_marker(artifact_handle, marker)
            if (
                isinstance(error, protocol.AtomicWriteError) and error.committed
            ) or marker_probe in {
                MARKER_EXACT,
                MARKER_MISMATCH,
                MARKER_UNKNOWN_IO,
            }:
                finalization_started = True
                raise
            try:
                os.unlink(FINALIZATION_STAGE, dir_fd=artifact_handle.descriptor)
                os.fsync(artifact_handle.descriptor)
            except BaseException:
                pass
            raise
        finalization_started = True
        if finalization_observer is not None:
            finalization_observer("prepared")
        _close_ledger(
            pathlib.Path(args.ledger),
            ledger,
            args,
            terminal["statistical_result"],
            "COMPLETE",
            atomic_writer,
        )
        ledger_closed = True
        if finalization_observer is not None:
            finalization_observer("ledger_closed")
        if campaign_write_observer is not None:
            campaign_write_observer(manifest_path, terminal)
        _publish_staged_campaign(artifact_handle)
        if finalization_observer is not None:
            finalization_observer("published")
        os.fsync(artifact_handle.descriptor)
        if finalization_observer is not None:
            finalization_observer("directory_synced")
        artifact_handle.validate_link()
        closed_ledger = _read_json(pathlib.Path(args.ledger), "evidence ledger")
        protocol.validate_ledger(closed_ledger)
        files, _directories = artifact_handle.inventory()
        _validate_completed_campaign(
            artifact_handle,
            args,
            closed_ledger,
            marker=marker,
            files=files,
        )
        _unlink_finalization_file(artifact_handle, FINALIZATION_MARKER)
        _unlink_finalization_file(artifact_handle, FINALIZATION_PUBLISH)
        _unlink_finalization_file(artifact_handle, FINALIZATION_STAGE)
        artifact_handle.validate_link()
        campaign = terminal
        return EXIT_BY_RESULT[("COMPLETE", terminal["statistical_result"])]
    except BaseException as error:
        if finalization_started:
            if not isinstance(error, Exception):
                raise error
            return 1
        if ledger_closed:
            if not isinstance(error, Exception):
                raise error
            return 1
        if manifest_path is None or campaign is None or ledger is None:
            if not isinstance(error, Exception):
                raise error
            return 1
        try:
            _synchronize_prefix_inventory(campaign, artifact_handle)
        except BaseException as inventory_error:
            if isinstance(error, Exception):
                error = protocol.ProtocolError(
                    f"{type(error).__name__}: {error}; "
                    "prefix-inventory-error: "
                    f"{type(inventory_error).__name__}: {inventory_error}"
                )
        terminal = _invalid_campaign(
            campaign,
            case=current["case"],
            pair=current["pair"],
            role=current["role"],
            reason=f"{type(error).__name__}: {error}",
        )
        try:
            write_campaign(terminal)
            _close_ledger(
                pathlib.Path(args.ledger), ledger, args, None, "INCONCLUSIVE", atomic_writer
            )
        except BaseException:
            if not isinstance(error, Exception):
                raise error
            return 1
        if not isinstance(error, Exception):
            raise
        return 2
    finally:
        primary = sys.exc_info()[1]
        close_failure: BaseException | None = None
        if pair_descriptor is not None:
            try:
                os.close(pair_descriptor)
            except BaseException as error:
                close_failure = error
        for executable in pinned_executables.values():
            try:
                executable.close()
            except BaseException as error:
                if close_failure is None:
                    close_failure = error
        for root_handle in (criterion_handle, artifact_handle):
            if root_handle is None:
                continue
            try:
                root_handle.close()
            except BaseException as error:
                if close_failure is None:
                    close_failure = error
        if primary is None and close_failure is not None:
            raise close_failure


def run_campaign(args, **runtime_seams) -> int:
    """Authoritatively revalidate persisted builds, then run one campaign."""
    try:
        config = build.BuildConfig(
            repository=pathlib.Path(args.repository),
            evidence_root=pathlib.Path(args.build_evidence_root),
            scratch_root=pathlib.Path(args.build_scratch_root),
            candidate_commit=args.candidate_commit,
            path=args.controlled_path,
            home=pathlib.Path(args.controlled_home),
            cargo_home=pathlib.Path(args.controlled_cargo_home),
        )
        validated = build.validate_build_set(config)
        authoritative_paths = {
            role: config.evidence_root / relative
            for role, relative in build.BUILD_MANIFEST_PATHS.items()
        }
    except Exception:
        return 1
    return _run_campaign(
        args,
        validated_builds=validated,
        authoritative_manifest_paths=authoritative_paths,
        **runtime_seams,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-kind", choices=protocol.LANE_NAMES, required=True
    )
    parser.add_argument("--baseline-build-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-build-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--repository", required=True, type=pathlib.Path)
    parser.add_argument("--build-evidence-root", required=True, type=pathlib.Path)
    parser.add_argument("--build-scratch-root", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--controlled-path", required=True)
    parser.add_argument("--controlled-home", required=True, type=pathlib.Path)
    parser.add_argument("--controlled-cargo-home", required=True, type=pathlib.Path)
    parser.add_argument("--ledger", required=True, type=pathlib.Path)
    parser.add_argument("--attempt-id", required=True, type=int)
    parser.add_argument("--artifact-root", required=True, type=pathlib.Path)
    parser.add_argument("--criterion-root", required=True, type=pathlib.Path)
    parser.add_argument(
        "--working-directory", type=pathlib.Path, default=pathlib.Path.cwd()
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--normalized-load-limit", type=float, default=0.25)
    return parser


def parse_args(argv=None):
    return build_argument_parser().parse_args(argv)


def main(argv=None) -> int:
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
