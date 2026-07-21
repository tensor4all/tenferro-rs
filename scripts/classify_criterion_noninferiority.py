#!/usr/bin/env python3
"""Validate and classify a terminal protocol-v2 Criterion campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    from scripts import phase2e_build as build
    from scripts import phase2e_protocol as protocol
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from scripts import phase2e_build as build
    from scripts import phase2e_protocol as protocol


THRESHOLD = 0.05
MONITOR_CADENCE_SECONDS = 1.0
MONITOR_JITTER_SECONDS = 0.25
PROTOCOL_VERSION = protocol.PROTOCOL_VERSION
PAIR_ORDERS = protocol.PAIR_ORDERS
RUN_ROLES = protocol.RUN_ROLES
THREAD_ENVIRONMENT = dict(protocol.THREAD_ENV)
CRITERION_SETTINGS = {
    "warm_up_seconds": 2,
    "measurement_seconds": 5,
    "sample_size": 100,
    "confidence_level": 0.95,
}
CANONICAL_CASES = protocol.CANONICAL_CASES
STATISTICAL_RESULTS = frozenset(("PASS", "FAIL", "INCONCLUSIVE"))
CLASSIFICATION_FILENAMES = ("classification.json", "summary.md")
RUNNER_FINALIZATION_FILES = {
    ".campaign-final.json",
    ".campaign-finalization.json",
    ".campaign-publish.json",
}
TRANSACTION_DIRECTORY = ".classification-transaction"
TRANSACTION_FILENAMES = (
    "staged-classification.json",
    "staged-summary.md",
    "commit.json",
    "pending-staged-classification.json",
    "pending-staged-summary.md",
    "pending-commit.json",
)

CAMPAIGN_FIELDS = {
    "protocol_version": int,
    "protocol_sha256": str,
    "classifier_sha256": str,
    "candidate_sha": str,
    "comparison_kind": str,
    "build_manifests": dict,
    "selected_cpu": int,
    "allowed_cpus": str,
    "allowed_cpu_count": int,
    "normalized_load_limit": (int, float),
    "thread_environment": dict,
    "orders": list,
    "criterion": dict,
    "criterion_binding": dict,
    "validity_state": str,
    "statistical_result": (str, type(None)),
    "completed_at": str,
    "cases": dict,
    "artifact_inventory": dict,
    "classification_artifacts": (dict, type(None)),
}
BUILD_RECORD_FIELDS = {
    "path": str,
    "sha256": str,
    "role": str,
    "executable_sha256": str,
    "executable_path": str,
    "executable_device": int,
    "executable_inode": int,
    "snapshot_sha256": str,
    "snapshot_device": int,
    "snapshot_inode": int,
}
CASE_FIELDS = {
    "benchmark": str,
    "statistical_result": str,
    "pairs": dict,
}
PAIR_FIELDS = {
    "order": str,
    "validity_path": str,
    "validity_sha256": str,
}
VALIDITY_FIELDS = {
    "protocol_version": int,
    "case": str,
    "pair": int,
    "order": str,
    "selected_cpu": int,
    "allowed_cpu_count": int,
    "validity_state": str,
    "runs": list,
    "artifacts": dict,
}
RUN_FIELDS = {
    "role": str,
    "binary": str,
    "binary_sha256": str,
    "validity_state": str,
    "exit_status": int,
    "stdout_artifact": str,
    "stderr_artifact": str,
    "process_started_monotonic": (int, float),
    "process_ended_monotonic": (int, float),
    "monitor_samples": list,
    "argv": list,
    "environment": dict,
    "environment_sha256": str,
    "executable": dict,
    "criterion_binding": dict,
    "process_group_cleanup": dict,
}
EXECUTABLE_FIELDS = {
    "logical_path": str,
    "source_device": int,
    "source_inode": int,
    "snapshot_device": int,
    "snapshot_inode": int,
    "snapshot_sha256": str,
    "launch_path": str,
}
CRITERION_BINDING_FIELDS = {
    "logical_path": str,
    "actual_home": str,
    "device": int,
    "inode": int,
}
PROCESS_GROUP_CLEANUP_FIELDS = {
    "survivor_observed": bool,
    "term_signal_sent": bool,
    "kill_signal_sent": bool,
    "failures": list,
}
MONITOR_SAMPLE_FIELDS = {
    "sequence": int,
    "phase": str,
    "monotonic_seconds": (int, float),
    "observed_affinity": str,
    "normalized_load": (int, float),
    "cargo_processes": list,
    "rustc_processes": list,
}
MONITOR_ARTIFACT_FIELDS = {
    "protocol_version": int,
    "case": str,
    "pair": int,
    "runs": dict,
}
ARTIFACT_RECORD_FIELDS = {"sha256": str}
OUTPUT_ARTIFACT_FIELDS = {"path": str, "sha256": str}
MAX_ALLOWED_CPU_COUNT = 65_536


def _parse_cpu_inventory_intervals(value: str) -> tuple[tuple[tuple[int, int], ...], int]:
    if not isinstance(value, str) or not value:
        raise protocol.ProtocolError("campaign.json: invalid allowed CPU inventory")
    intervals: list[tuple[int, int]] = []
    cardinality = 0
    previous_last = -2
    try:
        for component in value.split(","):
            match = re.fullmatch(r"(0|[1-9][0-9]*)(?:-(0|[1-9][0-9]*))?", component)
            if match is None:
                raise ValueError
            first = int(match.group(1))
            last = first if match.group(2) is None else int(match.group(2))
            if (
                (match.group(2) is not None and last <= first)
                or first <= previous_last + 1
            ):
                raise ValueError
            width = last - first + 1
            cardinality += width
            if cardinality > MAX_ALLOWED_CPU_COUNT:
                raise ValueError
            intervals.append((first, last))
            previous_last = last
    except (ValueError, ArithmeticError) as error:
        raise protocol.ProtocolError(
            "campaign.json: invalid allowed CPU inventory"
        ) from error
    return tuple(intervals), cardinality


def parse_cpu_inventory(value: str) -> set[int]:
    """Parse a bounded canonical comma/range CPU inventory."""
    intervals, _cardinality = _parse_cpu_inventory_intervals(value)
    cpus: set[int] = set()
    for first, last in intervals:
        cpus.update(range(first, last + 1))
    return cpus


def invert_interval(lower: float, upper: float, point: float) -> tuple[float, ...]:
    """Invert a B/A relative-change estimate to A/B orientation."""
    lower, upper, point = (
        _finite_real(value, "relative-change value")
        for value in (lower, upper, point)
    )
    if lower <= -1.0 or upper <= -1.0 or point <= -1.0:
        raise protocol.ProtocolError(
            "relative-change values must be greater than -1"
        )
    return (
        1.0 / (1.0 + upper) - 1.0,
        1.0 / (1.0 + lower) - 1.0,
        1.0 / (1.0 + point) - 1.0,
    )


def classify_case(intervals: Sequence[Mapping[str, float]]) -> str:
    """Apply the frozen three-interval non-inferiority rule."""
    if len(intervals) != 3:
        raise ValueError("classification requires exactly three intervals")
    if all(interval["upper"] <= 0.05 for interval in intervals):
        return "PASS"
    if sum(interval["lower"] > 0.05 for interval in intervals) >= 2:
        return "FAIL"
    return "INCONCLUSIVE"


def campaign_result(results: Mapping[str, str]) -> str:
    """Reduce case classifications using FAIL-before-INCONCLUSIVE precedence."""
    if any(value == "FAIL" for value in results.values()):
        return "FAIL"
    if any(value == "INCONCLUSIVE" for value in results.values()):
        return "INCONCLUSIVE"
    return "PASS"


def sentinel_breached(
    lower: float, upper: float, threshold: float = THRESHOLD
) -> bool:
    """Return whether an A/A interval lies wholly outside the drift band."""
    return lower > threshold or upper < -threshold


def _read_json(path: pathlib.Path, context: str) -> Any:
    content, _digest = _read_regular_snapshot(path)
    return _decode_json(content, context)


def _decode_json(content: bytes, context: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise protocol.ProtocolError(
                    f"{context} contains duplicate JSON key: {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            content.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise protocol.ProtocolError(f"{context} is not valid JSON: {error}") from error


def _read_regular_snapshot(path: pathlib.Path) -> tuple[bytes, str]:
    """Read one absolute regular path through held no-follow directory fds."""
    absolute = pathlib.Path(os.path.abspath(path))
    parts = absolute.parts
    retained = _retained_proc_components(absolute)
    descriptors: list[int] = []
    directory_links: list[tuple[int, str, tuple[int, int]]] = []
    file_identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    try:
        if retained is None:
            parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            components = parts[1:-1]
            leaf = parts[-1]
        else:
            root_descriptor, relative_parts = retained
            if not relative_parts:
                raise protocol.ProtocolError(
                    f"evidence path is not a regular file: {absolute}"
                )
            parent = os.dup(root_descriptor)
            components = relative_parts[:-1]
            leaf = relative_parts[-1]
        descriptors.append(parent)
        for component in components:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            directory_links.append(
                (parent, component, _directory_identity(os.fstat(child)))
            )
            parent = child
            descriptors.append(child)
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise protocol.ProtocolError(f"evidence path is not regular: {absolute}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if file_identity(before) != file_identity(after) or file_identity(
            after
        ) != file_identity(current):
            raise protocol.ProtocolError(f"evidence path changed while snapshotting: {absolute}")
        for ancestor, component, expected in directory_links:
            current_directory = os.stat(
                component, dir_fd=ancestor, follow_symlinks=False
            )
            if _directory_identity(current_directory) != expected:
                raise protocol.ProtocolError(
                    f"evidence parent changed while snapshotting: {absolute}"
                )
        content = b"".join(chunks)
        return content, hashlib.sha256(content).hexdigest()
    except protocol.ProtocolError:
        raise
    except OSError as error:
        raise protocol.ProtocolError(f"cannot snapshot evidence path {absolute}: {error}") from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _retained_proc_components(
    path: pathlib.Path,
) -> tuple[int, tuple[str, ...]] | None:
    absolute = pathlib.Path(os.path.abspath(path))
    parts = absolute.parts
    if len(parts) < 5 or parts[:4] != ("/", "proc", "self", "fd"):
        return None
    try:
        descriptor = int(parts[4])
        opened = os.fstat(descriptor)
    except (ValueError, OSError) as error:
        raise protocol.ProtocolError(f"invalid retained root path: {absolute}") from error
    if not stat.S_ISDIR(opened.st_mode):
        raise protocol.ProtocolError(f"retained root is not a directory: {absolute}")
    return descriptor, tuple(parts[5:])


def _open_retained_directory_path(path: pathlib.Path, *, create: bool = False) -> int:
    retained = _retained_proc_components(path)
    if retained is None:
        return os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    root_descriptor, components = retained
    descriptor = os.dup(root_descriptor)
    try:
        for component in components:
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
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        raise


def read_json(path: pathlib.Path) -> Any:
    """Read one JSON artifact with the evidence-mode parser."""
    return _read_json(pathlib.Path(path), str(path))


def read_change(path: pathlib.Path) -> tuple[float, float, float]:
    """Read and validate the mean interval from copied Criterion evidence."""
    content, _digest = _read_regular_snapshot(pathlib.Path(path))
    return _read_change_content(content, str(path))


def _read_change_content(content: bytes, context: str) -> tuple[float, float, float]:
    payload = _decode_json(content, context)
    if type(payload) is not dict or type(payload.get("mean")) is not dict:
        raise protocol.ProtocolError(f"{context}: missing mean estimate")
    estimate = payload["mean"]
    confidence = estimate.get("confidence_interval")
    if type(confidence) is not dict:
        raise protocol.ProtocolError(f"{context}: missing confidence interval")
    values = (
        confidence.get("lower_bound"),
        confidence.get("upper_bound"),
        estimate.get("point_estimate"),
    )
    lower, upper, point = (
        _finite_real(value, f"{context}: estimate value") for value in values
    )
    if lower > point or point > upper:
        raise protocol.ProtocolError(
            f"{context}: point estimate is outside its confidence interval"
        )
    return lower, upper, point


def _finite_real(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise protocol.ProtocolError(f"{label} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise protocol.ProtocolError(f"{label} must be a finite real number") from error
    if not math.isfinite(converted):
        raise protocol.ProtocolError(f"{label} must be a finite real number")
    return converted


def file_sha256(path: pathlib.Path) -> str:
    """Hash one regular artifact without following its final symlink."""
    return protocol.sha256_file(pathlib.Path(path))


def require_sha256(value: Any, label: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise protocol.ProtocolError(f"{label} must be a lowercase SHA-256")


def _require_commit(value: Any, label: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise protocol.ProtocolError(
            f"{label} must be a 40-character lowercase hexadecimal commit"
        )


def require_file(path: pathlib.Path) -> None:
    path = pathlib.Path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing campaign artifact: {path}") from None
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect campaign artifact {path}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise protocol.ProtocolError(
            f"campaign artifact is not a regular file: {path}"
        )


def _canonical_regular_file(path: pathlib.Path, context: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.is_absolute():
        raise protocol.ProtocolError(f"{context} path must be absolute")
    require_file(path)
    if _retained_proc_components(path) is not None:
        return path
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve {context} path: {error}") from error
    if canonical != path:
        raise protocol.ProtocolError(f"{context} path must be canonical")
    return path


def _relative_artifact_path(root: pathlib.Path, value: str, context: str) -> pathlib.Path:
    if type(value) is not str or not value:
        raise protocol.ProtocolError(f"{context} path must be a nonempty string")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise protocol.ProtocolError(f"{context} path escapes the campaign root")
    if relative.as_posix() != value:
        raise protocol.ProtocolError(f"{context} path is not canonical POSIX text")
    path = root.joinpath(*relative.parts)
    require_file(path)
    if _retained_proc_components(root) is not None:
        return path
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve {context} path: {error}") from error
    if canonical != path:
        raise protocol.ProtocolError(f"{context} path is not canonical")
    return path


def _artifact_record(record: Any, context: str) -> str:
    if type(record) is dict and "sha256" not in record:
        raise protocol.ProtocolError(f"{context}: missing SHA-256")
    protocol.validate_manifest_fields(
        record, ARTIFACT_RECORD_FIELDS, context=context
    )
    digest = record["sha256"]
    require_sha256(digest, f"{context} SHA-256")
    return digest


def _validate_build_manifests(
    campaign: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    records = campaign["build_manifests"]
    if set(records) != {"baseline", "candidate"}:
        raise protocol.ProtocolError(
            "campaign.json: build manifest inventory must contain baseline and candidate"
        )
    expected_baseline_role = {
        "direct-current-main": "direct-current-main-baseline",
        "common-lock-normalized": "common-lock-normalized-baseline",
    }.get(campaign["comparison_kind"])
    if expected_baseline_role is None:
        raise protocol.ProtocolError("campaign.json: invalid comparison kind")
    expected_roles = {
        "baseline": expected_baseline_role,
        "candidate": "candidate",
    }
    executable_shas: dict[str, str] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for identity in ("baseline", "candidate"):
        record = records[identity]
        protocol.validate_manifest_fields(
            record, BUILD_RECORD_FIELDS, context=f"{identity} build record"
        )
        require_sha256(record["sha256"], f"{identity} build manifest SHA-256")
        require_sha256(
            record["executable_sha256"], f"{identity} executable SHA-256"
        )
        require_sha256(record["snapshot_sha256"], f"{identity} snapshot SHA-256")
        if record["role"] != expected_roles[identity]:
            raise protocol.ProtocolError(f"{identity} build role is inconsistent")
        path = _canonical_regular_file(
            pathlib.Path(record["path"]), f"{identity} build manifest"
        )
        manifest_content, manifest_digest = _read_regular_snapshot(path)
        if manifest_digest != record["sha256"]:
            raise protocol.ProtocolError(f"{identity} build manifest digest mismatch")
        manifest = _decode_json(manifest_content, f"{identity} build manifest")
        if type(manifest) is not dict:
            raise protocol.ProtocolError(f"{identity} build manifest is not an object")
        expected_identity = {
            "protocol_version": PROTOCOL_VERSION,
            "role": record["role"],
            "executable_sha256": record["executable_sha256"],
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                raise protocol.ProtocolError(
                    f"{identity} build manifest identity differs: {field}"
                )
        if identity == "candidate" and manifest.get("head") != campaign["candidate_sha"]:
            raise protocol.ProtocolError("candidate build commit identity differs")
        build.validate_build_manifest(manifest)
        executable_metadata = os.stat(
            manifest["executable"], follow_symlinks=False
        )
        expected_executable = {
            "executable_path": manifest["executable"],
            "executable_device": executable_metadata.st_dev,
            "executable_inode": executable_metadata.st_ino,
            "snapshot_sha256": manifest["executable_sha256"],
        }
        for field, expected_value in expected_executable.items():
            if record[field] != expected_value:
                raise protocol.ProtocolError(
                    f"{identity} executable record differs: {field}"
                )
        if _read_regular_snapshot(pathlib.Path(manifest["executable"]))[1] != manifest[
            "executable_sha256"
        ]:
            raise protocol.ProtocolError(f"{identity} executable digest mismatch")
        manifests[identity] = manifest
        executable_shas[identity] = record["executable_sha256"]
    build.validate_pair(
        campaign["comparison_kind"], manifests["baseline"], manifests["candidate"]
    )
    return executable_shas, manifests


def _validate_monitor_samples(
    samples: Any,
    *,
    selected_cpu: int,
    load_limit: float,
    process_started: float,
    process_ended: float,
    context: str,
) -> None:
    if type(samples) is not list or len(samples) < 2:
        raise protocol.ProtocolError(
            f"{context}: monitor samples require start and end observations"
        )
    previous_timestamp: float | None = None
    for sequence, sample in enumerate(samples):
        protocol.validate_manifest_fields(
            sample, MONITOR_SAMPLE_FIELDS, context=f"{context} monitor sample"
        )
        if sample["sequence"] != sequence:
            raise protocol.ProtocolError(
                f"{context}: monitor sample sequence is not complete"
            )
        phase = sample["phase"]
        if phase not in {"start", "periodic", "end"}:
            raise protocol.ProtocolError(f"{context}: invalid monitor sample phase")
        if sequence == 0 and phase != "start":
            raise protocol.ProtocolError(
                f"{context}: monitor samples require start and end observations"
            )
        if sequence == len(samples) - 1 and phase != "end":
            raise protocol.ProtocolError(
                f"{context}: monitor samples require start and end observations"
            )
        if 0 < sequence < len(samples) - 1 and phase != "periodic":
            raise protocol.ProtocolError(
                f"{context}: interior monitor samples must be periodic"
            )
        if sample["observed_affinity"] != str(selected_cpu):
            raise protocol.ProtocolError(f"{context}: benchmark affinity mismatch")
        timestamp = _finite_real(
            sample["monotonic_seconds"], f"{context}: monitor timestamp"
        )
        if previous_timestamp is not None:
            gap = timestamp - previous_timestamp
            if gap <= 0.0 or gap > MONITOR_CADENCE_SECONDS + MONITOR_JITTER_SECONDS:
                raise protocol.ProtocolError(
                    f"{context}: monitor cadence is incomplete"
                )
        previous_timestamp = timestamp
        load = _finite_real(
            sample["normalized_load"],
            f"{context}: monitor sample normalized load",
        )
        if load < 0.0 or load > load_limit:
            raise protocol.ProtocolError(
                f"{context}: monitor sample normalized load exceeds limit"
            )
        if sample["cargo_processes"] or sample["rustc_processes"]:
            raise protocol.ProtocolError(
                f"{context}: monitor sample contains an overlapping build process"
            )
    first = _finite_real(samples[0]["monotonic_seconds"], f"{context}: monitor start")
    last = _finite_real(samples[-1]["monotonic_seconds"], f"{context}: monitor end")
    if (
        abs(first - process_started) > MONITOR_JITTER_SECONDS
        or abs(last - process_ended) > MONITOR_JITTER_SECONDS
    ):
        raise protocol.ProtocolError(f"{context}: monitor endpoint coverage is incomplete")


def _validate_run(
    run: Any,
    *,
    role: str,
    binary: str,
    binary_sha256: str,
    selected_cpu: int,
    load_limit: float,
    context: str,
    expected_argv_tail: list[str],
    build_manifest: Mapping[str, Any],
    build_record: Mapping[str, Any],
    criterion_binding: Mapping[str, Any],
    candidate_environment: Mapping[str, str],
) -> None:
    protocol.validate_manifest_fields(run, RUN_FIELDS, context=f"{context} {role}")
    expected = {
        "role": role,
        "binary": binary,
        "binary_sha256": binary_sha256,
        "validity_state": "COMPLETE",
        "exit_status": 0,
        "stdout_artifact": f"{role}.stdout.log",
        "stderr_artifact": f"{role}.stderr.log",
    }
    for field, value in expected.items():
        if run[field] != value:
            raise protocol.ProtocolError(
                f"{context} {role}: run identity differs: {field}"
            )
    protocol.validate_manifest_fields(
        run["executable"], EXECUTABLE_FIELDS, context=f"{context} {role} executable"
    )
    executable = run["executable"]
    expected_executable = {
        "logical_path": build_manifest["executable"],
        "source_device": build_record["executable_device"],
        "source_inode": build_record["executable_inode"],
        "snapshot_sha256": binary_sha256,
        "snapshot_device": build_record["snapshot_device"],
        "snapshot_inode": build_record["snapshot_inode"],
    }
    for field, value in expected_executable.items():
        if executable[field] != value:
            raise protocol.ProtocolError(
                f"{context} {role}: executable identity differs: {field}"
            )
    require_sha256(executable["snapshot_sha256"], f"{context} {role} snapshot")
    if (
        type(executable["snapshot_device"]) is not int
        or type(executable["snapshot_inode"]) is not int
        or executable["snapshot_device"] < 0
        or executable["snapshot_inode"] <= 0
        or re.fullmatch(r"/proc/self/fd/[0-9]+", executable["launch_path"])
        is None
    ):
        raise protocol.ProtocolError(f"{context} {role}: invalid executable snapshot")
    expected_argv = [executable["launch_path"], *expected_argv_tail]
    if run["argv"] != expected_argv:
        raise protocol.ProtocolError(f"{context} {role}: argv differs")
    protocol.validate_manifest_fields(
        run["criterion_binding"],
        CRITERION_BINDING_FIELDS,
        context=f"{context} {role} Criterion binding",
    )
    if run["criterion_binding"] != criterion_binding:
        raise protocol.ProtocolError(f"{context} {role}: Criterion binding differs")
    expected_environment = protocol.runtime_environment(
        path=candidate_environment["PATH"],
        home=candidate_environment["HOME"],
        criterion_home=criterion_binding["actual_home"],
    )
    if run["environment"] != expected_environment:
        raise protocol.ProtocolError(f"{context} {role}: environment differs")
    require_sha256(run["environment_sha256"], f"{context} {role} environment")
    if run["environment_sha256"] != protocol.sha256_json(expected_environment):
        raise protocol.ProtocolError(
            f"{context} {role}: environment digest differs"
        )
    protocol.validate_manifest_fields(
        run["process_group_cleanup"],
        PROCESS_GROUP_CLEANUP_FIELDS,
        context=f"{context} {role} process group cleanup",
    )
    if run["process_group_cleanup"] != {
        "survivor_observed": False,
        "term_signal_sent": False,
        "kill_signal_sent": False,
        "failures": [],
    }:
        raise protocol.ProtocolError(
            f"{context} {role}: process group cleanup was required"
        )
    process_started = _finite_real(
        run["process_started_monotonic"], f"{context} {role}: process start"
    )
    process_ended = _finite_real(
        run["process_ended_monotonic"], f"{context} {role}: process end"
    )
    if process_ended < process_started:
        raise protocol.ProtocolError(f"{context} {role}: process interval is reversed")
    _validate_monitor_samples(
        run["monitor_samples"],
        selected_cpu=selected_cpu,
        load_limit=load_limit,
        process_started=process_started,
        process_ended=process_ended,
        context=f"{context} {role}",
    )


def _validate_pair(
    root: pathlib.Path,
    case: str,
    pair: int,
    order: str,
    entry: Any,
    *,
    selected_cpu: int,
    allowed_count: int,
    load_limit: float,
    binary_shas: Mapping[str, str],
    build_manifests: Mapping[str, Mapping[str, Any]],
    build_records: Mapping[str, Mapping[str, Any]],
    criterion_binding: Mapping[str, Any],
    candidate_environment: Mapping[str, str],
    inventory: Mapping[str, Any],
    snapshots: Mapping[str, bytes],
) -> tuple[dict[str, float | int | str], set[str]]:
    context = f"{case}/pair{pair}"
    protocol.validate_manifest_fields(entry, PAIR_FIELDS, context=context)
    if entry["order"] != order:
        raise protocol.ProtocolError(f"{context}: campaign pair order differs")
    expected_relative = f"{case}/pair{pair}/validity.json"
    if entry["validity_path"] != expected_relative:
        raise protocol.ProtocolError(f"{context}: validity path is inconsistent")
    require_sha256(entry["validity_sha256"], f"{context} validity SHA-256")
    validity_path = _relative_artifact_path(root, expected_relative, context)
    if hashlib.sha256(snapshots[expected_relative]).hexdigest() != entry["validity_sha256"]:
        raise protocol.ProtocolError(f"{context}: validity digest mismatch")

    validity = _decode_json(snapshots[expected_relative], f"{context} validity")
    protocol.validate_manifest_fields(
        validity, VALIDITY_FIELDS, context=f"{context} validity"
    )
    expected_validity = {
        "protocol_version": PROTOCOL_VERSION,
        "case": case,
        "pair": pair,
        "order": order,
        "selected_cpu": selected_cpu,
        "allowed_cpu_count": allowed_count,
        "validity_state": "COMPLETE",
    }
    for field, expected in expected_validity.items():
        if validity[field] != expected:
            raise protocol.ProtocolError(
                f"{context}: validity identity differs: {field}"
            )

    expected_local_names = {
        "change-estimates.json",
        "sentinel-change-estimates.json",
        "monitor-samples.json",
        *(
            f"{role}.{stream}.log"
            for role in RUN_ROLES
            for stream in ("stdout", "stderr")
        ),
    }
    local_artifacts = validity["artifacts"]
    if set(local_artifacts) != expected_local_names:
        raise protocol.ProtocolError(f"{context}: local artifact inventory differs")
    pair_paths: set[str] = {expected_relative}
    pair_directory = validity_path.parent
    for name in sorted(expected_local_names):
        digest = _artifact_record(
            local_artifacts[name], f"{context} artifact {name}"
        )
        path = _relative_artifact_path(root, f"{case}/pair{pair}/{name}", context)
        relative = path.relative_to(root).as_posix()
        if (
            path.parent != pair_directory
            or hashlib.sha256(snapshots[relative]).hexdigest() != digest
        ):
            raise protocol.ProtocolError(f"{context}: artifact digest mismatch: {name}")
        top_record = inventory.get(relative)
        if top_record != {"sha256": digest}:
            raise protocol.ProtocolError(
                f"{context}: campaign artifact inventory differs: {name}"
            )
        pair_paths.add(relative)
    if inventory.get(expected_relative) != {"sha256": entry["validity_sha256"]}:
        raise protocol.ProtocolError(
            f"{context}: campaign validity artifact inventory differs"
        )

    runs = validity["runs"]
    if type(runs) is not list or len(runs) != 4:
        raise protocol.ProtocolError(f"{context}: validity requires exactly four runs")
    target_identities = (
        ("baseline", "candidate") if order == "A/B" else ("candidate", "baseline")
    )
    run_identities = ("candidate", *target_identities, "candidate")
    target_name = f"phase2e-target-{case}-p{pair}"
    sentinel_name = f"phase2e-sentinel-{case}-p{pair}"
    argv_tails = (
        ["--bench", CANONICAL_CASES["lazy_neg_1"], "--save-baseline", sentinel_name, "--noplot"],
        ["--bench", CANONICAL_CASES[case], "--save-baseline", target_name, "--noplot"],
        ["--bench", CANONICAL_CASES[case], "--baseline", target_name, "--noplot"],
        ["--bench", CANONICAL_CASES["lazy_neg_1"], "--baseline", sentinel_name, "--noplot"],
    )
    for run, role, binary, argv_tail in zip(
        runs, RUN_ROLES, run_identities, argv_tails
    ):
        _validate_run(
            run,
            role=role,
            binary=binary,
            binary_sha256=binary_shas[binary],
            selected_cpu=selected_cpu,
            load_limit=load_limit,
            context=context,
            expected_argv_tail=argv_tail,
            build_manifest=build_manifests[binary],
            build_record=build_records[binary],
            criterion_binding=criterion_binding,
            candidate_environment=candidate_environment,
        )

    monitor_path = pair_directory / "monitor-samples.json"
    monitor_relative = monitor_path.relative_to(root).as_posix()
    monitor = _decode_json(
        snapshots[monitor_relative], f"{context} monitor artifact"
    )
    protocol.validate_manifest_fields(
        monitor, MONITOR_ARTIFACT_FIELDS, context=f"{context} monitor artifact"
    )
    if (
        monitor["protocol_version"] != PROTOCOL_VERSION
        or monitor["case"] != case
        or monitor["pair"] != pair
        or set(monitor["runs"]) != set(RUN_ROLES)
    ):
        raise protocol.ProtocolError(f"{context}: monitor artifact identity differs")
    for run in runs:
        if monitor["runs"].get(run["role"]) != run["monitor_samples"]:
            raise protocol.ProtocolError(
                f"{context}: monitor artifact does not match complete run samples"
            )

    sentinel_path = pair_directory / "sentinel-change-estimates.json"
    sentinel = _read_change_content(
        snapshots[sentinel_path.relative_to(root).as_posix()], str(sentinel_path)
    )
    if sentinel_breached(sentinel[0], sentinel[1]):
        raise protocol.ProtocolError(f"{context}: sentinel interval breaches drift band")
    estimate_path = pair_directory / "change-estimates.json"
    estimate = _read_change_content(
        snapshots[estimate_path.relative_to(root).as_posix()], str(estimate_path)
    )
    oriented = invert_interval(*estimate) if order == "B/A" else estimate
    return (
        {
            "pair": pair,
            "source_order": order,
            "order": "A/B",
            "lower": oriented[0],
            "upper": oriented[1],
            "point": oriented[2],
        },
        pair_paths,
    )


def _validate_artifact_inventory(
    root: pathlib.Path, inventory: Any
) -> tuple[dict[str, dict[str, str]], dict[str, bytes]]:
    if type(inventory) is not dict:
        raise protocol.ProtocolError("campaign.json: artifact inventory must be an object")
    validated: dict[str, dict[str, str]] = {}
    snapshots: dict[str, bytes] = {}
    for relative, record in inventory.items():
        if type(relative) is not str:
            raise protocol.ProtocolError("campaign.json: artifact inventory path is invalid")
        digest = _artifact_record(record, f"artifact inventory {relative}")
        path = _relative_artifact_path(root, relative, f"artifact inventory {relative}")
        content, observed_digest = _read_regular_snapshot(path)
        if observed_digest != digest:
            raise protocol.ProtocolError(
                f"campaign.json: artifact inventory digest mismatch: {relative}"
            )
        validated[relative] = {"sha256": digest}
        snapshots[relative] = content
    return validated, snapshots


def _classification_artifact_records(
    root: pathlib.Path, records: Any
) -> tuple[dict[str, dict[str, str]] | None, set[str], dict[str, bytes]]:
    if records is None:
        return None, set(), {}
    if type(records) is not dict or set(records) != set(CLASSIFICATION_FILENAMES):
        raise protocol.ProtocolError(
            "campaign.json: classification artifacts must contain both outputs"
        )
    validated: dict[str, dict[str, str]] = {}
    relatives: set[str] = set()
    contents: dict[str, bytes] = {}
    for name in CLASSIFICATION_FILENAMES:
        record = records[name]
        protocol.validate_manifest_fields(
            record, OUTPUT_ARTIFACT_FIELDS, context=f"classification artifact {name}"
        )
        require_sha256(record["sha256"], f"classification artifact {name} SHA-256")
        path = _relative_artifact_path(
            root, record["path"], f"classification artifact {name}"
        )
        if path.name != name or (
            root != path.parent and root not in path.parents
        ):
            raise protocol.ProtocolError(
                f"classification artifact {name} is outside the campaign root"
            )
        content, digest = _read_regular_snapshot(path)
        if digest != record["sha256"]:
            raise protocol.ProtocolError(f"classification output digest mismatch: {name}")
        validated[name] = dict(record)
        contents[name] = content
        relatives.add(path.relative_to(root).as_posix())
    parents = {pathlib.Path(record["path"]).parent for record in validated.values()}
    if len(parents) != 1:
        raise protocol.ProtocolError("classification artifacts use different output dirs")
    return validated, relatives, contents


def _observed_normative_inventory(
    root: pathlib.Path, *, ignored_files: set[str] | None = None
) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    if _retained_proc_components(root) is not None:
        root_fd = _open_retained_directory_path(root)

        def visit(directory_fd: int, prefix: str) -> None:
            for name in sorted(os.listdir(directory_fd)):
                relative = f"{prefix}/{name}" if prefix else name
                metadata = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
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
                else:
                    files.add(relative)

        try:
            visit(root_fd, "")
        finally:
            os.close(root_fd)
        files.difference_update(ignored_files or set())
        return files, directories
    try:
        paths = tuple(root.rglob("*"))
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot enumerate campaign artifact inventory: {error}"
        ) from error
    for path in paths:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise protocol.ProtocolError(f"cannot inspect campaign artifact: {path}") from error
        if stat.S_ISDIR(metadata.st_mode):
            directories.add(path.relative_to(root).as_posix())
            continue
        files.add(path.relative_to(root).as_posix())
    files.difference_update(ignored_files or set())
    return files, directories


def _validated_classification(
    campaign_path: pathlib.Path,
    recovery_output_dir: pathlib.Path | None = None,
    terminal_payload: Mapping[str, Any] | None = None,
    ignored_root_files: set[str] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pathlib.Path,
    dict[str, dict[str, str]] | None,
    dict[str, bytes],
]:
    campaign_path = pathlib.Path(campaign_path)
    if campaign_path.name != "campaign.json":
        raise protocol.ProtocolError("evidence mode requires campaign.json")
    campaign_path = _canonical_regular_file(
        pathlib.Path(os.path.abspath(campaign_path)), "campaign manifest"
    )
    root = campaign_path.parent
    persisted_campaign = _read_json(campaign_path, "campaign.json")
    if terminal_payload is None:
        campaign = persisted_campaign
    else:
        campaign = copy.deepcopy(terminal_payload)
        _validate_terminal_derivation(persisted_campaign, campaign)
    protocol.validate_manifest_fields(
        campaign, CAMPAIGN_FIELDS, context="campaign.json"
    )

    if campaign["protocol_version"] != PROTOCOL_VERSION:
        raise protocol.ProtocolError("campaign.json: unsupported protocol version")
    require_sha256(campaign["protocol_sha256"], "campaign protocol SHA-256")
    require_sha256(campaign["classifier_sha256"], "campaign classifier SHA-256")
    if campaign["protocol_sha256"] != file_sha256(pathlib.Path(protocol.__file__)):
        raise protocol.ProtocolError("campaign.json: protocol digest mismatch")
    if campaign["classifier_sha256"] != file_sha256(pathlib.Path(__file__)):
        raise protocol.ProtocolError("campaign.json: classifier digest mismatch")
    _require_commit(campaign["candidate_sha"], "campaign candidate SHA")
    if campaign["validity_state"] != "COMPLETE":
        raise protocol.ProtocolError(
            "campaign.json: validity_state must be COMPLETE before classification"
        )
    declared_campaign_result = campaign["statistical_result"]
    if declared_campaign_result not in STATISTICAL_RESULTS:
        raise protocol.ProtocolError(
            "campaign.json: terminal statistical result must be non-null"
        )
    if not campaign["completed_at"]:
        raise protocol.ProtocolError("campaign.json: completed_at must be nonempty")
    if campaign["orders"] != list(PAIR_ORDERS):
        raise protocol.ProtocolError("campaign.json: pair order differs")
    if campaign["criterion"] != CRITERION_SETTINGS:
        raise protocol.ProtocolError("campaign.json: Criterion settings differ")
    protocol.validate_manifest_fields(
        campaign["criterion_binding"],
        CRITERION_BINDING_FIELDS,
        context="campaign.json Criterion binding",
    )
    criterion_binding = campaign["criterion_binding"]
    if (
        not pathlib.Path(criterion_binding["logical_path"]).is_absolute()
        or re.fullmatch(r"/proc/self/fd/[0-9]+", criterion_binding["actual_home"])
        is None
        or type(criterion_binding["device"]) is not int
        or type(criterion_binding["inode"]) is not int
        or criterion_binding["device"] < 0
        or criterion_binding["inode"] <= 0
    ):
        raise protocol.ProtocolError("campaign.json: invalid Criterion root binding")
    if campaign["thread_environment"] != THREAD_ENVIRONMENT:
        raise protocol.ProtocolError("campaign.json: thread environment differs")

    selected_cpu = campaign["selected_cpu"]
    allowed_count = campaign["allowed_cpu_count"]
    if (
        type(selected_cpu) is not int
        or type(allowed_count) is not int
        or selected_cpu < 0
        or not 0 < allowed_count <= MAX_ALLOWED_CPU_COUNT
    ):
        raise protocol.ProtocolError("campaign.json: invalid CPU identity")
    allowed_intervals, observed_count = _parse_cpu_inventory_intervals(
        campaign["allowed_cpus"]
    )
    selected_is_allowed = any(
        first <= selected_cpu <= last for first, last in allowed_intervals
    )
    if observed_count != allowed_count or not selected_is_allowed:
        raise protocol.ProtocolError("campaign.json: allowed CPU inventory differs")
    load_limit = _finite_real(
        campaign["normalized_load_limit"], "campaign normalized load limit"
    )
    if load_limit != 0.25:
        raise protocol.ProtocolError("campaign.json: normalized load limit differs")
    binary_shas, build_manifests = _validate_build_manifests(campaign)

    inventory, snapshots = _validate_artifact_inventory(
        root, campaign["artifact_inventory"]
    )
    output_records, output_relatives, output_contents = _classification_artifact_records(
        root, campaign["classification_artifacts"]
    )
    for name, record in (output_records or {}).items():
        relative = record["path"]
        if inventory.get(relative) != {"sha256": record["sha256"]}:
            raise protocol.ProtocolError(
                f"campaign.json: classification artifact inventory differs: {name}"
            )

    case_records = campaign["cases"]
    if set(case_records) != set(CANONICAL_CASES):
        missing = sorted(set(CANONICAL_CASES) - set(case_records))
        extra = sorted(set(case_records) - set(CANONICAL_CASES))
        raise protocol.ProtocolError(
            "campaign.json: incomplete canonical case inventory; "
            f"missing={missing}, extra={extra}"
        )
    output_cases: dict[str, dict[str, Any]] = {}
    input_paths: set[str] = set()
    results: dict[str, str] = {}
    for case in sorted(CANONICAL_CASES):
        record = case_records[case]
        protocol.validate_manifest_fields(record, CASE_FIELDS, context=f"case {case}")
        if record["benchmark"] != CANONICAL_CASES[case]:
            raise protocol.ProtocolError(f"case {case}: benchmark identity differs")
        if record["statistical_result"] not in STATISTICAL_RESULTS:
            raise protocol.ProtocolError(f"case {case}: invalid statistical result")
        pair_records = record["pairs"]
        if set(pair_records) != {"1", "2", "3"}:
            raise protocol.ProtocolError(f"case {case}: incomplete pair inventory")
        intervals: list[dict[str, float | int | str]] = []
        for pair, order in enumerate(PAIR_ORDERS, start=1):
            interval, paths = _validate_pair(
                root,
                case,
                pair,
                order,
                pair_records[str(pair)],
                selected_cpu=selected_cpu,
                allowed_count=allowed_count,
                load_limit=load_limit,
                binary_shas=binary_shas,
                build_manifests=build_manifests,
                build_records=campaign["build_manifests"],
                criterion_binding=campaign["criterion_binding"],
                candidate_environment=build_manifests["candidate"]["environment"],
                inventory=inventory,
                snapshots=snapshots,
            )
            intervals.append(interval)
            input_paths.update(paths)
        result = classify_case(intervals)
        if result != record["statistical_result"]:
            raise protocol.ProtocolError(
                f"case {case}: declared case result differs from recalculation"
            )
        results[case] = result
        output_cases[case] = {
            "benchmark": record["benchmark"],
            "statistical_result": result,
            "intervals": intervals,
        }

    recalculated_campaign_result = campaign_result(results)
    if recalculated_campaign_result != declared_campaign_result:
        raise protocol.ProtocolError(
            "campaign.json: declared campaign result differs from recalculation"
        )
    expected_inventory = input_paths | output_relatives
    if set(inventory) != expected_inventory:
        missing = sorted(expected_inventory - set(inventory))
        extra = sorted(set(inventory) - expected_inventory)
        raise protocol.ProtocolError(
            "campaign.json: normative artifact inventory differs; "
            f"missing={missing}, extra={extra}"
        )
    observed, observed_directories = _observed_normative_inventory(
        root,
        ignored_files=(
            ignored_root_files
            if ignored_root_files is not None
            else (RUNNER_FINALIZATION_FILES if terminal_payload is not None else None)
        ),
    )
    expected_files = {"campaign.json", *expected_inventory}
    if output_records is None and recovery_output_dir is not None:
        recovery_relative = recovery_output_dir.relative_to(root)
        expected_files.update(
            path.relative_to(root).as_posix()
            for path in (
                *(recovery_output_dir / name for name in CLASSIFICATION_FILENAMES),
                *(
                    recovery_output_dir / TRANSACTION_DIRECTORY / name
                    for name in TRANSACTION_FILENAMES
                ),
            )
            if path.exists()
        )
    if observed != expected_files:
        missing = sorted(expected_files - observed)
        extra = sorted(observed - expected_files)
        raise protocol.ProtocolError(
            "campaign root artifact inventory differs; "
            f"missing={missing}, extra={extra}"
        )
    expected_directories: set[str] = set()
    for relative in expected_files:
        expected_directories.update(
            parent.as_posix()
            for parent in pathlib.PurePosixPath(relative).parents
            if parent.as_posix() != "."
        )
    if recovery_output_dir is not None:
        output_relative = recovery_output_dir.relative_to(root)
        if output_relative.parts:
            expected_directories.add(output_relative.as_posix())
            expected_directories.update(
                parent.as_posix()
                for parent in output_relative.parents
                if parent.as_posix() != "."
            )
        transaction = recovery_output_dir / TRANSACTION_DIRECTORY
        if output_records is None and transaction.exists():
            expected_directories.add(transaction.relative_to(root).as_posix())
    if observed_directories != expected_directories:
        missing = sorted(expected_directories - observed_directories)
        extra = sorted(observed_directories - expected_directories)
        raise protocol.ProtocolError(
            "campaign root directory inventory differs; "
            f"missing={missing}, extra={extra}"
        )

    input_inventory = {
        path: inventory[path] for path in sorted(input_paths)
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": campaign["protocol_sha256"],
        "classifier_sha256": campaign["classifier_sha256"],
        "candidate_sha": campaign["candidate_sha"],
        "comparison_kind": campaign["comparison_kind"],
        "threshold": THRESHOLD,
        "validity_state": "COMPLETE",
        "statistical_result": recalculated_campaign_result,
        "build_manifest_sha256": {
            identity: campaign["build_manifests"][identity]["sha256"]
            for identity in ("baseline", "candidate")
        },
        "input_artifact_inventory_sha256": protocol.sha256_json(input_inventory),
        "cases": output_cases,
    }
    return payload, campaign, root, output_records, output_contents


def _validate_terminal_derivation(
    running: Any, terminal: Any
) -> None:
    protocol.validate_manifest_fields(
        running, CAMPAIGN_FIELDS, context="running campaign.json"
    )
    protocol.validate_manifest_fields(
        terminal, CAMPAIGN_FIELDS, context="terminal campaign view"
    )
    expected = copy.deepcopy(terminal)
    outputs = expected.get("classification_artifacts")
    if type(outputs) is dict:
        for record in outputs.values():
            if type(record) is dict and type(record.get("path")) is str:
                expected["artifact_inventory"].pop(record["path"], None)
    expected["classification_artifacts"] = None
    expected["validity_state"] = "RUNNING"
    expected["statistical_result"] = None
    expected["completed_at"] = ""
    if type(expected.get("cases")) is dict:
        for record in expected["cases"].values():
            if type(record) is dict:
                record["statistical_result"] = None
    if running != expected:
        raise protocol.ProtocolError(
            "terminal view does not derive from the persisted RUNNING campaign"
        )


def load_validated_campaign(root: pathlib.Path) -> list[tuple[str, list[tuple], str]]:
    """Validate one protocol-v2 campaign without generating output artifacts."""
    payload, _campaign, _root, _records, _contents = _validated_classification(
        pathlib.Path(root) / "campaign.json"
    )
    return [
        (
            case,
            [
                (interval["lower"], interval["upper"], interval["point"])
                for interval in record["intervals"]
            ],
            record["statistical_result"],
        )
        for case, record in payload["cases"].items()
    ]


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise protocol.ProtocolError(f"classification payload is not JSON: {error}") from error
    return (rendered + "\n").encode("utf-8")


def _format_interval(interval: Mapping[str, Any]) -> str:
    return (
        f"{100.0 * interval['lower']:+.2f}.."
        f"{100.0 * interval['upper']:+.2f} "
        f"({100.0 * interval['point']:+.2f})"
    )


def _render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 2E Criterion classification",
        "",
        "| Case | Pair 1 | Pair 2 | Pair 3 | Class |",
        "|---|---:|---:|---:|---|",
    ]
    counts: Counter[str] = Counter()
    for case in sorted(payload["cases"]):
        record = payload["cases"][case]
        intervals = [_format_interval(item) for item in record["intervals"]]
        result = record["statistical_result"]
        counts[result] += 1
        lines.append(
            f"| {case} | {intervals[0]} | {intervals[1]} | "
            f"{intervals[2]} | {result} |"
        )
    lines.extend(
        [
            "",
            (
                f"Summary: {counts['PASS']} PASS / {counts['FAIL']} FAIL / "
                f"{counts['INCONCLUSIVE']} INCONCLUSIVE; "
                f"campaign={payload['statistical_result']}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _output_directory(root: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    output_dir = pathlib.Path(os.path.abspath(output_dir))
    if root != output_dir and root not in output_dir.parents:
        raise protocol.ProtocolError("classification output dir is outside campaign root")
    if _retained_proc_components(root) is not None:
        descriptor = _open_retained_directory_path(output_dir, create=True)
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise protocol.ProtocolError(
                    "classification output dir is not a directory"
                )
        finally:
            os.close(descriptor)
        return output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = output_dir.lstat()
        canonical = output_dir.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot prepare classification output dir: {error}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or canonical != output_dir:
        raise protocol.ProtocolError(
            "classification output dir must be a canonical regular directory"
        )
    return output_dir


def _output_records(
    root: pathlib.Path,
    paths: Mapping[str, pathlib.Path],
    contents: Mapping[str, bytes],
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(contents[name]).hexdigest(),
        }
        for name, path in paths.items()
    }


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _fsync_fd(descriptor: int) -> None:
    os.fsync(descriptor)


def _regular_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _read_open_regular(
    descriptor: int, directory_fd: int, name: str
) -> tuple[bytes, str, tuple[int, int]]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise protocol.ProtocolError(f"transaction entry is not regular: {name}")
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        _regular_identity(before) != _regular_identity(after)
        or _regular_identity(after) != _regular_identity(current)
    ):
        raise protocol.ProtocolError(f"transaction entry changed: {name}")
    content = b"".join(chunks)
    return (
        content,
        hashlib.sha256(content).hexdigest(),
        _directory_identity(after),
    )


def _read_regular_at_with_identity(
    directory_fd: int, name: str
) -> tuple[bytes, str, tuple[int, int]]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=directory_fd,
    )
    failure: BaseException | None = None
    result: tuple[bytes, str, tuple[int, int]] | None = None
    try:
        result = _read_open_regular(descriptor, directory_fd, name)
    except BaseException as error:
        failure = error
    try:
        os.close(descriptor)
    except BaseException as error:
        if failure is None:
            failure = error
    if failure is not None:
        raise failure
    if result is None:
        raise RuntimeError("regular-file read completed without a result")
    return result


def _read_regular_at(directory_fd: int, name: str) -> tuple[bytes, str]:
    content, digest, _identity = _read_regular_at_with_identity(directory_fd, name)
    return content, digest


def _verify_directory_path(path: pathlib.Path, descriptor: int) -> None:
    current = os.stat(
        path, follow_symlinks=_retained_proc_components(path) is not None
    )
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode) or _directory_identity(
        current
    ) != _directory_identity(opened):
        raise protocol.ProtocolError(f"transaction directory changed: {path}")


def _pending_stage_name(stable_name: str) -> str:
    return f"pending-{stable_name}"


def _stage_file_at(directory_fd: int, stable_name: str, content: bytes) -> None:
    pending_name = _pending_stage_name(stable_name)
    try:
        observed, _digest = _read_regular_at(directory_fd, stable_name)
    except FileNotFoundError:
        pass
    else:
        if observed != content:
            raise protocol.ProtocolError(
                f"classification transaction staged hash differs: {stable_name}"
            )
        try:
            pending, _digest = _read_regular_at(directory_fd, pending_name)
        except FileNotFoundError:
            return
        if pending != content:
            raise protocol.ProtocolError(
                f"classification transaction pending staged hash differs: {pending_name}"
            )
        os.unlink(pending_name, dir_fd=directory_fd)
        _fsync_fd(directory_fd)
        return

    try:
        pending, _digest = _read_regular_at(directory_fd, pending_name)
    except FileNotFoundError:
        descriptor = os.open(
            pending_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
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
                    raise OSError("short transaction staging write")
                view = view[written:]
            _fsync_fd(descriptor)
        except BaseException as error:
            failure = error
        try:
            os.close(descriptor)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            try:
                os.unlink(pending_name, dir_fd=directory_fd)
                _fsync_fd(directory_fd)
            except OSError:
                pass
            raise failure
    else:
        if pending != content:
            raise protocol.ProtocolError(
                f"classification transaction pending staged hash differs: {pending_name}"
            )
    try:
        os.link(
            pending_name,
            stable_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        observed, _digest = _read_regular_at(directory_fd, stable_name)
        if observed != content:
            raise protocol.ProtocolError(
                f"classification transaction staged hash differs: {stable_name}"
            )
    _fsync_fd(directory_fd)
    os.unlink(pending_name, dir_fd=directory_fd)
    _fsync_fd(directory_fd)


def _verify_regular_link(
    directory_fd: int, name: str, expected_identity: tuple[int, int]
) -> None:
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or _directory_identity(current) != expected_identity
    ):
        raise protocol.ProtocolError(f"staged transaction entry changed: {name}")


def _remove_created_final(
    output_fd: int, final_name: str, owned_identity: tuple[int, int]
) -> None:
    try:
        current = os.stat(final_name, dir_fd=output_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect failed classification publication: {final_name}"
        ) from error
    if _directory_identity(current) != owned_identity:
        return
    try:
        os.unlink(final_name, dir_fd=output_fd)
        _fsync_fd(output_fd)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot clean failed classification publication: {final_name}"
        ) from error


def _publish_stage(
    transaction_fd: int,
    stage_name: str,
    output_fd: int,
    final_name: str,
    expected: bytes,
) -> None:
    stage_fd = os.open(
        stage_name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=transaction_fd,
    )
    failure: BaseException | None = None
    try:
        staged, _digest, stage_identity = _read_open_regular(
            stage_fd, transaction_fd, stage_name
        )
        if staged != expected:
            raise protocol.ProtocolError(
                f"classification transaction staged hash differs: {stage_name}"
            )
        _verify_regular_link(transaction_fd, stage_name, stage_identity)
        try:
            os.link(
                stage_name,
                final_name,
                src_dir_fd=transaction_fd,
                dst_dir_fd=output_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed, _digest = _read_regular_at(output_fd, final_name)
            if observed != expected:
                raise protocol.ProtocolError(
                    f"existing final output hash differs: {final_name}"
                )
        else:
            try:
                _verify_regular_link(transaction_fd, stage_name, stage_identity)
                observed, _digest, final_identity = _read_regular_at_with_identity(
                    output_fd, final_name
                )
                if final_identity != stage_identity or observed != expected:
                    raise protocol.ProtocolError(
                        f"classification publication identity differs: {final_name}"
                    )
            except BaseException:
                try:
                    _remove_created_final(output_fd, final_name, stage_identity)
                except BaseException:
                    pass
                raise
        _fsync_fd(output_fd)
    except BaseException as error:
        failure = error
    try:
        os.close(stage_fd)
    except BaseException as error:
        if failure is None:
            failure = error
    if failure is not None:
        raise failure


def _run_output_transaction(
    output_dir: pathlib.Path,
    contents: Mapping[str, bytes],
) -> None:
    output_fd = _open_retained_directory_path(output_dir)
    transaction_fd: int | None = None
    try:
        _verify_directory_path(output_dir, output_fd)
        try:
            transaction_fd = os.open(
                TRANSACTION_DIRECTORY,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=output_fd,
            )
        except FileNotFoundError:
            existing = []
            for name in CLASSIFICATION_FILENAMES:
                try:
                    observed, _digest = _read_regular_at(output_fd, name)
                except FileNotFoundError:
                    continue
                existing.append(name)
                if observed != contents[name]:
                    raise protocol.ProtocolError(
                        f"existing final output hash differs: {name}"
                    )
            if len(existing) == len(CLASSIFICATION_FILENAMES):
                _fsync_fd(output_fd)
                _verify_directory_path(output_dir, output_fd)
                return
            if existing:
                raise protocol.ProtocolError(
                    "classification final output exists without transaction ownership"
                )
            os.mkdir(TRANSACTION_DIRECTORY, mode=0o700, dir_fd=output_fd)
            _fsync_fd(output_fd)
            transaction_fd = os.open(
                TRANSACTION_DIRECTORY,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=output_fd,
            )

        transaction_identity = _directory_identity(os.fstat(transaction_fd))
        observed = set(os.listdir(transaction_fd))
        if not observed.issubset(set(TRANSACTION_FILENAMES)):
            raise protocol.ProtocolError("classification transaction has unknown partials")
        stages = {
            "classification.json": "staged-classification.json",
            "summary.md": "staged-summary.md",
        }
        for name, stage_name in stages.items():
            _stage_file_at(transaction_fd, stage_name, contents[name])
        commit_payload = {
            "version": 1,
            "outputs": {
                name: {
                    "path": name,
                    "sha256": hashlib.sha256(contents[name]).hexdigest(),
                }
                for name in CLASSIFICATION_FILENAMES
            },
        }
        commit_content = _canonical_json_bytes(commit_payload)
        try:
            existing_commit = _read_regular_at(transaction_fd, "commit.json")[0]
        except FileNotFoundError:
            pass
        else:
            if _decode_json(
                existing_commit, "classification transaction commit"
            ) != commit_payload:
                raise protocol.ProtocolError(
                    "classification transaction commit differs"
                )
        _stage_file_at(transaction_fd, "commit.json", commit_content)
        parsed_commit = _decode_json(
            _read_regular_at(transaction_fd, "commit.json")[0],
            "classification transaction commit",
        )
        if parsed_commit != commit_payload:
            raise protocol.ProtocolError("classification transaction commit differs")
        _verify_transaction_identity(output_fd, transaction_fd, transaction_identity)
        # summary.md is human-readable state; classification.json is the final marker.
        for name in ("summary.md", "classification.json"):
            _publish_stage(
                transaction_fd, stages[name], output_fd, name, contents[name]
            )
            _verify_directory_path(output_dir, output_fd)
            _verify_transaction_identity(output_fd, transaction_fd, transaction_identity)
        _cleanup_output_transaction(
            output_fd, transaction_fd, output_dir, transaction_identity
        )
    finally:
        primary_failure = sys.exc_info()[1]
        close_failure: BaseException | None = None
        for descriptor in (transaction_fd, output_fd):
            if descriptor is None:
                continue
            try:
                os.close(descriptor)
            except BaseException as error:
                if close_failure is None:
                    close_failure = error
        if primary_failure is None and close_failure is not None:
            raise close_failure


def _cleanup_output_transaction(
    output_fd: int,
    transaction_fd: int,
    output_dir: pathlib.Path,
    transaction_identity: tuple[int, int],
) -> None:
    """Remove only the fixed owned transaction after both finals are durable."""
    _verify_transaction_identity(output_fd, transaction_fd, transaction_identity)
    for name in TRANSACTION_FILENAMES:
        try:
            os.unlink(name, dir_fd=transaction_fd)
        except FileNotFoundError:
            pass
        _fsync_fd(transaction_fd)
    _verify_transaction_identity(output_fd, transaction_fd, transaction_identity)
    os.rmdir(TRANSACTION_DIRECTORY, dir_fd=output_fd)
    _fsync_fd(output_fd)
    _verify_directory_path(output_dir, output_fd)


def _verify_transaction_identity(
    output_fd: int,
    transaction_fd: int,
    expected: tuple[int, int],
) -> None:
    current = os.stat(
        TRANSACTION_DIRECTORY, dir_fd=output_fd, follow_symlinks=False
    )
    if (
        not stat.S_ISDIR(current.st_mode)
        or _directory_identity(current) != expected
        or _directory_identity(os.fstat(transaction_fd)) != expected
    ):
        raise protocol.ProtocolError("classification transaction directory changed")


def _classify_validated(
    campaign_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    terminal_payload: Mapping[str, Any] | None,
    ignored_root_files: set[str] | None = None,
) -> dict[str, Any]:
    campaign_path = _canonical_regular_file(
        pathlib.Path(os.path.abspath(campaign_path)), "campaign manifest"
    )
    root = campaign_path.parent
    output_dir = _output_directory(root, output_dir)
    payload, _campaign, root, registered, registered_contents = (
        _validated_classification(
            campaign_path,
            recovery_output_dir=output_dir,
            terminal_payload=terminal_payload,
            ignored_root_files=ignored_root_files,
        )
    )
    paths = {
        "classification.json": output_dir / "classification.json",
        "summary.md": output_dir / "summary.md",
    }
    summary = _render_summary(payload)
    expected_bytes = {
        "classification.json": _canonical_json_bytes(payload),
        "summary.md": summary.encode("utf-8"),
    }

    if registered is None:
        _run_output_transaction(output_dir, expected_bytes)
        records = _output_records(root, paths, expected_bytes)
    else:
        expected_paths = {
            name: path.relative_to(root).as_posix() for name, path in paths.items()
        }
        if any(registered[name]["path"] != expected_paths[name] for name in paths):
            raise protocol.ProtocolError(
                "registered classification artifacts do not match output dir"
            )
        for name in paths:
            if registered_contents[name] != expected_bytes[name]:
                raise protocol.ProtocolError(
                    f"registered classification output content differs: {name}"
                )
        records = _output_records(root, paths, registered_contents)
        if records != registered:
            raise protocol.ProtocolError("registered classification output digest differs")

    result = dict(payload)
    result["output_artifacts"] = records
    return result


def classify_campaign(
    campaign_path: pathlib.Path, output_dir: pathlib.Path
) -> dict[str, Any]:
    """Recompute a persisted COMPLETE campaign and create or verify outputs."""
    return _classify_validated(
        campaign_path, output_dir, terminal_payload=None
    )


def classify_terminal_view(
    running_campaign_path: pathlib.Path,
    terminal_payload: Mapping[str, Any],
    output_dir: pathlib.Path,
    *,
    root_descriptor: int | None = None,
    retained_root_observer: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Classify an in-memory terminal view derived from persisted RUNNING state."""
    if root_descriptor is None:
        return _classify_validated(
            running_campaign_path,
            output_dir,
            terminal_payload=terminal_payload,
        )
    logical_campaign = pathlib.Path(os.path.abspath(running_campaign_path))
    if logical_campaign.name != "campaign.json":
        raise protocol.ProtocolError("terminal view requires campaign.json")
    logical_root = logical_campaign.parent
    logical_output = pathlib.Path(os.path.abspath(output_dir))
    try:
        output_relative = logical_output.relative_to(logical_root)
    except ValueError as error:
        raise protocol.ProtocolError(
            "classification output dir is outside campaign root"
        ) from error
    retained_root, expected_identity = _retained_root_path(root_descriptor)
    if retained_root_observer is not None:
        retained_root_observer()
    result = _classify_validated(
        retained_root / "campaign.json",
        retained_root / output_relative,
        terminal_payload=terminal_payload,
    )
    if _directory_identity(os.fstat(root_descriptor)) != expected_identity:
        raise protocol.ProtocolError("terminal view root descriptor changed")
    try:
        logical_current = os.stat(logical_root, follow_symlinks=False)
    except OSError as error:
        raise protocol.ProtocolError("terminal view root identity changed") from error
    if (
        not stat.S_ISDIR(logical_current.st_mode)
        or _directory_identity(logical_current) != expected_identity
    ):
        raise protocol.ProtocolError("terminal view root identity changed")
    return result


def classify_campaign_retained(
    campaign_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    root_descriptor: int,
    ignored_root_files: set[str] | None = None,
) -> dict[str, Any]:
    """Fully revalidate COMPLETE evidence through a retained campaign root."""
    logical_campaign = pathlib.Path(os.path.abspath(campaign_path))
    if logical_campaign.name != "campaign.json":
        raise protocol.ProtocolError("retained classification requires campaign.json")
    logical_root = logical_campaign.parent
    logical_output = pathlib.Path(os.path.abspath(output_dir))
    try:
        output_relative = logical_output.relative_to(logical_root)
    except ValueError as error:
        raise protocol.ProtocolError(
            "classification output dir is outside campaign root"
        ) from error
    retained_root, expected_identity = _retained_root_path(root_descriptor)
    result = _classify_validated(
        retained_root / "campaign.json",
        retained_root / output_relative,
        terminal_payload=None,
        ignored_root_files=ignored_root_files,
    )
    if _directory_identity(os.fstat(root_descriptor)) != expected_identity:
        raise protocol.ProtocolError("retained campaign root descriptor changed")
    try:
        logical_current = os.stat(logical_root, follow_symlinks=False)
    except OSError as error:
        raise protocol.ProtocolError("retained campaign root identity changed") from error
    if (
        not stat.S_ISDIR(logical_current.st_mode)
        or _directory_identity(logical_current) != expected_identity
    ):
        raise protocol.ProtocolError("retained campaign root identity changed")
    return result


def _retained_root_path(
    root_descriptor: int,
) -> tuple[pathlib.Path, tuple[int, int]]:
    try:
        opened = os.fstat(root_descriptor)
        proc_path = pathlib.Path(f"/proc/self/fd/{root_descriptor}")
        current = os.stat(proc_path)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot resolve retained campaign root: {error}"
        ) from error
    identity = _directory_identity(opened)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or _directory_identity(current) != identity
    ):
        raise protocol.ProtocolError("retained campaign root identity differs")
    return proc_path, identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()
    campaign_path = args.root if args.root.name == "campaign.json" else args.root / "campaign.json"
    output_dir = args.output_dir if args.output_dir is not None else campaign_path.parent
    result = classify_campaign(campaign_path, output_dir)
    print(
        (campaign_path.parent / result["output_artifacts"]["summary.md"]["path"])
        .read_text()
    )


if __name__ == "__main__":
    main()
