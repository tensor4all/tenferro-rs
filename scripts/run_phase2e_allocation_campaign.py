#!/usr/bin/env python3
"""Run one indivisible Phase 2E allocation comparison."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import pathlib
import stat
import sys
from collections.abc import Callable, Mapping
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol
from scripts.run_phase1_eager_campaign import (
    EXECUTABLE_SEALS,
    PinnedDirectory,
    PinnedExecutable,
    _sha256_open_file,
)


PROCESS_DEADLINE_SECONDS = 30
REPETITIONS = 4096
OBSERVATION_ORDERS = protocol.PAIR_ORDERS
EXPECTED_LAUNCH_COUNT = 2 * len(OBSERVATION_ORDERS) * len(protocol.CANONICAL_CASES)
EXIT_BY_GATE = {"PASS": 0, "FAIL": 3}
RECORD_FIELDS = {
    "allocated_bytes",
    "allocation_count",
    "allocation_failures",
    "case",
    "checksum",
    "counter_overflow",
    "repetitions",
}
PROTOCOL_SHA256 = protocol.sha256_json(
    {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "case_inventory": list(protocol.CANONICAL_CASES),
        "observation_orders": list(OBSERVATION_ORDERS),
        "record_fields": sorted(RECORD_FIELDS),
        "repetitions": REPETITIONS,
    }
)
FINALIZATION_STAGE = ".allocation-final.json"
FINALIZATION_MARKER = ".allocation-finalization.json"
FINALIZATION_PUBLISH = ".allocation-publish.json"
FINALIZATION_FILES = {
    FINALIZATION_STAGE,
    FINALIZATION_MARKER,
    FINALIZATION_PUBLISH,
}


def _strict_json_object(payload: str, context: str) -> dict[str, Any]:
    if type(payload) is not str or not payload.endswith("\n") or payload.count("\n") != 1:
        raise protocol.ProtocolError(f"{context} framing mismatch")
    body = payload[:-1]
    if not body.startswith("{") or not body.endswith("}"):
        raise protocol.ProtocolError(f"{context} is not one compact JSON object")
    in_string = False
    escaped = False
    for character in body:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character.isspace():
            raise protocol.ProtocolError(f"{context} is not compact JSON")
    def reject_duplicates(pairs):
        decoded_object = {}
        for key, value in pairs:
            if key in decoded_object:
                raise protocol.ProtocolError(f"{context} contains duplicate key: {key}")
            decoded_object[key] = value
        return decoded_object

    try:
        decoded = json.loads(
            body,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                protocol.ProtocolError(f"{context} contains non-finite number: {value}")
            ),
        )
    except protocol.ProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise protocol.ProtocolError(f"{context} is malformed JSON") from error
    if type(decoded) is not dict:
        raise protocol.ProtocolError(f"{context} is not a JSON object")
    return decoded


def parse_probe_record(stdout: str, requested_case: str, returncode: int | None) -> dict[str, Any]:
    """Parse one successful probe record under the exact wire contract."""
    if type(returncode) is not int or returncode != 0:
        raise protocol.ProtocolError("allocation probe exit/record relation is invalid")
    record = _strict_json_object(stdout, "allocation probe stdout")
    if set(record) != RECORD_FIELDS:
        raise protocol.ProtocolError("allocation probe record schema mismatch")
    for name in ("allocated_bytes", "allocation_count", "allocation_failures"):
        value = record[name]
        if type(value) is not int or value < 0 or value > (1 << 64) - 1:
            raise protocol.ProtocolError(f"allocation probe {name} is not u64")
    if type(record["case"]) is not str or record["case"] != requested_case:
        raise protocol.ProtocolError("allocation probe case does not match request")
    checksum = record["checksum"]
    if type(checksum) is not float or not math.isfinite(checksum):
        raise protocol.ProtocolError("allocation probe checksum is not finite f64")
    if type(record["counter_overflow"]) is not bool:
        raise protocol.ProtocolError("allocation probe overflow flag is not bool")
    if type(record["repetitions"]) is not int or record["repetitions"] != REPETITIONS:
        raise protocol.ProtocolError("allocation probe repetition contract mismatch")
    if record["allocation_failures"] != 0 or record["counter_overflow"]:
        raise protocol.ProtocolError("allocation probe counters report invalid measurement")
    return record


def _read_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    try:
        decoded = json.loads(build._read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise protocol.ProtocolError(f"{context} is malformed") from error
    if type(decoded) is not dict:
        raise protocol.ProtocolError(f"{context} is not an object")
    return decoded


def _validate_executable(path: pathlib.Path, digest: str) -> None:
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect allocation probe executable: {error}"
        ) from error
    if (
        not path.is_absolute()
        or canonical != path
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(path, os.X_OK)
        or protocol.sha256_file(path) != digest
    ):
        raise protocol.ProtocolError("allocation probe executable identity mismatch")


def _validate_pinned_executable(pinned: PinnedExecutable) -> None:
    """Validate retained source and sealed snapshot without following its logical path."""
    source = os.fstat(pinned.source_descriptor)
    snapshot = os.fstat(pinned.descriptor)
    if (
        (source.st_dev, source.st_ino) != (pinned.device, pinned.inode)
        or _sha256_open_file(pinned.source_descriptor) != pinned.digest
        or (snapshot.st_dev, snapshot.st_ino)
        != (pinned.snapshot_device, pinned.snapshot_inode)
        or fcntl.fcntl(pinned.descriptor, fcntl.F_GET_SEALS) != EXECUTABLE_SEALS
        or _sha256_open_file(pinned.descriptor) != pinned.digest
    ):
        raise protocol.ProtocolError(
            f"pinned allocation executable changed: {pinned.logical_path}"
        )


def _validate_inputs(
    comparison_kind: str,
    probe_manifests: Mapping[str, Mapping[str, Any]],
    tenferro_manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    if comparison_kind not in protocol.LANE_NAMES:
        raise protocol.ProtocolError("invalid allocation comparison kind")
    roles = tuple(build.BUILD_MANIFEST_PATHS)
    if tuple(probe_manifests) != roles or tuple(tenferro_manifests) != roles:
        raise protocol.ProtocolError("allocation build role inventory mismatch")
    baseline_role = {
        "direct-current-main": "direct-current-main-baseline",
        "common-lock-normalized": "common-lock-normalized-baseline",
    }[comparison_kind]
    expected_inventory = list(protocol.CANONICAL_CASES)
    reference_source = None
    required = {
        "protocol_version",
        "role",
        "head",
        "target",
        "profile",
        "validity_state",
        "generated_root",
        "target_dir",
        "executable",
        "executable_sha256",
        "lock_name",
        "lock_sha256",
        "cargo_config_chain",
        "config_chain_sha256",
        "resolved_features",
        "resolved_features_sha256",
        "template_sha256",
        "source_sha256",
        "generated_manifest_sha256",
        "generated_source_sha256",
        "case_inventory",
        "repetitions",
        "build_commands",
        "build_environment",
        "environment",
        "toolchain_sha256",
        "tenferro_build_manifest_sha256",
    }
    for role in roles:
        probe = probe_manifests[role]
        tenferro = tenferro_manifests[role]
        if type(probe) is not dict or set(probe) != required:
            raise protocol.ProtocolError(f"{role} probe build manifest schema mismatch")
        if (
            probe["protocol_version"] != protocol.PROTOCOL_VERSION
            or probe["role"] != role
            or probe["profile"] != "bench"
            or probe["validity_state"] != "COMPLETE"
            or probe["case_inventory"] != expected_inventory
            or probe["repetitions"] != REPETITIONS
        ):
            raise protocol.ProtocolError(f"{role} probe build identity mismatch")
        expected_lock = "direct-probe" if role == roles[0] else "common-probe"
        if probe["lock_name"] != expected_lock:
            raise protocol.ProtocolError(f"{role} probe lock ownership mismatch")
        if probe["source_sha256"] != probe["generated_source_sha256"]:
            raise protocol.ProtocolError(f"{role} generated probe source mismatch")
        source = (
            probe["template_sha256"],
            probe["source_sha256"],
            probe["generated_source_sha256"],
        )
        if reference_source is None:
            reference_source = source
        elif source != reference_source:
            raise protocol.ProtocolError("allocation probe source differs across roles")
        if type(tenferro) is not dict or tenferro.get("role") != role:
            raise protocol.ProtocolError(f"{role} tenferro build identity mismatch")
        if probe["toolchain_sha256"] != protocol.sha256_json(
            tenferro.get("toolchain", {})
        ):
            raise protocol.ProtocolError(f"{role} probe toolchain digest mismatch")
        if probe["tenferro_build_manifest_sha256"] != protocol.sha256_json(tenferro):
            raise protocol.ProtocolError(f"{role} tenferro build digest mismatch")
        if probe["head"] != tenferro.get("head"):
            raise protocol.ProtocolError(f"{role} probe HEAD differs from tenferro build")
        executable = pathlib.Path(probe["executable"])
        generated_root = pathlib.Path(probe["generated_root"])
        target_dir = pathlib.Path(probe["target_dir"])
        if (
            not generated_root.is_absolute()
            or not target_dir.is_absolute()
            or executable.parent.parent != target_dir
            or type(probe["build_commands"]) is not list
            or type(probe["build_environment"]) is not dict
        ):
            raise protocol.ProtocolError(f"{role} probe build provenance mismatch")
        _validate_executable(executable, probe["executable_sha256"])
        environment = probe["environment"]
        if type(environment) is not dict or environment != protocol.runtime_environment(
            path=environment.get("PATH"), home=environment.get("HOME")
        ):
            raise protocol.ProtocolError(f"{role} probe environment is not sealed")
    return baseline_role, "candidate"


def _initial_manifest(
    args,
    probes: Mapping[str, Mapping[str, Any]],
    tenferro: Mapping[str, Mapping[str, Any]],
    baseline_role: str,
    pinned_executables: Mapping[str, PinnedExecutable],
) -> dict[str, Any]:
    relevant = (baseline_role, "candidate")
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "comparison_kind": args.comparison_kind,
        "attempt_id": args.attempt_id,
        "validity_state": "RUNNING",
        "gate": None,
        "case_inventory": list(protocol.CANONICAL_CASES),
        "repetitions": REPETITIONS,
        "observation_orders": list(OBSERVATION_ORDERS),
        "expected_launch_count": EXPECTED_LAUNCH_COUNT,
        "launch_count": 0,
        "observations": [],
        "invalid_reason": None,
        "probe_builds": {role: copy.deepcopy(probes[role]) for role in relevant},
        "tenferro_builds": {role: copy.deepcopy(tenferro[role]) for role in relevant},
        "role_locks": {role: probes[role]["lock_sha256"] for role in relevant},
        "executable_identities": {
            role: {
                "logical_path": str(pinned_executables[role].logical_path),
                "source_device": pinned_executables[role].device,
                "source_inode": pinned_executables[role].inode,
                "snapshot_device": pinned_executables[role].snapshot_device,
                "snapshot_inode": pinned_executables[role].snapshot_inode,
                "sha256": pinned_executables[role].digest,
            }
            for role in relevant
        },
    }


def _close_ledger(
    path: pathlib.Path,
    ledger: dict[str, Any],
    args,
    gate: str | None,
    validity: str,
    atomic_writer: Callable[[pathlib.Path, Any], None],
) -> dict[str, Any]:
    closed = protocol.close_attempt(
        ledger,
        "allocation",
        args.comparison_kind,
        args.attempt_id,
        gate,
        validity_state=validity,
    )
    atomic_writer(path, closed)
    return closed


def _allocation_attempt(ledger: Mapping[str, Any], args) -> Mapping[str, Any]:
    matches = [
        attempt
        for attempt in ledger["attempts"]
        if attempt["stage"] == "allocation"
        and attempt["lane"] == args.comparison_kind
        and attempt["attempt_id"] == args.attempt_id
    ]
    if len(matches) != 1:
        raise protocol.ProtocolError("allocation finalization attempt is not unique")
    return matches[0]


def _finalization_marker(terminal: Mapping[str, Any], args, exit_code: int) -> dict[str, Any]:
    return {
        "version": 1,
        "candidate_sha": terminal["tenferro_builds"]["candidate"]["head"],
        "comparison_kind": args.comparison_kind,
        "attempt_id": args.attempt_id,
        "campaign_sha256": protocol.sha256_json(terminal),
        "validity_state": terminal["validity_state"],
        "gate": terminal["gate"],
        "exit_code": exit_code,
    }


def _publish_staged_allocation(root: PinnedDirectory) -> None:
    try:
        os.link(
            FINALIZATION_STAGE,
            FINALIZATION_PUBLISH,
            src_dir_fd=root.descriptor,
            dst_dir_fd=root.descriptor,
            follow_symlinks=False,
        )
    except FileExistsError:
        pass
    _require_same_regular_inode(root, FINALIZATION_STAGE, FINALIZATION_PUBLISH)
    os.replace(
        FINALIZATION_PUBLISH,
        "allocation.json",
        src_dir_fd=root.descriptor,
        dst_dir_fd=root.descriptor,
    )
    os.fsync(root.descriptor)


def _require_same_regular_inode(
    root: PinnedDirectory, first: str, second: str
) -> None:
    staged = os.stat(first, dir_fd=root.descriptor, follow_symlinks=False)
    partial = os.stat(second, dir_fd=root.descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(staged.st_mode)
        or not stat.S_ISREG(partial.st_mode)
        or (staged.st_dev, staged.st_ino) != (partial.st_dev, partial.st_ino)
    ):
        raise protocol.ProtocolError("allocation publish partial is not staged inode")


def _cleanup_finalization(root: PinnedDirectory) -> None:
    for name in (FINALIZATION_MARKER, FINALIZATION_PUBLISH, FINALIZATION_STAGE):
        try:
            os.unlink(name, dir_fd=root.descriptor)
        except FileNotFoundError:
            pass
    os.fsync(root.descriptor)


def _validate_terminal_allocation(
    terminal: Mapping[str, Any], ledger: Mapping[str, Any], args
) -> int:
    validity = terminal.get("validity_state")
    gate = terminal.get("gate")
    if (
        terminal.get("protocol_version") != protocol.PROTOCOL_VERSION
        or terminal.get("protocol_sha256") != PROTOCOL_SHA256
        or terminal.get("comparison_kind") != args.comparison_kind
        or terminal.get("attempt_id") != args.attempt_id
        or terminal.get("expected_launch_count") != EXPECTED_LAUNCH_COUNT
        or validity not in ("COMPLETE", "INCONCLUSIVE")
        or (validity == "COMPLETE" and gate not in EXIT_BY_GATE)
        or (validity == "INCONCLUSIVE" and gate is not None)
        or terminal.get("tenferro_builds", {}).get("candidate", {}).get("head")
        != ledger["candidate_sha"]
    ):
        raise protocol.ProtocolError("terminal allocation identity differs")
    return EXIT_BY_GATE[gate] if validity == "COMPLETE" else 2


def _require_closed_allocation_attempt(
    ledger: Mapping[str, Any], args, terminal: Mapping[str, Any]
) -> None:
    attempt = _allocation_attempt(ledger, args)
    expected_state = (
        "COMPLETE"
        if terminal["validity_state"] == "COMPLETE"
        else "INCONCLUSIVE"
    )
    if (
        ledger["active_attempt_id"] is not None
        or attempt["state"] != expected_state
        or attempt["validity_state"] != terminal["validity_state"]
        or attempt["statistical_result"] != terminal["gate"]
    ):
        raise protocol.ProtocolError("terminal allocation ledger differs")


def _root_json_commit_state(
    root: PinnedDirectory, relative: str, expected: Mapping[str, Any]
) -> str:
    try:
        content = root.read_regular(relative)
    except FileNotFoundError:
        return "ABSENT"
    except BaseException:
        return "UNKNOWN"
    return (
        "EXACT"
        if content == protocol._canonical_json_bytes(expected)
        else "MISMATCH"
    )


def _inconclusive_terminal(
    terminal: Mapping[str, Any], reason: BaseException | str
) -> dict[str, Any]:
    fallback = copy.deepcopy(dict(terminal))
    fallback["validity_state"] = "INCONCLUSIVE"
    fallback["gate"] = None
    fallback["invalid_reason"] = (
        reason
        if isinstance(reason, str)
        else f"{type(reason).__name__}: {reason}"
    )
    return fallback


def _finish_staged_allocation(
    root: PinnedDirectory,
    ledger_path: pathlib.Path,
    ledger: dict[str, Any],
    args,
    terminal: dict[str, Any],
    exit_code: int,
    atomic_writer: Callable[[pathlib.Path, Any], None],
) -> int:
    marker = _finalization_marker(terminal, args, exit_code)
    root.atomic_json(FINALIZATION_MARKER, marker)
    _close_ledger(
        ledger_path,
        ledger,
        args,
        terminal["gate"],
        terminal["validity_state"],
        atomic_writer,
    )
    _publish_staged_allocation(root)
    _cleanup_finalization(root)
    return exit_code


def _finalize_allocation(
    root: PinnedDirectory,
    ledger_path: pathlib.Path,
    ledger: dict[str, Any],
    args,
    terminal: dict[str, Any],
    exit_code: int,
    atomic_writer: Callable[[pathlib.Path, Any], None],
) -> int:
    try:
        root.atomic_json(FINALIZATION_STAGE, terminal)
    except BaseException as primary:
        state = _root_json_commit_state(root, FINALIZATION_STAGE, terminal)
        if state == "EXACT":
            raise
        if state != "ABSENT":
            raise protocol.ProtocolError(
                f"allocation stage commit state is {state.lower()}"
            ) from primary
        fallback = _inconclusive_terminal(terminal, primary)
        try:
            root.atomic_json(FINALIZATION_STAGE, fallback)
            _finish_staged_allocation(
                root,
                ledger_path,
                ledger,
                args,
                fallback,
                2,
                atomic_writer,
            )
        except BaseException as secondary:
            build._record_suppressed_failure(
                primary, "allocation fallback finalization", secondary
            )
            raise primary
        if isinstance(primary, Exception):
            return 2
        raise primary
    return _finish_staged_allocation(
        root,
        ledger_path,
        ledger,
        args,
        terminal,
        exit_code,
        atomic_writer,
    )


def _recover_finalization(args, atomic_writer) -> int | None:
    artifact_path = pathlib.Path(os.path.abspath(args.artifact_root))
    try:
        metadata = artifact_path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        raise protocol.ProtocolError("allocation artifact root is not a directory")
    root = PinnedDirectory(artifact_path)
    outcome: int | None = None
    primary: BaseException | None = None
    try:
        files, directories = root.inventory()
        if (
            directories
            or "allocation.json" not in files
            or not files <= ({"allocation.json"} | FINALIZATION_FILES)
        ):
            raise protocol.ProtocolError("allocation recovery inventory is invalid")
        persisted = _read_root_json(root, "allocation.json")
        ledger_path = pathlib.Path(args.ledger).resolve(strict=True)
        ledger = _read_json(ledger_path, "allocation ledger")
        protocol.validate_ledger(ledger)
        active = ledger["active_attempt_id"]
        marker_present = FINALIZATION_MARKER in files
        stage_present = FINALIZATION_STAGE in files
        if persisted.get("validity_state") in ("COMPLETE", "INCONCLUSIVE"):
            if active is not None:
                raise protocol.ProtocolError("terminal allocation has active ledger attempt")
            exit_code = _validate_terminal_allocation(persisted, ledger, args)
            _require_closed_allocation_attempt(ledger, args, persisted)
            persisted_digest = protocol.sha256_json(persisted)
            if stage_present and protocol.sha256_json(
                _read_root_json(root, FINALIZATION_STAGE)
            ) != persisted_digest:
                raise protocol.ProtocolError("terminal allocation stage differs")
            if stage_present:
                _require_same_regular_inode(
                    root, "allocation.json", FINALIZATION_STAGE
                )
            if FINALIZATION_PUBLISH in files:
                if not stage_present:
                    raise protocol.ProtocolError(
                        "allocation publish partial has no stage"
                    )
                _require_same_regular_inode(
                    root, FINALIZATION_STAGE, FINALIZATION_PUBLISH
                )
            if marker_present:
                terminal_marker = _read_root_json(root, FINALIZATION_MARKER)
                if terminal_marker.get("campaign_sha256") != persisted_digest:
                    raise protocol.ProtocolError("terminal allocation marker differs")
            _cleanup_finalization(root)
            outcome = exit_code
            return outcome
        if persisted.get("validity_state") != "RUNNING":
            raise protocol.ProtocolError("allocation finalization state is unreachable")
        if marker_present and not stage_present:
            raise protocol.ProtocolError("allocation marker exists without its stage")
        if stage_present:
            terminal = _read_root_json(root, FINALIZATION_STAGE)
        else:
            terminal = _inconclusive_terminal(
                persisted, "allocation finalization stage was not committed"
            )
            root.atomic_json(FINALIZATION_STAGE, terminal)
            stage_present = True
        exit_code = _validate_terminal_allocation(terminal, ledger, args)
        expected_marker = _finalization_marker(terminal, args, exit_code)
        if marker_present:
            marker = _read_root_json(root, FINALIZATION_MARKER)
        else:
            try:
                root.atomic_json(FINALIZATION_MARKER, expected_marker)
            except BaseException as primary:
                if (
                    _root_json_commit_state(
                        root, FINALIZATION_MARKER, expected_marker
                    )
                    != "EXACT"
                ):
                    raise
                if not isinstance(primary, Exception):
                    raise
            marker = expected_marker
        if (
            set(marker)
            != {
                "version",
                "candidate_sha",
                "comparison_kind",
                "attempt_id",
                "campaign_sha256",
                "validity_state",
                "gate",
                "exit_code",
            }
            or
            marker.get("version") != 1
            or marker.get("comparison_kind") != args.comparison_kind
            or marker.get("attempt_id") != args.attempt_id
            or marker.get("campaign_sha256") != protocol.sha256_json(terminal)
            or marker.get("candidate_sha") != ledger["candidate_sha"]
            or marker.get("exit_code") not in (0, 2, 3)
        ):
            raise protocol.ProtocolError("allocation finalization marker differs")
        if marker != expected_marker:
            raise protocol.ProtocolError("allocation finalization marker content differs")
        if marker.get("exit_code") != exit_code:
            raise protocol.ProtocolError("allocation finalization exit differs")
        if active == args.attempt_id:
            _close_ledger(
                ledger_path,
                ledger,
                args,
                terminal["gate"],
                terminal["validity_state"],
                atomic_writer,
            )
        elif active is not None:
            raise protocol.ProtocolError("allocation ledger has foreign active attempt")
        else:
            _require_closed_allocation_attempt(ledger, args, terminal)
        _publish_staged_allocation(root)
        _cleanup_finalization(root)
        outcome = exit_code
        return outcome
    except BaseException as error:
        if isinstance(error, Exception) and not isinstance(
            error, protocol.ProtocolError
        ):
            primary = protocol.ProtocolError(
                f"allocation recovery failed: {error}"
            )
            raise primary from error
        primary = error
        raise
    finally:
        try:
            root.close()
        except BaseException as error:
            if primary is not None:
                build._record_suppressed_failure(
                    primary, "allocation recovery root close", error
                )
            elif outcome is None:
                raise


def _read_root_json(root: PinnedDirectory, relative: str) -> dict[str, Any]:
    try:
        decoded = json.loads(root.read_regular(relative))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise protocol.ProtocolError(
            f"allocation finalization JSON is malformed: {relative}"
        ) from error
    if type(decoded) is not dict:
        raise protocol.ProtocolError(
            f"allocation finalization JSON is not an object: {relative}"
        )
    return decoded


def _run_comparison_in_root(
    args,
    *,
    artifact_handle: PinnedDirectory,
    probe_manifests: Mapping[str, Mapping[str, Any]],
    tenferro_manifests: Mapping[str, Mapping[str, Any]],
    pinned_executables: Mapping[str, PinnedExecutable],
    command_runner: Callable[..., build.CommandResult] = build.run_bounded_command,
    atomic_writer: Callable[[pathlib.Path, Any], None] = protocol.atomic_write_json,
) -> int:
    """Run exactly one whole allocation lane attempt."""
    baseline_role, candidate_role = _validate_inputs(
        args.comparison_kind, probe_manifests, tenferro_manifests
    )
    ledger_path = pathlib.Path(args.ledger).resolve(strict=True)
    ledger = _read_json(ledger_path, "allocation ledger")
    protocol.validate_ledger(ledger)
    if ledger["candidate_sha"] != probe_manifests[candidate_role]["head"]:
        raise protocol.ProtocolError("allocation ledger candidate differs")
    ledger = protocol.open_attempt(
        ledger, "allocation", args.comparison_kind, args.attempt_id
    )
    atomic_writer(ledger_path, ledger)
    campaign = _initial_manifest(
        args,
        probe_manifests,
        tenferro_manifests,
        baseline_role,
        pinned_executables,
    )
    gate = "PASS"
    invalid_reason = None
    try:
        artifact_handle.atomic_json("allocation.json", campaign)
        for case in protocol.CANONICAL_CASES:
            per_binary: dict[str, list[tuple[int, int]]] = {
                baseline_role: [],
                candidate_role: [],
            }
            for observation, order in enumerate(OBSERVATION_ORDERS, start=1):
                identities = (
                    (baseline_role, candidate_role)
                    if order == "A/B"
                    else (candidate_role, baseline_role)
                )
                pair_records = {}
                for position, role in enumerate(identities, start=1):
                    probe = probe_manifests[role]
                    pinned = pinned_executables[role]
                    _validate_pinned_executable(pinned)
                    command = (str(pinned.launch_path), case)
                    result = command_runner(
                        command,
                        cwd=pathlib.Path(args.working_directory),
                        environment=probe["environment"],
                        deadline_seconds=PROCESS_DEADLINE_SECONDS,
                        inherited_descriptors=(pinned.descriptor,),
                    )
                    campaign["launch_count"] += 1
                    observation_record = {
                        "launch_index": campaign["launch_count"],
                        "case": case,
                        "observation": observation,
                        "order": order,
                        "position": position,
                        "role": role,
                        "record": None,
                    }
                    try:
                        if not isinstance(result, build.CommandResult):
                            raise protocol.ProtocolError(
                                "allocation runner returned foreign result"
                            )
                        if (
                            result.argv != command
                            or result.cwd != str(pathlib.Path(args.working_directory))
                            or result.environment
                            != dict(sorted(probe["environment"].items()))
                            or result.deadline_seconds != PROCESS_DEADLINE_SECONDS
                            or result.inherited_descriptors
                            != (pinned.descriptor,)
                            or result.validity_state != "COMPLETE"
                            or result.stderr
                        ):
                            raise protocol.ProtocolError(
                                "allocation process provenance is invalid"
                            )
                        parsed = parse_probe_record(
                            result.stdout, case, result.returncode
                        )
                    except Exception as error:
                        observation_record["invalid_reason"] = (
                            f"{type(error).__name__}: {error}"
                        )
                        campaign["observations"].append(observation_record)
                        raise
                    pair_records[role] = parsed
                    per_binary[role].append(
                        (parsed["allocation_count"], parsed["allocated_bytes"])
                    )
                    observation_record["record"] = parsed
                    campaign["observations"].append(observation_record)
                    artifact_handle.atomic_json("allocation.json", campaign)
                baseline = pair_records[baseline_role]
                candidate = pair_records[candidate_role]
                if (
                    candidate["allocation_count"] > baseline["allocation_count"]
                    or candidate["allocated_bytes"] > baseline["allocated_bytes"]
                ):
                    gate = "FAIL"
            for role, values in per_binary.items():
                if len(set(values)) != 1:
                    raise protocol.ProtocolError(
                        f"allocation observations are inconsistent within {role}"
                    )
        if campaign["launch_count"] != EXPECTED_LAUNCH_COUNT:
            raise protocol.ProtocolError("allocation launch inventory is incomplete")
    except Exception as error:
        invalid_reason = f"{type(error).__name__}: {error}"
    except BaseException as primary:
        campaign["validity_state"] = "INCONCLUSIVE"
        campaign["gate"] = None
        campaign["invalid_reason"] = f"{type(primary).__name__}: {primary}"
        try:
            _finalize_allocation(
                artifact_handle, ledger_path, ledger, args, campaign, 2, atomic_writer
            )
        except BaseException as secondary:
            build._record_suppressed_failure(primary, "allocation finalization", secondary)
        raise
    if invalid_reason is not None:
        campaign["validity_state"] = "INCONCLUSIVE"
        campaign["gate"] = None
        campaign["invalid_reason"] = invalid_reason
        return _finalize_allocation(
            artifact_handle, ledger_path, ledger, args, campaign, 2, atomic_writer
        )
    campaign["validity_state"] = "COMPLETE"
    campaign["gate"] = gate
    return _finalize_allocation(
        artifact_handle,
        ledger_path,
        ledger,
        args,
        campaign,
        EXIT_BY_GATE[gate],
        atomic_writer,
    )


def _run_comparison(
    args,
    *,
    probe_manifests: Mapping[str, Mapping[str, Any]],
    tenferro_manifests: Mapping[str, Mapping[str, Any]],
    command_runner: Callable[..., build.CommandResult] = build.run_bounded_command,
    atomic_writer: Callable[[pathlib.Path, Any], None] = protocol.atomic_write_json,
) -> int:
    """Pin a fresh attempt root and preserve primary failures while closing it."""
    _validate_inputs(args.comparison_kind, probe_manifests, tenferro_manifests)
    recovered = _recover_finalization(args, atomic_writer)
    if recovered is not None:
        return recovered
    artifact_handle: PinnedDirectory | None = None
    pinned_executables: dict[str, PinnedExecutable] = {}
    primary: BaseException | None = None
    outcome: int | None = None
    try:
        baseline_role, candidate_role = _validate_inputs(
            args.comparison_kind, probe_manifests, tenferro_manifests
        )
        for role in (baseline_role, candidate_role):
            probe = probe_manifests[role]
            pinned_executables[role] = PinnedExecutable.open(
                pathlib.Path(probe["executable"]), probe["executable_sha256"]
            )
        artifact_handle = PinnedDirectory.create_fresh(args.artifact_root)
        outcome = _run_comparison_in_root(
            args,
            artifact_handle=artifact_handle,
            probe_manifests=probe_manifests,
            tenferro_manifests=tenferro_manifests,
            pinned_executables=pinned_executables,
            command_runner=command_runner,
            atomic_writer=atomic_writer,
        )
        return outcome
    except BaseException as error:
        if isinstance(error, Exception) and not isinstance(
            error, protocol.ProtocolError
        ):
            primary = protocol.ProtocolError(
                f"allocation campaign failed: {error}"
            )
            raise primary from error
        primary = error
        raise
    finally:
        resources = ([artifact_handle] if artifact_handle is not None else []) + list(
            reversed(tuple(pinned_executables.values()))
        )
        close_failure: BaseException | None = None
        for resource in resources:
            try:
                resource.close()
            except BaseException as error:
                if primary is not None:
                    build._record_suppressed_failure(
                        primary, "allocation pinned resource close", error
                    )
                elif close_failure is None:
                    close_failure = error
        if primary is None and close_failure is not None and outcome is None:
            if isinstance(close_failure, Exception):
                raise protocol.ProtocolError(
                    f"cannot close allocation pinned resource: {close_failure}"
                ) from close_failure
            raise close_failure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-kind", choices=protocol.LANE_NAMES, required=True)
    parser.add_argument("--ledger", required=True, type=pathlib.Path)
    parser.add_argument("--attempt-id", required=True, type=int)
    parser.add_argument("--artifact-root", required=True, type=pathlib.Path)
    parser.add_argument("--working-directory", required=True, type=pathlib.Path)
    parser.add_argument("--probe-manifest-root", required=True, type=pathlib.Path)
    parser.add_argument("--tenferro-manifest-root", required=True, type=pathlib.Path)
    parser.add_argument("--repository", required=True, type=pathlib.Path)
    return parser


def parse_args(argv=None):
    return build_argument_parser().parse_args(argv)


def run_campaign(
    args,
    *,
    command_runner: Callable[..., build.CommandResult] = build.run_bounded_command,
) -> int:
    tenferro = {
        role: _read_json(
            pathlib.Path(args.tenferro_manifest_root) / relative,
            f"{role} tenferro manifest",
        )
        for role, relative in build.BUILD_MANIFEST_PATHS.items()
    }
    for manifest in tenferro.values():
        build.validate_build_manifest(manifest)
    probes = build.validate_allocation_probe_set(
        pathlib.Path(args.probe_manifest_root),
        tenferro,
        repository=pathlib.Path(args.repository),
    )
    return _run_comparison(
        args,
        probe_manifests=probes,
        tenferro_manifests=tenferro,
        command_runner=command_runner,
    )


def main(argv=None) -> int:
    try:
        return run_campaign(parse_args(argv))
    except protocol.ProtocolError as error:
        print(f"phase2e allocation campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
