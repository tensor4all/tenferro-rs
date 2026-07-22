#!/usr/bin/env python3
"""Collect and compose provenance-bound Phase 2E dispatch evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol

CPU_FILTER = "phase2e_characterization_evidence"
AD_FILTER = "phase2e_eager_characterization_evidence"
TEST_DEADLINE_SECONDS = 120
BENCH_ROW_DEADLINE_SECONDS = 30
TERMINATION_GRACE_SECONDS = 5
EVIDENCE_ENVIRONMENT_KEY = "TENFERRO_PHASE2E_EVIDENCE_DIR"


class ExecutionFailure(protocol.ProtocolError):
    def __init__(
        self, message: str, *, kind: str, stdout: str = "", stderr: str = "",
        termination: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.stdout = stdout
        self.stderr = stderr
        self.termination = dict(termination or {})

SOURCE_HOT_ITEMS = (
    ("crates/tenferro-ad/src/eager.rs", "pub fn with_eager_session<R: Send>("),
    ("crates/tenferro-ad/src/eager.rs", "pub(crate) fn materialize_value("),
    ("crates/tenferro-ad/src/eager.rs", "pub fn materialized(&self)"),
    ("crates/tenferro-ad/src/eager.rs", "pub(crate) fn materialized_arc(&self)"),
    ("crates/tenferro-ad/src/eager_backend.rs", "macro_rules! dispatch"),
    ("crates/tenferro-ad/src/eager_backend.rs", "fn with_backend_session<R: Send>("),
    ("crates/tenferro-cpu/src/backend.rs", "fn acquire_execution_permit("),
    ("crates/tenferro-cpu/src/backend.rs", "fn run_backend_session_cached<R: Send>("),
    ("crates/tenferro-cpu/src/backend.rs", "fn with_backend_session<R: Send>("),
    ("crates/tenferro-cpu/src/engine.rs", "pub(crate) fn new_managed("),
    ("crates/tenferro-cpu/src/elementwise.rs", "pub(crate) fn typed_add_with_pool<T>("),
    ("crates/tenferro-cpu/src/phase2e_observe.rs", "pub(crate) fn record_typed_add_worker()"),
    ("crates/tenferro-cpu/src/exec_session.rs", "fn run_native<R: Send>("),
    ("crates/tenferro-cpu/src/exec_session.rs", "fn run_native_fresh<R: FreshCpuOutput + Send>("),
    ("crates/tenferro-cpu/src/provider.rs", "pub(crate) fn enter<R: Send>("),
    ("crates/tenferro-cpu/src/provider.rs", "pub(crate) fn new(domain: &'a CpuResourceDomain"),
    ("crates/tenferro-cpu/src/provider.rs", "pub(crate) fn submit_outer("),
    ("crates/tenferro-cpu/src/provider.rs", "pub(crate) fn preferred_engine_mode("),
    ("crates/tenferro-cpu/src/provider.rs", "pub(crate) fn preferred_provider_mode("),
    ("crates/tenferro-cpu/src/dot_runtime.rs", "pub(crate) fn execute_dot_general_into("),
    ("crates/tenferro-cpu/src/dot_runtime.rs", "pub(crate) fn execute_grouped_gemm("),
    ("crates/tenferro-cpu/src/dot_runtime.rs", "fn execute_grouped_outer_typed<T>("),
)
BANNED_IDENTIFIER_TOKENS = ("TypeId", "Any", "HashMap")
BANNED_DISPATCH_PATTERNS = ("string_key", ".get(operation_name)", ".get(&format!(")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_keys() -> tuple[set[str], set[str]]:
    cpu: set[str] = set()
    ad: set[str] = set()
    for ownership in ("managed-exact", "external-exact", "external-advisory"):
        for budget in (1, 2, 4):
            cpu.update(
                f"{ownership}/budget-{budget}/{surface}"
                for surface in ("D-N", "D-D", "G-O")
            )
            ad.update(
                f"{ownership}/budget-{budget}/{surface}"
                for surface in ("E-N", "E-D")
            )
    cpu.update(("external-no-outer/budget-2/U-O", "external-no-inner/budget-2/U-I"))
    return cpu, ad


def expected_row_contract(key: str) -> tuple[list[int], str]:
    surface = key.rsplit("/", 1)[-1]
    budget = int(key.split("/budget-", 1)[1].split("/", 1)[0])
    if surface == "U-O":
        return [0, 1, 1, 0, 0, 0], "UnsupportedOuter"
    if surface == "U-I":
        return [0, 1, 1, 1, 0, 1], "Sequential"
    if surface == "G-O" and budget > 1:
        return [0, 1, 1, 0, 1, 2 * budget + 1], "Outer"
    session = 1 if surface.startswith("E-") else 0
    provider = 1 if surface in {"D-D", "E-D", "G-O"} else 0
    return [session, 1, 1, 1, 0, provider], (
        "Sequential" if budget == 1 else "Inner"
    )


def _validate_hardware_skip(row: Mapping[str, Any]) -> None:
    skip = row.get("hardware_skip")
    if skip is None:
        return
    if not isinstance(skip, Mapping):
        raise protocol.ProtocolError("hardware_skip must be null or a typed object")
    if skip.get("kind") not in {"InsufficientAllowedCpus", "InsufficientNumaNodes"}:
        raise protocol.ProtocolError("unknown hardware skip kind")
    if not isinstance(skip.get("required"), int) or not isinstance(skip.get("available"), int):
        raise protocol.ProtocolError("hardware skip requires integer required/available fields")


def validate_partition(artifact: Mapping[str, Any], owner: str) -> list[dict[str, Any]]:
    expected_cpu, expected_ad = canonical_keys()
    expected = expected_cpu if owner == "cpu" else expected_ad
    required_count = 29 if owner == "cpu" else 18
    if artifact.get("owner") != owner:
        raise protocol.ProtocolError(f"{owner} artifact has wrong owner")
    rows = artifact.get("characterization")
    if not isinstance(rows, list) or len(rows) != required_count:
        raise protocol.ProtocolError(f"{owner} artifact must contain {required_count} rows")
    keys: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("owner") != owner:
            raise protocol.ProtocolError(f"{owner} row has wrong owner")
        key = row.get("key")
        if not isinstance(key, str):
            raise protocol.ProtocolError("row key must be a string")
        keys.append(key)
        _validate_hardware_skip(row)
        for field in (
            "numerical_passed", "typed_error_recovered", "unwind_recovered",
            "post_recovery_passed",
        ):
            if row.get(field) is not True:
                raise protocol.ProtocolError(f"row {key} failed {field}")
        if owner == "cpu":
            counts = row.get("counts")
            if not isinstance(counts, list) or len(counts) != 6 or not all(type(x) is int for x in counts):
                raise protocol.ProtocolError(f"row {key} has invalid measured count vector")
            expected_counts, expected_mode = expected_row_contract(key)
            if counts != expected_counts or row.get("mode") != expected_mode:
                raise protocol.ProtocolError(f"row {key} differs from its count/mode contract")
            if key.endswith("/U-O") and (
                row.get("typed_error_kind") != "Scheduling"
                or row.get("typed_error_source")
                != "CPU domain executor scheduling failed: CPU domain CpuDomainId(9) does not support Outer mode"
            ):
                raise protocol.ProtocolError("U-O lacks its exact pre-submit scheduling source")
            recovery = row.get("recovery")
            if (
                not isinstance(recovery, dict)
                or recovery.get("fresh_reset") is not True
                or recovery.get("numerical_passed") is not True
                or recovery.get("subset_passed") is not True
                or not isinstance(recovery.get("counts"), list)
                or len(recovery["counts"]) != 6
                or not all(type(value) is int for value in recovery["counts"])
                or not isinstance(recovery.get("mode"), str)
                or not isinstance(recovery.get("observed_cpus"), list)
            ):
                raise protocol.ProtocolError(f"CPU row {key} lacks its fresh recovery record")
            if not key.endswith("/U-O") and (
                recovery["counts"] != counts
                or recovery["mode"] != row.get("mode")
                or not recovery["observed_cpus"]
                or not all(type(cpu) is int and cpu >= 0 for cpu in recovery["observed_cpus"])
            ):
                raise protocol.ProtocolError(f"CPU row {key} recovery rerun differs")
        else:
            if "counts" in row or "mode" in row:
                raise protocol.ProtocolError("AD raw rows may not supply downstream counts or modes")
            if row.get("session_entry") != 1:
                raise protocol.ProtocolError(f"AD row {key} did not measure one eager session")
            session_entry_cpus = row.get("session_entry_cpus")
            if (
                not isinstance(session_entry_cpus, list)
                or not session_entry_cpus
                or not all(type(cpu) is int and cpu >= 0 for cpu in session_entry_cpus)
            ):
                raise protocol.ProtocolError(f"AD row {key} lacks its session-entry observation")
            audit = row.get("placement_audit")
            budget = row.get("budget")
            if (
                not isinstance(budget, int)
                or not isinstance(audit, list)
                or len(audit) != budget
                or any(
                    not isinstance(item, list)
                    or len(item) != 2
                    or not all(type(value) is int and value >= 0 for value in item)
                    for item in audit
                )
                or {item[0] for item in audit} != set(range(budget))
            ):
                raise protocol.ProtocolError(f"AD row {key} lacks its all-worker placement audit")
            declared = row.get("declared_cpus")
            if not isinstance(declared, list) or not all(
                type(cpu) is int and cpu >= 0 for cpu in declared
            ):
                raise protocol.ProtocolError(f"AD row {key} has invalid declared CPUs")
            if not key.startswith("external-advisory/") and (
                not declared or any(item[1] not in declared for item in audit)
            ):
                raise protocol.ProtocolError(f"AD row {key} placement audit escaped its CPU set")
            expected_vector = "borrowed-add" if key.endswith("/E-N") else "borrowed-dot"
            if row.get("downstream_vector") != expected_vector:
                raise protocol.ProtocolError(f"AD row {key} names the wrong CPU downstream vector")
            expected_provider = 0 if key.endswith("/E-N") else 1
            if row.get("actual_provider") != expected_provider:
                raise protocol.ProtocolError(f"AD row {key} provider observation differs")
            if row.get("actual_install") not in (None, 1) or row.get("actual_submit") not in (None, 0):
                raise protocol.ProtocolError(f"AD row {key} executor observation differs")
            operation_workers = row.get("operation_workers")
            if key.endswith("/E-N") and (
                not isinstance(operation_workers, list)
                or not operation_workers
                or any(
                    not isinstance(item, list)
                    or len(item) != 2
                    or not all(type(value) is int and value >= 0 for value in item)
                    for item in operation_workers
                )
            ):
                raise protocol.ProtocolError(
                    f"AD row {key} lacks actual operation worker observations"
                )
            if (
                key.endswith("/E-N")
                and not key.startswith("external-advisory/")
                and any(item[1] not in declared for item in operation_workers)
            ):
                raise protocol.ProtocolError(
                    f"AD row {key} operation worker escaped its CPU set"
                )
            recovery = row.get("recovery")
            recovery_workers = recovery.get("operation_workers") if isinstance(recovery, dict) else None
            recovery_cpus = recovery.get("observed_cpus") if isinstance(recovery, dict) else None
            if (
                not isinstance(recovery, dict)
                or recovery.get("fresh_reset") is not True
                or recovery.get("session_entry") != 1
                or recovery.get("actual_install") not in (None, 1)
                or recovery.get("actual_submit") not in (None, 0)
                or recovery.get("actual_provider") != expected_provider
                or recovery.get("numerical_passed") is not True
                or recovery.get("subset_passed") is not True
                or not isinstance(recovery_cpus, list)
                or not recovery_cpus
                or not all(type(cpu) is int and cpu >= 0 for cpu in recovery_cpus)
            ):
                raise protocol.ProtocolError(f"AD row {key} lacks its fresh recovery record")
            if key.endswith("/E-N") and (
                not isinstance(recovery_workers, list)
                or not recovery_workers
                or any(
                    not isinstance(item, list)
                    or len(item) != 2
                    or not all(type(value) is int and value >= 0 for value in item)
                    for item in recovery_workers
                )
                or (
                    not key.startswith("external-advisory/")
                    and any(item[1] not in declared for item in recovery_workers)
                )
            ):
                raise protocol.ProtocolError(f"AD row {key} recovery lacks operation workers")
        observed_cpus = row.get("observed_cpus")
        allows_empty_observation = key.endswith("/U-O")
        if (
            row.get("hardware_skip") is None and not allows_empty_observation
            and (
                not isinstance(observed_cpus, list)
                or not observed_cpus
                or not all(type(cpu) is int and cpu >= 0 for cpu in observed_cpus)
            )
        ):
            raise protocol.ProtocolError(f"row {key} lacks actual CPU observations")
    if len(set(keys)) != len(keys):
        raise protocol.ProtocolError(f"{owner} artifact contains duplicate row keys")
    if set(keys) != expected:
        raise protocol.ProtocolError(f"{owner} artifact inventory differs from the canonical keys")
    return rows


def compose_characterization(cpu: Mapping[str, Any], ad: Mapping[str, Any]) -> dict[str, Any]:
    cpu_rows = validate_partition(cpu, "cpu")
    ad_rows = validate_partition(ad, "ad")
    vectors = cpu.get("canonical_vectors", [])
    expected_vectors = [
        [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1],
    ]
    if vectors != expected_vectors:
        raise protocol.ProtocolError("CPU artifact must own ten direct/borrowed vectors")
    entries = ad.get("session_entries")
    if entries != [1, 1, 1, 1, 1]:
        raise protocol.ProtocolError("AD session-entry proof must be five single entries")
    cpu_by_key = {row["key"]: row for row in cpu_rows}
    composed_ad = []
    for raw in ad_rows:
        ownership_budget = raw["key"].rsplit("/", 1)[0]
        direct_surface = "D-N" if raw["surface"] == "E-N" else "D-D"
        downstream_row = cpu_by_key[f"{ownership_budget}/{direct_surface}"]
        vector_index = 6 if raw["surface"] == "E-N" else 9
        vector = vectors[vector_index]
        counts = [raw["session_entry"], *vector[1:]]
        expected_counts, _expected_mode = expected_row_contract(raw["key"])
        if counts != expected_counts:
            raise protocol.ProtocolError(f"AD row {raw['key']} composed to the wrong vector")
        if raw["actual_install"] is not None and raw["actual_install"] != counts[3]:
            raise protocol.ProtocolError(f"AD row {raw['key']} install observation disagrees")
        if raw["actual_submit"] is not None and raw["actual_submit"] != counts[4]:
            raise protocol.ProtocolError(f"AD row {raw['key']} submit observation disagrees")
        if raw["actual_provider"] != counts[5]:
            raise protocol.ProtocolError(f"AD row {raw['key']} provider observation disagrees")
        composed = dict(raw)
        composed["counts"] = counts
        composed["mode"] = downstream_row["mode"]
        composed["downstream_mode_source"] = downstream_row["key"]
        recovery = dict(raw["recovery"])
        recovery_counts = [recovery["session_entry"], *vector[1:]]
        if recovery_counts != counts:
            raise protocol.ProtocolError(f"AD row {raw['key']} recovery composed incorrectly")
        recovery["counts"] = recovery_counts
        recovery["mode"] = downstream_row["recovery"]["mode"]
        recovery["downstream_mode_source"] = downstream_row["key"]
        composed["recovery"] = recovery
        composed_ad.append(composed)
    cross_socket = cpu.get("cross_socket_locality")
    if not isinstance(cross_socket, dict):
        raise protocol.ProtocolError("CPU artifact lacks cross-socket execution evidence")
    _validate_hardware_skip(cross_socket)
    usable_nodes = cross_socket.get("usable_numa_nodes")
    probes = cross_socket.get("probes")
    if type(usable_nodes) is not int or usable_nodes < 0 or not isinstance(probes, list):
        raise protocol.ProtocolError("cross-socket evidence has invalid availability")
    if cross_socket.get("hardware_skip") is None:
        if (
            usable_nodes < 2
            or len(probes) != 2
            or len({probe.get("node") for probe in probes if isinstance(probe, dict)}) != 2
        ):
            raise protocol.ProtocolError("cross-socket evidence did not execute two nodes")
        for probe in probes:
            if not isinstance(probe, dict):
                raise protocol.ProtocolError("cross-socket probe is invalid")
            declared = probe.get("declared_cpus")
            observed = probe.get("observed_cpus")
            if (
                probe.get("numerical_passed") is not True
                or probe.get("subset_passed") is not True
                or not isinstance(declared, list)
                or not declared
                or not isinstance(observed, list)
                or not observed
                or any(cpu not in declared for cpu in observed)
            ):
                raise protocol.ProtocolError("cross-socket probe lacks executed locality work")
    elif probes or cross_socket["hardware_skip"] != {
        "kind": "InsufficientNumaNodes", "required": 2, "available": usable_nodes,
    }:
        raise protocol.ProtocolError("cross-socket skip is not tied to unavailable hardware")
    rows = [*cpu_rows, *composed_ad]
    if len({row["key"] for row in rows}) != 47:
        raise protocol.ProtocolError("composed characterization must contain 47 unique rows")
    return {
        "validity_state": "PASS", "row_count": 47, "rows": rows,
        "cross_socket_locality": dict(cross_socket),
    }


def attach_hardware_validity(
    characterization: dict[str, Any], *, available_cpus: int, usable_numa_nodes: int,
) -> None:
    if available_cpus < 1 or usable_numa_nodes < 0:
        raise protocol.ProtocolError("hardware availability counts are invalid")
    for row in characterization["rows"]:
        budget = row["budget"]
        row["affinity_hardware_skip"] = (
            {
                "kind": "InsufficientAllowedCpus",
                "required": budget,
                "available": available_cpus,
            }
            if row["surface"] not in {"U-O", "U-I"} and available_cpus < budget
            else None
        )
    cross_socket = characterization.get("cross_socket_locality")
    if (
        not isinstance(cross_socket, dict)
        or cross_socket.get("usable_numa_nodes") != usable_numa_nodes
    ):
        raise protocol.ProtocolError("cross-socket evidence disagrees with gate hardware discovery")


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for field in value.strip().split(","):
        if not field:
            continue
        bounds = field.split("-", 1)
        start = int(bounds[0])
        stop = int(bounds[-1])
        cpus.update(range(start, stop + 1))
    return cpus


def hardware_availability() -> tuple[int, int]:
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(
        range(os.cpu_count() or 1)
    )
    usable_nodes = 0
    for cpu_list in pathlib.Path("/sys/devices/system/node").glob("node[0-9]*/cpulist"):
        try:
            node_cpus = _parse_cpu_list(cpu_list.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        usable_nodes += bool(node_cpus & allowed)
    return len(allowed), usable_nodes


def source_item(source: str, signature: str) -> str:
    """Return one Rust item, including its balanced outer brace pair."""
    start = source.find(signature)
    if start < 0:
        raise protocol.ProtocolError(f"dispatch source item is missing: {signature}")
    opening = source.find("{", start)
    if opening < 0:
        raise protocol.ProtocolError(f"dispatch source item has no body: {signature}")
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise protocol.ProtocolError(f"dispatch source item is unbalanced: {signature}")


def validate_source_contract(repository: pathlib.Path) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    sources: dict[str, str] = {}
    for relative, signature in SOURCE_HOT_ITEMS:
        path = repository / relative
        source = sources.setdefault(relative, path.read_text(encoding="utf-8"))
        dispatch_item = source_item(source, signature)
        for token in BANNED_IDENTIFIER_TOKENS:
            if token in dispatch_item:
                raise protocol.ProtocolError(
                    f"banned dispatch identifier {token} in {relative}: {signature}"
                )
        for pattern in BANNED_DISPATCH_PATTERNS:
            if pattern in dispatch_item:
                raise protocol.ProtocolError(
                    f"banned string/format dispatch in {relative}: {signature}"
                )
        inventory.append({
            "path": relative,
            "signature": signature,
            "source_sha256": sha256_file(path),
            "item_sha256": hashlib.sha256(dispatch_item.encode("utf-8")).hexdigest(),
        })
    production = (repository / "crates/tenferro-ad/src/eager_backend.rs").read_text()
    for fixture in ("struct RecordingBackend", "delegate_recording_backend_methods", "impl TensorElementwise for RecordingBackend"):
        if fixture in production:
            raise protocol.ProtocolError("RecordingBackend fixture leaked into production source")
    return inventory


def validate_external_scratch_root(
    repository: pathlib.Path, evidence_root: pathlib.Path, scratch_root: pathlib.Path,
) -> pathlib.Path:
    """Require scratch storage to be external to repository and evidence trees."""
    repository = pathlib.Path(repository).resolve(strict=True)
    evidence_root = pathlib.Path(evidence_root).resolve(strict=False)
    scratch_root = pathlib.Path(scratch_root).resolve(strict=True)
    for name, protected in (("repository", repository), ("evidence_root", evidence_root)):
        if (
            scratch_root == protected
            or scratch_root in protected.parents
            or protected in scratch_root.parents
        ):
            raise protocol.ProtocolError(
                f"scratch_root must be external and disjoint from {name}"
            )
    return scratch_root


def _validate_runtime_environment(
    environment: Mapping[str, str],
    *,
    criterion_home: str | None = None,
    affinity_row: str | None = None,
    affinity_file: str | None = None,
) -> dict[str, str]:
    if not isinstance(environment, dict):
        raise protocol.ProtocolError("runtime environment must be an exact dictionary")
    expected = protocol.runtime_environment(
        path=environment.get("PATH", ""),
        home=environment.get("HOME", ""),
        criterion_home=criterion_home,
        affinity_row=affinity_row,
        affinity_file=affinity_file,
    )
    added = {
        "CRITERION_HOME",
        "TENFERRO_PHASE2E_AFFINITY_ROW",
        "TENFERRO_PHASE2E_AFFINITY_FILE",
    }
    base = {name: value for name, value in expected.items() if name not in added}
    if environment != base:
        raise protocol.ProtocolError("runtime environment differs from the sealed allowlist")
    return expected


def validate_test_build_manifest(
    manifest: Mapping[str, Any], *, package: str, candidate: str,
    repository: pathlib.Path, common_lock: pathlib.Path,
) -> pathlib.Path:
    if manifest.get("validity_state") != "COMPLETE" or manifest.get("package") != package:
        raise protocol.ProtocolError("test build manifest identity mismatch")
    if manifest.get("candidate") != candidate:
        raise protocol.ProtocolError("test binary was not built from the candidate")
    if manifest.get("protocol_version") != protocol.PROTOCOL_VERSION:
        raise protocol.ProtocolError("test build protocol version mismatch")
    expected_protocol = sha256_file(repository / "scripts/phase2e_protocol.py")
    if manifest.get("protocol_sha256") != expected_protocol:
        raise protocol.ProtocolError("test build protocol digest mismatch")
    tree = subprocess.run(
        ("git", "ls-tree", "-r", "-z", "--full-tree", candidate),
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout
    expected_tree = hashlib.sha256(tree.encode()).hexdigest()
    if manifest.get("candidate_tree_sha256") != expected_tree:
        raise protocol.ProtocolError("test build candidate tree digest mismatch")
    expected_sources = {
        relative: sha256_file(repository / relative)
        for relative in build.TASK7_SOURCE_PATHS
    }
    if manifest.get("source_inventory") != dict(sorted(expected_sources.items())):
        raise protocol.ProtocolError("test build source inventory mismatch")
    expected_lock = sha256_file(common_lock)
    if manifest.get("common_lock_sha256") != expected_lock:
        raise protocol.ProtocolError("test build common lock digest mismatch")
    if tuple(manifest.get("argv", ())) != build.DISPATCH_TEST_COMMANDS[package]:
        raise protocol.ProtocolError("evidence rejects a non-contract Cargo test build")
    if manifest.get("requested_features") != ["cpu-faer"] or manifest.get("no_default_features") is not True:
        raise protocol.ProtocolError("test build used the wrong feature graph request")
    if manifest.get("compiler_configuration") != {
        "observer_cfg": build.DISPATCH_OBSERVER_CFG,
        "rustflags": build.DISPATCH_RUSTFLAGS,
    }:
        raise protocol.ProtocolError("test build compiler configuration differs")
    target = manifest.get("target")
    if not isinstance(target, str) or not target:
        raise protocol.ProtocolError("test build manifest lacks the host target")
    expected_query = build.feature_query_command(
        target, package=package, requested_features=("cpu-faer",),
        no_default_features=True
    )
    if tuple(manifest.get("feature_query_argv", ())) != expected_query:
        raise protocol.ProtocolError("package feature query differs from the locked contract")
    feature_graph = manifest.get("feature_graph")
    if not isinstance(feature_graph, str) or hashlib.sha256(feature_graph.encode()).hexdigest() != manifest.get("feature_graph_sha256"):
        raise protocol.ProtocolError("package feature graph bytes differ from their digest")
    executable = pathlib.Path(str(manifest.get("executable", ""))).resolve(strict=True)
    if sha256_file(executable) != manifest.get("executable_sha256"):
        raise protocol.ProtocolError("test executable digest mismatch")
    if manifest.get("source_sha256") != expected_tree or manifest.get("lock_sha256") != expected_lock:
        raise protocol.ProtocolError("test build source/lock identity mismatch")
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or environment != build.dispatch_cargo_environment(
        path=environment.get("PATH", ""), home=environment.get("HOME", ""),
        cargo_home=environment.get("CARGO_HOME", ""),
        target_dir=environment.get("CARGO_TARGET_DIR", ""),
    ):
        raise protocol.ProtocolError("test build environment is not exact and sealed")
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict):
        raise protocol.ProtocolError("test build toolchain manifest is invalid")
    for tool in ("git", "cargo", "rustc"):
        identity = toolchain.get(tool)
        if not isinstance(identity, dict):
            raise protocol.ProtocolError(f"test build lacks exact {tool} identity")
        path = pathlib.Path(str(identity.get("path", ""))).resolve(strict=True)
        if sha256_file(path) != identity.get("sha256"):
            raise protocol.ProtocolError(f"test build {tool} digest mismatch")
    return executable


def validate_candidate_worktree(repository: pathlib.Path, candidate: str) -> None:
    if len(candidate) != 40 or any(character not in "0123456789abcdef" for character in candidate):
        raise protocol.ProtocolError("candidate must be a full lowercase Git SHA")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout
    if head != candidate or status:
        raise protocol.ProtocolError("evidence requires the clean immutable candidate worktree")


def run_bounded(
    argv: Sequence[str], *, cwd: pathlib.Path, environment: Mapping[str, str], deadline: int,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(argv), cwd=cwd, env=dict(environment), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=deadline)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        killed = False
        try:
            stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            killed = True
            stdout, stderr = process.communicate()
        raise ExecutionFailure(
            "validity inconclusive: deadline-exceeded", kind="DeadlineExceeded",
            stdout=stdout, stderr=stderr,
            termination={
                "term_sent": True, "grace_seconds": TERMINATION_GRACE_SECONDS,
                "kill_sent": killed, "reaped": process.poll() is not None,
                "returncode": process.returncode,
            },
        ) from error
    return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)


def run_test_executable(
    executable: pathlib.Path, filter_name: str, *, repository: pathlib.Path,
    evidence_root: pathlib.Path, environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    runtime_environment = _validate_runtime_environment(environment)
    runtime_environment[EVIDENCE_ENVIRONMENT_KEY] = str(evidence_root.resolve())
    argv = (str(executable.resolve()), filter_name, "--nocapture")
    result = run_bounded(
        argv, cwd=repository, environment=runtime_environment, deadline=TEST_DEADLINE_SECONDS
    )
    if result.returncode != 0:
        raise ExecutionFailure(
            f"direct test executable failed: {filter_name}", kind="NonzeroExit",
            stdout=result.stdout, stderr=result.stderr,
            termination={"reaped": True, "returncode": result.returncode},
        )
    return result


def validate_bench_build_manifest(
    manifest: Mapping[str, Any], *, owner: str, candidate: str,
    repository: pathlib.Path, common_lock: pathlib.Path,
) -> pathlib.Path:
    expected = build.CHARACTERIZATION_BENCH_COMMANDS[owner]
    package = "tenferro-cpu" if owner == "cpu" else "tenferro-ad"
    if (
        manifest.get("validity_state") != "COMPLETE"
        or manifest.get("candidate") != candidate
        or manifest.get("package") != package
        or manifest.get("bench") != ("numa_execution" if owner == "cpu" else "phase2e_characterization")
    ):
        raise protocol.ProtocolError("characterization bench candidate identity mismatch")
    if manifest.get("protocol_version") != protocol.PROTOCOL_VERSION or manifest.get(
        "protocol_sha256"
    ) != sha256_file(repository / "scripts/phase2e_protocol.py"):
        raise protocol.ProtocolError("characterization bench protocol identity mismatch")
    tree = subprocess.run(
        ("git", "ls-tree", "-r", "-z", "--full-tree", candidate), cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout
    tree_digest = hashlib.sha256(tree.encode()).hexdigest()
    expected_sources = {
        relative: sha256_file(repository / relative) for relative in build.TASK7_SOURCE_PATHS
    }
    if (
        manifest.get("candidate_tree_sha256") != tree_digest
        or manifest.get("source_sha256") != tree_digest
        or manifest.get("source_inventory") != dict(sorted(expected_sources.items()))
    ):
        raise protocol.ProtocolError("characterization bench source identity mismatch")
    lock_digest = sha256_file(common_lock)
    if manifest.get("common_lock_sha256") != lock_digest or manifest.get("lock_sha256") != lock_digest:
        raise protocol.ProtocolError("characterization bench lock identity mismatch")
    if tuple(manifest.get("argv", ())) != expected:
        raise protocol.ProtocolError("characterization bench build argv differs from contract")
    if manifest.get("requested_features") != ["cpu-faer"] or manifest.get("no_default_features") is not True:
        raise protocol.ProtocolError("characterization bench feature graph differs from contract")
    executable = pathlib.Path(str(manifest.get("executable", ""))).resolve(strict=True)
    if sha256_file(executable) != manifest.get("executable_sha256"):
        raise protocol.ProtocolError("characterization bench executable digest mismatch")
    target = manifest.get("target")
    if not isinstance(target, str) or not target:
        raise protocol.ProtocolError("characterization bench target is invalid")
    query = build.feature_query_command(
        target, package=package, requested_features=("cpu-faer",), no_default_features=True,
    )
    if tuple(manifest.get("feature_query_argv", ())) != query:
        raise protocol.ProtocolError("characterization bench feature query differs")
    graph = manifest.get("feature_graph")
    if not isinstance(graph, str) or hashlib.sha256(graph.encode()).hexdigest() != manifest.get(
        "feature_graph_sha256"
    ):
        raise protocol.ProtocolError("characterization bench feature graph digest mismatch")
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or environment != protocol.cargo_environment(
        path=environment.get("PATH", ""), home=environment.get("HOME", ""),
        cargo_home=environment.get("CARGO_HOME", ""),
        target_dir=environment.get("CARGO_TARGET_DIR", ""),
    ):
        raise protocol.ProtocolError("characterization bench environment is not sealed")
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict):
        raise protocol.ProtocolError("characterization bench toolchain is invalid")
    for tool in ("git", "cargo", "rustc"):
        identity = toolchain.get(tool)
        if not isinstance(identity, dict):
            raise protocol.ProtocolError(f"characterization bench lacks {tool} identity")
        tool_path = pathlib.Path(str(identity.get("path", ""))).resolve(strict=True)
        if sha256_file(tool_path) != identity.get("sha256"):
            raise protocol.ProtocolError(f"characterization bench {tool} digest mismatch")
    return executable


def run_bench_row(
    executable: pathlib.Path, row_key: str, *, repository: pathlib.Path,
    environment: Mapping[str, str], criterion_home: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    affinity_file = str((criterion_home / "affinity.json").resolve())
    runtime_environment = _validate_runtime_environment(
        environment,
        criterion_home=str(criterion_home.resolve()),
        affinity_row=row_key,
        affinity_file=affinity_file,
    )
    argv = (str(executable.resolve()), row_key, "--bench", "--noplot")
    result = run_bounded(
        argv, cwd=repository, environment=runtime_environment,
        deadline=BENCH_ROW_DEADLINE_SECONDS,
    )
    if result.returncode != 0:
        raise ExecutionFailure(
            f"characterization bench row failed: {row_key}", kind="NonzeroExit",
            stdout=result.stdout, stderr=result.stderr,
            termination={"reaped": True, "returncode": result.returncode},
        )
    return result


def _write_new_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise protocol.ProtocolError(f"normative artifact already exists: {path}") from error


def capture_bench_row(
    executable: pathlib.Path, row_key: str, *, repository: pathlib.Path,
    environment: Mapping[str, str], criterion_home: pathlib.Path,
    evidence_root: pathlib.Path, hardware_skip: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one fresh Criterion row and preserve its normative output bytes."""
    row_id = row_key.replace("/", "__")
    if hardware_skip is not None:
        _validate_hardware_skip({"hardware_skip": hardware_skip})
        return {
            "row_id": row_id,
            "key": row_key,
            "hardware_skip": dict(hardware_skip),
            "latency_ns": None,
            "artifacts": {},
        }
    result = run_bench_row(
        executable, row_key, repository=repository, environment=environment,
        criterion_home=criterion_home,
    )
    estimates = list(criterion_home.glob("*/new/estimates.json"))
    if len(estimates) != 1:
        raise protocol.ProtocolError(
            f"row {row_key} produced {len(estimates)} Criterion estimate artifacts"
        )
    destination = evidence_root / "characterization" / "rows" / row_id
    affinity_path = criterion_home / "affinity.json"
    if not affinity_path.is_file():
        raise protocol.ProtocolError(f"row {row_key} lacks its actual fixture affinity artifact")
    affinity = json.loads(affinity_path.read_text(encoding="utf-8"))
    budget = int(row_key.split("/budget-", 1)[1].split("/", 1)[0])
    ownership = row_key.split("/", 1)[0]
    observations = affinity.get("observations")
    declared = affinity.get("declared_cpus")
    if (
        affinity.get("key") != row_key
        or affinity.get("ownership") != ownership
        or affinity.get("budget") != budget
        or affinity.get("worker_count") != budget
        or not isinstance(declared, list)
        or not isinstance(observations, list)
        or len(observations) != budget
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not all(type(value) is int and value >= 0 for value in item)
            for item in observations
        )
        or {item[0] for item in observations} != set(range(budget))
    ):
        raise protocol.ProtocolError(f"row {row_key} has invalid fixture affinity evidence")
    exact = ownership != "external-advisory"
    if affinity.get("guarantee") != (
        "ExactDeclared" if exact else "AdvisoryDeclared"
    ) or (exact and (not declared or any(item[1] not in declared for item in observations))):
        raise protocol.ProtocolError(f"row {row_key} fixture escaped exact placement")
    artifacts = {
        "stdout": (destination / "stdout.log", result.stdout.encode()),
        "stderr": (destination / "stderr.log", result.stderr.encode()),
        "criterion_estimates": (destination / "estimates.json", estimates[0].read_bytes()),
        "fixture_affinity": (destination / "affinity.json", affinity_path.read_bytes()),
    }
    for path, payload in artifacts.values():
        _write_new_bytes(path, payload)
    parsed = json.loads(artifacts["criterion_estimates"][1])
    try:
        mean = parsed["mean"]
        interval = mean["confidence_interval"]
        latency = {
            "point_estimate": float(mean["point_estimate"]),
            "lower_bound": float(interval["lower_bound"]),
            "upper_bound": float(interval["upper_bound"]),
            "confidence_level": float(interval["confidence_level"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise protocol.ProtocolError(f"row {row_key} has invalid Criterion mean data") from error
    if (
        not all(math.isfinite(value) for value in latency.values())
        or latency["confidence_level"] != 0.95
        or latency["lower_bound"] > latency["point_estimate"]
        or latency["point_estimate"] > latency["upper_bound"]
    ):
        raise protocol.ProtocolError(f"row {row_key} has invalid Criterion 95% CI")
    return {
        "row_id": row_id,
        "key": row_key,
        "hardware_skip": None,
        "latency_ns": latency,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, (path, _payload) in artifacts.items()
        },
    }


def atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise protocol.ProtocolError(f"JSON artifact is not an object: {path}")
    return value


def validate_terminal_evidence(
    evidence_root: pathlib.Path, *, candidate: str, repository: pathlib.Path,
    source_inventory: Sequence[Mapping[str, str]], common_lock: pathlib.Path,
) -> None:
    dispatch_path = evidence_root / "dispatch-gates/manifest.json"
    characterization_path = evidence_root / "characterization/manifest.json"
    dispatch = _read_json(dispatch_path)
    characterization = _read_json(characterization_path)
    protocol_digest = sha256_file(repository / "scripts/phase2e_protocol.py")
    tree = subprocess.run(
        ("git", "ls-tree", "-r", "-z", "--full-tree", candidate), cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout
    tree_digest = hashlib.sha256(tree.encode()).hexdigest()
    sources = [dict(item) for item in source_inventory]
    lock_digest = sha256_file(common_lock)
    for name, manifest in (("dispatch", dispatch), ("characterization", characterization)):
        expected = {
            "validity_state": "PASS", "candidate": candidate,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": protocol_digest,
            "candidate_tree_sha256": tree_digest,
            "source_inventory": sources, "common_lock_sha256": lock_digest,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise protocol.ProtocolError(f"terminal {name} manifest differs at {field}")
    if dispatch.get("row_count") != 47 or characterization.get("row_count") != 47:
        raise protocol.ProtocolError("terminal manifests must bind 47 composed rows")
    rows = characterization.get("rows")
    if not isinstance(rows, list) or len(rows) != 47:
        raise protocol.ProtocolError("terminal characterization row inventory is invalid")
    rows_by_key = {row.get("key"): row for row in rows if isinstance(row, dict)}
    for row in rows:
        if not isinstance(row, dict) or row.get("hardware_skip") is not None:
            raise protocol.ProtocolError("correctness rows must never hardware-skip")
        _validate_hardware_skip({"hardware_skip": row.get("affinity_hardware_skip")})
    cross_socket = characterization.get("cross_socket_locality")
    if not isinstance(cross_socket, dict):
        raise protocol.ProtocolError("cross-socket hardware validity is missing")
    _validate_hardware_skip(cross_socket)
    expected_files = {
        common_lock.resolve(), dispatch_path.resolve(), characterization_path.resolve()
    }
    for package, short in (("tenferro-cpu", "cpu"), ("tenferro-ad", "ad")):
        record = dispatch.get(short)
        if not isinstance(record, dict):
            raise protocol.ProtocolError(f"dispatch terminal lacks {short}")
        artifact = evidence_root / "dispatch-gates" / f"{short}-evidence.json"
        build_path = evidence_root / build.DISPATCH_BUILD_MANIFEST_PATHS[package]
        expected_files.update((artifact.resolve(), build_path.resolve()))
        if record.get("artifact") != str(artifact.resolve()) or record.get("sha256") != sha256_file(artifact):
            raise protocol.ProtocolError(f"dispatch {short} evidence digest mismatch")
        if record.get("build_manifest") != {
            "path": str(build_path.resolve()), "sha256": sha256_file(build_path)
        }:
            raise protocol.ProtocolError(f"dispatch {short} build manifest digest mismatch")
        build_manifest = _read_json(build_path)
        if record.get("executable_sha256") != build_manifest.get("executable_sha256"):
            raise protocol.ProtocolError(f"dispatch {short} executable digest mismatch")
        for stream in ("stdout", "stderr"):
            path = evidence_root / "dispatch-gates" / f"{short}-{stream}.log"
            expected_files.add(path.resolve())
            if record.get(stream) != {"path": str(path.resolve()), "sha256": sha256_file(path)}:
                raise protocol.ProtocolError(f"dispatch {short} {stream} digest mismatch")
    expected_latency = (canonical_keys()[0] | canonical_keys()[1]) - {
        "external-no-outer/budget-2/U-O", "external-no-inner/budget-2/U-I"
    }
    latency_rows = characterization.get("latency_rows")
    if (
        characterization.get("latency_row_count") != 45
        or not isinstance(latency_rows, list)
        or {row.get("key") for row in latency_rows if isinstance(row, dict)} != expected_latency
    ):
        raise protocol.ProtocolError("terminal characterization latency inventory mismatch")
    for record in latency_rows:
        key = record["key"]
        row_id = key.replace("/", "__")
        if record.get("row_id") != row_id:
            raise protocol.ProtocolError(f"latency row id mismatch: {key}")
        artifact_records = record.get("artifacts")
        if not isinstance(artifact_records, dict):
            raise protocol.ProtocolError(f"latency artifacts missing: {key}")
        hardware_skip = record.get("hardware_skip")
        _validate_hardware_skip(record)
        expected_skip = rows_by_key[key].get("affinity_hardware_skip")
        if hardware_skip != expected_skip:
            raise protocol.ProtocolError(f"latency {key} hardware skip mismatch")
        if hardware_skip is not None:
            if record.get("latency_ns") is not None or artifact_records:
                raise protocol.ProtocolError(f"skipped latency {key} fabricated measurement data")
            continue
        row_root = evidence_root / "characterization" / "rows" / row_id
        for name, filename in (
            ("stdout", "stdout.log"), ("stderr", "stderr.log"),
            ("criterion_estimates", "estimates.json"),
            ("fixture_affinity", "affinity.json"),
        ):
            path = row_root / filename
            expected_files.add(path.resolve())
            if artifact_records.get(name) != {
                "path": str(path.resolve()), "sha256": sha256_file(path)
            }:
                raise protocol.ProtocolError(f"latency {key} {name} digest mismatch")
        estimates = _read_json(row_root / "estimates.json")
        mean = estimates.get("mean", {})
        interval = mean.get("confidence_interval", {}) if isinstance(mean, dict) else {}
        if record.get("latency_ns") != {
            "point_estimate": float(mean.get("point_estimate")),
            "lower_bound": float(interval.get("lower_bound")),
            "upper_bound": float(interval.get("upper_bound")),
            "confidence_level": float(interval.get("confidence_level")),
        }:
            raise protocol.ProtocolError(f"latency {key} parsed estimate mismatch")
    for owner in ("cpu", "ad"):
        build_path = evidence_root / build.CHARACTERIZATION_BUILD_MANIFEST_PATHS[owner]
        expected_files.add(build_path.resolve())
        if characterization.get("bench_build_manifests", {}).get(owner) != {
            "path": str(build_path.resolve()), "sha256": sha256_file(build_path)
        }:
            raise protocol.ProtocolError(f"characterization {owner} build digest mismatch")
    actual_files = {path.resolve() for path in evidence_root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise protocol.ProtocolError("terminal evidence recursive file inventory mismatch")


def _run_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=pathlib.Path)
    parser.add_argument("--evidence-root", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--common-lock", required=True, type=pathlib.Path)
    parser.add_argument("--scratch-root", required=True, type=pathlib.Path)
    parser.add_argument("--path", required=True)
    parser.add_argument("--home", required=True, type=pathlib.Path)
    parser.add_argument("--cargo-home", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    validate_candidate_worktree(args.repository, args.candidate)
    scratch_root = validate_external_scratch_root(
        args.repository, args.evidence_root, args.scratch_root
    )
    source_inventory = validate_source_contract(args.repository)
    evidence_root = args.evidence_root.resolve()
    if evidence_root.exists():
        if any(evidence_root.iterdir()):
            raise protocol.ProtocolError("Task 7 evidence root must be initially empty")
    else:
        evidence_root.mkdir(mode=0o700, parents=True)
    common_lock = args.common_lock.resolve(strict=True)
    common_destination = evidence_root / build.LOCK_PATHS["common"]
    common_destination.parent.mkdir(parents=True)
    with common_lock.open("rb") as source:
        build._write_new_regular(common_destination, source.read())
    build.build_dispatch_and_characterization_artifacts(
        repository=args.repository,
        evidence_root=evidence_root,
        scratch_root=scratch_root,
        candidate=args.candidate,
        path=args.path,
        home=args.home,
        cargo_home=args.cargo_home,
    )
    manifests = {}
    runtime_environment = protocol.runtime_environment(
        path=args.path, home=str(args.home.resolve(strict=True))
    )
    for package, short, filter_name in (
        ("tenferro-cpu", "cpu", CPU_FILTER), ("tenferro-ad", "ad", AD_FILTER)
    ):
        manifest = _read_json(evidence_root / build.DISPATCH_BUILD_MANIFEST_PATHS[package])
        executable = validate_test_build_manifest(
            manifest, package=package, candidate=args.candidate,
            repository=args.repository, common_lock=common_destination,
        )
        result = run_test_executable(
            executable, filter_name, repository=args.repository,
            evidence_root=evidence_root, environment=runtime_environment,
        )
        artifact = evidence_root / "dispatch-gates" / f"{short}-evidence.json"
        stdout_path = evidence_root / "dispatch-gates" / f"{short}-stdout.log"
        stderr_path = evidence_root / "dispatch-gates" / f"{short}-stderr.log"
        _write_new_bytes(stdout_path, result.stdout.encode())
        _write_new_bytes(stderr_path, result.stderr.encode())
        build_manifest_path = evidence_root / build.DISPATCH_BUILD_MANIFEST_PATHS[package]
        manifests[short] = {
            "artifact": str(artifact.resolve()), "sha256": sha256_file(artifact),
            "stdout": {"path": str(stdout_path.resolve()), "sha256": sha256_file(stdout_path)},
            "stderr": {"path": str(stderr_path.resolve()), "sha256": sha256_file(stderr_path)},
            "build_manifest": {
                "path": str(build_manifest_path.resolve()),
                "sha256": sha256_file(build_manifest_path),
            },
            "executable_sha256": manifest["executable_sha256"],
        }
    cpu = _read_json(pathlib.Path(manifests["cpu"]["artifact"]))
    ad = _read_json(pathlib.Path(manifests["ad"]["artifact"]))
    characterization = compose_characterization(cpu, ad)
    available_cpus, usable_numa_nodes = hardware_availability()
    attach_hardware_validity(
        characterization,
        available_cpus=available_cpus,
        usable_numa_nodes=usable_numa_nodes,
    )
    bench_digests = {}
    bench_manifests = {}
    latency_rows = []
    criterion_root = scratch_root / "task7-criterion"
    criterion_root.mkdir(mode=0o700, exist_ok=False)
    try:
        for owner, surfaces in (("cpu", {"D-N", "D-D", "G-O"}), ("ad", {"E-N", "E-D"})):
            manifest = _read_json(
                evidence_root / build.CHARACTERIZATION_BUILD_MANIFEST_PATHS[owner]
            )
            executable = validate_bench_build_manifest(
                manifest, owner=owner, candidate=args.candidate,
                repository=args.repository, common_lock=common_destination,
            )
            for row in sorted(characterization["rows"], key=lambda value: value["key"]):
                if row["surface"] in surfaces:
                    row_scratch = criterion_root / row["key"].replace("/", "__")
                    row_scratch.mkdir(mode=0o700)
                    try:
                        latency_rows.append(capture_bench_row(
                            executable, row["key"], repository=args.repository,
                            environment=runtime_environment,
                            criterion_home=row_scratch, evidence_root=evidence_root,
                            hardware_skip=row["affinity_hardware_skip"],
                        ))
                    finally:
                        shutil.rmtree(row_scratch, ignore_errors=True)
            bench_digests[owner] = manifest["executable_sha256"]
            bench_path = evidence_root / build.CHARACTERIZATION_BUILD_MANIFEST_PATHS[owner]
            bench_manifests[owner] = {
                "path": str(bench_path.resolve()), "sha256": sha256_file(bench_path)
            }
    finally:
        shutil.rmtree(criterion_root, ignore_errors=True)
    if len(latency_rows) != 45 or len({row["key"] for row in latency_rows}) != 45:
        raise protocol.ProtocolError("characterization must preserve exactly 45 latency rows")
    characterization["bench_executable_sha256"] = bench_digests
    characterization["latency_row_count"] = len(latency_rows)
    characterization["latency_rows"] = latency_rows
    characterization["candidate"] = args.candidate
    characterization["protocol_version"] = protocol.PROTOCOL_VERSION
    characterization["protocol_sha256"] = sha256_file(
        args.repository / "scripts/phase2e_protocol.py"
    )
    characterization["candidate_tree_sha256"] = _read_json(
        evidence_root / build.CHARACTERIZATION_BUILD_MANIFEST_PATHS["cpu"]
    )["candidate_tree_sha256"]
    characterization["source_inventory"] = source_inventory
    characterization["common_lock_sha256"] = sha256_file(common_destination)
    characterization["bench_build_manifests"] = bench_manifests
    atomic_write_json(
        evidence_root / "dispatch-gates/manifest.json",
        {
            "validity_state": "PASS", "candidate": args.candidate,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": sha256_file(args.repository / "scripts/phase2e_protocol.py"),
            "candidate_tree_sha256": characterization["candidate_tree_sha256"],
            "source_inventory": source_inventory,
            "common_lock_sha256": sha256_file(common_destination),
            "row_count": characterization["row_count"], **manifests,
        },
    )
    atomic_write_json(evidence_root / "characterization/manifest.json", characterization)
    validate_terminal_evidence(
        evidence_root, candidate=args.candidate, repository=args.repository,
        source_inventory=source_inventory, common_lock=common_destination,
    )
    return 0


def _own_inconclusive(
    evidence_root: pathlib.Path, candidate: str, error: protocol.ProtocolError,
) -> None:
    kind = getattr(error, "kind", "ProtocolError")
    termination = getattr(error, "termination", {})
    failure = {
        "validity_state": "INCONCLUSIVE", "candidate": candidate,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "failure": {
            "kind": kind,
            "message": str(error),
            "termination": termination,
        },
    }
    if hasattr(error, "stdout") and hasattr(error, "stderr"):
        for name, payload in (("stdout", error.stdout), ("stderr", error.stderr)):
            path = evidence_root / "failure" / f"{name}.log"
            if not path.exists():
                _write_new_bytes(path, payload.encode())
            failure["failure"][name] = {
                "path": str(path.resolve()), "sha256": sha256_file(path)
            }
    atomic_write_json(evidence_root / "dispatch-gates/manifest.json", failure)
    atomic_write_json(evidence_root / "characterization/manifest.json", failure)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else __import__("sys").argv[1:])
    try:
        return _run_main(arguments)
    except protocol.ProtocolError as error:
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--evidence-root", type=pathlib.Path)
        probe.add_argument("--candidate", default="UNKNOWN")
        known, _unknown = probe.parse_known_args(arguments)
        if known.evidence_root is not None:
            root = known.evidence_root.resolve()
            if (root / build.LOCK_PATHS["common"]).is_file():
                _own_inconclusive(root, known.candidate, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
