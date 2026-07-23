#!/usr/bin/env python3
"""Build and validate provenance-bound Phase 2E benchmark binaries.

The module deliberately treats source trees, lock files, Cargo inputs, and the
resulting executable as one build identity.  All external commands run in a
new process group with fixed deadlines so a failed build cannot outlive the
evidence orchestrator.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import phase2e_protocol as protocol


_SEAL_NAMES = ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
INHERITED_EXECUTABLE_SEALS = (
    None
    if not hasattr(fcntl, "F_GET_SEALS")
    or any(not hasattr(fcntl, name) for name in _SEAL_NAMES)
    else sum(getattr(fcntl, name) for name in _SEAL_NAMES)
)


IMPLEMENTATION_BASELINE = "85855e272b1495611deb601a9ee06f3546772c3c"
HARNESS_COMMIT = "4471d6145c4d8793de3a96f8d99400c24ca8c6d1"
OLD_STRIDED = "10fc972d3c0f8cdfd4ecb45d21d815aebfd7d1f2"
COMMON_STRIDED = "6b0b4a46b7dd9a9ea1677a0d596c0b4adab1acbc"

BENCH_COMMAND = (
    "cargo",
    "bench",
    "--locked",
    "--no-run",
    "-p",
    "tenferro-ad",
    "--bench",
    "eager_dispatch_baseline",
    "--no-default-features",
    "--features",
    "cpu-faer",
)
REQUESTED_FEATURES = ("cpu-faer",)
DISPATCH_OBSERVER_CFG = "tenferro_phase2e_operation_observe"
DISPATCH_RUSTFLAGS = (
    "--check-cfg=cfg(tenferro_phase2e_operation_observe) "
    "--cfg=tenferro_phase2e_operation_observe"
)
TASK7_SOURCE_PATHS = (
    "Cargo.toml",
    "crates/tenferro-cpu/Cargo.toml",
    "crates/tenferro-ad/Cargo.toml",
    "crates/tenferro-cpu/src/backend.rs",
    "crates/tenferro-cpu/src/engine.rs",
    "crates/tenferro-cpu/src/elementwise.rs",
    "crates/tenferro-cpu/src/domain_executor.rs",
    "crates/tenferro-cpu/src/provider.rs",
    "crates/tenferro-cpu/src/exec_session.rs",
    "crates/tenferro-cpu/src/dot_runtime.rs",
    "crates/tenferro-cpu/src/phase2e_observe.rs",
    "crates/tenferro-cpu/src/tests/phase2e.rs",
    "crates/tenferro-cpu/benches/numa_execution.rs",
    "crates/tenferro-ad/src/eager.rs",
    "crates/tenferro-ad/src/eager_backend.rs",
    "crates/tenferro-ad/src/eager/tests/phase2e.rs",
    "crates/tenferro-ad/benches/phase2e_characterization.rs",
    "scripts/phase2e_protocol.py",
    "scripts/phase2e_build.py",
    "scripts/run_phase2e_gates.py",
)

DISPATCH_TEST_COMMANDS = MappingProxyType(
    {
        package: (
            "cargo",
            "test",
            "--locked",
            "--no-run",
            "-p",
            package,
            "--lib",
            "--no-default-features",
            "--features",
            ",".join(REQUESTED_FEATURES),
            "--message-format=json",
        )
        for package in ("tenferro-cpu", "tenferro-ad")
    }
)


def dispatch_cargo_environment(
    *, path: str, home: str, cargo_home: str, target_dir: str
) -> dict[str, str]:
    """Construct the exact sealed Cargo environment for dispatch evidence."""
    environment = protocol.cargo_environment(
        path=path,
        home=home,
        cargo_home=cargo_home,
        target_dir=target_dir,
    )
    environment["RUSTFLAGS"] = DISPATCH_RUSTFLAGS
    return environment
CHARACTERIZATION_BENCH_COMMANDS = MappingProxyType(
    {
        "cpu": (
            "cargo", "bench", "--locked", "--no-run", "-p", "tenferro-cpu",
            "--bench", "numa_execution", "--no-default-features", "--features",
            "cpu-faer", "--message-format=json",
        ),
        "ad": (
            "cargo", "bench", "--locked", "--no-run", "-p", "tenferro-ad",
            "--bench", "phase2e_characterization", "--no-default-features", "--features",
            "cpu-faer", "--message-format=json",
        ),
    }
)
DISPATCH_TEST_DEADLINE_SECONDS = 120
CHARACTERIZATION_ROW_DEADLINE_SECONDS = 30
DISPATCH_TERMINATION_GRACE_SECONDS = 5


class Task7BuildFailure(protocol.ProtocolError):
    def __init__(self, message: str, result: "CommandResult") -> None:
        super().__init__(message)
        self.kind = "NonzeroExit"
        self.stdout = result.stdout
        self.stderr = result.stderr
        self.termination = {"reaped": True, "returncode": result.returncode}
DISPATCH_BUILD_MANIFEST_PATHS = MappingProxyType(
    {
        "tenferro-cpu": pathlib.Path("dispatch-gates/cpu-test-build.json"),
        "tenferro-ad": pathlib.Path("dispatch-gates/ad-test-build.json"),
    }
)
CHARACTERIZATION_BUILD_MANIFEST_PATHS = MappingProxyType(
    {
        "cpu": pathlib.Path("characterization/cpu-bench-build.json"),
        "ad": pathlib.Path("characterization/eager-bench-build.json"),
    }
)


def select_cargo_executable(messages: str, package: str, *, bench: str | None = None) -> pathlib.Path:
    """Select the sole package-owned executable from Cargo JSON messages."""
    matches: list[pathlib.Path] = []
    for line in messages.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact" or not message.get("executable"):
            continue
        target = message.get("target", {})
        package_id = str(message.get("package_id", ""))
        owns_artifact = package_id.startswith(f"{package} ") or re.search(
            rf"(?:/|#){re.escape(package)}(?:#|@|$)", package_id
        ) is not None
        if not owns_artifact:
            continue
        kinds = target.get("kind", [])
        if bench is None:
            if "lib" not in kinds or not message.get("profile", {}).get("test"):
                continue
        elif target.get("name") != bench or "bench" not in kinds:
            continue
        matches.append(pathlib.Path(message["executable"]))
    if len(matches) != 1:
        raise protocol.ProtocolError(
            f"expected one Cargo executable for {package}, found {len(matches)}"
        )
    return matches[0]


def dispatch_build_provenance(
    *, package: str, candidate: str, source_sha256: str, lock_sha256: str,
    feature_graph_sha256: str, argv: tuple[str, ...], environment: Mapping[str, str],
    executable: pathlib.Path, target: str, toolchain: Mapping[str, str],
    protocol_sha256: str, source_inventory: Mapping[str, str], feature_graph: str,
) -> dict[str, Any]:
    """Bind one dispatch executable to candidate source, lock, graph, and build inputs."""
    expected = DISPATCH_TEST_COMMANDS.get(package)
    if expected is None or tuple(argv) != expected:
        raise protocol.ProtocolError("dispatch test build argv differs from the locked contract")
    sealed = dispatch_cargo_environment(
        path=environment.get("PATH", ""),
        home=environment.get("HOME", ""),
        cargo_home=environment.get("CARGO_HOME", ""),
        target_dir=environment.get("CARGO_TARGET_DIR", ""),
    )
    if dict(environment) != sealed:
        raise protocol.ProtocolError("dispatch test build environment is not sealed")
    executable = executable.resolve(strict=True)
    return {
        "validity_state": "COMPLETE",
        "candidate": candidate,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "package": package,
        "source_sha256": source_sha256,
        "candidate_tree_sha256": source_sha256,
        "source_inventory": dict(sorted(source_inventory.items())),
        "lock_sha256": lock_sha256,
        "common_lock_sha256": lock_sha256,
        "feature_graph_sha256": feature_graph_sha256,
        "feature_graph": feature_graph,
        "feature_query_argv": list(
            feature_query_command(
                target,
                package=package,
                requested_features=REQUESTED_FEATURES,
                no_default_features=True,
            )
        ),
        "requested_features": list(REQUESTED_FEATURES),
        "compiler_configuration": {
            "observer_cfg": DISPATCH_OBSERVER_CFG,
            "rustflags": DISPATCH_RUSTFLAGS,
        },
        "no_default_features": True,
        "target": target,
        "toolchain": dict(toolchain),
        "profile": "test",
        "argv": list(argv),
        "environment": dict(sorted(environment.items())),
        "executable": str(executable),
        "executable_sha256": protocol.sha256_file(executable),
    }


def _build_task7_artifacts(
    *, repository: pathlib.Path, evidence_root: pathlib.Path, scratch_root: pathlib.Path,
    candidate: str, path: str, home: pathlib.Path, cargo_home: pathlib.Path,
    kinds: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Build exactly the requested candidate-owned Task 7 executable family."""
    if not kinds or not kinds <= {"dispatch", "characterization"}:
        raise protocol.ProtocolError("Task 7 build kind is invalid")
    repository = pathlib.Path(repository).resolve(strict=True)
    evidence_root = pathlib.Path(evidence_root).resolve(strict=True)
    scratch_root = pathlib.Path(scratch_root).resolve(strict=True)
    _validate_commit(candidate, "dispatch candidate")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout
    if head != candidate or status:
        raise protocol.ProtocolError("dispatch builds require the clean immutable candidate")
    common_lock = evidence_root / LOCK_PATHS["common"]
    lock_payload = _read_regular_bytes(common_lock)
    installed_lock = repository / "Cargo.lock"
    if installed_lock.exists() and _read_regular_bytes(installed_lock) != lock_payload:
        raise protocol.ProtocolError("candidate Cargo.lock differs from the root-owned common lock")
    if not installed_lock.exists():
        _write_new_regular(installed_lock, lock_payload)
    validate_controlled_cargo_home(cargo_home)
    tools = resolve_toolchain(path)
    validate_resolved_toolchain(tools)
    probe_environment = protocol.runtime_environment(
        path=path, home=str(pathlib.Path(home).resolve(strict=True))
    )
    cargo_probe = run_bounded_command(
        (str(tools.cargo.path), "--version"), cwd=repository,
        environment=probe_environment, deadline_seconds=QUERY_DEADLINE_SECONDS,
        executable_identity=tools.cargo,
    )
    rustc_probe = run_bounded_command(
        (str(tools.rustc.path), "-vV"), cwd=repository,
        environment=probe_environment, deadline_seconds=QUERY_DEADLINE_SECONDS,
        executable_identity=tools.rustc,
    )
    if cargo_probe.returncode != 0 or rustc_probe.returncode != 0:
        failed = cargo_probe if cargo_probe.returncode != 0 else rustc_probe
        raise Task7BuildFailure("Task 7 toolchain probe failed", failed)
    target = _rustc_host(rustc_probe.stdout, "Task 7")
    tree = subprocess.run(
        ("git", "ls-tree", "-r", "-z", "--full-tree", "HEAD"), cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout
    source_sha256 = sha256_bytes(tree.encode())
    lock_sha256 = sha256_bytes(lock_payload)
    protocol_sha256 = protocol.sha256_file(repository / "scripts/phase2e_protocol.py")
    source_inventory = {
        relative: protocol.sha256_file(repository / relative)
        for relative in TASK7_SOURCE_PATHS
    }
    toolchain = _toolchain_manifest(
        tools, cargo_probe.stdout.strip(), rustc_probe.stdout.strip()
    )
    specs = ([
        ("dispatch", package, None, command, DISPATCH_BUILD_MANIFEST_PATHS[package])
        for package, command in DISPATCH_TEST_COMMANDS.items()
    ] if "dispatch" in kinds else []) + ([
        (
            "characterization", "tenferro-cpu" if owner == "cpu" else "tenferro-ad",
            "numa_execution" if owner == "cpu" else "phase2e_characterization",
            command, CHARACTERIZATION_BUILD_MANIFEST_PATHS[owner],
        )
        for owner, command in CHARACTERIZATION_BENCH_COMMANDS.items()
    ] if "characterization" in kinds else [])
    manifests: dict[str, dict[str, Any]] = {}
    for index, (kind, package, bench, command, relative) in enumerate(specs):
        target_dir = scratch_root / f"task7-{kind}-target-{index}-{package}"
        try:
            target_dir.mkdir(mode=0o700)
        except FileExistsError as error:
            raise protocol.ProtocolError(f"Task 7 target is not fresh: {target_dir}") from error
        environment_constructor = (
            dispatch_cargo_environment
            if kind == "dispatch"
            else protocol.cargo_environment
        )
        environment = environment_constructor(
            path=path,
            home=str(pathlib.Path(home).resolve(strict=True)),
            cargo_home=str(pathlib.Path(cargo_home).resolve(strict=True)),
            target_dir=str(target_dir),
        )
        requested_features = REQUESTED_FEATURES
        feature_argv = feature_query_command(
            target, package=package, requested_features=requested_features,
            no_default_features=True,
        )
        feature_result = run_bounded_command(
            feature_argv, cwd=repository, environment=environment,
            deadline_seconds=QUERY_DEADLINE_SECONDS,
        )
        if feature_result.returncode != 0:
            raise Task7BuildFailure(f"Task 7 feature query failed for {package}", feature_result)
        build_result = run_bounded_command(
            command, cwd=repository, environment=environment,
            deadline_seconds=BUILD_DEADLINE_SECONDS,
        )
        if build_result.returncode != 0:
            raise Task7BuildFailure(f"Task 7 executable build failed for {package}", build_result)
        executable = select_cargo_executable(build_result.stdout, package, bench=bench)
        manifest = dispatch_build_provenance(
            package=package, candidate=candidate, source_sha256=source_sha256,
            lock_sha256=lock_sha256,
            feature_graph_sha256=sha256_bytes(feature_result.stdout.encode()),
            argv=command, environment=environment, executable=executable,
            target=target, toolchain=toolchain, protocol_sha256=protocol_sha256,
            source_inventory=source_inventory, feature_graph=feature_result.stdout,
        ) if kind == "dispatch" else {
            "validity_state": "COMPLETE", "candidate": candidate, "package": package,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256,
            "bench": bench, "source_sha256": source_sha256,
            "candidate_tree_sha256": source_sha256,
            "source_inventory": dict(sorted(source_inventory.items())),
            "lock_sha256": lock_sha256, "common_lock_sha256": lock_sha256,
            "feature_graph_sha256": sha256_bytes(feature_result.stdout.encode()),
            "feature_graph": feature_result.stdout,
            "requested_features": list(requested_features), "no_default_features": True,
            "feature_query_argv": list(feature_argv), "target": target,
            "toolchain": toolchain, "profile": "bench", "argv": list(command),
            "environment": environment, "executable": str(executable.resolve(strict=True)),
            "executable_sha256": protocol.sha256_file(executable),
        }
        destination = evidence_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        protocol.atomic_write_json(destination, manifest)
        manifests[str(relative)] = manifest
    return manifests


def build_dispatch_artifacts(**kwargs) -> dict[str, dict[str, Any]]:
    """Build only the two dispatch-test executables."""
    return _build_task7_artifacts(**kwargs, kinds=frozenset({"dispatch"}))


def build_characterization_artifacts(**kwargs) -> dict[str, dict[str, Any]]:
    """Build only the two characterization benchmark executables."""
    return _build_task7_artifacts(**kwargs, kinds=frozenset({"characterization"}))


def build_dispatch_and_characterization_artifacts(
    **kwargs,
) -> dict[str, dict[str, Any]]:
    """Preserve the owning CLI's atomic all-four Task 7 build behavior."""
    return _build_task7_artifacts(
        **kwargs, kinds=frozenset({"dispatch", "characterization"})
    )

INVARIANT_FIELDS = frozenset(
    {
        "protocol_version",
        "toolchain",
        "target",
        "profile",
        "requested_features",
        "provider",
        "benchmark_sha256",
        "benchmark_stanza_sha256",
        "command_template",
        "config_chain_sha256",
    }
)
ROLE_FIELDS = frozenset(
    {
        "role",
        "head",
        "tracked_tree_sha256",
        "resolved_features_sha256",
        "lock_sha256",
        "worktree",
        "target_dir",
        "executable",
        "executable_sha256",
    }
)
AUDIT_FIELDS = frozenset(
    {
        "validity_state",
        "source_delta",
        "commands",
        "environment",
        "cargo_config_chain",
    }
)

ROOT_CARGO_PATH = pathlib.Path("Cargo.toml")
AD_CARGO_PATH = pathlib.Path("crates/tenferro-ad/Cargo.toml")
BENCH_SOURCE_PATH = pathlib.Path(
    "crates/tenferro-ad/benches/eager_dispatch_baseline.rs"
)
FROZEN_HARNESS_PATHS = (AD_CARGO_PATH, BENCH_SOURCE_PATH)
STRIDED_DEPENDENCIES = (
    "strided-view",
    "strided-traits",
    "strided-perm",
    "strided-kernel",
    "strided-einsum2",
)
BASELINE_DELTAS = ("frozen-benchmark-harness", "five-strided-pins")
BENCH_STANZA = (
    "[[bench]]\n"
    'name = "eager_dispatch_baseline"\n'
    "harness = false\n\n"
)

_LOCK_PATHS = {
    "direct": pathlib.Path("builds/locks/direct-current-main.Cargo.lock"),
    "common": pathlib.Path("builds/locks/common.Cargo.lock"),
    "direct-probe": pathlib.Path(
        "builds/locks/direct-current-main-probe.Cargo.lock"
    ),
    "common-probe": pathlib.Path("builds/locks/common-probe.Cargo.lock"),
}
LOCK_PATHS = MappingProxyType(_LOCK_PATHS)

_BUILD_MANIFEST_PATHS = {
    "direct-current-main-baseline": pathlib.Path(
        "builds/direct-current-main-baseline.json"
    ),
    "common-lock-normalized-baseline": pathlib.Path(
        "builds/common-lock-normalized-baseline.json"
    ),
    "candidate": pathlib.Path("builds/candidate.json"),
}
BUILD_MANIFEST_PATHS = MappingProxyType(_BUILD_MANIFEST_PATHS)

_PROBE_BUILD_MANIFEST_PATHS = {
    "direct-current-main-baseline": pathlib.Path(
        "builds/probes/direct-current-main-baseline.json"
    ),
    "common-lock-normalized-baseline": pathlib.Path(
        "builds/probes/common-lock-normalized-baseline.json"
    ),
    "candidate": pathlib.Path("builds/probes/candidate.json"),
}
PROBE_BUILD_MANIFEST_PATHS = MappingProxyType(_PROBE_BUILD_MANIFEST_PATHS)

LOCK_COMMAND = ("cargo", "generate-lockfile")
METADATA_COMMAND = ("cargo", "metadata", "--locked", "--format-version", "1")
QUERY_DEADLINE_SECONDS = 300
BUILD_DEADLINE_SECONDS = 1800
TERMINATION_GRACE_SECONDS = 5

ALLOCATION_PROBE_SOURCE_ROOT = pathlib.Path("scripts/phase2e/allocation-probe")
ALLOCATION_PROBE_TEMPLATE = pathlib.Path("Cargo.toml.in")
ALLOCATION_PROBE_SOURCES = (pathlib.Path("src/main.rs"), pathlib.Path("src/tests.rs"))
ALLOCATION_PROBE_BINARY = "phase2e-allocation-probe"
ALLOCATION_PROBE_ROOT_PLACEHOLDER = "__TENFERRO_REPOSITORY_ROOT__"
ALLOCATION_PROBE_FMT_DEADLINE_SECONDS = 300
ALLOCATION_PROBE_COMMAND_DEADLINE_SECONDS = 1800
ALLOCATION_PROBE_LIST_DEADLINE_SECONDS = 30

_ROLE_SOURCE_DELTAS = {
    "direct-current-main-baseline": ("frozen-benchmark-harness",),
    "common-lock-normalized-baseline": (
        "frozen-benchmark-harness",
        "five-strided-pins",
    ),
    "candidate": (),
}
_BUILD_ROLES = tuple(_ROLE_SOURCE_DELTAS)


@dataclasses.dataclass(frozen=True)
class WorktreeSpec:
    """One dedicated build worktree and its starting commit."""

    role: str
    path: pathlib.Path
    start_commit: str


@dataclasses.dataclass(frozen=True)
class AllocationProbeBuildSpec:
    """One role-bound external allocation-probe build identity."""

    role: str
    repository: pathlib.Path
    lock_name: str
    manifest_path: pathlib.Path
    profile: str


@dataclasses.dataclass(frozen=True)
class CommandSpec:
    """One exact Cargo command and its wall-clock deadline."""

    name: str
    argv: tuple[str, ...]
    deadline_seconds: int

    def to_manifest(self) -> dict[str, Any]:
        """Return the deterministic JSON form recorded in a build manifest."""
        return {
            "name": self.name,
            "argv": list(self.argv),
            "deadline_seconds": self.deadline_seconds,
        }


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Complete captured outcome of one bounded child process."""

    argv: tuple[str, ...]
    cwd: str
    environment: dict[str, str]
    deadline_seconds: int
    returncode: int | None
    stdout: str
    stderr: str
    validity_state: str
    failure_reason: str | None
    terminated: bool
    killed: bool
    inherited_descriptors: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class AllocationProbeVerification:
    """Immutable source, lock, binary, and case-inventory proof."""

    template_sha256: str
    source_sha256: dict[str, str]
    generated_manifest_sha256: str
    generated_source_sha256: dict[str, str]
    lock_sha256: str
    binary_sha256: str
    case_inventory: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class WorktreeProof:
    """Source and configuration identity observed around one build."""

    head: str
    tracked_tree_sha256: str
    benchmark_sha256: str
    benchmark_stanza_sha256: str
    cargo_config_chain: tuple[dict[str, str], ...]


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    """Caller-owned roots and identity for one three-role build set."""

    repository: pathlib.Path
    evidence_root: pathlib.Path
    scratch_root: pathlib.Path
    candidate_commit: str
    path: str
    home: pathlib.Path
    cargo_home: pathlib.Path
    expected_toolchain: ResolvedToolchain | None = None
    expected_host_target: str | None = None


@dataclasses.dataclass(frozen=True)
class BuildSetResult:
    """Terminal validity and manifests from one build-set invocation."""

    validity_state: str
    manifests: dict[str, dict[str, Any]]
    failure: CommandResult | None


@dataclasses.dataclass(frozen=True)
class ResolvedTool:
    """Canonical executable identity frozen before evidence mutation."""

    name: str
    path: pathlib.Path
    sha256: str


@dataclasses.dataclass(frozen=True)
class ResolvedToolchain:
    """Minimal controlled PATH and the three required executable identities."""

    path: str
    git: ResolvedTool
    cargo: ResolvedTool
    rustc: ResolvedTool


def resolve_toolchain(path: str) -> ResolvedToolchain:
    """Resolve a canonical minimal PATH containing exactly the required tools."""
    if not isinstance(path, str) or not path:
        raise protocol.ProtocolError("controlled PATH must be a nonempty string")
    raw_components = path.split(os.pathsep)
    components: list[pathlib.Path] = []
    for raw in raw_components:
        candidate = pathlib.Path(raw)
        if not raw or not candidate.is_absolute():
            raise protocol.ProtocolError("controlled PATH components must be absolute")
        try:
            metadata = candidate.lstat()
            canonical = candidate.resolve(strict=True)
        except OSError as error:
            raise protocol.ProtocolError(
                f"cannot inspect controlled PATH component {candidate}: {error}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or canonical != candidate:
            raise protocol.ProtocolError(
                f"controlled PATH component is not a canonical regular directory: {candidate}"
            )
        if candidate in components:
            raise protocol.ProtocolError(
                f"controlled PATH component is duplicated: {candidate}"
            )
        components.append(candidate)

    required = ("git", "cargo", "rustc")
    candidates: dict[str, list[pathlib.Path]] = {name: [] for name in required}
    used_directories: set[pathlib.Path] = set()
    for directory in components:
        for name in required:
            executable = directory / name
            if executable.exists() or executable.is_symlink():
                candidates[name].append(executable)
                used_directories.add(directory)
    unused = [directory for directory in components if directory not in used_directories]
    if unused:
        raise protocol.ProtocolError(
            f"controlled PATH contains unneeded components: {unused}"
        )

    resolved: dict[str, ResolvedTool] = {}
    for name in required:
        matches = candidates[name]
        if len(matches) != 1:
            raise protocol.ProtocolError(
                f"controlled PATH must resolve {name} exactly once, found {len(matches)}"
            )
        executable = matches[0]
        resolved[name] = _resolve_tool(name, executable)
    normalized_path = os.pathsep.join(str(component) for component in components)
    return ResolvedToolchain(
        path=normalized_path,
        git=resolved["git"],
        cargo=resolved["cargo"],
        rustc=resolved["rustc"],
    )


def _resolve_tool(name: str, path: pathlib.Path) -> ResolvedTool:
    path = pathlib.Path(path)
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect {name} executable {path}: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or canonical != path:
        raise protocol.ProtocolError(
            f"{name} executable must be a canonical regular file: {path}"
        )
    if not os.access(path, os.X_OK):
        raise protocol.ProtocolError(f"{name} executable is not executable: {path}")
    return ResolvedTool(name=name, path=path, sha256=protocol.sha256_file(path))


def validate_resolved_tool(tool: ResolvedTool) -> None:
    """Reject replacement or metadata drift of a previously resolved tool."""
    if not isinstance(tool, ResolvedTool) or tool.name not in ("git", "cargo", "rustc"):
        raise protocol.ProtocolError("resolved tool identity is invalid")
    observed = _resolve_tool(tool.name, tool.path)
    if observed != tool:
        raise protocol.ProtocolError(f"resolved {tool.name} executable changed: {tool.path}")


def validate_resolved_toolchain(tools: ResolvedToolchain) -> None:
    """Revalidate every executable and the minimal normalized PATH."""
    if not isinstance(tools, ResolvedToolchain):
        raise protocol.ProtocolError("resolved toolchain identity is invalid")
    observed = resolve_toolchain(tools.path)
    if observed != tools:
        raise protocol.ProtocolError("resolved toolchain changed")


def _validate_expected_build_identity(
    config: BuildConfig,
    observed: ResolvedToolchain,
) -> None:
    """Require the live build to match its immutable stage identity."""
    if not isinstance(config.expected_toolchain, ResolvedToolchain):
        raise protocol.ProtocolError("build config omits sealed toolchain identity")
    if config.expected_toolchain != observed:
        raise protocol.ProtocolError("build toolchain differs from sealed identity")
    if (
        type(config.expected_host_target) is not str
        or re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.]*"
            r"(?:-[A-Za-z0-9_][A-Za-z0-9_.]*){2,}",
            config.expected_host_target,
        )
        is None
    ):
        raise protocol.ProtocolError("build config host target is invalid")


def normalized_root_cargo(payload: bytes) -> bytes:
    """Replace exactly the five declared strided revisions."""
    if not isinstance(payload, bytes):
        raise protocol.ProtocolError("root Cargo.toml payload must be bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise protocol.ProtocolError("root Cargo.toml is not UTF-8") from error

    if text.count(OLD_STRIDED) != len(STRIDED_DEPENDENCIES):
        raise protocol.ProtocolError(
            "root Cargo.toml does not contain exactly five old strided revisions"
        )

    rendered_lines: list[str] = []
    replaced: set[str] = set()
    for line in text.splitlines(keepends=True):
        matching = [
            dependency
            for dependency in STRIDED_DEPENDENCIES
            if re.match(rf"^{re.escape(dependency)}\s*=", line)
        ]
        if matching:
            dependency = matching[0]
            if dependency in replaced or line.count(OLD_STRIDED) != 1:
                raise protocol.ProtocolError(
                    f"invalid old strided pin for dependency {dependency}"
                )
            line = line.replace(OLD_STRIDED, COMMON_STRIDED)
            replaced.add(dependency)
        rendered_lines.append(line)

    if replaced != set(STRIDED_DEPENDENCIES):
        missing = sorted(set(STRIDED_DEPENDENCIES) - replaced)
        raise protocol.ProtocolError(f"missing strided dependency pins: {missing}")
    rendered = "".join(rendered_lines)
    if OLD_STRIDED in rendered or rendered.count(COMMON_STRIDED) < len(replaced):
        raise protocol.ProtocolError("strided revision replacement was incomplete")
    return rendered.encode("utf-8")


def validate_source_delta(
    role: str,
    baseline_files: Mapping[pathlib.Path, bytes],
    harness_files: Mapping[pathlib.Path, bytes],
    changed_files: Mapping[pathlib.Path, bytes],
) -> None:
    """Accept only the frozen harness and optional five-pin normalization."""
    if role not in (
        "direct-current-main-baseline",
        "common-lock-normalized-baseline",
    ):
        raise protocol.ProtocolError(f"invalid baseline role: {role}")
    mappings = (baseline_files, harness_files, changed_files)
    if any(not isinstance(item, Mapping) for item in mappings):
        raise protocol.ProtocolError("source snapshots must be mappings")

    try:
        baseline_ad = baseline_files[AD_CARGO_PATH]
        harness_ad = harness_files[AD_CARGO_PATH]
        harness_bench = harness_files[BENCH_SOURCE_PATH]
    except KeyError as error:
        raise protocol.ProtocolError(f"source snapshot is missing {error.args[0]}") from error
    if not all(isinstance(item, bytes) for item in (baseline_ad, harness_ad, harness_bench)):
        raise protocol.ProtocolError("source snapshot payloads must be bytes")

    stanza = BENCH_STANZA.encode("utf-8")
    if harness_ad.count(stanza) != 1 or harness_ad.replace(stanza, b"", 1) != baseline_ad:
        raise protocol.ProtocolError(
            "harness Cargo.toml delta is not the exact benchmark stanza"
        )

    expected = {
        AD_CARGO_PATH: harness_ad,
        BENCH_SOURCE_PATH: harness_bench,
    }
    if role == "common-lock-normalized-baseline":
        try:
            expected[ROOT_CARGO_PATH] = normalized_root_cargo(
                baseline_files[ROOT_CARGO_PATH]
            )
        except KeyError as error:
            raise protocol.ProtocolError("baseline is missing root Cargo.toml") from error

    if set(changed_files) != set(expected):
        raise protocol.ProtocolError(
            "baseline delta paths do not match the predeclared source delta"
        )
    for path, expected_bytes in expected.items():
        actual = changed_files[path]
        if not isinstance(actual, bytes) or actual != expected_bytes:
            raise protocol.ProtocolError(f"baseline delta has an unexpected hunk: {path}")


def worktree_specs(scratch_root: pathlib.Path, candidate_commit: str) -> tuple[WorktreeSpec, ...]:
    """Return the three distinct worktrees required for one build set."""
    scratch_root = pathlib.Path(scratch_root)
    _validate_commit(candidate_commit, "candidate commit")
    return (
        WorktreeSpec(
            "direct-current-main-baseline",
            scratch_root / "direct-current-main-baseline",
            IMPLEMENTATION_BASELINE,
        ),
        WorktreeSpec(
            "common-lock-normalized-baseline",
            scratch_root / "common-lock-normalized-baseline",
            IMPLEMENTATION_BASELINE,
        ),
        WorktreeSpec("candidate", scratch_root / "candidate", candidate_commit),
    )


def allocation_probe_build_specs(
    config: BuildConfig, tenferro_manifests: Mapping[str, Mapping[str, Any]]
) -> tuple[AllocationProbeBuildSpec, ...]:
    """Return the fixed three role-bound allocation-probe build identities."""
    if tuple(tenferro_manifests) != tuple(BUILD_MANIFEST_PATHS):
        raise protocol.ProtocolError("allocation probe build role inventory mismatch")
    specs = []
    for role in BUILD_MANIFEST_PATHS:
        manifest = tenferro_manifests[role]
        if type(manifest) is not dict or manifest.get("role") != role:
            raise protocol.ProtocolError(f"allocation probe source role mismatch: {role}")
        lock_name = (
            "direct-probe"
            if role == "direct-current-main-baseline"
            else "common-probe"
        )
        specs.append(
            AllocationProbeBuildSpec(
                role=role,
                repository=pathlib.Path(config.scratch_root) / role,
                lock_name=lock_name,
                manifest_path=pathlib.Path(config.evidence_root)
                / PROBE_BUILD_MANIFEST_PATHS[role],
                profile="bench",
            )
        )
    return tuple(specs)


def prepare_fresh_worktree_destination(path: pathlib.Path) -> pathlib.Path:
    """Create or accept an empty directory reserved for a new worktree."""
    return protocol.prepare_empty_root(pathlib.Path(path))


class GitSourceControl:
    """Git-backed materialization and worktree provenance verifier."""

    def __init__(
        self,
        repository: pathlib.Path,
        *,
        path: str,
        home: pathlib.Path,
        tools: ResolvedToolchain | None = None,
        implementation_baseline: str = IMPLEMENTATION_BASELINE,
        harness_commit: str = HARNESS_COMMIT,
    ) -> None:
        self.repository = pathlib.Path(repository).resolve()
        self.tools = resolve_toolchain(path) if tools is None else tools
        validate_resolved_toolchain(self.tools)
        if self.tools.path != path:
            raise protocol.ProtocolError("Git source-control PATH is not normalized")
        self.path = self.tools.path
        self.home = pathlib.Path(home).resolve()
        self.implementation_baseline = implementation_baseline
        self.harness_commit = harness_commit
        _validate_commit(implementation_baseline, "implementation baseline")
        _validate_commit(harness_commit, "harness commit")

    def validate_role_source(
        self, role: str, head: str, expected_candidate: str
    ) -> None:
        """Revalidate a declared candidate or exact baseline measurement commit."""
        _validate_commit(head, f"{role} build HEAD")
        _validate_commit(expected_candidate, "expected candidate commit")
        if role == "candidate":
            if head != expected_candidate:
                raise protocol.ProtocolError("candidate build HEAD mismatch")
            observed = self._git(("rev-parse", f"{head}^{{commit}}"), cwd=self.repository)
            if observed.strip() != head:
                raise protocol.ProtocolError("candidate build HEAD is not a repository commit")
            return
        if role not in (
            "direct-current-main-baseline",
            "common-lock-normalized-baseline",
        ):
            raise protocol.ProtocolError(f"invalid build source role: {role}")
        self._validate_measurement_commit(role, head)

    def create_worktree(self, spec: WorktreeSpec) -> None:
        """Attach one detached, empty-destination worktree at its declared commit."""
        if not isinstance(spec, WorktreeSpec):
            raise protocol.ProtocolError("invalid worktree specification")
        self._git(
            ("worktree", "add", "--detach", str(spec.path), spec.start_commit),
            cwd=self.repository,
        )
        observed = self._git(("rev-parse", "HEAD"), cwd=spec.path).strip()
        if observed != spec.start_commit:
            raise protocol.ProtocolError(
                f"fresh worktree HEAD mismatch for {spec.role}: {observed}"
            )

    def materialize_baseline(self, spec: WorktreeSpec) -> str:
        """Create and verify one direct or five-pin normalized measurement commit."""
        if spec.role not in (
            "direct-current-main-baseline",
            "common-lock-normalized-baseline",
        ):
            raise protocol.ProtocolError(f"cannot materialize non-baseline role {spec.role}")
        if spec.start_commit != self.implementation_baseline:
            raise protocol.ProtocolError("baseline worktree starts at the wrong commit")
        self._git(
            (
                "checkout",
                self.harness_commit,
                "--",
                str(AD_CARGO_PATH),
                str(BENCH_SOURCE_PATH),
            ),
            cwd=spec.path,
        )
        paths = [AD_CARGO_PATH, BENCH_SOURCE_PATH]
        if spec.role == "common-lock-normalized-baseline":
            root_cargo = spec.path / ROOT_CARGO_PATH
            _replace_regular_bytes(
                root_cargo,
                normalized_root_cargo(_read_regular_bytes(root_cargo)),
            )
            paths.append(ROOT_CARGO_PATH)
        self._git(("add", "--", *(str(path) for path in paths)), cwd=spec.path)
        self._validate_staged_delta(spec)
        commit_environment = {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
        self._git(
            (
                "-c",
                "user.name=Phase 2E Builder",
                "-c",
                "user.email=phase2e-builder@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                f"perf: materialize {spec.role}",
            ),
            cwd=spec.path,
            extra_environment=commit_environment,
        )
        head = self._git(("rev-parse", "HEAD"), cwd=spec.path).strip()
        _validate_commit(head, f"{spec.role} measurement commit")
        self._validate_measurement_commit(spec.role, head)
        return head

    def validate_worktree(
        self,
        path: pathlib.Path,
        expected_head: str,
        expected_lock: pathlib.Path,
    ) -> WorktreeProof:
        """Recompute clean tracked, ignored-lock, config, and benchmark proofs."""
        path = pathlib.Path(path).resolve()
        head = self._git(("rev-parse", "HEAD"), cwd=path).strip()
        if head != expected_head:
            raise protocol.ProtocolError(
                f"worktree HEAD mismatch: expected {expected_head}, got {head}"
            )
        status = self._git(
            ("status", "--porcelain=v1", "--untracked-files=all"), cwd=path
        )
        if status:
            raise protocol.ProtocolError(f"worktree contains a source delta: {status!r}")
        tracked = {
            pathlib.Path(item)
            for item in self._git(("ls-files", "-z"), cwd=path).split("\0")
            if item
        }
        ignored = {
            pathlib.Path(item)
            for item in self._git(
                ("ls-files", "--others", "--ignored", "--exclude-standard", "-z"),
                cwd=path,
            ).split("\0")
            if item
        }
        expected_ignored = {pathlib.Path("Cargo.lock")}
        validate_ignored_inventory(ignored, expected_ignored)
        validate_filesystem_inventory(
            path, expected_ignored, tracked_paths=tracked
        )
        installed_lock = path / "Cargo.lock"
        if protocol.sha256_file(installed_lock) != protocol.sha256_file(expected_lock):
            raise protocol.ProtocolError("worktree Cargo.lock is not root-owned input")

        tree_inventory = self._git(
            ("ls-tree", "-r", "-z", "--full-tree", "HEAD"), cwd=path
        ).encode("utf-8")
        benchmark = path / BENCH_SOURCE_PATH
        benchmark_sha256 = protocol.sha256_file(benchmark)
        ad_cargo = _read_regular_bytes(path / AD_CARGO_PATH)
        stanza = BENCH_STANZA.encode("utf-8")
        if ad_cargo.count(stanza) != 1:
            raise protocol.ProtocolError("benchmark target stanza is missing or duplicated")
        config_chain = self._cargo_config_chain(path, tracked)
        return WorktreeProof(
            head=head,
            tracked_tree_sha256=sha256_bytes(tree_inventory),
            benchmark_sha256=benchmark_sha256,
            benchmark_stanza_sha256=sha256_bytes(stanza),
            cargo_config_chain=tuple(config_chain),
        )

    def _validate_staged_delta(self, spec: WorktreeSpec) -> None:
        names = {
            pathlib.Path(item)
            for item in self._git(
                ("diff", "--cached", "--name-only", "-z"), cwd=spec.path
            ).split("\0")
            if item
        }
        changed = {path: _read_regular_bytes(spec.path / path) for path in names}
        validate_source_delta(
            spec.role,
            {
                ROOT_CARGO_PATH: self._show_bytes(
                    self.implementation_baseline, ROOT_CARGO_PATH
                ),
                AD_CARGO_PATH: self._show_bytes(
                    self.implementation_baseline, AD_CARGO_PATH
                ),
            },
            {
                AD_CARGO_PATH: self._show_bytes(self.harness_commit, AD_CARGO_PATH),
                BENCH_SOURCE_PATH: self._show_bytes(
                    self.harness_commit, BENCH_SOURCE_PATH
                ),
            },
            changed,
        )

    def _validate_measurement_commit(self, role: str, commit: str) -> None:
        names = {
            pathlib.Path(item)
            for item in self._git(
                (
                    "diff",
                    "--name-only",
                    "-z",
                    self.implementation_baseline,
                    commit,
                ),
                cwd=self.repository,
            ).split("\0")
            if item
        }
        changed = {path: self._show_bytes(commit, path) for path in names}
        validate_source_delta(
            role,
            {
                ROOT_CARGO_PATH: self._show_bytes(
                    self.implementation_baseline, ROOT_CARGO_PATH
                ),
                AD_CARGO_PATH: self._show_bytes(
                    self.implementation_baseline, AD_CARGO_PATH
                ),
            },
            {
                AD_CARGO_PATH: self._show_bytes(self.harness_commit, AD_CARGO_PATH),
                BENCH_SOURCE_PATH: self._show_bytes(
                    self.harness_commit, BENCH_SOURCE_PATH
                ),
            },
            changed,
        )

    def _cargo_config_chain(
        self, worktree: pathlib.Path, tracked: set[pathlib.Path]
    ) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        locations = (worktree, *worktree.parents)
        for directory in locations:
            for name in ("config", "config.toml"):
                candidate = directory / ".cargo" / name
                if not candidate.exists() and not candidate.is_symlink():
                    continue
                try:
                    relative = candidate.relative_to(worktree)
                except ValueError as error:
                    raise protocol.ProtocolError(
                        f"ancestor Cargo config is forbidden: {candidate}"
                    ) from error
                if relative not in tracked:
                    raise protocol.ProtocolError(
                        f"untracked repository Cargo config is forbidden: {relative}"
                    )
                entries.append(
                    {"path": str(relative), "sha256": protocol.sha256_file(candidate)}
                )
        return sorted(entries, key=lambda item: item["path"])

    def _show_bytes(self, commit: str, path: pathlib.Path) -> bytes:
        return self._git(("show", f"{commit}:{path.as_posix()}"), cwd=self.repository).encode(
            "utf-8"
        )

    def _git(
        self,
        arguments: tuple[str, ...],
        *,
        cwd: pathlib.Path,
        extra_environment: Mapping[str, str] | None = None,
    ) -> str:
        environment = protocol.runtime_environment(
            path=self.path, home=str(self.home)
        )
        if extra_environment is not None:
            environment.update(extra_environment)
        outcome = run_bounded_command(
            (str(self.tools.git.path), *arguments),
            cwd=cwd,
            environment=environment,
            deadline_seconds=QUERY_DEADLINE_SECONDS,
            executable_identity=self.tools.git,
        )
        if outcome.validity_state != "COMPLETE":
            raise protocol.ProtocolError(
                f"Git command failed: {outcome.argv!r}; stderr={outcome.stderr!r}"
            )
        return outcome.stdout


def _record_suppressed_failure(
    primary: BaseException, operation: str, secondary: BaseException
) -> None:
    """Attach best-effort diagnostics without replacing an active exception."""
    try:
        primary.add_note(
            f"suppressed {operation} failure: {type(secondary).__name__}: {secondary}"
        )
    except BaseException:
        pass


def _read_regular_bytes(path: pathlib.Path) -> bytes:
    path = pathlib.Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
    except BaseException as error:
        if isinstance(error, Exception):
            raise protocol.ProtocolError(
                f"cannot open lock source {path}: {error}"
            ) from error
        raise
    primary: BaseException | None = None
    chunks: list[bytes] = []
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise protocol.ProtocolError(f"lock source is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except BaseException as error:
        if isinstance(error, protocol.ProtocolError) or not isinstance(error, Exception):
            primary = error
            raise
        wrapped = protocol.ProtocolError(f"cannot read lock source {path}: {error}")
        primary = wrapped
        raise wrapped from error
    finally:
        if descriptor is not None:
            owned_descriptor = descriptor
            descriptor = None
            try:
                os.close(owned_descriptor)
            except BaseException as error:
                if primary is not None:
                    _record_suppressed_failure(primary, "lock source close", error)
                elif isinstance(error, Exception):
                    raise protocol.ProtocolError(
                        f"cannot close lock source {path}: {error}"
                    ) from error
                else:
                    raise
    return b"".join(chunks)


def _replace_regular_bytes(path: pathlib.Path, payload: bytes) -> None:
    path = pathlib.Path(path)
    if not isinstance(payload, bytes):
        raise protocol.ProtocolError("replacement payload must be bytes")
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.replace-", suffix=".tmp", dir=path.parent
        )
        temporary = pathlib.Path(temporary_name)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short source replacement write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot replace source file {path}: {error}") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def copy_root_owned_lock(
    evidence_root: pathlib.Path, lock_name: str, source: pathlib.Path
) -> pathlib.Path:
    """Atomically create one immutable root-owned lock copy without overwrite."""
    if lock_name not in LOCK_PATHS:
        raise protocol.ProtocolError(f"unknown Phase 2E lock role: {lock_name}")
    evidence_root = pathlib.Path(evidence_root)
    destination = evidence_root / LOCK_PATHS[lock_name]
    payload = _read_regular_bytes(pathlib.Path(source))
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot create lock evidence directory {destination.parent}: {error}"
        ) from error
    if destination.exists() or destination.is_symlink():
        raise protocol.ProtocolError(f"root-owned lock already exists: {destination}")

    descriptor: int | None = None
    temporary: pathlib.Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.copy-",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = pathlib.Path(temporary_name)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short lock write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise protocol.ProtocolError(
            f"root-owned lock already exists: {destination}"
        ) from error
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot create root-owned lock {destination}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if protocol.sha256_file(destination) != sha256_bytes(payload):
        raise protocol.ProtocolError(f"root-owned lock copy changed: {destination}")
    return destination


def install_root_owned_lock(
    root_owned_lock: pathlib.Path, worktree: pathlib.Path
) -> pathlib.Path:
    """Atomically install exact root-owned lock bytes as worktree Cargo.lock."""
    payload = _read_regular_bytes(pathlib.Path(root_owned_lock))
    worktree = pathlib.Path(worktree)
    destination = worktree / "Cargo.lock"
    descriptor: int | None = None
    temporary: pathlib.Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".Cargo.lock.install-", suffix=".tmp", dir=worktree
        )
        temporary = pathlib.Path(temporary_name)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short Cargo.lock write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
        directory = os.open(worktree, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot install root-owned Cargo.lock in {worktree}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if protocol.sha256_file(destination) != sha256_bytes(payload):
        raise protocol.ProtocolError(f"installed Cargo.lock changed: {destination}")
    return destination


def validate_ignored_inventory(
    actual_paths: set[pathlib.Path], allowed_paths: set[pathlib.Path]
) -> None:
    """Require the ignored-file inventory to equal its explicit allowlist."""
    actual = {_validate_relative_path(path) for path in actual_paths}
    allowed = {_validate_relative_path(path) for path in allowed_paths}
    if actual != allowed:
        raise protocol.ProtocolError(
            f"ignored inventory mismatch: expected {sorted(map(str, allowed))}, "
            f"got {sorted(map(str, actual))}"
        )


def validate_filesystem_inventory(
    root: pathlib.Path,
    allowed_ignored: set[pathlib.Path],
    *,
    tracked_paths: set[pathlib.Path] | None = None,
) -> None:
    """Reject every worktree path outside Git, tracked files, and root Cargo.lock."""
    root = pathlib.Path(root)
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise protocol.ProtocolError(f"cannot inspect worktree {root}: {error}") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise protocol.ProtocolError(f"worktree is not a regular directory: {root}")

    ignored = {_validate_relative_path(path) for path in allowed_ignored}
    tracked = {
        _validate_relative_path(path) for path in (tracked_paths or set())
    }
    allowed_files = ignored | tracked | {pathlib.Path(".git")}
    allowed_directories: set[pathlib.Path] = set()
    for path in allowed_files:
        allowed_directories.update(path.parents)
    allowed_directories.discard(pathlib.Path("."))

    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = pathlib.Path(directory)
        for name in [*directory_names, *file_names]:
            path = current / name
            relative = path.relative_to(root)
            if relative == pathlib.Path(".git") or relative.parts[:1] == (".git",):
                continue
            if path.is_dir() and not path.is_symlink():
                if relative in allowed_directories:
                    continue
            elif relative in allowed_files:
                if relative in ignored:
                    metadata = path.lstat()
                    if not stat.S_ISREG(metadata.st_mode):
                        raise protocol.ProtocolError(
                            f"allowed ignored path is not a regular file: {relative}"
                        )
                continue
            raise protocol.ProtocolError(f"unexpected worktree path: {relative}")


def validate_controlled_cargo_home(cargo_home: pathlib.Path) -> None:
    """Accept pre-seeded Cargo caches but reject config and credentials."""
    cargo_home = pathlib.Path(cargo_home)
    try:
        metadata = cargo_home.lstat()
    except OSError as error:
        raise protocol.ProtocolError(f"cannot inspect CARGO_HOME {cargo_home}: {error}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise protocol.ProtocolError(f"CARGO_HOME is not a regular directory: {cargo_home}")
    for name in ("config", "config.toml", "credentials", "credentials.toml"):
        path = cargo_home / name
        if path.exists() or path.is_symlink():
            raise protocol.ProtocolError(f"controlled CARGO_HOME contains forbidden {name}")


def feature_query_command(
    target: str,
    *,
    package: str | None,
    requested_features: tuple[str, ...],
    no_default_features: bool,
    manifest_path: pathlib.Path | None = None,
) -> tuple[str, ...]:
    """Construct a build-matching Cargo feature query."""
    if not isinstance(target, str) or not target:
        raise protocol.ProtocolError("feature-query target must be a nonempty string")
    if (package is None) == (manifest_path is None):
        raise protocol.ProtocolError(
            "feature query requires exactly one package or manifest path"
        )
    if not requested_features or any(
        not isinstance(feature, str) or not feature for feature in requested_features
    ):
        raise protocol.ProtocolError("requested feature tuple must be nonempty strings")
    argv = ["cargo", "tree", "--locked", "--target", target]
    if package is not None:
        argv.extend(("-p", package))
    else:
        if manifest_path is None:
            raise AssertionError("manifest path disappeared after XOR validation")
        manifest = pathlib.Path(manifest_path)
        if not manifest.is_absolute():
            raise protocol.ProtocolError("feature-query manifest path must be absolute")
        argv.extend(("--manifest-path", str(manifest)))
    if no_default_features:
        argv.append("--no-default-features")
    argv.extend(("--features", ",".join(requested_features), "-e", "features"))
    return tuple(argv)


def timing_feature_command(target: str) -> tuple[str, ...]:
    """Return the exact timing-binary feature query required by protocol v2."""
    return feature_query_command(
        target,
        package="tenferro-ad",
        requested_features=REQUESTED_FEATURES,
        no_default_features=True,
    )


def validate_feature_query(
    argv: tuple[str, ...],
    *,
    target: str,
    package: str | None,
    requested_features: tuple[str, ...],
    no_default_features: bool,
    manifest_path: pathlib.Path | None = None,
) -> None:
    """Reject workspace-default or otherwise non-build-matching queries."""
    expected = feature_query_command(
        target,
        package=package,
        requested_features=requested_features,
        no_default_features=no_default_features,
        manifest_path=manifest_path,
    )
    if tuple(argv) != expected:
        raise protocol.ProtocolError(
            f"feature query does not match build inputs: expected {expected!r}"
        )


def _absolute_tool_command(
    template: tuple[str, ...], tool: ResolvedTool
) -> tuple[str, ...]:
    if not template or template[0] != tool.name:
        raise protocol.ProtocolError(
            f"command template does not belong to resolved {tool.name}"
        )
    return (str(tool.path), *template[1:])


def build_command_plan(
    target: str, cargo: ResolvedTool
) -> tuple[CommandSpec, ...]:
    """Return absolute metadata, feature, and bench-build commands in fixed order."""
    if cargo.name != "cargo":
        raise protocol.ProtocolError("build command plan requires the resolved Cargo tool")
    return (
        CommandSpec(
            "metadata",
            _absolute_tool_command(METADATA_COMMAND, cargo),
            QUERY_DEADLINE_SECONDS,
        ),
        CommandSpec(
            "features",
            _absolute_tool_command(timing_feature_command(target), cargo),
            QUERY_DEADLINE_SECONDS,
        ),
        CommandSpec(
            "build",
            _absolute_tool_command(BENCH_COMMAND, cargo),
            BUILD_DEADLINE_SECONDS,
        ),
    )


def allocation_probe_command_plan(
    manifest: pathlib.Path, binary: pathlib.Path, cargo: str
) -> tuple[CommandSpec, ...]:
    """Return the five exact external-probe verification commands."""
    manifest = pathlib.Path(manifest)
    binary = pathlib.Path(binary)
    cargo_path = pathlib.Path(cargo)
    if not manifest.is_absolute() or not binary.is_absolute() or not cargo_path.is_absolute():
        raise protocol.ProtocolError("allocation probe command paths must be absolute")
    return (
        CommandSpec(
            "fmt",
            (str(cargo_path), "fmt", "--manifest-path", str(manifest), "--", "--check"),
            ALLOCATION_PROBE_FMT_DEADLINE_SECONDS,
        ),
        CommandSpec(
            "test",
            (str(cargo_path), "test", "--manifest-path", str(manifest)),
            ALLOCATION_PROBE_COMMAND_DEADLINE_SECONDS,
        ),
        CommandSpec(
            "clippy",
            (
                str(cargo_path),
                "clippy",
                "--manifest-path",
                str(manifest),
                "--locked",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ),
            ALLOCATION_PROBE_COMMAND_DEADLINE_SECONDS,
        ),
        CommandSpec(
            "build",
            (
                str(cargo_path),
                "build",
                "--locked",
                "--profile",
                "bench",
                "--manifest-path",
                str(manifest),
            ),
            ALLOCATION_PROBE_COMMAND_DEADLINE_SECONDS,
        ),
        CommandSpec(
            "list-cases",
            (str(binary), "--list-cases"),
            ALLOCATION_PROBE_LIST_DEADLINE_SECONDS,
        ),
    )


def allocation_probe_build_only_command_plan(
    manifest: pathlib.Path, binary: pathlib.Path, cargo: str, target: str
) -> tuple[CommandSpec, ...]:
    """Return the exact feature proof, locked build, and inventory query."""
    full = allocation_probe_command_plan(manifest, binary, cargo)
    return (
        CommandSpec(
            "features",
            (
                str(pathlib.Path(cargo)),
                "tree",
                "--locked",
                "--manifest-path",
                str(pathlib.Path(manifest)),
                "--target",
                target,
                "-e",
                "features",
            ),
            QUERY_DEADLINE_SECONDS,
        ),
        full[3],
        full[4],
    )


def _allocation_probe_cargo_config_chain(root: pathlib.Path) -> list[dict[str, str]]:
    """Prove an external generated crate cannot inherit an ancestor Cargo config."""
    root = pathlib.Path(root)
    for directory in (root, *root.parents):
        for name in ("config", "config.toml"):
            candidate = directory / ".cargo" / name
            if candidate.exists() or candidate.is_symlink():
                raise protocol.ProtocolError(
                    f"foreign allocation-probe Cargo config is forbidden: {candidate}"
                )
    return []


def _signal_group(
    pid: int, requested_signal: signal.Signals, signal_process_group: Callable[[int, int], None]
) -> None:
    try:
        signal_process_group(pid, requested_signal)
    except ProcessLookupError:
        return
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot signal process group {pid} with {requested_signal.name}: {error}"
        ) from error


def _timeout_text(value: str | bytes | None, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _timeout_streams(
    error: subprocess.TimeoutExpired, stdout: str, stderr: str
) -> tuple[str, str]:
    return (
        _timeout_text(error.output, stdout),
        _timeout_text(error.stderr, stderr),
    )


def _try_signal_group(
    pid: int,
    requested_signal: signal.Signals,
    signal_process_group: Callable[[int, int], None],
) -> bool:
    try:
        _signal_group(pid, requested_signal, signal_process_group)
    except protocol.ProtocolError:
        return False
    return True


def _best_effort_signal_group(
    pid: int,
    requested_signal: signal.Signals,
    signal_process_group: Callable[[int, int], None],
) -> bool:
    try:
        return _try_signal_group(pid, requested_signal, signal_process_group)
    except BaseException:
        return False


def _force_kill_and_bounded_drain(
    process: Any,
    signal_process_group: Callable[[int, int], None],
) -> None:
    """Best-effort final cleanup that cannot replace an active exception."""
    _best_effort_signal_group(
        process.pid, signal.SIGKILL, signal_process_group
    )
    try:
        process.communicate(timeout=TERMINATION_GRACE_SECONDS)
    except BaseException:
        pass


def _finish_timed_out_command(
    argv: tuple[str, ...],
    *,
    cwd: pathlib.Path,
    actual_environment: Mapping[str, str],
    deadline_seconds: int,
    process: Any,
    timeout: subprocess.TimeoutExpired,
    signal_process_group: Callable[[int, int], None],
    executable_identity: ResolvedTool | None,
    inherited_descriptors: tuple[int, ...],
) -> CommandResult:
    """Bound and record normal timeout cleanup; cancellation is guarded outside."""
    stdout, stderr = _timeout_streams(timeout, "", "")
    cleanup_failures: list[str] = []
    terminated = _try_signal_group(
        process.pid, signal.SIGTERM, signal_process_group
    )
    killed = False
    if not terminated:
        cleanup_failures.append("term-signal-failed")
        killed = _try_signal_group(
            process.pid, signal.SIGKILL, signal_process_group
        )
        if not killed:
            cleanup_failures.append("kill-signal-failed")
        try:
            stdout, stderr = process.communicate(
                timeout=TERMINATION_GRACE_SECONDS
            )
        except subprocess.TimeoutExpired as drain_timeout:
            stdout, stderr = _timeout_streams(
                drain_timeout, stdout, stderr
            )
            cleanup_failures.append("post-kill-drain-timeout")
        except Exception:
            cleanup_failures.append("post-kill-drain-failed")
    else:
        try:
            stdout, stderr = process.communicate(
                timeout=TERMINATION_GRACE_SECONDS
            )
        except subprocess.TimeoutExpired as grace_timeout:
            stdout, stderr = _timeout_streams(
                grace_timeout, stdout, stderr
            )
        except Exception:
            cleanup_failures.append("term-drain-failed")
        # communicate() only observes the leader and its pipes; descendants in
        # the process group may still be alive after it returns successfully.
        killed = _try_signal_group(
            process.pid, signal.SIGKILL, signal_process_group
        )
        if not killed:
            cleanup_failures.append("kill-signal-failed")
        try:
            stdout, stderr = process.communicate(
                timeout=TERMINATION_GRACE_SECONDS
            )
        except subprocess.TimeoutExpired as drain_timeout:
            stdout, stderr = _timeout_streams(
                drain_timeout, stdout, stderr
            )
            cleanup_failures.append("post-kill-drain-timeout")
        except Exception:
            cleanup_failures.append("post-kill-drain-failed")
    if executable_identity is not None:
        validate_resolved_tool(executable_identity)
    reason = "deadline-exceeded"
    if cleanup_failures:
        reason += ":" + "+".join(cleanup_failures)
    return CommandResult(
        argv=tuple(argv),
        cwd=str(pathlib.Path(cwd)),
        environment=dict(actual_environment),
        deadline_seconds=deadline_seconds,
        returncode=getattr(process, "returncode", None),
        stdout=stdout,
        stderr=stderr,
        validity_state="INCONCLUSIVE",
        failure_reason=reason,
        terminated=terminated,
        killed=killed,
        inherited_descriptors=inherited_descriptors,
    )


def run_bounded_command(
    argv: tuple[str, ...],
    *,
    cwd: pathlib.Path,
    environment: Mapping[str, str],
    deadline_seconds: int,
    process_factory: Callable[..., Any] = subprocess.Popen,
    signal_process_group: Callable[[int, int], None] = os.killpg,
    executable_identity: ResolvedTool | None = None,
    inherited_descriptors: tuple[int, ...] = (),
) -> CommandResult:
    """Run one child group and convert timeout/nonzero outcomes to INCONCLUSIVE."""
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise protocol.ProtocolError("command argv must contain nonempty strings")
    if type(deadline_seconds) is not int or deadline_seconds <= 0:
        raise protocol.ProtocolError("command deadline must be a positive integer")
    if (
        type(inherited_descriptors) is not tuple
        or len(inherited_descriptors) > 1
        or any(
            type(descriptor) is not int or descriptor <= 2
            for descriptor in inherited_descriptors
        )
        or len(set(inherited_descriptors)) != len(inherited_descriptors)
    ):
        raise protocol.ProtocolError("inherited descriptor contract is invalid")
    proc_fd_prefix = "/proc/self/fd/"
    names_proc_fd = argv[0].startswith(proc_fd_prefix)
    if inherited_descriptors:
        descriptor = inherited_descriptors[0]
        if INHERITED_EXECUTABLE_SEALS is None:
            raise protocol.ProtocolError(
                "inherited executable descriptors require Linux file seals"
            )
        try:
            metadata = os.fstat(descriptor)
            inheritable = os.get_inheritable(descriptor)
            seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        except OSError as error:
            raise protocol.ProtocolError(
                f"cannot inspect inherited descriptor {descriptor}: {error}"
            ) from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o111 == 0
            or inheritable
            or seals != INHERITED_EXECUTABLE_SEALS
            or argv[0] != f"{proc_fd_prefix}{descriptor}"
        ):
            raise protocol.ProtocolError(
                "inherited descriptor is not the exact CLOEXEC launch snapshot"
            )
    elif names_proc_fd:
        raise protocol.ProtocolError(
            "proc-fd launch requires one explicit inherited descriptor"
        )
    actual_environment = dict(sorted(environment.items()))
    if any(
        type(key) is not str or type(value) is not str
        for key, value in actual_environment.items()
    ):
        raise protocol.ProtocolError("command environment must contain only strings")
    if executable_identity is not None:
        validate_resolved_tool(executable_identity)
        if argv[0] != str(executable_identity.path):
            raise protocol.ProtocolError(
                "launched argv does not name the resolved executable"
            )

    try:
        process_arguments = {
            "cwd": str(pathlib.Path(cwd)),
            "env": actual_environment,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "start_new_session": True,
        }
        if inherited_descriptors:
            process_arguments["pass_fds"] = inherited_descriptors
        process = process_factory(
            list(argv),
            **process_arguments,
        )
    except OSError as error:
        raise protocol.ProtocolError(f"cannot launch command {argv!r}: {error}") from error

    try:
        stdout, stderr = process.communicate(timeout=deadline_seconds)
    except subprocess.TimeoutExpired as timeout:
        try:
            return _finish_timed_out_command(
                argv,
                cwd=cwd,
                actual_environment=actual_environment,
                deadline_seconds=deadline_seconds,
                process=process,
                timeout=timeout,
                signal_process_group=signal_process_group,
                executable_identity=executable_identity,
                inherited_descriptors=inherited_descriptors,
            )
        except BaseException:
            _force_kill_and_bounded_drain(process, signal_process_group)
            raise
    except BaseException:
        term_succeeded = _best_effort_signal_group(
            process.pid, signal.SIGTERM, signal_process_group
        )
        if term_succeeded:
            try:
                process.communicate(timeout=TERMINATION_GRACE_SECONDS)
            except BaseException:
                pass
        _force_kill_and_bounded_drain(process, signal_process_group)
        raise

    if executable_identity is not None:
        validate_resolved_tool(executable_identity)
    returncode = getattr(process, "returncode", None)
    complete = returncode == 0
    return CommandResult(
        argv=tuple(argv),
        cwd=str(pathlib.Path(cwd)),
        environment=actual_environment,
        deadline_seconds=deadline_seconds,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        validity_state="COMPLETE" if complete else "INCONCLUSIVE",
        failure_reason=None if complete else "nonzero-exit",
        terminated=False,
        killed=False,
        inherited_descriptors=inherited_descriptors,
    )


def _toolchain_manifest(
    tools: ResolvedToolchain, cargo_version: str, rustc_version: str
) -> dict[str, Any]:
    if not cargo_version or not rustc_version:
        raise protocol.ProtocolError("toolchain probe returned empty version output")
    return {
        "git": {
            "path": str(tools.git.path),
            "sha256": tools.git.sha256,
        },
        "cargo": {
            "path": str(tools.cargo.path),
            "sha256": tools.cargo.sha256,
            "version": cargo_version,
        },
        "rustc": {
            "path": str(tools.rustc.path),
            "sha256": tools.rustc.sha256,
            "version": rustc_version,
        },
    }


def _rustc_host(stdout: str, context: str) -> str:
    match = re.search(r"^host:\s*(\S+)\s*$", stdout, re.MULTILINE)
    if match is None:
        raise protocol.ProtocolError(f"{context} rustc probe did not report host target")
    return match.group(1)


def build_all(
    config: BuildConfig,
) -> BuildSetResult:
    """Build with authoritative dependencies and reopen persisted evidence."""
    _validate_build_config(config)
    tools = resolve_toolchain(config.path)
    validate_resolved_toolchain(tools)
    _validate_expected_build_identity(config, tools)
    validate_controlled_cargo_home(config.cargo_home)
    source_control = GitSourceControl(
        config.repository,
        path=tools.path,
        home=config.home,
        tools=tools,
    )
    result = _build_all_with_dependencies(
        config,
        source_control=source_control,
        command_runner=run_bounded_command,
    )
    if result.validity_state != "COMPLETE":
        return result
    manifests = validate_build_set(config)
    return BuildSetResult("COMPLETE", manifests, None)


def _build_all_with_dependencies(
    config: BuildConfig,
    *,
    source_control: Any,
    command_runner: Callable[..., CommandResult],
) -> BuildSetResult:
    """Test seam for one build after dependencies establish their authority."""
    _validate_build_config(config)
    tools = resolve_toolchain(config.path)
    validate_resolved_toolchain(tools)
    _validate_expected_build_identity(config, tools)
    scratch_root = prepare_fresh_worktree_destination(config.scratch_root)
    protocol.prepare_empty_root(config.home)
    validate_controlled_cargo_home(config.cargo_home)

    specs = worktree_specs(scratch_root, config.candidate_commit)
    source_tools = getattr(source_control, "tools", None)
    if not isinstance(source_tools, ResolvedToolchain):
        raise protocol.ProtocolError(
            "source-control context has no resolved toolchain identity"
        )
    if source_tools != tools:
        raise protocol.ProtocolError(
            "source-control executable identity differs from build toolchain"
        )
    for spec in specs:
        prepare_fresh_worktree_destination(spec.path)
        source_control.create_worktree(spec)

    heads = {"candidate": config.candidate_commit}
    for spec in specs[:2]:
        heads[spec.role] = source_control.materialize_baseline(spec)
        _validate_commit(heads[spec.role], f"{spec.role} measurement commit")

    target_dirs = {
        spec.role: scratch_root / "targets" / spec.role for spec in specs
    }
    for target_dir in target_dirs.values():
        try:
            target_dir.mkdir(parents=True)
        except OSError as error:
            raise protocol.ProtocolError(
                f"cannot create external Cargo target directory {target_dir}: {error}"
            ) from error

    role_toolchains: dict[str, dict[str, Any]] = {}
    role_targets: dict[str, str] = {}
    for spec in specs:
        environment = protocol.cargo_environment(
            path=tools.path,
            home=str(config.home),
            cargo_home=str(config.cargo_home),
            target_dir=str(target_dirs[spec.role]),
        )
        rustc = _run_build_command(
            command_runner,
            (str(tools.rustc.path), "--version", "--verbose"),
            cwd=spec.path,
            environment=environment,
            deadline_seconds=QUERY_DEADLINE_SECONDS,
            executable_identity=tools.rustc,
        )
        if rustc.validity_state != "COMPLETE":
            return BuildSetResult("INCONCLUSIVE", {}, rustc)
        host_target = _rustc_host(rustc.stdout, f"role-local {spec.role}")
        if host_target != config.expected_host_target:
            raise protocol.ProtocolError(
                f"role-local host target differs from sealed target: {spec.role}"
            )
        cargo = _run_build_command(
            command_runner,
            (str(tools.cargo.path), "--version", "--verbose"),
            cwd=spec.path,
            environment=environment,
            deadline_seconds=QUERY_DEADLINE_SECONDS,
            executable_identity=tools.cargo,
        )
        if cargo.validity_state != "COMPLETE":
            return BuildSetResult("INCONCLUSIVE", {}, cargo)
        role_targets[spec.role] = host_target
        role_toolchains[spec.role] = _toolchain_manifest(
            tools, cargo.stdout.strip(), rustc.stdout.strip()
        )

    reference_role = specs[0].role
    for spec in specs[1:]:
        if role_targets[spec.role] != role_targets[reference_role]:
            raise protocol.ProtocolError(
                f"role-local host target differs: {reference_role} vs {spec.role}"
            )
        if role_toolchains[spec.role] != role_toolchains[reference_role]:
            raise protocol.ProtocolError(
                f"role-local toolchain differs: {reference_role} vs {spec.role}"
            )
    host_target = role_targets[reference_role]

    specs_by_role = {spec.role: spec for spec in specs}
    lock_generators = {
        "direct": specs_by_role["direct-current-main-baseline"],
        "common": specs_by_role["candidate"],
    }
    generated_locks: dict[str, pathlib.Path] = {}
    for lock_name, spec in lock_generators.items():
        environment = protocol.cargo_environment(
            path=tools.path,
            home=str(config.home),
            cargo_home=str(config.cargo_home),
            target_dir=str(target_dirs[spec.role]),
        )
        outcome = _run_build_command(
            command_runner,
            _absolute_tool_command(LOCK_COMMAND, tools.cargo),
            cwd=spec.path,
            environment=environment,
            deadline_seconds=QUERY_DEADLINE_SECONDS,
            executable_identity=tools.cargo,
        )
        if outcome.validity_state != "COMPLETE":
            return BuildSetResult("INCONCLUSIVE", {}, outcome)
        generated = spec.path / "Cargo.lock"
        protocol.sha256_file(generated)
        generated_locks[lock_name] = copy_root_owned_lock(
            config.evidence_root, lock_name, generated
        )

    role_locks = {
        "direct-current-main-baseline": generated_locks["direct"],
        "common-lock-normalized-baseline": generated_locks["common"],
        "candidate": generated_locks["common"],
    }
    for spec in specs:
        install_root_owned_lock(role_locks[spec.role], spec.path)

    manifests: dict[str, dict[str, Any]] = {}
    for spec in specs:
        owned_lock = role_locks[spec.role]
        before = source_control.validate_worktree(
            spec.path, heads[spec.role], owned_lock
        )
        _validate_worktree_proof(before, heads[spec.role])
        environment = protocol.cargo_environment(
            path=tools.path,
            home=str(config.home),
            cargo_home=str(config.cargo_home),
            target_dir=str(target_dirs[spec.role]),
        )
        outcomes: dict[str, CommandResult] = {}
        command_plan = build_command_plan(host_target, tools.cargo)
        for command in command_plan:
            outcome = _run_build_command(
                command_runner,
                command.argv,
                cwd=spec.path,
                environment=environment,
                deadline_seconds=command.deadline_seconds,
                executable_identity=tools.cargo,
            )
            if outcome.validity_state != "COMPLETE":
                return BuildSetResult("INCONCLUSIVE", {}, outcome)
            outcomes[command.name] = outcome
        try:
            metadata = json.loads(outcomes["metadata"].stdout)
        except (TypeError, ValueError) as error:
            raise protocol.ProtocolError("cargo metadata returned malformed JSON") from error
        if type(metadata) is not dict or type(metadata.get("packages")) is not list:
            raise protocol.ProtocolError("cargo metadata output has an invalid schema")
        if not outcomes["features"].stdout.strip():
            raise protocol.ProtocolError("Cargo feature graph is empty")
        executable = parse_bench_executable(
            outcomes["build"].stdout + "\n" + outcomes["build"].stderr,
            target_dirs[spec.role],
        )
        after = source_control.validate_worktree(
            spec.path, heads[spec.role], owned_lock
        )
        _validate_worktree_proof(after, heads[spec.role])
        if before != after:
            raise protocol.ProtocolError(
                f"worktree provenance changed during build: {spec.role}"
            )
        config_chain = [dict(item) for item in before.cargo_config_chain]
        manifest = {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "toolchain": role_toolchains[spec.role],
            "target": role_targets[spec.role],
            "profile": "bench",
            "requested_features": list(REQUESTED_FEATURES),
            "provider": "Faer",
            "benchmark_sha256": before.benchmark_sha256,
            "benchmark_stanza_sha256": before.benchmark_stanza_sha256,
            "command_template": list(BENCH_COMMAND),
            "config_chain_sha256": protocol.sha256_json(config_chain),
            "role": spec.role,
            "head": heads[spec.role],
            "tracked_tree_sha256": before.tracked_tree_sha256,
            "resolved_features_sha256": sha256_bytes(
                outcomes["features"].stdout.encode("utf-8")
            ),
            "lock_sha256": protocol.sha256_file(owned_lock),
            "worktree": str(spec.path.resolve()),
            "target_dir": str(target_dirs[spec.role].resolve()),
            "executable": str(executable),
            "executable_sha256": protocol.sha256_file(executable),
            "validity_state": "COMPLETE",
            "source_delta": list(_ROLE_SOURCE_DELTAS[spec.role]),
            "commands": [
                command.to_manifest() for command in command_plan
            ],
            "environment": environment,
            "cargo_config_chain": config_chain,
        }
        validate_build_manifest(manifest)
        manifests[spec.role] = manifest

    validate_pair(
        "direct-current-main",
        manifests["direct-current-main-baseline"],
        manifests["candidate"],
    )
    validate_pair(
        "common-lock-normalized",
        manifests["common-lock-normalized-baseline"],
        manifests["candidate"],
    )
    for role, manifest in manifests.items():
        destination = config.evidence_root / BUILD_MANIFEST_PATHS[role]
        destination.parent.mkdir(parents=True, exist_ok=True)
        protocol.atomic_write_json(destination, manifest)
        try:
            destination.chmod(0o444)
        except OSError as error:
            raise protocol.ProtocolError(
                f"cannot make build manifest read-only {destination}: {error}"
            ) from error
    return BuildSetResult("COMPLETE", manifests, None)


def _run_build_command(
    command_runner: Callable[..., CommandResult],
    argv: tuple[str, ...],
    *,
    cwd: pathlib.Path,
    environment: Mapping[str, str],
    deadline_seconds: int,
    executable_identity: ResolvedTool,
) -> CommandResult:
    validate_resolved_tool(executable_identity)
    if not argv or argv[0] != str(executable_identity.path):
        raise protocol.ProtocolError(
            "build command does not use its resolved absolute executable"
        )
    outcome = command_runner(
        argv,
        cwd=pathlib.Path(cwd),
        environment=dict(environment),
        deadline_seconds=deadline_seconds,
    )
    validate_resolved_tool(executable_identity)
    if not isinstance(outcome, CommandResult):
        raise protocol.ProtocolError("command runner returned an invalid result")
    if outcome.argv != tuple(argv):
        raise protocol.ProtocolError("command result argv does not match launched argv")
    if pathlib.Path(outcome.cwd) != pathlib.Path(cwd):
        raise protocol.ProtocolError("command result cwd does not match launched cwd")
    if outcome.environment != dict(sorted(environment.items())):
        raise protocol.ProtocolError("command result environment does not match launch")
    if outcome.deadline_seconds != deadline_seconds:
        raise protocol.ProtocolError("command result deadline does not match launch")
    if outcome.validity_state not in ("COMPLETE", "INCONCLUSIVE"):
        raise protocol.ProtocolError("command result validity state is invalid")
    return outcome


def _validate_worktree_proof(proof: WorktreeProof, expected_head: str) -> None:
    if not isinstance(proof, WorktreeProof):
        raise protocol.ProtocolError("source-control adapter returned invalid proof")
    if proof.head != expected_head:
        raise protocol.ProtocolError("worktree proof HEAD mismatch")
    _validate_commit(proof.head, "worktree proof HEAD")
    for name, value in (
        ("tracked tree", proof.tracked_tree_sha256),
        ("benchmark", proof.benchmark_sha256),
        ("benchmark stanza", proof.benchmark_stanza_sha256),
    ):
        _validate_sha256(value, name)
    if any(type(item) is not dict for item in proof.cargo_config_chain):
        raise protocol.ProtocolError("Cargo config chain entries must be dictionaries")


def _validate_build_config(config: BuildConfig) -> None:
    if not isinstance(config, BuildConfig):
        raise protocol.ProtocolError("build config has an invalid type")
    _validate_commit(config.candidate_commit, "candidate commit")
    protocol.cargo_environment(
        path=config.path,
        home=str(config.home),
        cargo_home=str(config.cargo_home),
        target_dir=str(config.scratch_root / "validation-target"),
    )
    for name in ("repository", "evidence_root"):
        path = pathlib.Path(getattr(config, name))
        try:
            metadata = path.lstat()
        except OSError as error:
            raise protocol.ProtocolError(f"cannot inspect {name} {path}: {error}") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise protocol.ProtocolError(f"{name} is not a regular directory: {path}")
        if not path.is_absolute():
            raise protocol.ProtocolError(f"{name} must be absolute: {path}")
    for name in ("scratch_root", "home", "cargo_home"):
        if not pathlib.Path(getattr(config, name)).is_absolute():
            raise protocol.ProtocolError(f"{name} must be absolute")
    resolved = {
        name: pathlib.Path(getattr(config, name)).resolve()
        for name in ("repository", "evidence_root", "scratch_root", "home", "cargo_home")
    }
    for outer_name, outer in resolved.items():
        for inner_name, inner in resolved.items():
            if outer_name >= inner_name:
                continue
            pair = {outer_name, inner_name}
            if pair == {"repository", "evidence_root"}:
                repository = resolved["repository"]
                evidence_root = resolved["evidence_root"]
                if repository in evidence_root.parents:
                    continue
            if outer == inner or outer in inner.parents or inner in outer.parents:
                raise protocol.ProtocolError(
                    f"build roots must be disjoint: {outer_name} and {inner_name}"
                )


def parse_bench_executable(output: str, target_dir: pathlib.Path) -> pathlib.Path:
    """Parse exactly one Cargo bench executable and bind it below target dir."""
    if not isinstance(output, str):
        raise protocol.ProtocolError("Cargo build output must be text")
    candidates: list[pathlib.Path] = []
    pattern = re.compile(r"^Executable\s+.+\s+\((.+)\)$")
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match is not None:
            candidates.append(pathlib.Path(match.group(1)))
    if len(candidates) != 1:
        raise protocol.ProtocolError(
            f"expected exactly one Cargo bench executable, found {len(candidates)}"
        )
    target = pathlib.Path(os.path.abspath(target_dir))
    try:
        target_metadata = target.lstat()
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect CARGO_TARGET_DIR {target}: {error}"
        ) from error
    if not stat.S_ISDIR(target_metadata.st_mode):
        raise protocol.ProtocolError(f"CARGO_TARGET_DIR is not a regular directory: {target}")
    executable = candidates[0]
    if not executable.is_absolute():
        executable = target / executable
    executable = pathlib.Path(os.path.abspath(executable))
    try:
        executable.relative_to(target)
    except ValueError as error:
        raise protocol.ProtocolError(
            f"bench executable is outside CARGO_TARGET_DIR: {executable}"
        ) from error
    protocol.sha256_file(executable)
    return executable


def _toolchain_from_manifest(payload: Any) -> ResolvedToolchain:
    if type(payload) is not dict or set(payload) != {"git", "cargo", "rustc"}:
        raise protocol.ProtocolError("build manifest toolchain schema mismatch")
    tools: dict[str, ResolvedTool] = {}
    for name in ("git", "cargo", "rustc"):
        item = payload[name]
        expected_fields = {"path", "sha256"}
        if name != "git":
            expected_fields.add("version")
        if type(item) is not dict or set(item) != expected_fields:
            raise protocol.ProtocolError(
                f"build manifest {name} tool schema mismatch"
            )
        if name != "git" and (
            type(item["version"]) is not str or not item["version"]
        ):
            raise protocol.ProtocolError(
                f"build manifest {name} version must be nonempty"
            )
        if type(item["path"]) is not str or not item["path"]:
            raise protocol.ProtocolError(
                f"build manifest {name} path must be nonempty text"
            )
        _validate_sha256(item["sha256"], f"{name} executable")
        tool = ResolvedTool(
            name=name,
            path=pathlib.Path(item["path"]),
            sha256=item["sha256"],
        )
        validate_resolved_tool(tool)
        tools[name] = tool
    directories: list[pathlib.Path] = []
    for name in ("git", "cargo", "rustc"):
        directory = tools[name].path.parent
        if directory not in directories:
            directories.append(directory)
    return ResolvedToolchain(
        path=os.pathsep.join(map(str, directories)),
        git=tools["git"],
        cargo=tools["cargo"],
        rustc=tools["rustc"],
    )


def validate_build_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate one strict COMPLETE build manifest and re-hash its executable."""
    schema: dict[str, type | tuple[type, ...]] = {
        **{name: object for name in INVARIANT_FIELDS | ROLE_FIELDS},
        **{name: object for name in AUDIT_FIELDS},
    }
    schema.update(
        {
            "protocol_version": int,
            "toolchain": dict,
            "target": str,
            "profile": str,
            "requested_features": list,
            "provider": str,
            "benchmark_sha256": str,
            "benchmark_stanza_sha256": str,
            "command_template": list,
            "config_chain_sha256": str,
            "role": str,
            "head": str,
            "tracked_tree_sha256": str,
            "resolved_features_sha256": str,
            "lock_sha256": str,
            "worktree": str,
            "target_dir": str,
            "executable": str,
            "executable_sha256": str,
            "validity_state": str,
            "source_delta": list,
            "commands": list,
            "environment": dict,
            "cargo_config_chain": list,
        }
    )
    protocol.validate_manifest_fields(manifest, schema, context="build manifest")
    if manifest["protocol_version"] != protocol.PROTOCOL_VERSION:
        raise protocol.ProtocolError("build manifest protocol version mismatch")
    role = manifest["role"]
    if role not in _BUILD_ROLES:
        raise protocol.ProtocolError(f"invalid build role: {role}")
    if manifest["validity_state"] != "COMPLETE":
        raise protocol.ProtocolError("build manifest is not validity COMPLETE")
    if manifest["profile"] != "bench":
        raise protocol.ProtocolError("build manifest profile must be bench")
    if manifest["requested_features"] != list(REQUESTED_FEATURES):
        raise protocol.ProtocolError("build manifest requested feature tuple mismatch")
    if manifest["provider"] != "Faer":
        raise protocol.ProtocolError("build manifest provider must be Faer")
    if manifest["command_template"] != list(BENCH_COMMAND):
        raise protocol.ProtocolError("build manifest command template mismatch")
    if manifest["source_delta"] != list(_ROLE_SOURCE_DELTAS[role]):
        raise protocol.ProtocolError("build manifest source delta mismatch")
    _validate_commit(manifest["head"], "build HEAD")
    for field in (
        "tracked_tree_sha256",
        "resolved_features_sha256",
        "lock_sha256",
        "benchmark_sha256",
        "benchmark_stanza_sha256",
        "config_chain_sha256",
        "executable_sha256",
    ):
        _validate_sha256(manifest[field], field)

    tools = _toolchain_from_manifest(manifest["toolchain"])

    expected_commands = [
        command.to_manifest()
        for command in build_command_plan(manifest["target"], tools.cargo)
    ]
    if manifest["commands"] != expected_commands:
        raise protocol.ProtocolError("build manifest actual command sequence mismatch")
    if protocol.sha256_json(manifest["cargo_config_chain"]) != manifest["config_chain_sha256"]:
        raise protocol.ProtocolError("Cargo config chain digest mismatch")

    environment = manifest["environment"]
    target_dir = pathlib.Path(manifest["target_dir"])
    worktree = pathlib.Path(manifest["worktree"])
    executable = pathlib.Path(manifest["executable"])
    if not target_dir.is_absolute() or not worktree.is_absolute() or not executable.is_absolute():
        raise protocol.ProtocolError("build manifest paths must be absolute")
    expected_environment = protocol.cargo_environment(
        path=environment.get("PATH"),
        home=environment.get("HOME"),
        cargo_home=environment.get("CARGO_HOME"),
        target_dir=str(target_dir),
    )
    if environment != expected_environment:
        raise protocol.ProtocolError("build manifest environment is not sealed")
    path_components = environment["PATH"].split(os.pathsep)
    tool_directories = {
        str(tool.path.parent) for tool in (tools.git, tools.cargo, tools.rustc)
    }
    if set(path_components) != tool_directories or len(path_components) != len(
        tool_directories
    ):
        raise protocol.ProtocolError(
            "build manifest PATH is not the minimal resolved tool path"
        )
    try:
        executable.resolve().relative_to(target_dir.resolve())
    except ValueError as error:
        raise protocol.ProtocolError("build executable is outside target directory") from error
    if protocol.sha256_file(executable) != manifest["executable_sha256"]:
        raise protocol.ProtocolError("build executable digest mismatch")


def validate_pair(
    comparison_kind: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Validate invariant equality and only predeclared role differences."""
    validate_build_manifest(baseline)
    validate_build_manifest(candidate)
    expected_baseline_role = {
        "direct-current-main": "direct-current-main-baseline",
        "common-lock-normalized": "common-lock-normalized-baseline",
    }.get(comparison_kind)
    if expected_baseline_role is None:
        raise protocol.ProtocolError(f"invalid comparison kind: {comparison_kind}")
    if baseline["role"] != expected_baseline_role or candidate["role"] != "candidate":
        raise protocol.ProtocolError("build manifest roles do not match comparison kind")
    for field in INVARIANT_FIELDS:
        if baseline[field] != candidate[field]:
            raise protocol.ProtocolError(f"build invariant field differs: {field}")
    if baseline["commands"] != candidate["commands"]:
        raise protocol.ProtocolError("actual Cargo command sequence differs by role")
    if baseline["cargo_config_chain"] != candidate["cargo_config_chain"]:
        raise protocol.ProtocolError("Cargo config chain differs by role")
    if _normalized_environment(baseline) != _normalized_environment(candidate):
        raise protocol.ProtocolError("sealed Cargo environment differs beyond target path")
    if comparison_kind == "common-lock-normalized":
        if baseline["lock_sha256"] != candidate["lock_sha256"]:
            raise protocol.ProtocolError("normalized pair does not share the common lock")
    elif baseline["lock_sha256"] == candidate["lock_sha256"]:
        raise protocol.ProtocolError("direct comparison unexpectedly shares one lock")


def validate_build_set(
    config: BuildConfig,
) -> dict[str, dict[str, Any]]:
    """Reopen persisted builds using the authoritative Git source validator."""
    _validate_build_config(config)
    tools = resolve_toolchain(config.path)
    validate_resolved_toolchain(tools)
    if config.expected_toolchain is not None:
        _validate_expected_build_identity(config, tools)
    validate_controlled_cargo_home(config.cargo_home)
    source_control = GitSourceControl(
        config.repository,
        path=tools.path,
        home=config.home,
        tools=tools,
    )
    return _validate_build_set_with_source_control(
        config,
        source_control,
        command_runner=run_bounded_command,
    )


def _validate_build_set_with_source_control(
    config: BuildConfig,
    source_control: Any,
    *,
    command_runner: Callable[..., CommandResult],
) -> dict[str, dict[str, Any]]:
    """Test seam for validation after the caller establishes source authority."""
    _validate_build_config(config)
    tools = resolve_toolchain(config.path)
    validate_resolved_toolchain(tools)
    if config.expected_toolchain is not None:
        _validate_expected_build_identity(config, tools)
    validate_controlled_cargo_home(config.cargo_home)
    source_tools = getattr(source_control, "tools", None)
    if not isinstance(source_tools, ResolvedToolchain):
        raise protocol.ProtocolError(
            "source-control context has no resolved toolchain identity"
        )
    if source_tools != tools:
        raise protocol.ProtocolError(
            "source-control executable identity differs from validation toolchain"
        )
    if not callable(
        getattr(source_control, "validate_role_source", None)
    ) or not callable(getattr(source_control, "validate_worktree", None)):
        raise protocol.ProtocolError(
            "source-control context cannot authoritatively revalidate build inputs"
        )

    evidence_root = pathlib.Path(config.evidence_root)
    manifests: dict[str, dict[str, Any]] = {}
    for role, relative in BUILD_MANIFEST_PATHS.items():
        path = evidence_root / relative
        decoded = protocol.decode_canonical_json_bytes(
            _read_regular_bytes(path), f"build manifest {path}"
        )
        if type(decoded) is not dict:
            raise protocol.ProtocolError(f"build manifest is not an object: {path}")
        validate_build_manifest(decoded)
        if decoded["role"] != role:
            raise protocol.ProtocolError(f"build manifest stored under wrong role: {path}")
        if (
            config.expected_toolchain is not None
            and (
                decoded["target"] != config.expected_host_target
                or any(
                    decoded["toolchain"][name]["path"]
                    != str(getattr(config.expected_toolchain, name).path)
                    or decoded["toolchain"][name]["sha256"]
                    != getattr(config.expected_toolchain, name).sha256
                    for name in ("git", "cargo", "rustc")
                )
            )
        ):
            raise protocol.ProtocolError(
                f"build manifest differs from sealed identity: {role}"
            )
        manifests[role] = decoded

    direct_sha256 = protocol.sha256_file(evidence_root / LOCK_PATHS["direct"])
    common_sha256 = protocol.sha256_file(evidence_root / LOCK_PATHS["common"])
    if manifests["direct-current-main-baseline"]["lock_sha256"] != direct_sha256:
        raise protocol.ProtocolError("direct baseline is not bound to root-owned direct lock")
    for role in ("common-lock-normalized-baseline", "candidate"):
        if manifests[role]["lock_sha256"] != common_sha256:
            raise protocol.ProtocolError(
                f"{role} is not bound to the root-owned common lock"
            )

    role_locks = {
        "direct-current-main-baseline": evidence_root / LOCK_PATHS["direct"],
        "common-lock-normalized-baseline": evidence_root / LOCK_PATHS["common"],
        "candidate": evidence_root / LOCK_PATHS["common"],
    }
    for role, manifest in manifests.items():
        expected_worktree = _canonical_directory(
            pathlib.Path(config.scratch_root) / role,
            f"{role} worktree",
        )
        expected_target = _canonical_directory(
            pathlib.Path(config.scratch_root) / "targets" / role,
            f"{role} target directory",
        )
        if pathlib.Path(manifest["worktree"]) != expected_worktree:
            raise protocol.ProtocolError(f"{role} worktree path is not authoritative")
        if pathlib.Path(manifest["target_dir"]) != expected_target:
            raise protocol.ProtocolError(f"{role} target path is not authoritative")
        executable = pathlib.Path(manifest["executable"])
        try:
            canonical_executable = executable.resolve(strict=True)
            canonical_executable.relative_to(expected_target)
        except (OSError, ValueError) as error:
            raise protocol.ProtocolError(
                f"{role} executable is not under its authoritative target"
            ) from error
        if canonical_executable != executable:
            raise protocol.ProtocolError(f"{role} executable path is not canonical")

        manifest_tools = _toolchain_from_manifest(manifest["toolchain"])
        if any(
            getattr(manifest_tools, name) != getattr(tools, name)
            for name in ("git", "cargo", "rustc")
        ):
            raise protocol.ProtocolError(f"{role} tool executable identity changed")
        environment = manifest["environment"]
        if environment["PATH"] != tools.path:
            raise protocol.ProtocolError(f"{role} PATH differs from controlled tools")
        if environment["HOME"] != str(pathlib.Path(config.home)):
            raise protocol.ProtocolError(f"{role} HOME differs from validation context")
        if environment["CARGO_HOME"] != str(pathlib.Path(config.cargo_home)):
            raise protocol.ProtocolError(
                f"{role} CARGO_HOME differs from validation context"
            )

        source_control.validate_role_source(
            role, manifest["head"], config.candidate_commit
        )
        proof = source_control.validate_worktree(
            expected_worktree,
            manifest["head"],
            role_locks[role],
        )
        _validate_worktree_proof(proof, manifest["head"])
        expected_proof = {
            "tracked_tree_sha256": proof.tracked_tree_sha256,
            "benchmark_sha256": proof.benchmark_sha256,
            "benchmark_stanza_sha256": proof.benchmark_stanza_sha256,
            "cargo_config_chain": [dict(item) for item in proof.cargo_config_chain],
        }
        for field, observed in expected_proof.items():
            if manifest[field] != observed:
                raise protocol.ProtocolError(
                    f"{role} persisted source proof differs: {field}"
                )
        if protocol.sha256_json(expected_proof["cargo_config_chain"]) != manifest[
            "config_chain_sha256"
        ]:
            raise protocol.ProtocolError(
                f"{role} persisted Cargo config proof differs"
            )
        _revalidate_role_build_observations(
            role,
            manifest,
            tools=tools,
            command_runner=command_runner,
        )
    validate_pair(
        "direct-current-main",
        manifests["direct-current-main-baseline"],
        manifests["candidate"],
    )
    validate_pair(
        "common-lock-normalized",
        manifests["common-lock-normalized-baseline"],
        manifests["candidate"],
    )
    return manifests


def _revalidate_role_build_observations(
    role: str,
    manifest: Mapping[str, Any],
    *,
    tools: ResolvedToolchain,
    command_runner: Callable[..., CommandResult],
) -> None:
    worktree = pathlib.Path(manifest["worktree"])
    environment = manifest["environment"]
    probes = (
        (
            "rustc version",
            (str(tools.rustc.path), "--version", "--verbose"),
            tools.rustc,
        ),
        (
            "cargo version",
            (str(tools.cargo.path), "--version", "--verbose"),
            tools.cargo,
        ),
        (
            "resolved features",
            _absolute_tool_command(
                timing_feature_command(manifest["target"]), tools.cargo
            ),
            tools.cargo,
        ),
    )
    outcomes: dict[str, CommandResult] = {}
    for name, argv, executable_identity in probes:
        outcome = _run_build_command(
            command_runner,
            argv,
            cwd=worktree,
            environment=environment,
            deadline_seconds=QUERY_DEADLINE_SECONDS,
            executable_identity=executable_identity,
        )
        if outcome.validity_state != "COMPLETE":
            raise protocol.ProtocolError(
                f"{role} persisted {name} probe did not complete"
            )
        outcomes[name] = outcome

    toolchain = manifest["toolchain"]
    rustc_version = outcomes["rustc version"].stdout.strip()
    cargo_version = outcomes["cargo version"].stdout.strip()
    if rustc_version != toolchain["rustc"]["version"]:
        raise protocol.ProtocolError(f"{role} persisted rustc version differs")
    if cargo_version != toolchain["cargo"]["version"]:
        raise protocol.ProtocolError(f"{role} persisted cargo version differs")
    if _rustc_host(rustc_version, f"persisted {role}") != manifest["target"]:
        raise protocol.ProtocolError(f"{role} persisted host target differs")
    features = outcomes["resolved features"].stdout
    if not features.strip():
        raise protocol.ProtocolError(f"{role} persisted Cargo feature graph is empty")
    if sha256_bytes(features.encode("utf-8")) != manifest[
        "resolved_features_sha256"
    ]:
        raise protocol.ProtocolError(
            f"{role} persisted resolved feature graph differs"
        )


def _allocation_probe_input_bytes(
    repository: pathlib.Path,
) -> tuple[pathlib.Path, bytes, dict[pathlib.Path, bytes]]:
    repository = pathlib.Path(repository)
    try:
        metadata = repository.lstat()
        canonical = repository.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect allocation probe repository {repository}: {error}"
        ) from error
    if not repository.is_absolute() or not stat.S_ISDIR(metadata.st_mode) or canonical != repository:
        raise protocol.ProtocolError(
            "allocation probe repository must be an absolute canonical directory"
        )
    source_root = repository / ALLOCATION_PROBE_SOURCE_ROOT
    try:
        root_metadata = source_root.lstat()
        canonical_source_root = source_root.resolve(strict=True)
        root_entries = {entry.name: entry for entry in os.scandir(source_root)}
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect allocation probe source inventory: {error}"
        ) from error
    if canonical_source_root != source_root or not stat.S_ISDIR(root_metadata.st_mode) or set(root_entries) != {
        ALLOCATION_PROBE_TEMPLATE.name,
        "src",
    }:
        raise protocol.ProtocolError("allocation probe source inventory mismatch")
    if root_entries[ALLOCATION_PROBE_TEMPLATE.name].is_symlink() or not root_entries[
        ALLOCATION_PROBE_TEMPLATE.name
    ].is_file(follow_symlinks=False):
        raise protocol.ProtocolError("allocation probe template is not a regular file")
    src_entry = root_entries["src"]
    if src_entry.is_symlink() or not src_entry.is_dir(follow_symlinks=False):
        raise protocol.ProtocolError("allocation probe src is not a regular directory")
    try:
        source_entries = {entry.name: entry for entry in os.scandir(source_root / "src")}
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect allocation probe src inventory: {error}"
        ) from error
    expected_names = {path.name for path in ALLOCATION_PROBE_SOURCES}
    if set(source_entries) != expected_names:
        raise protocol.ProtocolError("allocation probe src inventory mismatch")
    for name, entry in source_entries.items():
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise protocol.ProtocolError(
                f"allocation probe source is not a regular file: {name}"
            )
    template = _read_regular_bytes(source_root / ALLOCATION_PROBE_TEMPLATE)
    sources = {
        relative: _read_regular_bytes(source_root / relative)
        for relative in ALLOCATION_PROBE_SOURCES
    }
    return source_root, template, sources


def _toml_escape_string_content(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _render_allocation_probe_manifest(
    template: bytes, repository: pathlib.Path
) -> bytes:
    try:
        text = template.decode("utf-8")
    except UnicodeDecodeError as error:
        raise protocol.ProtocolError("allocation probe template is not UTF-8") from error
    if text.count(ALLOCATION_PROBE_ROOT_PLACEHOLDER) != 3:
        raise protocol.ProtocolError(
            "allocation probe template must use exactly one repository-root token kind"
        )
    without_placeholder = text.replace(ALLOCATION_PROBE_ROOT_PLACEHOLDER, "")
    if re.search(r"__[A-Z][A-Z0-9_]+__", without_placeholder):
        raise protocol.ProtocolError("allocation probe template contains a foreign token")
    rendered = text.replace(
        ALLOCATION_PROBE_ROOT_PLACEHOLDER,
        _toml_escape_string_content(str(repository)),
    ).encode("utf-8")
    _validate_allocation_probe_manifest(rendered, repository)
    return rendered


def _validate_allocation_probe_manifest(payload: bytes, repository: pathlib.Path) -> None:
    try:
        decoded = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise protocol.ProtocolError("generated allocation probe manifest is invalid") from error
    if set(decoded) != {"package", "dependencies"}:
        raise protocol.ProtocolError("allocation probe top-level manifest schema mismatch")
    package = decoded["package"]
    expected_package = {
        "name": ALLOCATION_PROBE_BINARY,
        "version": "0.0.0",
        "edition": "2021",
        "publish": False,
    }
    if type(package) is not dict or set(package) != set(expected_package):
        raise protocol.ProtocolError("allocation probe package schema mismatch")
    for field, expected in expected_package.items():
        observed = package[field]
        if type(observed) is not type(expected) or observed != expected:
            raise protocol.ProtocolError(
                f"allocation probe package field mismatch: {field}"
            )
    dependencies = decoded.get("dependencies")
    expected_names = {"tenferro-ad", "tenferro-cpu", "tenferro-tensor"}
    if type(dependencies) is not dict or set(dependencies) != expected_names:
        raise protocol.ProtocolError("allocation probe dependency set mismatch")
    for name in sorted(expected_names):
        item = dependencies[name]
        expected_path = repository / "crates" / name
        if type(item) is not dict or item.get("default-features") is not False:
            raise protocol.ProtocolError(f"allocation probe dependency contract mismatch: {name}")
        if set(item) - {"path", "default-features", "features"}:
            raise protocol.ProtocolError(f"allocation probe dependency fields mismatch: {name}")
        path_value = item.get("path")
        if type(path_value) is not str:
            raise protocol.ProtocolError(f"allocation probe dependency path is invalid: {name}")
        path = pathlib.Path(path_value)
        try:
            canonical = path.resolve(strict=True)
        except OSError as error:
            raise protocol.ProtocolError(
                f"allocation probe dependency path cannot be resolved: {name}"
            ) from error
        if not path.is_absolute() or path != expected_path or canonical != expected_path:
            raise protocol.ProtocolError(f"allocation probe dependency path mismatch: {name}")
        expected_features = ["cpu-faer"] if name != "tenferro-tensor" else None
        if item.get("features") != expected_features:
            if not (expected_features is None and "features" not in item):
                raise protocol.ProtocolError(
                    f"allocation probe dependency features mismatch: {name}"
                )


def _write_new_regular(path: pathlib.Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o644)
    except BaseException as error:
        if isinstance(error, Exception):
            raise protocol.ProtocolError(
                f"cannot create generated probe file {path}: {error}"
            ) from error
        raise
    primary: BaseException | None = None
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short generated probe write")
            offset += written
        os.fsync(descriptor)
    except BaseException as error:
        if isinstance(error, protocol.ProtocolError) or not isinstance(error, Exception):
            primary = error
            raise
        wrapped = protocol.ProtocolError(f"cannot write generated probe file {path}: {error}")
        primary = wrapped
        raise wrapped from error
    finally:
        if descriptor is not None:
            owned_descriptor = descriptor
            descriptor = None
            try:
                os.close(owned_descriptor)
            except BaseException as error:
                if primary is not None:
                    _record_suppressed_failure(primary, "generated probe file close", error)
                elif isinstance(error, Exception):
                    raise protocol.ProtocolError(
                        f"cannot close generated probe file {path}: {error}"
                    ) from error
                else:
                    raise


def _validate_generated_probe_inventory(root: pathlib.Path, *, lock_required: bool) -> None:
    expected_root = {"Cargo.toml", "src"}
    if lock_required:
        expected_root.add("Cargo.lock")
    try:
        entries = {entry.name: entry for entry in os.scandir(root)}
    except OSError as error:
        raise protocol.ProtocolError(f"cannot inspect generated probe inventory: {error}") from error
    if set(entries) != expected_root:
        raise protocol.ProtocolError("generated allocation probe inventory mismatch")
    for name in expected_root - {"src"}:
        if entries[name].is_symlink() or not entries[name].is_file(follow_symlinks=False):
            raise protocol.ProtocolError(f"generated probe path is not regular: {name}")
    if entries["src"].is_symlink() or not entries["src"].is_dir(follow_symlinks=False):
        raise protocol.ProtocolError("generated probe src is not a regular directory")
    src_entries = {entry.name: entry for entry in os.scandir(root / "src")}
    if set(src_entries) != {"main.rs", "tests.rs"}:
        raise protocol.ProtocolError("generated allocation probe src inventory mismatch")
    if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in src_entries.values()):
        raise protocol.ProtocolError("generated allocation probe source is not regular")


def _probe_digests(
    root: pathlib.Path, template: bytes, sources: Mapping[pathlib.Path, bytes]
) -> tuple[str, dict[str, str], str, dict[str, str]]:
    source_sha256 = {str(path): sha256_bytes(payload) for path, payload in sources.items()}
    generated_sources = {
        str(path): protocol.sha256_file(root / path) for path in ALLOCATION_PROBE_SOURCES
    }
    return (
        sha256_bytes(template),
        source_sha256,
        protocol.sha256_file(root / "Cargo.toml"),
        generated_sources,
    )


def _validate_probe_state(
    repository: pathlib.Path,
    root: pathlib.Path,
    template: bytes,
    sources: Mapping[pathlib.Path, bytes],
    *,
    lock_required: bool,
) -> None:
    _, current_template, current_sources = _allocation_probe_input_bytes(repository)
    if current_template != template or current_sources != sources:
        raise protocol.ProtocolError("allocation probe source changed during verification")
    _validate_generated_probe_inventory(root, lock_required=lock_required)
    if _read_regular_bytes(root / "Cargo.toml") != _render_allocation_probe_manifest(
        template, repository
    ):
        raise protocol.ProtocolError("generated allocation probe manifest changed")
    for relative, payload in sources.items():
        if _read_regular_bytes(root / relative) != payload:
            raise protocol.ProtocolError(f"generated allocation probe source changed: {relative}")


def _seed_probe_cargo_home(cargo_home: pathlib.Path, cache_source: pathlib.Path) -> None:
    cargo_home.mkdir()
    cache_source = pathlib.Path(cache_source)
    for name in ("registry", "git"):
        source = cache_source / name
        try:
            metadata = source.lstat()
            canonical = source.resolve(strict=True)
        except OSError as error:
            raise protocol.ProtocolError(f"cannot inspect Cargo cache {source}: {error}") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise protocol.ProtocolError(f"Cargo cache is not a directory: {source}")
        os.symlink(canonical, cargo_home / name, target_is_directory=True)
    (cargo_home / ".package-cache").touch()
    validate_controlled_cargo_home(cargo_home)


def _regular_executable_sha256(path: pathlib.Path) -> str:
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot inspect allocation probe binary {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or canonical != path or not os.access(path, os.X_OK):
        raise protocol.ProtocolError("allocation probe binary is not a canonical executable")
    return protocol.sha256_file(path)


def _require_probe_command(
    step: CommandSpec,
    result: CommandResult,
    *,
    cwd: pathlib.Path,
    environment: Mapping[str, str],
) -> None:
    if result.argv != step.argv or result.deadline_seconds != step.deadline_seconds:
        raise protocol.ProtocolError(f"allocation probe {step.name} result identity mismatch")
    if result.cwd != str(cwd) or result.environment != dict(sorted(environment.items())):
        raise protocol.ProtocolError(f"allocation probe {step.name} result provenance mismatch")
    if result.validity_state != "COMPLETE" or result.returncode != 0:
        raise protocol.ProtocolError(
            f"allocation probe {step.name} failed: {result.failure_reason or 'nonzero-exit'}"
        )


def _cleanup_probe_root(root: pathlib.Path, primary: BaseException | None) -> None:
    try:
        shutil.rmtree(root)
    except BaseException as error:
        if primary is not None:
            _record_suppressed_failure(primary, "allocation probe root cleanup", error)
        elif isinstance(error, Exception):
            raise protocol.ProtocolError(
                f"cannot clean allocation probe temporary root {root}: {error}"
            ) from error
        else:
            raise


def _verify_allocation_probe_with_dependencies(
    repository: pathlib.Path,
    *,
    cargo: ResolvedTool,
    command_runner: Callable[..., CommandResult],
    temporary_root_factory: Callable[[], pathlib.Path],
    cache_source: pathlib.Path,
) -> AllocationProbeVerification:
    """Test seam for the authoritative five-step allocation-probe verifier."""
    if not isinstance(cargo, ResolvedTool) or cargo.name != "cargo":
        raise protocol.ProtocolError("allocation probe verifier requires Cargo identity")
    validate_resolved_tool(cargo)
    source_root, template, sources = _allocation_probe_input_bytes(repository)
    del source_root
    root: pathlib.Path | None = None
    primary: BaseException | None = None
    try:
        root = pathlib.Path(temporary_root_factory())
        root_metadata = root.lstat()
        if (
            not root.is_absolute()
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root.resolve(strict=True) != root
        ):
            raise protocol.ProtocolError("allocation probe temporary root is invalid")
        if any(root.iterdir()):
            raise protocol.ProtocolError("allocation probe temporary root is not empty")
        generated = root / "generated"
        generated_src = generated / "src"
        target = root / "target"
        home = root / "home"
        cargo_home = root / "cargo-home"
        generated_src.mkdir(parents=True)
        target.mkdir()
        home.mkdir()
        _seed_probe_cargo_home(cargo_home, cache_source)
        manifest_payload = _render_allocation_probe_manifest(template, repository)
        _write_new_regular(generated / "Cargo.toml", manifest_payload)
        for relative, payload in sources.items():
            _write_new_regular(generated / relative, payload)
        _validate_probe_state(
            repository, generated, template, sources, lock_required=False
        )

        git_path = shutil.which("git")
        if git_path is None:
            raise protocol.ProtocolError("cannot locate Git for the sealed Cargo PATH")
        path_components = [cargo.path.parent, pathlib.Path(git_path).resolve().parent]
        controlled_path = os.pathsep.join(
            str(path) for index, path in enumerate(path_components) if path not in path_components[:index]
        )
        environment = protocol.cargo_environment(
            path=controlled_path,
            home=str(home),
            cargo_home=str(cargo_home),
            target_dir=str(target),
        )
        manifest = generated / "Cargo.toml"
        binary = target / "release" / ALLOCATION_PROBE_BINARY
        plan = allocation_probe_command_plan(manifest, binary, str(cargo.path))
        lock_sha256: str | None = None
        binary_sha256: str | None = None
        list_output: str | None = None
        for index, step in enumerate(plan):
            executable_identity = cargo if index < 4 else None
            if step.name == "list-cases":
                binary_sha256 = _regular_executable_sha256(binary)
            result = command_runner(
                step.argv,
                cwd=generated,
                environment=environment,
                deadline_seconds=step.deadline_seconds,
                executable_identity=executable_identity,
            )
            _require_probe_command(
                step, result, cwd=generated, environment=environment
            )
            if step.name == "test":
                _validate_probe_state(
                    repository, generated, template, sources, lock_required=True
                )
                lock_sha256 = protocol.sha256_file(generated / "Cargo.lock")
            elif index < 1:
                _validate_probe_state(
                    repository, generated, template, sources, lock_required=False
                )
            else:
                _validate_probe_state(
                    repository, generated, template, sources, lock_required=True
                )
                if lock_sha256 is None or protocol.sha256_file(
                    generated / "Cargo.lock"
                ) != lock_sha256:
                    raise protocol.ProtocolError("allocation probe Cargo.lock changed")
            if step.name == "list-cases":
                if binary_sha256 != _regular_executable_sha256(binary):
                    raise protocol.ProtocolError("allocation probe binary changed during list-cases")
                if result.stderr:
                    raise protocol.ProtocolError("allocation probe list-cases wrote stderr")
                list_output = result.stdout

        expected_output = json.dumps(
            list(protocol.CANONICAL_CASES), separators=(",", ":")
        ) + "\n"
        if list_output != expected_output:
            raise protocol.ProtocolError("allocation probe list-cases inventory mismatch")
        decoded = json.loads(list_output)
        if type(decoded) is not list or any(type(case) is not str for case in decoded):
            raise protocol.ProtocolError("allocation probe list-cases schema mismatch")
        inventory = tuple(decoded)
        if inventory != tuple(protocol.CANONICAL_CASES):
            raise protocol.ProtocolError("allocation probe list-cases order mismatch")
        if lock_sha256 is None or binary_sha256 is None:
            raise AssertionError("allocation probe proof was not completed")
        template_sha256, source_sha256, manifest_sha256, generated_sha256 = _probe_digests(
            generated, template, sources
        )
        return AllocationProbeVerification(
            template_sha256=template_sha256,
            source_sha256=source_sha256,
            generated_manifest_sha256=manifest_sha256,
            generated_source_sha256=generated_sha256,
            lock_sha256=lock_sha256,
            binary_sha256=binary_sha256,
            case_inventory=inventory,
        )
    except OSError as error:
        wrapped = protocol.ProtocolError(
            f"allocation probe verification filesystem failure: {error}"
        )
        primary = wrapped
        raise wrapped from error
    except BaseException as error:
        primary = error
        raise
    finally:
        if root is not None:
            _cleanup_probe_root(root, primary)


def _discover_cargo() -> ResolvedTool:
    rustup = shutil.which("rustup")
    if rustup is None:
        raise protocol.ProtocolError("cannot locate rustup to resolve Cargo")
    try:
        completed = subprocess.run(
            [rustup, "which", "cargo"],
            capture_output=True,
            text=True,
            timeout=ALLOCATION_PROBE_LIST_DEADLINE_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise protocol.ProtocolError(f"cannot resolve Cargo through rustup: {error}") from error
    if completed.returncode != 0 or not completed.stdout.strip():
        raise protocol.ProtocolError(f"rustup cannot resolve Cargo: {completed.stderr.strip()}")
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise protocol.ProtocolError("rustup returned an ambiguous Cargo path")
    return _resolve_tool("cargo", pathlib.Path(lines[0]))


def verify_allocation_probe(repository: pathlib.Path) -> AllocationProbeVerification:
    """Verify the tracked external allocation probe in fresh owned state."""
    cargo = _discover_cargo()
    cache_path = pathlib.Path(
        os.environ.get("CARGO_HOME", str(pathlib.Path.home() / ".cargo"))
    )
    try:
        cache_source = cache_path.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot resolve Cargo cache source {cache_path}: {error}"
        ) from error
    return _verify_allocation_probe_with_dependencies(
        pathlib.Path(repository),
        cargo=cargo,
        command_runner=run_bounded_command,
        temporary_root_factory=lambda: pathlib.Path(
            tempfile.mkdtemp(prefix="phase2e-allocation-probe.")
        ),
        cache_source=cache_source,
    )


def _probe_build_manifest(
    spec: AllocationProbeBuildSpec,
    *,
    tenferro_manifest: Mapping[str, Any],
    generated: pathlib.Path,
    owned_lock: pathlib.Path,
    template: bytes,
    sources: Mapping[pathlib.Path, bytes],
    executable: pathlib.Path,
    inventory: tuple[str, ...],
    build_commands: tuple[CommandSpec, ...],
    build_environment: Mapping[str, str],
    runtime_environment: Mapping[str, str],
    target: str,
    cargo_config_chain: list[dict[str, str]],
    resolved_features: str,
) -> dict[str, Any]:
    template_sha256, source_sha256, manifest_sha256, generated_sha256 = (
        _probe_digests(generated, template, sources)
    )
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "role": spec.role,
        "head": tenferro_manifest["head"],
        "target": target,
        "profile": spec.profile,
        "validity_state": "COMPLETE",
        "generated_root": str(generated.resolve()),
        "target_dir": str(executable.parents[1].resolve()),
        "executable": str(executable.resolve()),
        "executable_sha256": _regular_executable_sha256(executable),
        "lock_name": spec.lock_name,
        "lock_sha256": protocol.sha256_file(owned_lock),
        "cargo_config_chain": cargo_config_chain,
        "config_chain_sha256": protocol.sha256_json(cargo_config_chain),
        "resolved_features": resolved_features,
        "resolved_features_sha256": sha256_bytes(resolved_features.encode("utf-8")),
        "template_sha256": template_sha256,
        "source_sha256": source_sha256,
        "generated_manifest_sha256": manifest_sha256,
        "generated_source_sha256": generated_sha256,
        "case_inventory": list(inventory),
        "repetitions": 4096,
        "build_commands": [command.to_manifest() for command in build_commands],
        "build_environment": dict(build_environment),
        "environment": dict(runtime_environment),
        "toolchain_sha256": protocol.sha256_json(tenferro_manifest.get("toolchain", {})),
        "tenferro_build_manifest_sha256": protocol.sha256_json(tenferro_manifest),
    }


def _build_allocation_probe_set_with_dependencies(
    config: BuildConfig,
    tenferro_manifests: Mapping[str, Mapping[str, Any]],
    *,
    tools: ResolvedToolchain,
    command_runner: Callable[..., CommandResult],
) -> dict[str, dict[str, Any]]:
    """Build three role-bound probes after timing builds establish authority."""
    validate_resolved_toolchain(tools)
    specs = allocation_probe_build_specs(config, tenferro_manifests)
    _source_root, template, sources = _allocation_probe_input_bytes(config.repository)
    generated_by_role: dict[str, pathlib.Path] = {}
    target_by_role: dict[str, pathlib.Path] = {}
    build_environments: dict[str, dict[str, str]] = {}
    for spec in specs:
        generated = protocol.prepare_empty_root(
            pathlib.Path(config.scratch_root) / "allocation-probes" / spec.role
        )
        (generated / "src").mkdir()
        target = protocol.prepare_empty_root(
            pathlib.Path(config.scratch_root) / "allocation-probe-targets" / spec.role
        )
        _write_new_regular(
            generated / "Cargo.toml",
            _render_allocation_probe_manifest(template, spec.repository),
        )
        for relative, payload in sources.items():
            _write_new_regular(generated / relative, payload)
        generated_by_role[spec.role] = generated
        target_by_role[spec.role] = target
        build_environments[spec.role] = protocol.cargo_environment(
            path=tools.path,
            home=str(pathlib.Path(config.home).resolve()),
            cargo_home=str(pathlib.Path(config.cargo_home).resolve()),
            target_dir=str(target.resolve()),
        )
        _allocation_probe_cargo_config_chain(generated)

    lock_generators = {
        "direct-probe": specs[0].role,
        "common-probe": "candidate",
    }
    owned_locks: dict[str, pathlib.Path] = {}
    for lock_name, role in lock_generators.items():
        result = _run_build_command(
            command_runner,
            _absolute_tool_command(LOCK_COMMAND, tools.cargo),
            cwd=generated_by_role[role],
            environment=build_environments[role],
            deadline_seconds=QUERY_DEADLINE_SECONDS,
            executable_identity=tools.cargo,
        )
        if result.validity_state != "COMPLETE":
            raise protocol.ProtocolError(f"{lock_name} generation was inconclusive")
        owned_locks[lock_name] = copy_root_owned_lock(
            config.evidence_root,
            lock_name,
            generated_by_role[role] / "Cargo.lock",
        )

    manifests: dict[str, dict[str, Any]] = {}
    expected_inventory = tuple(protocol.CANONICAL_CASES)
    for spec in specs:
        generated = generated_by_role[spec.role]
        target = target_by_role[spec.role]
        owned_lock = owned_locks[spec.lock_name]
        install_root_owned_lock(owned_lock, generated)
        binary = target / "release" / ALLOCATION_PROBE_BINARY
        host_target = tenferro_manifests[spec.role]["target"]
        plan = allocation_probe_build_only_command_plan(
            generated / "Cargo.toml", binary, str(tools.cargo.path), host_target
        )
        list_output = None
        resolved_features = None
        for step in plan:
            binary_before = (
                _regular_executable_sha256(binary)
                if step.name == "list-cases"
                else None
            )
            result = command_runner(
                step.argv,
                cwd=generated,
                environment=build_environments[spec.role],
                deadline_seconds=step.deadline_seconds,
                executable_identity=tools.cargo
                if step.name in ("features", "build")
                else None,
            )
            if step.name == "features":
                if not result.stdout.strip() or result.stderr:
                    raise protocol.ProtocolError(
                        "allocation probe resolved feature graph is invalid"
                    )
                resolved_features = result.stdout
            _require_probe_command(
                step,
                result,
                cwd=generated,
                environment=build_environments[spec.role],
            )
            if step.name == "list-cases":
                if binary_before != _regular_executable_sha256(binary):
                    raise protocol.ProtocolError(
                        "allocation probe executable changed during inventory query"
                    )
                if result.stderr:
                    raise protocol.ProtocolError("allocation probe inventory wrote stderr")
                list_output = result.stdout
        expected_output = json.dumps(list(expected_inventory), separators=(",", ":")) + "\n"
        if list_output != expected_output or resolved_features is None:
            raise protocol.ProtocolError(f"{spec.role} allocation probe inventory mismatch")
        _current_root, current_template, current_sources = _allocation_probe_input_bytes(
            config.repository
        )
        if current_template != template or current_sources != sources:
            raise protocol.ProtocolError("allocation probe source changed during role builds")
        _validate_generated_probe_inventory(generated, lock_required=True)
        if _read_regular_bytes(generated / "Cargo.toml") != _render_allocation_probe_manifest(
            template, spec.repository
        ):
            raise protocol.ProtocolError(f"{spec.role} generated probe manifest changed")
        for relative, payload in sources.items():
            if _read_regular_bytes(generated / relative) != payload:
                raise protocol.ProtocolError(
                    f"{spec.role} generated probe source changed: {relative}"
                )
        if protocol.sha256_file(generated / "Cargo.lock") != protocol.sha256_file(owned_lock):
            raise protocol.ProtocolError(f"{spec.role} allocation probe lock changed")
        runtime_environment = protocol.runtime_environment(
            path=tools.path, home=str(pathlib.Path(config.home).resolve())
        )
        manifest = _probe_build_manifest(
            spec,
            tenferro_manifest=tenferro_manifests[spec.role],
            generated=generated,
            owned_lock=owned_lock,
            template=template,
            sources=sources,
            executable=binary,
            inventory=expected_inventory,
            build_commands=plan,
            build_environment=build_environments[spec.role],
            runtime_environment=runtime_environment,
            target=host_target,
            cargo_config_chain=_allocation_probe_cargo_config_chain(generated),
            resolved_features=resolved_features,
        )
        spec.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        protocol.atomic_write_json(spec.manifest_path, manifest)
        spec.manifest_path.chmod(0o444)
        manifests[spec.role] = manifest
    return _validate_allocation_probe_set_with_dependencies(
        config.evidence_root,
        tenferro_manifests,
        repository=config.repository,
        command_runner=command_runner,
    )


def build_allocation_probe_set(config: BuildConfig) -> dict[str, dict[str, Any]]:
    """Validate timing builds and create three immutable allocation probes."""
    tenferro_manifests = validate_build_set(config)
    tools = resolve_toolchain(config.path)
    validate_controlled_cargo_home(config.cargo_home)
    return _build_allocation_probe_set_with_dependencies(
        config,
        tenferro_manifests,
        tools=tools,
        command_runner=run_bounded_command,
    )


def _validate_allocation_probe_set_with_dependencies(
    evidence_root: pathlib.Path,
    tenferro_manifests: Mapping[str, Mapping[str, Any]],
    *,
    repository: pathlib.Path,
    command_runner: Callable[..., CommandResult] = run_bounded_command,
) -> dict[str, dict[str, Any]]:
    """Reopen and validate the three persisted role-bound probe manifests."""
    evidence_root = pathlib.Path(evidence_root)
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
    lock_digests = {
        name: sha256_bytes(_read_regular_bytes(evidence_root / LOCK_PATHS[name]))
        for name in ("direct-probe", "common-probe")
    }
    manifests = {}
    _source_root, tracked_template, tracked_sources = _allocation_probe_input_bytes(
        pathlib.Path(repository)
    )
    reference_source = None
    for role, relative in PROBE_BUILD_MANIFEST_PATHS.items():
        path = evidence_root / relative
        decoded = protocol.decode_canonical_json_bytes(
            _read_regular_bytes(path), f"probe build manifest {path}"
        )
        if type(decoded) is not dict or set(decoded) != required:
            raise protocol.ProtocolError(f"{role} probe build manifest schema mismatch")
        expected_lock = (
            "direct-probe"
            if role == "direct-current-main-baseline"
            else "common-probe"
        )
        tenferro = tenferro_manifests.get(role)
        if (
            type(decoded["protocol_version"]) is not int
            or decoded["protocol_version"] != protocol.PROTOCOL_VERSION
            or decoded["role"] != role
            or decoded["profile"] != "bench"
            or decoded["validity_state"] != "COMPLETE"
            or decoded["lock_name"] != expected_lock
            or decoded["lock_sha256"] != lock_digests[expected_lock]
            or decoded["case_inventory"] != list(protocol.CANONICAL_CASES)
            or type(decoded["repetitions"]) is not int
            or decoded["repetitions"] != 4096
            or type(tenferro) is not dict
            or decoded["head"] != tenferro.get("head")
            or decoded["target"] != tenferro.get("target")
            or decoded["tenferro_build_manifest_sha256"]
            != protocol.sha256_json(tenferro)
            or decoded["toolchain_sha256"]
            != protocol.sha256_json(tenferro.get("toolchain", {}))
        ):
            raise protocol.ProtocolError(f"{role} probe build identity mismatch")
        for field in (
            "executable_sha256",
            "lock_sha256",
            "config_chain_sha256",
            "resolved_features_sha256",
            "template_sha256",
            "generated_manifest_sha256",
            "toolchain_sha256",
            "tenferro_build_manifest_sha256",
        ):
            _validate_sha256(decoded[field], f"{role} probe {field}")
        for mapping_name in ("source_sha256", "generated_source_sha256"):
            mapping = decoded[mapping_name]
            if type(mapping) is not dict or set(mapping) != {
                str(path) for path in ALLOCATION_PROBE_SOURCES
            }:
                raise protocol.ProtocolError(f"{role} probe source digest schema mismatch")
            for name, digest in mapping.items():
                _validate_sha256(digest, f"{role} probe {mapping_name}/{name}")
        expected_source_sha256 = {
            str(path): sha256_bytes(payload) for path, payload in tracked_sources.items()
        }
        if (
            decoded["template_sha256"] != sha256_bytes(tracked_template)
            or decoded["source_sha256"] != expected_source_sha256
            or decoded["generated_source_sha256"] != expected_source_sha256
        ):
            raise protocol.ProtocolError(f"{role} generated probe source digest mismatch")
        if (
            decoded["cargo_config_chain"]
            != _allocation_probe_cargo_config_chain(
                generated_root := pathlib.Path(decoded["generated_root"])
            )
            or decoded["config_chain_sha256"]
            != protocol.sha256_json(decoded["cargo_config_chain"])
            or type(decoded["resolved_features"]) is not str
            or not decoded["resolved_features"].strip()
            or decoded["resolved_features_sha256"]
            != sha256_bytes(decoded["resolved_features"].encode("utf-8"))
        ):
            raise protocol.ProtocolError(f"{role} probe Cargo proof mismatch")
        source_identity = (
            decoded["template_sha256"],
            decoded["source_sha256"],
            decoded["generated_source_sha256"],
        )
        if reference_source is None:
            reference_source = source_identity
        elif source_identity != reference_source:
            raise protocol.ProtocolError("probe source identity differs across roles")
        executable = pathlib.Path(decoded["executable"])
        tenferro_root = _canonical_directory(
            pathlib.Path(tenferro["worktree"]), f"{role} tenferro worktree"
        )
        expected_generated_root = (
            tenferro_root.parent / "allocation-probes" / role
        )
        generated_root = _canonical_directory(
            generated_root, f"{role} generated probe root"
        )
        if generated_root != expected_generated_root:
            raise protocol.ProtocolError(
                f"{role} generated probe root is not authoritative"
            )
        _validate_generated_probe_inventory(generated_root, lock_required=True)
        if _read_regular_bytes(
            generated_root / "Cargo.toml"
        ) != _render_allocation_probe_manifest(tracked_template, tenferro_root):
            raise protocol.ProtocolError(f"{role} persisted generated manifest differs")
        if sha256_bytes(
            _read_regular_bytes(generated_root / "Cargo.toml")
        ) != decoded["generated_manifest_sha256"]:
            raise protocol.ProtocolError(
                f"{role} persisted generated manifest digest differs"
            )
        for relative, payload in tracked_sources.items():
            if _read_regular_bytes(generated_root / relative) != payload:
                raise protocol.ProtocolError(
                    f"{role} persisted generated source differs: {relative}"
                )
        if sha256_bytes(_read_regular_bytes(generated_root / "Cargo.lock")) != decoded[
            "lock_sha256"
        ]:
            raise protocol.ProtocolError(f"{role} persisted generated lock differs")
        target_dir = _canonical_directory(
            pathlib.Path(decoded["target_dir"]), f"{role} probe target directory"
        )
        expected_target_dir = (
            tenferro_root.parent / "allocation-probe-targets" / role
        )
        if target_dir != expected_target_dir:
            raise protocol.ProtocolError(f"{role} probe target is not authoritative")
        if executable.parent.parent != target_dir:
            raise protocol.ProtocolError(f"{role} probe executable is outside target")
        if _regular_executable_sha256(executable) != decoded["executable_sha256"]:
            raise protocol.ProtocolError(f"{role} probe executable changed")
        build_environment = decoded["build_environment"]
        if type(build_environment) is not dict or build_environment != protocol.cargo_environment(
            path=build_environment.get("PATH"),
            home=build_environment.get("HOME"),
            cargo_home=build_environment.get("CARGO_HOME"),
            target_dir=str(target_dir),
        ):
            raise protocol.ProtocolError(f"{role} probe build environment is not sealed")
        tenferro_environment = tenferro.get("environment")
        if (
            type(tenferro_environment) is not dict
            or any(
                build_environment[name] != tenferro_environment[name]
                for name in ("PATH", "HOME", "CARGO_HOME")
            )
        ):
            raise protocol.ProtocolError(f"{role} probe build environment differs")
        commands = decoded["build_commands"]
        if type(commands) is not list or not commands or type(commands[0]) is not dict:
            raise protocol.ProtocolError(f"{role} probe build command schema mismatch")
        argv = commands[0].get("argv")
        if type(argv) is not list or not argv:
            raise protocol.ProtocolError(f"{role} probe build command is missing")
        tools = _toolchain_from_manifest(tenferro["toolchain"])
        if argv[0] != str(tools.cargo.path):
            raise protocol.ProtocolError(f"{role} probe build Cargo identity differs")
        expected_commands = [
            command.to_manifest()
            for command in allocation_probe_build_only_command_plan(
                generated_root / "Cargo.toml", executable, argv[0], decoded["target"]
            )
        ]
        if commands != expected_commands:
            raise protocol.ProtocolError(f"{role} probe build commands differ")
        feature_step = allocation_probe_build_only_command_plan(
            generated_root / "Cargo.toml", executable, argv[0], decoded["target"]
        )[0]
        feature_result = command_runner(
            feature_step.argv,
            cwd=generated_root,
            environment=build_environment,
            deadline_seconds=feature_step.deadline_seconds,
            executable_identity=tools.cargo,
        )
        _require_probe_command(
            feature_step,
            feature_result,
            cwd=generated_root,
            environment=build_environment,
        )
        if feature_result.stderr or feature_result.stdout != decoded["resolved_features"]:
            raise protocol.ProtocolError(f"{role} persisted resolved feature graph differs")
        _validate_generated_probe_inventory(generated_root, lock_required=True)
        if (
            _read_regular_bytes(generated_root / "Cargo.toml")
            != _render_allocation_probe_manifest(tracked_template, tenferro_root)
            or any(
                _read_regular_bytes(generated_root / relative) != payload
                for relative, payload in tracked_sources.items()
            )
            or sha256_bytes(_read_regular_bytes(generated_root / "Cargo.lock"))
            != decoded["lock_sha256"]
            or _regular_executable_sha256(executable)
            != decoded["executable_sha256"]
        ):
            raise protocol.ProtocolError(
                f"{role} persisted probe changed during feature validation"
            )
        environment = decoded["environment"]
        if type(environment) is not dict or environment != protocol.runtime_environment(
            path=environment.get("PATH"), home=environment.get("HOME")
        ):
            raise protocol.ProtocolError(f"{role} probe runtime environment is not sealed")
        if any(
            environment[name] != tenferro_environment[name]
            for name in ("PATH", "HOME")
        ):
            raise protocol.ProtocolError(f"{role} probe runtime environment differs")
        manifests[role] = decoded
    direct = manifests["direct-current-main-baseline"]
    common = manifests["common-lock-normalized-baseline"]
    if (
        direct["head"] == common["head"]
        or direct["executable"] == common["executable"]
    ):
        raise protocol.ProtocolError(
            "direct and common allocation baselines are not distinct"
        )
    return manifests


def validate_allocation_probe_set(
    evidence_root: pathlib.Path,
    tenferro_manifests: Mapping[str, Mapping[str, Any]],
    *,
    repository: pathlib.Path,
) -> dict[str, dict[str, Any]]:
    """Authoritatively re-run and validate persisted allocation-probe evidence."""
    return _validate_allocation_probe_set_with_dependencies(
        evidence_root,
        tenferro_manifests,
        repository=repository,
        command_runner=run_bounded_command,
    )


def _canonical_directory(path: pathlib.Path, context: str) -> pathlib.Path:
    path = pathlib.Path(path)
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot inspect {context} {path}: {error}") from error
    if not stat.S_ISDIR(metadata.st_mode) or canonical != path:
        raise protocol.ProtocolError(f"{context} is not a canonical regular directory: {path}")
    return canonical


def sha256_bytes(payload: bytes) -> str:
    """Hash an in-memory immutable build input."""
    return hashlib.sha256(payload).hexdigest()


def _normalized_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    environment = dict(manifest["environment"])
    if environment.get("CARGO_TARGET_DIR") != manifest["target_dir"]:
        raise protocol.ProtocolError("CARGO_TARGET_DIR is not bound to role target_dir")
    environment["CARGO_TARGET_DIR"] = "<ROLE_TARGET_DIR>"
    return environment


def _validate_relative_path(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_absolute() or path == pathlib.Path(".") or ".." in path.parts:
        raise protocol.ProtocolError(f"inventory path must be relative and contained: {path}")
    return path


def _validate_commit(value: str, context: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise protocol.ProtocolError(f"{context} must be a full lowercase SHA-1")


def _validate_sha256(value: str, context: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise protocol.ProtocolError(f"{context} must be a lowercase SHA-256")


def main(argv: list[str] | None = None) -> int:
    """Run one strict Phase 2E build-helper subcommand."""
    parser = argparse.ArgumentParser(prog="phase2e_build.py", exit_on_error=False)
    subparsers = parser.add_subparsers(dest="command")
    verify = subparsers.add_parser("verify-allocation-probe", exit_on_error=False)
    verify.add_argument("--repository", required=True)
    try:
        arguments = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit) as error:
        print(f"phase2e build error: {error}", file=sys.stderr)
        return 2
    if arguments.command != "verify-allocation-probe":
        print("phase2e build error: a supported subcommand is required", file=sys.stderr)
        return 2
    try:
        verify_allocation_probe(pathlib.Path(arguments.repository))
    except protocol.ProtocolError as error:
        print(f"phase2e build error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
