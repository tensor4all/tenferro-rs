#!/usr/bin/env python3
"""Own, validate, and preserve one atomic Phase 2E evidence campaign."""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import json
import os
import pathlib
import re
import secrets
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import phase2e_build as build
from scripts import phase2e_protocol as protocol


AGGREGATE_MANIFEST = "phase2e-evidence.json"
PROGRESS_MANIFEST = ".phase2e-progress.json"
STAGE_CONTEXT = ".phase2e-stage-context.json"
ABANDONMENT_SEAL = "abandoned-inventory.json"
PROCESS_JOURNAL = ".phase2e-process-journal.json"
TIMING_BUILD_FAILURE = "timing-build-failure.json"
ORCHESTRATOR_LOCK_NAME = protocol.ORCHESTRATOR_LOCK_NAME
INDEX_PATH = pathlib.Path("docs/worklogs/2026-07-21-phase-2e-index.json")
INDEX_LOCK_PATH = pathlib.Path("docs/worklogs/.phase2e-index.lock")
WORKLOG_PATH = pathlib.Path("docs/worklogs/2026-07-21-phase-2e-noninferiority.md")
PRESERVATION_BRANCH = "origin/codex/execution-engine-through-phase9"
ISSUE_NUMBER = 1436
_TIMING_BUILD_ROLES = tuple(build.BUILD_MANIFEST_PATHS)
_TIMEOUT_FAILURE_COMPONENTS = frozenset(
    {
        "term-signal-failed",
        "kill-signal-failed",
        "post-kill-drain-timeout",
        "post-kill-drain-failed",
        "term-drain-failed",
    }
)


def _has_ascii_control(value: str) -> bool:
    return any(ord(character) <= 0x1F or ord(character) == 0x7F for character in value)


def _is_canonical_issue_comment_url(url: Any) -> bool:
    return (
        type(url) is str
        and re.fullmatch(
            rf"https://github\.com/tensor4all/tenferro-rs/issues/{ISSUE_NUMBER}"
            r"#issuecomment-[1-9][0-9]*",
            url,
        )
        is not None
    )


def campaign_index_paths(repository: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Return the only repository-owned Phase 2E index and lock paths."""
    repository = pathlib.Path(repository)
    if (
        not repository.is_absolute()
        or repository.resolve(strict=True) != repository
        or not repository.is_dir()
    ):
        raise protocol.ProtocolError("Phase 2E repository is not a directory")
    return repository / INDEX_PATH, repository / INDEX_LOCK_PATH


def _canonical_repository(repository: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve the one Git repository that owns the fixed Phase 2E index."""
    requested = pathlib.Path.cwd() if repository is None else pathlib.Path(repository)
    try:
        canonical = requested.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve Phase 2E repository: {error}") from error
    if not canonical.is_dir():
        raise protocol.ProtocolError("Phase 2E repository is not a directory")
    result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=canonical,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise protocol.ProtocolError("cannot infer the Phase 2E Git repository")
    try:
        top_level = pathlib.Path(result.stdout.strip()).resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve Git top level: {error}") from error
    if repository is not None and canonical != top_level:
        raise protocol.ProtocolError("--repository must name the exact Git top level")
    return top_level


def _canonical_evidence_root(
    repository: pathlib.Path, root: pathlib.Path, *, must_exist: bool
) -> pathlib.Path:
    """Canonicalize an evidence root and keep it in the owned artifacts tree."""
    root = pathlib.Path(root)
    if not root.is_absolute():
        root = repository / root
    try:
        canonical = root.resolve(strict=must_exist)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve evidence root: {error}") from error
    artifacts = (repository / "docs" / "worklogs" / "artifacts").resolve(strict=True)
    if artifacts not in canonical.parents or canonical == artifacts:
        raise protocol.ProtocolError(
            "evidence root must be a child of docs/worklogs/artifacts"
        )
    if must_exist and not canonical.is_dir():
        raise protocol.ProtocolError("evidence root is not a directory")
    if not must_exist and canonical.exists():
        raise protocol.ProtocolError("start requires an absent evidence root")
    return canonical


def _require_fixed_index_argument(
    repository: pathlib.Path, supplied: pathlib.Path
) -> pathlib.Path:
    supplied = pathlib.Path(supplied)
    if not supplied.is_absolute():
        supplied = repository / supplied
    canonical = supplied.resolve(strict=False)
    expected, _lock = campaign_index_paths(repository)
    if canonical != expected:
        raise protocol.ProtocolError(
            f"--index must equal the fixed repository path {INDEX_PATH.as_posix()}"
        )
    return expected


def _infer_repository_for_root(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    requested = pathlib.Path(root)
    if not requested.is_absolute():
        requested = pathlib.Path.cwd() / requested
    try:
        canonical = requested.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve evidence root: {error}") from error
    if canonical != requested or not canonical.is_dir():
        raise protocol.ProtocolError(
            "evidence root must be an absolute canonical directory"
        )
    repositories = []
    for ancestor in canonical.parents:
        if (
            ancestor.name == "artifacts"
            and ancestor.parent.name == "worklogs"
            and ancestor.parent.parent.name == "docs"
        ):
            repositories.append(ancestor.parent.parent.parent)
    if len(repositories) != 1:
        raise protocol.ProtocolError(
            "evidence root does not identify one canonical repository"
        )
    repository = _canonical_repository(repositories[0])
    return repository, _canonical_evidence_root(
        repository, canonical, must_exist=True
    )

ALLOCATION_LANE_STAGES = tuple(
    f"allocation/{lane}" for lane in protocol.LANE_NAMES
)
TIMING_LANE_STAGES = tuple(f"timing/{lane}" for lane in protocol.LANE_NAMES)

STAGE_ORDER = (
    "timing-builds",
    "probe-builds",
    *ALLOCATION_LANE_STAGES,
    "dispatch-builds",
    "dispatch-gates",
    "characterization-builds",
    "characterization",
    *TIMING_LANE_STAGES,
    "aggregate-validation",
)
RETRYABLE_STAGES = frozenset((*ALLOCATION_LANE_STAGES, *TIMING_LANE_STAGES))

STAGE_WORKER_PREFIX = (
    str(pathlib.Path(sys.executable).resolve(strict=True)),
    "-m",
    "scripts.run_phase2e",
    "_stage-worker",
)


def stage_argv(
    stage: str, context: pathlib.Path, context_sha256: str
) -> tuple[str, ...]:
    """Construct the only executable shape accepted for a fixed child stage."""
    context = pathlib.Path(context)
    _require_sha(context_sha256, sha256=True, context="stage context digest")
    if stage not in STAGE_ORDER or not context.is_absolute():
        raise protocol.ProtocolError("stage worker identity is invalid")
    return (
        *STAGE_WORKER_PREFIX,
        "--stage", stage,
        "--context", str(context),
        "--context-sha256", context_sha256,
    )


def validate_stage_argv(
    stage: str,
    argv: Sequence[str],
    context: pathlib.Path,
    context_sha256: str,
) -> None:
    """Reject every caller-selected executable, shell, and argument surface."""
    if tuple(argv) != stage_argv(stage, context, context_sha256):
        raise protocol.ProtocolError("stage argv differs from the internal contract")


def command_contract_digest() -> str:
    """Digest stage order and argv templates without run-local context paths."""
    return protocol.sha256_json(
        {
            "version": 1,
            "stages": list(STAGE_ORDER),
            "worker_prefix": list(STAGE_WORKER_PREFIX),
            "worker_arguments": [
                "--stage", "<stage>", "--context", "<absolute-path>",
                "--context-sha256", "<sha256>",
            ],
            "protocol_version": protocol.PROTOCOL_VERSION,
        }
    )


def stage_context_contract_digest(context: Mapping[str, Any]) -> str:
    """Bind the executable template to every immutable worker input byte."""
    payload = dict(context)
    payload.pop("command_contract_digest", None)
    return protocol.sha256_json(
        {"template_digest": command_contract_digest(), "context": payload}
    )


STAGE_CONTEXT_FIELDS = frozenset(
    {
        "version",
        "repository",
        "evidence_root",
        "scratch_parent",
        "candidate_sha",
        "candidate_tree_sha256",
        "reservation_id",
        "experiment_identity_digest",
        "command_contract_digest",
        "path",
        "tool_paths",
        "tool_sha256",
        "host_target",
        "home",
        "cargo_home",
        "index",
        "index_lock",
    }
)


def _validate_stage_context_record(
    context: Mapping[str, Any],
) -> None:
    """Validate immutable context values without consulting external roots."""
    if (
        type(context) is not dict
        or set(context) != STAGE_CONTEXT_FIELDS
        or type(context["version"]) is not int
        or context["version"] != 1
    ):
        raise protocol.ProtocolError("stage context schema differs")
    path_names = (
        "repository",
        "evidence_root",
        "scratch_parent",
        "home",
        "cargo_home",
        "index",
        "index_lock",
    )
    paths = []
    for name in path_names:
        value = context[name]
        candidate = pathlib.Path(value) if type(value) is str else pathlib.Path()
        if (
            type(value) is not str
            or not candidate.is_absolute()
            or pathlib.Path(os.path.normpath(value)) != candidate
        ):
            raise protocol.ProtocolError(
                f"stage context {name} is not canonical absolute"
            )
        paths.append(candidate)
    repository, evidence, scratch, home, cargo_home, _index, _index_lock = paths
    roots = (scratch, home, cargo_home)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ) or any(
        runtime == protected
        or runtime in protected.parents
        or protected in runtime.parents
        for runtime in roots
        for protected in (repository, evidence)
    ):
        raise protocol.ProtocolError("stage filesystem roots are not disjoint")
    if type(context["path"]) is not str or not context["path"]:
        raise protocol.ProtocolError("stage context controlled PATH is invalid")
    components = context["path"].split(os.pathsep)
    if (
        len(set(components)) != len(components)
        or any(
            not component
            or not pathlib.Path(component).is_absolute()
            or pathlib.Path(os.path.normpath(component))
            != pathlib.Path(component)
            for component in components
        )
    ):
        raise protocol.ProtocolError("stage context controlled PATH is invalid")
    tool_paths = context["tool_paths"]
    if (
        type(tool_paths) is not dict
        or set(tool_paths) != {"git", "cargo", "rustc"}
        or any(
            type(tool_paths[name]) is not str
            or pathlib.Path(tool_paths[name]).name != name
            or not pathlib.Path(tool_paths[name]).is_absolute()
            or pathlib.Path(os.path.normpath(tool_paths[name]))
            != pathlib.Path(tool_paths[name])
            or pathlib.Path(tool_paths[name]).parent
            not in tuple(pathlib.Path(component) for component in components)
            for name in ("git", "cargo", "rustc")
        )
        or len(set(tool_paths.values())) != 3
    ):
        raise protocol.ProtocolError("stage context resolved tool paths are invalid")
    tool_sha256 = context["tool_sha256"]
    if (
        type(tool_sha256) is not dict
        or set(tool_sha256) != {"git", "cargo", "rustc"}
        or any(
            type(tool_sha256[name]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", tool_sha256[name]) is None
            for name in ("git", "cargo", "rustc")
        )
    ):
        raise protocol.ProtocolError("stage context tool digests are invalid")
    if (
        type(context["host_target"]) is not str
        or re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.]*"
            r"(?:-[A-Za-z0-9_][A-Za-z0-9_.]*){2,}",
            context["host_target"],
        )
        is None
    ):
        raise protocol.ProtocolError("stage context host target is invalid")
    if (
        type(context["reservation_id"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", context["reservation_id"]) is None
    ):
        raise protocol.ProtocolError("stage context reservation id is invalid")
    _require_sha(context["candidate_sha"], sha256=False, context="context candidate")
    for name in (
        "candidate_tree_sha256",
        "experiment_identity_digest",
        "command_contract_digest",
    ):
        _require_sha(context[name], sha256=True, context=name)
    if context["command_contract_digest"] != stage_context_contract_digest(context):
        raise protocol.ProtocolError("stage context command contract differs")


def load_stage_context(
    path: pathlib.Path,
    expected_sha256: str | None = None,
    *,
    require_fresh_scratch: bool = True,
) -> dict[str, Any]:
    """Load the exact immutable worker context before reserving ACTIVE."""
    path = pathlib.Path(path)
    if not path.is_absolute():
        raise protocol.ProtocolError("stage context path must be absolute")
    if pathlib.Path(os.path.normpath(path)) != path:
        raise protocol.ProtocolError("stage context path must be canonical")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot securely open stage context: {error}"
        ) from error
    primary: BaseException | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise protocol.ProtocolError("stage context is not a regular file")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            payload = stream.read()
    except OSError as error:
        wrapped = protocol.ProtocolError(
            f"cannot securely open stage context: {error}"
        )
        primary = wrapped
        raise wrapped from error
    except BaseException as error:
        primary = error
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as close_error:
                if primary is not None:
                    try:
                        primary.add_note(
                            "suppressed stage context close failure: "
                            f"{close_error}"
                        )
                    except BaseException:
                        pass
                else:
                    raise protocol.ProtocolError(
                        f"cannot close stage context: {close_error}"
                    ) from close_error
    if expected_sha256 is not None:
        _require_sha(expected_sha256, sha256=True, context="bound context digest")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise protocol.ProtocolError("stage context bytes differ from bound digest")
    context = protocol.decode_canonical_json_bytes(payload, "stage context")
    if type(context) is not dict:
        raise protocol.ProtocolError("stage context is not an object")
    _validate_stage_context_record(context)
    resolved_tools = build.resolve_toolchain(context["path"])
    expected_tools = _stage_context_toolchain(context)
    if resolved_tools != expected_tools:
        raise protocol.ProtocolError("stage context resolved tool identity differs")
    try:
        repository = pathlib.Path(context["repository"]).resolve(strict=True)
        scratch = pathlib.Path(context["scratch_parent"]).resolve(strict=True)
        home = pathlib.Path(context["home"]).resolve(strict=True)
        cargo_home = pathlib.Path(context["cargo_home"]).resolve(strict=True)
        evidence = pathlib.Path(context["evidence_root"]).resolve(strict=False)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot resolve stage context filesystem root: {error}"
        ) from error
    roots = (scratch, home, cargo_home)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ) or any(
        runtime == protected
        or runtime in protected.parents
        or protected in runtime.parents
        for runtime in roots
        for protected in (repository, evidence)
    ):
        raise protocol.ProtocolError("stage filesystem roots are not disjoint")
    if not all(path.is_dir() for path in (repository, scratch, home, cargo_home)):
        raise protocol.ProtocolError("stage repository/runtime root is not a directory")
    build.validate_controlled_cargo_home(cargo_home)
    if require_fresh_scratch and next(scratch.iterdir(), None) is not None:
        raise protocol.ProtocolError("stage scratch parent is not fresh")
    return context


def _context_path(context: Mapping[str, Any], name: str) -> pathlib.Path:
    return pathlib.Path(context[name])


def _build_config(context: Mapping[str, Any]) -> build.BuildConfig:
    return build.BuildConfig(
        repository=_context_path(context, "repository"),
        evidence_root=_context_path(context, "evidence_root"),
        scratch_root=_context_path(context, "scratch_parent"),
        candidate_commit=context["candidate_sha"],
        path=context["path"],
        home=_context_path(context, "home"),
        cargo_home=_context_path(context, "cargo_home"),
        expected_toolchain=_stage_context_toolchain(context),
        expected_host_target=context["host_target"],
    )


def _stage_context_toolchain(
    context: Mapping[str, Any],
) -> build.ResolvedToolchain:
    """Reconstruct the exact sealed tool identities without filesystem reads."""
    tools = {
        name: build.ResolvedTool(
            name=name,
            path=pathlib.Path(context["tool_paths"][name]),
            sha256=context["tool_sha256"][name],
        )
        for name in ("git", "cargo", "rustc")
    }
    return build.ResolvedToolchain(
        path=context["path"],
        git=tools["git"],
        cargo=tools["cargo"],
        rustc=tools["rustc"],
    )


def _next_attempt(context: Mapping[str, Any], stage: str, lane: str) -> int:
    ledger = _read_json(
        _context_path(context, "evidence_root") / "evidence-ledger.json",
        "stage worker ledger",
    )
    protocol.validate_ledger(ledger)
    for stage_record in ledger["stages"]:
        if stage_record["name"] == stage:
            for lane_record in stage_record["lanes"]:
                if lane_record["name"] == lane:
                    return len(lane_record["attempt_ids"]) + 1
    raise protocol.ProtocolError("stage worker lane is absent from ledger")


def _timing_builds(context: Mapping[str, Any]) -> int:
    root = _context_path(context, "evidence_root")
    identity = protocol.PreparedRootIdentity(root)
    try:
        _require_absent_timing_build_failure(identity)
        result = build.build_all(_build_config(context))
        if not isinstance(result, build.BuildSetResult):
            raise protocol.ProtocolError("timing build returned an invalid result")
        if result.validity_state == "COMPLETE":
            if result.failure is not None:
                raise protocol.ProtocolError(
                    "complete timing build retained failure evidence"
                )
            return 0
        if result.validity_state != "INCONCLUSIVE":
            raise protocol.ProtocolError("timing build validity state is invalid")
        if not isinstance(result.failure, build.CommandResult):
            raise protocol.ProtocolError(
                "inconclusive timing build omitted typed failure evidence"
            )
        payload = _timing_build_failure_payload(result.failure)
        validate_timing_build_failure(payload, context)
        _require_absent_timing_build_failure(identity)
        identity.revalidate()
        protocol.atomic_write_json_new_at(
            identity.descriptor, TIMING_BUILD_FAILURE, payload
        )
        identity.revalidate()
        return 2
    finally:
        identity.close()


def _require_absent_timing_build_failure(
    identity: protocol.PreparedRootIdentity,
) -> None:
    """Reserve the one timing-build failure artifact name without following links."""
    identity.revalidate()
    try:
        os.stat(
            TIMING_BUILD_FAILURE,
            dir_fd=identity.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot inspect timing build failure evidence: {error}"
        ) from error
    raise protocol.ProtocolError("timing build failure evidence already exists")


def _timing_build_failure_payload(
    failure: build.CommandResult,
) -> dict[str, Any]:
    """Serialize one typed command failure without dropping captured fields."""
    return {
        "version": 1,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "stage": "timing-builds",
        "validity_state": failure.validity_state,
        "failure_reason": failure.failure_reason,
        "command": {
            "argv": list(failure.argv),
            "cwd": failure.cwd,
            "environment": dict(sorted(failure.environment.items())),
            "deadline_seconds": failure.deadline_seconds,
            "returncode": failure.returncode,
            "stdout": failure.stdout,
            "stderr": failure.stderr,
            "termination": {
                "terminated": failure.terminated,
                "killed": failure.killed,
            },
            "inherited_descriptors": list(failure.inherited_descriptors),
        },
    }


def validate_timing_build_failure(
    payload: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    require_live_paths: bool = True,
) -> None:
    """Validate the exact canonical timing-build failure evidence schema."""
    if type(payload) is not dict or set(payload) != {
        "version",
        "protocol_version",
        "stage",
        "validity_state",
        "failure_reason",
        "command",
    }:
        raise protocol.ProtocolError("timing build failure evidence schema is invalid")
    if (
        type(payload["version"]) is not int
        or payload["version"] != 1
        or type(payload["protocol_version"]) is not int
        or payload["protocol_version"] != protocol.PROTOCOL_VERSION
        or payload["stage"] != "timing-builds"
        or payload["validity_state"] != "INCONCLUSIVE"
        or type(payload["failure_reason"]) is not str
        or not payload["failure_reason"]
    ):
        raise protocol.ProtocolError("timing build failure evidence header is invalid")
    command = payload["command"]
    if type(command) is not dict or set(command) != {
        "argv",
        "cwd",
        "environment",
        "deadline_seconds",
        "returncode",
        "stdout",
        "stderr",
        "termination",
        "inherited_descriptors",
    }:
        raise protocol.ProtocolError("timing build command evidence schema is invalid")
    argv = command["argv"]
    environment = command["environment"]
    returncode = command["returncode"]
    descriptors = command["inherited_descriptors"]
    if (
        type(argv) is not list
        or not argv
        or any(type(part) is not str or not part for part in argv)
        or type(command["cwd"]) is not str
        or not pathlib.Path(command["cwd"]).is_absolute()
        or type(environment) is not dict
        or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        )
        or type(command["deadline_seconds"]) is not int
        or command["deadline_seconds"] <= 0
        or (
            returncode is not None
            and type(returncode) is not int
        )
        or type(command["stdout"]) is not str
        or type(command["stderr"]) is not str
        or type(descriptors) is not list
        or len(descriptors) > 1
        or any(type(descriptor) is not int or descriptor <= 2 for descriptor in descriptors)
        or len(set(descriptors)) != len(descriptors)
    ):
        raise protocol.ProtocolError("timing build command evidence is invalid")
    termination = command["termination"]
    if (
        type(termination) is not dict
        or set(termination) != {"terminated", "killed"}
        or type(termination["terminated"]) is not bool
        or type(termination["killed"]) is not bool
    ):
        raise protocol.ProtocolError("timing build termination evidence is invalid")
    if descriptors:
        raise protocol.ProtocolError(
            "timing build commands cannot inherit executable descriptors"
        )

    role = _timing_build_failure_role(
        command["cwd"], context, require_live_paths=require_live_paths
    )
    target_dir = str(
        pathlib.Path(context["scratch_parent"]) / "targets" / role
    )
    if require_live_paths:
        expected_environment = protocol.cargo_environment(
            path=context["path"],
            home=context["home"],
            cargo_home=context["cargo_home"],
            target_dir=target_dir,
        )
    else:
        expected_environment = {
            "PATH": context["path"],
            "HOME": context["home"],
            "LC_ALL": "C",
            "TZ": "UTC",
            **protocol.THREAD_ENV,
            "CARGO_HOME": context["cargo_home"],
            "CARGO_TARGET_DIR": target_dir,
            "CARGO_INCREMENTAL": "0",
            "CARGO_NET_OFFLINE": "true",
        }
    if environment != expected_environment:
        raise protocol.ProtocolError(
            "timing build failure environment differs from sealed role environment"
        )
    expected_deadline = _timing_build_failure_deadline(
        tuple(argv),
        role=role,
        path=context["path"],
        tool_paths=context["tool_paths"],
        tool_sha256=context["tool_sha256"],
        host_target=context["host_target"],
        require_live_paths=require_live_paths,
    )
    if command["deadline_seconds"] != expected_deadline:
        raise protocol.ProtocolError(
            "timing build failure deadline differs from canonical command"
        )

    reason = payload["failure_reason"]
    terminated = termination["terminated"]
    killed = termination["killed"]
    if reason == "nonzero-exit":
        if (
            type(returncode) is not int
            or returncode == 0
            or terminated
            or killed
        ):
            raise protocol.ProtocolError(
                "nonzero timing build failure fields are inconsistent"
            )
        return
    if reason == "deadline-exceeded":
        components: list[str] = []
    elif reason.startswith("deadline-exceeded:"):
        components = reason.split(":", 1)[1].split("+")
        if (
            not components
            or any(component not in _TIMEOUT_FAILURE_COMPONENTS for component in components)
            or len(set(components)) != len(components)
        ):
            raise protocol.ProtocolError(
                "timing build timeout failure reason is invalid"
            )
    else:
        raise protocol.ProtocolError("timing build failure reason is invalid")
    prefixes: list[list[str]]
    if terminated:
        prefixes = [[], ["term-drain-failed"]]
    else:
        prefixes = [["term-signal-failed"]]
    if not killed:
        prefixes = [prefix + ["kill-signal-failed"] for prefix in prefixes]
    accepted = {
        tuple(prefix + suffix)
        for prefix in prefixes
        for suffix in (
            [],
            ["post-kill-drain-timeout"],
            ["post-kill-drain-failed"],
        )
    }
    if tuple(components) not in accepted:
        raise protocol.ProtocolError(
            "timing build timeout termination fields are inconsistent"
        )


def _timing_build_failure_role(
    cwd: str,
    context: Mapping[str, Any],
    *,
    require_live_paths: bool,
) -> str:
    """Bind a captured build cwd to one exact existing role worktree."""
    if type(context) is not dict or set(context) != STAGE_CONTEXT_FIELDS:
        raise protocol.ProtocolError("timing build stage context schema differs")
    if type(context["version"]) is not int or context["version"] != 1:
        raise protocol.ProtocolError("timing build stage context version differs")
    scratch = pathlib.Path(context["scratch_parent"])
    candidate = pathlib.Path(cwd)
    if not require_live_paths:
        if (
            not scratch.is_absolute()
            or pathlib.Path(os.path.normpath(scratch)) != scratch
            or not candidate.is_absolute()
            or pathlib.Path(os.path.normpath(candidate)) != candidate
        ):
            raise protocol.ProtocolError(
                "timing build cwd is not an exact canonical role worktree"
            )
        for role in _TIMING_BUILD_ROLES:
            if candidate == scratch / role:
                return role
        raise protocol.ProtocolError(
            "timing build cwd is not a canonical build role"
        )
    try:
        canonical_scratch = scratch.resolve(strict=True)
        canonical_candidate = candidate.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot resolve timing build role worktree: {error}"
        ) from error
    if (
        not scratch.is_absolute()
        or canonical_scratch != scratch
        or not candidate.is_absolute()
        or canonical_candidate != candidate
        or not candidate.is_dir()
    ):
        raise protocol.ProtocolError(
            "timing build cwd is not an exact canonical role worktree"
        )
    for role in _TIMING_BUILD_ROLES:
        if candidate == scratch / role:
            return role
    raise protocol.ProtocolError("timing build cwd is not a canonical build role")


def _timing_build_failure_deadline(
    argv: tuple[str, ...],
    *,
    role: str,
    path: str,
    tool_paths: Mapping[str, Any],
    tool_sha256: Mapping[str, Any],
    host_target: str,
    require_live_paths: bool,
) -> int:
    """Bind a failure to one exact command eligible for its build role."""
    if require_live_paths:
        tools = build.resolve_toolchain(path)
        expected = build.ResolvedToolchain(
            path=path,
            **{
                name: build.ResolvedTool(
                    name,
                    pathlib.Path(tool_paths[name]),
                    tool_sha256[name],
                )
                for name in ("git", "cargo", "rustc")
            },
        )
        if tools != expected:
            raise protocol.ProtocolError(
                "timing build resolved tool identity differs from stage context"
            )
    cargo = tool_paths["cargo"]
    rustc = tool_paths["rustc"]
    fixed_commands = {
        (rustc, "--version", "--verbose"): build.QUERY_DEADLINE_SECONDS,
        (cargo, "--version", "--verbose"): build.QUERY_DEADLINE_SECONDS,
        (cargo, *build.METADATA_COMMAND[1:]): build.QUERY_DEADLINE_SECONDS,
        (cargo, *build.BENCH_COMMAND[1:]): build.BUILD_DEADLINE_SECONDS,
    }
    if role in {"direct-current-main-baseline", "candidate"}:
        fixed_commands[(cargo, *build.LOCK_COMMAND[1:])] = (
            build.QUERY_DEADLINE_SECONDS
        )
    deadline = fixed_commands.get(argv)
    if deadline is not None:
        return deadline

    if argv == (cargo, *build.timing_feature_command(host_target)[1:]):
        return build.QUERY_DEADLINE_SECONDS
    raise protocol.ProtocolError(
        "timing build failure argv is not a canonical role command"
    )


def _probe_builds(context: Mapping[str, Any]) -> int:
    build.build_allocation_probe_set(_build_config(context))
    return 0


def _allocation(context: Mapping[str, Any], lane: str) -> int:
    from scripts import run_phase2e_allocation_campaign as allocation

    root = _context_path(context, "evidence_root")
    attempt = _next_attempt(context, "allocation", lane)
    return allocation.main(
        [
            "--comparison-kind", lane,
            "--ledger", str(root / "evidence-ledger.json"),
            "--attempt-id", str(attempt),
            "--artifact-root", str(root / "attempts" / "allocation" / lane / str(attempt)),
            "--working-directory", context["repository"],
            "--probe-manifest-root", str(root),
            "--tenferro-manifest-root", str(root),
            "--repository", context["repository"],
        ]
    )


def _gate_root(context: Mapping[str, Any]) -> pathlib.Path:
    return _context_path(context, "evidence_root") / "gate-collector"


def _dispatch_builds(context: Mapping[str, Any]) -> int:
    """Initialize Task 7 ownership and build only dispatch executables."""
    root = _context_path(context, "evidence_root")
    gate_root = _gate_root(context)
    protocol.prepare_empty_root(gate_root)
    common_destination = gate_root / build.LOCK_PATHS["common"]
    common_destination.parent.mkdir(parents=True)
    build._write_new_regular(
        common_destination, (root / build.LOCK_PATHS["common"]).read_bytes()
    )
    gate_scratch = _context_path(context, "scratch_parent") / "dispatch-builds"
    gate_scratch.mkdir(mode=0o700, exist_ok=False)
    build.build_dispatch_artifacts(
        repository=_context_path(context, "repository"), evidence_root=gate_root,
        scratch_root=gate_scratch, candidate=context["candidate_sha"],
        path=context["path"], home=_context_path(context, "home"),
        cargo_home=_context_path(context, "cargo_home"),
    )
    return 0


def _validate_gate_component(context: Mapping[str, Any], component: str) -> int:
    manifest = _read_json(_gate_root(context) / component / "manifest.json", component)
    if manifest.get("candidate") != context["candidate_sha"]:
        raise protocol.ProtocolError(f"{component} candidate differs")
    state = manifest.get("gating_result", manifest.get("validity_state"))
    return 0 if state == "PASS" else 2


def _dispatch_gates(context: Mapping[str, Any]) -> int:
    from scripts import run_phase2e_gates as gates

    gates.run_dispatch_gate_stage(
        repository=_context_path(context, "repository"),
        evidence_root=_gate_root(context), candidate=context["candidate_sha"],
        path=context["path"], home=_context_path(context, "home"),
    )
    return 0


def _characterization_builds(context: Mapping[str, Any]) -> int:
    gate_root = _gate_root(context)
    scratch = _context_path(context, "scratch_parent") / "characterization-builds"
    scratch.mkdir(mode=0o700, exist_ok=False)
    build.build_characterization_artifacts(
        repository=_context_path(context, "repository"), evidence_root=gate_root,
        scratch_root=scratch, candidate=context["candidate_sha"],
        path=context["path"], home=_context_path(context, "home"),
        cargo_home=_context_path(context, "cargo_home"),
    )
    return 0


def _characterization(context: Mapping[str, Any]) -> int:
    from scripts import run_phase2e_gates as gates

    scratch = _context_path(context, "scratch_parent") / "characterization"
    scratch.mkdir(mode=0o700, exist_ok=False)
    gates.run_characterization_stage(
        repository=_context_path(context, "repository"),
        evidence_root=_gate_root(context), candidate=context["candidate_sha"],
        scratch_root=scratch, path=context["path"],
        home=_context_path(context, "home"),
    )
    return 0


def _timing(context: Mapping[str, Any], lane: str) -> int:
    from scripts import run_phase1_eager_campaign as timing

    root = _context_path(context, "evidence_root")
    attempt = _next_attempt(context, "timing", lane)
    baseline_role = (
        "direct-current-main-baseline"
        if lane == "direct-current-main"
        else "common-lock-normalized-baseline"
    )
    return timing.main(
        [
            "--comparison-kind", lane,
            "--baseline-build-manifest", str(root / build.BUILD_MANIFEST_PATHS[baseline_role]),
            "--candidate-build-manifest", str(root / build.BUILD_MANIFEST_PATHS["candidate"]),
            "--repository", context["repository"],
            "--build-evidence-root", str(root),
            "--build-scratch-root", context["scratch_parent"],
            "--candidate-commit", context["candidate_sha"],
            "--controlled-path", context["path"],
            "--controlled-home", context["home"],
            "--controlled-cargo-home", context["cargo_home"],
            "--ledger", str(root / "evidence-ledger.json"),
            "--attempt-id", str(attempt),
            "--artifact-root", str(root / "attempts" / "timing" / lane / str(attempt)),
            "--criterion-root", str(_context_path(context, "scratch_parent") / f"timing-{lane}-{attempt}"),
            "--working-directory", context["repository"],
        ]
    )


def _aggregate_validation(context: Mapping[str, Any]) -> int:
    # Sealing is performed by the parent after it has durably recorded this
    # worker's own child record.  This worker proves the ledger itself is closed.
    status = _terminal_ledger_status(
        _read_json(
            _context_path(context, "evidence_root") / "evidence-ledger.json",
            "aggregate ledger",
        )
    )
    return {
        "PASS": 0,
        "VALIDITY_INCONCLUSIVE": 2,
        "FAIL": 3,
        "STATISTICAL_INCONCLUSIVE": 4,
    }[status]


STAGE_HANDLERS: dict[str, Callable[[Mapping[str, Any]], int]] = {
    "timing-builds": _timing_builds,
    "probe-builds": _probe_builds,
    "allocation/direct-current-main": lambda context: _allocation(context, "direct-current-main"),
    "allocation/common-lock-normalized": lambda context: _allocation(context, "common-lock-normalized"),
    "dispatch-builds": _dispatch_builds,
    "dispatch-gates": _dispatch_gates,
    "characterization-builds": _characterization_builds,
    "characterization": _characterization,
    "timing/direct-current-main": lambda context: _timing(context, "direct-current-main"),
    "timing/common-lock-normalized": lambda context: _timing(context, "common-lock-normalized"),
    "aggregate-validation": _aggregate_validation,
}


class GuardedRootPath(os.PathLike[str]):
    """Revalidate a held root whenever stage code materializes its pathname."""

    def __init__(self, identity: protocol.PreparedRootIdentity) -> None:
        self.identity = identity

    def __fspath__(self) -> str:
        self.identity.revalidate()
        return str(self.identity.path)

    def __str__(self) -> str:
        return self.__fspath__()


def execute_stage_worker(
    stage: str, context_path: pathlib.Path, context_sha256: str
) -> int:
    """Dispatch one exact private worker stage through its registered owner."""
    if stage not in STAGE_ORDER or stage not in STAGE_HANDLERS:
        raise protocol.ProtocolError("stage worker has no registered owner")
    context = load_stage_context(
        context_path, context_sha256, require_fresh_scratch=False
    )
    validate_worker_binding(stage, context, context_path, context_sha256)
    root_identity = protocol.PreparedRootIdentity(
        pathlib.Path(context["evidence_root"])
    )
    try:
        guarded_context = dict(context)
        guarded_context["evidence_root"] = GuardedRootPath(root_identity)
        root_identity.revalidate()
        result = STAGE_HANDLERS[stage](guarded_context)
        root_identity.revalidate()
        if type(result) is not int or result not in {0, 2, 3, 4}:
            raise protocol.ProtocolError("stage worker returned an invalid status")
        return result
    finally:
        root_identity.close()


def validate_worker_binding(
    stage: str,
    context: Mapping[str, Any],
    context_path: pathlib.Path,
    context_sha256: str,
) -> None:
    """Revalidate ACTIVE, Git candidate, and any checkpoint before stage action."""
    index = _read_index(pathlib.Path(context["index"]))
    if index_state(index) != "ACTIVE":
        raise protocol.ProtocolError("stage worker reservation is not ACTIVE")
    active = index["events"][-1]
    expected = {
        "root": context["evidence_root"],
        "reservation_id": context["reservation_id"],
        "candidate_sha": context["candidate_sha"],
        "candidate_tree_sha256": context["candidate_tree_sha256"],
        "experiment_identity_digest": context["experiment_identity_digest"],
        "command_contract_digest": context["command_contract_digest"],
        "context_sha256": context_sha256,
    }
    if any(active.get(name) != value for name, value in expected.items()):
        raise protocol.ProtocolError("stage worker identity differs from ACTIVE")
    repository = pathlib.Path(context["repository"])
    head = _git(repository, "rev-parse", "HEAD", text=True).strip()
    tree = _git(repository, "ls-tree", "-r", "-z", "--full-tree", head)
    if (
        head != context["candidate_sha"]
        or hashlib.sha256(tree).hexdigest() != context["candidate_tree_sha256"]
    ):
        raise protocol.ProtocolError("stage worker candidate provenance differs")
    progress_path = pathlib.Path(context["evidence_root"]) / PROGRESS_MANIFEST
    process_identity = _linux_process_identity(os.getpid())
    running_process = {
        "stage": stage,
        "argv": list(stage_argv(stage, context_path, context_sha256)),
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "start_ticks": process_identity["start_ticks"],
    }
    if progress_path.exists():
        progress = validate_progress(
            pathlib.Path(context["evidence_root"]),
            running_process=running_process,
        )
        validate_resume_identity(
            active,
            context,
            context_sha256,
            progress,
            context_path=context_path,
        )
    else:
        _validate_process_journal_handoff(
            pathlib.Path(context["evidence_root"]),
            protocol.sha256_json({"version": 1, "entries": []}),
            running_process,
        )

TERMINAL_STATUSES = frozenset(
    {"PASS", "FAIL", "STATISTICAL_INCONCLUSIVE", "VALIDITY_INCONCLUSIVE", "ABANDONED"}
)
RETRYABLE_STATUSES = frozenset({"VALIDITY_INCONCLUSIVE", "ABANDONED"})
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_sha(value: str, *, sha256: bool, context: str) -> None:
    expression = SHA256_RE if sha256 else SHA1_RE
    if type(value) is not str or expression.fullmatch(value) is None:
        raise protocol.ProtocolError(f"{context} is not a canonical digest")


def experiment_identity_digest(
    inventory: Iterable[tuple[str, str, str]], contract: Mapping[str, Any]
) -> str:
    """Digest the semantic Git tree and immutable campaign contract.

    Commit metadata and worklog-only blobs are intentionally excluded.  Git
    mode, path, and blob identity remain distinct inputs for every other path.
    """
    canonical = []
    for item in inventory:
        if type(item) is not tuple or len(item) != 3:
            raise protocol.ProtocolError("Git tree inventory entry is invalid")
        mode, path, blob = item
        if not all(type(value) is str for value in item) or not path:
            raise protocol.ProtocolError("Git tree inventory entry is invalid")
        if path == "docs/worklogs" or path.startswith("docs/worklogs/"):
            continue
        canonical.append((mode, path, blob))
    if len({path for _, path, _ in canonical}) != len(canonical):
        raise protocol.ProtocolError("Git tree inventory contains a duplicate path")
    payload = {"contract": dict(contract), "git_tree": sorted(canonical)}
    return protocol.sha256_json(payload)


def parse_git_tree(payload: bytes) -> tuple[tuple[str, str, str], ...]:
    """Parse ``git ls-tree -r -z --full-tree`` output without losing path bytes."""
    if type(payload) is not bytes:
        raise protocol.ProtocolError("Git tree payload must be bytes")
    result = []
    for record in payload.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, kind, blob = header.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise protocol.ProtocolError("Git tree payload is malformed") from error
        if kind != "blob" or "\x00" in path:
            raise protocol.ProtocolError("Git tree contains an unsupported entry")
        result.append((mode, path, blob))
    return tuple(result)


def git_experiment_identity(
    repository: pathlib.Path, revision: str, contract: Mapping[str, Any]
) -> str:
    """Compute the canonical experiment identity for one Git revision."""
    result = subprocess.run(
        ("git", "ls-tree", "-r", "-z", "--full-tree", revision),
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise protocol.ProtocolError("cannot read candidate Git tree")
    return experiment_identity_digest(parse_git_tree(result.stdout), contract)


def canonical_experiment_contract() -> dict[str, Any]:
    """Return the immutable protocol/build/classifier contract for Task 10."""
    from scripts import classify_criterion_noninferiority as classifier

    return {
        "version": 1,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "implementation_baseline": build.IMPLEMENTATION_BASELINE,
        "harness_commit": build.HARNESS_COMMIT,
        "old_strided": build.OLD_STRIDED,
        "common_strided": build.COMMON_STRIDED,
        "requested_features": list(build.REQUESTED_FEATURES),
        "bench_command": list(build.BENCH_COMMAND),
        "dispatch_test_commands": {
            package: list(command)
            for package, command in build.DISPATCH_TEST_COMMANDS.items()
        },
        "characterization_bench_commands": {
            owner: list(command)
            for owner, command in build.CHARACTERIZATION_BENCH_COMMANDS.items()
        },
        "criterion_threshold": classifier.THRESHOLD,
        "criterion_settings": dict(classifier.CRITERION_SETTINGS),
        "pair_orders": list(protocol.PAIR_ORDERS),
        "run_roles": list(protocol.RUN_ROLES),
        "cases": dict(protocol.CANONICAL_CASES),
        "stage_order": list(STAGE_ORDER),
    }


def _candidate_provenance(
    repository: pathlib.Path, candidate: str
) -> tuple[str, str]:
    """Resolve one exact commit and derive its tree and experiment identities."""
    _require_sha(candidate, sha256=False, context="candidate SHA")
    resolved = _git(
        repository, "rev-parse", f"{candidate}^{{commit}}", text=True
    ).strip()
    if resolved != candidate:
        raise protocol.ProtocolError("--candidate must be the exact commit SHA")
    tree = _git(repository, "ls-tree", "-r", "-z", "--full-tree", candidate)
    tree_sha256 = hashlib.sha256(tree).hexdigest()
    identity = git_experiment_identity(
        repository, candidate, canonical_experiment_contract()
    )
    validate_candidate_provenance(
        repository, candidate, tree_sha256, identity, canonical_experiment_contract()
    )
    return tree_sha256, identity


def _minimal_toolchain_path() -> str:
    """Derive the canonical minimal PATH containing Git, Cargo, and rustc."""
    directories: list[pathlib.Path] = []
    for name in ("git", "cargo", "rustc"):
        found = shutil.which(name)
        if found is None:
            raise protocol.ProtocolError(f"required executable is unavailable: {name}")
        candidate = pathlib.Path(found)
        if candidate.is_symlink() and name in {"cargo", "rustc"}:
            rustup = shutil.which("rustup")
            if rustup is None:
                raise protocol.ProtocolError(
                    f"cannot resolve rustup proxy for required executable: {name}"
                )
            result = subprocess.run(
                (rustup, "which", name),
                capture_output=True,
                check=False,
                text=True,
            )
            if result.returncode:
                raise protocol.ProtocolError(
                    f"rustup cannot resolve required executable: {name}"
                )
            candidate = pathlib.Path(result.stdout.strip())
        try:
            executable = candidate.resolve(strict=True)
        except OSError as error:
            raise protocol.ProtocolError(
                f"cannot resolve required executable {name}: {error}"
            ) from error
        directory = executable.parent
        if directory not in directories:
            directories.append(directory)
    path = os.pathsep.join(str(directory) for directory in directories)
    return build.resolve_toolchain(path).path


def _canonical_cargo_home() -> pathlib.Path:
    raw = os.environ.get("CARGO_HOME")
    if raw is None:
        ambient_home = os.environ.get("HOME")
        if not ambient_home:
            raise protocol.ProtocolError("neither CARGO_HOME nor HOME is available")
        raw = str(pathlib.Path(ambient_home) / ".cargo")
    try:
        cargo_home = pathlib.Path(raw).resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot resolve source CARGO_HOME: {error}"
        ) from error
    if not cargo_home.is_dir():
        raise protocol.ProtocolError("source CARGO_HOME is not a directory")
    return cargo_home


def _execution_home_paths(
    scratch_parent: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    scratch = pathlib.Path(scratch_parent)
    return (
        scratch.parent / f".{scratch.name}-phase2e-home",
        scratch.parent / f".{scratch.name}-phase2e-cargo-home",
    )


def _is_directory_identity(
    metadata: os.stat_result, device: int, inode: int
) -> bool:
    return stat.S_ISDIR(metadata.st_mode) and (
        metadata.st_dev,
        metadata.st_ino,
    ) == (device, inode)


@dataclass
class _OwnedExecutionHome:
    path: pathlib.Path
    descriptor: int
    device: int
    inode: int

    def close(self, primary: BaseException | None = None) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor < 0:
            return
        try:
            os.close(descriptor)
        except OSError as error:
            if primary is not None:
                _add_cleanup_note(
                    primary,
                    f"suppressed execution-home descriptor close failure for "
                    f"{self.path}: {error}",
                )
            else:
                raise protocol.ProtocolError(
                    f"cannot close execution-home descriptor for "
                    f"{self.path}: {error}"
                ) from error


def _add_cleanup_note(primary: BaseException, text: str) -> None:
    try:
        primary.add_note(text)
    except BaseException:
        pass


def _empty_retained_directory(descriptor: int) -> None:
    """Remove contents through a held leaf fd, never through its parent name."""
    try:
        shutil.rmtree(".", dir_fd=descriptor)
    except OSError as error:
        if error.errno == errno.EINVAL and error.filename == ".":
            return
        raise
    raise protocol.ProtocolError(
        "retained-directory rmtree unexpectedly removed its root"
    )


class _PreparedExecutionHomes:
    """Retain parent and leaf identities until ACTIVE or safe rollback."""

    def __init__(self, parent: pathlib.Path, descriptor: int) -> None:
        self.parent = pathlib.Path(parent)
        self.parent_descriptor = descriptor
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            failure = protocol.ProtocolError(
                f"cannot hold execution-home parent identity: {error}"
            )
            try:
                os.close(descriptor)
            except OSError as close_error:
                _add_cleanup_note(
                    failure,
                    "suppressed execution-home parent descriptor close "
                    f"failure for {self.parent}: {close_error}",
                )
            self.parent_descriptor = -1
            raise failure from error
        self.parent_device = metadata.st_dev
        self.parent_inode = metadata.st_ino
        self.owned: list[_OwnedExecutionHome] = []

    def __iter__(self):
        return iter(home.path for home in self.owned)

    def revalidate(self) -> None:
        try:
            held_parent = os.fstat(self.parent_descriptor)
            current_parent = self.parent.lstat()
        except OSError as error:
            raise protocol.ProtocolError(
                f"execution-home parent identity disappeared: {error}"
            ) from error
        if not (
            _is_directory_identity(
                held_parent, self.parent_device, self.parent_inode
            )
            and _is_directory_identity(
                current_parent, self.parent_device, self.parent_inode
            )
        ):
            raise protocol.ProtocolError(
                "execution-home parent identity changed"
            )
        for home in self.owned:
            try:
                held = os.fstat(home.descriptor)
                current = os.stat(
                    home.path.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise protocol.ProtocolError(
                    f"execution-home identity disappeared: {home.path}: {error}"
                ) from error
            if not (
                _is_directory_identity(held, home.device, home.inode)
                and _is_directory_identity(current, home.device, home.inode)
            ):
                raise protocol.ProtocolError(
                    f"execution-home identity changed: {home.path}"
                )

    def close(self, primary: BaseException | None = None) -> None:
        failure = primary
        for home in reversed(self.owned):
            try:
                home.close(failure)
            except protocol.ProtocolError as error:
                failure = error
        descriptor, self.parent_descriptor = self.parent_descriptor, -1
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as error:
                if failure is None:
                    failure = protocol.ProtocolError(
                        "cannot close execution-home parent descriptor: "
                        f"{error}"
                    )
                else:
                    _add_cleanup_note(
                        failure,
                        "suppressed execution-home parent descriptor close "
                        f"failure for {self.parent}: {error}",
                    )
        if primary is None and failure is not None:
            raise failure

    def rollback(self, primary: BaseException) -> None:
        """Delete only leaves still bound to their retained identities."""
        try:
            try:
                held_parent = os.fstat(self.parent_descriptor)
                current_parent = self.parent.lstat()
            except OSError as error:
                _add_cleanup_note(
                    primary,
                    f"execution-home cleanup identity failure for parent "
                    f"{self.parent}: {error}",
                )
                return
            if not (
                _is_directory_identity(
                    held_parent, self.parent_device, self.parent_inode
                )
                and _is_directory_identity(
                    current_parent, self.parent_device, self.parent_inode
                )
            ):
                _add_cleanup_note(
                    primary,
                    "execution-home cleanup identity failure: parent changed",
                )
                return
            for home in reversed(self.owned):
                try:
                    held = os.fstat(home.descriptor)
                except OSError as error:
                    _add_cleanup_note(
                        primary,
                        f"execution-home cleanup identity failure for "
                        f"{home.path}: {error}",
                    )
                    continue
                if not _is_directory_identity(
                    held, home.device, home.inode
                ):
                    _add_cleanup_note(
                        primary,
                        f"execution-home cleanup identity failure for "
                        f"{home.path}: retained leaf changed",
                    )
                    continue
                try:
                    _empty_retained_directory(home.descriptor)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        primary,
                        "suppressed execution-home cleanup failure for "
                        f"{home.path}: {type(cleanup_error).__name__}: "
                        f"{cleanup_error}",
                    )
                    continue
                quarantine = (
                    f".phase2e-cleanup-{secrets.token_hex(16)}"
                )
                try:
                    os.rename(
                        home.path.name,
                        quarantine,
                        src_dir_fd=self.parent_descriptor,
                        dst_dir_fd=self.parent_descriptor,
                    )
                    os.fsync(self.parent_descriptor)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        primary,
                        "suppressed execution-home cleanup failure for "
                        f"{home.path}: {type(cleanup_error).__name__}: "
                        f"{cleanup_error}",
                    )
                    continue
                try:
                    quarantined = os.stat(
                        quarantine,
                        dir_fd=self.parent_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as cleanup_error:
                    _add_cleanup_note(
                        primary,
                        "execution-home quarantine identity failure for "
                        f"{home.path}: {cleanup_error}; retained as "
                        f"{self.parent / quarantine}",
                    )
                    continue
                if not _is_directory_identity(
                    quarantined, home.device, home.inode
                ):
                    _add_cleanup_note(
                        primary,
                        "execution-home quarantine identity mismatch for "
                        f"{home.path}; retained as {self.parent / quarantine}",
                    )
                    continue
                try:
                    remaining = os.listdir(home.descriptor)
                except OSError as cleanup_error:
                    _add_cleanup_note(
                        primary,
                        "execution-home tombstone verification failure for "
                        f"{home.path}: {cleanup_error}; retained as "
                        f"{self.parent / quarantine}",
                    )
                    continue
                state = "empty" if not remaining else "nonempty"
                _add_cleanup_note(
                    primary,
                    f"execution-home cleanup retained {state} owned tombstone "
                    f"for {home.path}: {self.parent / quarantine}",
                )
        finally:
            self.close(primary)


def _prepare_execution_homes(
    scratch_parent: pathlib.Path,
    *,
    source_cargo_home: pathlib.Path | None = None,
) -> _PreparedExecutionHomes:
    """Create isolated homes exposing only Cargo's reusable source caches."""
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise protocol.ProtocolError(
            "controlled execution homes require symlink-attack-resistant rmtree"
        )
    try:
        scratch = pathlib.Path(scratch_parent).resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot resolve scratch parent for execution homes: {error}"
        ) from error
    home, cargo_home = _execution_home_paths(scratch)
    parent = home.parent
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(parent, parent_flags)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot hold execution-home parent identity: {error}"
        ) from error
    prepared = _PreparedExecutionHomes(parent, parent_descriptor)
    try:
        prepared.revalidate()
        try:
            source = pathlib.Path(
                source_cargo_home
                if source_cargo_home is not None
                else _canonical_cargo_home()
            ).resolve(strict=True)
        except OSError as error:
            raise protocol.ProtocolError(
                f"cannot resolve source CARGO_HOME: {error}"
            ) from error
        for destination in (home, cargo_home):
            try:
                os.mkdir(
                    destination.name,
                    mode=0o700,
                    dir_fd=prepared.parent_descriptor,
                )
            except FileExistsError as error:
                raise protocol.ProtocolError(
                    f"controlled execution home is not fresh: {destination}"
                ) from error
            except OSError as error:
                raise protocol.ProtocolError(
                    f"cannot create controlled execution home {destination}: "
                    f"{error}"
                ) from error
            try:
                descriptor = os.open(
                    destination.name,
                    parent_flags,
                    dir_fd=prepared.parent_descriptor,
                )
            except OSError as error:
                failure = protocol.ProtocolError(
                    f"cannot hold controlled execution home {destination}: "
                    f"{error}"
                )
                _add_cleanup_note(
                    failure,
                    "execution-home cleanup skipped because ownership "
                    f"identity was not retained: {destination}",
                )
                raise failure from error
            try:
                metadata = os.fstat(descriptor)
            except OSError as error:
                failure = protocol.ProtocolError(
                    f"cannot hold controlled execution home {destination}: "
                    f"{error}"
                )
                try:
                    os.close(descriptor)
                except OSError as close_error:
                    _add_cleanup_note(
                        failure,
                        "suppressed execution-home descriptor close failure "
                        f"for {destination}: {close_error}",
                    )
                _add_cleanup_note(
                    failure,
                    "execution-home cleanup skipped because ownership "
                    f"identity was not retained: {destination}",
                )
                raise failure from error
            prepared.owned.append(
                _OwnedExecutionHome(
                    destination,
                    descriptor,
                    metadata.st_dev,
                    metadata.st_ino,
                )
            )
            prepared.revalidate()
        for name in ("git", "registry"):
            cache = source / name
            if not cache.exists():
                continue
            try:
                target = cache.resolve(strict=True)
            except OSError as error:
                raise protocol.ProtocolError(
                    f"cannot resolve source Cargo {name} cache: {error}"
                ) from error
            if not target.is_dir():
                raise protocol.ProtocolError(
                    f"source Cargo {name} cache is not a directory"
                )
            try:
                os.symlink(
                    target,
                    name,
                    target_is_directory=True,
                    dir_fd=prepared.owned[1].descriptor,
                )
            except OSError as error:
                raise protocol.ProtocolError(
                    f"cannot link controlled Cargo cache {name}: {error}"
                ) from error
            prepared.revalidate()
        build.validate_controlled_cargo_home(cargo_home)
        prepared.revalidate()
        try:
            resolved = home.resolve(strict=True), cargo_home.resolve(strict=True)
        except OSError as error:
            raise protocol.ProtocolError(
                f"cannot resolve controlled execution home: {error}"
            ) from error
        if resolved != (home, cargo_home):
            raise protocol.ProtocolError(
                "controlled execution home path is not canonical"
            )
        return prepared
    except BaseException as primary:
        prepared.rollback(primary)
        raise


def _close_committed_execution_homes(
    execution_homes: _PreparedExecutionHomes,
) -> None:
    """Report cleanup failure without replacing a committed lifecycle result."""
    try:
        execution_homes.close()
    except Exception as error:
        print(
            f"execution-home cleanup warning: {error}",
            file=sys.stderr,
        )


def _preflight_offline_feature_queries(context: Mapping[str, Any]) -> None:
    """Prove the isolated Cargo cache can resolve Task 7 before ACTIVE."""
    repository = _context_path(context, "repository")
    scratch = _context_path(context, "scratch_parent")
    target_dir = scratch / ".phase2e-preflight-target"
    try:
        target_dir.mkdir(mode=0o700)
    except FileExistsError as error:
        raise protocol.ProtocolError("offline preflight target is not fresh") from error
    tools = build.resolve_toolchain(context["path"])
    environment = protocol.cargo_environment(
        path=context["path"],
        home=context["home"],
        cargo_home=context["cargo_home"],
        target_dir=str(target_dir),
    )
    primary: BaseException | None = None
    try:
        rustc = build.run_bounded_command(
            (str(tools.rustc.path), "-vV"),
            cwd=repository,
            environment=environment,
            deadline_seconds=build.QUERY_DEADLINE_SECONDS,
        )
        if rustc.returncode != 0:
            raise protocol.ProtocolError("offline preflight rustc probe failed")
        target = build._rustc_host(rustc.stdout, "offline preflight")
        if target != context["host_target"]:
            raise protocol.ProtocolError(
                "offline preflight host target differs from stage context"
            )
        for package in ("tenferro-cpu", "tenferro-ad"):
            result = build.run_bounded_command(
                build.feature_query_command(
                    target,
                    package=package,
                    requested_features=build.REQUESTED_FEATURES,
                    no_default_features=True,
                ),
                cwd=repository,
                environment=environment,
                deadline_seconds=build.QUERY_DEADLINE_SECONDS,
            )
            if result.returncode != 0:
                raise protocol.ProtocolError(
                    f"offline Task 7 feature query failed for {package}"
                )
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            shutil.rmtree(target_dir)
        except BaseException as cleanup_error:
            if primary is not None:
                try:
                    primary.add_note(
                        "suppressed offline preflight cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                except BaseException:
                    pass
            elif isinstance(cleanup_error, Exception):
                raise protocol.ProtocolError(
                    f"cannot clean offline preflight target: {cleanup_error}"
                ) from cleanup_error
            else:
                raise


def _measure_stage_host_target(
    *,
    repository: pathlib.Path,
    scratch: pathlib.Path,
    path: str,
    home: pathlib.Path,
    cargo_home: pathlib.Path,
    tools: build.ResolvedToolchain,
) -> str:
    """Measure and bind the exact resolved rustc host before root mutation."""
    environment = protocol.cargo_environment(
        path=path,
        home=str(home),
        cargo_home=str(cargo_home),
        target_dir=str(scratch / ".phase2e-host-target"),
    )
    build.validate_resolved_tool(tools.rustc)
    result = build.run_bounded_command(
        (str(tools.rustc.path), "--version", "--verbose"),
        cwd=repository,
        environment=environment,
        deadline_seconds=build.QUERY_DEADLINE_SECONDS,
    )
    build.validate_resolved_tool(tools.rustc)
    if result.returncode != 0:
        raise protocol.ProtocolError("stage host-target rustc probe failed")
    return build._rustc_host(result.stdout, "stage host-target probe")


def _prepare_start_context(
    *,
    repository: pathlib.Path,
    root: pathlib.Path,
    scratch_parent: pathlib.Path,
    candidate: str,
) -> tuple[dict[str, Any], str, str]:
    """Derive every immutable identity and context field from public inputs."""
    try:
        scratch = pathlib.Path(scratch_parent).resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve scratch parent: {error}") from error
    if not scratch.is_dir() or next(scratch.iterdir(), None) is not None:
        raise protocol.ProtocolError("scratch parent must be an existing empty directory")
    for protected in (repository, root):
        if scratch == protected or scratch in protected.parents or protected in scratch.parents:
            raise protocol.ProtocolError("scratch parent must be external and disjoint")
    candidate_tree_sha256, experiment_identity = _candidate_provenance(
        repository, candidate
    )
    reservation_id = secrets.token_hex(32)
    campaign_identity = protocol.sha256_json(
        {
            "version": 1,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "candidate_sha": candidate,
            "experiment_identity_digest": experiment_identity,
            "evidence_root": str(root),
        }
    )
    home, cargo_home = _execution_home_paths(scratch)
    index, index_lock = campaign_index_paths(repository)
    path = _minimal_toolchain_path()
    tools = build.resolve_toolchain(path)
    host_target = _measure_stage_host_target(
        repository=repository,
        scratch=scratch,
        path=path,
        home=home,
        cargo_home=cargo_home,
        tools=tools,
    )
    context = {
        "version": 1,
        "repository": str(repository),
        "evidence_root": str(root),
        "scratch_parent": str(scratch),
        "candidate_sha": candidate,
        "candidate_tree_sha256": candidate_tree_sha256,
        "reservation_id": reservation_id,
        "experiment_identity_digest": experiment_identity,
        "command_contract_digest": "",
        "path": path,
        "tool_paths": {
            name: str(getattr(tools, name).path)
            for name in ("git", "cargo", "rustc")
        },
        "tool_sha256": {
            name: getattr(tools, name).sha256
            for name in ("git", "cargo", "rustc")
        },
        "host_target": host_target,
        "home": str(home),
        "cargo_home": str(cargo_home),
        "index": str(index),
        "index_lock": str(index_lock),
    }
    context["command_contract_digest"] = stage_context_contract_digest(context)
    context_sha256 = hashlib.sha256(
        protocol._canonical_json_bytes(context)
    ).hexdigest()
    return context, context_sha256, campaign_identity


def validate_candidate_provenance(
    repository: pathlib.Path,
    candidate: str,
    candidate_tree_sha256: str,
    identity_digest: str,
    contract: Mapping[str, Any],
) -> None:
    """Bind start to the clean exact commit, full tree, and canonical identity."""
    head = _git(repository, "rev-parse", "HEAD", text=True).strip()
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        text=True,
    )
    if head != candidate or status:
        raise protocol.ProtocolError(
            "start requires the clean exact candidate worktree"
        )
    tree = _git(repository, "ls-tree", "-r", "-z", "--full-tree", candidate)
    if hashlib.sha256(tree).hexdigest() != candidate_tree_sha256:
        raise protocol.ProtocolError("candidate full-tree digest differs")
    if git_experiment_identity(repository, candidate, contract) != identity_digest:
        raise protocol.ProtocolError("canonical experiment identity differs")


def new_campaign_index() -> dict[str, Any]:
    """Create the append-only global reservation index."""
    return {"version": 1, "events": [], "current_evidence_root": None}


def _validate_index(index: Mapping[str, Any]) -> None:
    if (
        type(index) is not dict
        or set(index) - {"version", "events", "current_evidence_root", "audit"}
        or index.get("version") != 1
        or type(index.get("events")) is not list
        or index.get("current_evidence_root") is not None
        and type(index.get("current_evidence_root")) is not str
    ):
        raise protocol.ProtocolError("Phase 2E index schema is invalid")
    states: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    terminal_statuses: dict[str, str] = {}
    terminal_digests: dict[str, str] = {}
    active_global = None
    expected_current_root = None
    for ordinal, event in enumerate(index["events"], start=1):
        if type(event) is not dict or event.get("ordinal") != ordinal:
            raise protocol.ProtocolError("Phase 2E index history is not append-only")
        reservation = event.get("reservation_id")
        if type(reservation) is not str or not reservation:
            raise protocol.ProtocolError("index event lacks a reservation id")
        kind = event.get("event")
        if kind == "ACTIVE":
            if set(event) != {
                "ordinal",
                "event",
                "reservation_id",
                "candidate_sha",
                "candidate_tree_sha256",
                "root",
                "protocol_version",
                "experiment_identity_digest",
                "campaign_identity_digest",
                "command_contract_digest",
                "context_sha256",
            }:
                raise protocol.ProtocolError("ACTIVE event schema is invalid")
            if reservation in states or active_global is not None:
                raise protocol.ProtocolError(
                    "index has overlapping or reused reservation"
                )
            for name in (
                "candidate_sha",
                "candidate_tree_sha256",
                "experiment_identity_digest",
                "campaign_identity_digest",
                "command_contract_digest",
                "context_sha256",
                "root",
            ):
                if type(event.get(name)) is not str or not event[name]:
                    raise protocol.ProtocolError(f"ACTIVE event lacks {name}")
            if _has_ascii_control(event["root"]):
                raise protocol.ProtocolError(
                    "ACTIVE evidence root contains a control character"
                )
            _require_sha(event["candidate_sha"], sha256=False, context="candidate SHA")
            for name in (
                "candidate_tree_sha256",
                "experiment_identity_digest",
                "campaign_identity_digest",
                "command_contract_digest",
                "context_sha256",
            ):
                _require_sha(event[name], sha256=True, context=name)
            states[reservation] = "ACTIVE"
            records[reservation] = event
            active_global = reservation
        elif kind == "TERMINAL":
            if set(event) != {
                "ordinal",
                "event",
                "reservation_id",
                "status",
                "root_digest",
                "ledger_sha256",
                "candidate_sha",
                "candidate_tree_sha256",
                "root",
                "experiment_identity_digest",
                "campaign_identity_digest",
                "command_contract_digest",
                "context_sha256",
            }:
                raise protocol.ProtocolError("TERMINAL event schema is invalid")
            if states.get(reservation) != "ACTIVE" or active_global != reservation:
                raise protocol.ProtocolError("TERMINAL event does not close ACTIVE")
            if event.get("status") not in TERMINAL_STATUSES:
                raise protocol.ProtocolError("TERMINAL status is invalid")
            _require_sha(event.get("root_digest"), sha256=True, context="root digest")
            _require_sha(
                event.get("ledger_sha256"), sha256=True, context="ledger digest"
            )
            if any(
                event[name] != records[reservation][name]
                for name in (
                    "candidate_sha",
                    "candidate_tree_sha256",
                    "root",
                    "experiment_identity_digest",
                    "campaign_identity_digest",
                    "command_contract_digest",
                    "context_sha256",
                )
            ):
                raise protocol.ProtocolError("TERMINAL event identity differs")
            states[reservation] = "PENDING_PRESERVATION"
            terminal_statuses[reservation] = event["status"]
            terminal_digests[reservation] = event["root_digest"]
            active_global = reservation
        elif kind == "PRESERVED":
            if set(event) != {
                "ordinal",
                "event",
                "reservation_id",
                "preservation_commit",
                "issue_url",
                "candidate_sha",
                "root",
                "status",
                "root_digest",
                "ledger_sha256",
                "candidate_tree_sha256",
                "experiment_identity_digest",
                "campaign_identity_digest",
                "command_contract_digest",
                "context_sha256",
            }:
                raise protocol.ProtocolError("PRESERVED event schema is invalid")
            if (
                states.get(reservation) != "PENDING_PRESERVATION"
                or active_global != reservation
            ):
                raise protocol.ProtocolError("PRESERVED event has no pending terminal")
            _require_sha(
                event.get("preservation_commit"),
                sha256=False,
                context="preservation commit",
            )
            if not _is_canonical_issue_comment_url(event.get("issue_url")):
                raise protocol.ProtocolError("PRESERVED event issue URL is invalid")
            if any(
                event[name] != records[reservation][name]
                for name in (
                    "candidate_sha",
                    "candidate_tree_sha256",
                    "root",
                    "experiment_identity_digest",
                    "campaign_identity_digest",
                    "command_contract_digest",
                    "context_sha256",
                )
            ) or (
                event["status"] != terminal_statuses[reservation]
                or event["root_digest"] != terminal_digests[reservation]
                or event["ledger_sha256"]
                != next(
                    prior["ledger_sha256"]
                    for prior in reversed(index["events"][: event["ordinal"] - 1])
                    if prior["reservation_id"] == reservation
                    and prior["event"] == "TERMINAL"
                )
            ):
                raise protocol.ProtocolError("PRESERVED event identity differs")
            states[reservation] = "PRESERVED"
            active_global = None
            if terminal_statuses[reservation] == "PASS":
                expected_current_root = records[reservation]["root"]
        else:
            raise protocol.ProtocolError("index event kind is invalid")
    if index["current_evidence_root"] != expected_current_root:
        raise protocol.ProtocolError(
            "current evidence root is not the latest preserved PASS"
        )


def index_state(index: Mapping[str, Any]) -> str:
    _validate_index(index)
    if not index["events"]:
        return "EMPTY"
    return {
        "ACTIVE": "ACTIVE",
        "TERMINAL": "PENDING_PRESERVATION",
        "PRESERVED": "PRESERVED",
    }[index["events"][-1]["event"]]


def _append(index: Mapping[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(index)
    event["ordinal"] = len(updated["events"]) + 1
    updated["events"].append(event)
    _validate_index(updated)
    return updated


def record_active(
    index: Mapping[str, Any],
    *,
    reservation_id: str,
    candidate_sha: str,
    candidate_tree_sha256: str,
    root: str,
    experiment_identity_digest: str,
    campaign_identity_digest: str,
    command_digest: str | None = None,
    context_sha256: str | None = None,
) -> dict[str, Any]:
    """Reserve one globally unique evidence root before initialization."""
    _validate_index(index)
    if index_state(index) in {"ACTIVE", "PENDING_PRESERVATION"}:
        raise protocol.ProtocolError("another Phase 2E campaign is globally reserved")
    if any(event.get("root") == root for event in index["events"]):
        raise protocol.ProtocolError("evidence root was already reserved")
    prior = [
        event
        for event in index["events"]
        if event.get("event") == "ACTIVE"
        and event.get("experiment_identity_digest") == experiment_identity_digest
    ]
    if prior:
        reservation = prior[-1]["reservation_id"]
        matching = [
            event
            for event in index["events"]
            if event["reservation_id"] == reservation
        ]
        if (
            len(matching) != 3
            or matching[-1]["event"] != "PRESERVED"
            or matching[-2].get("status") not in RETRYABLE_STATUSES
            or prior[-1]["candidate_sha"] != candidate_sha
        ):
            raise protocol.ProtocolError(
                "canonical experiment identity is permanently closed"
            )
    command_digest = command_digest or command_contract_digest()
    context_sha256 = context_sha256 or "0" * 64
    _require_sha(command_digest, sha256=True, context="command contract digest")
    _require_sha(context_sha256, sha256=True, context="context digest")
    return _append(
        index,
        {
            "event": "ACTIVE",
            "reservation_id": reservation_id,
            "candidate_sha": candidate_sha,
            "candidate_tree_sha256": candidate_tree_sha256,
            "root": root,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "experiment_identity_digest": experiment_identity_digest,
            "campaign_identity_digest": campaign_identity_digest,
            "command_contract_digest": command_digest,
            "context_sha256": context_sha256,
        },
    )


def record_terminal(
    index: Mapping[str, Any],
    *,
    reservation_id: str,
    status: str,
    root_digest: str,
    ledger_sha256: str = "0" * 64,
) -> dict[str, Any]:
    """Close ACTIVE and make its root pending preservation."""
    _validate_index(index)
    if (
        index_state(index) != "ACTIVE"
        or index["events"][-1]["reservation_id"] != reservation_id
    ):
        raise protocol.ProtocolError("reservation is not the global ACTIVE campaign")
    if status not in TERMINAL_STATUSES:
        raise protocol.ProtocolError("terminal status is invalid")
    active = index["events"][-1]
    return _append(
        index,
        {
            "event": "TERMINAL",
            "reservation_id": reservation_id,
            "status": status,
            "root_digest": root_digest,
            "ledger_sha256": ledger_sha256,
            **{
                name: active[name]
                for name in (
                    "candidate_sha",
                    "candidate_tree_sha256",
                    "root",
                    "experiment_identity_digest",
                    "campaign_identity_digest",
                    "command_contract_digest",
                    "context_sha256",
                )
            },
        },
    )


def record_preserved_event(
    index: Mapping[str, Any],
    *,
    reservation_id: str,
    preservation_commit: str,
    issue_url: str,
) -> dict[str, Any]:
    """Record remote Git and issue preservation after both are verified."""
    _validate_index(index)
    if index_state(index) == "PRESERVED":
        preserved = index["events"][-1]
        if (
            preserved["reservation_id"] == reservation_id
            and preserved["preservation_commit"] == preservation_commit
            and preserved["issue_url"] == issue_url
        ):
            return copy.deepcopy(index)
        raise protocol.ProtocolError("PRESERVED replay differs")
    if (
        index_state(index) != "PENDING_PRESERVATION"
        or index["events"][-1]["reservation_id"] != reservation_id
    ):
        raise protocol.ProtocolError("reservation is not pending preservation")
    if not _is_canonical_issue_comment_url(issue_url):
        raise protocol.ProtocolError("preservation report is not on issue #1436")
    updated = copy.deepcopy(index)
    active = next(
        event
        for event in reversed(index["events"])
        if event["reservation_id"] == reservation_id and event["event"] == "ACTIVE"
    )
    terminal = index["events"][-1]
    updated["events"].append(
        {
            "ordinal": len(updated["events"]) + 1,
            "event": "PRESERVED",
            "reservation_id": reservation_id,
            "preservation_commit": preservation_commit,
            "issue_url": issue_url,
            "candidate_sha": active["candidate_sha"],
            "root": active["root"],
            "status": terminal["status"],
            "root_digest": terminal["root_digest"],
            "ledger_sha256": terminal["ledger_sha256"],
            **{
                name: active[name]
                for name in (
                    "candidate_tree_sha256",
                    "experiment_identity_digest",
                    "campaign_identity_digest",
                    "command_contract_digest",
                    "context_sha256",
                )
            },
        }
    )
    matching = [
        event
        for event in updated["events"]
        if event["reservation_id"] == reservation_id
    ]
    if matching[-2]["status"] == "PASS":
        updated["current_evidence_root"] = matching[0]["root"]
    _validate_index(updated)
    return updated


def _open_directory_descriptor(path: pathlib.Path) -> int:
    path = pathlib.Path(path)
    if not path.is_absolute():
        raise protocol.ProtocolError("trusted directory path must be absolute")
    try:
        descriptor = os.open(
            path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot open trusted directory {path}: {error}"
        ) from error
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise protocol.ProtocolError(
                    f"trusted directory traverses an invalid component: {path}: {error}"
                ) from error
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or (
            hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()
        ):
            raise protocol.ProtocolError("trusted directory is not owned")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_lock_identity(
    directory_descriptor: int, name: str, expected: os.stat_result
) -> None:
    try:
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as error:
        raise protocol.ProtocolError(f"lock identity disappeared: {name}: {error}") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise protocol.ProtocolError(f"lock identity changed: {name}")


@contextmanager
def exclusive_lock_at(directory_descriptor: int, name: str):
    """Hold one no-follow regular lock relative to a retained directory."""
    try:
        prior = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        prior = None
    except OSError as error:
        raise protocol.ProtocolError(f"cannot inspect lock {name}: {error}") from error
    if prior is not None and not stat.S_ISREG(prior.st_mode):
        raise protocol.ProtocolError(f"lock is not a regular file: {name}")
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise protocol.ProtocolError(f"cannot securely open lock {name}: {error}") from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or prior is not None
        and (prior.st_dev, prior.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        os.close(descriptor)
        raise protocol.ProtocolError(f"lock identity or mode differs: {name}")
    if prior is None:
        os.fsync(directory_descriptor)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_lock_identity(directory_descriptor, name, metadata)
        yield descriptor
        _validate_lock_identity(directory_descriptor, name, metadata)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def exclusive_lock(path: pathlib.Path):
    """Hold one process-exclusive lock without following any path symlink."""
    path = pathlib.Path(path)
    parent = _open_directory_descriptor(path.parent)
    try:
        with exclusive_lock_at(parent, path.name) as descriptor:
            yield descriptor
    finally:
        os.close(parent)


def _read_regular_path(path: pathlib.Path, context: str) -> bytes:
    path = pathlib.Path(path)
    parent = _open_directory_descriptor(path.parent)
    descriptor = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after):
            raise protocol.ProtocolError(f"{context} is not a stable regular file")
        return b"".join(chunks)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot securely read {context}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _read_regular_at(directory_descriptor: int, name: str, context: str) -> bytes:
    """Read one stable regular file without reopening its parent pathname."""
    descriptor = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_descriptor,
        )
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
            raise protocol.ProtocolError(f"{context} is not a stable regular file")
        return b"".join(chunks)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot securely read {context}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


class CampaignIndexTransaction:
    """Descriptor-bound I/O for one locked repository campaign index."""

    def __init__(
        self,
        repository: pathlib.Path,
        directory: pathlib.Path,
        descriptor: int,
        identity: tuple[int, int],
    ) -> None:
        self.repository = repository
        self.directory = directory
        self.descriptor = descriptor
        self.identity = identity

    def revalidate(self) -> None:
        try:
            current = os.stat(self.directory, follow_symlinks=False)
            held = os.fstat(self.descriptor)
        except OSError as error:
            raise protocol.ProtocolError(
                f"campaign index directory identity disappeared: {error}"
            ) from error
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != self.identity
            or (held.st_dev, held.st_ino) != self.identity
        ):
            raise protocol.ProtocolError("campaign index directory identity changed")

    def exists(self) -> bool:
        self.revalidate()
        try:
            metadata = os.stat(
                pathlib.Path(INDEX_PATH).name,
                dir_fd=self.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise protocol.ProtocolError("Phase 2E index is not a regular file")
        return True

    def read_bytes(self) -> bytes:
        self.revalidate()
        return _read_regular_at(
            self.descriptor, pathlib.Path(INDEX_PATH).name, "Phase 2E index"
        )

    def read(self) -> dict[str, Any]:
        decoded = protocol.decode_canonical_json_bytes(
            self.read_bytes(), "Phase 2E index"
        )
        _validate_index(decoded)
        return decoded

    def write(self, payload: Mapping[str, Any]) -> None:
        self.revalidate()
        name = pathlib.Path(INDEX_PATH).name
        try:
            current = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise protocol.ProtocolError("Phase 2E index write target is not regular")
        protocol.atomic_write_json_at(self.descriptor, name, payload)
        self.revalidate()


@contextmanager
def campaign_index_transaction(repository: pathlib.Path):
    """Retain the exact worklog directory for lock and all index I/O."""
    index_path, lock_path = campaign_index_paths(repository)
    directory = index_path.parent
    descriptor = _open_directory_descriptor(directory)
    metadata = os.fstat(descriptor)
    transaction = CampaignIndexTransaction(
        pathlib.Path(repository),
        directory,
        descriptor,
        (metadata.st_dev, metadata.st_ino),
    )
    try:
        with exclusive_lock_at(descriptor, lock_path.name):
            transaction.revalidate()
            yield transaction
            transaction.revalidate()
    finally:
        os.close(descriptor)


def _regular_path_exists(path: pathlib.Path, context: str) -> bool:
    path = pathlib.Path(path)
    parent = _open_directory_descriptor(path.parent)
    try:
        try:
            metadata = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise protocol.ProtocolError(f"{context} is not a regular file")
        return True
    finally:
        os.close(parent)


def _atomic_write_path(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path = pathlib.Path(path)
    parent = _open_directory_descriptor(path.parent)
    try:
        try:
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise protocol.ProtocolError(f"write target is not regular: {path}")
        protocol.atomic_write_json_at(parent, path.name, payload)
    finally:
        os.close(parent)


def _read_index(path: pathlib.Path) -> dict[str, Any]:
    payload = _read_regular_path(path, "Phase 2E index")
    decoded = protocol.decode_canonical_json_bytes(payload, "Phase 2E index")
    _validate_index(decoded)
    return decoded


def mutate_index(
    repository: pathlib.Path,
    mutation: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Serialize one atomic read-modify-write operation on the campaign index."""
    with campaign_index_transaction(repository) as transaction:
        current = transaction.read()
        updated = mutation(copy.deepcopy(current))
        if type(updated) is not dict:
            raise protocol.ProtocolError("index mutation returned a non-object")
        transaction.write(updated)
        return updated


def run_fixed_stages(
    root: pathlib.Path,
    environment: Mapping[str, str],
    runner: Callable[[str, Mapping[str, str]], int],
    *,
    completed: Sequence[str] = (),
    identity: Mapping[str, str] | None = None,
    root_identity: protocol.PreparedRootIdentity | None = None,
    _locked: bool = False,
) -> int:
    """Run remaining children in fixed order and durably hash every checkpoint."""
    root = pathlib.Path(root)
    if not _locked:
        with exclusive_lock(root / ORCHESTRATOR_LOCK_NAME):
            return run_fixed_stages(
                root,
                environment,
                runner,
                completed=completed,
                identity=identity,
                root_identity=root_identity,
                _locked=True,
            )
    expected_environment = protocol.runtime_environment(
        path=environment.get("PATH", ""), home=environment.get("HOME", "")
    )
    if dict(environment) != expected_environment:
        raise protocol.ProtocolError("child environment is not sealed")
    children = []
    for stage in completed:
        if stage != STAGE_ORDER[len(children)]:
            raise protocol.ProtocolError("completed child order is not a prefix")
        children.append({"stage": stage, "exit_code": 0})
    for stage in STAGE_ORDER[len(children):]:
        if root_identity is not None:
            root_identity.revalidate()
        before = _root_inventory(root, root_identity)
        code = runner(stage, dict(environment))
        if root_identity is not None:
            root_identity.revalidate()
        if type(code) is not int or code not in {0, 2, 3, 4}:
            raise protocol.ProtocolError("child returned an unsupported exit code")
        children.append({"stage": stage, "exit_code": code})
        _write_child_record(
            root,
            children,
            before,
            environment=environment,
            identity=identity,
            root_identity=root_identity,
        )
        _write_progress(
            root, children, identity=identity, root_identity=root_identity
        )
        if stage == "aggregate-validation" and identity is not None:
            if root_identity is not None:
                root_identity.revalidate()
            seal_root(
                root,
                candidate_sha=identity["candidate_sha"],
                reservation_id=identity["reservation_id"],
                experiment_identity_digest=identity["experiment_identity_digest"],
            )
            return code
        if code:
            return code
    if identity is not None:
        if root_identity is not None:
            root_identity.revalidate()
        seal_root(
            root,
            candidate_sha=identity["candidate_sha"],
            reservation_id=identity["reservation_id"],
            experiment_identity_digest=identity["experiment_identity_digest"],
        )
    return 0


def _child_record_path(root: pathlib.Path, ordinal: int, stage: str) -> pathlib.Path:
    return root / "children" / f"{ordinal:02d}-{stage.replace('/', '__')}.json"


def _root_io_path(
    root: pathlib.Path, identity: protocol.PreparedRootIdentity | None
) -> pathlib.Path:
    if identity is None:
        return pathlib.Path(root)
    identity.revalidate()
    return pathlib.Path(f"/proc/self/fd/{identity.descriptor}")


def _root_inventory(
    root: pathlib.Path,
    identity: protocol.PreparedRootIdentity | None,
    *,
    excluded: frozenset[str] = frozenset(),
) -> dict[str, str]:
    if identity is None:
        return protocol.regular_file_inventory(root, excluded=excluded)
    identity.revalidate()
    inventory = protocol.regular_file_inventory_at(
        identity.descriptor, excluded=excluded
    )
    identity.revalidate()
    return inventory


def _atomic_write_root_json(
    root: pathlib.Path,
    relative: pathlib.PurePosixPath,
    payload: Mapping[str, Any],
    identity: protocol.PreparedRootIdentity | None,
) -> None:
    if identity is None:
        path = pathlib.Path(root) / pathlib.Path(relative)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        protocol.atomic_write_json(path, payload)
        return
    identity.revalidate()
    descriptor = os.dup(identity.descriptor)
    try:
        for component in relative.parts[:-1]:
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
        protocol.atomic_write_json_at(descriptor, relative.name, payload)
        identity.revalidate()
    finally:
        os.close(descriptor)


def _write_child_record(
    root: pathlib.Path,
    children: Sequence[Mapping[str, Any]],
    before: Mapping[str, str],
    *,
    environment: Mapping[str, str],
    identity: Mapping[str, str] | None = None,
    root_identity: protocol.PreparedRootIdentity | None = None,
) -> None:
    child = children[-1]
    ordinal = len(children)
    relative = pathlib.PurePosixPath(
        "children", f"{ordinal:02d}-{child['stage'].replace('/', '__')}.json"
    )
    after = _root_inventory(root, root_identity)
    changed = {
        relative: digest
        for relative, digest in after.items()
        if before.get(relative) != digest
    }
    removed = sorted(set(before) - set(after))
    bound = dict(identity or {})
    context_path = pathlib.Path(bound.get("context_path", "/context.json"))
    context_sha256 = bound.get("context_sha256", "0" * 64)
    _atomic_write_root_json(
        root,
        relative,
        {
            "version": 1,
            "ordinal": ordinal,
            "stage": child["stage"],
            "attempt": sum(
                previous["stage"] == child["stage"] for previous in children
            ),
            "exit_code": child["exit_code"],
            "candidate_sha": bound.get("candidate_sha", "a" * 40),
            "reservation_id": bound.get("reservation_id", "reservation-1"),
            "experiment_identity_digest": bound.get(
                "experiment_identity_digest", "b" * 64
            ),
            "command_contract_digest": bound.get(
                "command_contract_digest", command_contract_digest()
            ),
            "context_sha256": context_sha256,
            "argv": list(stage_argv(child["stage"], context_path, context_sha256)),
            "environment": dict(sorted(environment.items())),
            "before_inventory_sha256": protocol.sha256_json(dict(before)),
            "after_inventory_sha256": protocol.sha256_json(after),
            "changed": changed,
            "removed": removed,
        },
        root_identity,
    )


def _write_progress(
    root: pathlib.Path,
    children: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, str] | None = None,
    root_identity: protocol.PreparedRootIdentity | None = None,
) -> None:
    bound = dict(identity or {})
    io_root = _root_io_path(root, root_identity)
    ledger_path = io_root / "evidence-ledger.json"
    if ledger_path.exists():
        ledger = _read_json(ledger_path, "progress ledger")
        candidate_sha = ledger["candidate_sha"]
        ledger_sha256 = protocol.sha256_file(ledger_path)
    else:
        # Unit-level scheduler tests deliberately exercise checkpoint ordering
        # without constructing the campaign protocol fixture.
        candidate_sha = bound.get("candidate_sha", "a" * 40)
        ledger_sha256 = "0" * 64
    process_journal_path = io_root / PROCESS_JOURNAL
    process_journal_sha256 = (
        protocol.sha256_json(
            _read_process_journal_at(root_identity)
            if root_identity is not None
            else validate_process_journal(root)
        )
        if process_journal_path.exists()
        else "0" * 64
    )
    _atomic_write_root_json(
        root,
        pathlib.PurePosixPath(PROGRESS_MANIFEST),
        {
            "version": 2,
            "stage_order": list(STAGE_ORDER),
            "children": list(children),
            "protocol_version": protocol.PROTOCOL_VERSION,
            "candidate_sha": candidate_sha,
            "candidate_tree_sha256": bound.get("candidate_tree_sha256", "c" * 64),
            "reservation_id": bound.get("reservation_id", "reservation-1"),
            "experiment_identity_digest": bound.get(
                "experiment_identity_digest", "b" * 64
            ),
            "command_contract_digest": bound.get(
                "command_contract_digest", command_contract_digest()
            ),
            "context_sha256": bound.get("context_sha256", "0" * 64),
            "repository": bound.get("repository", str(root.resolve())),
            "evidence_root": bound.get("evidence_root", str(root.resolve())),
            "scratch_parent": bound.get("scratch_parent", "/tmp"),
            "path": bound.get("path", "/bin"),
            "home": bound.get("home", "/tmp"),
            "cargo_home": bound.get("cargo_home", "/tmp"),
            "index": bound.get("index", "/tmp/index.json"),
            "index_lock": bound.get("index_lock", "/tmp/index.lock"),
            "context_path": bound.get("context_path", "/context.json"),
            "ledger_sha256": ledger_sha256,
            "process_journal_sha256": process_journal_sha256,
            "inventory": _root_inventory(
                root,
                root_identity,
                excluded=frozenset(
                    {PROGRESS_MANIFEST, AGGREGATE_MANIFEST, PROCESS_JOURNAL}
                ),
            ),
        },
        root_identity,
    )


def rerun_invalid_stage(
    root: pathlib.Path,
    environment: Mapping[str, str],
    runner: Callable[[str, Mapping[str, str]], int],
    *,
    identity: Mapping[str, str] | None = None,
    root_identity: protocol.PreparedRootIdentity | None = None,
    _locked: bool = False,
) -> int:
    """Append one fresh whole-stage attempt after retryable validity failure."""
    root = pathlib.Path(root)
    if not _locked:
        with exclusive_lock(root / ORCHESTRATOR_LOCK_NAME):
            return rerun_invalid_stage(
                root,
                environment,
                runner,
                identity=identity,
                root_identity=root_identity,
                _locked=True,
            )
    expected = protocol.runtime_environment(
        path=environment.get("PATH", ""), home=environment.get("HOME", "")
    )
    if dict(environment) != expected:
        raise protocol.ProtocolError("child environment is not sealed")
    progress = validate_progress(root)
    children = list(progress["children"])
    if not children or children[-1].get("exit_code") != 2:
        raise protocol.ProtocolError(
            "latest stage is not retryable validity INCONCLUSIVE"
        )
    stage = children[-1].get("stage")
    if stage not in RETRYABLE_STAGES:
        raise protocol.ProtocolError(
            "latest stage is not a canonical retryable allocation or timing lane"
        )
    if root_identity is not None:
        root_identity.revalidate()
    before = _root_inventory(root, root_identity)
    code = runner(stage, dict(environment))
    if root_identity is not None:
        root_identity.revalidate()
    if type(code) is not int or code not in {0, 2, 3, 4}:
        raise protocol.ProtocolError("child returned an unsupported exit code")
    children.append({"stage": stage, "exit_code": code})
    _write_child_record(
        root,
        children,
        before,
        environment=environment,
        identity=identity,
        root_identity=root_identity,
    )
    _write_progress(root, children, identity=identity, root_identity=root_identity)
    return code


def continue_after_retry(
    root: pathlib.Path,
    environment: Mapping[str, str],
    runner: Callable[[str, Mapping[str, str]], int],
    *,
    identity: Mapping[str, str] | None = None,
    root_identity: protocol.PreparedRootIdentity | None = None,
    _locked: bool = False,
) -> int:
    """Continue only after a retained replacement attempt passed."""
    root = pathlib.Path(root)
    if not _locked:
        with exclusive_lock(root / ORCHESTRATOR_LOCK_NAME):
            return continue_after_retry(
                root,
                environment,
                runner,
                identity=identity,
                root_identity=root_identity,
                _locked=True,
            )
    expected = protocol.runtime_environment(
        path=environment.get("PATH", ""), home=environment.get("HOME", "")
    )
    if dict(environment) != expected:
        raise protocol.ProtocolError("child environment is not sealed")
    progress = validate_progress(root)
    children = list(progress["children"])
    if (
        len(children) < 2
        or children[-1].get("exit_code") != 0
        or children[-2].get("exit_code") != 2
        or children[-1].get("stage") != children[-2].get("stage")
        or children[-1].get("stage") not in RETRYABLE_STAGES
    ):
        raise protocol.ProtocolError("replacement attempt has not passed")
    start = STAGE_ORDER.index(children[-1]["stage"]) + 1
    for stage in STAGE_ORDER[start:]:
        if root_identity is not None:
            root_identity.revalidate()
        before = _root_inventory(root, root_identity)
        code = runner(stage, dict(environment))
        if root_identity is not None:
            root_identity.revalidate()
        if type(code) is not int or code not in {0, 2, 3, 4}:
            raise protocol.ProtocolError("child returned an unsupported exit code")
        children.append({"stage": stage, "exit_code": code})
        _write_child_record(
            root,
            children,
            before,
            environment=environment,
            identity=identity,
            root_identity=root_identity,
        )
        _write_progress(root, children, identity=identity, root_identity=root_identity)
        if stage == "aggregate-validation" and identity is not None:
            if root_identity is not None:
                root_identity.revalidate()
            seal_root(
                root,
                candidate_sha=identity["candidate_sha"],
                reservation_id=identity["reservation_id"],
                experiment_identity_digest=identity["experiment_identity_digest"],
            )
            return code
        if code:
            return code
    if identity is not None:
        if root_identity is not None:
            root_identity.revalidate()
        seal_root(
            root,
            candidate_sha=identity["candidate_sha"],
            reservation_id=identity["reservation_id"],
            experiment_identity_digest=identity["experiment_identity_digest"],
        )
    return 0


def initialize_campaign(
    *,
    repository: pathlib.Path,
    root: pathlib.Path,
    reservation_id: str,
    candidate_sha: str,
    candidate_tree_sha256: str,
    experiment_identity_digest: str,
    campaign_identity_digest: str,
    command_digest: str | None = None,
    context_sha256: str | None = None,
    initializer: Callable[[protocol.PreparedRootIdentity], None] | None = None,
) -> int:
    """Reserve globally, initialize locally, or self-seal atomically on failure."""
    root = pathlib.Path(root)
    index_path, _index_lock = campaign_index_paths(repository)
    with campaign_index_transaction(repository) as index_transaction:
        require_remote_index(
            repository,
            index_path,
            allow_absent=True,
            transaction=index_transaction,
        )
        if index_transaction.exists():
            current = index_transaction.read()
        else:
            current = new_campaign_index()
            index_transaction.write(current)
        updated = record_active(
            current,
            reservation_id=reservation_id,
            candidate_sha=candidate_sha,
            candidate_tree_sha256=candidate_tree_sha256,
            root=str(root),
            experiment_identity_digest=experiment_identity_digest,
            campaign_identity_digest=campaign_identity_digest,
            command_digest=command_digest,
            context_sha256=context_sha256,
        )
        root_identity = protocol.prepare_empty_root_identity(root)
        try:
            root_lock = ORCHESTRATOR_LOCK_NAME
            with exclusive_lock_at(root_identity.descriptor, root_lock):
                active_committed = False
                try:
                    try:
                        index_transaction.write(updated)
                    except protocol.AtomicWriteError as error:
                        active_committed = error.committed
                        if not active_committed:
                            raise
                        raise
                    else:
                        active_committed = True
                    protocol.atomic_write_json_at(
                        root_identity.descriptor,
                        PROCESS_JOURNAL,
                        {"version": 1, "entries": []},
                    )
                    root_identity.revalidate()
                    if initializer is None:
                        protocol.atomic_write_json_at(
                            root_identity.descriptor,
                            "evidence-ledger.json",
                            protocol.new_ledger(candidate_sha),
                        )
                    else:
                        root_identity.revalidate()
                        initializer(root_identity)
                        root_identity.revalidate()
                    root_identity.revalidate()
                except BaseException:
                    if not active_committed:
                        try:
                            active_committed = index_transaction.read() == updated
                        except BaseException:
                            active_committed = False
                    if not active_committed:
                        raise
                    seal = seal_abandoned_root(
                        root,
                        identity=root_identity,
                        abandonment_kind="INITIALIZATION_FAILURE",
                    )
                    terminal = record_terminal(
                        updated,
                        reservation_id=reservation_id,
                        status="ABANDONED",
                        root_digest=protocol.sha256_json(seal),
                        ledger_sha256=seal["inventory"].get(
                            "evidence-ledger.json", "0" * 64
                        ),
                    )
                    index_transaction.write(terminal)
                    return 5
        finally:
            root_identity.close()
    return 0


def _terminal_ledger_status(ledger: Mapping[str, Any]) -> str:
    protocol.validate_ledger(dict(ledger))
    if ledger["active_attempt_id"] is not None:
        raise protocol.ProtocolError("aggregate ledger still has a RUNNING attempt")
    lane_results = [
        lane["result"]
        for stage in ledger["stages"]
        for lane in stage["lanes"]
    ]
    lane_states = [
        lane["state"]
        for stage in ledger["stages"]
        for lane in stage["lanes"]
    ]
    if any(state == "RETRYABLE" for state in lane_states):
        return "VALIDITY_INCONCLUSIVE"
    if any(state != "COMPLETE" for state in lane_states):
        raise protocol.ProtocolError("aggregate ledger is not terminal")
    if "FAIL" in lane_results:
        return "FAIL"
    if "INCONCLUSIVE" in lane_results:
        return "STATISTICAL_INCONCLUSIVE"
    return "PASS"


def _read_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise protocol.ProtocolError(f"cannot read {context}: {error}") from error
    decoded = protocol.decode_canonical_json_bytes(payload, context)
    if type(decoded) is not dict:
        raise protocol.ProtocolError(f"{context} is not an object")
    return decoded


def required_root_paths(ledger: Mapping[str, Any]) -> frozenset[str]:
    """Derive the closed canonical Task 8A inventory from contracts and attempts."""
    required = {
        "evidence-ledger.json",
        PROGRESS_MANIFEST,
        STAGE_CONTEXT,
        ORCHESTRATOR_LOCK_NAME,
        "dispatch-gates/manifest.json",
        "characterization/manifest.json",
        *(path.as_posix() for path in build.LOCK_PATHS.values()),
        *(path.as_posix() for path in build.BUILD_MANIFEST_PATHS.values()),
        *(path.as_posix() for path in build.PROBE_BUILD_MANIFEST_PATHS.values()),
        *(path.as_posix() for path in build.DISPATCH_BUILD_MANIFEST_PATHS.values()),
        *(
            path.as_posix()
            for path in build.CHARACTERIZATION_BUILD_MANIFEST_PATHS.values()
        ),
        *(
            f"attempts/{attempt['stage']}/{attempt['lane']}/"
            f"{attempt['attempt_id']}/manifest.json"
            for attempt in ledger["attempts"]
        ),
        *(
            f"children/{index:02d}-{stage.replace('/', '__')}.json"
            for index, stage in enumerate(STAGE_ORDER, start=1)
        ),
    }
    return frozenset(required)


def _canonical_root_paths(root: pathlib.Path, ledger: Mapping[str, Any]) -> frozenset[str]:
    """Resolve the real combined-gate ownership tree."""
    gate_root = root / "gate-collector"
    if not gate_root.is_dir():
        raise protocol.ProtocolError("Task 8A gate-collector root is missing")
    required = set(required_root_paths(ledger))
    if (root / PROCESS_JOURNAL).exists():
        required.add(PROCESS_JOURNAL)
    required.difference_update(
        path for path in tuple(required) if path.startswith("attempts/")
    )
    for relative in (
        "dispatch-gates/manifest.json",
        "characterization/manifest.json",
        *(path.as_posix() for path in build.DISPATCH_BUILD_MANIFEST_PATHS.values()),
        *(path.as_posix() for path in build.CHARACTERIZATION_BUILD_MANIFEST_PATHS.values()),
    ):
        required.discard(relative)
    required.update(
        f"gate-collector/{relative}"
        for relative in protocol.regular_file_inventory(gate_root)
    )
    progress_path = root / PROGRESS_MANIFEST
    if progress_path.exists():
        progress = _read_json(progress_path, "canonical progress")
        required.difference_update(
            path for path in tuple(required) if path.startswith("children/")
        )
        required.update(
            _child_record_path(root, ordinal, child["stage"])
            .relative_to(root)
            .as_posix()
            for ordinal, child in enumerate(progress["children"], start=1)
        )
    for attempt in ledger["attempts"]:
        artifact = (
            pathlib.Path(attempt["artifact_root"])
            if attempt["artifact_root"] is not None
            else root
            / "attempts"
            / attempt["stage"]
            / attempt["lane"]
            / str(attempt["attempt_id"])
        )
        if artifact.is_dir():
            resolved = artifact.resolve(strict=True)
            root_resolved = root.resolve(strict=True)
            if root_resolved not in resolved.parents:
                raise protocol.ProtocolError("attempt artifact escapes aggregate root")
            required.update(
                path.relative_to(root).as_posix()
                for path in resolved.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
    return frozenset(required)


def validate_semantic_root(root: pathlib.Path) -> None:
    """Reopen required manifests with their owning semantic validators."""
    root = pathlib.Path(root).resolve(strict=True)
    ledger = _read_json(root / "evidence-ledger.json", "semantic ledger")
    protocol.validate_ledger(ledger)
    if (root / PROCESS_JOURNAL).exists():
        validate_process_journal(root, require_entries=True)
    candidate = ledger["candidate_sha"]
    for relative in build.LOCK_PATHS.values():
        try:
            payload = (root / relative).read_bytes()
        except OSError as error:
            raise protocol.ProtocolError(f"cannot read owned lock: {relative}") from error
        if not payload:
            raise protocol.ProtocolError(f"owned lock is empty: {relative}")
    tenferro = {}
    for role, relative in build.BUILD_MANIFEST_PATHS.items():
        manifest = _read_json(root / relative, f"{role} build manifest")
        build.validate_build_manifest(manifest)
        if manifest["head"] != candidate or manifest["role"] != role:
            raise protocol.ProtocolError("build manifest aggregate identity differs")
        lock_key = (
            "direct" if role == "direct-current-main-baseline" else "common"
        )
        if manifest["lock_sha256"] != protocol.sha256_file(
            root / build.LOCK_PATHS[lock_key]
        ):
            raise protocol.ProtocolError("build manifest lock digest differs")
        tenferro[role] = manifest
    progress_path = root / PROGRESS_MANIFEST
    progress = validate_progress(root)
    repository = pathlib.Path(progress["repository"])
    probes = build.validate_allocation_probe_set(
        root, tenferro, repository=repository
    )
    from scripts import run_phase1_eager_campaign as timing
    from scripts import run_phase2e_allocation_campaign as allocation

    for attempt in ledger["attempts"]:
        artifact = (
            pathlib.Path(attempt["artifact_root"])
            if attempt["artifact_root"] is not None
            else root / "attempts" / attempt["stage"] / attempt["lane"]
            / str(attempt["attempt_id"])
        )
        if attempt["stage"] == "allocation":
            allocation.validate_completed_attempt(
                artifact, ledger, comparison_kind=attempt["lane"],
                attempt_id=attempt["attempt_id"], probe_manifests=probes,
                tenferro_manifests=tenferro,
            )
        elif attempt["state"] in {"COMPLETE", "INCONCLUSIVE"}:
            timing.validate_retained_attempt(
                artifact, ledger, comparison_kind=attempt["lane"],
                attempt_id=attempt["attempt_id"],
            )
    gate_root = root / "gate-collector"
    if not gate_root.is_dir():
        raise protocol.ProtocolError("Task 8A gate-collector root is missing")
    from scripts import run_phase2e_gates as gates

    common_lock = gate_root / build.LOCK_PATHS["common"]
    gates.validate_terminal_evidence(
        gate_root,
        candidate=candidate,
        repository=repository,
        source_inventory=gates.validate_source_contract(repository),
        common_lock=common_lock,
    )


def _canonical_existing_root(root: pathlib.Path) -> pathlib.Path:
    root = pathlib.Path(root)
    try:
        canonical = root.resolve(strict=True)
    except OSError as error:
        raise protocol.ProtocolError(f"cannot resolve evidence root: {error}") from error
    if not root.is_absolute() or canonical != root or not root.is_dir():
        raise protocol.ProtocolError("evidence root must be a canonical directory")
    return root


def seal_root(
    root: pathlib.Path,
    *,
    candidate_sha: str,
    reservation_id: str,
    experiment_identity_digest: str,
) -> dict[str, Any]:
    """Write the aggregate manifest after all four measured lanes close."""
    _require_sha(candidate_sha, sha256=False, context="candidate SHA")
    _require_sha(experiment_identity_digest, sha256=True, context="experiment identity")
    root = _canonical_existing_root(root)
    validate_semantic_root(root)
    ledger = _read_json(root / "evidence-ledger.json", "evidence ledger")
    if ledger.get("candidate_sha") != candidate_sha:
        raise protocol.ProtocolError(
            "ledger candidate differs from aggregate candidate"
        )
    status = _terminal_ledger_status(ledger)
    gate_prefix = "gate-collector/"
    for relative in ("dispatch-gates/manifest.json", "characterization/manifest.json"):
        relative = gate_prefix + relative
        child = _read_json(root / relative, relative)
        if (
            child.get("candidate") != candidate_sha
            or child.get("gating_result", child.get("validity_state")) != "PASS"
        ):
            raise protocol.ProtocolError(f"{relative} does not pass every gate")
    inventory = protocol.regular_file_inventory(
        root, excluded=frozenset({AGGREGATE_MANIFEST})
    )
    required = _canonical_root_paths(root, ledger)
    if set(inventory) != required:
        raise protocol.ProtocolError(
            "root inventory is not the canonical Task 8A set: "
            f"missing={sorted(required - set(inventory))}, "
            f"extra={sorted(set(inventory) - required)}"
        )
    progress = _read_json(root / PROGRESS_MANIFEST, "aggregate progress")
    if progress["context_sha256"] != inventory[STAGE_CONTEXT]:
        raise protocol.ProtocolError(
            "aggregate progress differs from normative context bytes"
        )
    bound_command_digest = progress.get(
        "command_contract_digest", command_contract_digest()
    )
    _require_sha(bound_command_digest, sha256=True, context="aggregate command contract")
    manifest = {
        "version": 1,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "candidate_sha": candidate_sha,
        "reservation_id": reservation_id,
        "experiment_identity_digest": experiment_identity_digest,
        "command_contract_digest": bound_command_digest,
        "context_sha256": progress["context_sha256"],
        "status": status,
        "stage_order": list(STAGE_ORDER),
        "ledger_sha256": inventory["evidence-ledger.json"],
        "inventory": inventory,
    }
    protocol.atomic_write_json(root / AGGREGATE_MANIFEST, manifest)
    return manifest


def _validate_retained_timing_build_failure(
    root: pathlib.Path,
    inventory: Mapping[str, str],
    *,
    expected_evidence_root: pathlib.Path,
) -> None:
    """Rebind retained timing failure bytes to their immutable stage context."""
    if TIMING_BUILD_FAILURE not in inventory:
        return
    if STAGE_CONTEXT not in inventory:
        raise protocol.ProtocolError(
            "timing build failure evidence has no stage context"
        )
    artifact = root / TIMING_BUILD_FAILURE
    if protocol.sha256_file(artifact) != inventory[TIMING_BUILD_FAILURE]:
        raise protocol.ProtocolError(
            "timing build failure digest differs from owning inventory"
        )
    context_path = root / STAGE_CONTEXT
    if protocol.sha256_file(context_path) != inventory[STAGE_CONTEXT]:
        raise protocol.ProtocolError(
            "timing build stage context digest differs from owning inventory"
        )
    context = _read_json(context_path, "retained timing build stage context")
    _validate_stage_context_record(context)
    if pathlib.Path(context["evidence_root"]) != expected_evidence_root:
        raise protocol.ProtocolError(
            "timing build failure context names another evidence root"
        )
    payload = _read_json(artifact, "timing build failure evidence")
    validate_timing_build_failure(
        payload,
        context,
        require_live_paths=False,
    )


def _validate_root(
    root: pathlib.Path,
    *,
    expected_evidence_root: pathlib.Path,
) -> str:
    """Validate physical bytes against their immutable original root identity."""
    root = _canonical_existing_root(root)
    expected_evidence_root = _canonical_existing_root(expected_evidence_root)
    abandonment_path = root / ABANDONMENT_SEAL
    if abandonment_path.exists():
        seal = _read_json(abandonment_path, "abandonment seal")
        if set(seal) != {
            "version",
            "protocol_version",
            "status",
            "abandonment_kind",
            "inventory",
        } or (
            type(seal["version"]) is not int
            or seal["version"] != 1
            or type(seal["protocol_version"]) is not int
            or seal["protocol_version"] != protocol.PROTOCOL_VERSION
            or type(seal["status"]) is not str
            or seal["status"] != "ABANDONED"
            or type(seal["abandonment_kind"]) is not str
            or seal["abandonment_kind"]
            not in {"INITIALIZATION_FAILURE", "UNCAUGHT_INTERRUPTION"}
            or type(seal["inventory"]) is not dict
        ):
            raise protocol.ProtocolError("abandonment seal schema is invalid")
        protocol.validate_regular_file_inventory(
            root, seal["inventory"], excluded=frozenset({ABANDONMENT_SEAL})
        )
        _validate_retained_timing_build_failure(
            root,
            seal["inventory"],
            expected_evidence_root=expected_evidence_root,
        )
        return "ABANDONED"
    if os.path.lexists(root / TIMING_BUILD_FAILURE):
        raise protocol.ProtocolError(
            "complete evidence root contains timing build failure evidence"
        )
    manifest = _read_json(root / AGGREGATE_MANIFEST, "aggregate manifest")
    validate_semantic_root(root)
    required = {
        "version",
        "protocol_version",
        "candidate_sha",
        "reservation_id",
        "experiment_identity_digest",
        "command_contract_digest",
        "context_sha256",
        "status",
        "stage_order",
        "ledger_sha256",
        "inventory",
    }
    if set(manifest) != required or manifest["stage_order"] != list(STAGE_ORDER):
        raise protocol.ProtocolError("aggregate manifest schema or stage order differs")
    protocol.validate_regular_file_inventory(
        root,
        manifest["inventory"],
        excluded=frozenset({AGGREGATE_MANIFEST}),
    )
    ledger = _read_json(root / "evidence-ledger.json", "evidence ledger")
    if ledger.get("candidate_sha") != manifest["candidate_sha"]:
        raise protocol.ProtocolError(
            "ledger candidate differs from aggregate candidate"
        )
    progress = _read_json(root / PROGRESS_MANIFEST, "aggregate progress")
    if manifest["command_contract_digest"] != progress.get(
        "command_contract_digest", command_contract_digest()
    ) or manifest["context_sha256"] != progress.get("context_sha256"):
        raise protocol.ProtocolError("aggregate command contract differs")
    if manifest["context_sha256"] != manifest["inventory"][STAGE_CONTEXT]:
        raise protocol.ProtocolError(
            "aggregate context digest differs from normative context bytes"
        )
    if set(manifest["inventory"]) != _canonical_root_paths(root, ledger):
        raise protocol.ProtocolError("aggregate inventory contract differs")
    status = _terminal_ledger_status(ledger)
    if status != manifest["status"]:
        raise protocol.ProtocolError("aggregate status differs from ledger")
    if manifest["ledger_sha256"] != protocol.sha256_file(root / "evidence-ledger.json"):
        raise protocol.ProtocolError("aggregate ledger digest differs")
    gate_prefix = "gate-collector/"
    for relative in ("dispatch-gates/manifest.json", "characterization/manifest.json"):
        relative = gate_prefix + relative
        child = _read_json(root / relative, relative)
        if (
            child.get("candidate") != manifest["candidate_sha"]
            or child.get("gating_result", child.get("validity_state")) != "PASS"
        ):
            raise protocol.ProtocolError(f"{relative} does not pass every gate")
    return status


def validate_root(root: pathlib.Path) -> str:
    """Cryptographically reconstruct one complete aggregate evidence root."""
    root = _canonical_existing_root(root)
    return _validate_root(root, expected_evidence_root=root)


def seal_abandoned_root(
    root: pathlib.Path,
    *,
    identity: protocol.PreparedRootIdentity | None = None,
    abandonment_kind: str = "UNCAUGHT_INTERRUPTION",
) -> dict[str, Any]:
    """Own every preexisting regular byte after an unresumable interruption."""
    if abandonment_kind not in {
        "INITIALIZATION_FAILURE",
        "UNCAUGHT_INTERRUPTION",
    }:
        raise protocol.ProtocolError("abandonment kind differs")
    root = pathlib.Path(root)
    owned_identity = identity is None
    identity = identity or protocol.PreparedRootIdentity(root)
    inventory = protocol.regular_file_inventory_at(
        identity.descriptor, excluded=frozenset({ABANDONMENT_SEAL})
    )
    seal = {
        "version": 1,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "status": "ABANDONED",
        "abandonment_kind": abandonment_kind,
        "inventory": inventory,
    }
    try:
        protocol.atomic_write_json_at(identity.descriptor, ABANDONMENT_SEAL, seal)
        return seal
    finally:
        if owned_identity:
            identity.close()


@dataclass(frozen=True)
class GitBlob:
    mode: str
    object_id: str
    content: bytes


def validate_git_blob_inventory(
    root: pathlib.Path, staged_blobs: Mapping[str, bytes | GitBlob]
) -> None:
    """Require Git objects and modes to equal every root-owned regular file."""
    if type(staged_blobs) is not dict or any(
        type(path) is not str
        or not isinstance(payload, (bytes, GitBlob))
        for path, payload in staged_blobs.items()
    ):
        raise protocol.ProtocolError("Git blob inventory schema is invalid")
    inventory = protocol.regular_file_inventory(root)
    if set(staged_blobs) != set(inventory):
        raise protocol.ProtocolError("Git blob inventory has missing or extra paths")
    for relative, digest in inventory.items():
        entry = staged_blobs[relative]
        content = entry if isinstance(entry, bytes) else entry.content
        if isinstance(entry, GitBlob):
            path_mode = stat.S_IMODE((pathlib.Path(root) / relative).stat().st_mode)
            expected_mode = "100755" if path_mode & 0o111 else "100644"
            if entry.mode != expected_mode:
                raise protocol.ProtocolError(f"Git mode differs: {relative}")
            if re.fullmatch(r"[0-9a-f]{40}", entry.object_id) is None:
                raise protocol.ProtocolError(f"Git blob identity differs: {relative}")
        if hashlib.sha256(content).hexdigest() != digest:
            raise protocol.ProtocolError(f"Git blob differs: {relative}")


def validate_preservation_comment(
    issue_url: str,
    body: str,
    *,
    preservation_commit: str,
    root: str,
    candidate_sha: str,
    status: str,
) -> None:
    """Parse one frozen four-line #1436 preservation report."""
    if not _is_canonical_issue_comment_url(issue_url) or type(body) is not str:
        raise protocol.ProtocolError(
            "preservation comment is not permanent issue #1436"
        )
    _require_sha(
        preservation_commit, sha256=False, context="preservation commit"
    )
    _require_sha(candidate_sha, sha256=False, context="candidate SHA")
    if status not in TERMINAL_STATUSES:
        raise protocol.ProtocolError("preservation comment status is invalid")
    if type(root) is not str or _has_ascii_control(root):
        raise protocol.ProtocolError("preservation comment root is not canonical")
    root_path = pathlib.PurePosixPath(root)
    if (
        not root_path.is_absolute()
        or root_path.as_posix() != root
        or ".." in root_path.parts
    ):
        raise protocol.ProtocolError("preservation comment root is not canonical")
    expected_lines = [
        f"preservation_commit: {preservation_commit}",
        f"evidence_root: {root}",
        f"candidate: {candidate_sha}",
        f"terminal_status: {status}",
    ]
    expected = "\n".join(expected_lines) + "\n"
    if body != expected or body.splitlines() != expected_lines:
        raise protocol.ProtocolError("preservation comment format or identity differs")


def validate_preservation_comment_proof(
    issue_url: str,
    proof: Mapping[str, Any],
    *,
    preservation_commit: str,
    root: str,
    candidate_sha: str,
    status: str,
) -> None:
    """Validate canonical GitHub API metadata, not only caller-supplied text."""
    match = (
        re.fullmatch(
            rf"https://github\.com/tensor4all/tenferro-rs/issues/{ISSUE_NUMBER}"
            r"#issuecomment-([1-9][0-9]*)",
            issue_url,
        )
        if type(issue_url) is str
        else None
    )
    if match is None or type(proof) is not dict or set(proof) != {
        "id",
        "html_url",
        "issue_url",
        "body",
    }:
        raise protocol.ProtocolError("preservation comment proof schema differs")
    comment_id = int(match.group(1))
    if (
        type(proof["id"]) is not int
        or proof["id"] != comment_id
        or proof["html_url"] != issue_url
        or proof["issue_url"]
        != f"https://api.github.com/repos/tensor4all/tenferro-rs/issues/{ISSUE_NUMBER}"
    ):
        raise protocol.ProtocolError("preservation comment proof association differs")
    validate_preservation_comment(
        issue_url,
        proof["body"],
        preservation_commit=preservation_commit,
        root=root,
        candidate_sha=candidate_sha,
        status=status,
    )


def _parse_git_blob_listing(
    payload: bytes, *, selector: str, root_relative: str
) -> list[tuple[str, str, str]]:
    prefix = root_relative.rstrip("/") + "/"
    result = []
    for record in payload.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            fields = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise protocol.ProtocolError("Git evidence inventory is malformed") from error
        if selector == ":":
            if len(fields) != 3 or fields[2] != "0":
                raise protocol.ProtocolError("staged Git inventory has nonzero stage")
            mode, object_id = fields[:2]
        else:
            if len(fields) != 3 or fields[1] != "blob":
                raise protocol.ProtocolError("commit Git inventory is not blobs")
            mode, object_id = fields[0], fields[2]
        if mode not in {"100644", "100755"}:
            raise protocol.ProtocolError(f"Git evidence mode is forbidden: {mode}")
        if re.fullmatch(r"[0-9a-f]{40}", object_id) is None:
            raise protocol.ProtocolError("Git evidence object id is invalid")
        if not path.startswith(prefix):
            raise protocol.ProtocolError("Git evidence path escapes its root")
        relative = path.removeprefix(prefix)
        pure = pathlib.PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts:
            raise protocol.ProtocolError("Git evidence path is invalid")
        result.append((mode, object_id, relative))
    if len({relative for _, _, relative in result}) != len(result):
        raise protocol.ProtocolError("Git evidence inventory has duplicate paths")
    return result


def git_blob_inventory(
    repository: pathlib.Path, selector: str, root_relative: str
) -> dict[str, GitBlob]:
    """Read one exact root inventory from the index (``:``) or a commit."""
    if selector != ":" and re.fullmatch(r"[0-9a-f]{40}", selector) is None:
        raise protocol.ProtocolError("Git selector must be index or exact commit")
    command = ["git", "ls-files", "-z", "--stage", "--", root_relative]
    if selector != ":":
        command = [
            "git",
            "ls-tree",
            "-r",
            "-z",
            selector,
            "--",
            root_relative,
        ]
    listing = subprocess.run(command, cwd=repository, capture_output=True, check=False)
    if listing.returncode:
        raise protocol.ProtocolError("cannot enumerate Git evidence blobs")
    result = {}
    for mode, object_id, relative in _parse_git_blob_listing(
        listing.stdout, selector=selector, root_relative=root_relative
    ):
        blob = subprocess.run(
            ("git", "cat-file", "blob", object_id),
            cwd=repository,
            capture_output=True,
            check=False,
        )
        if blob.returncode:
            raise protocol.ProtocolError(f"cannot read Git evidence blob: {relative}")
        result[relative] = GitBlob(mode, object_id, blob.stdout)
    return result


def validate_git_selector(
    repository: pathlib.Path,
    root: pathlib.Path,
    *,
    selector: str,
    terminal_event: Mapping[str, Any] | None = None,
) -> None:
    """Validate exact staged/committed objects and a fresh reconstruction."""
    repository = pathlib.Path(repository).resolve(strict=True)
    root = pathlib.Path(root).resolve(strict=True)
    try:
        relative = root.relative_to(repository).as_posix()
    except ValueError as error:
        raise protocol.ProtocolError("evidence root is outside repository") from error
    entries = git_blob_inventory(repository, selector, relative)
    validate_git_blob_inventory(root, entries)
    with tempfile.TemporaryDirectory() as directory:
        reconstructed = pathlib.Path(directory) / "root"
        reconstructed.mkdir(mode=0o700)
        for name, entry in entries.items():
            destination = reconstructed / name
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            destination.write_bytes(entry.content)
            destination.chmod(0o755 if entry.mode == "100755" else 0o644)
        status = _validate_root(
            reconstructed,
            expected_evidence_root=root,
        )
        if terminal_event is not None:
            if terminal_event.get("status") != status:
                raise protocol.ProtocolError("Git root status differs from TERMINAL")
            if status == "ABANDONED":
                seal = _read_json(
                    reconstructed / ABANDONMENT_SEAL, "reconstructed abandonment seal"
                )
                root_digest = protocol.sha256_json(seal)
                ledger_digest = seal["inventory"].get(
                    "evidence-ledger.json", "0" * 64
                )
            else:
                root_digest = protocol.sha256_file(
                    reconstructed / AGGREGATE_MANIFEST
                )
                ledger_digest = protocol.sha256_file(
                    reconstructed / "evidence-ledger.json"
                )
            if (
                terminal_event.get("root_digest") != root_digest
                or terminal_event.get("ledger_sha256") != ledger_digest
            ):
                raise protocol.ProtocolError(
                    "Git root digest differs from TERMINAL"
                )


def _git_exact_blob(
    repository: pathlib.Path, selector: str, relative: str
) -> GitBlob:
    if selector != ":" and re.fullmatch(r"[0-9a-f]{40}", selector) is None:
        raise protocol.ProtocolError("Git selector must be index or exact commit")
    if selector == ":":
        command = ("git", "ls-files", "-z", "--stage", "--", relative)
    else:
        command = ("git", "ls-tree", "-z", selector, "--", relative)
    listing = subprocess.run(
        command, cwd=repository, capture_output=True, check=False
    )
    if listing.returncode:
        raise protocol.ProtocolError(f"cannot inspect Git object: {relative}")
    parsed = _parse_git_blob_listing(
        listing.stdout,
        selector=selector,
        root_relative=str(pathlib.PurePosixPath(relative).parent),
    )
    expected_name = pathlib.PurePosixPath(relative).name
    if len(parsed) != 1 or parsed[0][2] != expected_name:
        raise protocol.ProtocolError(f"Git object is missing or duplicated: {relative}")
    mode, object_id, _name = parsed[0]
    blob = subprocess.run(
        ("git", "cat-file", "blob", object_id),
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if blob.returncode:
        raise protocol.ProtocolError(f"cannot read Git object: {relative}")
    return GitBlob(mode, object_id, blob.stdout)


def _require_git_path_absent(
    repository: pathlib.Path, selector: str, relative: str
) -> None:
    if selector != ":" and re.fullmatch(r"[0-9a-f]{40}", selector) is None:
        raise protocol.ProtocolError("Git selector must be index or exact commit")
    command = (
        ("git", "ls-files", "-z", "--stage", "--", relative)
        if selector == ":"
        else ("git", "ls-tree", "-z", selector, "--", relative)
    )
    listing = subprocess.run(
        command, cwd=repository, capture_output=True, check=False
    )
    if listing.returncode:
        raise protocol.ProtocolError(f"cannot inspect forbidden Git path: {relative}")
    if listing.stdout:
        raise protocol.ProtocolError(f"forbidden Git path is present: {relative}")


def validate_preservation_objects(
    repository: pathlib.Path,
    *,
    selector: str,
    root: pathlib.Path,
    index_path: pathlib.Path,
    worklog: pathlib.Path,
    terminal_event: Mapping[str, Any],
    expected_index: Mapping[str, Any] | None = None,
) -> None:
    """Validate root, durable index, and curated worklog as exact Git blobs."""
    repository = pathlib.Path(repository).resolve(strict=True)
    root = pathlib.Path(root).resolve(strict=True)
    index_path = pathlib.Path(index_path).resolve(strict=True)
    worklog = pathlib.Path(worklog).resolve(strict=True)
    _require_git_path_absent(
        repository, selector, pathlib.Path(INDEX_LOCK_PATH).as_posix()
    )
    for path, context in ((index_path, "index"), (worklog, "worklog")):
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError as error:
            raise protocol.ProtocolError(
                f"preservation {context} is outside repository"
            ) from error
        entry = _git_exact_blob(repository, selector, relative)
        expected = (
            protocol._canonical_json_bytes(expected_index)
            if context == "index" and expected_index is not None
            else path.read_bytes()
        )
        if entry.mode != "100644" or entry.content != expected:
            raise protocol.ProtocolError(
                f"preservation {context} object differs"
            )
    validate_git_selector(
        repository,
        root,
        selector=selector,
        terminal_event=terminal_event,
    )


def validate_progress(
    root: pathlib.Path,
    *,
    running_process: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-hash the complete root before any retry/continuation mutation."""
    progress = _read_json(root / PROGRESS_MANIFEST, "Phase 2E progress")
    if (
        set(progress)
        != {
            "version",
            "stage_order",
            "children",
            "inventory",
            "protocol_version",
            "candidate_sha",
            "candidate_tree_sha256",
            "reservation_id",
            "experiment_identity_digest",
            "command_contract_digest",
            "context_sha256",
            "repository",
            "evidence_root",
            "scratch_parent",
            "path",
            "home",
            "cargo_home",
            "index",
            "index_lock",
            "context_path",
            "ledger_sha256",
            "process_journal_sha256",
        }
        or progress["version"] != 2
        or progress["protocol_version"] != protocol.PROTOCOL_VERSION
        or progress["stage_order"] != list(STAGE_ORDER)
        or type(progress["children"]) is not list
    ):
        raise protocol.ProtocolError("Phase 2E progress schema is invalid")
    _require_sha(
        progress["command_contract_digest"],
        sha256=True,
        context="progress command contract",
    )
    _require_sha(progress["context_sha256"], sha256=True, context="progress context")
    _require_sha(
        progress["process_journal_sha256"],
        sha256=True,
        context="progress process journal",
    )
    protocol.validate_regular_file_inventory(
        root,
        progress["inventory"],
        excluded=frozenset(
            {PROGRESS_MANIFEST, AGGREGATE_MANIFEST, PROCESS_JOURNAL}
        ),
    )
    if running_process is None:
        process_journal_path = root / PROCESS_JOURNAL
        if progress["process_journal_sha256"] == "0" * 64:
            if process_journal_path.exists():
                raise protocol.ProtocolError(
                    "synthetic progress unexpectedly has a process journal"
                )
        else:
            journal = validate_process_journal(root)
            if (
                protocol.sha256_json(journal)
                != progress["process_journal_sha256"]
            ):
                raise protocol.ProtocolError("process journal snapshot differs")
    else:
        _validate_process_journal_handoff(
            root,
            progress["process_journal_sha256"],
            running_process,
        )
    cursor = 0
    previous: Mapping[str, Any] | None = None
    child_schema = {
        "version", "ordinal", "stage", "attempt", "exit_code",
        "command_contract_digest", "before_inventory_sha256",
        "context_sha256", "candidate_sha", "reservation_id",
        "experiment_identity_digest", "argv", "environment",
        "after_inventory_sha256", "changed", "removed",
    }
    attempts: dict[str, int] = {}
    for ordinal, child in enumerate(progress["children"], start=1):
        if (
            type(child) is not dict
            or set(child) != {"stage", "exit_code"}
            or type(child["exit_code"]) is not int
            or child["exit_code"] not in {0, 2, 3, 4}
        ):
            raise protocol.ProtocolError("progress child schema is invalid")
        retry = previous is not None and previous["exit_code"] == 2
        if retry and previous["stage"] not in RETRYABLE_STAGES:
            raise protocol.ProtocolError(
                "progress retries a non-retryable stage"
            )
        if not retry and cursor >= len(STAGE_ORDER):
            raise protocol.ProtocolError("progress contains children after completion")
        expected_stage = previous["stage"] if retry else STAGE_ORDER[cursor]
        if child["stage"] != expected_stage:
            raise protocol.ProtocolError("progress child order differs")
        attempts[child["stage"]] = attempts.get(child["stage"], 0) + 1
        record = _read_json(
            _child_record_path(root, ordinal, child["stage"]),
            f"child record {ordinal}",
        )
        if (
            set(record) != child_schema
            or record["version"] != 1
            or record["ordinal"] != ordinal
            or record["stage"] != child["stage"]
            or record["attempt"] != attempts[child["stage"]]
            or record["exit_code"] != child["exit_code"]
            or record["command_contract_digest"] != progress["command_contract_digest"]
            or record["context_sha256"] != progress["context_sha256"]
            or record["candidate_sha"] != progress["candidate_sha"]
            or record["reservation_id"] != progress["reservation_id"]
            or record["experiment_identity_digest"]
            != progress["experiment_identity_digest"]
            or record["argv"] != list(
                stage_argv(
                    child["stage"], pathlib.Path(progress["context_path"]),
                    progress["context_sha256"],
                )
            )
            or record["environment"] != protocol.runtime_environment(
                path=progress["path"], home=progress["home"]
            )
        ):
            raise protocol.ProtocolError("child record identity differs")
        for name in ("before_inventory_sha256", "after_inventory_sha256"):
            _require_sha(record[name], sha256=True, context=name)
        if type(record["changed"]) is not dict or type(record["removed"]) is not list:
            raise protocol.ProtocolError("child record inventory delta is invalid")
        for relative, digest in record["changed"].items():
            if (
                type(relative) is not str
                or not relative
                or pathlib.PurePosixPath(relative).is_absolute()
                or ".." in pathlib.PurePosixPath(relative).parts
            ):
                raise protocol.ProtocolError("child record changed path is invalid")
            _require_sha(digest, sha256=True, context="child changed digest")
        if any(
            type(relative) is not str
            or not relative
            or pathlib.PurePosixPath(relative).is_absolute()
            or ".." in pathlib.PurePosixPath(relative).parts
            for relative in record["removed"]
        ) or len(record["removed"]) != len(set(record["removed"])):
            raise protocol.ProtocolError("child record removed paths are invalid")
        if child["exit_code"] == 0:
            cursor += 1
        elif ordinal != len(progress["children"]):
            if child["exit_code"] != 2:
                raise protocol.ProtocolError("terminal child is not last")
        previous = child
    ledger_path = root / "evidence-ledger.json"
    if ledger_path.exists():
        ledger = _read_json(ledger_path, "progress ledger")
        protocol.validate_ledger(ledger)
        if ledger["active_attempt_id"] is not None:
            raise protocol.ProtocolError("progress retains a RUNNING attempt")
        if (
            progress["candidate_sha"] != ledger["candidate_sha"]
            or progress["ledger_sha256"] != protocol.sha256_file(ledger_path)
        ):
            raise protocol.ProtocolError("progress identity differs from ledger")
        measured = {
            f"{stage['name']}/{lane['name']}": lane
            for stage in ledger["stages"]
            for lane in stage["lanes"]
        }
        for stage_name, lane in measured.items():
            matching = [
                child for child in progress["children"] if child["stage"] == stage_name
            ]
            if len(lane["attempt_ids"]) != len(matching):
                raise protocol.ProtocolError("progress attempt count differs from ledger")
            if matching:
                expected_state = "COMPLETE" if matching[-1]["exit_code"] in {0, 3, 4} else "RETRYABLE"
                if lane["state"] != expected_state:
                    raise protocol.ProtocolError("progress current lane differs from ledger")
    elif progress["ledger_sha256"] != "0" * 64:
        raise protocol.ProtocolError("progress ledger is absent but not marked synthetic")
    return progress


def _validate_process_journal_handoff(
    root: pathlib.Path,
    completed_sha256: str,
    running_process: Mapping[str, Any],
) -> None:
    """Validate one exact mutable RUNNING suffix over a retained journal prefix."""
    _require_sha(
        completed_sha256,
        sha256=True,
        context="completed process journal",
    )
    expected_fields = {"stage", "argv", "pid", "pgid", "start_ticks"}
    if (
        type(running_process) is not dict
        or set(running_process) != expected_fields
        or running_process["stage"] not in STAGE_ORDER
        or type(running_process["argv"]) is not list
        or any(type(value) is not str for value in running_process["argv"])
        or any(
            type(running_process[name]) is not int
            for name in ("pid", "pgid", "start_ticks")
        )
    ):
        raise protocol.ProtocolError("running process identity differs")
    journal = validate_process_journal(root, require_entries=True)
    running = journal["entries"][-1]
    prefix = {"version": journal["version"], "entries": journal["entries"][:-1]}
    _validate_process_journal_payload(prefix)
    if (
        protocol.sha256_json(prefix) != completed_sha256
        or any(entry["state"] == "RUNNING" for entry in prefix["entries"])
    ):
        raise protocol.ProtocolError("process journal completed prefix differs")
    if (
        running["state"] != "RUNNING"
        or any(running[name] != running_process[name] for name in expected_fields)
    ):
        raise protocol.ProtocolError("process journal RUNNING binding differs")


def _linux_process_identity(pid: int) -> dict[str, int]:
    try:
        content = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = content[content.rindex(")") + 2 :].split()
        start_ticks = int(fields[19])
    except (OSError, ValueError, IndexError) as error:
        raise protocol.ProtocolError(
            f"cannot read spawned process identity: {error}"
        ) from error
    return {"pid": pid, "start_ticks": start_ticks}


def validate_process_journal(
    root: pathlib.Path, *, require_entries: bool = False
) -> dict[str, Any]:
    payload = protocol.decode_canonical_json_bytes(
        _read_regular_path(pathlib.Path(root) / PROCESS_JOURNAL, "process journal"),
        "process journal",
    )
    return _validate_process_journal_payload(payload, require_entries=require_entries)


def _read_process_journal_at(
    identity: protocol.PreparedRootIdentity, *, require_entries: bool = False
) -> dict[str, Any]:
    identity.revalidate()
    payload = protocol.decode_canonical_json_bytes(
        _read_regular_at(identity.descriptor, PROCESS_JOURNAL, "process journal"),
        "process journal",
    )
    return _validate_process_journal_payload(payload, require_entries=require_entries)


def _validate_process_journal_payload(
    payload: Any, *, require_entries: bool = False
) -> dict[str, Any]:
    if (
        type(payload) is not dict
        or set(payload) != {"version", "entries"}
        or payload["version"] != 1
        or type(payload["entries"]) is not list
        or require_entries and not payload["entries"]
    ):
        raise protocol.ProtocolError("process journal schema differs")
    identities = set()
    for ordinal, entry in enumerate(payload["entries"], start=1):
        protocol.validate_manifest_fields(
            entry,
            {
                "ordinal": int,
                "stage": str,
                "argv": list,
                "pid": int,
                "pgid": int,
                "start_ticks": int,
                "state": str,
                "exit_code": (int, type(None)),
                "signals": list,
                "reaped": bool,
            },
            context="process journal entry",
        )
        process_identity = (entry["pid"], entry["start_ticks"])
        if (
            entry["ordinal"] != ordinal
            or entry["stage"] not in STAGE_ORDER
            or type(entry["argv"]) is not list
            or any(type(value) is not str for value in entry["argv"])
            or entry["pid"] <= 0
            or entry["pgid"] <= 0
            or entry["start_ticks"] < 0
            or process_identity in identities
            or entry["state"] not in {"RUNNING", "EXITED", "TERMINATED"}
            or any(value not in {"TERM", "KILL"} for value in entry["signals"])
            or (entry["state"] == "RUNNING" and (entry["reaped"] or entry["exit_code"] is not None))
            or (entry["state"] != "RUNNING" and not entry["reaped"])
        ):
            raise protocol.ProtocolError("process journal entry differs")
        identities.add(process_identity)
    return payload


def _write_process_journal(
    root: pathlib.Path,
    journal: Mapping[str, Any],
    *,
    identity: protocol.PreparedRootIdentity | None = None,
) -> None:
    owned = identity is None
    identity = identity or protocol.PreparedRootIdentity(pathlib.Path(root))
    try:
        identity.revalidate()
        protocol.atomic_write_json_at(identity.descriptor, PROCESS_JOURNAL, journal)
        identity.revalidate()
    finally:
        if owned:
            identity.close()


def _start_process_journal_entry(
    root: pathlib.Path,
    *,
    stage: str,
    argv: Sequence[str],
    pid: int,
    pgid: int,
    identity: Mapping[str, int],
    root_identity: protocol.PreparedRootIdentity | None = None,
) -> int:
    journal = (
        _read_process_journal_at(root_identity)
        if root_identity is not None
        else validate_process_journal(root)
    )
    entry = {
        "ordinal": len(journal["entries"]) + 1,
        "stage": stage,
        "argv": list(argv),
        "pid": pid,
        "pgid": pgid,
        "start_ticks": identity["start_ticks"],
        "state": "RUNNING",
        "exit_code": None,
        "signals": [],
        "reaped": False,
    }
    journal["entries"].append(entry)
    _write_process_journal(root, journal, identity=root_identity)
    return entry["ordinal"]


def _finish_process_journal_entry(
    root: pathlib.Path,
    ordinal: int,
    *,
    state: str,
    exit_code: int,
    signals: Sequence[str],
    root_identity: protocol.PreparedRootIdentity | None = None,
) -> None:
    journal = (
        _read_process_journal_at(root_identity, require_entries=True)
        if root_identity is not None
        else validate_process_journal(root, require_entries=True)
    )
    entry = journal["entries"][ordinal - 1]
    if entry["ordinal"] != ordinal or entry["state"] != "RUNNING":
        raise protocol.ProtocolError("process journal completion differs")
    entry.update(
        state=state,
        exit_code=exit_code,
        signals=list(signals),
        reaped=True,
    )
    _write_process_journal(root, journal, identity=root_identity)


def _terminate_and_reap_process(
    process,
    pgid: int,
    *,
    kill_process_group: Callable[[int, int], None],
    termination_grace_seconds: float,
) -> tuple[int, list[str], list[BaseException]]:
    """Best-effort signal one group while independently guaranteeing a reap."""
    signals = []
    signal_errors: list[BaseException] = []

    def send(sig: int, name: str) -> None:
        try:
            kill_process_group(pgid, sig)
        except ProcessLookupError:
            # The child/group may exit between Popen and cleanup.  wait still
            # owns the authoritative reap below.
            return
        except BaseException as error:
            signal_errors.append(error)
        else:
            signals.append(name)

    send(signal.SIGTERM, "TERM")
    try:
        code = process.wait(timeout=termination_grace_seconds)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL, "KILL")
        code = process.wait()
    return code, signals, signal_errors


def _subprocess_stage_runner(
    context: pathlib.Path,
    context_sha256: str,
    repository: pathlib.Path,
    *,
    root: pathlib.Path | None = None,
    root_identity: protocol.PreparedRootIdentity | None = None,
    process_factory=subprocess.Popen,
    process_identity: Callable[[int], Mapping[str, int]] = _linux_process_identity,
    process_group: Callable[[int], int] = os.getpgid,
    kill_process_group: Callable[[int, int], None] = os.killpg,
    deadline_seconds: float = 1800.0,
    termination_grace_seconds: float = 5.0,
) -> Callable[[str, Mapping[str, str]], int]:
    root = pathlib.Path(root) if root is not None else None

    def run(stage: str, environment: Mapping[str, str]) -> int:
        argv = stage_argv(stage, context, context_sha256)
        validate_stage_argv(stage, argv, context, context_sha256)
        process = process_factory(
            argv,
            cwd=repository,
            env=dict(environment),
            start_new_session=True,
        )
        pgid = process.pid
        ordinal = None
        reaped = False
        try:
            if root is None:
                raise protocol.ProtocolError("stage subprocess runner lacks evidence root")
            if root_identity is not None:
                root_identity.revalidate()
            identity = process_identity(process.pid)
            if identity.get("pid") != process.pid:
                raise protocol.ProtocolError("spawned process identity differs")
            pgid = process_group(process.pid)
            if root_identity is not None:
                root_identity.revalidate()
            ordinal = _start_process_journal_entry(
                root,
                stage=stage,
                argv=argv,
                pid=process.pid,
                pgid=pgid,
                identity=identity,
                root_identity=root_identity,
            )
            code = process.wait(timeout=deadline_seconds)
            reaped = True
        except BaseException as primary:
            try:
                signal_errors = []
                signals = []
                if not reaped:
                    code, signals, signal_errors = _terminate_and_reap_process(
                        process,
                        pgid,
                        kill_process_group=kill_process_group,
                        termination_grace_seconds=termination_grace_seconds,
                    )
                    reaped = True
                if ordinal is not None:
                    _finish_process_journal_entry(
                        root,
                        ordinal,
                        state="TERMINATED",
                        exit_code=code,
                        signals=signals,
                        root_identity=root_identity,
                    )
            except BaseException as cleanup_error:
                raise protocol.ProtocolError(
                    f"cannot terminate and reap stage process: {cleanup_error}"
                ) from cleanup_error
            cleanup_note = None
            if signal_errors:
                details = "; ".join(str(error) for error in signal_errors)
                cleanup_note = f"stage cleanup signal failed after reap: {details}"
                primary.add_note(cleanup_note)
            if isinstance(primary, subprocess.TimeoutExpired):
                timeout_error = protocol.ProtocolError(
                    "stage process deadline exceeded"
                )
                if cleanup_note is not None:
                    timeout_error.add_note(cleanup_note)
                raise timeout_error from primary
            raise
        _finish_process_journal_entry(
            root,
            ordinal,
            state="EXITED",
            exit_code=code,
            signals=[],
            root_identity=root_identity,
        )
        return code

    return run


def validate_resume_identity(
    active: Mapping[str, Any],
    stage_context: Mapping[str, Any],
    context_sha256: str,
    progress: Mapping[str, Any],
    *,
    context_path: pathlib.Path | None = None,
) -> None:
    """Reject every mutable resume identity before a child can mutate evidence."""
    for name in (
        "candidate_sha", "candidate_tree_sha256",
        "experiment_identity_digest", "command_contract_digest",
    ):
        if active[name] != stage_context[name]:
            raise protocol.ProtocolError(f"stage context differs from ACTIVE at {name}")
    if active["context_sha256"] != context_sha256:
        raise protocol.ProtocolError("stage context digest differs from ACTIVE")
    for name in (
        "candidate_sha", "candidate_tree_sha256", "reservation_id",
        "experiment_identity_digest", "command_contract_digest", "context_sha256",
        "repository", "evidence_root", "scratch_parent", "path", "home", "cargo_home",
        "index", "index_lock",
        "context_path",
    ):
        if name == "context_path" and context_path is not None:
            expected = str(context_path)
        else:
            expected = active[name] if name in active else stage_context[name]
        if progress[name] != expected:
            raise protocol.ProtocolError(f"retry progress differs at {name}")


def _validate_terminal_root_binding(
    active: Mapping[str, Any], root: pathlib.Path, status: str
) -> None:
    """Bind terminal complete evidence to every identity carried by ACTIVE."""
    if status == "ABANDONED":
        return
    manifest = _read_json(root / AGGREGATE_MANIFEST, "aggregate manifest")
    progress = _read_json(root / PROGRESS_MANIFEST, "aggregate progress")
    ledger = _read_json(root / "evidence-ledger.json", "evidence ledger")
    expected_manifest = {
        "candidate_sha": active["candidate_sha"],
        "reservation_id": active["reservation_id"],
        "experiment_identity_digest": active["experiment_identity_digest"],
        "command_contract_digest": active["command_contract_digest"],
        "context_sha256": active["context_sha256"],
    }
    if any(manifest.get(name) != value for name, value in expected_manifest.items()):
        raise protocol.ProtocolError("terminal aggregate identity differs from ACTIVE")
    if (
        progress.get("candidate_sha") != active["candidate_sha"]
        or progress.get("candidate_tree_sha256") != active["candidate_tree_sha256"]
        or progress.get("reservation_id") != active["reservation_id"]
        or progress.get("experiment_identity_digest")
        != active["experiment_identity_digest"]
        or progress.get("command_contract_digest")
        != active["command_contract_digest"]
        or progress.get("context_sha256") != active["context_sha256"]
        or ledger.get("candidate_sha") != active["candidate_sha"]
    ):
        raise protocol.ProtocolError("terminal progress identity differs from ACTIVE")


@contextmanager
def active_campaign_lock(
    repository: pathlib.Path,
    root: pathlib.Path,
    *,
    reservation_id: str,
):
    """Hold the fixed index then the ACTIVE root lock for one operation."""
    with campaign_index_transaction(repository) as index_transaction:
        index = index_transaction.read()
        if index_state(index) != "ACTIVE":
            raise protocol.ProtocolError("campaign reservation is finalized")
        active = index["events"][-1]
        root = _canonical_existing_root(root)
        if (
            active["reservation_id"] != reservation_id
            or active["root"] != str(root)
        ):
            raise protocol.ProtocolError("active operation identity differs")
        identity = protocol.PreparedRootIdentity(root)
        try:
            with exclusive_lock_at(identity.descriptor, ORCHESTRATOR_LOCK_NAME):
                identity.revalidate()
                yield active, identity
                identity.revalidate()
        finally:
            identity.close()


def record_index_root(
    *,
    repository: pathlib.Path,
    root: pathlib.Path,
    reservation_id: str,
    abandoned: bool = False,
) -> dict[str, Any]:
    """Validate and transition one ACTIVE reservation to pending preservation."""
    with campaign_index_transaction(repository) as index_transaction:
        index = index_transaction.read()
        root = _canonical_existing_root(root)
        state = index_state(index)
        if state == "ACTIVE":
            active = index["events"][-1]
        elif state == "PENDING_PRESERVATION":
            terminal = index["events"][-1]
            active = next(
                event
                for event in reversed(index["events"])
                if event["event"] == "ACTIVE"
                and event["reservation_id"] == terminal["reservation_id"]
            )
        else:
            raise protocol.ProtocolError("campaign has no recordable reservation")
        if (
            active["reservation_id"] != reservation_id
            or active["root"] != str(root)
        ):
            raise protocol.ProtocolError("record-index identity differs from ACTIVE")
        identity = protocol.PreparedRootIdentity(root)
        try:
            with exclusive_lock_at(identity.descriptor, ORCHESTRATOR_LOCK_NAME):
                if state == "PENDING_PRESERVATION":
                    status = validate_root(root)
                    if status == "ABANDONED":
                        seal = _read_json(
                            root / ABANDONMENT_SEAL, "abandonment seal"
                        )
                        digest = protocol.sha256_json(seal)
                        ledger_digest = seal["inventory"].get(
                            "evidence-ledger.json", "0" * 64
                        )
                        if seal["abandonment_kind"] != "UNCAUGHT_INTERRUPTION":
                            raise protocol.ProtocolError(
                                "initialization abandonment cannot be replayed by record-index"
                            )
                    else:
                        digest = protocol.sha256_file(root / AGGREGATE_MANIFEST)
                        ledger_digest = protocol.sha256_file(
                            root / "evidence-ledger.json"
                        )
                    if abandoned != (status == "ABANDONED") or (
                        terminal["status"] != status
                        or terminal["root_digest"] != digest
                        or terminal["ledger_sha256"] != ledger_digest
                    ):
                        raise protocol.ProtocolError(
                            "record-index replay differs from terminal evidence"
                        )
                    identity.revalidate()
                    return index
                if abandoned:
                    journal = _read_process_journal_at(identity)
                    for process_group in sorted(
                        {entry["pgid"] for entry in journal["entries"]}
                    ):
                        try:
                            os.killpg(process_group, 0)
                        except ProcessLookupError:
                            continue
                        except PermissionError as error:
                            raise protocol.ProtocolError(
                                "cannot confirm process-group state"
                            ) from error
                        raise protocol.ProtocolError(
                            "a recorded process group is still live"
                        )
                    seal = seal_abandoned_root(
                        root,
                        identity=identity,
                        abandonment_kind="UNCAUGHT_INTERRUPTION",
                    )
                    status = "ABANDONED"
                    digest = protocol.sha256_json(seal)
                    ledger_digest = seal["inventory"].get(
                        "evidence-ledger.json", "0" * 64
                    )
                else:
                    status = validate_root(root)
                    if status == "ABANDONED":
                        raise protocol.ProtocolError(
                            "normal record-index cannot record abandoned evidence"
                        )
                    digest = protocol.sha256_file(root / AGGREGATE_MANIFEST)
                    ledger_digest = protocol.sha256_file(
                        root / "evidence-ledger.json"
                    )
                _validate_terminal_root_binding(active, root, status)
                identity.revalidate()
                updated = record_terminal(
                    index,
                    reservation_id=reservation_id,
                    status=status,
                    root_digest=digest,
                    ledger_sha256=ledger_digest,
                )
                index_transaction.write(updated)
                return updated
        finally:
            identity.close()


def fetch_comment(url: str) -> dict[str, Any]:
    """Fetch one permanent GitHub comment representation."""
    match = re.fullmatch(
        r"https://github\.com/([^/]+)/([^/]+)/issues/1436#issuecomment-([0-9]+)",
        url,
    )
    if match is None:
        raise protocol.ProtocolError("preservation comment URL is not canonical")
    owner, repository, comment_id = match.groups()
    api_url = (
        f"https://api.github.com/repos/{owner}/{repository}/issues/comments/"
        f"{comment_id}"
    )
    request = urllib.request.Request(
        api_url, headers={"Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot fetch preservation comment: {error}"
        ) from error
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise protocol.ProtocolError(
            "preservation comment response is malformed"
        ) from error
    if type(decoded) is not dict:
        raise protocol.ProtocolError("preservation comment response is not an object")
    proof = {
        name: decoded.get(name)
        for name in ("id", "html_url", "issue_url", "body")
    }
    if (
        type(proof["id"]) is not int
        or any(type(proof[name]) is not str for name in ("html_url", "issue_url", "body"))
    ):
        raise protocol.ProtocolError("preservation comment response lacks proof fields")
    return proof


def _git(repository: pathlib.Path, *argv: str, text: bool = False):
    result = subprocess.run(
        ("git", *argv), cwd=repository, capture_output=True, check=False, text=text
    )
    if result.returncode:
        raise protocol.ProtocolError(f"Git command failed: {' '.join(argv)}")
    return result.stdout


def validate_canonical_origin_url(url: str) -> None:
    """Allow only canonical transport forms for tensor4all/tenferro-rs."""
    if url not in {
        "https://github.com/tensor4all/tenferro-rs.git",
        "git@github.com:tensor4all/tenferro-rs.git",
        "ssh://git@github.com/tensor4all/tenferro-rs.git",
    }:
        raise protocol.ProtocolError("origin URL is not canonical tenferro-rs")


def validate_remote_preservation(
    repository: pathlib.Path, preservation_commit: str
) -> None:
    """Fetch the one preservation branch and prove commit reachability."""
    _require_sha(
        preservation_commit, sha256=False, context="preservation commit"
    )
    origin = _git(repository, "remote", "get-url", "origin", text=True).strip()
    validate_canonical_origin_url(origin)
    _git(repository, "fetch", "origin", "codex/execution-engine-through-phase9")
    _git(
        repository,
        "merge-base",
        "--is-ancestor",
        preservation_commit,
        PRESERVATION_BRANCH,
    )


def require_remote_index(
    repository: pathlib.Path,
    index_path: pathlib.Path,
    *,
    allow_absent: bool,
    transaction: CampaignIndexTransaction | None = None,
) -> None:
    """Require the local durable index to equal the fetched branch blob."""
    if transaction is not None:
        transaction.revalidate()
    origin = _git(repository, "remote", "get-url", "origin", text=True).strip()
    validate_canonical_origin_url(origin)
    _git(repository, "fetch", "origin", "codex/execution-engine-through-phase9")
    if transaction is not None:
        transaction.revalidate()
    relative = pathlib.Path(INDEX_PATH).as_posix()
    result = subprocess.run(
        ("git", "show", f"{PRESERVATION_BRANCH}:{relative}"),
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if transaction is not None:
        transaction.revalidate()
    if result.returncode:
        exists = (
            transaction.exists()
            if transaction is not None
            else _regular_path_exists(index_path, "Phase 2E index")
        )
        if allow_absent and not exists:
            return
        raise protocol.ProtocolError("remote branch lacks the durable Phase 2E index")
    if (
        not (
            transaction.exists()
            if transaction is not None
            else _regular_path_exists(index_path, "Phase 2E index")
        )
        or result.stdout
        != (
            transaction.read_bytes()
            if transaction is not None
            else _read_regular_path(index_path, "Phase 2E index")
        )
    ):
        raise protocol.ProtocolError("local Phase 2E index is not pushed byte-for-byte")


def record_preserved(
    *,
    repository: pathlib.Path,
    root: pathlib.Path,
    reservation_id: str,
    preservation_commit: str,
    issue_url: str,
    worklog: pathlib.Path,
    remote_validator: Callable[[pathlib.Path, str], None] = validate_remote_preservation,
    comment_fetcher: Callable[[str], Mapping[str, Any]] = fetch_comment,
) -> dict[str, Any]:
    """Verify remote Git/root/index/comment preservation, then append PRESERVED."""
    _index_path, _index_lock = campaign_index_paths(repository)
    with campaign_index_transaction(repository) as index_transaction:
        index = index_transaction.read()
        state = index_state(index)
        replay = state == "PRESERVED"
        if state not in {"PENDING_PRESERVATION", "PRESERVED"}:
            raise protocol.ProtocolError("no campaign is pending preservation")
        if replay:
            preserved = index["events"][-1]
            if (
                preserved["reservation_id"] != reservation_id
                or preserved["preservation_commit"] != preservation_commit
                or preserved["issue_url"] != issue_url
            ):
                raise protocol.ProtocolError("record-preserved replay differs")
            pending = copy.deepcopy(index)
            pending["events"].pop()
            previous_passes = [
                event["root"]
                for event in pending["events"]
                if event["event"] == "PRESERVED" and event["status"] == "PASS"
            ]
            pending["current_evidence_root"] = (
                previous_passes[-1] if previous_passes else None
            )
            _validate_index(pending)
        else:
            pending = index
        active = next(
            event
            for event in reversed(pending["events"])
            if event["reservation_id"] == reservation_id and event["event"] == "ACTIVE"
        )
        terminal = pending["events"][-1]
        root = _canonical_existing_root(root)
        if active["root"] != str(root) or terminal["root"] != str(root):
            raise protocol.ProtocolError("record-preserved root differs from pending")
        worklog = pathlib.Path(worklog).resolve(strict=True)
        if (
            worklog.parent != pathlib.Path(repository) / INDEX_PATH.parent
            or worklog.suffix != ".md"
        ):
            raise protocol.ProtocolError("curated worklog path is not canonical")
        root_identity = protocol.PreparedRootIdentity(root)
        try:
            with exclusive_lock_at(root_identity.descriptor, ORCHESTRATOR_LOCK_NAME):
                root_identity.revalidate()
                remote_validator(pathlib.Path(repository), preservation_commit)
                index_transaction.revalidate()
                validate_preservation_objects(
                    repository,
                    selector=preservation_commit,
                    root=root,
                    index_path=pathlib.Path(repository) / INDEX_PATH,
                    worklog=worklog,
                    terminal_event=terminal,
                    expected_index=pending,
                )
                root_identity.revalidate()
                validate_preservation_comment_proof(
                    issue_url,
                    comment_fetcher(issue_url),
                    preservation_commit=preservation_commit,
                    root=active["root"],
                    candidate_sha=active["candidate_sha"],
                    status=terminal["status"],
                )
                root_identity.revalidate()
                if replay:
                    return index
                updated = record_preserved_event(
                    index,
                    reservation_id=reservation_id,
                    preservation_commit=preservation_commit,
                    issue_url=issue_url,
                )
                index_transaction.write(updated)
                return updated
        finally:
            root_identity.close()


def _exit_for_status(status: str, *, require_pass: bool) -> int:
    if status == "PASS":
        return 0
    if not require_pass:
        return 0
    if status == "VALIDITY_INCONCLUSIVE":
        return 2
    if status == "STATISTICAL_INCONCLUSIVE":
        return 4
    return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_phase2e.py", exit_on_error=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", exit_on_error=False)
    validate.add_argument("--evidence-root", required=True, type=pathlib.Path)
    validate.add_argument("--print-status", action="store_true")
    validate.add_argument("--require-pass", action="store_true")
    validate.add_argument("--git-index", action="store_true")
    compare = subparsers.add_parser("compare-experiment-identity", exit_on_error=False)
    compare.add_argument("--repository", required=True, type=pathlib.Path)
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--contract", required=True, type=pathlib.Path)
    start = subparsers.add_parser("start", exit_on_error=False)
    start.add_argument("--repository", required=True, type=pathlib.Path)
    start.add_argument("--candidate", required=True)
    start.add_argument("--evidence-root", required=True, type=pathlib.Path)
    start.add_argument("--index", required=True, type=pathlib.Path)
    start.add_argument("--scratch-parent", required=True, type=pathlib.Path)
    for name in ("rerun-invalid-lane", "continue"):
        command = subparsers.add_parser(name, exit_on_error=False)
        command.add_argument("--evidence-root", required=True, type=pathlib.Path)
    record = subparsers.add_parser("record-index", exit_on_error=False)
    record.add_argument("--evidence-root", required=True, type=pathlib.Path)
    record.add_argument("--index", required=True, type=pathlib.Path)
    record.add_argument("--abandoned", action="store_true")
    record.add_argument("--confirm-no-live-processes", action="store_true")
    preserved = subparsers.add_parser("record-preserved", exit_on_error=False)
    preserved.add_argument("--evidence-root", required=True, type=pathlib.Path)
    preserved.add_argument("--index", required=True, type=pathlib.Path)
    preserved.add_argument("--preservation-commit", required=True)
    preserved.add_argument("--issue-comment-url", required=True)
    return parser


def _build_stage_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_phase2e.py _stage-worker", exit_on_error=False
    )
    parser.add_argument("--stage", required=True, choices=STAGE_ORDER)
    parser.add_argument("--context", required=True, type=pathlib.Path)
    parser.add_argument("--context-sha256", required=True)
    return parser


def _active_for_root(
    repository: pathlib.Path,
    root: pathlib.Path,
    *,
    allowed_states: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the fixed durable index and locate the reservation owning ``root``."""
    with campaign_index_transaction(repository) as transaction:
        index = transaction.read()
    state = index_state(index)
    if state not in allowed_states:
        raise protocol.ProtocolError("campaign index state is not valid for this command")
    current = index["events"][-1]
    reservation_id = current["reservation_id"]
    active = next(
        event
        for event in reversed(index["events"])
        if event["event"] == "ACTIVE"
        and event["reservation_id"] == reservation_id
    )
    if active["root"] != str(root):
        raise protocol.ProtocolError("evidence root differs from the durable reservation")
    return index, active


def _load_bound_context(
    repository: pathlib.Path,
    root: pathlib.Path,
    active: Mapping[str, Any],
) -> tuple[pathlib.Path, dict[str, Any], str]:
    """Load only the normative context bound by ACTIVE."""
    context_path = root / STAGE_CONTEXT
    context_sha256 = active["context_sha256"]
    context = load_stage_context(
        context_path, context_sha256, require_fresh_scratch=False
    )
    expected = {
        "repository": str(repository),
        "evidence_root": str(root),
        "index": str(repository / INDEX_PATH),
        "index_lock": str(repository / INDEX_LOCK_PATH),
        "candidate_sha": active["candidate_sha"],
        "candidate_tree_sha256": active["candidate_tree_sha256"],
        "reservation_id": active["reservation_id"],
        "experiment_identity_digest": active["experiment_identity_digest"],
        "command_contract_digest": active["command_contract_digest"],
    }
    for name, value in expected.items():
        if context[name] != value:
            raise protocol.ProtocolError(f"normative context differs at {name}")
    head = _git(repository, "rev-parse", "HEAD", text=True).strip()
    tree = _git(repository, "ls-tree", "-r", "-z", "--full-tree", head)
    if (
        head != active["candidate_sha"]
        or hashlib.sha256(tree).hexdigest() != active["candidate_tree_sha256"]
        or git_experiment_identity(
            repository, head, canonical_experiment_contract()
        )
        != active["experiment_identity_digest"]
    ):
        raise protocol.ProtocolError("current candidate provenance differs from ACTIVE")
    context["context_path"] = str(context_path)
    return context_path, context, context_sha256


def main(argv: Sequence[str] | None = None) -> int:
    try:
        raw_argv = tuple(sys.argv[1:] if argv is None else argv)
        if raw_argv[:1] == ("_stage-worker",):
            args = _build_stage_worker_parser().parse_args(raw_argv[1:])
            return execute_stage_worker(
                args.stage, args.context, args.context_sha256
            )
        args = build_parser().parse_args(raw_argv)
        if args.command == "validate":
            repository, root = _infer_repository_for_root(args.evidence_root)
            status = validate_root(root)
            if args.git_index:
                index_path, _lock_path = campaign_index_paths(repository)
                index = _read_index(index_path)
                terminal = next(
                    (
                        event
                        for event in reversed(index["events"])
                        if event["event"] == "TERMINAL"
                        and event["root"] == str(root)
                    ),
                    None,
                )
                if terminal is None:
                    raise protocol.ProtocolError(
                        "staged root has no matching TERMINAL event"
                    )
                validate_preservation_objects(
                    repository,
                    selector=":",
                    root=root,
                    index_path=index_path,
                    worklog=repository / WORKLOG_PATH,
                    terminal_event=terminal,
                )
            if args.print_status:
                print(status)
            return _exit_for_status(status, require_pass=args.require_pass)
        if args.command == "compare-experiment-identity":
            contract = _read_json(args.contract, "experiment contract")
            left = git_experiment_identity(args.repository, args.left, contract)
            right = git_experiment_identity(args.repository, args.right, contract)
            print(left)
            return 0 if left == right else 1
        if args.command == "start":
            repository = _canonical_repository(args.repository)
            root = _canonical_evidence_root(
                repository, args.evidence_root, must_exist=False
            )
            _require_fixed_index_argument(repository, args.index)
            stage_context, context_sha256, campaign_identity = _prepare_start_context(
                repository=repository,
                root=root,
                scratch_parent=args.scratch_parent,
                candidate=args.candidate,
            )
            execution_homes: _PreparedExecutionHomes | None = None
            try:
                execution_homes = _prepare_execution_homes(
                    pathlib.Path(stage_context["scratch_parent"])
                )
                home, cargo_home = execution_homes
                if (
                    home != pathlib.Path(stage_context["home"])
                    or cargo_home != pathlib.Path(stage_context["cargo_home"])
                ):
                    raise protocol.ProtocolError(
                        "derived execution home paths changed during start"
                    )
                _preflight_offline_feature_queries(stage_context)
                context_path = root / STAGE_CONTEXT

                def initialize_root(
                    root_identity: protocol.PreparedRootIdentity,
                ) -> None:
                    root_identity.revalidate()
                    protocol.atomic_write_json_at(
                        root_identity.descriptor, STAGE_CONTEXT, stage_context
                    )
                    root_identity.revalidate()
                    protocol.atomic_write_json_at(
                        root_identity.descriptor,
                        "evidence-ledger.json",
                        protocol.new_ledger(args.candidate),
                    )
                    root_identity.revalidate()

                code = initialize_campaign(
                    repository=repository,
                    root=root,
                    reservation_id=stage_context["reservation_id"],
                    candidate_sha=args.candidate,
                    candidate_tree_sha256=stage_context["candidate_tree_sha256"],
                    experiment_identity_digest=stage_context[
                        "experiment_identity_digest"
                    ],
                    campaign_identity_digest=campaign_identity,
                    command_digest=stage_context["command_contract_digest"],
                    context_sha256=context_sha256,
                    initializer=initialize_root,
                )
            except BaseException as primary:
                if execution_homes is not None:
                    execution_homes.rollback(primary)
                raise
            _close_committed_execution_homes(execution_homes)
            if code == 5:
                print("ABANDONED_INITIALIZATION")
                return 5
            stage_context = load_stage_context(
                context_path, context_sha256, require_fresh_scratch=False
            )
            stage_context["context_path"] = str(context_path)
            environment = protocol.runtime_environment(
                path=stage_context["path"], home=stage_context["home"]
            )
            with active_campaign_lock(
                repository,
                root,
                reservation_id=stage_context["reservation_id"],
            ) as (_active, root_identity):
                return run_fixed_stages(
                    root,
                    environment,
                    _subprocess_stage_runner(
                        context_path,
                        context_sha256,
                        repository,
                        root=root,
                        root_identity=root_identity,
                    ),
                    identity={
                        **stage_context,
                        "context_sha256": context_sha256,
                        "context_path": str(context_path),
                    },
                    root_identity=root_identity,
                    _locked=True,
                )
        if args.command in {"rerun-invalid-lane", "continue"}:
            repository, root = _infer_repository_for_root(args.evidence_root)
            _index, preliminary = _active_for_root(
                repository, root, allowed_states=frozenset({"ACTIVE"})
            )
            with active_campaign_lock(
                repository,
                root,
                reservation_id=preliminary["reservation_id"],
            ) as (active, root_identity):
                context_path, stage_context, context_sha256 = _load_bound_context(
                    repository, root, active
                )
                environment = protocol.runtime_environment(
                    path=stage_context["path"], home=stage_context["home"]
                )
                runner = _subprocess_stage_runner(
                    context_path,
                    context_sha256,
                    repository,
                    root=root,
                    root_identity=root_identity,
                )
                root_identity.revalidate()
                progress = validate_progress(root)
                validate_resume_identity(
                    active,
                    stage_context,
                    context_sha256,
                    progress,
                    context_path=context_path,
                )
                if args.command == "rerun-invalid-lane":
                    return rerun_invalid_stage(
                        root,
                        environment,
                        runner,
                        identity={
                            **stage_context,
                            "context_sha256": context_sha256,
                            "context_path": str(context_path),
                        },
                        root_identity=root_identity,
                        _locked=True,
                    )
                return continue_after_retry(
                    root,
                    environment,
                    runner,
                    identity={
                        **stage_context,
                        "context_sha256": context_sha256,
                        "context_path": str(context_path),
                    },
                    root_identity=root_identity,
                    _locked=True,
                )
        if args.command == "record-index":
            repository, root = _infer_repository_for_root(args.evidence_root)
            _require_fixed_index_argument(repository, args.index)
            if args.abandoned != args.confirm_no_live_processes:
                raise protocol.ProtocolError(
                    "--confirm-no-live-processes is required exactly with --abandoned"
                )
            _index, active = _active_for_root(
                repository,
                root,
                allowed_states=frozenset({"ACTIVE", "PENDING_PRESERVATION"}),
            )
            updated = record_index_root(
                repository=repository,
                root=root,
                reservation_id=active["reservation_id"],
                abandoned=args.abandoned,
            )
            print(index_state(updated))
            return 0
        if args.command == "record-preserved":
            repository, root = _infer_repository_for_root(args.evidence_root)
            _require_fixed_index_argument(repository, args.index)
            _index, active = _active_for_root(
                repository,
                root,
                allowed_states=frozenset({"PENDING_PRESERVATION", "PRESERVED"}),
            )
            updated = record_preserved(
                repository=repository,
                root=root,
                reservation_id=active["reservation_id"],
                preservation_commit=args.preservation_commit,
                issue_url=args.issue_comment_url,
                worklog=repository / WORKLOG_PATH,
            )
            print(index_state(updated))
            return 0
        raise protocol.ProtocolError(f"unsupported command: {args.command}")
    except (argparse.ArgumentError, protocol.ProtocolError) as error:
        print(f"phase2e orchestrator error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
