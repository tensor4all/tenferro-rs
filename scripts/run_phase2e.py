#!/usr/bin/env python3
"""Own, validate, and preserve one atomic Phase 2E evidence campaign."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import phase2e_protocol as protocol


AGGREGATE_MANIFEST = "phase2e-evidence.json"
PROGRESS_MANIFEST = AGGREGATE_MANIFEST
ABANDONMENT_SEAL = "abandoned-inventory.json"
INDEX_PATH = pathlib.Path("docs/worklogs/phase2e-index.json")
INDEX_LOCK_PATH = pathlib.Path("docs/worklogs/.phase2e-index.lock")
PRESERVATION_BRANCH = "origin/codex/execution-engine-through-phase9"
ISSUE_NUMBER = 1436

STAGE_ORDER = (
    "timing-builds",
    "probe-builds",
    "allocation/direct-current-main",
    "allocation/common-lock-normalized",
    "dispatch-builds",
    "dispatch-gates",
    "characterization-builds",
    "characterization",
    "timing/direct-current-main",
    "timing/common-lock-normalized",
    "aggregate-validation",
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
                "root",
            ):
                if type(event.get(name)) is not str or not event[name]:
                    raise protocol.ProtocolError(f"ACTIVE event lacks {name}")
            _require_sha(event["candidate_sha"], sha256=False, context="candidate SHA")
            for name in (
                "candidate_tree_sha256",
                "experiment_identity_digest",
                "campaign_identity_digest",
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
                "candidate_sha",
                "root",
                "experiment_identity_digest",
            }:
                raise protocol.ProtocolError("TERMINAL event schema is invalid")
            if states.get(reservation) != "ACTIVE" or active_global != reservation:
                raise protocol.ProtocolError("TERMINAL event does not close ACTIVE")
            if event.get("status") not in TERMINAL_STATUSES:
                raise protocol.ProtocolError("TERMINAL status is invalid")
            _require_sha(event.get("root_digest"), sha256=True, context="root digest")
            if any(
                event[name] != records[reservation][name]
                for name in ("candidate_sha", "root", "experiment_identity_digest")
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
                "experiment_identity_digest",
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
            if type(event.get("issue_url")) is not str:
                raise protocol.ProtocolError("PRESERVED event lacks issue URL")
            if any(
                event[name] != records[reservation][name]
                for name in ("candidate_sha", "root", "experiment_identity_digest")
            ) or (
                event["status"] != terminal_statuses[reservation]
                or event["root_digest"] != terminal_digests[reservation]
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
        },
    )


def record_terminal(
    index: Mapping[str, Any],
    *,
    reservation_id: str,
    status: str,
    root_digest: str,
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
            "candidate_sha": active["candidate_sha"],
            "root": active["root"],
            "experiment_identity_digest": active["experiment_identity_digest"],
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
    if (
        index_state(index) != "PENDING_PRESERVATION"
        or index["events"][-1]["reservation_id"] != reservation_id
    ):
        raise protocol.ProtocolError("reservation is not pending preservation")
    if f"/issues/{ISSUE_NUMBER}#issuecomment-" not in issue_url:
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
            "experiment_identity_digest": active["experiment_identity_digest"],
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


@contextmanager
def exclusive_lock(path: pathlib.Path):
    """Hold one process-exclusive advisory lock on an exact regular path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_index(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise protocol.ProtocolError(f"cannot read Phase 2E index: {error}") from error
    decoded = protocol.decode_canonical_json_bytes(payload, "Phase 2E index")
    _validate_index(decoded)
    return decoded


def mutate_index(
    index_path: pathlib.Path,
    mutation: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    lock_path: pathlib.Path,
) -> dict[str, Any]:
    """Serialize one atomic read-modify-write operation on the campaign index."""
    with exclusive_lock(lock_path):
        current = _read_index(index_path)
        updated = mutation(copy.deepcopy(current))
        if type(updated) is not dict:
            raise protocol.ProtocolError("index mutation returned a non-object")
        protocol.atomic_write_json(index_path, updated)
        return updated


def run_fixed_stages(
    root: pathlib.Path,
    environment: Mapping[str, str],
    runner: Callable[[str, Mapping[str, str]], int],
    *,
    completed: Sequence[str] = (),
    _locked: bool = False,
) -> int:
    """Run remaining children in fixed order and durably hash every checkpoint."""
    root = pathlib.Path(root)
    if not _locked:
        with exclusive_lock(root / ".orchestrator.lock"):
            return run_fixed_stages(
                root, environment, runner, completed=completed, _locked=True
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
        code = runner(stage, dict(environment))
        if type(code) is not int or code not in {0, 2, 3, 4}:
            raise protocol.ProtocolError("child returned an unsupported exit code")
        children.append({"stage": stage, "exit_code": code})
        _write_progress(root, children)
        if code:
            return code
    return 0


def _write_progress(root: pathlib.Path, children: Sequence[Mapping[str, Any]]) -> None:
    protocol.atomic_write_json(
        root / PROGRESS_MANIFEST,
        {
            "version": 1,
            "stage_order": list(STAGE_ORDER),
            "children": list(children),
            "inventory": protocol.regular_file_inventory(
                root, excluded=frozenset({PROGRESS_MANIFEST})
            ),
        },
    )


def rerun_invalid_stage(
    root: pathlib.Path,
    environment: Mapping[str, str],
    runner: Callable[[str, Mapping[str, str]], int],
    *,
    _locked: bool = False,
) -> int:
    """Append one fresh whole-stage attempt after retryable validity failure."""
    root = pathlib.Path(root)
    if not _locked:
        with exclusive_lock(root / ".orchestrator.lock"):
            return rerun_invalid_stage(
                root, environment, runner, _locked=True
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
    code = runner(stage, dict(environment))
    if type(code) is not int or code not in {0, 2, 3, 4}:
        raise protocol.ProtocolError("child returned an unsupported exit code")
    children.append({"stage": stage, "exit_code": code})
    _write_progress(root, children)
    return code


def continue_after_retry(
    root: pathlib.Path,
    environment: Mapping[str, str],
    runner: Callable[[str, Mapping[str, str]], int],
    *,
    _locked: bool = False,
) -> int:
    """Continue only after a retained replacement attempt passed."""
    root = pathlib.Path(root)
    if not _locked:
        with exclusive_lock(root / ".orchestrator.lock"):
            return continue_after_retry(root, environment, runner, _locked=True)
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
    ):
        raise protocol.ProtocolError("replacement attempt has not passed")
    start = STAGE_ORDER.index(children[-1]["stage"]) + 1
    for stage in STAGE_ORDER[start:]:
        code = runner(stage, dict(environment))
        if type(code) is not int or code not in {0, 2, 3, 4}:
            raise protocol.ProtocolError("child returned an unsupported exit code")
        children.append({"stage": stage, "exit_code": code})
        _write_progress(root, children)
        if code:
            return code
    return 0


def initialize_campaign(
    *,
    index_path: pathlib.Path,
    index_lock: pathlib.Path,
    root: pathlib.Path,
    reservation_id: str,
    candidate_sha: str,
    candidate_tree_sha256: str,
    experiment_identity_digest: str,
    campaign_identity_digest: str,
    initializer: Callable[[pathlib.Path], None] | None = None,
) -> int:
    """Reserve globally, initialize locally, or self-seal atomically on failure."""
    root = pathlib.Path(root)
    with exclusive_lock(index_lock):
        current = _read_index(index_path)
        updated = record_active(
            current,
            reservation_id=reservation_id,
            candidate_sha=candidate_sha,
            candidate_tree_sha256=candidate_tree_sha256,
            root=str(root),
            experiment_identity_digest=experiment_identity_digest,
            campaign_identity_digest=campaign_identity_digest,
        )
        protocol.prepare_empty_root(root)
        root_lock = root / ".orchestrator.lock"
        with exclusive_lock(root_lock):
            protocol.atomic_write_json(index_path, updated)
            try:
                if initializer is None:
                    protocol.atomic_write_json(
                        root / "evidence-ledger.json",
                        protocol.new_ledger(candidate_sha),
                    )
                else:
                    initializer(root)
            except BaseException:
                seal = seal_abandoned_root(root)
                terminal = record_terminal(
                    updated,
                    reservation_id=reservation_id,
                    status="ABANDONED",
                    root_digest=protocol.sha256_json(seal),
                )
                protocol.atomic_write_json(index_path, terminal)
                return 5
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
    root = pathlib.Path(root)
    ledger = _read_json(root / "evidence-ledger.json", "evidence ledger")
    status = _terminal_ledger_status(ledger)
    for relative in ("dispatch-gates/manifest.json", "characterization/manifest.json"):
        child = _read_json(root / relative, relative)
        if (
            child.get("candidate") != candidate_sha
            or child.get("gating_result") != "PASS"
        ):
            raise protocol.ProtocolError(f"{relative} does not pass every gate")
    inventory = protocol.regular_file_inventory(
        root, excluded=frozenset({AGGREGATE_MANIFEST})
    )
    manifest = {
        "version": 1,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "candidate_sha": candidate_sha,
        "reservation_id": reservation_id,
        "experiment_identity_digest": experiment_identity_digest,
        "status": status,
        "stage_order": list(STAGE_ORDER),
        "ledger_sha256": inventory["evidence-ledger.json"],
        "inventory": inventory,
    }
    protocol.atomic_write_json(root / AGGREGATE_MANIFEST, manifest)
    return manifest


def validate_root(root: pathlib.Path) -> str:
    """Cryptographically reconstruct one complete aggregate evidence root."""
    root = pathlib.Path(root)
    abandonment_path = root / ABANDONMENT_SEAL
    if abandonment_path.exists():
        seal = _read_json(abandonment_path, "abandonment seal")
        if set(seal) != {"version", "protocol_version", "status", "inventory"} or (
            seal["version"] != 1
            or seal["protocol_version"] != protocol.PROTOCOL_VERSION
            or seal["status"] != "ABANDONED"
        ):
            raise protocol.ProtocolError("abandonment seal schema is invalid")
        protocol.validate_regular_file_inventory(
            root, seal["inventory"], excluded=frozenset({ABANDONMENT_SEAL})
        )
        return "ABANDONED"
    manifest = _read_json(root / AGGREGATE_MANIFEST, "aggregate manifest")
    required = {
        "version",
        "protocol_version",
        "candidate_sha",
        "reservation_id",
        "experiment_identity_digest",
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
    status = _terminal_ledger_status(ledger)
    if status != manifest["status"]:
        raise protocol.ProtocolError("aggregate status differs from ledger")
    if manifest["ledger_sha256"] != protocol.sha256_file(root / "evidence-ledger.json"):
        raise protocol.ProtocolError("aggregate ledger digest differs")
    for relative in ("dispatch-gates/manifest.json", "characterization/manifest.json"):
        child = _read_json(root / relative, relative)
        if (
            child.get("candidate") != manifest["candidate_sha"]
            or child.get("gating_result") != "PASS"
        ):
            raise protocol.ProtocolError(f"{relative} does not pass every gate")
    return status


def seal_abandoned_root(root: pathlib.Path) -> dict[str, Any]:
    """Own every preexisting regular byte after an unresumable interruption."""
    root = pathlib.Path(root)
    inventory = protocol.regular_file_inventory(
        root, excluded=frozenset({ABANDONMENT_SEAL})
    )
    seal = {
        "version": 1,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "status": "ABANDONED",
        "inventory": inventory,
    }
    protocol.atomic_write_json(root / ABANDONMENT_SEAL, seal)
    return seal


def validate_git_blob_inventory(
    root: pathlib.Path, staged_blobs: Mapping[str, bytes]
) -> None:
    """Require Git's staged/reconstructed bytes to equal every root-owned byte."""
    if type(staged_blobs) is not dict or any(
        type(path) is not str or type(payload) is not bytes
        for path, payload in staged_blobs.items()
    ):
        raise protocol.ProtocolError("Git blob inventory schema is invalid")
    inventory = protocol.regular_file_inventory(root)
    if set(staged_blobs) != set(inventory):
        raise protocol.ProtocolError("Git blob inventory has missing or extra paths")
    for relative, digest in inventory.items():
        if hashlib.sha256(staged_blobs[relative]).hexdigest() != digest:
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
    """Bind one permanent #1436 report to the exact preserved campaign."""
    if (
        type(issue_url) is not str
        or f"/issues/{ISSUE_NUMBER}#issuecomment-" not in issue_url
        or type(body) is not str
    ):
        raise protocol.ProtocolError(
            "preservation comment is not permanent issue #1436"
        )
    for value in (preservation_commit, root, candidate_sha, status):
        if type(value) is not str or value not in body:
            raise protocol.ProtocolError("preservation comment omits campaign identity")


def git_blob_inventory(
    repository: pathlib.Path, selector: str, root_relative: str
) -> dict[str, bytes]:
    """Read one exact root inventory from the index (``:``) or a commit."""
    prefix = root_relative.rstrip("/") + "/"
    command = ["git", "ls-files", "-z", "--stage", "--", root_relative]
    if selector != ":":
        command = [
            "git",
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            selector,
            "--",
            root_relative,
        ]
    listing = subprocess.run(command, cwd=repository, capture_output=True, check=False)
    if listing.returncode:
        raise protocol.ProtocolError("cannot enumerate Git evidence blobs")
    paths = []
    for record in listing.stdout.split(b"\0"):
        if not record:
            continue
        if selector == ":":
            try:
                _metadata, raw = record.split(b"\t", 1)
            except ValueError as error:
                raise protocol.ProtocolError(
                    "staged Git inventory is malformed"
                ) from error
        else:
            raw = record
        try:
            path = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise protocol.ProtocolError("Git evidence path is not UTF-8") from error
        if not path.startswith(prefix):
            raise protocol.ProtocolError("Git evidence path escapes its root")
        paths.append(path)
    result = {}
    for path in paths:
        spec = f":{path}" if selector == ":" else f"{selector}:{path}"
        blob = subprocess.run(
            ("git", "show", spec), cwd=repository, capture_output=True, check=False
        )
        if blob.returncode:
            raise protocol.ProtocolError(f"cannot read Git evidence blob: {path}")
        result[path.removeprefix(prefix)] = blob.stdout
    return result


def validate_git_selector(
    repository: pathlib.Path,
    root: pathlib.Path,
    *,
    selector: str,
) -> None:
    """Validate staged or committed evidence, including ignored files."""
    repository = pathlib.Path(repository).resolve(strict=True)
    root = pathlib.Path(root).resolve(strict=True)
    try:
        relative = root.relative_to(repository).as_posix()
    except ValueError as error:
        raise protocol.ProtocolError("evidence root is outside repository") from error
    validate_git_blob_inventory(
        root, git_blob_inventory(repository, selector, relative)
    )


def validate_progress(root: pathlib.Path) -> dict[str, Any]:
    """Re-hash the complete root before any retry/continuation mutation."""
    progress = _read_json(root / PROGRESS_MANIFEST, "Phase 2E progress")
    if (
        set(progress) != {"version", "stage_order", "children", "inventory"}
        or progress["version"] != 1
        or progress["stage_order"] != list(STAGE_ORDER)
        or type(progress["children"]) is not list
    ):
        raise protocol.ProtocolError("Phase 2E progress schema is invalid")
    protocol.validate_regular_file_inventory(
        root, progress["inventory"], excluded=frozenset({PROGRESS_MANIFEST})
    )
    return progress


def _load_command_plan(path: pathlib.Path) -> dict[str, tuple[str, ...]]:
    plan = _read_json(path, "stage command plan")
    if set(plan) != set(STAGE_ORDER):
        raise protocol.ProtocolError("stage command plan inventory differs")
    result = {}
    for stage in STAGE_ORDER:
        argv = plan[stage]
        if (
            type(argv) is not list
            or not argv
            or any(type(argument) is not str or not argument for argument in argv)
            or not pathlib.Path(argv[0]).is_absolute()
        ):
            raise protocol.ProtocolError(f"stage command is invalid: {stage}")
        result[stage] = tuple(argv)
    return result


def _subprocess_stage_runner(
    plan: Mapping[str, tuple[str, ...]], repository: pathlib.Path
) -> Callable[[str, Mapping[str, str]], int]:
    def run(stage: str, environment: Mapping[str, str]) -> int:
        result = subprocess.run(
            plan[stage],
            cwd=repository,
            env=dict(environment),
            check=False,
            start_new_session=True,
        )
        return result.returncode

    return run


def record_index_root(
    *,
    index_path: pathlib.Path,
    index_lock: pathlib.Path,
    root: pathlib.Path,
    reservation_id: str,
    abandoned: bool = False,
    confirm_no_live_processes: bool = False,
    process_groups: Sequence[int] = (),
) -> dict[str, Any]:
    """Validate and transition one ACTIVE reservation to pending preservation."""
    with exclusive_lock(index_lock):
        with exclusive_lock(root / ".orchestrator.lock"):
            index = _read_index(index_path)
            if abandoned:
                if not confirm_no_live_processes:
                    raise protocol.ProtocolError(
                        "abandonment requires no-live-process confirmation"
                    )
                for process_group in process_groups:
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
                seal = seal_abandoned_root(root)
                status = "ABANDONED"
                digest = protocol.sha256_json(seal)
            else:
                status = validate_root(root)
                digest = protocol.sha256_file(root / AGGREGATE_MANIFEST)
            updated = record_terminal(
                index,
                reservation_id=reservation_id,
                status=status,
                root_digest=digest,
            )
            protocol.atomic_write_json(index_path, updated)
            return updated


def fetch_comment(url: str) -> str:
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
    body = decoded.get("body") if type(decoded) is dict else None
    if type(body) is not str:
        raise protocol.ProtocolError("preservation comment response lacks body")
    return body


def _git(repository: pathlib.Path, *argv: str, text: bool = False):
    result = subprocess.run(
        ("git", *argv), cwd=repository, capture_output=True, check=False, text=text
    )
    if result.returncode:
        raise protocol.ProtocolError(f"Git command failed: {' '.join(argv)}")
    return result.stdout


def require_remote_index(
    repository: pathlib.Path, index_path: pathlib.Path, *, allow_absent: bool
) -> None:
    """Require the local durable index to equal the fetched branch blob."""
    _git(repository, "fetch", "origin", "codex/execution-engine-through-phase9")
    relative = index_path.resolve().relative_to(repository.resolve()).as_posix()
    result = subprocess.run(
        ("git", "show", f"{PRESERVATION_BRANCH}:{relative}"),
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        if allow_absent and not index_path.exists():
            return
        raise protocol.ProtocolError("remote branch lacks the durable Phase 2E index")
    if not index_path.exists() or result.stdout != index_path.read_bytes():
        raise protocol.ProtocolError("local Phase 2E index is not pushed byte-for-byte")


def record_preserved(
    *,
    repository: pathlib.Path,
    index_path: pathlib.Path,
    index_lock: pathlib.Path,
    root: pathlib.Path,
    reservation_id: str,
    preservation_commit: str,
    issue_url: str,
    comment_fetcher: Callable[[str], str] = fetch_comment,
) -> dict[str, Any]:
    """Verify remote Git/root/index/comment preservation, then append PRESERVED."""
    with exclusive_lock(index_lock):
        index = _read_index(index_path)
        if index_state(index) != "PENDING_PRESERVATION":
            raise protocol.ProtocolError("no campaign is pending preservation")
        _git(repository, "fetch", "origin", "codex/execution-engine-through-phase9")
        _git(
            repository,
            "merge-base",
            "--is-ancestor",
            preservation_commit,
            PRESERVATION_BRANCH,
        )
        relative_index = (
            index_path.resolve().relative_to(repository.resolve()).as_posix()
        )
        committed_index = _git(
            repository, "show", f"{preservation_commit}:{relative_index}"
        )
        if committed_index != index_path.read_bytes():
            raise protocol.ProtocolError(
                "preservation commit has the wrong pending index"
            )
        validate_git_selector(repository, root, selector=preservation_commit)
        active = next(
            event
            for event in reversed(index["events"])
            if event["reservation_id"] == reservation_id and event["event"] == "ACTIVE"
        )
        terminal = index["events"][-1]
        validate_preservation_comment(
            issue_url,
            comment_fetcher(issue_url),
            preservation_commit=preservation_commit,
            root=active["root"],
            candidate_sha=active["candidate_sha"],
            status=terminal["status"],
        )
        updated = record_preserved_event(
            index,
            reservation_id=reservation_id,
            preservation_commit=preservation_commit,
            issue_url=issue_url,
        )
        protocol.atomic_write_json(index_path, updated)
        return updated


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
    validate.add_argument("--root", required=True, type=pathlib.Path)
    validate.add_argument("--require-pass", action="store_true")
    validate.add_argument("--git-index", action="store_true")
    validate.add_argument("--repository", type=pathlib.Path)
    compare = subparsers.add_parser("compare-experiment-identity", exit_on_error=False)
    compare.add_argument("--repository", required=True, type=pathlib.Path)
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--contract", required=True, type=pathlib.Path)
    for name in ("start", "rerun-invalid-lane", "continue"):
        command = subparsers.add_parser(name, exit_on_error=False)
        command.add_argument("--repository", required=True, type=pathlib.Path)
        command.add_argument("--root", required=True, type=pathlib.Path)
        command.add_argument("--index", required=True, type=pathlib.Path)
        command.add_argument("--index-lock", required=True, type=pathlib.Path)
        command.add_argument("--commands", required=True, type=pathlib.Path)
        command.add_argument("--path", required=True)
        command.add_argument("--home", required=True)
        if name == "start":
            command.add_argument("--reservation-id", required=True)
            command.add_argument("--candidate", required=True)
            command.add_argument("--candidate-tree-sha256", required=True)
            command.add_argument("--experiment-identity-digest", required=True)
            command.add_argument("--campaign-identity-digest", required=True)
            command.add_argument("--contract", required=True, type=pathlib.Path)
    record = subparsers.add_parser("record-index", exit_on_error=False)
    record.add_argument("--index", required=True, type=pathlib.Path)
    record.add_argument("--index-lock", required=True, type=pathlib.Path)
    record.add_argument("--root", required=True, type=pathlib.Path)
    record.add_argument("--reservation-id", required=True)
    record.add_argument("--abandoned", action="store_true")
    record.add_argument("--confirm-no-live-processes", action="store_true")
    record.add_argument("--process-group", action="append", type=int, default=[])
    preserved = subparsers.add_parser("record-preserved", exit_on_error=False)
    preserved.add_argument("--repository", required=True, type=pathlib.Path)
    preserved.add_argument("--index", required=True, type=pathlib.Path)
    preserved.add_argument("--index-lock", required=True, type=pathlib.Path)
    preserved.add_argument("--root", required=True, type=pathlib.Path)
    preserved.add_argument("--reservation-id", required=True)
    preserved.add_argument("--preservation-commit", required=True)
    preserved.add_argument("--issue-url", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "validate":
            status = validate_root(args.root)
            if args.git_index:
                if args.repository is None:
                    raise protocol.ProtocolError("--git-index requires --repository")
                validate_git_selector(args.repository, args.root, selector=":")
            print(status)
            return _exit_for_status(status, require_pass=args.require_pass)
        if args.command == "compare-experiment-identity":
            contract = _read_json(args.contract, "experiment contract")
            left = git_experiment_identity(args.repository, args.left, contract)
            right = git_experiment_identity(args.repository, args.right, contract)
            print(left)
            return 0 if left == right else 1
        if args.command == "start":
            contract = _read_json(args.contract, "experiment contract")
            validate_candidate_provenance(
                args.repository,
                args.candidate,
                args.candidate_tree_sha256,
                args.experiment_identity_digest,
                contract,
            )
            require_remote_index(args.repository, args.index, allow_absent=True)
            if not args.index.exists():
                args.index.parent.mkdir(parents=True, exist_ok=True)
                protocol.atomic_write_json(args.index, new_campaign_index())
            code = initialize_campaign(
                index_path=args.index,
                index_lock=args.index_lock,
                root=args.root,
                reservation_id=args.reservation_id,
                candidate_sha=args.candidate,
                candidate_tree_sha256=args.candidate_tree_sha256,
                experiment_identity_digest=args.experiment_identity_digest,
                campaign_identity_digest=args.campaign_identity_digest,
            )
            if code == 5:
                print("ABANDONED_INITIALIZATION")
                return 5
            plan = _load_command_plan(args.commands)
            environment = protocol.runtime_environment(path=args.path, home=args.home)
            return run_fixed_stages(
                args.root,
                environment,
                _subprocess_stage_runner(plan, args.repository),
            )
        if args.command in {"rerun-invalid-lane", "continue"}:
            plan = _load_command_plan(args.commands)
            environment = protocol.runtime_environment(path=args.path, home=args.home)
            runner = _subprocess_stage_runner(plan, args.repository)
            with exclusive_lock(args.index_lock):
                index = _read_index(args.index)
                if index_state(index) != "ACTIVE":
                    raise protocol.ProtocolError("campaign reservation is finalized")
                active = index["events"][-1]
                if pathlib.Path(active["root"]) != args.root:
                    raise protocol.ProtocolError("active reservation root differs")
                with exclusive_lock(args.root / ".orchestrator.lock"):
                    if args.command == "rerun-invalid-lane":
                        return rerun_invalid_stage(
                            args.root, environment, runner, _locked=True
                        )
                    return continue_after_retry(
                        args.root, environment, runner, _locked=True
                    )
        if args.command == "record-index":
            updated = record_index_root(
                index_path=args.index,
                index_lock=args.index_lock,
                root=args.root,
                reservation_id=args.reservation_id,
                abandoned=args.abandoned,
                confirm_no_live_processes=args.confirm_no_live_processes,
                process_groups=args.process_group,
            )
            print(index_state(updated))
            return 0
        if args.command == "record-preserved":
            updated = record_preserved(
                repository=args.repository,
                index_path=args.index,
                index_lock=args.index_lock,
                root=args.root,
                reservation_id=args.reservation_id,
                preservation_commit=args.preservation_commit,
                issue_url=args.issue_url,
            )
            print(index_state(updated))
            return 0
        raise protocol.ProtocolError(f"unsupported command: {args.command}")
    except (argparse.ArgumentError, protocol.ProtocolError) as error:
        print(f"phase2e orchestrator error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
