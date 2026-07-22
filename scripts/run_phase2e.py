#!/usr/bin/env python3
"""Own, validate, and preserve one atomic Phase 2E evidence campaign."""

from __future__ import annotations

import argparse
import copy
import fcntl
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import phase2e_protocol as protocol


AGGREGATE_MANIFEST = "phase2e-evidence.json"
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
            }:
                raise protocol.ProtocolError("TERMINAL event schema is invalid")
            if states.get(reservation) != "ACTIVE" or active_global != reservation:
                raise protocol.ProtocolError("TERMINAL event does not close ACTIVE")
            if event.get("status") not in TERMINAL_STATUSES:
                raise protocol.ProtocolError("TERMINAL status is invalid")
            _require_sha(event.get("root_digest"), sha256=True, context="root digest")
            states[reservation] = "PENDING_PRESERVATION"
            terminal_statuses[reservation] = event["status"]
            active_global = reservation
        elif kind == "PRESERVED":
            if set(event) != {
                "ordinal",
                "event",
                "reservation_id",
                "preservation_commit",
                "issue_url",
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
            states[reservation] = "PRESERVED"
            active_global = None
            if terminal_statuses[reservation] == "PASS":
                expected_current_root = records[reservation]["root"]
        else:
            raise protocol.ProtocolError("index event kind is invalid")
    if index["current_evidence_root"] != expected_current_root:
        raise protocol.ProtocolError("current evidence root is not the latest preserved PASS")


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
    return _append(
        index,
        {
            "event": "TERMINAL",
            "reservation_id": reservation_id,
            "status": status,
            "root_digest": root_digest,
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
    updated["events"].append(
        {
            "ordinal": len(updated["events"]) + 1,
            "event": "PRESERVED",
            "reservation_id": reservation_id,
            "preservation_commit": preservation_commit,
            "issue_url": issue_url,
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
    compare = subparsers.add_parser("compare-experiment-identity", exit_on_error=False)
    compare.add_argument("--repository", required=True, type=pathlib.Path)
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--contract", required=True, type=pathlib.Path)
    for name in (
        "start",
        "rerun-invalid-lane",
        "continue",
        "record-index",
        "record-preserved",
    ):
        subparsers.add_parser(name, exit_on_error=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "validate":
            status = validate_root(args.root)
            print(status)
            return _exit_for_status(status, require_pass=args.require_pass)
        if args.command == "compare-experiment-identity":
            contract = _read_json(args.contract, "experiment contract")
            left = git_experiment_identity(args.repository, args.left, contract)
            right = git_experiment_identity(args.repository, args.right, contract)
            print(left)
            return 0 if left == right else 1
        raise protocol.ProtocolError(f"{args.command} requires campaign arguments")
    except (argparse.ArgumentError, protocol.ProtocolError) as error:
        print(f"phase2e orchestrator error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
