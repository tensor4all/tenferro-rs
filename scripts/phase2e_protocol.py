#!/usr/bin/env python3
"""Protocol-v2 primitives for the Phase 2E evidence campaign.

Atomic replacement in this module deliberately targets Linux/POSIX filesystems:
every write uses a unique sibling created with ``O_EXCL``, fsyncs the file,
replaces the destination, and then fsyncs the containing directory.  A
pre-commit failure retains any created temporary as evidence.  A directory
fsync failure after replacement reports that the complete target was installed
but its crash durability is unknown; replacement is never rolled back.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import stat
import tempfile
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class ProtocolError(Exception):
    """Report a protocol contract or evidence I/O failure."""


class AtomicWriteError(ProtocolError):
    """Report an atomic-write failure and whether replacement committed."""

    def __init__(self, message: str, *, committed: bool) -> None:
        super().__init__(message)
        self.committed = committed


class AtomicWriteDurabilityError(AtomicWriteError):
    """Report that replacement committed but directory durability is unknown."""

    def __init__(self, message: str) -> None:
        super().__init__(message, committed=True)


PROTOCOL_VERSION = 2
PAIR_ORDERS = ("A/B", "B/A", "A/B")
RUN_ROLES = ("sentinel_before", "first_target", "second_target", "sentinel_after")
STAGE_NAMES = ("allocation", "timing")
LANE_NAMES = ("direct-current-main", "common-lock-normalized")

_THREAD_ENV = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
THREAD_ENV = MappingProxyType(_THREAD_ENV)


def _canonical_cases() -> dict[str, str]:
    cases: dict[str, str] = {}
    operations = (
        ("neg", "neg_f64"),
        ("add", "add_f64"),
        ("reduce", "reduce_sum_f64"),
        ("slice", "slice_f64"),
    )
    for mode in ("lazy", "materialized"):
        for tag, benchmark in operations:
            for size in (1, 8, 64):
                cases[f"{mode}_{tag}_{size}"] = (
                    f"eager_dispatch_baseline/{mode}/{benchmark}/{size}"
                )
        for size in (1, 2):
            cases[f"{mode}_dot_{size}"] = (
                f"eager_dispatch_baseline/{mode}/dot_general_f64/{size}"
            )
    return cases


CANONICAL_CASES = MappingProxyType(_canonical_cases())


def prepare_empty_root(root: pathlib.Path) -> pathlib.Path:
    """Create or accept an empty regular directory and reject all other roots."""
    root = pathlib.Path(root)
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        try:
            root.mkdir(parents=True)
        except OSError as error:
            raise ProtocolError(f"cannot create evidence root {root}: {error}") from error
        return root
    except OSError as error:
        raise ProtocolError(f"cannot inspect evidence root {root}: {error}") from error

    if not stat.S_ISDIR(metadata.st_mode):
        raise ProtocolError(f"evidence root is not a regular directory: {root}")
    try:
        first_entry = next(root.iterdir(), None)
    except OSError as error:
        raise ProtocolError(f"cannot inspect evidence root {root}: {error}") from error
    if first_entry is not None:
        raise ProtocolError(f"evidence root is not empty: {root}")
    return root


def runtime_environment(
    *, path: str, home: str, criterion_home: str | None = None
) -> dict[str, str]:
    """Construct the runtime allowlist without inheriting ambient variables."""
    if not isinstance(path, str) or not isinstance(home, str):
        raise ProtocolError("PATH and HOME must be strings")
    if criterion_home is not None and not isinstance(criterion_home, str):
        raise ProtocolError("CRITERION_HOME must be a string when supplied")
    result = {
        "PATH": path,
        "HOME": home,
        "LC_ALL": "C",
        "TZ": "UTC",
        **THREAD_ENV,
    }
    if criterion_home is not None:
        result["CRITERION_HOME"] = criterion_home
    return result


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
        raise ProtocolError(f"payload is not canonical JSON: {error}") from error
    return (rendered + "\n").encode("utf-8")


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    """Replace *path* with canonical JSON under the documented commit boundary."""
    path = pathlib.Path(path)
    encoded = _canonical_json_bytes(payload)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    try:
        directory_descriptor = os.open(path.parent, directory_flags)
    except OSError as error:
        raise AtomicWriteError(
            f"cannot open parent directory for {path}: {error}", committed=False
        ) from error

    failure: BaseException | None = None
    cause: BaseException | None = None
    committed = False
    try:
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.write-", suffix=".tmp", dir=path.parent
            )
        except OSError as error:
            failure = AtomicWriteError(
                f"cannot create temporary for {path}: {error}", committed=False
            )
            cause = error
        else:
            temporary = pathlib.Path(temporary_name)
            stream = None
            write_failure: BaseException | None = None
            try:
                stream = os.fdopen(descriptor, "wb")
                written = stream.write(encoded)
                if written != len(encoded):
                    raise OSError(
                        f"short JSON write: wrote {written} of {len(encoded)} bytes"
                    )
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException as error:
                write_failure = error

            if stream is None:
                close_failure = _close_descriptor(descriptor)
            else:
                try:
                    stream.close()
                except OSError as error:
                    close_failure = error
                else:
                    close_failure = None
            if write_failure is None:
                write_failure = close_failure

            if write_failure is not None:
                if isinstance(write_failure, OSError):
                    failure = AtomicWriteError(
                        f"cannot write and fsync temporary for {path}: {write_failure}",
                        committed=False,
                    )
                    cause = write_failure
                else:
                    failure = write_failure

            if failure is None:
                try:
                    os.replace(temporary, path)
                except OSError as error:
                    failure = AtomicWriteError(
                        f"cannot replace {path}: {error}", committed=False
                    )
                    cause = error
                else:
                    committed = True

            if failure is None:
                try:
                    os.fsync(directory_descriptor)
                except OSError as error:
                    failure = AtomicWriteDurabilityError(
                        f"replacement committed for {path}, but parent fsync failed: {error}"
                    )
                    cause = error
    finally:
        close_failure = _close_descriptor(directory_descriptor)
        if failure is None and close_failure is not None:
            failure = AtomicWriteError(
                f"cannot close parent directory for {path}: {close_failure}",
                committed=committed,
            )
            cause = close_failure

    if failure is not None:
        if cause is not None:
            raise failure from cause
        raise failure


def _close_descriptor(descriptor: int) -> OSError | None:
    try:
        os.close(descriptor)
    except OSError as error:
        return error
    return None


def sha256_file(path: pathlib.Path) -> str:
    """Hash a Linux/POSIX regular file without following a final symlink."""
    path = pathlib.Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProtocolError(f"cannot open regular evidence file {path}: {error}") from error

    digest = hashlib.sha256()
    primary_error: BaseException | None = None
    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise ProtocolError(f"cannot hash evidence file {path}: {error}") from error

        if not stat.S_ISREG(metadata.st_mode):
            raise ProtocolError(f"evidence path is not a regular file: {path}")

        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except OSError as error:
                raise ProtocolError(
                    f"cannot hash evidence file {path}: {error}"
                ) from error
            if not chunk:
                break
            digest.update(chunk)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_failure = _close_descriptor(descriptor)
        if close_failure is not None and primary_error is None:
            raise ProtocolError(
                f"cannot close evidence file {path}: {close_failure}"
            ) from close_failure

    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON representation."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def validate_manifest_fields(
    manifest: Mapping[str, Any],
    required_fields: Mapping[str, type | tuple[type, ...]],
    *,
    context: str = "manifest",
) -> None:
    """Validate an exact manifest field set with exact (not subclass) types."""
    if not isinstance(manifest, Mapping):
        raise ProtocolError(f"{context} must be a mapping")
    if not isinstance(required_fields, Mapping):
        raise ProtocolError("manifest schema must be a mapping")

    expected_names = set(required_fields)
    actual_names = set(manifest)
    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields {sorted(missing, key=repr)!r}")
        if extra:
            details.append(f"extra fields {sorted(extra, key=repr)!r}")
        raise ProtocolError(f"{context} has " + " and ".join(details))

    for name, expected in required_fields.items():
        accepted = expected if isinstance(expected, tuple) else (expected,)
        if not accepted or any(not isinstance(item, type) for item in accepted):
            raise ProtocolError(f"invalid schema type for field {name!r}")
        if type(manifest[name]) not in accepted:
            expected_names_text = ", ".join(item.__name__ for item in accepted)
            raise ProtocolError(
                f"{context} field {name!r} must have exact type "
                f"{expected_names_text}; got {type(manifest[name]).__name__}"
            )


def new_ledger(candidate_sha: str) -> dict[str, Any]:
    """Create an empty candidate-scoped allocation/timing attempt ledger."""
    if (
        not isinstance(candidate_sha, str)
        or len(candidate_sha) != 40
        or any(character not in "0123456789abcdef" for character in candidate_sha)
    ):
        raise ProtocolError("candidate SHA must be exactly 40 lowercase hexadecimal digits")

    stages = []
    for stage_name in STAGE_NAMES:
        stages.append(
            {
                "name": stage_name,
                "lanes": [
                    {
                        "name": LANE_NAMES[0],
                        "state": "READY",
                        "result": None,
                        "attempt_ids": [],
                    },
                    {
                        "name": LANE_NAMES[1],
                        "state": "BLOCKED",
                        "result": None,
                        "attempt_ids": [],
                    },
                ],
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_sha": candidate_sha,
        "next_transition_ordinal": 1,
        "active_attempt_id": None,
        "stages": stages,
        "attempts": [],
    }


def _stage_and_lane(
    ledger: dict[str, Any], stage_name: str, lane_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    for stage_record in ledger["stages"]:
        if stage_record["name"] == stage_name:
            for lane_record in stage_record["lanes"]:
                if lane_record["name"] == lane_name:
                    return stage_record, lane_record
            break
    raise ProtocolError(f"unknown ledger stage/lane: {stage_name}/{lane_name}")


def _validate_ledger(ledger: dict[str, Any]) -> None:
    validate_manifest_fields(
        ledger,
        {
            "protocol_version": int,
            "candidate_sha": str,
            "next_transition_ordinal": int,
            "active_attempt_id": (int, type(None)),
            "stages": list,
            "attempts": list,
        },
        context="evidence ledger",
    )
    if ledger["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError("evidence ledger protocol version mismatch")
    candidate_sha = ledger["candidate_sha"]
    if (
        len(candidate_sha) != 40
        or any(character not in "0123456789abcdef" for character in candidate_sha)
    ):
        raise ProtocolError("evidence ledger candidate SHA is invalid")
    if ledger["next_transition_ordinal"] <= 0:
        raise ProtocolError("next transition ordinal must be positive")

    if len(ledger["stages"]) != len(STAGE_NAMES):
        raise ProtocolError("evidence ledger stage inventory mismatch")
    registered_attempts: list[tuple[str, str, int]] = []
    for expected_stage_name, stage_record in zip(STAGE_NAMES, ledger["stages"]):
        validate_manifest_fields(
            stage_record,
            {"name": str, "lanes": list},
            context="ledger stage",
        )
        if stage_record["name"] != expected_stage_name:
            raise ProtocolError("evidence ledger stage order mismatch")
        if len(stage_record["lanes"]) != len(LANE_NAMES):
            raise ProtocolError("evidence ledger lane inventory mismatch")
        for expected_lane_name, lane_record in zip(LANE_NAMES, stage_record["lanes"]):
            validate_manifest_fields(
                lane_record,
                {"name": str, "state": str, "result": (str, type(None)), "attempt_ids": list},
                context="ledger lane",
            )
            if lane_record["name"] != expected_lane_name:
                raise ProtocolError("evidence ledger lane order mismatch")
            if lane_record["state"] not in {
                "BLOCKED",
                "READY",
                "RUNNING",
                "RETRYABLE",
                "COMPLETE",
            }:
                raise ProtocolError("evidence ledger lane state is invalid")
            if lane_record["result"] not in {None, "PASS", "FAIL", "INCONCLUSIVE"}:
                raise ProtocolError("evidence ledger lane result is invalid")
            if lane_record["state"] == "COMPLETE":
                if lane_record["result"] is None:
                    raise ProtocolError("complete ledger lane is missing its result")
            elif lane_record["result"] is not None:
                raise ProtocolError("non-complete ledger lane has a terminal result")
            for attempt_id in lane_record["attempt_ids"]:
                if type(attempt_id) is not int or attempt_id <= 0:
                    raise ProtocolError("ledger lane attempt ids must be positive integers")
            if any(
                attempt_id != expected_attempt_id
                for expected_attempt_id, attempt_id in enumerate(
                    lane_record["attempt_ids"], start=1
                )
            ):
                raise ProtocolError("ledger lane attempt ids are not sequential")
            registered_attempts.extend(
                (expected_stage_name, expected_lane_name, attempt_id)
                for attempt_id in lane_record["attempt_ids"]
            )

    attempts = ledger["attempts"]
    attempts_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    transition_events: list[tuple[int, str, tuple[str, str, int]]] = []
    running_attempts: list[dict[str, Any]] = []
    open_ordinals: list[int] = []
    for attempt in attempts:
        validate_manifest_fields(
            attempt,
            {
                "attempt_id": int,
                "stage": str,
                "lane": str,
                "open_ordinal": int,
                "close_ordinal": (int, type(None)),
                "state": str,
                "validity_state": str,
                "statistical_result": (str, type(None)),
            },
            context="ledger attempt",
        )
        attempt_id = attempt["attempt_id"]
        if attempt_id <= 0:
            raise ProtocolError("attempt id must be positive")
        key = (attempt["stage"], attempt["lane"], attempt_id)
        if key in attempts_by_key:
            raise ProtocolError("attempt is registered more than once")
        attempts_by_key[key] = attempt

        open_ordinal = attempt["open_ordinal"]
        close_ordinal = attempt["close_ordinal"]
        if open_ordinal <= 0:
            raise ProtocolError("attempt open ordinal must be positive")
        open_ordinals.append(open_ordinal)
        transition_events.append((open_ordinal, "OPEN", key))
        state = attempt["state"]
        validity_state = attempt["validity_state"]
        statistical_result = attempt["statistical_result"]
        if state == "RUNNING":
            if (
                validity_state != "RUNNING"
                or statistical_result is not None
                or close_ordinal is not None
            ):
                raise ProtocolError("running attempt has terminal fields")
            running_attempts.append(attempt)
        elif state == "INCONCLUSIVE":
            if (
                validity_state != "INCONCLUSIVE"
                or statistical_result is not None
                or close_ordinal is None
            ):
                raise ProtocolError("validity-inconclusive attempt is malformed")
        elif state == "COMPLETE":
            if (
                validity_state != "COMPLETE"
                or statistical_result
                not in {"PASS", "FAIL", "INCONCLUSIVE"}
                or close_ordinal is None
            ):
                raise ProtocolError("complete attempt is malformed")
        else:
            raise ProtocolError("attempt state is invalid")
        if close_ordinal is not None:
            if close_ordinal <= 0:
                raise ProtocolError("attempt close ordinal must be positive")
            if open_ordinal >= close_ordinal:
                raise ProtocolError("attempt close must follow its open transition")
            transition_events.append((close_ordinal, "CLOSE", key))

    if set(attempts_by_key) != set(registered_attempts):
        raise ProtocolError("attempt records do not match lane registrations")
    if open_ordinals != sorted(open_ordinals):
        raise ProtocolError("attempt records are not in append-only open order")

    if ledger["next_transition_ordinal"] != len(transition_events) + 1:
        raise ProtocolError(
            "next transition ordinal does not match the stored event count"
        )
    ordered_events = sorted(transition_events)
    if any(
        ordinal != expected_ordinal
        for expected_ordinal, (ordinal, _, _) in enumerate(ordered_events, start=1)
    ):
        raise ProtocolError("transition ordinals must be positive, unique, and gap-free")

    active_key: tuple[str, str, int] | None = None
    for _, event_kind, key in ordered_events:
        if event_kind == "OPEN":
            if active_key is not None:
                raise ProtocolError("attempt opened before the active attempt closed")
            active_key = key
        else:
            if active_key != key:
                raise ProtocolError("attempt close does not match the active open")
            active_key = None

    for stage_record in ledger["stages"]:
        direct_lane, normalized_lane = stage_record["lanes"]
        direct_state, direct_result = _summary_from_attempt_history(
            attempts_by_key, stage_record["name"], direct_lane
        )
        _validate_lane_summary(direct_lane, direct_state, direct_result)

        if direct_state == "COMPLETE" and direct_result == "PASS":
            normalized_state, normalized_result = _summary_from_attempt_history(
                attempts_by_key, stage_record["name"], normalized_lane
            )
            if normalized_lane["attempt_ids"]:
                direct_attempt = attempts_by_key[
                    (
                        stage_record["name"],
                        direct_lane["name"],
                        direct_lane["attempt_ids"][-1],
                    )
                ]
                first_normalized = attempts_by_key[
                    (stage_record["name"], normalized_lane["name"], 1)
                ]
                if (
                    first_normalized["open_ordinal"]
                    <= direct_attempt["close_ordinal"]
                ):
                    raise ProtocolError(
                        "normalized lane opened before the direct PASS close"
                    )
            else:
                normalized_state = "READY"
        else:
            if normalized_lane["attempt_ids"]:
                raise ProtocolError("normalized lane history requires direct PASS")
            normalized_state, normalized_result = "BLOCKED", None
        _validate_lane_summary(
            normalized_lane, normalized_state, normalized_result
        )

    active_attempt_id = ledger["active_attempt_id"]
    if running_attempts:
        if (
            len(running_attempts) != 1
            or active_attempt_id != running_attempts[0]["attempt_id"]
        ):
            raise ProtocolError("ledger active attempt registration mismatch")
        active_attempt = running_attempts[0]
        _, active_lane = _stage_and_lane(
            ledger, active_attempt["stage"], active_attempt["lane"]
        )
        if active_lane["state"] != "RUNNING":
            raise ProtocolError("active attempt lane is not running")
        running_key = (
            active_attempt["stage"],
            active_attempt["lane"],
            active_attempt["attempt_id"],
        )
        if active_key != running_key:
            raise ProtocolError("transition history does not end at the active attempt")
    elif active_attempt_id is not None:
        raise ProtocolError("ledger names an active attempt but none is running")
    elif active_key is not None:
        raise ProtocolError("transition history has an unclosed attempt")


def validate_ledger(ledger: dict[str, Any]) -> None:
    """Validate the complete persisted protocol-v2 ledger contract."""
    _validate_ledger(ledger)


def _summary_from_attempt_history(
    attempts_by_key: dict[tuple[str, str, int], dict[str, Any]],
    stage_name: str,
    lane_record: dict[str, Any],
) -> tuple[str, str | None]:
    attempt_ids = lane_record["attempt_ids"]
    if not attempt_ids:
        return "READY", None

    history = [
        attempts_by_key[(stage_name, lane_record["name"], attempt_id)]
        for attempt_id in attempt_ids
    ]
    for previous, following in zip(history, history[1:]):
        if previous["state"] != "INCONCLUSIVE":
            raise ProtocolError(
                "only validity-INCONCLUSIVE attempts may precede another lane attempt"
            )
        if previous["close_ordinal"] >= following["open_ordinal"]:
            raise ProtocolError(
                "next lane attempt opened before the retryable attempt closed"
            )

    latest = history[-1]
    if latest["state"] == "RUNNING":
        return "RUNNING", None
    if latest["state"] == "INCONCLUSIVE":
        return "RETRYABLE", None
    return "COMPLETE", latest["statistical_result"]


def _validate_lane_summary(
    lane_record: dict[str, Any], expected_state: str, expected_result: str | None
) -> None:
    if (
        lane_record["state"] != expected_state
        or lane_record["result"] != expected_result
    ):
        raise ProtocolError(
            f"lane {lane_record['name']} summary does not match its attempt history"
        )


def _find_attempt(
    attempts: list[dict[str, Any]],
    stage_name: str,
    lane_name: str,
    attempt_id: int,
) -> dict[str, Any]:
    matches = [
        attempt
        for attempt in attempts
        if attempt["stage"] == stage_name
        and attempt["lane"] == lane_name
        and attempt["attempt_id"] == attempt_id
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"attempt is not uniquely registered: {stage_name}/{lane_name}/{attempt_id}"
        )
    return matches[0]


def open_attempt(
    ledger: dict[str, Any], stage_name: str, lane_name: str, attempt_id: int
) -> dict[str, Any]:
    """Register and open the next whole-lane attempt."""
    _validate_ledger(ledger)
    if type(attempt_id) is not int or attempt_id <= 0:
        raise ProtocolError("attempt id must be a positive integer")
    if ledger["active_attempt_id"] is not None:
        raise ProtocolError("another attempt is already running")
    if stage_name not in STAGE_NAMES or lane_name not in LANE_NAMES:
        raise ProtocolError(f"unknown ledger stage/lane: {stage_name}/{lane_name}")

    updated = copy.deepcopy(ledger)
    _, lane_record = _stage_and_lane(updated, stage_name, lane_name)
    expected_attempt_id = len(lane_record["attempt_ids"]) + 1
    if attempt_id != expected_attempt_id:
        raise ProtocolError(
            f"lane attempt id must be {expected_attempt_id}; got {attempt_id}"
        )
    if lane_record["state"] not in {"READY", "RETRYABLE"}:
        raise ProtocolError(
            f"lane {stage_name}/{lane_name} cannot open from {lane_record['state']}"
        )
    lane_record["state"] = "RUNNING"
    lane_record["attempt_ids"].append(attempt_id)
    updated["attempts"].append(
        {
            "attempt_id": attempt_id,
            "stage": stage_name,
            "lane": lane_name,
            "open_ordinal": updated["next_transition_ordinal"],
            "close_ordinal": None,
            "state": "RUNNING",
            "validity_state": "RUNNING",
            "statistical_result": None,
        }
    )
    updated["next_transition_ordinal"] += 1
    updated["active_attempt_id"] = attempt_id
    _validate_ledger(updated)
    return updated


def close_attempt(
    ledger: dict[str, Any],
    stage_name: str,
    lane_name: str,
    attempt_id: int,
    statistical_result: str | None,
    *,
    validity_state: str = "COMPLETE",
) -> dict[str, Any]:
    """Close a running attempt as complete or validity-inconclusive.

    A validity ``INCONCLUSIVE`` has no statistical result and makes the whole
    lane retryable.  Validity ``COMPLETE`` requires ``PASS``, ``FAIL``, or a
    statistical ``INCONCLUSIVE`` and permanently closes the lane.
    """
    _validate_ledger(ledger)
    if type(attempt_id) is not int or attempt_id <= 0:
        raise ProtocolError("attempt id must be a positive integer")
    if validity_state == "INCONCLUSIVE":
        if statistical_result is not None:
            raise ProtocolError(
                "validity INCONCLUSIVE must not have a statistical result"
            )
    elif validity_state == "COMPLETE":
        if statistical_result not in {"PASS", "FAIL", "INCONCLUSIVE"}:
            raise ProtocolError(
                "validity COMPLETE requires PASS, FAIL, or statistical INCONCLUSIVE"
            )
    else:
        raise ProtocolError(f"unsupported validity state: {validity_state!r}")

    if ledger["active_attempt_id"] != attempt_id:
        raise ProtocolError(f"attempt {attempt_id} is not the active attempt")
    attempt = _find_attempt(ledger["attempts"], stage_name, lane_name, attempt_id)
    if attempt["state"] != "RUNNING":
        raise ProtocolError(f"attempt {attempt_id} is already terminal")

    updated = copy.deepcopy(ledger)
    updated_attempt = _find_attempt(
        updated["attempts"], stage_name, lane_name, attempt_id
    )
    _, lane_record = _stage_and_lane(updated, stage_name, lane_name)
    if lane_record["state"] != "RUNNING":
        raise ProtocolError("attempt lane is not running")

    updated["active_attempt_id"] = None
    updated_attempt["close_ordinal"] = updated["next_transition_ordinal"]
    updated["next_transition_ordinal"] += 1
    if validity_state == "INCONCLUSIVE":
        updated_attempt["state"] = "INCONCLUSIVE"
        updated_attempt["validity_state"] = "INCONCLUSIVE"
        lane_record["state"] = "RETRYABLE"
    else:
        updated_attempt["state"] = "COMPLETE"
        updated_attempt["validity_state"] = "COMPLETE"
        updated_attempt["statistical_result"] = statistical_result
        lane_record["state"] = "COMPLETE"
        lane_record["result"] = statistical_result
        if lane_name == LANE_NAMES[0] and statistical_result == "PASS":
            stage_record, _ = _stage_and_lane(updated, stage_name, lane_name)
            stage_record["lanes"][1]["state"] = "READY"

    _validate_ledger(updated)
    return updated
