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
import secrets
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

AFFINITY_ROWS = frozenset(
    f"{ownership}/budget-{budget}/{surface}"
    for ownership in ("managed-exact", "external-exact", "external-advisory")
    for budget in (1, 2, 4)
    for surface in ("D-N", "D-D", "G-O", "E-N", "E-D")
)


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
    root = pathlib.Path(os.path.abspath(root))
    current = pathlib.Path(root.anchor)
    for component in root.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise ProtocolError(f"cannot inspect evidence root {root}: {error}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ProtocolError(f"evidence root traverses a symbolic link: {current}")
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        try:
            root.mkdir(mode=0o700, parents=True)
        except OSError as error:
            raise ProtocolError(f"cannot create evidence root {root}: {error}") from error
        try:
            metadata = root.lstat()
        except OSError as error:
            raise ProtocolError(f"cannot inspect evidence root {root}: {error}") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProtocolError(f"evidence root is not a regular directory: {root}")
    except OSError as error:
        raise ProtocolError(f"cannot inspect evidence root {root}: {error}") from error

    if not stat.S_ISDIR(metadata.st_mode):
        raise ProtocolError(f"evidence root is not a regular directory: {root}")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise ProtocolError(f"evidence root is not owned by the current user: {root}")
    try:
        first_entry = next(root.iterdir(), None)
    except OSError as error:
        raise ProtocolError(f"cannot inspect evidence root {root}: {error}") from error
    if first_entry is not None:
        raise ProtocolError(f"evidence root is not empty: {root}")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        try:
            root.chmod(0o700)
            metadata = root.lstat()
        except OSError as error:
            raise ProtocolError(f"cannot make evidence root private: {root}: {error}") from error
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ProtocolError(f"evidence root is not private (mode must be 0700): {root}")
    return root


class PreparedRootIdentity:
    """Hold and revalidate one private evidence-root directory identity.

    Revalidation detects path-component replacement between protocol operations.
    The remaining threat boundary is a hostile same-UID process that wins the
    narrow interval after revalidation and before a nested pathname syscall;
    final normative file opens additionally use ``O_NOFOLLOW | O_EXCL``.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            self.descriptor = os.open(self.path, flags)
            metadata = os.fstat(self.descriptor)
        except OSError as error:
            raise ProtocolError(f"cannot hold evidence root {self.path}: {error}") from error
        self.device = metadata.st_dev
        self.inode = metadata.st_ino
        try:
            self.revalidate()
        except BaseException:
            self.close()
            raise

    def revalidate(self) -> None:
        current = pathlib.Path(self.path.anchor)
        for component in self.path.parts[1:]:
            current /= component
            try:
                metadata = current.lstat()
            except OSError as error:
                raise ProtocolError(
                    f"evidence root identity disappeared: {self.path}: {error}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ProtocolError(f"evidence root traverses a symbolic link: {current}")
        metadata = self.path.lstat()
        held = os.fstat(self.descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (self.device, self.inode)
            or (held.st_dev, held.st_ino) != (self.device, self.inode)
        ):
            raise ProtocolError(f"evidence root identity changed: {self.path}")

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def prepare_empty_root_identity(root: pathlib.Path) -> PreparedRootIdentity:
    """Prepare an empty private root and retain its device/inode identity."""
    return PreparedRootIdentity(prepare_empty_root(root))


def runtime_environment(
    *,
    path: str,
    home: str,
    criterion_home: str | None = None,
    affinity_row: str | None = None,
    affinity_file: str | None = None,
) -> dict[str, str]:
    """Construct the runtime allowlist without inheriting ambient variables."""
    if not isinstance(path, str) or not isinstance(home, str):
        raise ProtocolError("PATH and HOME must be strings")
    if criterion_home is not None and not isinstance(criterion_home, str):
        raise ProtocolError("CRITERION_HOME must be a string when supplied")
    if (affinity_row is None) != (affinity_file is None):
        raise ProtocolError("affinity row and file must be supplied together")
    if affinity_row is not None:
        if type(affinity_row) is not str or type(affinity_file) is not str:
            raise ProtocolError("affinity row and file must be strings")
        if affinity_row not in AFFINITY_ROWS:
            raise ProtocolError("affinity row key is not canonical")
        if criterion_home is None:
            raise ProtocolError("affinity parameters require CRITERION_HOME")
        criterion_root = pathlib.Path(criterion_home)
        destination = pathlib.Path(affinity_file)
        try:
            root_metadata = criterion_root.lstat()
            resolved_root = criterion_root.resolve(strict=True)
        except OSError as error:
            raise ProtocolError(
                f"CRITERION_HOME cannot be inspected for affinity evidence: {error}"
            ) from error
        expected_destination = criterion_root / "affinity.json"
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or resolved_root != criterion_root
            or destination != expected_destination
        ):
            raise ProtocolError(
                "affinity file must be exactly canonical CRITERION_HOME/affinity.json"
            )
        try:
            destination_metadata = destination.lstat()
        except FileNotFoundError:
            destination_metadata = None
        except OSError as error:
            raise ProtocolError(f"affinity file cannot be inspected: {error}") from error
        if destination_metadata is not None and not stat.S_ISREG(
            destination_metadata.st_mode
        ):
            raise ProtocolError("affinity file may not be a symlink or special file")
        try:
            resolved_destination = destination.resolve(
                strict=destination_metadata is not None
            )
        except OSError as error:
            raise ProtocolError(f"affinity file cannot be resolved: {error}") from error
        if resolved_destination != expected_destination:
            raise ProtocolError("affinity file path is not canonical")
    result = {
        "PATH": path,
        "HOME": home,
        "LC_ALL": "C",
        "TZ": "UTC",
        **THREAD_ENV,
    }
    if criterion_home is not None:
        result["CRITERION_HOME"] = criterion_home
    if affinity_row is not None:
        result["TENFERRO_PHASE2E_AFFINITY_ROW"] = affinity_row
        result["TENFERRO_PHASE2E_AFFINITY_FILE"] = affinity_file
    return result


def cargo_environment(
    *, path: str, home: str, cargo_home: str, target_dir: str
) -> dict[str, str]:
    """Construct the sealed Cargo environment for Phase 2E build processes."""
    values = {
        "PATH": path,
        "HOME": home,
        "CARGO_HOME": cargo_home,
        "CARGO_TARGET_DIR": target_dir,
    }
    for name, value in values.items():
        if not isinstance(value, str):
            raise ProtocolError(f"{name} must be a string")
        paths = value.split(os.pathsep) if name == "PATH" else [value]
        if not paths or any(
            not item or not pathlib.Path(item).is_absolute() for item in paths
        ):
            raise ProtocolError(f"{name} must be an absolute path")
        if name == "PATH":
            normalized: list[pathlib.Path] = []
            for item in paths:
                candidate = pathlib.Path(item)
                try:
                    metadata = candidate.lstat()
                    canonical = candidate.resolve(strict=True)
                except OSError as error:
                    raise ProtocolError(
                        f"PATH component cannot be inspected: {candidate}: {error}"
                    ) from error
                if not stat.S_ISDIR(metadata.st_mode) or canonical != candidate:
                    raise ProtocolError(
                        f"PATH component must be a canonical regular directory: {candidate}"
                    )
                if candidate in normalized:
                    raise ProtocolError(f"PATH component is duplicated: {candidate}")
                normalized.append(candidate)

    result = runtime_environment(path=path, home=home)
    result.update(
        {
            "CARGO_HOME": cargo_home,
            "CARGO_TARGET_DIR": target_dir,
            "CARGO_INCREMENTAL": "0",
            "CARGO_NET_OFFLINE": "true",
        }
    )
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


def decode_canonical_json_bytes(payload: bytes, context: str) -> Any:
    """Decode one exact canonical JSON document with strict number semantics."""
    if type(payload) is not bytes or type(context) is not str or not context:
        raise ProtocolError("canonical JSON decoder arguments are invalid")

    def reject_duplicates(pairs):
        decoded_object = {}
        for key, value in pairs:
            if key in decoded_object:
                raise ProtocolError(f"{context} contains duplicate key: {key}")
            decoded_object[key] = value
        return decoded_object

    try:
        text = payload.decode("utf-8")
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"{context} contains non-finite number: {value}")
            ),
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolError(f"{context} is malformed canonical JSON") from error
    if payload != _canonical_json_bytes(decoded):
        raise ProtocolError(f"{context} is not canonical JSON")
    return decoded


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


def atomic_write_json_at(
    directory_descriptor: int, name: str, payload: Any
) -> None:
    """Atomically replace one leaf JSON file below a caller-held directory fd."""
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
    ):
        raise ProtocolError("atomic JSON leaf name is invalid")
    try:
        directory_metadata = os.fstat(directory_descriptor)
    except OSError as error:
        raise AtomicWriteError(
            f"cannot inspect parent directory for {name}: {error}", committed=False
        ) from error
    if not stat.S_ISDIR(directory_metadata.st_mode):
        raise ProtocolError("atomic JSON parent descriptor is not a directory")
    encoded = _canonical_json_bytes(payload)
    temporary_name = f".{name}.write-{secrets.token_hex(16)}.tmp"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise AtomicWriteError(
            f"cannot create temporary for {name}: {error}", committed=False
        ) from error

    failure: BaseException | None = None
    try:
        view = memoryview(encoded)
        while view:
            try:
                written = os.write(descriptor, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError("short JSON write")
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
        if isinstance(failure, OSError):
            raise AtomicWriteError(
                f"cannot write and fsync temporary for {name}: {failure}",
                committed=False,
            ) from failure
        raise failure

    try:
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise AtomicWriteError(
            f"cannot replace {name}: {error}", committed=False
        ) from error
    try:
        os.fsync(directory_descriptor)
    except OSError as error:
        raise AtomicWriteDurabilityError(
            f"replacement committed for {name}, but parent fsync failed: {error}"
        ) from error


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
                "artifact_root": (str, type(None)),
                "artifact_device": (int, type(None)),
                "artifact_inode": (int, type(None)),
                "artifact_state": str,
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
        artifact_root = attempt["artifact_root"]
        artifact_device = attempt["artifact_device"]
        artifact_inode = attempt["artifact_inode"]
        artifact_state = attempt["artifact_state"]
        if attempt["stage"] == "timing":
            if (
                artifact_root is not None
                or artifact_device is not None
                or artifact_inode is not None
                or artifact_state != "NOT_APPLICABLE"
            ):
                raise ProtocolError("timing attempt has allocation artifact ownership")
        elif (
            type(artifact_root) is not str
            or not pathlib.Path(artifact_root).is_absolute()
            or os.path.normpath(artifact_root) != artifact_root
            or artifact_state not in {"RESERVED", "BOUND"}
        ):
            raise ProtocolError("allocation attempt artifact reservation is invalid")
        if artifact_state == "RESERVED":
            if artifact_device is not None or artifact_inode is not None:
                raise ProtocolError("reserved allocation artifact has an identity")
        elif artifact_state == "BOUND":
            if (
                type(artifact_device) is not int
                or type(artifact_inode) is not int
                or artifact_device < 0
                or artifact_inode < 0
                or artifact_device > (1 << 64) - 1
                or artifact_inode > (1 << 64) - 1
            ):
                raise ProtocolError("bound allocation artifact identity is invalid")
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
    ledger: dict[str, Any],
    stage_name: str,
    lane_name: str,
    attempt_id: int,
    *,
    artifact_root: str | None = None,
) -> dict[str, Any]:
    """Register and open the next whole-lane attempt."""
    _validate_ledger(ledger)
    if type(attempt_id) is not int or attempt_id <= 0:
        raise ProtocolError("attempt id must be a positive integer")
    if ledger["active_attempt_id"] is not None:
        raise ProtocolError("another attempt is already running")
    if stage_name not in STAGE_NAMES or lane_name not in LANE_NAMES:
        raise ProtocolError(f"unknown ledger stage/lane: {stage_name}/{lane_name}")
    if stage_name == "allocation":
        if (
            type(artifact_root) is not str
            or not pathlib.Path(artifact_root).is_absolute()
            or os.path.normpath(artifact_root) != artifact_root
        ):
            raise ProtocolError("allocation attempt requires a canonical artifact root")
        artifact_state = "RESERVED"
    else:
        if artifact_root is not None:
            raise ProtocolError("timing attempt cannot reserve an allocation artifact")
        artifact_state = "NOT_APPLICABLE"

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
            "artifact_root": artifact_root,
            "artifact_device": None,
            "artifact_inode": None,
            "artifact_state": artifact_state,
        }
    )
    updated["next_transition_ordinal"] += 1
    updated["active_attempt_id"] = attempt_id
    _validate_ledger(updated)
    return updated


def bind_attempt_artifact(
    ledger: dict[str, Any],
    stage_name: str,
    lane_name: str,
    attempt_id: int,
    *,
    artifact_root: str,
    artifact_device: int,
    artifact_inode: int,
) -> dict[str, Any]:
    """Bind a reserved allocation attempt to one pinned directory identity."""
    _validate_ledger(ledger)
    if ledger["active_attempt_id"] != attempt_id:
        raise ProtocolError("allocation artifact attempt is not active")
    attempt = _find_attempt(ledger["attempts"], stage_name, lane_name, attempt_id)
    if (
        stage_name != "allocation"
        or attempt["state"] != "RUNNING"
        or attempt["artifact_state"] != "RESERVED"
        or type(artifact_root) is not str
        or artifact_root != attempt["artifact_root"]
        or type(artifact_device) is not int
        or type(artifact_inode) is not int
        or artifact_device < 0
        or artifact_inode < 0
        or artifact_device > (1 << 64) - 1
        or artifact_inode > (1 << 64) - 1
    ):
        raise ProtocolError("allocation artifact binding differs from reservation")
    updated = copy.deepcopy(ledger)
    bound = _find_attempt(updated["attempts"], stage_name, lane_name, attempt_id)
    bound["artifact_device"] = artifact_device
    bound["artifact_inode"] = artifact_inode
    bound["artifact_state"] = "BOUND"
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
