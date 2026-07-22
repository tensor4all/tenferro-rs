#!/usr/bin/env python3
"""Collect and compose provenance-bound Phase 2E dispatch evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
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

SOURCE_HOT_ITEMS = (
    ("crates/tenferro-ad/src/eager.rs", "pub fn with_eager_session<R: Send>("),
    ("crates/tenferro-ad/src/eager_backend.rs", "macro_rules! dispatch"),
    ("crates/tenferro-ad/src/eager_backend.rs", "fn with_backend_session<R: Send>("),
    ("crates/tenferro-cpu/src/backend.rs", "fn with_backend_session<R: Send>("),
    ("crates/tenferro-cpu/src/exec_session.rs", "fn run_native<R: Send>("),
    ("crates/tenferro-cpu/src/exec_session.rs", "fn run_native_fresh<R: FreshCpuOutput + Send>("),
    ("crates/tenferro-cpu/src/provider.rs", "pub(crate) fn enter<R: Send>("),
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
        counts = row.get("counts")
        if not isinstance(counts, list) or len(counts) != 6 or not all(isinstance(x, int) for x in counts):
            raise protocol.ProtocolError(f"row {key} has invalid count vector")
        observed_cpus = row.get("observed_cpus")
        if (
            row.get("hardware_skip") is None
            and (
                not isinstance(observed_cpus, list)
                or not observed_cpus
                or not all(type(cpu) is int and cpu >= 0 for cpu in observed_cpus)
            )
        ):
            raise protocol.ProtocolError(f"row {key} lacks actual CPU observations")
        expected_counts, expected_mode = expected_row_contract(key)
        if counts != expected_counts or row.get("mode") != expected_mode:
            raise protocol.ProtocolError(f"row {key} differs from its count/mode contract")
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
    rows = [*cpu_rows, *ad_rows]
    if len({row["key"] for row in rows}) != 47:
        raise protocol.ProtocolError("composed characterization must contain 47 unique rows")
    return {"validity_state": "PASS", "row_count": 47, "rows": rows}


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


def validate_source_contract(repository: pathlib.Path) -> dict[str, str]:
    digests: dict[str, str] = {}
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
        digests[relative] = sha256_file(path)
    production = (repository / "crates/tenferro-ad/src/eager_backend.rs").read_text()
    for fixture in ("struct RecordingBackend", "delegate_recording_backend_methods", "impl TensorElementwise for RecordingBackend"):
        if fixture in production:
            raise protocol.ProtocolError("RecordingBackend fixture leaked into production source")
    return digests


def validate_test_build_manifest(
    manifest: Mapping[str, Any], *, package: str, candidate: str,
) -> pathlib.Path:
    if manifest.get("validity_state") != "COMPLETE" or manifest.get("package") != package:
        raise protocol.ProtocolError("test build manifest identity mismatch")
    if manifest.get("candidate") != candidate:
        raise protocol.ProtocolError("test binary was not built from the candidate")
    if tuple(manifest.get("argv", ())) != build.DISPATCH_TEST_COMMANDS[package]:
        raise protocol.ProtocolError("evidence rejects a non-contract Cargo test build")
    if manifest.get("requested_features") != ["cpu-faer"] or manifest.get("no_default_features") is not True:
        raise protocol.ProtocolError("test build used the wrong feature graph request")
    target = manifest.get("target")
    if not isinstance(target, str) or not target:
        raise protocol.ProtocolError("test build manifest lacks the host target")
    expected_query = build.feature_query_command(
        target, package=package, requested_features=("cpu-faer",), no_default_features=True
    )
    if tuple(manifest.get("feature_query_argv", ())) != expected_query:
        raise protocol.ProtocolError("package feature query differs from the locked contract")
    executable = pathlib.Path(str(manifest.get("executable", ""))).resolve(strict=True)
    if sha256_file(executable) != manifest.get("executable_sha256"):
        raise protocol.ProtocolError("test executable digest mismatch")
    for field in (
        "source_sha256", "lock_sha256", "feature_graph_sha256", "environment", "toolchain"
    ):
        if not manifest.get(field):
            raise protocol.ProtocolError(f"test build manifest lacks {field}")
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
        try:
            stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise protocol.ProtocolError("validity inconclusive: deadline-exceeded") from error
    return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)


def run_test_executable(
    executable: pathlib.Path, filter_name: str, *, repository: pathlib.Path,
    evidence_root: pathlib.Path, environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    runtime_environment = dict(environment)
    runtime_environment[EVIDENCE_ENVIRONMENT_KEY] = str(evidence_root.resolve())
    argv = (str(executable.resolve()), filter_name, "--nocapture")
    result = run_bounded(
        argv, cwd=repository, environment=runtime_environment, deadline=TEST_DEADLINE_SECONDS
    )
    if result.returncode != 0:
        raise protocol.ProtocolError(f"direct test executable failed: {filter_name}")
    return result


def validate_bench_build_manifest(
    manifest: Mapping[str, Any], *, owner: str, candidate: str,
) -> pathlib.Path:
    expected = build.CHARACTERIZATION_BENCH_COMMANDS[owner]
    if manifest.get("validity_state") != "COMPLETE" or manifest.get("candidate") != candidate:
        raise protocol.ProtocolError("characterization bench candidate identity mismatch")
    if tuple(manifest.get("argv", ())) != expected:
        raise protocol.ProtocolError("characterization bench build argv differs from contract")
    if manifest.get("requested_features") != ["cpu-faer"] or manifest.get("no_default_features") is not True:
        raise protocol.ProtocolError("characterization bench feature graph differs from contract")
    executable = pathlib.Path(str(manifest.get("executable", ""))).resolve(strict=True)
    if sha256_file(executable) != manifest.get("executable_sha256"):
        raise protocol.ProtocolError("characterization bench executable digest mismatch")
    for field in ("source_sha256", "lock_sha256", "feature_graph_sha256", "environment"):
        if not manifest.get(field):
            raise protocol.ProtocolError(f"characterization bench manifest lacks {field}")
    return executable


def run_bench_row(
    executable: pathlib.Path, row_key: str, *, repository: pathlib.Path,
    environment: Mapping[str, str], criterion_home: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    runtime_environment = dict(environment)
    runtime_environment["CRITERION_HOME"] = str(criterion_home.resolve())
    argv = (str(executable.resolve()), row_key, "--noplot")
    result = run_bounded(
        argv, cwd=repository, environment=runtime_environment,
        deadline=BENCH_ROW_DEADLINE_SECONDS,
    )
    if result.returncode != 0:
        raise protocol.ProtocolError(f"characterization bench row failed: {row_key}")
    return result


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=pathlib.Path)
    parser.add_argument("--evidence-root", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    validate_candidate_worktree(args.repository, args.candidate)
    validate_source_contract(args.repository)
    manifests = {}
    for package, short, filter_name in (
        ("tenferro-cpu", "cpu", CPU_FILTER), ("tenferro-ad", "ad", AD_FILTER)
    ):
        manifest = _read_json(args.evidence_root / build.DISPATCH_BUILD_MANIFEST_PATHS[package])
        executable = validate_test_build_manifest(manifest, package=package, candidate=args.candidate)
        run_test_executable(
            executable, filter_name, repository=args.repository,
            evidence_root=args.evidence_root, environment=manifest["environment"],
        )
        artifact = args.evidence_root / "dispatch-gates" / f"{short}-evidence.json"
        manifests[short] = {"sha256": sha256_file(artifact), "artifact": str(artifact)}
    cpu = _read_json(pathlib.Path(manifests["cpu"]["artifact"]))
    ad = _read_json(pathlib.Path(manifests["ad"]["artifact"]))
    characterization = compose_characterization(cpu, ad)
    bench_digests = {}
    criterion_root = args.evidence_root.parent / f".{args.evidence_root.name}-criterion"
    criterion_root.mkdir(mode=0o700, exist_ok=False)
    try:
        for owner, surfaces in (("cpu", {"D-N", "D-D", "G-O"}), ("ad", {"E-N", "E-D"})):
            manifest = _read_json(
                args.evidence_root / build.CHARACTERIZATION_BUILD_MANIFEST_PATHS[owner]
            )
            executable = validate_bench_build_manifest(
                manifest, owner=owner, candidate=args.candidate
            )
            for row in characterization["rows"]:
                if row["surface"] in surfaces:
                    run_bench_row(
                        executable, row["key"], repository=args.repository,
                        environment=manifest["environment"], criterion_home=criterion_root,
                    )
            bench_digests[owner] = manifest["executable_sha256"]
    finally:
        import shutil
        shutil.rmtree(criterion_root, ignore_errors=True)
    characterization["bench_executable_sha256"] = bench_digests
    atomic_write_json(args.evidence_root / "dispatch-gates/manifest.json", {"candidate": args.candidate, **manifests})
    atomic_write_json(args.evidence_root / "characterization/manifest.json", characterization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
