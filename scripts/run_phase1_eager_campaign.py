#!/usr/bin/env python3
"""Run one indivisible protocol-v2 Phase 2E timing campaign."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import pathlib
import platform
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

try:
    from scripts import classify_criterion_noninferiority as classification
    from scripts import phase2e_build as build
    from scripts import phase2e_protocol as protocol
except ModuleNotFoundError:
    import classify_criterion_noninferiority as classification
    import phase2e_build as build
    import phase2e_protocol as protocol


CANONICAL_CASES = protocol.CANONICAL_CASES
PAIR_ORDERS = protocol.PAIR_ORDERS
RUN_ROLES = protocol.RUN_ROLES
THREAD_ENVIRONMENT = dict(protocol.THREAD_ENV)
SENTINEL_BENCHMARK = CANONICAL_CASES["lazy_neg_1"]
QUIET_DEADLINE_SECONDS = 300
QUIET_POLL_SECONDS = 1
PROCESS_DEADLINE_SECONDS = 30
TERMINATION_GRACE_SECONDS = 5
EXIT_BY_RESULT = {
    ("COMPLETE", "PASS"): 0,
    ("INCONCLUSIVE", None): 2,
    ("COMPLETE", "FAIL"): 3,
    ("COMPLETE", "INCONCLUSIVE"): 4,
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def format_cpu_list(cpus) -> str:
    ordered = sorted(cpus)
    if not ordered:
        return ""
    ranges = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def criterion_directory(criterion_root: pathlib.Path, benchmark: str) -> pathlib.Path:
    components = benchmark.split("/")
    if len(components) != 4 or any(not component for component in components):
        raise protocol.ProtocolError(f"unexpected benchmark identifier: {benchmark}")
    group = f"{components[0]}_{components[1]}"
    return criterion_root / group / components[2] / components[3]


def run_identities(order: str) -> tuple[str, str, str, str]:
    if order == "A/B":
        return "candidate", "baseline", "candidate", "candidate"
    if order == "B/A":
        return "candidate", "candidate", "baseline", "candidate"
    raise protocol.ProtocolError(f"unsupported pair order: {order}")


def benchmark_command(
    binary: pathlib.Path,
    benchmark: str,
    comparison_option: str,
    comparison_name: str,
) -> tuple[str, ...]:
    return (
        str(binary),
        "--bench",
        benchmark,
        comparison_option,
        comparison_name,
        "--noplot",
    )


def exact_build_processes() -> list[dict[str, Any]]:
    processes = []
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        return processes
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name in {"cargo", "rustc"}:
            processes.append({"pid": int(entry.name), "name": name})
    return sorted(processes, key=lambda record: record["pid"])


def _normalized_load(load_provider: Callable[[], float], allowed_count: int) -> float:
    return float(load_provider()) / allowed_count


def _is_within(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_roots(
    artifact_root: pathlib.Path, criterion_root: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    artifact_input = pathlib.Path(os.path.abspath(artifact_root))
    criterion_input = pathlib.Path(os.path.abspath(criterion_root))
    artifact = artifact_input.resolve(strict=False)
    criterion = criterion_input.resolve(strict=False)
    if artifact == criterion or _is_within(artifact, criterion) or _is_within(
        criterion, artifact
    ):
        raise protocol.ProtocolError("artifact and Criterion roots must be disjoint")
    prepared_artifact = protocol.prepare_empty_root(artifact_input)
    prepared_criterion = protocol.prepare_empty_root(criterion_input)
    return (
        prepared_artifact.resolve(strict=True),
        prepared_criterion.resolve(strict=True),
    )


def _read_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    value = classification.read_json(path)
    if type(value) is not dict:
        raise protocol.ProtocolError(f"{context} must be a JSON object")
    return value


def _build_inputs(args) -> tuple[dict[str, Any], dict[str, pathlib.Path], dict[str, str]]:
    paths = {
        "baseline": pathlib.Path(args.baseline_build_manifest).resolve(strict=True),
        "candidate": pathlib.Path(args.candidate_build_manifest).resolve(strict=True),
    }
    manifests = {
        identity: _read_json(path, f"{identity} build manifest")
        for identity, path in paths.items()
    }
    build.validate_pair(args.comparison_kind, manifests["baseline"], manifests["candidate"])
    binaries = {
        identity: pathlib.Path(manifest["executable"]).resolve(strict=True)
        for identity, manifest in manifests.items()
    }
    binary_shas = {
        identity: protocol.sha256_file(binary) for identity, binary in binaries.items()
    }
    for identity, manifest in manifests.items():
        if binary_shas[identity] != manifest["executable_sha256"]:
            raise protocol.ProtocolError(f"{identity} executable digest changed")
    records = {
        identity: {
            "path": str(paths[identity]),
            "sha256": protocol.sha256_file(paths[identity]),
            "role": manifests[identity]["role"],
            "executable_sha256": binary_shas[identity],
        }
        for identity in ("baseline", "candidate")
    }
    return records, binaries, binary_shas


def _runtime_environment(
    candidate_manifest_path: pathlib.Path, criterion_root: pathlib.Path
) -> dict[str, str]:
    manifest = _read_json(candidate_manifest_path, "candidate build manifest")
    environment = manifest["environment"]
    return protocol.runtime_environment(
        path=environment["PATH"],
        home=environment["HOME"],
        criterion_home=str(criterion_root),
    )


def _sample_host(
    *,
    pid: int,
    phase: str,
    sequence: int,
    allowed_count: int,
    affinity_provider: Callable[[int], set[int]],
    load_provider: Callable[[], float],
    build_process_provider: Callable[[], list[dict[str, Any]]],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    try:
        affinity = format_cpu_list(affinity_provider(pid))
    except (ProcessLookupError, PermissionError, OSError):
        affinity = ""
    processes = build_process_provider()
    return {
        "sequence": sequence,
        "phase": phase,
        "monotonic_seconds": float(monotonic()),
        "observed_affinity": affinity,
        "normalized_load": _normalized_load(load_provider, allowed_count),
        "cargo_processes": [record for record in processes if record.get("name") == "cargo"],
        "rustc_processes": [record for record in processes if record.get("name") == "rustc"],
    }


def _best_effort_signal(
    pid: int, requested_signal: signal.Signals, signal_process_group
) -> bool:
    try:
        signal_process_group(pid, requested_signal)
    except ProcessLookupError:
        return True
    except BaseException:
        return False
    return True


def _terminate_group(process, signal_process_group) -> tuple[bool, bool, list[str]]:
    failures = []
    terminated = _best_effort_signal(process.pid, signal.SIGTERM, signal_process_group)
    if not terminated:
        failures.append("term-signal-failed")
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
        return terminated, False, failures
    except subprocess.TimeoutExpired:
        pass
    except BaseException as error:
        failures.append(f"term-wait-failed:{type(error).__name__}")
        if process.returncode is not None:
            return terminated, False, failures
    killed = _best_effort_signal(process.pid, signal.SIGKILL, signal_process_group)
    if not killed:
        failures.append("kill-signal-failed")
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        failures.append("kill-wait-timeout")
    except BaseException as error:
        failures.append(f"kill-wait-failed:{type(error).__name__}")
    return terminated, killed, failures


def _process_record(
    *,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    cwd: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    role: str,
    identity: str,
    binary_sha: str,
    selected_cpu: int,
    allowed_count: int,
    process_factory,
    signal_process_group,
    monotonic,
    sleep,
    affinity_provider,
    load_provider,
    build_process_provider,
) -> tuple[dict[str, Any], str | None]:
    preamble = {
        "argv": list(command),
        "environment": dict(sorted(environment.items())),
        "environment_sha256": protocol.sha256_json(dict(sorted(environment.items()))),
    }
    process = None
    samples = []
    timed_out = False
    cleanup_failures: list[str] = []
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        stdout.write(json.dumps(preamble, sort_keys=True) + "\n")
        stdout.flush()
        try:
            process = process_factory(
                list(command),
                cwd=str(cwd),
                env=dict(sorted(environment.items())),
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
                preexec_fn=lambda: os.sched_setaffinity(0, {selected_cpu}),
            )
            started = float(monotonic())
            samples.append(
                _sample_host(
                    pid=process.pid,
                    phase="start",
                    sequence=0,
                    allowed_count=allowed_count,
                    affinity_provider=affinity_provider,
                    load_provider=load_provider,
                    build_process_provider=build_process_provider,
                    monotonic=monotonic,
                )
            )
            deadline = started + PROCESS_DEADLINE_SECONDS
            while True:
                status = process.poll()
                if status is not None:
                    break
                now = float(monotonic())
                if now >= deadline:
                    timed_out = True
                    _terminated, _killed, cleanup_failures = _terminate_group(
                        process, signal_process_group
                    )
                    status = process.returncode
                    break
                sleep(min(1.0, deadline - now))
                if process.poll() is None:
                    if float(monotonic()) >= deadline:
                        timed_out = True
                        _terminated, _killed, cleanup_failures = _terminate_group(
                            process, signal_process_group
                        )
                        status = process.returncode
                        break
                    samples.append(
                        _sample_host(
                            pid=process.pid,
                            phase="periodic",
                            sequence=len(samples),
                            allowed_count=allowed_count,
                            affinity_provider=affinity_provider,
                            load_provider=load_provider,
                            build_process_provider=build_process_provider,
                            monotonic=monotonic,
                        )
                    )
            ended = max(float(monotonic()), started + 1e-9)
            samples.append(
                _sample_host(
                    pid=process.pid,
                    phase="end",
                    sequence=len(samples),
                    allowed_count=allowed_count,
                    affinity_provider=affinity_provider,
                    load_provider=load_provider,
                    build_process_provider=build_process_provider,
                    monotonic=lambda: ended,
                )
            )
        except BaseException:
            if process is not None:
                try:
                    _terminate_group(process, signal_process_group)
                except BaseException:
                    pass
            raise

    reason = None
    if timed_out:
        reason = "benchmark-process-timeout"
        if cleanup_failures:
            reason += ":" + "+".join(cleanup_failures)
    elif status != 0:
        reason = f"benchmark-process-exit:{status}"
    elif any(
        sample["observed_affinity"] != str(selected_cpu)
        or sample["normalized_load"] > 0.25
        or sample["cargo_processes"]
        or sample["rustc_processes"]
        for sample in samples
    ):
        reason = "benchmark-monitor-invalid"
    return (
        {
            "role": role,
            "binary": identity,
            "binary_sha256": binary_sha,
            "validity_state": "COMPLETE" if reason is None else "INCONCLUSIVE",
            "exit_status": int(status if status is not None else -1),
            "stdout_artifact": stdout_path.name,
            "stderr_artifact": stderr_path.name,
            "process_started_monotonic": started,
            "process_ended_monotonic": ended,
            "monitor_samples": samples,
        },
        reason,
    )


def _copy_regular(source: pathlib.Path, destination: pathlib.Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise protocol.ProtocolError(
            f"cannot open Criterion estimate {source}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise protocol.ProtocolError(f"Criterion estimate is not regular: {source}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = source.stat(follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise protocol.ProtocolError(f"Criterion estimate changed: {source}")
    finally:
        os.close(descriptor)
    output = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(b"".join(chunks))
        while view:
            written = os.write(output, view)
            if written <= 0:
                raise OSError("short Criterion estimate copy")
            view = view[written:]
        os.fsync(output)
    finally:
        os.close(output)


def _artifact_record(path: pathlib.Path) -> dict[str, str]:
    return {"sha256": protocol.sha256_file(path)}


def _record_artifact(
    campaign: dict[str, Any], root: pathlib.Path, path: pathlib.Path
) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    record = _artifact_record(path)
    if relative in campaign["artifact_inventory"]:
        raise protocol.ProtocolError(f"artifact registered more than once: {relative}")
    campaign["artifact_inventory"][relative] = record
    return record


def _synchronize_prefix_inventory(
    campaign: dict[str, Any], artifact_root: pathlib.Path
) -> None:
    """Make an invalid campaign's inventory match every durable prefix file."""
    discovered: dict[str, dict[str, str]] = {}
    for path in sorted(artifact_root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        relative = path.relative_to(artifact_root).as_posix()
        if relative == "campaign.json":
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise protocol.ProtocolError(
                f"invalid artifact-prefix file type: {relative}"
            )
        discovered[relative] = _artifact_record(path)
    campaign["artifact_inventory"] = discovered


def _quiet_host(
    *,
    allowed_count: int,
    load_limit: float,
    monotonic,
    sleep,
    load_provider,
    build_process_provider,
) -> str | None:
    started = float(monotonic())
    while True:
        load = _normalized_load(load_provider, allowed_count)
        processes = build_process_provider()
        if load <= load_limit and not processes:
            return None
        now = float(monotonic())
        if now - started >= QUIET_DEADLINE_SECONDS:
            return "quiet-host-timeout"
        sleep(min(float(QUIET_POLL_SECONDS), QUIET_DEADLINE_SECONDS - (now - started)))


def _pair_specs(case: str, benchmark: str, pair: int, order: str):
    identities = run_identities(order)
    target_name = f"phase2e-target-{case}-p{pair}"
    sentinel_name = f"phase2e-sentinel-{case}-p{pair}"
    return (
        (RUN_ROLES[0], identities[0], SENTINEL_BENCHMARK, "--save-baseline", sentinel_name),
        (RUN_ROLES[1], identities[1], benchmark, "--save-baseline", target_name),
        (RUN_ROLES[2], identities[2], benchmark, "--baseline", target_name),
        (RUN_ROLES[3], identities[3], SENTINEL_BENCHMARK, "--baseline", sentinel_name),
    )


def _write_monitor_artifact(
    pair_dir: pathlib.Path, case: str, pair: int, runs: list[dict[str, Any]]
) -> pathlib.Path:
    path = pair_dir / "monitor-samples.json"
    protocol.atomic_write_json(
        path,
        {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "case": case,
            "pair": pair,
            "runs": {run["role"]: copy.deepcopy(run["monitor_samples"]) for run in runs},
        },
    )
    return path


def _invalid_campaign(
    campaign: dict[str, Any], *, case: str, pair: int, role: str, reason: str
) -> dict[str, Any]:
    terminal = copy.deepcopy(campaign)
    terminal["validity_state"] = "INCONCLUSIVE"
    terminal["statistical_result"] = None
    terminal["completed_at"] = utc_now()
    terminal["invalid"] = {
        "case": case,
        "pair": pair,
        "role": role,
        "reason": reason,
    }
    terminal["prefix_inventory"] = copy.deepcopy(terminal["artifact_inventory"])
    return terminal


def _initial_campaign(
    args,
    *,
    build_records,
    selected_cpu: int,
    allowed_cpus: set[int],
) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.sha256_file(pathlib.Path(protocol.__file__)),
        "classifier_sha256": protocol.sha256_file(pathlib.Path(classification.__file__)),
        "candidate_sha": _read_json(
            pathlib.Path(args.candidate_build_manifest), "candidate build manifest"
        )["head"],
        "comparison_kind": args.comparison_kind,
        "build_manifests": build_records,
        "selected_cpu": selected_cpu,
        "allowed_cpus": format_cpu_list(allowed_cpus),
        "allowed_cpu_count": len(allowed_cpus),
        "normalized_load_limit": args.normalized_load_limit,
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "orders": list(PAIR_ORDERS),
        "criterion": dict(classification.CRITERION_SETTINGS),
        "validity_state": "RUNNING",
        "statistical_result": None,
        "completed_at": "",
        "cases": {
            case: {"benchmark": benchmark, "statistical_result": None, "pairs": {}}
            for case, benchmark in CANONICAL_CASES.items()
        },
        "artifact_inventory": {},
        "classification_artifacts": None,
    }


def _declare_results(campaign: dict[str, Any], root: pathlib.Path) -> str:
    results = {}
    for case in sorted(CANONICAL_CASES):
        intervals = []
        for pair, order in enumerate(PAIR_ORDERS, start=1):
            estimate = classification.read_change(root / case / f"pair{pair}/change-estimates.json")
            intervals.append(
                classification.invert_interval(*estimate) if order == "B/A" else {
                    "lower": estimate[0], "upper": estimate[1], "point": estimate[2]
                }
            )
        normalized = [
            value
            if isinstance(value, Mapping)
            else {"lower": value[0], "upper": value[1], "point": value[2]}
            for value in intervals
        ]
        result = classification.classify_case(normalized)
        campaign["cases"][case]["statistical_result"] = result
        results[case] = result
    return classification.campaign_result(results)


def _close_ledger(
    ledger_path: pathlib.Path,
    ledger: dict[str, Any],
    args,
    result: str | None,
    validity_state: str,
    atomic_writer,
) -> None:
    closed = protocol.close_attempt(
        ledger,
        "timing",
        args.comparison_kind,
        args.attempt_id,
        result,
        validity_state=validity_state,
    )
    atomic_writer(ledger_path, closed)


def run_campaign(
    args,
    *,
    process_factory=subprocess.Popen,
    signal_process_group=os.killpg,
    monotonic=time.monotonic,
    sleep=time.sleep,
    affinity_provider=os.sched_getaffinity,
    allowed_cpu_provider=lambda: set(os.sched_getaffinity(0)),
    load_provider=lambda: os.getloadavg()[0],
    build_process_provider=exact_build_processes,
    atomic_writer=protocol.atomic_write_json,
) -> int:
    manifest_path = None
    campaign = None
    ledger = None
    current = {"case": "<startup>", "pair": 0, "role": "<none>"}
    try:
        if args.comparison_kind not in protocol.LANE_NAMES:
            raise protocol.ProtocolError("invalid comparison kind")
        if args.normalized_load_limit != 0.25:
            raise protocol.ProtocolError("normalized load limit must be exactly 0.25")
        artifact_root, criterion_root = _prepare_roots(
            args.artifact_root, args.criterion_root
        )
        args.artifact_root = artifact_root
        args.criterion_root = criterion_root
        build_records, binaries, binary_shas = _build_inputs(args)
        for path in (
            pathlib.Path(args.baseline_build_manifest).resolve(),
            pathlib.Path(args.candidate_build_manifest).resolve(),
            pathlib.Path(args.ledger).resolve(),
        ):
            if _is_within(path, artifact_root) or _is_within(path, criterion_root):
                raise protocol.ProtocolError("read-only campaign input is inside a fresh root")
        ledger_path = pathlib.Path(args.ledger).resolve(strict=True)
        ledger = _read_json(ledger_path, "evidence ledger")
        allowed_cpus = set(allowed_cpu_provider())
        if not allowed_cpus:
            raise protocol.ProtocolError("process has no allowed CPUs")
        selected_cpu = min(allowed_cpus) if args.cpu is None else args.cpu
        if selected_cpu not in allowed_cpus:
            raise protocol.ProtocolError("selected CPU is not process-allowed")
        environment = _runtime_environment(
            pathlib.Path(args.candidate_build_manifest), criterion_root
        )
        campaign = _initial_campaign(
            args,
            build_records=build_records,
            selected_cpu=selected_cpu,
            allowed_cpus=allowed_cpus,
        )
        opened = protocol.open_attempt(
            ledger, "timing", args.comparison_kind, args.attempt_id
        )
        atomic_writer(ledger_path, opened)
        ledger = opened
        manifest_path = artifact_root / "campaign.json"
        atomic_writer(manifest_path, campaign)

        for case in sorted(CANONICAL_CASES):
            benchmark = CANONICAL_CASES[case]
            for pair, order in enumerate(PAIR_ORDERS, start=1):
                current = {"case": case, "pair": pair, "role": "quiet_wait"}
                quiet_error = _quiet_host(
                    allowed_count=len(allowed_cpus),
                    load_limit=args.normalized_load_limit,
                    monotonic=monotonic,
                    sleep=sleep,
                    load_provider=load_provider,
                    build_process_provider=build_process_provider,
                )
                if quiet_error is not None:
                    _synchronize_prefix_inventory(campaign, artifact_root)
                    terminal = _invalid_campaign(campaign, reason=quiet_error, **current)
                    atomic_writer(manifest_path, terminal)
                    _close_ledger(
                        pathlib.Path(args.ledger), ledger, args, None, "INCONCLUSIVE", atomic_writer
                    )
                    return 2

                pair_dir = artifact_root / case / f"pair{pair}"
                pair_dir.mkdir(parents=True)
                runs = []
                local_artifacts = {}
                invalid_reason = None
                for role, identity, run_benchmark, option, name in _pair_specs(
                    case, benchmark, pair, order
                ):
                    current = {"case": case, "pair": pair, "role": role}
                    stdout_path = pair_dir / f"{role}.stdout.log"
                    stderr_path = pair_dir / f"{role}.stderr.log"
                    record, invalid_reason = _process_record(
                        command=benchmark_command(
                            binaries[identity], run_benchmark, option, name
                        ),
                        environment=environment,
                        cwd=pathlib.Path(args.working_directory),
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        role=role,
                        identity=identity,
                        binary_sha=binary_shas[identity],
                        selected_cpu=selected_cpu,
                        allowed_count=len(allowed_cpus),
                        process_factory=process_factory,
                        signal_process_group=signal_process_group,
                        monotonic=monotonic,
                        sleep=sleep,
                        affinity_provider=affinity_provider,
                        load_provider=load_provider,
                        build_process_provider=build_process_provider,
                    )
                    runs.append(record)
                    for path in (stdout_path, stderr_path):
                        local_artifacts[path.name] = _record_artifact(
                            campaign, artifact_root, path
                        )
                    campaign["cases"][case]["active_pair"] = {
                        "pair": pair,
                        "order": order,
                        "runs": copy.deepcopy(runs),
                    }
                    atomic_writer(manifest_path, campaign)
                    if invalid_reason is not None:
                        break

                if invalid_reason is None:
                    target_source = criterion_directory(criterion_root, benchmark)
                    sentinel_source = criterion_directory(
                        criterion_root, SENTINEL_BENCHMARK
                    )
                    _copy_regular(
                        target_source / "change/estimates.json",
                        pair_dir / "change-estimates.json",
                    )
                    _copy_regular(
                        sentinel_source / "change/estimates.json",
                        pair_dir / "sentinel-change-estimates.json",
                    )
                    sentinel = classification.read_change(
                        pair_dir / "sentinel-change-estimates.json"
                    )
                    if classification.sentinel_breached(sentinel[0], sentinel[1]):
                        invalid_reason = "sentinel interval breaches drift band"

                monitor_path = _write_monitor_artifact(pair_dir, case, pair, runs)
                local_artifacts[monitor_path.name] = _record_artifact(
                    campaign, artifact_root, monitor_path
                )
                for estimate_name in (
                    "change-estimates.json",
                    "sentinel-change-estimates.json",
                ):
                    estimate_path = pair_dir / estimate_name
                    if estimate_path.is_file():
                        local_artifacts[estimate_name] = _record_artifact(
                            campaign, artifact_root, estimate_path
                        )
                validity = {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "case": case,
                    "pair": pair,
                    "order": order,
                    "selected_cpu": selected_cpu,
                    "allowed_cpu_count": len(allowed_cpus),
                    "validity_state": "COMPLETE" if invalid_reason is None else "INCONCLUSIVE",
                    "runs": runs,
                    "artifacts": local_artifacts,
                }
                if invalid_reason is not None:
                    validity["reason"] = invalid_reason
                validity_path = pair_dir / "validity.json"
                atomic_writer(validity_path, validity)
                validity_record = _record_artifact(campaign, artifact_root, validity_path)
                campaign["cases"][case].pop("active_pair", None)
                if invalid_reason is not None:
                    _synchronize_prefix_inventory(campaign, artifact_root)
                    terminal = _invalid_campaign(
                        campaign, reason=invalid_reason, **current
                    )
                    atomic_writer(manifest_path, terminal)
                    _close_ledger(
                        pathlib.Path(args.ledger), ledger, args, None, "INCONCLUSIVE", atomic_writer
                    )
                    return 2
                campaign["cases"][case]["pairs"][str(pair)] = {
                    "order": order,
                    "validity_path": validity_path.relative_to(artifact_root).as_posix(),
                    "validity_sha256": validity_record["sha256"],
                }
                atomic_writer(manifest_path, campaign)

        campaign["statistical_result"] = _declare_results(campaign, artifact_root)
        campaign["validity_state"] = "COMPLETE"
        campaign["completed_at"] = utc_now()
        atomic_writer(manifest_path, campaign)
        classified = classification.classify_campaign(manifest_path, artifact_root)
        campaign["classification_artifacts"] = classified["output_artifacts"]
        for record in classified["output_artifacts"].values():
            campaign["artifact_inventory"][record["path"]] = {
                "sha256": record["sha256"]
            }
        atomic_writer(manifest_path, campaign)
        verified = classification.classify_campaign(manifest_path, artifact_root)
        if verified["statistical_result"] != campaign["statistical_result"]:
            raise protocol.ProtocolError("classifier result differs after registration")
        _close_ledger(
            pathlib.Path(args.ledger),
            ledger,
            args,
            campaign["statistical_result"],
            "COMPLETE",
            atomic_writer,
        )
        return EXIT_BY_RESULT[("COMPLETE", campaign["statistical_result"])]
    except BaseException as error:
        if isinstance(error, protocol.AtomicWriteError):
            return 1
        if manifest_path is None or campaign is None or ledger is None:
            return 1
        try:
            _synchronize_prefix_inventory(campaign, pathlib.Path(args.artifact_root))
        except BaseException as inventory_error:
            error = protocol.ProtocolError(
                f"{type(error).__name__}: {error}; "
                f"prefix-inventory-error: {type(inventory_error).__name__}: {inventory_error}"
            )
        terminal = _invalid_campaign(
            campaign,
            case=current["case"],
            pair=current["pair"],
            role=current["role"],
            reason=f"{type(error).__name__}: {error}",
        )
        try:
            atomic_writer(manifest_path, terminal)
            _close_ledger(
                pathlib.Path(args.ledger), ledger, args, None, "INCONCLUSIVE", atomic_writer
            )
        except BaseException:
            return 1
        if not isinstance(error, Exception):
            raise
        return 2


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-kind", choices=protocol.LANE_NAMES, required=True
    )
    parser.add_argument("--baseline-build-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-build-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--ledger", required=True, type=pathlib.Path)
    parser.add_argument("--attempt-id", required=True, type=int)
    parser.add_argument("--artifact-root", required=True, type=pathlib.Path)
    parser.add_argument("--criterion-root", required=True, type=pathlib.Path)
    parser.add_argument(
        "--working-directory", type=pathlib.Path, default=pathlib.Path.cwd()
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--normalized-load-limit", type=float, default=0.25)
    return parser


def parse_args(argv=None):
    return build_argument_parser().parse_args(argv)


def main(argv=None) -> int:
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
