#!/usr/bin/env python3
"""Run the Phase 1 eager non-inferiority campaign with validity evidence."""

import argparse
import datetime
import json
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import time

import classify_criterion_noninferiority as classification


CANONICAL_CASES = classification.CANONICAL_CASES
PAIR_ORDERS = classification.PAIR_ORDERS
RUN_ROLES = classification.RUN_ROLES
THREAD_ENVIRONMENT = classification.THREAD_ENVIRONMENT
SENTINEL_BENCHMARK = CANONICAL_CASES["lazy_neg_1"]


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_cpu_list(value):
    cpus = set()
    for component in value.split(","):
        component = component.strip()
        if not component:
            continue
        if "-" in component:
            first, last = (int(item) for item in component.split("-", 1))
            if last < first:
                raise ValueError(f"invalid CPU range: {component}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(component))
    if not cpus:
        raise ValueError("CPU list must not be empty")
    return cpus


def format_cpu_list(cpus):
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


def criterion_directory(criterion_root, benchmark):
    components = benchmark.split("/")
    if len(components) != 4:
        raise ValueError(f"unexpected benchmark identifier: {benchmark}")
    group = f"{components[0]}_{components[1]}"
    return criterion_root / group / components[2] / components[3]


def run_identities(order):
    if order == "A/B":
        targets = ("baseline", "candidate")
    elif order == "B/A":
        targets = ("candidate", "baseline")
    else:
        raise ValueError(f"unsupported pair order: {order}")
    return ["candidate", targets[0], targets[1], "candidate"]


def benchmark_command(binary, benchmark, comparison_option, comparison_name):
    return [
        str(binary),
        "--bench",
        benchmark,
        comparison_option,
        comparison_name,
        "--noplot",
    ]


def exact_build_processes():
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


def normalized_load(allowed_cpu_count):
    return os.getloadavg()[0] / allowed_cpu_count


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as destination:
            json.dump(value, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha_record(path):
    return {"sha256": classification.file_sha256(path)}


def monitor_benchmark(
    binary,
    benchmark,
    comparison_option,
    comparison_name,
    role,
    binary_identity,
    binary_sha,
    selected_cpu,
    allowed_cpu_count,
    working_directory,
    log_path,
):
    command = benchmark_command(binary, benchmark, comparison_option, comparison_name)
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    start_load = normalized_load(allowed_cpu_count)
    violations = []
    overlaps = exact_build_processes()
    if overlaps:
        violations.append(f"cargo/rustc overlap at start: {overlaps!r}")
    affinities = set()
    started_at = utc_now()
    status = None
    completed = False

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.sched_setaffinity(0, {selected_cpu}),
            text=True,
        )
        try:
            while True:
                try:
                    affinity = os.sched_getaffinity(process.pid)
                    affinities.add(format_cpu_list(affinity))
                    if affinity != {selected_cpu}:
                        violations.append(
                            f"affinity mismatch: {format_cpu_list(affinity)}"
                        )
                except (ProcessLookupError, PermissionError):
                    pass
                overlaps = exact_build_processes()
                if overlaps:
                    violations.append(f"cargo/rustc overlap: {overlaps!r}")
                status = process.poll()
                if status is not None:
                    completed = True
                    break
                time.sleep(0.25)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

    end_overlaps = exact_build_processes()
    if end_overlaps:
        violations.append(f"cargo/rustc overlap at end: {end_overlaps!r}")
    end_load = normalized_load(allowed_cpu_count)
    affinity_text = str(selected_cpu) if affinities == {str(selected_cpu)} else ",".join(
        sorted(affinities)
    )
    return {
        "role": role,
        "binary": binary_identity,
        "binary_sha256": binary_sha,
        "completed": completed,
        "exit_status": status,
        "monitor_violations": sorted(set(violations)),
        "observed_affinity": affinity_text,
        "affinity_samples": sorted(affinities),
        "normalized_load_start": start_load,
        "normalized_load_end": end_load,
        "started_at": started_at,
        "ended_at": utc_now(),
        "command": command,
    }


def copy_estimate(source, destination):
    if not source.is_file():
        raise FileNotFoundError(f"Criterion did not produce {source}")
    shutil.copy2(source, destination)


def runs_are_valid(runs, selected_cpu, load_limit, order, binary_shas):
    if len(runs) != 4:
        return False, "pair does not contain four runs"
    try:
        for run, role, identity in zip(runs, RUN_ROLES, run_identities(order)):
            classification.validate_run(
                run,
                role,
                selected_cpu,
                load_limit,
                "attempt",
                identity,
                binary_shas[identity],
            )
    except ValueError as error:
        return False, str(error)
    return True, "valid"


def run_pair(
    *,
    case,
    benchmark,
    pair,
    order,
    attempt,
    binaries,
    binary_shas,
    selected_cpu,
    allowed_cpu_count,
    load_limit,
    working_directory,
    criterion_root,
    artifact_root,
):
    attempt_root = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{case}-pair{pair}-attempt{attempt}-", dir=artifact_root)
    )
    target_name = f"phase1-target-{case}-p{pair}-a{attempt}"
    sentinel_name = f"phase1-sentinel-{case}-p{pair}-a{attempt}"
    identities = run_identities(order)
    run_specs = (
        (
            "sentinel_before",
            identities[0],
            SENTINEL_BENCHMARK,
            "--save-baseline",
            sentinel_name,
        ),
        (
            "target_first",
            identities[1],
            benchmark,
            "--save-baseline",
            target_name,
        ),
        (
            "target_second",
            identities[2],
            benchmark,
            "--baseline",
            target_name,
        ),
        (
            "sentinel_after",
            identities[3],
            SENTINEL_BENCHMARK,
            "--baseline",
            sentinel_name,
        ),
    )
    runs = []
    artifact_error = None
    for role, identity, run_benchmark_name, option, name in run_specs:
        print(
            f"RUN case={case} pair={pair} attempt={attempt} role={role} "
            f"binary={identity}",
            flush=True,
        )
        record = monitor_benchmark(
            binaries[identity],
            run_benchmark_name,
            option,
            name,
            role,
            identity,
            binary_shas[identity],
            selected_cpu,
            allowed_cpu_count,
            working_directory,
            attempt_root / f"{role}.log",
        )
        runs.append(record)
        try:
            if role == "target_second":
                source = criterion_directory(criterion_root, benchmark)
                copy_estimate(
                    source / target_name / "estimates.json",
                    attempt_root / "target-first-estimates.json",
                )
                copy_estimate(
                    source / "new" / "estimates.json",
                    attempt_root / "target-second-estimates.json",
                )
                copy_estimate(
                    source / "change" / "estimates.json",
                    attempt_root / "change-estimates.json",
                )
            elif role == "sentinel_after":
                source = criterion_directory(criterion_root, SENTINEL_BENCHMARK)
                copy_estimate(
                    source / sentinel_name / "estimates.json",
                    attempt_root / "sentinel-first-estimates.json",
                )
                copy_estimate(
                    source / "new" / "estimates.json",
                    attempt_root / "sentinel-second-estimates.json",
                )
                copy_estimate(
                    source / "change" / "estimates.json",
                    attempt_root / "sentinel-change-estimates.json",
                )
        except FileNotFoundError as error:
            artifact_error = str(error)

    valid, reason = runs_are_valid(
        runs, selected_cpu, load_limit, order, binary_shas
    )
    target_change = attempt_root / "change-estimates.json"
    sentinel_change = attempt_root / "sentinel-change-estimates.json"
    if artifact_error is not None:
        valid, reason = False, artifact_error
    elif not target_change.is_file() or not sentinel_change.is_file():
        valid, reason = False, "target or sentinel change estimate is missing"
    else:
        sentinel = classification.read_change(sentinel_change)
        if classification.sentinel_breached(sentinel[0], sentinel[1]):
            valid, reason = False, "sentinel interval breaches drift band"

    validity = {
        "protocol_version": classification.PROTOCOL_VERSION,
        "case": case,
        "pair": pair,
        "order": order,
        "attempt": attempt,
        "selected_cpu": selected_cpu,
        "allowed_cpu_count": allowed_cpu_count,
        "valid": valid,
        "reason": reason,
        "runs": runs,
        "artifacts": {},
    }
    if target_change.is_file():
        validity["artifacts"][target_change.name] = sha_record(target_change)
    if sentinel_change.is_file():
        validity["artifacts"][sentinel_change.name] = sha_record(sentinel_change)
    atomic_write_json(attempt_root / "validity.json", validity)
    return attempt_root, validity


def wait_for_quiet_host(allowed_cpu_count, load_limit):
    while True:
        load = normalized_load(allowed_cpu_count)
        processes = exact_build_processes()
        if load <= load_limit and not processes:
            return
        print(
            f"WAIT normalized_load={load:.3f} build_processes={processes!r}",
            flush=True,
        )
        time.sleep(5)


def make_campaign(args, binaries, binary_shas, selected_cpu, allowed_cpus):
    lock_sha = classification.file_sha256(args.lock_file)
    if args.expected_lock_sha is not None and lock_sha != args.expected_lock_sha:
        raise ValueError(
            f"lock SHA mismatch: expected {args.expected_lock_sha}, observed {lock_sha}"
        )
    cases = {
        case: {"benchmark": benchmark, "pairs": {}}
        for case, benchmark in CANONICAL_CASES.items()
    }
    return {
        "protocol_version": classification.PROTOCOL_VERSION,
        "started_at": utc_now(),
        "lock_file": str(args.lock_file),
        "lock_sha256": lock_sha,
        "binaries": {
            "baseline": {
                "path": str(binaries["baseline"]),
                "sha256": binary_shas["baseline"],
                "source_revision": args.baseline_revision,
            },
            "candidate": {
                "path": str(binaries["candidate"]),
                "sha256": binary_shas["candidate"],
                "source_revision": args.candidate_revision,
            },
        },
        "selected_cpu": selected_cpu,
        "allowed_cpus": format_cpu_list(allowed_cpus),
        "allowed_cpu_count": len(allowed_cpus),
        "normalized_load_limit": args.normalized_load_limit,
        "thread_environment": THREAD_ENVIRONMENT,
        "orders": list(PAIR_ORDERS),
        "criterion": classification.CRITERION_SETTINGS,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "cases": cases,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-binary", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-binary", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--lock-file", required=True, type=pathlib.Path)
    parser.add_argument("--expected-lock-sha")
    parser.add_argument("--artifact-root", required=True, type=pathlib.Path)
    parser.add_argument("--working-directory", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--criterion-root", type=pathlib.Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--normalized-load-limit", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.normalized_load_limit != 0.25:
        raise ValueError("the predeclared normalized load limit is exactly 0.25")
    args.working_directory = args.working_directory.resolve()
    args.lock_file = args.lock_file.resolve()
    args.artifact_root = args.artifact_root.resolve()
    if args.criterion_root is None:
        args.criterion_root = args.working_directory / "target/criterion"
    else:
        args.criterion_root = args.criterion_root.resolve()
    binaries = {
        "baseline": args.baseline_binary.resolve(),
        "candidate": args.candidate_binary.resolve(),
    }
    for identity, binary in binaries.items():
        if not binary.is_file():
            raise FileNotFoundError(f"missing {identity} binary: {binary}")
    if not args.lock_file.is_file():
        raise FileNotFoundError(f"missing common lock file: {args.lock_file}")
    if args.artifact_root.exists() and any(args.artifact_root.iterdir()):
        raise ValueError(f"artifact root is not empty: {args.artifact_root}")
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    allowed_cpus = set(os.sched_getaffinity(0))
    selected_cpu = min(allowed_cpus) if args.cpu is None else args.cpu
    if selected_cpu not in allowed_cpus:
        raise ValueError(f"selected CPU {selected_cpu} is not process-allowed")
    binary_shas = {
        identity: classification.file_sha256(binary)
        for identity, binary in binaries.items()
    }
    campaign = make_campaign(args, binaries, binary_shas, selected_cpu, allowed_cpus)
    manifest_path = args.artifact_root / "campaign.json"
    atomic_write_json(manifest_path, campaign)

    print(
        f"START cases={len(CANONICAL_CASES)} cpu={selected_cpu} "
        f"allowed={format_cpu_list(allowed_cpus)} lock={campaign['lock_sha256']}",
        flush=True,
    )
    for case in sorted(CANONICAL_CASES):
        benchmark = CANONICAL_CASES[case]
        for pair, order in enumerate(PAIR_ORDERS, start=1):
            accepted = False
            for attempt in range(1, args.max_attempts + 1):
                wait_for_quiet_host(len(allowed_cpus), args.normalized_load_limit)
                attempt_root, validity = run_pair(
                    case=case,
                    benchmark=benchmark,
                    pair=pair,
                    order=order,
                    attempt=attempt,
                    binaries=binaries,
                    binary_shas=binary_shas,
                    selected_cpu=selected_cpu,
                    allowed_cpu_count=len(allowed_cpus),
                    load_limit=args.normalized_load_limit,
                    working_directory=args.working_directory,
                    criterion_root=args.criterion_root,
                    artifact_root=args.artifact_root,
                )
                if validity["valid"]:
                    pair_dir = args.artifact_root / case / f"pair{pair}"
                    pair_dir.parent.mkdir(parents=True, exist_ok=True)
                    attempt_root.rename(pair_dir)
                    validity_path = pair_dir / "validity.json"
                    campaign["cases"][case]["pairs"][str(pair)] = {
                        "order": order,
                        "validity": f"{case}/pair{pair}/validity.json",
                        "validity_sha256": classification.file_sha256(validity_path),
                    }
                    atomic_write_json(manifest_path, campaign)
                    print(
                        f"ACCEPT case={case} pair={pair} attempt={attempt}",
                        flush=True,
                    )
                    accepted = True
                    break
                rejected = (
                    args.artifact_root
                    / "_rejected"
                    / case
                    / f"pair{pair}"
                    / f"attempt{attempt}"
                )
                rejected.parent.mkdir(parents=True, exist_ok=True)
                attempt_root.rename(rejected)
                print(
                    f"REJECT case={case} pair={pair} attempt={attempt} "
                    f"reason={validity['reason']}",
                    flush=True,
                )
            if not accepted:
                raise RuntimeError(
                    f"could not obtain a valid {case}/pair{pair} in "
                    f"{args.max_attempts} attempts"
                )

    campaign["completed_at"] = utc_now()
    atomic_write_json(manifest_path, campaign)
    cases = classification.load_validated_campaign(args.artifact_root)
    print(classification.render_markdown(cases), flush=True)


if __name__ == "__main__":
    main()
