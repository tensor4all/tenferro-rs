#!/usr/bin/env python3
"""Run and classify the frozen incremental Householder QR performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "tenferro.incremental-householder-qr-performance.v1"
ALGORITHMS = ("compact", "bcgs2", "full-qr")
BACKENDS = ("faer", "blas", "cuda")
CYCLES = 7
BOOTSTRAP_SEED = 1735
BOOTSTRAP_SAMPLES = 10_000
SOURCE_COMMIT = "da0775a208006352f6e5eab18bc6bb09ca39a1f6"


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    initial_rank: int
    block_width: int
    max_rank: int
    repetitions: int
    algorithms: tuple[str, ...]
    kind: str


CASES = (
    Case("bond32", 2 * 32 * 32, 2, 3, 32, 5, ALGORITHMS, "primary"),
    Case("bond64", 2 * 64 * 64, 2, 3, 32, 5, ALGORITHMS, "primary"),
    Case("bond128", 2 * 128 * 128, 2, 3, 32, 3, ALGORITHMS, "primary"),
    Case("secondary-width1", 4096, 2, 1, 32, 5, ("compact", "bcgs2"), "secondary"),
    Case("secondary-width8", 4096, 2, 8, 32, 5, ("compact", "bcgs2"), "secondary"),
    Case("scaling-rank8", 32768, 8, 3, 11, 3, ("compact",), "scaling"),
    Case("scaling-rank16", 32768, 16, 3, 19, 3, ("compact",), "scaling"),
    Case("scaling-rank29", 32768, 29, 3, 32, 3, ("compact",), "scaling"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("target/iqr-performance"))
    parser.add_argument("--backend", choices=BACKENDS, action="append")
    return parser.parse_args()


def process_order(case: Case, cycle: int) -> tuple[str, ...]:
    return case.algorithms if cycle % 2 == 1 else tuple(reversed(case.algorithms))


def cargo_prefix(backend: str) -> list[str]:
    command = ["cargo", "bench", "-q", "-p", "tenferro-linalg"]
    if backend == "blas":
        command += ["--no-default-features", "--features", "cpu-blas,blas-openblas"]
    elif backend == "cuda":
        command += ["--features", "cuda"]
    command += ["--bench", "incremental_householder_qr", "--"]
    return command


def capture(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def environment_observation() -> dict[str, object]:
    load1 = float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    mhz = []
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("cpu mhz"):
            mhz.append(float(line.split(":", 1)[1]))
    gpu_clock = None
    gpu_throttle = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
        gpu_clock = float(fields[0])
        gpu_throttle = fields[1] or None
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    return {
        "load1": load1,
        "cpu_mhz": statistics.fmean(mhz) if mhz else None,
        "gpu_clock_mhz": gpu_clock,
        "gpu_throttle": gpu_throttle,
    }


def run_suite(root: Path, artifact_dir: Path, backends: list[str]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ledger = root / "docs/performance/incremental-householder-qr-bcgs2-ledger.md"
    if SOURCE_COMMIT not in ledger.read_text(encoding="utf-8"):
        raise ValueError("BCGS2 correspondence ledger is missing the pinned #694 commit")
    contract = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "tenferro-linalg",
            "--test",
            "integration",
            "incremental_qr_performance_gate_contract",
            "--",
            "--nocapture",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "source-contract.log").write_text(
        contract.stdout + contract.stderr, encoding="utf-8"
    )
    benchmark = root / "crates/tenferro-linalg/benches/incremental_householder_qr.rs"
    benchmark_sha256 = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    environment_path = artifact_dir / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "pre_run": environment_observation(),
                "git_head": git(root, "rev-parse", "HEAD"),
                "git_status": git(root, "status", "--porcelain"),
                "benchmark_sha256": benchmark_sha256,
                "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
                "release_profile": "release",
                "thread_environment": {
                    "RAYON_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                "uname": capture(["uname", "-a"]),
                "lscpu": capture(["lscpu"]),
                "numa": capture(["numactl", "--hardware"]),
                "affinity": capture(["taskset", "-pc", str(os.getpid())]),
                "nvidia_smi": capture([
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]),
                "nvcc": capture(["nvcc", "--version"]),
                "cargo_lock_sha256": hashlib.sha256((root / "Cargo.lock").read_bytes()).hexdigest(),
                "reference_medians_ms": {
                    "rust_src": [54.656, 108.898, 433.856],
                    "python_householder": [11.358, 56.101, 348.851],
                },
                "rustc": subprocess.run(
                    ["rustc", "--version", "--verbose"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "cases": [case.__dict__ for case in CASES],
                "cycles": CYCLES,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for backend in backends:
        output_path = artifact_dir / f"{backend}.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for case in CASES:
                for cycle in range(1, CYCLES + 1):
                    for order, algorithm in enumerate(process_order(case, cycle)):
                        command = cargo_prefix(backend) + [
                            "--backend",
                            backend,
                            "--algorithm",
                            algorithm,
                            "--rows",
                            str(case.rows),
                            "--initial-rank",
                            str(case.initial_rank),
                            "--block-width",
                            str(case.block_width),
                            "--max-rank",
                            str(case.max_rank),
                            "--warmups",
                            "3",
                            "--repetitions",
                            str(case.repetitions),
                            "--seed",
                            "7",
                        ]
                        env = os.environ.copy()
                        env.update(
                            RAYON_NUM_THREADS="1",
                            OPENBLAS_NUM_THREADS="1",
                            OMP_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                            CUBECL_DEBUG_LOG="0",
                        )
                        result = subprocess.run(
                            command,
                            cwd=root,
                            env=env,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        json_lines = [
                            line for line in result.stdout.splitlines() if line.startswith("{")
                        ]
                        if len(json_lines) != 1:
                            raise RuntimeError(
                                f"benchmark emitted {len(json_lines)} JSON records; "
                                f"stdout={result.stdout!r} stderr={result.stderr!r}"
                            )
                        record = json.loads(json_lines[0])
                        record.update(
                            gate_schema=SCHEMA,
                            case=case.name,
                            case_kind=case.kind,
                            cycle=cycle,
                            order=order,
                            environment=environment_observation(),
                            command=command,
                        )
                        output.write(json.dumps(record, sort_keys=True) + "\n")
                        output.flush()
                        print(f"completed backend={backend} case={case.name} cycle={cycle} algorithm={algorithm}")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_records(artifact_dir: Path, backends: list[str]) -> list[dict]:
    records = []
    for backend in backends:
        path = artifact_dir / f"{backend}.jsonl"
        if not path.exists():
            raise ValueError(f"missing artifact {path}")
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return records


def validate_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError("benchmark record must be an object")
    required = {
        "schema",
        "gate_schema",
        "backend",
        "algorithm",
        "case",
        "cycle",
        "order",
        "timings_ms",
        "repetitions",
        "reconstruction_relative_error",
        "orthogonality_relative_error",
        "r_relative_error",
        "environment",
        "command",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"benchmark record missing keys: {missing}")
    if record["schema"] != "tenferro.incremental-householder-qr-benchmark.v1":
        raise ValueError(f"unexpected benchmark schema: {record['schema']!r}")
    if record["gate_schema"] != SCHEMA:
        raise ValueError(f"unexpected gate schema: {record['gate_schema']!r}")
    if record["backend"] not in BACKENDS or record["algorithm"] not in ALGORITHMS:
        raise ValueError("invalid backend/algorithm in benchmark record")
    case_index = {case.name: case for case in CASES}
    case = case_index.get(record["case"])
    if case is None or record["algorithm"] not in case.algorithms:
        raise ValueError("invalid case/algorithm in benchmark record")
    if record["cycle"] not in range(1, CYCLES + 1):
        raise ValueError("invalid cycle in benchmark record")
    expected_order = process_order(case, int(record["cycle"]))
    if record["order"] not in range(len(expected_order)):
        raise ValueError("invalid order in benchmark record")
    if expected_order[record["order"]] != record["algorithm"]:
        raise ValueError("record algorithm does not match the frozen cycle order")
    timings = record["timings_ms"]
    if not isinstance(timings, list) or len(timings) != int(record["repetitions"]) or not timings:
        raise ValueError("timings_ms does not match the positive repetition count")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in timings):
        raise ValueError("timings_ms contains invalid values")
    if not isinstance(record["environment"], dict):
        raise ValueError("environment must be an object")
    for field in (
        "reconstruction_relative_error",
        "orthogonality_relative_error",
        "r_relative_error",
    ):
        if not math.isfinite(float(record[field])) or float(record[field]) < 0.0:
            raise ValueError(f"invalid {field}")


def median_timing(record: dict) -> float:
    return statistics.median(float(value) for value in record["timings_ms"])


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.median(sample))
    estimates.sort()
    return estimates[int(0.025 * len(estimates))], estimates[int(0.975 * len(estimates))]


def paired_ratio_ci(numerator: list[float], denominator: list[float]) -> tuple[float, float]:
    if len(numerator) != len(denominator):
        raise ValueError("paired ratio inputs have different lengths")
    ratios = [left / right for left, right in zip(numerator, denominator, strict=True)]
    return bootstrap_ci(ratios)


def observation_issues(observation: dict, pre: dict, backend: str) -> list[str]:
    issues = []
    if observation["load1"] > 1.5 * max(float(pre["load1"]), 0.1):
        issues.append("system load validity gate failed")
    if pre.get("cpu_mhz") and observation.get("cpu_mhz"):
        if abs(observation["cpu_mhz"] / pre["cpu_mhz"] - 1.0) > 0.10:
            issues.append("CPU frequency validity gate failed")
    if backend == "cuda" and observation.get("gpu_throttle") not in (
        None,
        "0x0000000000000000",
    ):
        issues.append("CUDA throttle reason active")
    if backend == "cuda" and pre.get("gpu_clock_mhz") and observation.get("gpu_clock_mhz"):
        if abs(observation["gpu_clock_mhz"] / pre["gpu_clock_mhz"] - 1.0) > 0.10:
            issues.append("GPU clock validity gate failed")
    return issues


def check_suite(artifact_dir: Path, backends: list[str]) -> int:
    environment = json.loads((artifact_dir / "environment.json").read_text(encoding="utf-8"))
    if not isinstance(environment, dict) or environment.get("schema") != SCHEMA:
        raise ValueError("unexpected or missing environment schema")
    if environment["git_status"]:
        raise ValueError("performance run used a dirty worktree")
    records = load_records(artifact_dir, backends)
    for record in records:
        validate_record(record)
    findings: list[str] = []
    inconclusive: list[str] = []
    summaries: dict[tuple[str, str, str], dict[str, object]] = {}
    ratios: dict[str, dict[str, object]] = {}
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "crates/tenferro-linalg/benches/incremental_householder_qr.rs"
    ledger = root / "docs/performance/incremental-householder-qr-bcgs2-ledger.md"
    if git(root, "rev-parse", "HEAD") != environment["git_head"]:
        findings.append("current HEAD differs from the measured candidate")
    if git(root, "status", "--porcelain"):
        findings.append("current worktree is dirty")
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != environment["benchmark_sha256"]:
        findings.append("benchmark source hash differs from the measured candidate")
    if hashlib.sha256(ledger.read_bytes()).hexdigest() != environment["ledger_sha256"]:
        findings.append("BCGS2 ledger hash differs from the measured candidate")
    if not (artifact_dir / "source-contract.log").exists():
        findings.append("missing exact-candidate source-contract log")
    pre = environment["pre_run"]
    for backend in backends:
        for case in CASES:
            for algorithm in case.algorithms:
                selected = [
                    record
                    for record in records
                    if record["backend"] == backend
                    and record["case"] == case.name
                    and record["algorithm"] == algorithm
                ]
                label = f"{backend}/{case.name}/{algorithm}"
                if len(selected) != CYCLES or {r["cycle"] for r in selected} != set(range(1, 8)):
                    findings.append(f"{label}: expected seven complete cycles")
                    continue
                medians = [median_timing(record) for record in selected]
                median = statistics.median(medians)
                cv = statistics.pstdev(medians) / statistics.fmean(medians)
                ci = bootstrap_ci(medians)
                summaries[(backend, case.name, algorithm)] = {
                    "median_ms": median,
                    "process_medians_ms": medians,
                    "ci95_ms": ci,
                    "cv": cv,
                }
                if any(value < 1.0 for value in medians):
                    inconclusive.append(f"{label}: a process median is below 1 ms")
                if cv > 0.10:
                    inconclusive.append(f"{label}: process-median CoV {cv:.3f} exceeds 0.10")
                tolerance = 2.0e-9 if backend == "cuda" else 5.0e-11
                for record in selected:
                    environment_issues = observation_issues(record["environment"], pre, backend)
                    inconclusive.extend(f"{label}: {issue}" for issue in environment_issues)
                    correctness = []
                    if record["reconstruction_relative_error"] > tolerance:
                        correctness.append(f"reconstruction error exceeded {tolerance}")
                    if record["orthogonality_relative_error"] > tolerance:
                        correctness.append(f"orthogonality error exceeded {tolerance}")
                    if algorithm == "compact" and record["r_relative_error"] > 1.0e-9:
                        correctness.append("R agreement exceeded 1e-9")
                    target = inconclusive if environment_issues else findings
                    target.extend(f"{label}: {issue}" for issue in correctness)
    for backend in backends:
        required = [
            (backend, f"bond{bond}", algorithm)
            for bond in (32, 64, 128)
            for algorithm in ALGORITHMS
        ] + [
            (backend, f"scaling-rank{rank}", "compact")
            for rank in (8, 16, 29)
        ]
        missing = [key for key in required if key not in summaries]
        if missing:
            findings.append(f"{backend}: performance gates missing summaries {missing}")
            continue
        for bond in (32, 64, 128):
            case = f"bond{bond}"
            compact = summaries[(backend, case, "compact")]["median_ms"]
            bcgs2 = summaries[(backend, case, "bcgs2")]["median_ms"]
            limit = 1.15 if bond == 32 else 1.0
            compact_values = summaries[(backend, case, "compact")]["process_medians_ms"]
            bcgs2_values = summaries[(backend, case, "bcgs2")]["process_medians_ms"]
            paired = [left / right for left, right in zip(compact_values, bcgs2_values, strict=True)]
            ratios[f"{backend}/{case}/compact-over-bcgs2"] = {
                "median": statistics.median(paired),
                "ci95": paired_ratio_ci(compact_values, bcgs2_values),
                "values": paired,
            }
            if not (backend == "cuda" and bond == 32) and compact > limit * bcgs2:
                findings.append(f"{backend}/{case}: compact exceeded {limit:.2f}x BCGS2")
        improvements = []
        for bond in (64, 128):
            case = f"bond{bond}"
            compact = summaries[(backend, case, "compact")]["median_ms"]
            bcgs2 = summaries[(backend, case, "bcgs2")]["median_ms"]
            improvements.append(1.0 - compact / bcgs2)
        if backend != "cuda" and max(improvements) < 0.05:
            findings.append(f"{backend}: no >=5% compact improvement at bond 64/128")
        compact = summaries[(backend, "bond128", "compact")]["median_ms"]
        full = summaries[(backend, "bond128", "full-qr")]["median_ms"]
        compact_values = summaries[(backend, "bond128", "compact")]["process_medians_ms"]
        full_values = summaries[(backend, "bond128", "full-qr")]["process_medians_ms"]
        paired_full = [left / right for left, right in zip(compact_values, full_values, strict=True)]
        ratios[f"{backend}/bond128/compact-over-full-qr"] = {
            "median": statistics.median(paired_full),
            "ci95": paired_ratio_ci(compact_values, full_values),
            "values": paired_full,
        }
        if compact > 0.5 * full:
            findings.append(f"{backend}/bond128: compact was not 2x faster than full QR")
        proxies = []
        for rank in (8, 16, 29):
            timing = summaries[(backend, f"scaling-rank{rank}", "compact")]["median_ms"]
            proxies.append(timing / (32768 * rank * 3))
        if max(proxies) / min(proxies) > 1.35:
            findings.append(f"{backend}: normalized append scaling exceeded 35%")
    report = {
        "schema": SCHEMA,
        "verdict": "FAIL" if findings else ("INCONCLUSIVE" if inconclusive else "PASS"),
        "findings": sorted(set(findings)),
        "inconclusive": sorted(set(inconclusive)),
        "summaries": {"/".join(key): value for key, value in sorted(summaries.items())},
        "ratios": ratios,
    }
    (artifact_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


def self_test() -> None:
    assert [case.name for case in CASES] == [
        "bond32",
        "bond64",
        "bond128",
        "secondary-width1",
        "secondary-width8",
        "scaling-rank8",
        "scaling-rank16",
        "scaling-rank29",
    ]
    assert process_order(CASES[0], 1) == ALGORITHMS
    assert process_order(CASES[0], 2) == tuple(reversed(ALGORITHMS))
    width8 = next(case for case in CASES if case.name == "secondary-width8")
    assert width8.initial_rank + ((width8.max_rank - width8.initial_rank) // 8) * 8 == 26
    assert next(case for case in CASES if case.name == "scaling-rank29").rows == 32768
    assert SOURCE_COMMIT == "da0775a208006352f6e5eab18bc6bb09ca39a1f6"
    low, high = bootstrap_ci([1.0] * 7)
    assert (low, high) == (1.0, 1.0)
    assert paired_ratio_ci([2.0] * 7, [4.0] * 7) == (0.5, 0.5)
    print("incremental-householder-qr-performance-self-test-ok")


def main() -> int:
    args = parse_args()
    backends = args.backend or list(BACKENDS)
    root = Path(__file__).resolve().parents[1]
    if args.self_test:
        self_test()
        return 0
    if args.run:
        run_suite(root, args.artifact_dir.resolve(), backends)
        return 0
    artifact_dir = args.artifact_dir.resolve()
    try:
        return check_suite(artifact_dir, backends)
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "verdict": "FAIL",
            "findings": [f"invalid or incomplete artifact: {type(error).__name__}: {error}"],
            "inconclusive": [],
            "summaries": {},
            "ratios": {},
        }
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
