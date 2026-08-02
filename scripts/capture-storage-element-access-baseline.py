#!/usr/bin/env python3
"""Capture the Criterion element-access baseline into the v1 report schema."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "tenferro.storage-element-access-baseline.v1"
BENCHMARK_PATH = "crates/tenferro-tensor/benches/element_access.rs"
CRITERION_ARGS = (
    "--warm-up-time",
    "2",
    "--measurement-time",
    "5",
    "--sample-size",
    "100",
    "--noplot",
)
WARM_UP_SECONDS = 2.0
MEASUREMENT_SECONDS = 5.0
SAMPLE_SIZE = 100
THREAD_VARIABLES = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
REQUIRED_CASES = (
    "element_access/2d/col_major/direct_slice/4096",
    "element_access/2d/col_major/direct_slice_mut/4096",
    "element_access/2d/col_major/get/4096",
    "element_access/2d/col_major/get_unchecked/4096",
    "element_access/2d/col_major/get_mut/4096",
    "rank_fixed/2d/col_major/get2/4096",
    "rank_fixed/3d/col_major/get3/4096",
    "linear_iteration/col_major/as_slice_iter",
    "linear_iteration/col_major/dynamic_tensor_iter",
    "linear_iteration/col_major/tensor_iter",
    "linear_iteration/col_major/dynamic_tensor_iter_mut",
    "strided_traversal/rectangular_transpose/logical_order_get/3840",
)
FILENAME_UNSAFE = set('?"/\\*<>:|^')


def _run(root: Path, command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _canonical_commit(root: Path) -> str:
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root).returncode != 0:
        raise RuntimeError("baseline capture requires a clean tracked worktree")
    return _run(root, ["git", "rev-parse", "--verify", "HEAD^{commit}"]).strip()


def _rustc_metadata(root: Path, env: dict[str, str]) -> tuple[str, str, str]:
    version = _run(root, ["rustc", "--version"]).strip()
    verbose = _run(root, ["rustc", "-vV"])
    fields = dict(
        line.split(": ", 1)
        for line in verbose.splitlines()
        if ": " in line
    )
    host = fields.get("host")
    if not host:
        raise RuntimeError("rustc -vV did not report a host target")
    return version, host, env.get("CARGO_BUILD_TARGET") or host


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def _environment(root: Path, env: dict[str, str]) -> dict[str, Any]:
    rustc_version, rustc_host, compilation_target = _rustc_metadata(root, env)
    logical_cpu_count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        cpu_affinity = sorted(os.sched_getaffinity(0))
    else:
        cpu_affinity = list(range(logical_cpu_count))
    thread_environment = {
        name: env.get(name, "") for name in THREAD_VARIABLES
    }
    for name, value in thread_environment.items():
        if not value.isdigit() or int(value) <= 0:
            raise RuntimeError(f"{name} must be a positive integer for baseline capture")
    return {
        "rustc_version": rustc_version,
        "rustc_host": rustc_host,
        "compilation_target": compilation_target,
        "os": f"{platform.system()} {platform.release()}".strip(),
        "architecture": platform.machine() or "unknown",
        "cpu_model": _cpu_model(),
        "logical_cpu_count": logical_cpu_count,
        "cpu_affinity": cpu_affinity,
        "cargo_version": _run(root, ["cargo", "--version"]).strip(),
        "RUSTFLAGS": env.get("RUSTFLAGS", ""),
        "CARGO_ENCODED_RUSTFLAGS": env.get("CARGO_ENCODED_RUSTFLAGS", ""),
        "thread_environment": thread_environment,
    }


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Criterion {field} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"Criterion {field} is not finite and positive")
    return value


def _nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Criterion {field} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"Criterion {field} is not finite and nonnegative")
    return value


def _confidence_level(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Criterion {field} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise RuntimeError(f"Criterion {field} is not a valid confidence level")
    return value


def _criterion_component(value: str) -> str:
    return "".join("_" if character in FILENAME_UNSAFE else character for character in value)[:64]


def _estimate_path(root: Path, case_id: str) -> Path:
    parts = case_id.split("/")
    if len(parts) < 2:
        raise RuntimeError(f"invalid benchmark case ID: {case_id}")
    if parts[-1].isdigit():
        if len(parts) < 3:
            raise RuntimeError(f"invalid valued benchmark case ID: {case_id}")
        group = "/".join(parts[:-2])
        function = parts[-2]
        value = parts[-1]
    else:
        group = "/".join(parts[:-1])
        function = parts[-1]
        value = None
    components = [_criterion_component(group), _criterion_component(function)]
    if value is not None:
        components.append(_criterion_component(value))
    return root / "target" / "criterion" / Path(*components) / "new" / "estimates.json"


def _estimate(root: Path, case_id: str) -> dict[str, Any]:
    estimate_path = _estimate_path(root, case_id)
    try:
        estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
        mean = estimate["mean"]
        interval = mean["confidence_interval"]
        return {
            "id": case_id,
            "estimate_ns": _positive(mean["point_estimate"], f"{case_id}.point_estimate"),
            "confidence_interval_ns": {
                "confidence_level": _confidence_level(
                    interval["confidence_level"], f"{case_id}.confidence_level"
                ),
                "lower_bound": _positive(
                    interval["lower_bound"], f"{case_id}.lower_bound"
                ),
                "upper_bound": _positive(
                    interval["upper_bound"], f"{case_id}.upper_bound"
                ),
            },
            "standard_error_ns": _nonnegative(
                mean["standard_error"], f"{case_id}.standard_error"
            ),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read Criterion estimate for {case_id}: {estimate_path}") from error


def capture(root: Path, output: Path) -> None:
    root = root.resolve()
    output = output if output.is_absolute() else root / output
    measured_commit = _canonical_commit(root)
    bench_env = os.environ.copy()
    for name in THREAD_VARIABLES:
        bench_env[name] = "1"
    command = [
        "cargo",
        "bench",
        "--locked",
        "-p",
        "tenferro-tensor",
        "--bench",
        "element_access",
        "--",
        *CRITERION_ARGS,
    ]
    _run(root, command, env=bench_env)
    cases = [_estimate(root, case_id) for case_id in REQUIRED_CASES]
    report = {
        "schema": SCHEMA,
        "measured_commit": measured_commit,
        "benchmark": {
            "path": BENCHMARK_PATH,
            "crate": "tenferro-tensor",
            "target": "element_access",
            "profile": "bench",
            "cargo_features": "default",
            "warm_up_seconds": WARM_UP_SECONDS,
            "measurement_seconds": MEASUREMENT_SECONDS,
            "sample_size": SAMPLE_SIZE,
            "time_unit": "ns",
        },
        "environment": _environment(root, bench_env),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/testing/storage-element-access-baseline.json"),
    )
    args = parser.parse_args()
    capture(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
