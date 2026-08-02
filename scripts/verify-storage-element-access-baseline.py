#!/usr/bin/env python3
"""Verify the immutable pre-redesign element-access benchmark report."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "tenferro.storage-element-access-baseline.v1"
BENCHMARK_PATH = "crates/tenferro-tensor/benches/element_access.rs"
REQUIRED_CASES = frozenset(
    {
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
    }
)

TOP_LEVEL_FIELDS = frozenset({"schema", "measured_commit", "benchmark", "environment", "cases"})
BENCHMARK_FIELDS = frozenset(
    {
        "path",
        "crate",
        "target",
        "profile",
        "cargo_features",
        "warm_up_seconds",
        "measurement_seconds",
        "sample_size",
        "time_unit",
    }
)
ENVIRONMENT_FIELDS = frozenset(
    {
        "rustc_version",
        "rustc_host",
        "compilation_target",
        "cargo_version",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "os",
        "architecture",
        "cpu_model",
        "logical_cpu_count",
        "cpu_affinity",
        "thread_environment",
    }
)
CASE_FIELDS = frozenset(
    {"id", "estimate_ns", "confidence_interval_ns", "standard_error_ns"}
)
INTERVAL_FIELDS = frozenset(
    {"confidence_level", "lower_bound", "upper_bound"}
)
THREAD_ENVIRONMENT_FIELDS = frozenset(
    {
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
)


class BaselineError(ValueError):
    """A report field violates the baseline contract."""


def _object(value: Any, field: str, expected_fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineError(f"{field} must be an object")
    actual = frozenset(value)
    if actual != expected_fields:
        raise BaselineError(
            f"{field} fields must be {sorted(expected_fields)}, got {sorted(actual)}"
        )
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineError(f"{field} must be a non-empty string")
    return value


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{field} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise BaselineError(f"{field} must be a finite positive number")
    return number


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise BaselineError(f"{field} must be a finite nonnegative number")
    return number


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BaselineError(f"{field} must be a positive integer")
    return value


def _canonical_commit(root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BaselineError(f"measured_commit does not resolve to a commit: {revision}")
    return result.stdout.strip()


def _verify_benchmark(root: Path, commit: str, value: Any) -> None:
    benchmark = _object(value, "benchmark", BENCHMARK_FIELDS)
    path = _string(benchmark["path"], "benchmark.path")
    expected_strings = {
        "path": BENCHMARK_PATH,
        "crate": "tenferro-tensor",
        "target": "element_access",
        "profile": "bench",
        "cargo_features": "default",
        "time_unit": "ns",
    }
    for field, expected in expected_strings.items():
        if benchmark[field] != expected:
            raise BaselineError(f"benchmark.{field} must be {expected!r}")
    warm_up_seconds = _positive_number(
        benchmark["warm_up_seconds"], "benchmark.warm_up_seconds"
    )
    measurement_seconds = _positive_number(
        benchmark["measurement_seconds"], "benchmark.measurement_seconds"
    )
    sample_size = _positive_integer(benchmark["sample_size"], "benchmark.sample_size")
    tracked = subprocess.run(
        ["git", "cat-file", "-t", f"{commit}:{path}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise BaselineError(
            f"benchmark.path is not tracked at measured_commit: {path}"
        )
    if tracked.stdout.strip() != "blob":
        raise BaselineError(f"benchmark.path must be a tracked file: {path}")


def _verify_environment(value: Any) -> None:
    environment = _object(value, "environment", ENVIRONMENT_FIELDS)
    for field in (
        "rustc_version",
        "rustc_host",
        "compilation_target",
        "cargo_version",
        "os",
        "architecture",
        "cpu_model",
    ):
        _string(environment[field], f"environment.{field}")
    for field in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
        if not isinstance(environment[field], str):
            raise BaselineError(f"environment.{field} must be a string")
    _positive_integer(environment["logical_cpu_count"], "environment.logical_cpu_count")
    affinity = environment["cpu_affinity"]
    if not isinstance(affinity, list) or not affinity:
        raise BaselineError("environment.cpu_affinity must be a non-empty integer array")
    if any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in affinity):
        raise BaselineError("environment.cpu_affinity must be a non-empty integer array")
    thread_environment = _object(
        environment["thread_environment"],
        "environment.thread_environment",
        THREAD_ENVIRONMENT_FIELDS,
    )
    for field, setting in thread_environment.items():
        if not isinstance(setting, str) or not setting.isdigit() or int(setting) <= 0:
            raise BaselineError(
                f"environment.thread_environment.{field} must be a positive integer string"
            )


def _verify_cases(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise BaselineError("cases must be a non-empty array")
    seen: set[str] = set()
    for index, raw_case in enumerate(value):
        case = _object(raw_case, f"cases[{index}]", CASE_FIELDS)
        case_id = _string(case["id"], f"cases[{index}].id")
        if case_id in seen:
            raise BaselineError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        estimate = _positive_number(case["estimate_ns"], f"cases[{index}].estimate_ns")
        _nonnegative_number(
            case["standard_error_ns"], f"cases[{index}].standard_error_ns"
        )
        interval = _object(
            case["confidence_interval_ns"],
            f"cases[{index}].confidence_interval_ns",
            INTERVAL_FIELDS,
        )
        confidence = _positive_number(
            interval["confidence_level"],
            f"cases[{index}].confidence_interval_ns.confidence_level",
        )
        lower = _positive_number(
            interval["lower_bound"],
            f"cases[{index}].confidence_interval_ns.lower_bound",
        )
        upper = _positive_number(
            interval["upper_bound"],
            f"cases[{index}].confidence_interval_ns.upper_bound",
        )
        if confidence >= 1.0:
            raise BaselineError("confidence level must be less than 1")
        if not lower <= estimate <= upper:
            raise BaselineError(
                f"case {case_id} estimate must lie inside its confidence interval"
            )
    missing = sorted(REQUIRED_CASES - seen)
    if missing:
        raise BaselineError(f"missing required benchmark cases: {missing}")


def verify_report(root: Path, report_path: Path) -> None:
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"cannot read baseline report: {error}") from error
    report = _object(raw, "report", TOP_LEVEL_FIELDS)
    if report["schema"] != SCHEMA:
        raise BaselineError(f"schema must be {SCHEMA!r}")
    measured_commit = _string(report["measured_commit"], "measured_commit")
    canonical_commit = _canonical_commit(root, measured_commit)
    if measured_commit != canonical_commit:
        raise BaselineError(
            "measured_commit must be the canonical Git object ID, not a revision alias"
        )
    _verify_benchmark(root, canonical_commit, report["benchmark"])
    _verify_environment(report["environment"])
    _verify_cases(report["cases"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_report(args.root.resolve(), args.report.resolve())
    except BaselineError as error:
        print(f"storage-element-access-baseline-error: {error}", file=sys.stderr)
        return 1
    print("storage-element-access-baseline-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
