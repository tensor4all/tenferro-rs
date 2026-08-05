#!/usr/bin/env python3
"""Measure the storage traversal benchmark and compare stable cases to baseline."""
from __future__ import annotations

import argparse
import json
import os
import re
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREADS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CASES = {
    "contiguous_read": "linear_iteration/col_major/as_slice_iter",
    "contiguous_write": "linear_iteration/col_major/tensor_iter_mut",
    "dynamic_contiguous": "linear_iteration/col_major/dynamic_tensor_iter",
    "fixed_rank": "rank_fixed/2d/col_major/get2/4096",
    "strided": "strided_traversal/rectangular_transpose/logical_order_get/3840",
    "empty": "linear_iteration/col_major/empty",
}
BASELINE_CASES = {
    "contiguous_read": "linear_iteration/col_major/as_slice_iter",
    "dynamic_contiguous": "linear_iteration/col_major/dynamic_tensor_iter",
    "fixed_rank": "rank_fixed/2d/col_major/get2/4096",
    "strided": "strided_traversal/rectangular_transpose/logical_order_get/3840",
}


def run(command: list[str], env: dict[str, str] | None = None) -> str:
    return subprocess.run(command, cwd=ROOT, env=env, check=True, text=True, capture_output=True).stdout.strip()


def criterion_path(case_id: str) -> Path:
    parts = case_id.split("/")
    if parts[-1].isdigit():
        group, function, value = "/".join(parts[:-2]), parts[-2], parts[-1]
        components = [group.replace("/", "_"), function, value]
    else:
        components = ["/".join(parts[:-1]).replace("/", "_"), parts[-1]]
    return ROOT / "target" / "criterion" / Path(*components) / "new" / "estimates.json"


def estimate(case_id: str) -> dict[str, float]:
    data = json.loads(criterion_path(case_id).read_text(encoding="utf-8"))
    mean = data["mean"]
    interval = mean["confidence_interval"]
    return {
        "estimate_ns": float(mean["point_estimate"]),
        "lower_bound_ns": float(interval["lower_bound"]),
        "upper_bound_ns": float(interval["upper_bound"]),
        "standard_error_ns": float(mean["standard_error"]),
    }


def environment(env: dict[str, str]) -> dict[str, object]:
    verbose = run(["rustc", "-vV"])
    fields = dict(line.split(": ", 1) for line in verbose.splitlines() if ": " in line)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    return {
        "rustc_version": run(["rustc", "--version"]),
        "cargo_version": run(["cargo", "--version"]),
        "target": fields.get("host", platform.machine()),
        "architecture": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.lower().startswith("model name")), platform.processor()),
        "cpu_affinity": affinity,
        "thread_environment": {name: env[name] for name in THREADS},
    }


def frozen_candidate() -> str:
    text = (ROOT / "docs/design/storage-contract-freeze.md").read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise ValueError("freeze report has no JSON record")
    record = json.loads(match.group(1))
    candidate = record.get("candidate_commit")
    if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise ValueError("freeze report has no full candidate commit")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-obligation", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = args.report if args.report.is_absolute() else ROOT / args.report
    if output.is_file():
        text = output.read_text(encoding="utf-8")
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            raise ValueError("existing performance report has no JSON record")
        saved = json.loads(match.group(1))
        if saved.get("candidate_commit") != frozen_candidate():
            raise ValueError("existing performance report does not match frozen candidate")
        if saved.get("result") != "pass":
            raise ValueError("existing performance report is not passing")
        print(json.dumps(saved, indent=2))
        return 0
    env = os.environ.copy()
    for name in THREADS:
        env[name] = "1"
    command = ["cargo", "bench", "--locked", "-p", "tenferro-tensor", "--bench", "element_access", "--", "--warm-up-time", "2", "--measurement-time", "5", "--sample-size", "100", "--noplot"]
    run(command, env)
    current = {name: {"id": case_id, **estimate(case_id)} for name, case_id in CASES.items()}
    baseline = json.loads((ROOT / args.baseline_report).read_text(encoding="utf-8"))
    baseline_by_id = {case["id"]: case for case in baseline["cases"]}
    comparisons = {}
    failures = []
    for name, case_id in BASELINE_CASES.items():
        old = baseline_by_id[case_id]
        now = current[name]
        ratio = now["estimate_ns"] / float(old["estimate_ns"])
        limit = 1.10 if name == "contiguous_read" else 1.15
        comparisons[name] = {"baseline_ns": old["estimate_ns"], "current_ns": now["estimate_ns"], "ratio": ratio, "limit": limit}
        if ratio > limit:
            failures.append(f"{name} is {ratio:.3f}x baseline (limit {limit:.2f}x)")
    result = {
        "schema": "tenferro.storage-traversal-performance.v1",
        "candidate_commit": run(["git", "rev-parse", "HEAD"]),
        "benchmark_path": "crates/tenferro-tensor/benches/element_access.rs",
        "baseline_obligation": args.baseline_obligation,
        "baseline_report": str(args.baseline_report),
        "baseline_measured_commit": baseline["measured_commit"],
        "command": " ".join(command),
        "environment": environment(env),
        "sample_size": 100,
        "warm_up_seconds": 2.0,
        "measurement_seconds": 5.0,
        "medians_ns": current,
        "comparisons": comparisons,
        "setup_measurements": {"prepare_map_bind_dispatch": "not part of element_access benchmark; covered by storage preparation contracts"},
        "result": "inconclusive" if failures else "pass",
        "reason": "; ".join(failures) if failures else "comparable traversal cases satisfy the documented limits; new write/empty cases are recorded without a historical baseline",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# Storage traversal performance\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
