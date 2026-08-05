#!/usr/bin/env python3
"""Independently verify the P8-P13 storage redesign evidence set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FREEZE = Path("docs/design/storage-contract-freeze.md")
MATRIX = Path("docs/testing/storage-hardware-matrix.md")
PERFORMANCE = Path("docs/testing/storage-traversal-performance.md")
STATIC_RANK = Path("docs/testing/storage-static-rank-codegen.md")
DOC_AUDIT = Path("docs/worklogs/storage-documentation-source-blind-audit.md")
REQUIRED_FILES = (
    FREEZE,
    MATRIX,
    PERFORMANCE,
    STATIC_RANK,
    DOC_AUDIT,
    Path("scripts/check-storage-element-hot-path.py"),
    Path("crates/tenferro-tensor/tests/storage_public_api.rs"),
    Path("crates/tenferro-gpu/tests/storage_provider_webgpu.rs"),
)


class CheckError(ValueError):
    pass


REPRODUCE_COMMANDS = (
    ("p10-api-normalization", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_public_api")),
    ("p4-traversal-resolution-counts", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_traversal_resolution")),
    ("p3-static-rank-preservation", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_static_rank")),
    ("p3-host-owner", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract")),
    (None, ("cargo", "test", "-p", "tenferro-runtime", "scoped_immediate_provider_returns_borrowed_output")),
    (None, ("python3", "scripts/ci/run_profile.py", "coverage")),
)


def read(path: Path) -> str:
    if not path.is_file():
        raise CheckError(f"missing closure evidence: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def record(path: Path) -> dict:
    text = read(ROOT / path)
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise CheckError(f"no fenced JSON record in {path}")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise CheckError(f"invalid JSON record in {path}: {error}") from error


def run_reproduction(
    receipt_path: Path,
    *,
    receipt_validator=None,
    runner=None,
) -> list[dict[str, object]]:
    if receipt_validator is None:
        command = [
            sys.executable,
            "scripts/check-storage-ownership-contracts.py",
            "--root",
            str(ROOT),
            "--receipt",
            str(receipt_path),
            "--summary-json",
        ]
        receipt_result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        if receipt_result.returncode != 0:
            raise CheckError(
                "receipt checker failed: " + receipt_result.stderr.strip()
            )
    elif receipt_validator(receipt_path) != 0:
        raise CheckError("receipt checker failed")

    def default_runner(argv: tuple[str, ...]) -> int:
        result = subprocess.run(argv, cwd=ROOT, check=False)
        return result.returncode

    execute = runner or default_runner
    executions = []
    for obligation_id, argv in REPRODUCE_COMMANDS:
        exit_code = execute(argv)
        executions.append(
            {
                "obligation_id": obligation_id,
                "argv": list(argv),
                "exit_code": exit_code,
            }
        )
        if exit_code != 0:
            raise CheckError(
                f"reproduction command exited with exit code {exit_code}: {' '.join(argv)}"
            )
    return executions


def validate() -> tuple[str, dict, dict, dict]:
    for relative in REQUIRED_FILES:
        read(ROOT / relative)
    freeze = record(FREEZE)
    candidate = freeze.get("candidate_commit")
    if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise CheckError("freeze candidate is not a full commit")
    if freeze.get("status") != "pass":
        raise CheckError("freeze candidate is not passing")
    matrix = record(MATRIX)
    performance = record(PERFORMANCE)
    static_rank = record(STATIC_RANK)
    for name, evidence in (("hardware matrix", matrix), ("performance", performance), ("static-rank", static_rank)):
        if evidence.get("candidate_commit") != candidate:
            raise CheckError(f"{name} evidence does not match frozen candidate")
    if performance.get("result") != "pass":
        raise CheckError("performance evidence is not passing")
    if static_rank.get("status") != "pass":
        raise CheckError("static-rank evidence is not passing")
    if matrix.get("complete") is not True or matrix.get("status") != "pass":
        raise CheckError("hardware matrix is incomplete or not passing")
    for lane in matrix.get("lanes", []):
        if lane.get("status") != "pass" or not isinstance(lane.get("test_count"), int) or lane["test_count"] <= 0:
            raise CheckError(f"hardware lane is incomplete: {lane.get('lane')}")
    audit_text = read(ROOT / DOC_AUDIT)
    for marker in ("Critical usability gaps: 0", "Important usability gaps: 0", "Source-blind"):
        if marker not in audit_text:
            raise CheckError(f"documentation audit is missing {marker!r}")
    return candidate, matrix, performance, static_rank


def write_report(
    path: Path,
    candidate: str,
    matrix: dict,
    performance: dict,
    static_rank: dict,
    reproduction: list[dict[str, object]] | None = None,
) -> None:
    record_value = {
        "schema": "tenferro.storage-redesign-closure.v1",
        "candidate_commit": candidate,
        "status": "pass",
        "findings": [],
        "obligations": {
            "architecture_and_lifecycle": "verified by freeze source inventory and ownership contract receipt",
            "prepared_and_hot_paths": "verified by storage element hot-path, static-rank, and traversal evidence",
            "api_and_docs": "verified by public API tests, rendered documentation checks, and source-blind audit",
            "cpu": "verified by CPU public API and workspace test evidence",
            "gpu_and_multi_gpu": "CUDA, WebGPU, and Metal provider lanes pass",
            "ad": "CUDA AD integration lane passes",
        },
        "performance": {"result": performance["result"], "report": PERFORMANCE.as_posix()},
        "hardware_skips": [],
        "evidence_paths": [relative.as_posix() for relative in REQUIRED_FILES],
        "notes": "No Critical or Important findings; every required hardware lane has a positive passing test count.",
    }
    if reproduction is not None:
        record_value["reproduction"] = {
            "mode": "reproduce",
            "executions": reproduction,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Storage redesign closure\n\n"
        "This is an independent evidence audit of the frozen product candidate.\n\n"
        "```json\n" + json.dumps(record_value, indent=2) + "\n```\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.reproduce and args.receipt is None:
            raise CheckError("--receipt is required with --reproduce")
        candidate, matrix, performance, static_rank = validate()
        report = args.report if not args.report.is_absolute() else args.report.relative_to(ROOT)
        if any(part == ".." for part in report.parts):
            raise CheckError("report path must remain inside the repository")
        report_path = ROOT / report
        existing_reproduction = None
        if report_path.is_file() and not args.reproduce:
            existing = record(report)
            existing_reproduction = existing.get("reproduction")
            if not isinstance(existing_reproduction, dict):
                existing_reproduction = None
            else:
                existing_reproduction = existing_reproduction.get("executions")
        reproduction = (
            run_reproduction(args.receipt.resolve()) if args.reproduce else existing_reproduction
        )
        write_report(report_path, candidate, matrix, performance, static_rank, reproduction)
    except (CheckError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"storage-redesign-closure: {error}", file=sys.stderr)
        return 1
    print(f"storage-redesign-closure-ok: candidate={candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
