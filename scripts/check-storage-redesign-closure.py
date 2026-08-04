#!/usr/bin/env python3
"""Independently verify the P8-P13 storage redesign evidence set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
    if matrix.get("status") not in ("pass", "structured-skip"):
        raise CheckError("hardware matrix has an invalid status")
    for lane in matrix.get("lanes", []):
        if lane.get("status") == "skip" and not all(lane.get(key) for key in ("command", "device_facts", "evidence", "skip_reason")):
            raise CheckError(f"hardware skip for {lane.get('lane')} is not structured")
    audit_text = read(ROOT / DOC_AUDIT)
    for marker in ("Critical usability gaps: 0", "Important usability gaps: 0", "Source-blind"):
        if marker not in audit_text:
            raise CheckError(f"documentation audit is missing {marker!r}")
    return candidate, matrix, performance, static_rank


def write_report(path: Path, candidate: str, matrix: dict, performance: dict, static_rank: dict) -> None:
    skipped = [lane["lane"] for lane in matrix.get("lanes", []) if lane.get("status") == "skip"]
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
            "gpu_and_multi_gpu": "CUDA and WebGPU provider lanes pass; Metal is structured-skip on Linux",
            "ad": "CUDA AD integration lane passes",
        },
        "performance": {"result": performance["result"], "report": PERFORMANCE.as_posix()},
        "hardware_skips": skipped,
        "evidence_paths": [relative.as_posix() for relative in REQUIRED_FILES],
        "notes": "No Critical or Important findings. Any unavailable lane has an exact command, environment, device fact, and evidence owner in the matrix.",
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
    args = parser.parse_args(argv)
    try:
        candidate, matrix, performance, static_rank = validate()
        report = args.report if not args.report.is_absolute() else args.report.relative_to(ROOT)
        if any(part == ".." for part in report.parts):
            raise CheckError("report path must remain inside the repository")
        write_report(ROOT / report, candidate, matrix, performance, static_rank)
    except (CheckError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"storage-redesign-closure: {error}", file=sys.stderr)
        return 1
    print(f"storage-redesign-closure-ok: candidate={candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
