#!/usr/bin/env python3
"""Verify and record the adapter-free storage ownership candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED = (
    Path("scripts/storage-ownership-contracts.toml"),
    Path("scripts/test-storage-ownership-contracts-v2.py"),
    Path("scripts/check-storage-element-hot-path.py"),
    Path("scripts/check-storage-static-rank-codegen.py"),
    Path("scripts/check-storage-contract-freeze.py"),
    Path("crates/tenferro-tensor/tests/storage_public_api.rs"),
    Path("crates/tenferro-gpu/tests/storage_provider_webgpu.rs"),
    Path("docs/storage-ownership.md"),
    Path("docs/guides/views-and-slicing.md"),
)
FORBIDDEN_SOURCE = (
    "pub struct ArcTensor",
    "pub struct Buffer<",
    "pub trait BackendBuffer<",
    "pub mod cuda_interop",
    "pub mod webgpu_interop",
    "webgpu_interop::allocate_raw",
    "webgpu_interop::finish_",
)


class CheckError(ValueError):
    pass


EVIDENCE_ALLOWLIST = frozenset(
    {
        "docs/design/storage-contract-freeze.md",
        "docs/testing/storage-hardware-matrix.md",
        "docs/testing/storage-static-rank-codegen.md",
        "docs/testing/storage-traversal-performance.md",
        "docs/worklogs/storage-documentation-source-blind-audit.md",
        "docs/worklogs/storage-redesign-closure.md",
        "docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md",
    }
)


def validate_evidence_paths(paths: set[str]) -> None:
    invalid = sorted(paths - EVIDENCE_ALLOWLIST)
    if invalid:
        raise CheckError(f"non-evidence path after candidate: {invalid[0]}")


def run(*args: str) -> str:
    result = subprocess.run(
        args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.strip()


def tracked_status() -> str:
    return run("git", "status", "--porcelain")


def existing_record(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise CheckError("existing freeze report has no fenced JSON record")
    import json

    record = json.loads(match.group(1))
    if record.get("status") != "pass":
        raise CheckError("existing freeze report is not passing")
    return record


def select_existing_record(record: dict[str, object] | None, *, refresh: bool) -> dict[str, object] | None:
    return None if refresh else record


def validate_candidate(report_path: Path, *, refresh: bool = False) -> tuple[str, str, bool]:
    previous = select_existing_record(existing_record(report_path), refresh=refresh)
    if previous is None and tracked_status():
        raise CheckError("candidate must be clean before freeze evidence is written")
    candidate = (
        str(previous["candidate_commit"])
        if previous is not None
        else run("git", "rev-parse", "HEAD")
    )
    if not HEX40.fullmatch(candidate):
        raise CheckError(f"candidate is not a full Git commit: {candidate!r}")
    base = (
        str(previous["base_commit"])
        if previous is not None
        else run("git", "merge-base", candidate, "origin/main")
    )
    if not HEX40.fullmatch(base):
        raise CheckError(f"base is not a full Git commit: {base!r}")
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            raise CheckError(f"required freeze path is missing: {relative}")
    handoff = list(ROOT.glob("HANDOFF-2026-07-25-tenferro-unification6-wip.md"))
    if handoff:
        raise CheckError("legacy handoff file remains")
    for path in ROOT.glob("crates/**/src/**/*.rs"):
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_SOURCE:
            if marker in text:
                raise CheckError(f"forbidden legacy source marker {marker!r} in {path.relative_to(ROOT)}")
    diff_check = subprocess.run(
        ("git", "diff", "--check", f"{base}...{candidate}"),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if diff_check.returncode:
        raise CheckError("candidate diff contains whitespace errors")
    if previous is not None:
        changed = subprocess.run(
            ("git", "diff", "--name-only", f"{candidate}..HEAD"),
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
        validate_evidence_paths(set(changed))
    return candidate, base, previous is not None


def write_report(path: Path, candidate: str, base: str) -> None:
    record = {
        "schema": "tenferro.storage-contract-freeze.v1",
        "candidate_commit": candidate,
        "base_commit": base,
        "status": "pass",
        "checks": {
            "clean_candidate": True,
            "required_paths": True,
            "legacy_handoff_removed": True,
            "source_inventory": True,
            "diff_check": True,
        },
        "evidence_paths": [relative.as_posix() for relative in REQUIRED],
        "notes": "Product/API/docs candidate is frozen; later commits are evidence-only.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Storage contract freeze\n\n"
        "The record below identifies the clean product candidate. Evidence-only "
        "commits must not change production/API/docs/checker semantics.\n\n"
        "```json\n" + json.dumps(record, indent=2) + "\n```\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = args.report if not args.report.is_absolute() else args.report.relative_to(ROOT)
        candidate, base, already_recorded = validate_candidate(
            ROOT / report, refresh=args.refresh
        )
        if any(part == ".." for part in report.parts):
            raise CheckError("report path must remain inside the repository")
        if not already_recorded:
            write_report(ROOT / report, candidate, base)
    except (CheckError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"storage-contract-freeze: {error}", file=sys.stderr)
        return 1
    print(f"storage-contract-freeze-ok: candidate={candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
