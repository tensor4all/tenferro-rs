#!/usr/bin/env python3
"""Compile the static-rank element probe and classify its generated loops."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES = ("tensor_static_rank_read_probe", "tensor_static_rank_write_probe")
FORBIDDEN = (
    "prepare_",
    "resolve_descriptor",
    "backend_buffer",
    "synchronize",
    "coordinate_decode",
    "__rust_alloc",
)


def run(*args: str) -> str:
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def select_existing_record(record: dict[str, object] | None, *, refresh: bool) -> dict[str, object] | None:
    return None if refresh else record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    saved = None
    if args.report.is_file() and not args.refresh:
        text = args.report.read_text(encoding="utf-8")
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            raise ValueError("existing static-rank report has no JSON record")
        saved = select_existing_record(json.loads(match.group(1)), refresh=args.refresh)
    if saved is not None:
        freeze_text = (ROOT / "docs/design/storage-contract-freeze.md").read_text(encoding="utf-8")
        freeze_match = re.search(r"```json\s*(\{.*?\})\s*```", freeze_text, re.DOTALL)
        if not freeze_match:
            raise ValueError("freeze report has no JSON record")
        frozen = json.loads(freeze_match.group(1))["candidate_commit"]
        if saved.get("candidate_commit") != frozen:
            raise ValueError("existing static-rank report does not match frozen candidate")
        if saved.get("status") != "pass":
            raise ValueError("existing static-rank report is not passing")
        print(json.dumps(saved, indent=2))
        return 0
    command = [
        "cargo",
        "rustc",
        "-p",
        "tenferro-tensor",
        "--bench",
        "element_access",
        "--release",
        "--",
        "--emit=asm",
    ]
    result = {
        "schema": "tenferro.storage-static-rank-codegen.v1",
        "candidate_commit": run("git", "rev-parse", "HEAD"),
        "command": " ".join(command),
        "rustc": run("rustc", "-Vv"),
        "target": run("rustc", "-Vv").split("host: ", 1)[-1].splitlines()[0],
        "probes": list(PROBES),
        "status": "inconclusive",
        "observations": [],
    }
    try:
        subprocess.run(command, cwd=ROOT, check=True, text=True)
    except subprocess.CalledProcessError as error:
        result["observations"].append(f"cargo rustc failed with exit code {error.returncode}")
        args.report.write_text(
            "# Static-rank storage codegen\n\n```json\n"
            + json.dumps(result, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        return 1

    assemblies = sorted(
        (ROOT / "target" / "release" / "deps").glob("element_access-*.s"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not assemblies:
        result["observations"].append("cargo emitted no assembly file")
    else:
        assembly_path = assemblies[0]
        assembly = assembly_path.read_text(encoding="utf-8", errors="replace")
        result["assembly"] = str(assembly_path.relative_to(ROOT))
        missing = [probe for probe in PROBES if probe not in assembly]
        if missing:
            result["observations"].append(f"missing probe symbols: {missing}")
        else:
            bad: dict[str, dict[str, list[str]]] = {}
            missing_loops: list[str] = []
            for probe in PROBES:
                start = assembly.find(f"{probe}:")
                size_match = re.search(rf"\.size\s+{re.escape(probe)}\b", assembly[start:])
                end = start + size_match.start() if size_match else len(assembly)
                function = assembly[start:end]
                lines = function.splitlines()
                labels = {
                    line[:-1]: index
                    for index, line in enumerate(lines)
                    if line.startswith(".") and line.endswith(":")
                }
                loops: list[tuple[int, int]] = []
                for index, line in enumerate(lines):
                    match = re.search(r"(?:j[a-z]+|loop[a-z]*)\s+([.A-Za-z0-9_]+)", line)
                    if not match or match.group(1) not in labels:
                        continue
                    target = labels[match.group(1)]
                    if target < index:
                        loops.append((target, index + 1))
                if not loops:
                    missing_loops.append(probe)
                    continue
                probe_bad = {}
                for target, end_index in loops:
                    body = "\n".join(lines[target:end_index])
                    # Keep only the arithmetic loop(s); error/unwind blocks also
                    # contain backward branches but are not element traversal.
                    if not re.search(r"\b(?:add|mul|sub|div)(?:pd|ps|sd|ss)\b", body):
                        continue
                    tokens = [token for token in FORBIDDEN if token in body]
                    if tokens:
                        probe_bad[f"line_{target}"] = tokens
                if probe_bad:
                    bad[probe] = probe_bad
            if missing_loops:
                result["observations"].append(f"missing backward loop: {missing_loops}")
            elif bad:
                result["observations"].append(f"forbidden assembly calls in loops: {bad}")
            else:
                result["observations"].append(
                    "both fixed-rank probes are present and their backward loops contain no prohibited setup calls"
                )
                result["status"] = "pass"
    args.report.write_text(
        "# Static-rank storage codegen\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
