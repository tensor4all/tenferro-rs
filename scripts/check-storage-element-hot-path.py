#!/usr/bin/env python3
"""Check that prepared storage element loops stay free of setup work.

This is intentionally a small source contract, not a Rust parser or a proof of
memory safety. The storage preparation APIs and unsafe blocks carry that proof;
this check protects the performance boundary from accidental provider/storage
lookups being reintroduced into the final loops.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FUNCTIONS = (
    ("crates/tenferro-tensor/src/storage/prepared.rs", "as_slice"),
    ("crates/tenferro-tensor/src/storage/prepared.rs", "as_slice_mut"),
    ("crates/tenferro-tensor/src/storage/prepared.rs", "iter_contiguous"),
    ("crates/tenferro-tensor/src/storage/prepared.rs", "iter_contiguous_mut"),
    ("crates/tenferro-tensor/src/storage/prepared.rs", "next"),
)
FORBIDDEN = (
    r"prepare_",
    r"resolve_descriptor",
    r"backend_buffer",
    r"provider_",
    r"synchronize",
    r"allocate",
    r"validate_",
    r"coordinate_decode",
    r"format!",
    r"Arc::",
    r"Rc::",
)


def _body_after(source: str, start: int) -> str:
    """Return the brace-balanced function body, ignoring Rust strings/comments."""
    open_brace = source.find("{", start)
    if open_brace < 0:
        raise ValueError("function has no body")
    depth = 0
    i = open_brace
    state = "code"
    while i < len(source):
        char = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if char == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if char == '"':
                state = "string"
                i += 1
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[open_brace + 1 : i]
        elif state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state == "string":
            if char == "\\":
                i += 2
                continue
            if char == '"':
                state = "code"
        i += 1
    raise ValueError("unterminated function body")


def _function_bodies(source: str, name: str) -> list[str]:
    bodies = []
    for match in re.finditer(rf"\bfn\s+{re.escape(name)}\s*\(", source):
        bodies.append(_body_after(source, match.start()))
    return bodies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    findings: list[dict[str, str]] = []
    checked = 0
    for relative, function in FUNCTIONS:
        path = ROOT / relative
        source = path.read_text(encoding="utf-8")
        bodies = _function_bodies(source, function)
        if not bodies:
            findings.append({"path": relative, "function": function, "reason": "missing"})
            continue
        for body in bodies:
            checked += 1
            # Comments and strings are irrelevant to this token-level contract.
            code = re.sub(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"", "", body, flags=re.S)
            for pattern in FORBIDDEN:
                if re.search(pattern, code):
                    findings.append(
                        {"path": relative, "function": function, "reason": pattern}
                    )
    result = {
        "schema": "tenferro.storage-element-hot-path.v1",
        "checked_functions": checked,
        "status": "fail" if findings else "pass",
        "findings": findings,
    }
    if args.report:
        args.report.write_text(
            "# Storage element hot-path contract\n\n"
            "```json\n"
            + json.dumps(result, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
    if findings:
        print(json.dumps(result, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
