#!/usr/bin/env python3
"""Check crate-boundary rules that are easy to regress mechanically."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def iter_files(path: pathlib.Path):
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            yield candidate


def check_tensor_has_no_gpu_runtime_deps() -> list[str]:
    forbidden = (
        "cubecl",
        "cudarc",
        "tenferro-gpu",
        "tenferro-internal-gpubackend",
    )
    paths = [ROOT / "tenferro-tensor" / "Cargo.toml", ROOT / "tenferro-tensor" / "src"]
    violations: list[str] = []
    for path in paths:
        for file_path in iter_files(path):
            text = file_path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                for token in forbidden:
                    if token in lowered:
                        rel = file_path.relative_to(ROOT)
                        violations.append(f"{rel}:{line_no}: forbidden tensor-core GPU token `{token}`")
    return violations


def main() -> int:
    violations = check_tensor_has_no_gpu_runtime_deps()
    if violations:
        print("crate-boundary check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("crate-boundary-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
