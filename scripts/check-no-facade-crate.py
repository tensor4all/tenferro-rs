#!/usr/bin/env python3
"""Fail if the removed root `tenferro` facade crate or dependencies return."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_PATHS = [
    ROOT / "tenferro" / "Cargo.toml",
]

SKIP_DIRS = {
    ".git",
    "target",
    "docs/plans",
    "docs/superpowers",
}

FORBIDDEN_SNIPPETS = [
    'path = "../' + 'tenferro"',
    'path = "../../' + 'tenferro"',
    "tenferro" + "/autodiff",
]


def is_skipped(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return any(rel == skip or rel.startswith(skip + "/") for skip in SKIP_DIRS)


def main() -> int:
    failures: list[str] = []
    for path in FORBIDDEN_PATHS:
        if path.exists():
            failures.append(f"forbidden path exists: {path.relative_to(ROOT)}")

    for path in ROOT.rglob("*"):
        if is_skipped(path) or not path.is_file():
            continue
        if path == Path(__file__).resolve():
            continue
        if path.suffix not in {".toml", ".md", ".rs", ".py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                failures.append(f"{path.relative_to(ROOT)} contains {snippet!r}")

    if failures:
        print("no-facade boundary check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("no-facade-boundary-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
