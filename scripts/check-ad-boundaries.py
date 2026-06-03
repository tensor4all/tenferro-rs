#!/usr/bin/env python3
"""Check that AD opt-in does not leak into runtime crate boundaries."""

from __future__ import annotations

from pathlib import Path
import re
import sys


FORBIDDEN = re.compile(r"\b(ADRule|EagerRuntime|EagerTensor|autodiff|chainrules|tidu)\b")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    runtime = repo / "crates" / "tenferro-runtime"
    checked = [runtime / "Cargo.toml"]
    checked.extend((runtime / "src").rglob("*.rs"))
    failed = False
    for path in checked:
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if FORBIDDEN.search(line):
                print(f"{path.relative_to(repo)}:{line_no}: {line}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
