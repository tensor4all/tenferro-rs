#!/usr/bin/env python3
"""Keep every public extension trait reachable from its crate prelude."""

from pathlib import Path
import re
import sys

CRATES = (
    "tenferro-tensor",
    "tenferro-runtime",
    "tenferro-einsum",
    "tenferro-linalg",
    "tenferro-fft",
    "tenferro-ad",
)
TRAIT_RE = re.compile(r"\bpub\s+trait\s+([A-Za-z_][A-Za-z0-9_]*Ext)\b")
USE_RE = re.compile(r"\bpub\s+use\s+[^;]+;")
NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*Ext)\b")


def public_extension_traits(crate: Path) -> set[str]:
    names: set[str] = set()
    for path in (crate / "src").rglob("*.rs"):
        if "tests" not in path.parts:
            names.update(TRAIT_RE.findall(path.read_text()))
    return names


def prelude_extension_traits(crate: Path) -> set[str]:
    text = (crate / "src" / "prelude.rs").read_text()
    names: set[str] = set()
    for statement in USE_RE.findall(text):
        names.update(NAME_RE.findall(statement))
    return names


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failed = False
    for crate_name in CRATES:
        crate = root / "crates" / crate_name
        public = public_extension_traits(crate)
        prelude = prelude_extension_traits(crate)
        missing = sorted(public - prelude)
        extra = sorted(prelude - public)
        if missing or extra:
            failed = True
            print(f"{crate_name}: public={sorted(public)} prelude={sorted(prelude)}")
            if missing:
                print(f"  missing from prelude: {', '.join(missing)}")
            if extra:
                print(f"  not declared public in crate: {', '.join(extra)}")
        else:
            print(f"{crate_name}: {len(public)} public *Ext traits covered")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
