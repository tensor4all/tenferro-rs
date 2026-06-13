#!/usr/bin/env python3
"""Count *real* `unsafe` usage in the tenferro-rs workspace.

A naive `grep -c unsafe` over-counts badly: it includes the word inside line
and block comments, `// SAFETY:` notes, doc comments, string literals, and test
code, and it picks up generated code under `target/`. This script strips
comments and string literals with a small state machine, skips build artifacts,
optionally removes test code, and then categorizes the remaining `unsafe`
tokens (block / fn / impl / trait).

Why this matters: `unsafe` density is a maintenance and audit signal. The real
number — production `unsafe` that a reviewer must reason about — is what counts,
not the raw grep hit count.

Usage:
    python3 scripts/count-unsafe.py                # scans crates/ and ext/
    python3 scripts/count-unsafe.py crates         # restrict to given roots
    python3 scripts/count-unsafe.py --include-tests

Output: raw vs. real counts, a per-category breakdown, and a per-crate table so
the source of any inflation is visible. Exit status is always 0; this is a
reporting tool, not a gate.
"""
import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {"target", ".git", ".worktrees", "third_party", "node_modules"}


def strip_comments_and_strings(src: str) -> str:
    """Remove // line comments, /* */ (nested) block comments, "..." strings,
    and r#"..."# raw strings, replacing removed spans with blanks so line
    structure survives. Char literals are left alone (lifetimes like 'a make
    them ambiguous, and they never contain the substring "unsafe")."""
    out = []
    i, n = 0, len(src)
    block_depth = 0
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if block_depth > 0:
            if c == "/" and nxt == "*":
                block_depth += 1
                out.append("  "); i += 2; continue
            if c == "*" and nxt == "/":
                block_depth -= 1
                out.append("  "); i += 2; continue
            out.append("\n" if c == "\n" else " "); i += 1; continue

        if c == "/" and nxt == "/":  # line comment
            while i < n and src[i] != "\n":
                out.append(" "); i += 1
            continue
        if c == "/" and nxt == "*":  # block comment start
            block_depth = 1
            out.append("  "); i += 2; continue
        if c == "r" and (nxt == '"' or nxt == "#"):  # raw string r"..."/r#"..."#
            j = i + 1
            hashes = 0
            while j < n and src[j] == "#":
                hashes += 1; j += 1
            if j < n and src[j] == '"':
                j += 1
                closer = '"' + "#" * hashes
                end = src.find(closer, j)
                end = n if end == -1 else end + len(closer)
                for k in range(i, end):
                    out.append("\n" if src[k] == "\n" else " ")
                i = end; continue
        if c == '"':  # normal string
            out.append(" "); i += 1
            while i < n:
                if src[i] == "\\":
                    out.append("  "); i += 2; continue
                if src[i] == '"':
                    out.append(" "); i += 1; break
                out.append("\n" if src[i] == "\n" else " "); i += 1
            continue

        out.append(c); i += 1
    return "".join(out)


def remove_cfg_test_modules(src: str) -> str:
    """Brace-match and remove `#[cfg(test)] mod ... { ... }` regions."""
    pat = re.compile(r"#\[cfg\(test\)\]")
    while True:
        m = pat.search(src)
        if not m:
            break
        brace = src.find("{", m.end())
        if brace == -1:
            src = src[: m.start()] + src[m.end():]
            continue
        depth, j = 0, brace
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        src = src[: m.start()] + src[j + 1:]
    return src


UNSAFE_TOKEN = re.compile(r"\bunsafe\b")
CATS = [
    ("unsafe fn", re.compile(r"\bunsafe\s+fn\b")),
    ("unsafe impl", re.compile(r"\bunsafe\s+impl\b")),
    ("unsafe trait", re.compile(r"\bunsafe\s+trait\b")),
    ("unsafe block {", re.compile(r"\bunsafe\s*\{")),
]


def is_test_file(path: str) -> bool:
    p = path.replace(os.sep, "/")
    return ("/tests/" in p or "/benches/" in p or "/examples/" in p
            or p.endswith("_test.rs") or p.endswith("_tests.rs"))


def crate_of(path: str) -> str:
    p = path.replace(os.sep, "/")
    m = re.search(r"(?:^|/)crates/([^/]+)/", p)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|/)ext/([^/]+)/", p)
    if m:
        return "ext/" + m.group(1)
    return os.path.basename(os.path.dirname(p))


def main(argv):
    include_tests = "--include-tests" in argv
    roots = [a for a in argv[1:] if not a.startswith("--")]
    if not roots:
        repo = Path(__file__).resolve().parent.parent
        roots = [str(repo / "crates"), str(repo / "ext")]

    files = []
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                if fn.endswith(".rs"):
                    files.append(os.path.join(dp, fn))

    totals = {"raw": 0, "stripped": 0, "real": 0}
    cat_totals = {name: 0 for name, _ in CATS}
    per_crate = {}

    for path in sorted(files):
        try:
            src = open(path, encoding="utf-8").read()
        except OSError:
            continue
        raw = len(UNSAFE_TOKEN.findall(src))
        if raw == 0:
            continue

        stripped = strip_comments_and_strings(src)
        stripped_n = len(UNSAFE_TOKEN.findall(stripped))

        if is_test_file(path) and not include_tests:
            prod = ""
        elif include_tests:
            prod = stripped
        else:
            prod = remove_cfg_test_modules(stripped)
        real = len(UNSAFE_TOKEN.findall(prod))

        totals["raw"] += raw
        totals["stripped"] += stripped_n
        totals["real"] += real

        pc = per_crate.setdefault(crate_of(path), {"raw": 0, "real": 0})
        pc["raw"] += raw
        pc["real"] += real
        for name, rx in CATS:
            cat_totals[name] += len(rx.findall(prod))

    scope = "incl. test code" if include_tests else "excl. test/bench/example code"
    print("=== unsafe keyword counts ===")
    print(f"  raw grep-style (incl comments/docs/strings):  {totals['raw']}")
    print(f"  after stripping comments + string literals:   {totals['stripped']}")
    print(f"  REAL ({scope}):  {totals['real']}")
    classified = sum(cat_totals.values())
    print()
    print("=== real unsafe by category ===")
    for name, _ in CATS:
        print(f"  {name:16s}: {cat_totals[name]}")
    print(f"  (classified {classified}/{totals['real']}; remainder is unsafe in "
          f"expression position, e.g. `pub unsafe fn`, `= unsafe {{`)")
    print()
    print("=== real unsafe per crate (real, raw) ===")
    for c in sorted(per_crate, key=lambda k: -per_crate[k]["real"]):
        pc = per_crate[c]
        print(f"  {pc['real']:4d}  (raw {pc['raw']:4d})  {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
