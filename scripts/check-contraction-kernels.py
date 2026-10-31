#!/usr/bin/env python3
"""Reject deep slice-iterator scalar contraction kernels in crates/*/src/**/*.rs.

This is a source heuristic, not Rust type checking: .iter()/iter_mut() and
borrowed for-loop operands are treated as slices. It deliberately does not
recognize arbitrary iterator adapters, macro expansion, or mul_add calls.
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Exact repository-relative files only; every future entry needs a reason.
ALLOWLIST = {}

NON_CODE = re.compile(
    r'//[^\n]*|/\*|(?:br|cr|r)(?P<hashes>\#*)"|"(?:\\.|[^"\\])*"'
    r"|'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\])'",
    re.DOTALL,
)
LOOP = re.compile(r"\b(?:for\s+[^;{}]*?\bin\s+[^;{}]+|while\s+[^;{}]+|loop\s*)\{")
SLICE_ITERATOR = re.compile(r"\.\s*iter(?:_mut)?\s*\(|\bin\s*&\s*(?:mut\b\s*)?\w")
ANNOTATION = re.compile(r"\s*// contraction-kernel-ok:\s*\S.*")
ASSIGNMENT = re.compile(
    r"\s*(?P<lhs>\*?\s*\w+(?:\s*\[[^;]+\])?)\s*(?P<op>\+=|=)\s*(?P<rhs>.+)",
    re.DOTALL,
)


def mask_non_code(source):
    """Preserve offsets and line numbers, including for nested comments/raw strings."""
    masked = list(source)
    offset = 0
    while match := NON_CODE.search(source, offset):
        end = match.end()
        if match.group() == "/*":
            depth = 1
            while depth and (marker := re.search(r"/\*|\*/", source[end:])):
                depth += 1 if marker.group() == "/*" else -1
                end += marker.end()
            if depth:
                end = len(source)
        elif match.group("hashes") is not None:
            terminator = '"' + match.group("hashes")
            close = source.find(terminator, end)
            end = len(source) if close < 0 else close + len(terminator)
        masked[match.start():end] = ["\n" if c == "\n" else " " for c in source[match.start():end]]
        offset = end
    return "".join(masked)


def has_product(expression):
    # Index arithmetic is not a scalar product; a unary dereference is not either.
    outside_indices = []
    depth = 0
    for char in expression:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if depth == 0:
            outside_indices.append(char)
    return re.search(r"[\w)\]]\s*\*\s*[*&\w(+-]", "".join(outside_indices)) is not None


def findings(source):
    """Return (line, loop depth) for unannotated scalar multiply-accumulates."""
    code = mask_non_code(source)
    lines = source.splitlines()
    loops = {}
    for match in LOOP.finditer(code):
        line = code.count("\n", 0, match.start()) + 1
        exempt = line > 1 and ANNOTATION.fullmatch(lines[line - 2]) is not None
        loops[match.end() - 1] = (match.group(), exempt)

    scopes = []
    start = 0
    result = []
    for delimiter in re.finditer(r"[{};]", code):
        token = delimiter.group()
        if token == "{":
            scopes.append(loops.get(delimiter.start()))
        else:
            active = [scope for scope in scopes if scope is not None]
            if len(active) >= 3 and not any(scope[1] for scope in active):
                statement = code[start:delimiter.start()]
                assignment = ASSIGNMENT.fullmatch(statement)
                if assignment and SLICE_ITERATOR.search(active[-1][0]):
                    lhs = re.sub(r"\s+", "", assignment["lhs"])
                    rhs = re.sub(r"\s+", "", assignment["rhs"])
                    if assignment["op"] == "=":
                        prefix = re.match(r"\(*" + re.escape(lhs) + r"\)*\+", rhs)
                        rhs = rhs[prefix.end():] if prefix else ""
                    if has_product(rhs):
                        position = start + assignment.start("lhs")
                        result.append((code.count("\n", 0, position) + 1, len(active)))
            if token == "}" and scopes:
                scopes.pop()
        start = delimiter.end()
    return result


def check(root):
    failures = []
    paths = sorted(root.glob("crates/*/src/**/*.rs"))
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if relative not in ALLOWLIST:
            for line, depth in findings(path.read_text(encoding="utf-8")):
                failures.append((relative, line, depth))
    if failures:
        for path, line, depth in failures:
            print(f"{path}:{line}: handwritten scalar contraction in slice-iterator loop nest (depth {depth})")
        print("Use einsum_tensors, tcontract, tlinalg, or the owning backend provider.")
        print("For an unavoidable loop, put // contraction-kernel-ok: <reason> immediately above it and explain it in the PR.")
        print("File allowlist:")
        for path, reason in sorted(ALLOWLIST.items()):
            print(f"  {path}: {reason}")
        if not ALLOWLIST:
            print("  (empty)")
        return 1
    print(f"contraction-kernels: {len(paths)} Rust files checked; no unannotated kernels found (allowlist: {len(ALLOWLIST)} files).")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (also used for regression fixtures)")
    args = parser.parse_args()
    return check(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
