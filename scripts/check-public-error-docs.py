#!/usr/bin/env python3
"""Audit public Rust ``Result`` APIs for concrete ``# Errors`` docs.

The Rust compiler cannot make ``clippy::missing_errors_doc`` a repository
contract by itself: it checks only whether a heading exists, and it does not
cover all public trait methods consistently across toolchain versions. This
small source audit complements Clippy by checking the public API surface and
requiring the section to name a concrete error variant or failure condition.
It intentionally does not rewrite source; missing documentation is a review
failure that must be fixed at the API's source of truth.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PUBLIC_FN_RE = re.compile(
    r"^\s*pub\s+(?!(?:\([^)]*\)))"
    r"(?:(?:async|const|unsafe)\s+)*fn\b"
)
TRAIT_FN_RE = re.compile(r"^\s*(?:(?:async|unsafe)\s+)*fn\b")
RESULT_RE = re.compile(r"\b(?:[A-Za-z_][\w:]*::)?Result\s*<")
TRAIT_RE = re.compile(r"^\s*pub\s+(?:unsafe\s+)?trait\b")
DOC_RE = re.compile(r"^\s*///")
DOC_ATTRIBUTE_RE = re.compile(r'^\s*#\[doc\s*=\s*("(?:\\.|[^"\\])*")\s*\]\s*$')
ATTRIBUTE_RE = re.compile(r"^\s*#\[")
ERROR_HEADING_RE = re.compile(r"^\s*#\s+Errors\s*$", re.IGNORECASE)
DEFERRED_HEADING_RE = re.compile(r"^\s*#\s+Deferred errors\s*$", re.IGNORECASE)
NEXT_HEADING_RE = re.compile(r"^\s*#\s+")
# A traced API may mention a deferred check in its prose. Such a statement is
# required to live under an explicit ``# Deferred errors`` heading so callers
# can distinguish graph-build failures from compile/execution failures.
DEFERRED_HINT_RE = re.compile(
    r"\b(?:deferred|compile(?:d)? or execution|compilation or execution|"
    r"after binding|at execution,)\b",
    re.IGNORECASE,
)
# A category label such as "backend error" or "validation failure" is not
# enough: the section must identify an observable variant or a condition that
# lets a caller decide how to recover. Keep this deliberately narrower than a
# prose spell-checker so boilerplate cannot satisfy the repository contract.
CONCRETE_ERROR_RE = re.compile(
    r"(?:"
    r"\bInfallible\b|"
    r"\b(?:Error|ErrorKind)::(?!Validation\b)[A-Za-z0-9_]+\b|"
    r"\bValidationError::[A-Za-z0-9_]+\b|"
    r"\b[A-Z][A-Za-z0-9_]*(?:Error|Failure|Mismatch|OutOfBounds|Overflow|"
    r"Underflow|Unavailable|NonConvergence|Singular|Unsupported|Invalid|"
    r"Poisoned|Conversion|State)\b|"
    r"\b(?:shape|rank|axis|dtype|configuration|config|input|output|"
    r"cotangent|tangent|operand|index|placement|buffer|metadata|program|"
    r"graph|cache|device|plugin|lock|file|stream|serialization|worker|"
    r"operation|extension|family|backend|runtime|divisor|exponent)\b"
    r"[^.\n]{0,80}\b(?:mismatch|out of bounds|overflow|underflow|invalid|"
    r"incompatible|missing|poisoned|unavailable|unsupported|singular|"
    r"non[- ]?convergen\w*|zero|failure|(?:do not )?match)\b|"
    r"\binvalid\s+(?:shape|shapes|rank|axis|axes|dtype|dtypes|configuration|"
    r"config|input|output|notation|subscripts|metadata|binding|indices|"
    r"positions|range|dimensions?)\b|"
    r"\b(?:no|missing|without)\b[^.\n]{0,50}\b(?:shape|rank|axis|dtype|"
    r"metadata|input|output|binding|executor|cache|runtime|plugin)\b|"
    r"\b(?:division by zero|zero divisor|non[- ]?convergen\w*|singular|"
    r"overflow|underflow|poisoned|out of bounds|missing [A-Za-z]|"
    r"unsupported[- ](?:operation|dtype|conversion|backend|extension|rule))\b"
    r")",
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    function: str
    reason: str


def rust_files(root: Path) -> list[Path]:
    excluded = {".git", ".codegraph", "target"}
    paths: list[Path] = []
    for path in root.rglob("*.rs"):
        if any(part in excluded for part in path.parts):
            continue
        paths.append(path)
    return sorted(paths)


def brace_delta(line: str) -> int:
    # This is a documentation audit, not a Rust parser. Removing comments and
    # string literals avoids the common false positives in examples while
    # keeping the trait-range tracking deterministic and dependency-free.
    code = re.sub(r'//.*$', "", line)
    code = re.sub(r'"(?:\\.|[^"\\])*"', "", code)
    return code.count("{") - code.count("}")


def preceding_docs(lines: list[str], index: int) -> tuple[str, bool]:
    docs: list[str] = []
    hidden = False
    cursor = index - 1
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if DOC_RE.match(lines[cursor]):
            docs.append(lines[cursor])
        elif (match := DOC_ATTRIBUTE_RE.match(lines[cursor])) is not None:
            # Procedural-macro source commonly uses #[doc = "..."] because
            # quote! cannot emit a literal /// block. Treat that attribute as
            # the generated rustdoc it represents so macro-generated Result
            # APIs are held to the same documentation contract.
            docs.append(f"/// {ast.literal_eval(match.group(1))}")
        elif ATTRIBUTE_RE.match(lines[cursor]):
            hidden = hidden or "doc(hidden)" in stripped
        elif stripped == "" and docs:
            break
        else:
            break
        cursor -= 1
    return "\n".join(reversed(docs)), hidden


def function_signature(lines: list[str], index: int) -> str:
    parts: list[str] = []
    for line in lines[index : index + 80]:
        parts.append(line)
        if ";" in line or "{" in line:
            break
    return " ".join(parts)


def function_name(signature: str) -> str:
    match = re.search(r"\bfn\s+([A-Za-z_][\w]*)", signature)
    return match.group(1) if match else "<anonymous>"


def doc_section(doc: str, heading: re.Pattern[str]) -> str | None:
    if not doc:
        return None
    cleaned = [line.split("///", 1)[-1].strip() for line in doc.splitlines()]
    start = next((index for index, line in enumerate(cleaned) if heading.match(line)), None)
    if start is None:
        return None
    body: list[str] = []
    for line in cleaned[start + 1 :]:
        if NEXT_HEADING_RE.match(line):
            break
        body.append(line)
    return "\n".join(body).strip()


def error_section(doc: str) -> str | None:
    return doc_section(doc, ERROR_HEADING_RE)


def deferred_section(doc: str) -> str | None:
    return doc_section(doc, DEFERRED_HEADING_RE)


def audit_file(path: Path, changed_lines: set[int] | None = None) -> list[Finding]:
    lines = path.read_text(encoding="utf-8").splitlines()
    findings: list[Finding] = []
    trait_depth: int | None = None
    depth = 0

    for index, line in enumerate(lines):
        if TRAIT_RE.match(line):
            trait_depth = depth + max(brace_delta(line), 0)

        is_public = bool(PUBLIC_FN_RE.match(line))
        is_trait_method = trait_depth is not None and bool(TRAIT_FN_RE.match(line))
        if is_public or is_trait_method:
            if changed_lines is not None and index + 1 not in changed_lines:
                depth += brace_delta(line)
                if trait_depth is not None and depth < trait_depth:
                    trait_depth = None
                continue
            signature = function_signature(lines, index)
            if RESULT_RE.search(signature):
                docs, hidden = preceding_docs(lines, index)
                if not hidden:
                    section = error_section(docs)
                    if section is None:
                        findings.append(
                            Finding(path, index + 1, function_name(signature), "missing # Errors")
                        )
                    elif not CONCRETE_ERROR_RE.search(re.sub(r"\s+", " ", section)):
                        findings.append(
                            Finding(
                                path,
                                index + 1,
                                function_name(signature),
                                "# Errors does not name a concrete variant or condition",
                            )
                        )
                    elif (
                        path.name == "traced.rs"
                        and DEFERRED_HINT_RE.search(docs)
                        and deferred_section(docs) is None
                    ):
                        findings.append(
                            Finding(
                                path,
                                index + 1,
                                function_name(signature),
                                "deferred validation must be documented under # Deferred errors",
                            )
                        )

        depth += brace_delta(line)
        if trait_depth is not None and depth < trait_depth:
            trait_depth = None

    return findings


def audit(
    root: Path,
    paths: list[Path] | None = None,
    changed_lines: dict[Path, set[int]] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths if paths is not None else rust_files(root):
        lines = changed_lines.get(path) if changed_lines is not None else None
        findings.extend(audit_file(path, lines))
    return findings


def changed_rust_lines(root: Path, revision: str) -> dict[Path, set[int]]:
    """Return added Rust source lines between ``revision`` and ``HEAD``."""

    diff = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            f"{revision}...HEAD",
            "--",
            "*.rs",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result: dict[Path, set[int]] = {}
    current: Path | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = (root / line.removeprefix("+++ b/")).resolve()
            result.setdefault(current, set())
            continue
        if current is None or not line.startswith("@@"):
            continue
        match = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        result[current].update(range(start, start + count))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--changed-from",
        metavar="REV",
        help="audit only Rust files changed between REV and HEAD",
    )
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = args.root_dir.resolve()
    paths = [((root / path).resolve() if not path.is_absolute() else path) for path in args.paths]
    changed_lines = None
    if args.changed_from:
        changed = subprocess.run(
            ["git", "diff", "--name-only", f"{args.changed_from}...HEAD", "--", "*.rs"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        paths = [(root / path).resolve() for path in changed]
        changed_lines = changed_rust_lines(root, args.changed_from)
    findings = audit(root, paths or None, changed_lines)
    if findings:
        print("public Result APIs with incomplete error documentation:", file=sys.stderr)
        for finding in findings:
            print(
                f"- {finding.path.relative_to(root)}:{finding.line}: "
                f"{finding.function}: {finding.reason}",
                file=sys.stderr,
            )
        print(
            "Each public Result API must document concrete variants/conditions "
            "under `# Errors`; traced symbolic APIs must also explain deferred "
            "validation where applicable.",
            file=sys.stderr,
        )
        return 1
    print("public-error-docs-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
