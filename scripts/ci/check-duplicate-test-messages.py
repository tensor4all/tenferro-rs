#!/usr/bin/env python3
"""Detect one assertion message copy-pasted across many test files.

`expect("...")` in test code carries the proof of a local invariant. When the
same sentence is pasted into every test that unwraps the same accessor, the
proof is not local any more: it is a claim about the whole repository that no
single site checks, and it drifts (#2492). The wording is also the only thing a
reader sees when the assertion fires, so 80 identical copies of "a preset ..."
explain nothing about the site that panicked.

This checker looks for one message literal used in more than `--max-files`
distinct test files. Short labels ("download", "upload", "CUDA engine") are
excluded: they name an operand, not an invariant, and repeating them is
readable. A message counts when it is a sentence - at least
`MIN_MESSAGE_CHARS` characters and `MIN_MESSAGE_WORDS` words.

Scope is test sources under `crates/*`: `src/**/tests/**`, `tests/**`,
`tests.rs` and `*_tests.rs`. Production code is out of scope; its `expect`
calls are a different question (REPOSITORY_RULES.md, Public Boundary Safety
Audits).

A reviewed exception is a `[[message]]` entry in
`scripts/ci/duplicate-test-messages-allowlist.toml` naming the literal, the
files it is reviewed in, and the reason one shared definition is unreachable.
An entry without a reason fails. The recorded file list is a ratchet: a new
file using the message fails the run, and a file that stopped using it is
reported so the entry can be pruned.

Usage:
    python3 scripts/ci/check-duplicate-test-messages.py [--report|--check] [--json]

Lives under `scripts/ci/` because a new top-level `scripts/*.py` is an unknown
path in the base revision's change policy, which reserves the paid GPU lane for
an otherwise Python-only diff (scripts/ci/change_policy.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path("scripts") / "ci" / "duplicate-test-messages-allowlist.toml"

# A call ending in a string literal: `expect("...")`. Multiline arguments are
# not matched, and neither is an `expect` that formats its message.
MESSAGE_PATTERN = re.compile(r'\.expect\(\s*"((?:[^"\\]|\\.)*)"\s*\)')

# A bare word names an operand; an invariant is a sentence.
MIN_MESSAGE_CHARS = 24
MIN_MESSAGE_WORDS = 4

# Two test files may share a sentence: a helper and its mirror in a sibling
# binary. Three is a pattern, and the pattern is what this check rejects.
DEFAULT_MAX_FILES = 2


def is_test_source(relative: str) -> bool:
    """Whether a repository-relative path is test code."""

    parts = relative.split("/")
    if len(parts) < 3 or parts[0] != "crates":
        return False
    name = parts[-1]
    if name == "tests.rs" or name.endswith("_tests.rs"):
        return True
    return "tests" in parts[:-1]


def is_message(literal: str) -> bool:
    """Whether a message literal states an invariant rather than naming an operand."""

    return len(literal) >= MIN_MESSAGE_CHARS and len(literal.split()) >= MIN_MESSAGE_WORDS


def scan(root: Path) -> dict[str, dict[str, int]]:
    """Map each message literal to the file -> occurrence count using it."""

    messages: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path in sorted(root.glob("crates/*/**/*.rs")):
        relative = path.relative_to(root).as_posix()
        if not is_test_source(relative):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in MESSAGE_PATTERN.finditer(text):
            literal = match.group(1)
            if is_message(literal):
                messages[literal][relative] += 1
    return {literal: dict(files) for literal, files in messages.items()}


def load_allowlist(path: Path) -> tuple[dict[str, dict], list[str]]:
    """Parse the reviewed exceptions, returning them and the reasons they are invalid."""

    if not path.is_file():
        return {}, []
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return {}, [f"{path}: unreadable allowlist: {error}"]

    entries: dict[str, dict] = {}
    errors: list[str] = []
    for entry in document.get("message", []):
        literal = entry.get("literal")
        reason = (entry.get("reason") or "").strip()
        files = entry.get("files") or []
        if not isinstance(literal, str) or not literal:
            errors.append(f"{path}: a [[message]] entry has no literal")
            continue
        if not reason:
            errors.append(f"{path}: {literal!r} has no reason")
            continue
        if not files:
            errors.append(f"{path}: {literal!r} names no files")
            continue
        entries[literal] = {"files": frozenset(files), "reason": reason}
    return entries, errors


def build_report(
    root: Path,
    allowlist_path: Path,
    max_files: int = DEFAULT_MAX_FILES,
) -> tuple[dict, list[str]]:
    """Return the report and the errors that make the run invalid regardless of findings."""

    messages = scan(root)
    allowlist, errors = load_allowlist(root / allowlist_path)

    findings: list[dict] = []
    reviewed: list[dict] = []
    for literal, files in sorted(messages.items()):
        entry = allowlist.get(literal)
        occurrences = sum(files.values())
        if len(files) <= max_files:
            if entry is not None:
                # The message no longer repeats: the exception can be removed.
                errors.append(
                    f"stale allowlist entry for {literal!r}: "
                    f"{len(files)} file(s) now, {len(entry['files'])} recorded"
                )
            continue
        finding = {
            "literal": literal,
            "files": sorted(files),
            "occurrences": occurrences,
        }
        if entry is None:
            findings.append(finding)
            continue
        unrecorded = sorted(set(files) - entry["files"])
        if unrecorded:
            findings.append({**finding, "unrecorded": unrecorded})
        else:
            reviewed.append({**finding, "reason": entry["reason"]})

    report = {
        "schema": "tenferro.duplicate-test-messages.v1",
        "max_files": max_files,
        "findings": findings,
        "reviewed": reviewed,
    }
    return report, errors


def _describe(finding: dict) -> str:
    lines = [
        f"{finding['literal']!r}: {len(finding['files'])} test files, "
        f"{finding['occurrences']} sites"
    ]
    for path in finding.get("unrecorded", []):
        lines.append(f"  new file: {path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_const",
        dest="mode",
        const="report",
        help="report findings without failing (the only mode a scheduled job may use)",
    )
    parser.add_argument(
        "--check",
        action="store_const",
        dest="mode",
        const="check",
        help="fail when a finding has no reviewed entry (default)",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help="how many test files may share one message",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT,
        help="repository root to scan",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=ALLOWLIST,
        help="allowlist path, relative to the root unless absolute",
    )
    arguments = parser.parse_args(argv)
    mode = arguments.mode or "check"
    allowlist = arguments.allowlist
    if not allowlist.is_absolute():
        allowlist = Path(allowlist)

    report, errors = build_report(arguments.root_dir, allowlist, arguments.max_files)
    if arguments.json:
        print(json.dumps({**report, "errors": errors}, indent=2, sort_keys=True))
    else:
        for finding in report["findings"]:
            print(_describe(finding))
        for entry in report["reviewed"]:
            print(f"reviewed: {_describe(entry)}")
        print(
            f"{len(report['findings'])} unreviewed duplicated message(s), "
            f"{len(report['reviewed'])} reviewed"
        )
        for error in errors:
            print(f"error: {error}", file=sys.stderr)

    if errors:
        return 1
    if mode == "report":
        return 0
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
