#!/usr/bin/env python3
"""Review PR diffs against REPOSITORY_RULES.md using a delta-scoped LLM check.

Local API key loading uses the ``python-dotenv`` library. Install dev helpers with:

    python3 -m pip install -r scripts/requirements-dev.txt

When ``.env`` exists at the repository root it is loaded automatically.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "REPOSITORY_RULES.md"
PROMPT_PATH = ROOT / "ai" / "prompts" / "repository-rules-review.md"
PROMPT_VERSION = "1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
MAX_DIFF_CHARS = 60_000
MAX_FILE_DIFF_CHARS = 40_000
MAX_FINDINGS_PER_CHUNK = 8
HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)
MAX_ROUTED_SECTION_LINES = 100
RUNTIME_AD_FORBIDDEN = re.compile(
    r"\b(ADRule|EagerRuntime|EagerTensor|autodiff|chainrules|tidu)\b"
)
SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bAuthorization:\s*Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
SECRET_NAME = (
    r"[\w.-]*(?:api[_-]?key|token|secret|password|passwd|pwd|client[_-]?secret|"
    r"private[_-]?key)[\w.-]*"
)
# A typed declaration puts the type between the name and the value:
#   const API_KEY: &str = "....";     let api_key: String = "...".into();
# Matching only `name <sep> value` redacts the type and leaves the literal, so
# allow a short annotation before the separator that precedes the value.
SECRET_ANNOTATION = r"(?:\s*:[^=\n]{0,40})?"
# The value alternatives are ordered quoted-first so a quoted secret is
# consumed whole. Putting the bare-token alternative first would stop at the
# first space inside a quoted passphrase and upload the remainder. (Spelling
# out an example assignment here would make this file trip its own guard.)
SECRET_VALUE = r"""(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s#]+)"""
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")(?P<sep>" + SECRET_ANNOTATION
    + r"\s*[:=]\s*)(?P<value>" + SECRET_VALUE + r")"
)
# Quoted credentials may legitimately contain spaces (a diceware passphrase is
# prose by construction), so the value shape cannot be the discriminator.
QUOTED_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")" + SECRET_ANNOTATION + r"\s*[:=]\s*"
    r"""(?:"[^"\r\n]{8,}"|'[^'\r\n]{8,}')"""
)
# A quote opened on this line and closed on a later one hides the value from
# any single-line pattern, so treat the opening alone as disqualifying.
UNTERMINATED_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")" + SECRET_ANNOTATION
    + r"""\s*[:=]\s*["'][^"'\r\n]*$"""
)
# Since the value may be prose, the name has to carry the discrimination.
# `token_type`, `secret_name`, and `key_path` describe a credential rather than
# holding one, and their values are ordinary text.
SECRET_NAME_METADATA = re.compile(
    r"(?i)[_-]?(?:type|kind|name|names|id|ids|len|length|size|count|path|paths"
    r"|file|files|dir|env|var|vars|header|prefix|suffix|field|fields|url|uri"
    r"|list|set|map|schema|format|scheme|class|enum|error|regex|pattern|label"
    r"|source|store|provider|policy|status|state|mode|version)$"
)


def is_credential_name(name: str) -> bool:
    """Whether a matched identifier holds a credential rather than describes one."""
    return not SECRET_NAME_METADATA.search(name)


SEVERITY_ALIASES = {
    "block": "block",
    "blocker": "block",
    "critical": "block",
    "error": "block",
    "fail": "block",
    "failure": "block",
    "warn": "warn",
    "warning": "warn",
    "minor": "warn",
    "info": "warn",
    "informational": "warn",
}
ALWAYS_SECTIONS = frozenset(
    {
        "Public Surface Discipline",
        "Public Boundary Safety Audits",
        "Invariant Markers",
        "Work Logs And Design Records",
        "No Ad Hoc Fixes",
    }
)

HUMAN_ONLY_SECTIONS = frozenset(
    {
        "Final Cross-Phase Multi-Agent Audit",
        "External Contribution Intake",
        "CI Cost Discipline",
        "Performance-Gated Experiment Protocol",
    }
)

SECTION_TRIGGERS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(r"(^|/)ad/|tenferro-ad/|linearize|transpose_rule|autodiff"),
        frozenset(
            {
                "Rule Source Of Truth",
                "AD Rule Coverage",
                "Oracle Gate",
            }
        ),
    ),
    (
        re.compile(r"tenferro-tensor(?:-core)?/|/layout|/view|strided"),
        frozenset(
            {
                "Performance-Sensitive Safety Contracts",
                "Complexity Budget",
                "Materialization And Copies",
                "Device Transfer And Backend Buffer Errors",
                "Dense Layout And Linear Algebra",
                "Range Checks And Slicing",
                "Tensor Core Data Model",
                "Performance Anti-Patterns",
                "Structured Error Classification",
                "Unsafe Code Boundary",
            }
        ),
    ),
    (
        re.compile(r"tenferro-cpu/|/kernel/|strided-kernel|strided-rs|strided-einsum"),
        frozenset(
            {
                "Performance-Sensitive Safety Contracts",
                "Materialization And Copies",
                "Dense Layout And Linear Algebra",
                "Range Checks And Slicing",
                "CPU Kernel Implementation",
                "Faer Integration",
                "Performance Anti-Patterns",
                "Cache Ownership",
                "CPU Threading Contract",
                "Structured Error Classification",
                "Unsafe Code Boundary",
            }
        ),
    ),
    (
        re.compile(r"tenferro-gpu/|cubecl|cuda|cutensor|cublas"),
        frozenset(
            {
                "Performance-Sensitive Safety Contracts",
                "Materialization And Copies",
                "Device Transfer And Backend Buffer Errors",
                "Dense Layout And Linear Algebra",
                "Range Checks And Slicing",
                "Performance Anti-Patterns",
                "Cache Ownership",
                "GPU Backend Contract",
                "Structured Error Classification",
                "Unsafe Code Boundary",
            }
        ),
    ),
    (
        re.compile(r"tenferro-runtime/|runtime/|cache|executor"),
        frozenset(
            {
                "Performance-Sensitive Safety Contracts",
                "Complexity Budget",
                "Device Transfer And Backend Buffer Errors",
                "Cache Ownership",
                "Structured Error Classification",
            }
        ),
    ),
    (
        re.compile(r"tenferro-linalg/|linalg|faer|lapack|blas|dot_general|gemm"),
        frozenset(
            {
                "Performance-Sensitive Safety Contracts",
                "Materialization And Copies",
                "Dense Layout And Linear Algebra",
                "Faer Integration",
                "Performance Anti-Patterns",
                "Cache Ownership",
                "Structured Error Classification",
                "Unsafe Code Boundary",
            }
        ),
    ),
    (
        re.compile(r"(^|/)benches/|^benchmarks/|criterion|benchmark"),
        frozenset(
            {
                "Performance-Sensitive Tests And Benchmarks",
            }
        ),
    ),
    (
        re.compile(r"tenferro-einsum/|tenferro-linalg/|tenferro-fft/|ext/"),
        frozenset({"Standard Extension Boundary", "Wrapper DRY And Codegen"}),
    ),
    (
        re.compile(r"/tests/|_tests\.rs$|tests/"),
        frozenset({"Unit Test Organization", "AD Rule Coverage"}),
    ),
    (
        re.compile(r"^docs/worklogs/"),
        frozenset({"Work Logs And Design Records"}),
    ),
    (
        re.compile(r"^docs/design/"),
        frozenset({"Work Logs And Design Records"}),
    ),
    (
        re.compile(r"\.md$|\.qmd$|^README|^docs/"),
        frozenset(
            {
                "Public Surface Drift",
                "Documentation Policy",
                "Naming Style",
            }
        ),
    ),
    (
        re.compile(r"\.rs$"),
        frozenset(
            {
                "File Organization",
                "Public API Convention",
                "Generic Over Scalar Type",
            }
        ),
    ),
)


# Signals in the changed lines themselves. Path routing cannot see that a file
# under a generic path just gained an `unsafe` block or a cache.
CONTENT_TRIGGERS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(r"\bunsafe\b|get_unchecked|from_raw_parts|\bas_ptr\b|\bas_mut_ptr\b"),
        frozenset({"Unsafe Code Boundary", "Range Checks And Slicing"}),
    ),
    (
        re.compile(r"rayon|par_iter|num_threads|thread_pool"),
        frozenset({"CPU Threading Contract"}),
    ),
    (
        re.compile(r"to_dense\(|materiali[sz]e|\bvec!\[|collect::<Vec"),
        frozenset({"Materialization And Copies"}),
    ),
    (
        re.compile(r"OnceLock|lazy_static|thread_local!|struct \w*Cache\b"),
        frozenset({"Cache Ownership"}),
    ),
    (
        re.compile(r"\bfaer\b|faer::"),
        frozenset({"Faer Integration"}),
    ),
    (
        re.compile(r"ADRule|linearize|transpose_rule|autodiff"),
        frozenset({"Rule Source Of Truth", "AD Rule Coverage"}),
    ),
    (
        re.compile(r"// INVARIANT:|#\[allow\("),
        frozenset({"Invariant Markers"}),
    ),
    (
        re.compile(r"#\[cfg\(test\)\]|\binclude!\("),
        frozenset({"Unit Test Organization"}),
    ),
)


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str
    rule_section: str
    file: str
    line: int | None
    summary: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "rule_section": self.rule_section,
            "file": self.file,
            "line": self.line,
            "summary": self.summary,
            "detail": self.detail,
        }


def run_git(args: list[str], cwd: Path = ROOT, *, check: bool = True) -> str:
    # Without this git C-quotes non-ASCII pathnames ("\346\227\245..."), and
    # the quoted form matches no real path: per_file_diffs yields nothing and
    # the extension-based deterministic checks never recognise the file.
    completed = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def untracked_files() -> list[str]:
    """Paths git does not track yet, honouring .gitignore."""
    output = run_git(["ls-files", "--others", "--exclude-standard"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def untracked_file_diff(path: str) -> str:
    """Synthesize an all-added diff for a file with no committed counterpart.

    `git diff <base>` compares the working tree against a commit, so an
    untracked path has no object to compare and is omitted entirely. In
    worktree mode -- the documented local preview -- that means a brand new
    file is reviewed by nothing at all and the preview reports a false pass.
    `--no-index` against /dev/null gives a normal diff without touching the
    index; it exits 1 because the inputs differ, which is not an error here.
    """
    return run_git(
        ["diff", "--unified=3", "--no-index", "--", "/dev/null", path],
        check=False,
    )


def changed_files(base: str, head: str, *, worktree: bool = False) -> list[str]:
    if worktree:
        output = run_git(["diff", "--name-only", base])
        tracked = [line.strip() for line in output.splitlines() if line.strip()]
        return sorted({*tracked, *untracked_files()})
    output = run_git(["diff", "--name-only", f"{base}...{head}"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def unified_diff(base: str, head: str, *, worktree: bool = False) -> str:
    if worktree:
        pieces = [run_git(["diff", "--unified=3", base])]
        pieces.extend(untracked_file_diff(path) for path in untracked_files())
        return "\n".join(piece for piece in pieces if piece.strip())
    return run_git(["diff", "--unified=3", f"{base}...{head}"])


def per_file_diffs(
    base: str,
    head: str,
    files: list[str],
    *,
    worktree: bool = False,
) -> dict[str, str]:
    diffs: dict[str, str] = {}
    untracked = set(untracked_files()) if worktree else set()
    for path in files:
        if path in untracked:
            diff = untracked_file_diff(path)
        elif worktree:
            diff = run_git(["diff", "--unified=3", base, "--", path])
        else:
            diff = run_git(["diff", "--unified=3", f"{base}...{head}", "--", path])
        if diff.strip():
            diffs[path] = diff
    return diffs


def configure_dotenv(*, explicit: Path | None, skip: bool) -> None:
    """Load environment variables with python-dotenv."""
    if skip:
        return

    path = explicit if explicit is not None else ROOT / ".env"
    if not path.is_file():
        if explicit is not None:
            print(f"dotenv file not found: {path}", file=sys.stderr)
            raise SystemExit(1)
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        print(
            "python-dotenv is required to load .env; install with: "
            "python3 -m pip install -r scripts/requirements-dev.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    load_dotenv(path, override=False)


def parse_repository_rules_sections(path: Path = RULES_PATH) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line.removeprefix("## ").strip()
            current_lines = [line]
            continue
        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def select_rule_sections(
    files: list[str],
    added: dict[str, list[tuple[int, str]]] | None = None,
) -> list[str]:
    """Pick the rule sections to show the reviewer.

    Path routing alone misses rule-relevant code added under a generic path:
    an `unsafe` block in a file whose name matches no trigger meant the unsafe
    rules were never supplied -- and the prompt forbids inventing requirements
    that were not supplied, making the rule unenforceable there. Content
    triggers close that gap without making every safety section unconditional.

    HUMAN_ONLY_SECTIONS is now subtracted explicitly. It used to be excluded
    only because no trigger happened to name one, which left the guarantee in
    "Performance-Gated Experiment Protocol" -- "intentionally not routed to the
    diff-scoped review bot" -- resting on an accident.
    """
    selected = set(ALWAYS_SECTIONS)
    for path in files:
        for pattern, section_names in SECTION_TRIGGERS:
            if pattern.search(path):
                selected.update(section_names)

    if added:
        for entries in added.values():
            for _line_no, text in entries:
                for pattern, section_names in CONTENT_TRIGGERS:
                    if pattern.search(text):
                        selected.update(section_names)
    return sorted(selected - HUMAN_ONLY_SECTIONS)


def build_rules_payload(section_names: list[str]) -> str:
    sections = parse_repository_rules_sections()
    chunks: list[str] = []
    for name in section_names:
        body = sections.get(name)
        if body:
            chunks.append(body)
    if not chunks:
        return sections.get("Public Surface Discipline", "")
    return "\n\n".join(chunks)


def added_lines_by_file(diff_text: str) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    current_file: str | None = None
    new_line = 0

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw = line.removeprefix("+++ b/").removeprefix("+++ ")
            if raw != "/dev/null":
                current_file = raw
                result.setdefault(current_file, set())
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            new_line = int(match.group(1)) if match else 0
            continue
        if current_file is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            result[current_file].add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        elif line.startswith(" "):
            new_line += 1

    return result


def added_lines_with_text(diff_text: str) -> dict[str, list[tuple[int, str]]]:
    """Map each file to its added ``(new_line_number, text)`` pairs."""
    result: dict[str, list[tuple[int, str]]] = {}
    current_file: str | None = None
    new_line = 0

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw = line.removeprefix("+++ b/").removeprefix("+++ ")
            current_file = None if raw == "/dev/null" else raw
            if current_file is not None:
                result.setdefault(current_file, [])
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            new_line = int(match.group(1)) if match else 0
            continue
        if current_file is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            result[current_file].append((new_line, line[1:]))
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        elif line.startswith(" "):
            new_line += 1

    return result


def split_diff_chunks(file_diffs: dict[str, str]) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for path in sorted(file_diffs):
        piece = file_diffs[path]
        if len(piece) > MAX_FILE_DIFF_CHARS:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            chunks.extend(split_large_file_diff(piece))
            continue

        if current_len + len(piece) > MAX_DIFF_CHARS and current:
            chunks.append("\n".join(current))
            current = [piece]
            current_len = len(piece)
        else:
            current.append(piece)
            current_len += len(piece)

    if current:
        chunks.append("\n".join(current))
    return chunks


def joined_line_len(lines: list[str]) -> int:
    return len("\n".join(lines))


def line_deltas(line: str) -> tuple[int, int]:
    """How one diff body line advances the old and new file cursors."""
    if line.startswith("+"):
        return (0, 1)
    if line.startswith("-"):
        return (1, 0)
    if line.startswith("\\"):
        return (0, 0)
    return (1, 1)


def format_hunk_header(
    old_start: int, old_count: int, new_start: int, new_count: int, suffix: str
) -> str:
    return f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}"


def split_overlong_diff_line(prefix: list[str], line: str) -> list[str]:
    prefix_len = joined_line_len(prefix)
    line_budget = MAX_FILE_DIFF_CHARS - prefix_len - 1
    if line_budget <= 0:
        return ["\n".join([*prefix, line])]

    marker = line[:1] if line[:1] in {"+", "-", " "} else ""
    payload = line[1:] if marker else line
    payload_budget = max(1, line_budget - len(marker))
    chunks: list[str] = []
    for start in range(0, len(payload), payload_budget):
        piece = f"{marker}{payload[start : start + payload_budget]}"
        chunks.append("\n".join([*prefix, piece]))
    return chunks


def split_oversized_hunk(header: list[str], hunk: list[str]) -> list[str]:
    """Split one oversized hunk, rewriting the header for every emitted chunk.

    Repeating the original header would tell the model that each chunk starts
    at the hunk's first line, so findings in later chunks come back with line
    numbers thousands of lines too small. `filter_findings` then drops them, or
    worse keeps them against an unrelated added line that happens to collide.
    """
    if not hunk:
        return []

    parsed = HUNK_HEADER.match(hunk[0])
    if parsed is None:
        # Not a header we can renumber; fall back to repeating it verbatim
        # rather than inventing offsets.
        return split_oversized_hunk_verbatim(header, hunk)

    old_start = int(parsed.group(1))
    new_start = int(parsed.group(3))
    suffix = parsed.group(5)

    chunks: list[str] = []
    body: list[str] = []
    body_old = body_new = 0
    cursor_old, cursor_new = old_start, new_start

    def flush() -> None:
        nonlocal body, body_old, body_new, cursor_old, cursor_new
        if not body:
            return
        hunk_header = format_hunk_header(
            cursor_old, body_old, cursor_new, body_new, suffix
        )
        chunks.append("\n".join([*header, hunk_header, *body]))
        cursor_old += body_old
        cursor_new += body_new
        body = []
        body_old = body_new = 0

    for line in hunk[1:]:
        delta_old, delta_new = line_deltas(line)
        single_header = format_hunk_header(
            cursor_old + body_old, delta_old, cursor_new + body_new, delta_new, suffix
        )
        if joined_line_len([*header, single_header, line]) > MAX_FILE_DIFF_CHARS:
            flush()
            single_header = format_hunk_header(
                cursor_old, delta_old, cursor_new, delta_new, suffix
            )
            chunks.extend(
                split_overlong_diff_line([*header, single_header], line)
            )
            cursor_old += delta_old
            cursor_new += delta_new
            continue

        candidate_header = format_hunk_header(
            cursor_old, body_old + delta_old, cursor_new, body_new + delta_new, suffix
        )
        if (
            body
            and joined_line_len([*header, candidate_header, *body, line])
            > MAX_FILE_DIFF_CHARS
        ):
            flush()
        body.append(line)
        body_old += delta_old
        body_new += delta_new

    flush()
    if not chunks:
        chunks.append("\n".join([*header, hunk[0]]))
    return chunks


def split_oversized_hunk_verbatim(header: list[str], hunk: list[str]) -> list[str]:
    """Fallback for a hunk header this script cannot parse."""
    hunk_header = hunk[0]
    prefix = [*header, hunk_header]
    chunks: list[str] = []
    current = list(prefix)

    for line in hunk[1:]:
        if joined_line_len([*prefix, line]) > MAX_FILE_DIFF_CHARS:
            if current != prefix:
                chunks.append("\n".join(current))
                current = list(prefix)
            chunks.extend(split_overlong_diff_line(prefix, line))
            continue

        candidate = [*current, line]
        if current != prefix and joined_line_len(candidate) > MAX_FILE_DIFF_CHARS:
            chunks.append("\n".join(current))
            current = [*prefix, line]
        else:
            current = candidate

    if current != prefix:
        chunks.append("\n".join(current))
    else:
        chunks.append("\n".join(prefix))
    return chunks


def split_large_file_diff(diff_text: str) -> list[str]:
    """Split one file diff while preserving file headers in every chunk."""
    lines = diff_text.splitlines()
    header: list[str] = []
    hunks: list[list[str]] = []
    current_hunk: list[str] | None = None

    for line in lines:
        if line.startswith("@@"):
            if current_hunk is not None:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk is None:
            header.append(line)
        else:
            current_hunk.append(line)

    if current_hunk is not None:
        hunks.append(current_hunk)
    if not hunks:
        return [
            diff_text[start : start + MAX_FILE_DIFF_CHARS]
            for start in range(0, len(diff_text), MAX_FILE_DIFF_CHARS)
        ]

    chunks: list[str] = []
    current_lines = list(header)
    current_len = joined_line_len(current_lines)

    for hunk in hunks:
        hunk_len = joined_line_len(hunk)
        separator_len = 1 if current_lines else 0
        if joined_line_len([*header, *hunk]) > MAX_FILE_DIFF_CHARS:
            if current_lines != header:
                chunks.append("\n".join(current_lines))
                current_lines = list(header)
                current_len = joined_line_len(current_lines)
            chunks.extend(split_oversized_hunk(header, hunk))
            continue

        if (
            current_lines != header
            and current_len + separator_len + hunk_len > MAX_FILE_DIFF_CHARS
        ):
            chunks.append("\n".join(current_lines))
            current_lines = list(header)
            current_len = len("\n".join(current_lines))

        if current_lines:
            current_len += 1
        current_lines.extend(hunk)
        current_len += hunk_len

    if current_lines != header:
        chunks.append("\n".join(current_lines))
    return chunks


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    def mask(match: re.Match[str]) -> str:
        if not is_credential_name(match.group("name")):
            return match.group(0)
        return f"{match.group('name')}{match.group('sep')}[REDACTED_SECRET]"

    return SECRET_ASSIGNMENT.sub(mask, redacted)


def redact_file_diffs(file_diffs: dict[str, str]) -> dict[str, str]:
    return {path: redact_sensitive_text(diff) for path, diff in file_diffs.items()}


def contains_sensitive_text(text: str) -> bool:
    if any(pattern.search(text) for pattern in SECRET_VALUE_PATTERNS):
        return True
    for pattern in (QUOTED_SECRET_ASSIGNMENT, UNTERMINATED_SECRET_ASSIGNMENT):
        for match in pattern.finditer(text):
            if is_credential_name(match.group("name")):
                return True
    return False


def added_diff_text(diff_text: str) -> str:
    return "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def sensitive_diff_location(diff_text: str) -> tuple[str, int] | None:
    current_file: str | None = None
    new_line: int | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw_path = line.removeprefix("+++ b/").removeprefix("+++ ")
            current_file = None if raw_path == "/dev/null" else raw_path
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)?", line)
            new_line = int(match.group(1)) if match else None
            continue
        if current_file is None or new_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if contains_sensitive_text(line[1:]):
                return current_file, new_line
            new_line += 1
        elif line.startswith(" "):
            new_line += 1
    return None


def sensitive_diff_finding(diff_text: str) -> Finding | None:
    location = sensitive_diff_location(diff_text)
    if location is None:
        return None
    file, line = location
    return Finding(
        id="sensitive-diff",
        severity="block",
        rule_section="External LLM Review",
        file=file,
        line=line,
        summary="Sensitive-looking diff content detected before LLM upload",
        detail=(
            "External LLM review was skipped because the diff contains token, "
            "secret, password, authorization header, or private-key shaped text. "
            "Remove the sensitive value or use a maintainer-approved waiver if "
            "this is a verified false positive."
        ),
    )


def parse_findings(raw: Any) -> tuple[str, list[Finding]]:
    if not isinstance(raw, dict):
        raise ValueError("model response must be a JSON object")

    verdict = raw.get("verdict")
    if verdict not in {"pass", "fail"}:
        raise ValueError("verdict must be 'pass' or 'fail'")

    findings_raw = raw.get("findings", [])
    if not isinstance(findings_raw, list):
        raise ValueError("findings must be a list")

    findings: list[Finding] = []
    for index, item in enumerate(findings_raw[:MAX_FINDINGS_PER_CHUNK]):
        if not isinstance(item, dict):
            raise ValueError(f"findings[{index}] must be an object")
        severity_raw = str(item.get("severity", "warn")).strip().lower()
        severity = SEVERITY_ALIASES.get(severity_raw, "warn")
        line = item.get("line")
        if line is not None and not isinstance(line, int):
            raise ValueError(f"findings[{index}].line must be an integer or null")
        findings.append(
            Finding(
                id=str(item.get("id", f"finding-{index + 1}")),
                severity=severity,
                rule_section=str(item.get("rule_section", "unknown")),
                file=str(item.get("file", "")),
                line=line,
                summary=str(item.get("summary", "")),
                detail=str(item.get("detail", "")),
            )
        )

    return verdict, findings


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as err:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"model response was not valid JSON: {err}") from err
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as embedded_err:
            raise ValueError(
                f"model response was not valid JSON: {embedded_err}"
            ) from embedded_err

    if not isinstance(parsed, dict):
        raise ValueError("parsed JSON must be an object")
    return parsed


def llm_response_error_finding(error: BaseException) -> Finding:
    return Finding(
        id="llm-review-unusable",
        severity="block",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="External LLM review did not complete",
        detail=(
            "The repository-rules review could not send the request or parse "
            f"the model response: {type(error).__name__}: {error}"
        ),
    )


def api_key_error_finding(reason: str) -> Finding:
    return Finding(
        id="llm-api-key-invalid",
        severity="block",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="DEEPSEEK_API_KEY is unusable",
        detail=(
            f"{reason} Re-set the repository secret; the value itself is never "
            "printed. Until then the LLM pass cannot run."
        ),
    )


def api_key_problem(api_key: str) -> str | None:
    """Describe why a key cannot be sent, without echoing the key.

    HTTP header values are encoded as latin-1, so a key carrying non-ASCII
    text -- a pasted ellipsis from a masked console display, or mojibake --
    raises UnicodeEncodeError before any request leaves the machine. That is a
    ValueError subclass, so it used to surface as "did not produce usable
    JSON", pointing the reader at the model instead of at the secret.
    """
    if not api_key:
        return "The secret is empty."
    if not api_key.isascii():
        offsets = [
            str(index)
            for index, char in enumerate(api_key)
            if not char.isascii()
        ]
        return (
            "The secret contains non-ASCII characters at offset(s) "
            f"{', '.join(offsets[:10])}, which cannot be sent in an HTTP "
            "Authorization header. A masked value copied from a console "
            "display is the usual cause."
        )
    if any(char.isspace() for char in api_key):
        return "The secret contains whitespace."
    return None


def filter_findings(
    findings: list[Finding],
    files: list[str],
    added_lines: dict[str, set[int]],
    *,
    allow_global: bool = True,
) -> list[Finding]:
    allowed_files = set(files)
    kept: list[Finding] = []
    for finding in findings:
        if not finding.file:
            if allow_global:
                kept.append(finding)
            continue
        if finding.file and finding.file not in allowed_files:
            continue
        if finding.line is None and finding.severity == "block":
            continue
        if finding.line is not None:
            if finding.line not in added_lines.get(finding.file, set()):
                continue
        kept.append(finding)
    return kept


def reconcile_verdict(findings: list[Finding]) -> str:
    return "fail" if any(item.severity == "block" for item in findings) else "pass"


# Every way the request can fail below the JSON layer. OSError covers
# socket.timeout, TimeoutError, ConnectionResetError, ssl.SSLError, and
# urllib.error.URLError/HTTPError. http.client.HTTPException is a sibling of
# OSError, not a subclass, so a truncated chunked response (IncompleteRead)
# needs naming separately. socket.timeout only aliases TimeoutError from
# Python 3.10 on, so listing TimeoutError alone is not enough on 3.9.
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    http.client.HTTPException,
)
NETWORK_RETRIES = 2
RETRY_BACKOFF_SECONDS = 5.0
# Wall-clock ceiling for all LLM traffic in one run. The workflow allows 20
# minutes; a run that blows past it is killed mid-request, so neither the
# exception handler nor the report ever executes and the gate fails with no
# diagnostic at all -- the exact failure this retry logic exists to prevent.
# Finishing under our own budget guarantees a report is always posted.
DEFAULT_BUDGET_SECONDS = 900.0


def call_deepseek(
    *,
    api_key: str,
    model: str,
    api_url: str,
    system_prompt: str,
    user_content: str,
    timeout: float,
    deadline: float | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(1, NETWORK_RETRIES + 1):
        attempt_timeout = timeout
        if deadline is not None:
            attempt_timeout = min(timeout, max(1.0, deadline - time.monotonic()))
        try:
            with urllib.request.urlopen(request, timeout=attempt_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except TRANSPORT_ERRORS as exc:
            # Timeouts and connection resets are common enough on large diffs
            # that one blocked PR per blip is not an acceptable failure mode.
            last_error = exc
            if attempt == NETWORK_RETRIES:
                raise
            if deadline is not None and time.monotonic() + RETRY_BACKOFF_SECONDS >= deadline:
                # Retrying would run past the budget and get the job killed,
                # which loses the report entirely. Fail now, with a diagnostic.
                raise
            print(
                f"LLM request attempt {attempt}/{NETWORK_RETRIES} failed "
                f"({type(exc).__name__}: {exc}); retrying in "
                f"{RETRY_BACKOFF_SECONDS:.0f}s",
                file=sys.stderr,
            )
            time.sleep(RETRY_BACKOFF_SECONDS)
    else:  # pragma: no cover - the loop either breaks or raises
        raise last_error if last_error else RuntimeError("no response")
    content = body["choices"][0]["message"]["content"]
    return extract_json_payload(content)


def review_chunk(
    *,
    api_key: str,
    model: str,
    api_url: str,
    system_prompt: str,
    rules_text: str,
    changed: list[str],
    diff_chunk: str,
    timeout: float,
    deadline: float | None = None,
) -> tuple[str, list[Finding]]:
    user_content = "\n\n".join(
        [
            f"Prompt version: {PROMPT_VERSION}",
            "Changed files:",
            "\n".join(f"- {path}" for path in changed),
            "Applicable REPOSITORY_RULES sections:",
            rules_text,
            "Output limits:",
            f"- Return at most {MAX_FINDINGS_PER_CHUNK} findings for this diff chunk.",
            "- Do not split one root cause into multiple findings.",
            "- Do not invent requirements that are not explicit in the supplied rules.",
            "Unified diff (review only added/changed lines):",
            diff_chunk,
        ]
    )
    parsed = call_deepseek(
        api_key=api_key,
        model=model,
        api_url=api_url,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout=timeout,
        deadline=deadline,
    )
    return parse_findings(parsed)


def merge_findings(all_findings: list[Finding]) -> list[Finding]:
    merged: dict[tuple[str, str, str, int | None], Finding] = {}
    for finding in all_findings:
        key = (finding.id, finding.file, finding.summary, finding.line)
        existing = merged.get(key)
        if existing is None or (
            finding.severity == "block" and existing.severity != "block"
        ):
            merged[key] = finding
    return list(merged.values())


def budget_exhausted_finding(reviewed: int, total: int) -> Finding:
    return Finding(
        id="llm-budget-exhausted",
        severity="warn",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="External LLM review stopped at its time budget",
        detail=(
            f"Reviewed {reviewed} of {total} diff chunk(s) before the "
            f"{DEFAULT_BUDGET_SECONDS:.0f}s budget ran out, so part of this "
            "diff was not reviewed. Deterministic checks still covered all of "
            "it. Split the PR or raise --budget-seconds to review the rest."
        ),
    )


def llm_skipped_finding(reason: str) -> Finding:
    return Finding(
        id="llm-skipped",
        severity="warn",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="External LLM review was skipped",
        detail=reason,
    )


def summarize_llm_review(
    *,
    chunk_sizes: list[int],
    elapsed_seconds: float,
    returned_count: int,
    kept_count: int,
) -> str:
    dropped = returned_count - kept_count
    return (
        f"LLM review: {len(chunk_sizes)} chunk(s) "
        f"({', '.join(f'{size} chars' for size in chunk_sizes)}) "
        f"in {elapsed_seconds:.1f}s; "
        f"{returned_count} finding(s) returned, {kept_count} kept, "
        f"{dropped} dropped by diff-anchor filtering."
    )


def format_report(
    *,
    base: str,
    head: str,
    verdict: str,
    findings: list[Finding],
    waived: bool,
    llm_summary: str | None = None,
) -> str:
    lines = [
        f"Repository rules review ({base}...{head})",
        f"Verdict: {verdict}",
    ]
    if llm_summary:
        lines.append(llm_summary)
    if waived:
        lines.append("Waived by maintainer label.")
    if not findings:
        lines.append("No findings.")
        return "\n".join(lines)

    lines.append("Findings:")
    for finding in findings:
        location = finding.file or "<unknown>"
        if finding.line is not None:
            location = f"{location}:{finding.line}"
        lines.append(
            f"- [{finding.severity}] {finding.id} ({finding.rule_section}) "
            f"{location}: {finding.summary}"
        )
        if finding.detail:
            lines.append(f"  {finding.detail}")
    return "\n".join(lines)


def runtime_boundary_files_in_worktree() -> list[str]:
    runtime = ROOT / "crates" / "tenferro-runtime"
    checked = [runtime / "Cargo.toml"]
    checked.extend(sorted((runtime / "src").rglob("*.rs")))
    return [path.relative_to(ROOT).as_posix() for path in checked if path.is_file()]


def runtime_boundary_files_at_ref(ref: str) -> list[str]:
    output = run_git(
        ["ls-tree", "-r", "--name-only", ref, "--", "crates/tenferro-runtime"]
    )
    return [
        path
        for path in output.splitlines()
        if path == "crates/tenferro-runtime/Cargo.toml"
        or (
            path.startswith("crates/tenferro-runtime/src/")
            and path.endswith(".rs")
        )
    ]


def runtime_boundary_text(path: str, *, ref: str | None, worktree: bool) -> str:
    if worktree:
        return (ROOT / path).read_text(encoding="utf-8")
    if ref is None:
        raise ValueError("ref is required when worktree is false")
    return run_git(["show", f"{ref}:{path}"])


def strip_code_comments(line: str, in_block_comment: bool) -> tuple[str, bool]:
    result: list[str] = []
    index = 0
    while index < len(line):
        if in_block_comment:
            end = line.find("*/", index)
            if end == -1:
                return "".join(result), True
            index = end + 2
            in_block_comment = False
            continue
        if line.startswith("/*", index):
            in_block_comment = True
            index += 2
            continue
        if line.startswith("//", index):
            break
        result.append(line[index])
        index += 1
    return "".join(result), in_block_comment


def scan_runtime_boundary_text(
    path: str,
    text: str,
    line_numbers: set[int] | None = None,
) -> list[str]:
    violations: list[str] = []
    in_block_comment = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        code_line, in_block_comment = strip_code_comments(line, in_block_comment)
        if line_numbers is not None and line_no not in line_numbers:
            continue
        if path.endswith("Cargo.toml") and code_line.lstrip().startswith("#"):
            code_line = ""
        if RUNTIME_AD_FORBIDDEN.search(code_line):
            violations.append(f"{path}:{line_no}: {line}")
    return violations


def runtime_ad_boundary_violations(
    *,
    ref: str | None,
    worktree: bool,
    changed_lines: dict[str, set[int]] | None = None,
) -> list[str]:
    resolved_ref = ref or "HEAD"
    if changed_lines is None:
        paths = (
            runtime_boundary_files_in_worktree()
            if worktree
            else runtime_boundary_files_at_ref(resolved_ref)
        )
    else:
        paths = sorted(
            path
            for path, lines in changed_lines.items()
            if lines
            and (
                path == "crates/tenferro-runtime/Cargo.toml"
                or (
                    path.startswith("crates/tenferro-runtime/src/")
                    and path.endswith(".rs")
                )
            )
        )
    violations: list[str] = []
    for path in paths:
        text = runtime_boundary_text(path, ref=resolved_ref, worktree=worktree)
        line_numbers = changed_lines.get(path) if changed_lines is not None else None
        violations.extend(scan_runtime_boundary_text(path, text, line_numbers))
    return violations


def deterministic_checks(
    files: list[str],
    *,
    head: str | None = None,
    worktree: bool = False,
    added_lines: dict[str, set[int]] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    runtime_touched = any(path.startswith("crates/tenferro-runtime/") for path in files)
    if runtime_touched:
        violations = runtime_ad_boundary_violations(
            ref=head,
            worktree=worktree,
            changed_lines=added_lines,
        )
        if violations:
            findings.append(
                Finding(
                    id="ad-boundary-runtime",
                    severity="block",
                    rule_section="Rule Source Of Truth",
                    file="crates/tenferro-runtime",
                    line=None,
                    summary="AD symbols leaked into tenferro-runtime boundary",
                    detail="\n".join(violations),
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Merge base ref")
    parser.add_argument("--head", default="HEAD", help="Head ref (default: HEAD)")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write machine-readable report JSON to this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM call; run diff/section selection and deterministic checks only",
    )
    parser.add_argument(
        "--waived",
        action="store_true",
        help="Treat review as waived (maintainer label in CI)",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        metavar="PATH",
        help="Load environment from this dotenv file (default: .env at repo root when present)",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Do not load .env even when the file exists",
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="Diff the working tree against --base (includes uncommitted changes)",
    )
    parser.add_argument(
        "--llm-skipped-reason",
        help="Record a maintainer-approved reason when --dry-run intentionally skips LLM review",
    )
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-url",
        default=os.environ.get("DEEPSEEK_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=DEFAULT_BUDGET_SECONDS,
        help="Wall-clock ceiling for all LLM requests, so the job is never "
        "killed before the report is written",
    )
    args = parser.parse_args(argv)

    if args.llm_skipped_reason and not args.dry_run:
        print("--llm-skipped-reason requires --dry-run", file=sys.stderr)
        return 1

    configure_dotenv(explicit=args.dotenv, skip=args.no_dotenv)

    if not RULES_PATH.is_file():
        print(f"Missing rules file: {RULES_PATH}", file=sys.stderr)
        return 1
    if not PROMPT_PATH.is_file():
        print(f"Missing prompt file: {PROMPT_PATH}", file=sys.stderr)
        return 1

    files = changed_files(args.base, args.head, worktree=args.worktree)
    if not files:
        report = {
            "verdict": "pass",
            "waived": args.waived,
            "findings": [],
            "summary": "No changed files; review skipped.",
        }
        print(json.dumps(report, indent=2))
        return 0

    diff_text = unified_diff(args.base, args.head, worktree=args.worktree)
    added_lines = added_lines_by_file(diff_text)
    added_text = added_lines_with_text(diff_text)
    section_names = select_rule_sections(files, added_text)
    rules_text = build_rules_payload(section_names)
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    findings = deterministic_checks(
        files,
        head=args.head,
        worktree=args.worktree,
        added_lines=added_lines,
    )
    sensitive_finding = sensitive_diff_finding(diff_text)
    if sensitive_finding:
        findings.append(sensitive_finding)
    if args.llm_skipped_reason:
        findings.append(llm_skipped_finding(args.llm_skipped_reason))

    if args.waived:
        report_body = format_report(
            base=args.base,
            head=args.head,
            verdict="pass",
            findings=findings,
            waived=True,
        )
        print(report_body)
        payload = {
            "verdict": "pass",
            "waived": True,
            "findings": [item.to_dict() for item in findings],
            "changed_files": files,
            "rule_sections": section_names,
        }
        if args.output_json:
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    llm_summary: str | None = None
    llm_stats: dict[str, Any] | None = None
    if not args.dry_run and not sensitive_finding:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            print("DEEPSEEK_API_KEY is not set", file=sys.stderr)
            return 1
        key_problem = api_key_problem(api_key)

        if key_problem:
            print(f"DEEPSEEK_API_KEY is unusable: {key_problem}", file=sys.stderr)
            findings.append(api_key_error_finding(key_problem))
            chunks = []
        else:
            file_diffs = per_file_diffs(
                args.base,
                args.head,
                files,
                worktree=args.worktree,
            )
            chunks = split_diff_chunks(redact_file_diffs(file_diffs))
        chunk_sizes = [len(chunk) for chunk in chunks]
        print(
            f"LLM review: model {args.model}, {len(chunks)} chunk(s), "
            f"sizes {chunk_sizes} chars",
            file=sys.stderr,
        )
        llm_findings: list[Finding] = []
        llm_started = time.monotonic()
        llm_deadline = llm_started + args.budget_seconds
        reviewed = 0
        for index, chunk in enumerate(chunks, start=1):
            chunk_started = time.monotonic()
            if chunk_started >= llm_deadline:
                print(
                    f"LLM review: budget exhausted after {reviewed}/{len(chunks)} "
                    "chunk(s)",
                    file=sys.stderr,
                )
                findings.append(budget_exhausted_finding(reviewed, len(chunks)))
                break
            try:
                _, chunk_findings = review_chunk(
                    api_key=api_key,
                    model=args.model,
                    api_url=args.api_url,
                    system_prompt=system_prompt,
                    rules_text=rules_text,
                    changed=files,
                    diff_chunk=chunk,
                    timeout=args.timeout,
                    deadline=llm_deadline,
                )
            except (KeyError, ValueError, *TRANSPORT_ERRORS) as exc:
                print(
                    f"LLM chunk {index}/{len(chunks)}: failed after "
                    f"{time.monotonic() - chunk_started:.1f}s: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                findings.append(llm_response_error_finding(exc))
                break
            print(
                f"LLM chunk {index}/{len(chunks)}: {len(chunk)} chars, "
                f"{len(chunk_findings)} finding(s), "
                f"{time.monotonic() - chunk_started:.1f}s",
                file=sys.stderr,
            )
            llm_findings.extend(chunk_findings)
            reviewed += 1
        merged_llm_findings = merge_findings(llm_findings)
        kept_llm_findings = filter_findings(
            merged_llm_findings,
            files,
            added_lines,
            allow_global=False,
        )
        llm_elapsed = time.monotonic() - llm_started
        llm_summary = summarize_llm_review(
            chunk_sizes=chunk_sizes,
            elapsed_seconds=llm_elapsed,
            returned_count=len(merged_llm_findings),
            kept_count=len(kept_llm_findings),
        )
        llm_stats = {
            "chunk_sizes": chunk_sizes,
            "elapsed_seconds": round(llm_elapsed, 3),
            "findings_returned": len(merged_llm_findings),
            "findings_kept": len(kept_llm_findings),
        }
        findings.extend(kept_llm_findings)

    block_findings = [item for item in findings if item.severity == "block"]
    verdict = reconcile_verdict(block_findings)

    report_body = format_report(
        base=args.base,
        head=args.head,
        verdict=verdict,
        findings=findings,
        waived=False,
        llm_summary=llm_summary,
    )
    print(report_body)

    payload = {
        "verdict": verdict,
        "waived": False,
        "findings": [item.to_dict() for item in findings],
        "block_findings": [item.to_dict() for item in block_findings],
        "changed_files": files,
        "rule_sections": section_names,
        "prompt_version": PROMPT_VERSION,
        "llm_review": llm_stats,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 1 if block_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
