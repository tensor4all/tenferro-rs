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
PROMPT_VERSION = "2"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
MAX_DIFF_CHARS = 60_000
MAX_FILE_DIFF_CHARS = 40_000
MAX_FINDINGS_PER_CHUNK = 8
HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)
# Inline `#[cfg(test)] mod ... { ... }` blocks are exempt only for genuinely
# tiny leaf modules (Unit Test Organization).
INLINE_TEST_EXEMPT_FILE_LINES = 150
INLINE_TEST_EXEMPT_BLOCK_LINES = 60
# Section routed for changed paths that match no other trigger or that live
# outside the established top-level directories.
FALLBACK_SECTION = "PR Content Hygiene"
KNOWN_TOP_LEVEL_DIRS = frozenset(
    {
        "crates",
        "docs",
        "scripts",
        "ext",
        "samples",
        "benches",
        "benchmarks",
        "ai",
        "src",
        "tests",
        ".github",
        ".claude",
        ".agents",
        ".opencode",
        ".kimi",
        ".cargo",
    }
)
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
SECRET_ANNOTATION = r"(?:[ \t]*:[^=\n]{0,40})?"
# The value alternatives are ordered quoted-first so a quoted secret is
# consumed whole. Putting the bare-token alternative first would stop at the
# first space inside a quoted passphrase and upload the remainder. (Spelling
# out an example assignment here would make this file trip its own guard.)
SECRET_VALUE = r"""(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s#]+)"""
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")(?P<sep>" + SECRET_ANNOTATION
    + r"[ \t]*[:=][ \t]*)(?P<value>" + SECRET_VALUE + r")"
)
# Quoted credentials may legitimately contain spaces (a diceware passphrase is
# prose by construction), so the value shape cannot be the discriminator.
QUOTED_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")" + SECRET_ANNOTATION + r"\s*[:=]\s*"
    r"""(?:"[^"\r\n]{8,}"|'[^'\r\n]{8,}')"""
)
# An assignment whose value has not started yet on this line: the credential
# lands on the following line, where no credential-shaped name is in sight.
OPEN_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")" + SECRET_ANNOTATION + r"[ \t]*[:=][ \t]*$"
)
# A value standing alone on its own line, as the continuation of the above.
# The value may be a bare token: `API_KEY =` followed by an unquoted secret.
# `awaiting_value` is only set for credential-named assignments, so accepting
# bare tokens here cannot fire on ordinary continuations.
#
# The bare alternative is restricted to the characters a credential literal is
# actually made of. An unrestricted `[^\s#]+` also matched an EXPRESSION, so
# `let api_key =` continued by `std::env::var("API_KEY")?;` was reported as a
# leaked credential and a valid credential-LOADING change could not pass the
# required gate. Call, path, and index syntax (`(`, `)`, `::`, `?`, `[`, `"`)
# is outside the class, so such continuations no longer match.
#
# `.` is excluded too, which costs the unquoted-JWT continuation shape (a
# quoted one still matches the alternatives above, and the `Bearer` form is
# covered by SECRET_VALUE_PATTERNS). A dotted token is far more often a FIELD
# ACCESS — `let api_key =` continued by `settings.api_key;` — and blocking
# credential-loading code on the required gate is the worse error of the two.
#
# Length and word-shape carry the rest of the discrimination, because a plain
# IDENTIFIER is spelled from the same alphabet as a credential:
# `let api_key =` continued by `configured_token;` or `ENV_API_KEY;` is
# ordinary code with no secret in the diff.
#   * 20 characters is the floor the file already uses for a bare credential
#     (`github_pat_…{20,}`, `gh[pousr]_…{20,}`, `sk-…{20,}`); the reported
#     identifiers are 16 and 11.
#   * a snake_case / SCREAMING_SNAKE_CASE token is an identifier by
#     convention, while base64/hex credentials mix case and digits, so
#     `IDENTIFIER_WORD_SHAPE` rejects the former even past the length floor.
# A long single-word identifier with no underscore remains a residual false
# positive; it is waivable, whereas the reverse error would upload a secret.
BARE_SECRET_VALUE = r"[A-Za-z0-9][A-Za-z0-9_~+/=-]{19,}"
IDENTIFIER_WORD_SHAPE = re.compile(
    r"^(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)$"
)
STANDALONE_VALUE = re.compile(
    r"""^[ \t]*(?P<value>"[^"\r\n]{4,}"|'[^'\r\n]{4,}'|"""
    + BARE_SECRET_VALUE
    + r""")[ \t]*[,;]?[ \t]*$"""
)


def is_standalone_secret_value(body: str) -> bool:
    """Whether a continuation line carries a credential value rather than code."""
    match = STANDALONE_VALUE.match(body)
    if not match:
        return False
    return not IDENTIFIER_WORD_SHAPE.match(match.group("value"))
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
                "PR Content Hygiene",
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
                # Inline `#[cfg(test)]` violations happen in normal src
                # files, so the unit-test rule must load for every Rust
                # change, not only for tests paths.
                "Unit Test Organization",
                # The doc-example mandate applies to Rust public items, not
                # only to Markdown changes.
                "Documentation Policy",
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
        matched = False
        for pattern, section_names in SECTION_TRIGGERS:
            if pattern.search(path):
                selected.update(section_names)
                matched = True
        # Paths outside every routed area or outside the established
        # top-level directories (for example `.superpowers/` or a new
        # `new-crate/src/lib.rs`) must load the PR-content rules even when
        # another trigger such as `.rs$` already matched.
        top_level = path.split("/", 1)[0] if "/" in path else None
        if not matched or (
            top_level is not None and top_level not in KNOWN_TOP_LEVEL_DIRS
        ):
            selected.add(FALLBACK_SECTION)

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


def files_with_unanchorable_deletions(diff_text: str) -> set[str]:
    """Files containing at least one hunk that removes lines and adds none.

    A finding about a deletion has no new-file line to point at, so the model
    is obliged to return ``line: null`` and `filter_findings` must keep the
    block. That obligation holds exactly where the diff gives it nothing to
    anchor to, and the unit of "nothing to anchor to" is the HUNK, not the
    file:

    * a replacement edit adds the lines that took the deleted ones' place, so
      a real finding about it can and should name one of them — judging this
      per file would disable the anti-generalization filter for any patch that
      happens to remove a line;
    * but an unrelated addition elsewhere in the same file is not a valid
      anchor for a deletion-only hunk — judging this per file would drop a
      real block about validation or a ``// SAFETY:`` comment removed in a
      mixed-hunk diff.

    Keyed by the new-side path, falling back to the old-side path when the file
    is deleted outright (``+++ /dev/null``) — that is the path a finding about
    the removal names, and without the fallback a whole-file deletion was
    omitted from the very set that exists to retain it.

    File headers are only read OUTSIDE a hunk. Inside one, ``--- validation``
    is the deletion of a source line reading ``-- validation``, not an
    old-file header; treating it as a header reset the hunk flags and lost the
    file. ``diff --git`` returns the parser to the header state.
    """
    result: set[str] = set()
    current_file: str | None = None
    old_file: str | None = None
    in_hunk = False
    hunk_deleted = False
    hunk_added = False

    def close_hunk() -> None:
        if current_file and hunk_deleted and not hunk_added:
            result.add(current_file)

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            close_hunk()
            in_hunk = False
            hunk_deleted = hunk_added = False
            current_file = old_file = None
            continue
        if line.startswith("@@"):
            close_hunk()
            in_hunk = True
            hunk_deleted = hunk_added = False
            continue
        if not in_hunk:
            if line.startswith("--- "):
                raw = line.removeprefix("--- a/").removeprefix("--- ")
                old_file = None if raw == "/dev/null" else raw
                continue
            if line.startswith("+++ "):
                raw = line.removeprefix("+++ b/").removeprefix("+++ ")
                current_file = old_file if raw == "/dev/null" else raw
                continue
            continue
        if current_file is None:
            continue
        if line.startswith("-"):
            hunk_deleted = True
        elif line.startswith("+"):
            hunk_added = True
    close_hunk()
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
    """Locate the first added line that must not be uploaded.

    Lines are examined in diff order rather than per file, because a credential
    can be split across two: the assignment stays unchanged on a context line
    and only the value line is replaced. Checking added lines in isolation sees
    a bare string literal with no credential-shaped name anywhere on it.
    """
    current_file: str | None = None
    new_line = 0
    # Set when the previous surviving line opened an assignment whose value has
    # not appeared yet, so the next line's literal is that value.
    awaiting_value = False

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw = line.removeprefix("+++ b/").removeprefix("+++ ")
            current_file = None if raw == "/dev/null" else raw
            awaiting_value = False
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            new_line = int(match.group(1)) if match else 0
            awaiting_value = False
            continue
        if current_file is None:
            continue

        if line.startswith("-") and not line.startswith("---"):
            # A deleted line neither survives nor breaks the continuation.
            continue
        if not (line.startswith("+") or line.startswith(" ")):
            continue

        body = line[1:]
        is_added = line.startswith("+") and not line.startswith("+++")

        if is_added and contains_sensitive_text(body):
            return current_file, new_line
        if is_added and awaiting_value and is_standalone_secret_value(body):
            return current_file, new_line

        opener = OPEN_SECRET_ASSIGNMENT.search(body)
        awaiting_value = bool(opener) and is_credential_name(opener.group("name"))
        if is_added:
            new_line += 1
        else:
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
    files_with_deletions: set[str] | None = None,
) -> list[Finding]:
    """Keep only findings anchored to something this diff actually changed.

    A file-level `block` is normally dropped, because an unanchored block is
    usually the model generalising about the file rather than about the diff.
    The exception is a violation introduced by *deleting* required validation,
    coverage, or a safety comment: there is no new-file line to point at, so
    the model must return `line: null`, and dropping it would let a
    deletion-only diff pass the gate.
    """
    allowed_files = set(files)
    deletions = files_with_deletions or set()
    kept: list[Finding] = []
    for finding in findings:
        if not finding.file:
            if allow_global:
                kept.append(finding)
            continue
        if finding.file and finding.file not in allowed_files:
            continue
        if (
            finding.line is None
            and finding.severity == "block"
            and finding.file not in deletions
        ):
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


def budget_exhausted_finding(reviewed: int, total: int, budget: float) -> Finding:
    return Finding(
        id="llm-budget-exhausted",
        severity="warn",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="External LLM review stopped at its time budget",
        detail=(
            f"Reviewed {reviewed} of {total} diff chunk(s) before the "
            f"{budget:.0f}s budget ran out, so part of this "
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


def strip_code_comments_and_literals(
    line: str, state: tuple[bool, bool, int | None]
) -> tuple[str, tuple[bool, bool, int | None]]:
    """Drop comments AND string/char literal contents, for brace counting.

    A brace inside a literal is not structural: `let expected = "}";` inside an
    inline test module ended the block at that line, so a test added below fell
    outside the computed span and evaded the audit entirely. Only the scanners
    that COUNT braces use this; `scan_runtime_boundary_text` keeps
    `strip_code_comments`, because a forbidden symbol appearing inside a string
    is still worth reporting there.

    `state` is `(in_block_comment, in_string, raw_hashes)`; `raw_hashes` is the
    `#` count of an open raw string (`r#"..."#`), or `None` when not in one.
    Rust normal strings and raw strings may span lines, hence the carried state.
    """
    in_block_comment, in_string, raw_hashes = state
    result: list[str] = []
    index = 0
    length = len(line)
    while index < length:
        if in_block_comment:
            end = line.find("*/", index)
            if end == -1:
                return "".join(result), (True, False, None)
            index = end + 2
            in_block_comment = False
            continue
        if raw_hashes is not None:
            closer = '"' + "#" * raw_hashes
            end = line.find(closer, index)
            if end == -1:
                return "".join(result), (False, False, raw_hashes)
            index = end + len(closer)
            raw_hashes = None
            continue
        if in_string:
            index, in_string = _scan_string_body(line, index)
            continue
        if line.startswith("/*", index):
            in_block_comment = True
            index += 2
            continue
        if line.startswith("//", index):
            break
        raw = RAW_STRING_OPEN.match(line, index)
        if raw:
            raw_hashes = len(raw.group(1))
            index = raw.end()
            continue
        if line[index] == '"':
            index, in_string = _scan_string_body(line, index + 1)
            continue
        if line[index] == "'":
            # `'a` is a lifetime, `'x'` a char literal. Only the literal hides
            # braces, and only it has a closing quote on the same line.
            char = CHAR_LITERAL.match(line, index)
            if char:
                index = char.end()
            else:
                index += 1
            continue
        result.append(line[index])
        index += 1
    return "".join(result), (in_block_comment, in_string, raw_hashes)


def _scan_string_body(line: str, index: int) -> tuple[int, bool]:
    """Consume a normal string from `index`; return (next index, still open)."""
    while index < len(line):
        if line[index] == "\\":
            index += 2
            continue
        if line[index] == '"':
            return index + 1, False
        index += 1
    return index, True


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


RUST_TEST_PATH = re.compile(r"(^|/)tests(/|\.rs$)|_tests\.rs$|(^|/)benches/|(^|/)examples/")
INLINE_TEST_MOD = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{")
# `union` belongs here with the other public type forms: a public union is as
# much a documented public type as a struct or enum, and omitting it let one
# slip past the doc-example audit entirely.
PUB_ITEM = re.compile(
    r"^\s*pub\s+(?:async\s+|unsafe\s+|const\s+|extern\s+\"[^\"]*\"\s+)*"
    r"(fn|struct|enum|union|trait|type)\s+([A-Za-z_]\w*)"
)
AI_REPORT_PATH = re.compile(r"(^|/)\.superpowers/|-report\.md$")
RUST_MOD_OPEN = re.compile(r"^(?:pub(\([^)]*\))?\s+)?mod\s+\w+\s*\{")
# `r"..."`, `r#"..."#`, `br"..."` — the byte/raw prefixes rustc accepts.
RAW_STRING_OPEN = re.compile(r'(?:b?r)(#*)"')
# `'x'`, `'\\n'`, `'\\u{1F600}'` — a lifetime has no closing quote.
CHAR_LITERAL = re.compile(r"'(?:\\\\(?:u\\{[0-9a-fA-F]{1,6}\\}|x[0-9a-fA-F]{2}|.)|[^\\\\'])'")
CFG_ATTR = re.compile(r"^#\[cfg\((.*)\)\]\s*$")
CFG_PREDICATE_CALL = re.compile(r"^(all|any|not)\s*\((.*)\)$", re.DOTALL)


def split_cfg_operands(expression: str) -> list[str]:
    """Split a cfg operand list on commas that are not inside parens or strings."""
    operands: list[str] = []
    depth = 0
    in_string = False
    current: list[str] = []
    for char in expression:
        if in_string:
            current.append(char)
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        operands.append(tail)
    return [operand for operand in operands if operand]


def cfg_expression_enables_test(expression: str, *, negated: bool = False) -> bool:
    """True when `test` appears in a POSITIVE position of a cfg expression.

    The cfg grammar nests, so polarity has to be tracked structurally rather
    than by deleting a literal `not(test)` substring: in
    `not(any(test, feature = "cuda"))` the item is compiled when `test` is
    *off*, yet a bare `test` token is still present. Each enclosing `not`
    flips polarity, so a `test` operand under an odd number of them does not
    gate the item on tests.
    """
    expression = expression.strip()
    if not expression:
        return False
    call = CFG_PREDICATE_CALL.match(expression)
    if call:
        name, inner = call.group(1), call.group(2)
        if name == "not":
            return cfg_expression_enables_test(inner, negated=not negated)
        return any(
            cfg_expression_enables_test(operand, negated=negated)
            for operand in split_cfg_operands(inner)
        )
    # A leaf: a bare identifier (`test`, `unix`) or a `key = "value"` predicate.
    return expression == "test" and not negated


def is_cfg_test_attr(line: str) -> bool:
    """True for cfg attributes that compile the item *because* tests are on.

    Recognizes `#[cfg(test)]`, `#[cfg(all(test, ...))]`, and
    `#[cfg(all(..., test))]`. Attributes where every `test` operand sits under
    an odd number of `not`s — `#[cfg(not(test))]`,
    `#[cfg(not(any(test, feature = "cuda")))]` — describe production-only items
    and are not test gates.
    """
    match = CFG_ATTR.match(line)
    if not match:
        return False
    return cfg_expression_enables_test(match.group(1))


# A doc-example line that binds or names a path without calling anything.
VACUOUS_EXAMPLE_LINE = re.compile(
    r"^(?:let\s+_\w*\s*(?::[^=]{0,120})?=\s*)?[A-Za-z_][\w:<>]*;$"
)
# Boilerplate lines that neither prove nor disprove real usage. Comment-only
# lines belong here too: prose above `let _method = Widget::spin;` does not
# turn the binding into real API usage, and leaving comments in the classified
# set let an assignment-only example escape the audit entirely.
VACUOUS_IGNORE_LINE = re.compile(r"^(?:use\s|#|//)")
# A rustdoc fence with no info string is Rust; an info string is a
# comma/space-separated attribute list, and rustdoc only treats it as code when
# every token is a Rust attribute. A ```text / ```bash / ```toml block is
# prose or another language's syntax, not a doctest, so classifying its
# contents as a vacuous EXAMPLE was a false positive on ordinary docs.
RUST_FENCE_ATTRS = frozenset(
    {
        "rust",
        "ignore",
        "no_run",
        "should_panic",
        "compile_fail",
        "edition2015",
        "edition2018",
        "edition2021",
        "edition2024",
        "standalone_crate",
    }
)


def is_rust_doc_fence(info: str) -> bool:
    """Whether a rustdoc fence info string opens a Rust doctest."""
    tokens = [token for token in re.split(r"[,\s]+", info.strip()) if token]
    if not tokens:
        return True
    return all(
        token in RUST_FENCE_ATTRS or token.startswith("ignore-") for token in tokens
    )


CRATE_CARGO_TOML = re.compile(r"^crates/(tenferro-[\w-]+)/Cargo\.toml$")
# TOML permits any spacing around `=`, so `{workspace=true,optional=true}` is
# as valid as the expanded form. Matching the literal `optional = true` string
# recorded compact optional entries as production diagram edges.
OPTIONAL_TRUE = re.compile(r"\boptional\s*=\s*true\b")
DEPENDENCY_DIAGRAM_DOC = "docs/architecture/tenferro-crates.md"
# Edges to these targets are documented in the "Additional internal
# dependencies" prose of the architecture doc rather than in the diagram
# block, so the diagram-sync check skips them.
DIAGRAM_PROSE_TARGETS = frozenset(
    {"tenferro-core-ops", "tenferro-internal-extension-macros"}
)


def changed_file_text(path: str, *, ref: str | None, worktree: bool) -> str | None:
    try:
        return runtime_boundary_text(path, ref=ref, worktree=worktree)
    except (OSError, ValueError, subprocess.CalledProcessError):
        # Deleted or unreadable at this ref; nothing to scan.
        return None


def read_attribute(lines: list[str], start: int) -> tuple[str, int]:
    """Join a possibly multiline `#[...]` attribute into one string.

    Returns the whitespace-normalized attribute text and the 0-based index of
    its last line.
    """
    text = lines[start].strip()
    end = start
    while (
        text.count("[") > text.count("]") or text.count("(") > text.count(")")
    ) and end + 1 < len(lines):
        end += 1
        text += " " + lines[end].strip()
    return text, end


def rust_inline_test_blocks(text: str) -> list[tuple[int, int]]:
    """Return 1-based (start, end) line spans of inline `#[cfg(test)] mod` blocks."""
    lines = text.splitlines()
    total = len(lines)
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < total:
        stripped = lines[index].strip()
        if stripped.startswith("#[cfg("):
            attr_text, attr_end = read_attribute(lines, index)
        else:
            attr_text, attr_end = stripped, index
        if is_cfg_test_attr(attr_text):
            probe = attr_end + 1
            while probe < total:
                probe_stripped = lines[probe].strip()
                if not probe_stripped:
                    probe += 1
                elif probe_stripped.startswith("#["):
                    _, probe = read_attribute(lines, probe)
                    probe += 1
                else:
                    break
            if probe < total and INLINE_TEST_MOD.match(lines[probe].strip()):
                depth = 0
                end = total - 1
                scan_state: tuple[bool, bool, int | None] = (False, False, None)
                for cursor in range(probe, total):
                    code, scan_state = strip_code_comments_and_literals(
                        lines[cursor], scan_state
                    )
                    depth += code.count("{") - code.count("}")
                    if cursor >= probe and depth <= 0:
                        end = cursor
                        break
                blocks.append((index + 1, end + 1))
                index = end
        index += 1
    return blocks


def inline_test_block_line_total(text: str | None) -> int:
    """Total number of lines this file spends on inline `#[cfg(test)]` blocks."""
    if text is None:
        return 0
    return sum(end - start + 1 for start, end in rust_inline_test_blocks(text))


def inline_test_module_findings(
    files: list[str],
    *,
    ref: str | None,
    base: str | None = None,
    worktree: bool,
    added_lines: dict[str, set[int]],
) -> list[Finding]:
    """Report inline `#[cfg(test)]` blocks this diff adds to or grows.

    Growth is judged against the BASE revision, not from the presence of a
    touched line: extraction work that shrinks an oversized block while still
    editing a line inside what remains is progress toward the rule, and
    warning about it would penalize exactly the cleanup the rule asks for.
    Blocks cannot be matched one-to-one across revisions (they move, split and
    merge), so the comparison is the file's net inline-test line count.
    """
    findings: list[Finding] = []
    for path in sorted(files):
        if not path.endswith(".rs") or RUST_TEST_PATH.search(path):
            continue
        touched = added_lines.get(path)
        if not touched:
            continue
        text = changed_file_text(path, ref=ref, worktree=worktree)
        if text is None:
            continue
        grew = True
        if base is not None:
            base_total = inline_test_block_line_total(
                changed_file_text(path, ref=base, worktree=False)
            )
            grew = inline_test_block_line_total(text) > base_total
        file_lines = len(text.splitlines())
        for start, end in rust_inline_test_blocks(text):
            block_lines = end - start + 1
            if not any(start <= line <= end for line in touched):
                continue
            # A file whose inline-test total did not grow is mid-extraction, so
            # an edit inside a surviving block is progress, not a violation.
            # A block whose OPENER is itself an added line is new, though, and
            # a PR that shrinks one block while adding another can lower the
            # total while still introducing a fresh violation — judge that
            # block on its own rather than letting the file-level total hide it.
            if not grew and start not in touched:
                continue
            if (
                file_lines < INLINE_TEST_EXEMPT_FILE_LINES
                and block_lines <= INLINE_TEST_EXEMPT_BLOCK_LINES
            ):
                continue
            findings.append(
                Finding(
                    id="inline-test-module",
                    severity="warn",
                    rule_section="Unit Test Organization",
                    file=path,
                    line=start,
                    summary=(
                        "Inline #[cfg(test)] module added or grown in a "
                        "non-tiny production file"
                    ),
                    detail=(
                        f"{path}:{start} carries an inline test block of "
                        f"~{block_lines} lines in a {file_lines}-line file. "
                        "Unit Test Organization requires module-local "
                        "src/<module>/tests/*.rs files with only "
                        "`#[cfg(test)] mod tests;` left in the source."
                    ),
                )
            )
    return findings


def rust_private_mod_spans(text: str) -> list[tuple[int, int]]:
    """Return 1-based line spans of inline modules that are not plain-`pub`.

    Items inside `mod x { ... }` or `pub(crate) mod x { ... }` are not part of
    the public API even when declared `pub` (the sealed-trait pattern), so the
    doc-example mandate does not apply to them.
    """
    lines = text.splitlines()
    total = len(lines)
    spans: list[tuple[int, int]] = []
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        match = RUST_MOD_OPEN.match(stripped)
        if not match:
            continue
        if stripped.startswith("pub ") and not stripped.startswith("pub("):
            continue
        depth = 0
        end = total - 1
        scan_state: tuple[bool, bool, int | None] = (False, False, None)
        for cursor in range(index, total):
            code, scan_state = strip_code_comments_and_literals(
                lines[cursor], scan_state
            )
            depth += code.count("{") - code.count("}")
            if cursor >= index and depth <= 0:
                end = cursor
                break
        spans.append((index + 1, end + 1))
    return spans


def doc_block_above(lines: list[str], item_index: int) -> tuple[bool, bool]:
    """Return (has_examples, is_doc_hidden) for the doc/attr block above an item."""
    has_examples = False
    hidden = False
    cursor = item_index - 1
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if stripped.startswith("///"):
            if "# Examples" in stripped:
                has_examples = True
        elif stripped.startswith("#["):
            if "doc(hidden)" in stripped:
                hidden = True
        elif stripped.startswith("//"):
            # Plain comments may sit between the doc block and the item.
            pass
        elif stripped.endswith("]") or stripped.endswith(")]"):
            # Continuation tail of a multi-line attribute.
            pass
        else:
            break
        cursor -= 1
    return has_examples, hidden


MOD_DECLARATION = re.compile(
    r"^\s*(pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_]\w*)\s*;"
)


def pub_use_exports(text: str, name: str) -> set[str] | str | None:
    """Item names a declaring file re-exports from module `name` via `pub use`.

    Returns None when no plain `pub use` references the module, the string
    "all" for glob or whole-module re-exports, and otherwise the set of
    re-exported source item names.
    """
    exports: set[str] = set()
    found = False
    for match in re.finditer(r"^\s*pub\s+use\s+([^;]+);", text, re.M):
        clause = re.sub(r"\s+", " ", match.group(1)).strip()
        clause = re.sub(r"^(crate::|self::|super::)+", "", clause)
        if not re.match(re.escape(name) + r"(::|$)", clause):
            continue
        found = True
        rest = clause[len(name) :]
        if not rest.startswith("::"):
            return "all"
        rest = rest[2:].strip()
        if rest == "*":
            return "all"
        if rest.startswith("{"):
            for part in rest.strip("{} ").split(","):
                part = part.strip()
                if not part:
                    continue
                if part == "*" or "::" in part:
                    return "all"
                exports.add(part.split(" as ")[0].strip())
        else:
            exports.add(rest.split(" as ")[0].strip())
    if not found:
        return None
    return exports


def module_public_item_filter(
    path: str,
    *,
    ref: str | None,
    worktree: bool,
    max_depth: int = 12,
) -> tuple[bool, set[str] | None]:
    """Follow out-of-line `mod` declarations toward the crate root.

    Returns (reachable, item_filter). A `pub fn` in a file declared by a
    non-`pub` `mod foo;` is not part of the crate's public API unless the
    declaring file re-exports it via a plain `pub use`; a selective re-export
    yields an item-name filter. When a declaration cannot be located
    (macro-generated or `#[path]`-mapped modules), assume public so real gaps
    are not hidden.
    """
    item_filter: set[str] | None = None
    current = Path(path)
    for _ in range(max_depth):
        name = current.stem
        if name in ("lib", "main"):
            return True, item_filter
        parent = current.parent
        if name == "mod":
            name = parent.name
            parent = parent.parent
        candidates = [parent / "mod.rs", parent.with_suffix(".rs")]
        if parent.name == "src":
            candidates = [parent / "lib.rs", parent / "main.rs"]
        declaration = None
        declaring_file = None
        for candidate in candidates:
            text = changed_file_text(
                candidate.as_posix(), ref=ref, worktree=worktree
            )
            if text is None:
                continue
            for line in text.splitlines():
                match = MOD_DECLARATION.match(line)
                if match and match.group(2) == name:
                    declaration = match
                    declaring_file = candidate
                    break
            if declaration:
                break
        if declaration is None:
            return True, item_filter
        visibility = declaration.group(1) or ""
        if not visibility.startswith("pub") or visibility.startswith("pub("):
            # A private module can still expose items via a plain `pub use`
            # re-export in the declaring file; a selective re-export exposes
            # only the named items.
            declaring_text = changed_file_text(
                declaring_file.as_posix(), ref=ref, worktree=worktree
            )
            exports = (
                pub_use_exports(declaring_text, name)
                if declaring_text is not None
                else None
            )
            if exports is None:
                return False, None
            if exports != "all" and item_filter is None:
                item_filter = set(exports)
        current = declaring_file
    return True, item_filter


def module_publicly_reachable(
    path: str,
    *,
    ref: str | None,
    worktree: bool,
    max_depth: int = 12,
) -> bool:
    reachable, _ = module_public_item_filter(
        path, ref=ref, worktree=worktree, max_depth=max_depth
    )
    return reachable


def missing_doc_example_findings(
    files: list[str],
    *,
    ref: str | None,
    worktree: bool,
    added_lines: dict[str, set[int]],
) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(files):
        if not path.endswith(".rs") or RUST_TEST_PATH.search(path):
            continue
        touched = added_lines.get(path)
        if not touched:
            continue
        text = changed_file_text(path, ref=ref, worktree=worktree)
        if text is None:
            continue
        reachable, item_filter = module_public_item_filter(
            path, ref=ref, worktree=worktree
        )
        if not reachable:
            continue
        lines = text.splitlines()
        skip_spans = rust_inline_test_blocks(text) + rust_private_mod_spans(text)
        missing: list[str] = []
        first_line: int | None = None
        for line_no in sorted(touched):
            if line_no > len(lines):
                continue
            match = PUB_ITEM.match(lines[line_no - 1])
            if not match:
                continue
            if any(start <= line_no <= end for start, end in skip_spans):
                continue
            if item_filter is not None and match.group(2) not in item_filter:
                # The declaring module re-exports selectively and this item
                # is not part of the public surface.
                continue
            has_examples, hidden = doc_block_above(lines, line_no - 1)
            if has_examples or hidden:
                continue
            missing.append(f"{match.group(1)} {match.group(2)} (line {line_no})")
            if first_line is None:
                first_line = line_no
        if missing:
            shown = ", ".join(missing[:10])
            extra = f", +{len(missing) - 10} more" if len(missing) > 10 else ""
            findings.append(
                Finding(
                    id="missing-doc-examples",
                    severity="warn",
                    rule_section="Documentation Policy",
                    file=path,
                    line=first_line,
                    summary=(
                        "New public items lack the mandatory /// # Examples "
                        "doctest"
                    ),
                    detail=(
                        f"{path} adds public items without a `# Examples` "
                        f"doc section: {shown}{extra}. Every public type, "
                        "trait, and function must include a runnable doc "
                        "example."
                    ),
                )
            )
    return findings


def vacuous_doc_example_findings(
    files: list[str],
    *,
    ref: str | None,
    worktree: bool,
    added_lines: dict[str, set[int]],
) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(files):
        if not path.endswith(".rs"):
            continue
        touched = added_lines.get(path)
        if not touched:
            continue
        text = changed_file_text(path, ref=ref, worktree=worktree)
        if text is None:
            continue
        lines = text.splitlines()
        in_fence = False
        fence_is_rust = True
        fence_start = 0
        fence_code: list[str] = []
        reported: list[int] = []
        for line_no, raw in enumerate(lines, start=1):
            stripped = raw.strip()
            if not stripped.startswith("///"):
                in_fence = False
                fence_code = []
                continue
            body = stripped[3:].strip()
            if body.startswith("```"):
                if in_fence:
                    code = [
                        item
                        for item in fence_code
                        if item and not VACUOUS_IGNORE_LINE.match(item)
                    ]
                    fenced_span = range(fence_start, line_no + 1)
                    if (
                        fence_is_rust
                        and code
                        and all(VACUOUS_EXAMPLE_LINE.match(item) for item in code)
                        and any(line in touched for line in fenced_span)
                    ):
                        reported.append(fence_start)
                    in_fence = False
                    fence_code = []
                else:
                    in_fence = True
                    fence_is_rust = is_rust_doc_fence(body[3:])
                    fence_start = line_no
            elif in_fence:
                fence_code.append(body)
        for start in reported:
            findings.append(
                Finding(
                    id="vacuous-doc-example",
                    severity="warn",
                    rule_section="Documentation Policy",
                    file=path,
                    line=start,
                    summary="Doc example demonstrates no usage",
                    detail=(
                        f"{path}:{start} contains a doc example consisting "
                        "only of path/assignment statements (no calls). It "
                        "satisfies the doctest gate without showing how to "
                        "use the API; replace it with a real usage example."
                    ),
                )
            )
    return findings


def ai_report_file_findings(
    files: list[str],
    *,
    ref: str | None = None,
    worktree: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(files):
        if not AI_REPORT_PATH.search(path):
            continue
        if path.startswith("docs/worklogs/"):
            continue
        if changed_file_text(path, ref=ref, worktree=worktree) is None:
            # The change deletes the report file; that is the remediation,
            # not a violation.
            continue
        findings.append(
            Finding(
                id="ai-report-file",
                severity="warn",
                rule_section="PR Content Hygiene",
                file=path,
                line=None,
                summary="Standalone AI-generated report file in the PR",
                detail=(
                    f"{path} looks like an AI-generated analysis or task "
                    "report committed as a standalone file. Fold durable "
                    "content into docs/worklogs/ and drop the report file."
                ),
            )
        )
    return findings


# `[target.'cfg(unix)'.dependencies]` is an ordinary production dependency
# table for that target. Matching the section name exactly ignored it, so a
# real production edge read as absent — a matching diagram edge was reported
# stale, and a missing one passed. The target spec may be quoted (and then may
# itself contain dots), so strip it before classifying the section.
# `[target.<spec>.dev-dependencies]` keeps falling through, exactly like the
# plain `[dev-dependencies]` table.
TARGET_TABLE_PREFIX = re.compile(r"^target\.(?:'[^']*'|\"[^\"]*\"|[^.]+)\.")


def parse_cargo_tenferro_dependencies(text: str) -> set[str]:
    deps: set[str] = set()
    section = ""
    table_name: str | None = None
    table_optional = False
    for raw in text.splitlines():
        line = raw.strip()
        header = re.match(r"^\[(.+)\]$", line)
        if header:
            if table_name and not table_optional:
                deps.add(table_name)
            table_name = None
            table_optional = False
            section = TARGET_TABLE_PREFIX.sub("", header.group(1))
            table = re.match(r"^dependencies\.(tenferro-[\w-]+)$", section)
            if table:
                table_name = table.group(1)
            continue
        if section == "dependencies":
            entry = re.match(r"^(tenferro-[\w-]+)(?:\.[\w-]+)?\s*=", line)
            if entry and not OPTIONAL_TRUE.search(line):
                deps.add(entry.group(1))
        elif table_name is not None and OPTIONAL_TRUE.match(line):
            table_optional = True
    if table_name and not table_optional:
        deps.add(table_name)
    return deps


def parse_dependency_diagram(doc_text: str) -> dict[str, set[str]] | None:
    heading = doc_text.find("Dependency Direction")
    if heading == -1:
        return None
    fence_start = doc_text.find("```", heading)
    if fence_start == -1:
        return None
    fence_end = doc_text.find("```", fence_start + 3)
    if fence_end == -1:
        return None
    block = doc_text[fence_start:fence_end].splitlines()[1:]
    edges: dict[str, set[str]] = {}
    current: str | None = None
    for raw in block:
        entry = re.match(r"^(tenferro-[\w-]+)\s*(?:->\s*(.*))?$", raw)
        if entry:
            current = entry.group(1)
            edges.setdefault(current, set())
            targets = entry.group(2) or ""
        elif raw.startswith((" ", "\t")) and current is not None:
            targets = raw.strip().removeprefix("->").strip()
        else:
            current = None
            continue
        for target in targets.split(","):
            target = target.strip()
            if target.startswith("tenferro-"):
                edges[current].add(target)
    return edges


def list_crate_manifests(*, ref: str | None, worktree: bool) -> list[str]:
    if worktree:
        return sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "crates").glob("tenferro-*/Cargo.toml")
        )
    output = run_git(["ls-tree", "-r", "--name-only", ref or "HEAD", "--", "crates"])
    return sorted(path for path in output.splitlines() if CRATE_CARGO_TOML.match(path))


def dependency_diagram_findings(
    files: list[str],
    *,
    ref: str | None,
    worktree: bool,
) -> list[Finding]:
    changed_manifests = [path for path in files if CRATE_CARGO_TOML.match(path)]
    doc_changed = DEPENDENCY_DIAGRAM_DOC in files
    if not changed_manifests and not doc_changed:
        return []
    doc_text = changed_file_text(DEPENDENCY_DIAGRAM_DOC, ref=ref, worktree=worktree)
    if doc_text is None:
        return []
    diagram = parse_dependency_diagram(doc_text)
    if diagram is None:
        return []
    # A diagram-only change must be validated against every crate manifest,
    # not only the manifests touched by the same PR.
    manifests = set(changed_manifests)
    findings: list[Finding] = []
    if doc_changed:
        all_manifests = list_crate_manifests(ref=ref, worktree=worktree)
        manifests.update(all_manifests)
        # Enumerating the manifests only covers `manifest_crates - diagram`.
        # The opposite direction — a diagram node with no manifest at all —
        # never enters the loop below, so an invented or long-stale crate
        # entry passed the audit. Compare it only here, where the enumeration
        # makes the crate set authoritative; with just the PR's own manifests
        # every untouched crate would look invented.
        manifest_crates = {
            match.group(1)
            for match in (CRATE_CARGO_TOML.match(path) for path in all_manifests)
            if match
        }
        # Sources only. A node that appears solely as an edge TARGET may
        # legitimately be a prose-documented crate or a non-crate box, so
        # judging those needs its own rule; a source line asserts "this crate
        # exists and depends on ...", which a missing manifest contradicts.
        for crate in sorted(set(diagram) - manifest_crates - DIAGRAM_PROSE_TARGETS):
            findings.append(
                Finding(
                    id="dependency-diagram-drift",
                    severity="warn",
                    rule_section="Documentation Policy",
                    file=DEPENDENCY_DIAGRAM_DOC,
                    line=None,
                    summary=f"Dependency diagram names {crate}, which has no manifest",
                    detail=(
                        f"The Dependency Direction diagram in "
                        f"{DEPENDENCY_DIAGRAM_DOC} references {crate}, but no "
                        f"crates/{crate}/Cargo.toml exists at this revision. "
                        "Remove the invented or stale entry, or add the crate, "
                        "in the same PR (Diagram Consistency)."
                    ),
                )
            )
    for path in sorted(manifests):
        crate_match = CRATE_CARGO_TOML.match(path)
        if not crate_match:
            continue
        crate = crate_match.group(1)
        cargo_text = changed_file_text(path, ref=ref, worktree=worktree)
        if cargo_text is None:
            if crate in diagram:
                findings.append(
                    Finding(
                        id="dependency-diagram-drift",
                        severity="warn",
                        rule_section="Documentation Policy",
                        file=DEPENDENCY_DIAGRAM_DOC,
                        line=None,
                        summary=(
                            f"Deleted crate {crate} still has a dependency "
                            "diagram entry"
                        ),
                        detail=(
                            f"{path} is removed by this change but "
                            f"{DEPENDENCY_DIAGRAM_DOC} still lists {crate} in "
                            "the Dependency Direction diagram. Remove the "
                            "stale entry in the same PR (Diagram "
                            "Consistency)."
                        ),
                    )
                )
            continue
        cargo_deps = {
            dep
            for dep in parse_cargo_tenferro_dependencies(cargo_text)
            if dep not in DIAGRAM_PROSE_TARGETS
        }
        diagram_deps = {
            dep
            for dep in diagram.get(crate, set())
            if dep not in DIAGRAM_PROSE_TARGETS
        }
        if crate not in diagram:
            # A crate absent from the diagram block is acceptable only when
            # the architecture doc covers it elsewhere (for example the
            # internal-crate prose and layer diagram); a genuinely new crate
            # must be added to the doc even when it has no tenferro
            # dependencies.
            if cargo_deps or crate not in doc_text:
                findings.append(
                    Finding(
                        id="dependency-diagram-drift",
                        severity="warn",
                        rule_section="Documentation Policy",
                        file=DEPENDENCY_DIAGRAM_DOC,
                        line=None,
                        summary=f"{crate} is missing from the dependency diagram",
                        detail=(
                            f"{path} exists but {DEPENDENCY_DIAGRAM_DOC} has "
                            f"no diagram entry for {crate}"
                            + (
                                " and does not mention the crate anywhere"
                                if crate not in doc_text
                                else ""
                            )
                            + ". Diagram Consistency requires the diagram to "
                            "match the implementation in the same PR."
                        ),
                    )
                )
            continue
        missing = sorted(cargo_deps - diagram_deps)
        stale = sorted(diagram_deps - cargo_deps)
        if missing or stale:
            parts = []
            if missing:
                parts.append(f"missing from the diagram: {', '.join(missing)}")
            if stale:
                parts.append(f"stale in the diagram: {', '.join(stale)}")
            findings.append(
                Finding(
                    id="dependency-diagram-drift",
                    severity="warn",
                    rule_section="Documentation Policy",
                    file=DEPENDENCY_DIAGRAM_DOC,
                    line=None,
                    summary=f"Dependency diagram is out of sync for {crate}",
                    detail=(
                        f"{crate} production dependencies disagree with the "
                        f"Dependency Direction diagram in "
                        f"{DEPENDENCY_DIAGRAM_DOC}: {'; '.join(parts)}. "
                        "Update the diagram in the same PR (Diagram "
                        "Consistency)."
                    ),
                )
            )
    return findings


def deterministic_checks(
    files: list[str],
    *,
    head: str | None = None,
    base: str | None = None,
    worktree: bool = False,
    added_lines: dict[str, set[int]] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(ai_report_file_findings(files, ref=head, worktree=worktree))
    if added_lines is not None:
        findings.extend(
            inline_test_module_findings(
                files,
                ref=head,
                base=base,
                worktree=worktree,
                added_lines=added_lines,
            )
        )
        findings.extend(
            missing_doc_example_findings(
                files, ref=head, worktree=worktree, added_lines=added_lines
            )
        )
        findings.extend(
            vacuous_doc_example_findings(
                files, ref=head, worktree=worktree, added_lines=added_lines
            )
        )
    findings.extend(
        dependency_diagram_findings(files, ref=head, worktree=worktree)
    )
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
    deleted_from = files_with_unanchorable_deletions(diff_text)
    section_names = select_rule_sections(files, added_text)
    rules_text = build_rules_payload(section_names)
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    findings = deterministic_checks(
        files,
        head=args.head,
        base=args.base,
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
                findings.append(
                    budget_exhausted_finding(
                        reviewed, len(chunks), args.budget_seconds
                    )
                )
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
                # A transport failure at the cumulative deadline is the budget
                # running out, not an unusable model. Reporting it as a block
                # would fail the gate for exactly the case the budget exists
                # to degrade gracefully.
                if isinstance(exc, TRANSPORT_ERRORS) and (
                    time.monotonic() >= llm_deadline
                ):
                    findings.append(
                        budget_exhausted_finding(
                            reviewed, len(chunks), args.budget_seconds
                        )
                    )
                else:
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
            files_with_deletions=deleted_from,
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
