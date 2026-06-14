#!/usr/bin/env python3
"""Review PR diffs against REPOSITORY_RULES.md using a delta-scoped LLM check.

Local API key loading uses the ``python-dotenv`` library. Install dev helpers with:

    python3 -m pip install -r scripts/requirements-dev.txt

When ``.env`` exists at the repository root it is loaded automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "REPOSITORY_RULES.md"
PROMPT_PATH = ROOT / "ai" / "prompts" / "repository-rules-review.md"
PROMPT_VERSION = "1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_DIFF_CHARS = 120_000
MAX_FILE_DIFF_CHARS = 40_000
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
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b("
    r"[\w.-]*(?:api[_-]?key|token|secret|password|passwd|pwd|client[_-]?secret|"
    r"private[_-]?key)[\w.-]*"
    r"\s*[:=]\s*)"
    r"([^\s#]+)"
)
STRICT_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b"
    r"[\w.-]*(?:api[_-]?key|token|secret|password|passwd|pwd|client[_-]?secret|"
    r"private[_-]?key)[\w.-]*"
    r"\s*[:=]\s*"
    r"(?!os\.environ|re\.compile|\[REDACTED_SECRET\]|tuple\[)"
    r"[A-Za-z0-9_./+=:@-]{12,}"
)

ALWAYS_SECTIONS = frozenset(
    {
        "Public Surface Discipline",
        "Work Logs And Design Records",
        "No Ad Hoc Fixes",
    }
)

SECTION_TRIGGERS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(r"(^|/)ad/|linearize|transpose_rule|autodiff"),
        frozenset(
            {
                "Rule Source Of Truth",
                "AD Rule Coverage",
                "Oracle Gate",
            }
        ),
    ),
    (
        re.compile(r"tenferro-cpu/|/kernel/|strided"),
        frozenset({"Performance And Layout Rules", "Unsafe Code Boundary"}),
    ),
    (
        re.compile(r"tenferro-gpu/|cubecl|cuda|cutensor|cublas"),
        frozenset({"Performance And Layout Rules", "Unsafe Code Boundary"}),
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


def run_git(args: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def changed_files(base: str, head: str, *, worktree: bool = False) -> list[str]:
    if worktree:
        output = run_git(["diff", "--name-only", base])
    else:
        output = run_git(["diff", "--name-only", f"{base}...{head}"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def unified_diff(base: str, head: str, *, worktree: bool = False) -> str:
    if worktree:
        return run_git(["diff", "--unified=3", base])
    return run_git(["diff", "--unified=3", f"{base}...{head}"])


def per_file_diffs(
    base: str,
    head: str,
    files: list[str],
    *,
    worktree: bool = False,
) -> dict[str, str]:
    diffs: dict[str, str] = {}
    for path in files:
        if worktree:
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


def select_rule_sections(files: list[str]) -> list[str]:
    selected = set(ALWAYS_SECTIONS)
    for path in files:
        for pattern, section_names in SECTION_TRIGGERS:
            if pattern.search(path):
                selected.update(section_names)
    return sorted(selected)


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
    current_len = len("\n".join(current_lines))

    for hunk in hunks:
        hunk_len = len("\n".join(hunk))
        separator_len = 1 if current_lines else 0
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
    return SECRET_ASSIGNMENT.sub(r"\1[REDACTED_SECRET]", redacted)


def redact_file_diffs(file_diffs: dict[str, str]) -> dict[str, str]:
    return {path: redact_sensitive_text(diff) for path, diff in file_diffs.items()}


def contains_sensitive_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_VALUE_PATTERNS) or bool(
        STRICT_SECRET_ASSIGNMENT.search(text)
    )


def sensitive_diff_finding(diff_text: str) -> Finding | None:
    if not contains_sensitive_text(diff_text):
        return None
    return Finding(
        id="sensitive-diff",
        severity="block",
        rule_section="External LLM Review",
        file="",
        line=None,
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
    for index, item in enumerate(findings_raw):
        if not isinstance(item, dict):
            raise ValueError(f"findings[{index}] must be an object")
        severity = item.get("severity")
        if severity not in {"block", "warn"}:
            raise ValueError(f"findings[{index}].severity must be block or warn")
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
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("parsed JSON must be an object")
    return parsed


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
        if finding.line is not None and finding.file in added_lines:
            if finding.line not in added_lines[finding.file]:
                continue
        kept.append(finding)
    return kept


def reconcile_verdict(findings: list[Finding]) -> str:
    return "fail" if any(item.severity == "block" for item in findings) else "pass"


def call_deepseek(
    *,
    api_key: str,
    model: str,
    api_url: str,
    system_prompt: str,
    user_content: str,
    timeout: float,
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
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
) -> tuple[str, list[Finding]]:
    user_content = "\n\n".join(
        [
            f"Prompt version: {PROMPT_VERSION}",
            "Changed files:",
            "\n".join(f"- {path}" for path in changed),
            "Applicable REPOSITORY_RULES sections:",
            rules_text,
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


def format_report(
    *,
    base: str,
    head: str,
    verdict: str,
    findings: list[Finding],
    waived: bool,
) -> str:
    lines = [
        f"Repository rules review ({base}...{head})",
        f"Verdict: {verdict}",
    ]
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


def scan_runtime_boundary_text(
    path: str,
    text: str,
    line_numbers: set[int] | None = None,
) -> list[str]:
    violations: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if line_numbers is not None and line_no not in line_numbers:
            continue
        if RUNTIME_AD_FORBIDDEN.search(line):
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
    ad_touched = any(
        "ad/" in path or "linearize" in path or "transpose_rule" in path
        for path in files
    )
    runtime_touched = any(path.startswith("crates/tenferro-runtime/") for path in files)
    if ad_touched and runtime_touched:
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
    parser.add_argument("--timeout", type=float, default=120.0)
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
    section_names = select_rule_sections(files)
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

    if not args.dry_run and not sensitive_finding:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("DEEPSEEK_API_KEY is not set", file=sys.stderr)
            return 1

        file_diffs = per_file_diffs(
            args.base,
            args.head,
            files,
            worktree=args.worktree,
        )
        chunks = split_diff_chunks(redact_file_diffs(file_diffs))
        llm_findings: list[Finding] = []
        for chunk in chunks:
            _, chunk_findings = review_chunk(
                api_key=api_key,
                model=args.model,
                api_url=args.api_url,
                system_prompt=system_prompt,
                rules_text=rules_text,
                changed=files,
                diff_chunk=chunk,
                timeout=args.timeout,
            )
            llm_findings.extend(chunk_findings)
        findings.extend(
            filter_findings(
                merge_findings(llm_findings),
                files,
                added_lines,
                allow_global=False,
            )
        )

    block_findings = [item for item in findings if item.severity == "block"]
    verdict = reconcile_verdict(block_findings)

    report_body = format_report(
        base=args.base,
        head=args.head,
        verdict=verdict,
        findings=findings,
        waived=False,
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
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 1 if block_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
