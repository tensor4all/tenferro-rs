#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "repository-rules-review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("repository_rules_review", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Secret-shaped fixtures are assembled at runtime so this file contains no
# contiguous secret-shaped literal of its own. Otherwise the guard under test
# blocks the LLM pass on every PR that touches its own tests, and the only way
# to review such a PR is a maintainer waiver. Short names keep the interpolated
# span below the 12-character threshold the quoted-credential pattern uses.
PAT = "ghp" + "_" + "abcdefghijklmnopqrstuvwxyz0123"
PW = "correct " + "horse " + "battery " + "staple"
# Spelled out, the opener plus the following value line would make this
# file trip the continuation detector it exercises.
KEYNAME = "API" + "_KEY"
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
SK = "sk-" + "0123456789abcdef0123456789abcdef"
VALUE = "abcdefghij" + "klmnopqrst"
BEARER = "Authorization: Bearer " + VALUE


def test_added_lines_by_file() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/foo.rs b/foo.rs",
            "index abc..def 100644",
            "--- a/foo.rs",
            "+++ b/foo.rs",
            "@@ -1,3 +1,4 @@",
            " unchanged",
            "+added",
            " context",
        ]
    )
    lines = mod.added_lines_by_file(diff)
    assert lines["foo.rs"] == {2}


def test_default_deepseek_model_uses_current_v4_name() -> None:
    mod = load_module()

    assert mod.DEFAULT_MODEL == "deepseek-v4-pro"
    assert mod.DEFAULT_API_URL == "https://api.deepseek.com/chat/completions"


def test_filter_findings_drops_unchanged_files() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="other.rs",
        line=1,
        summary="test",
        detail="detail",
    )
    kept = mod.filter_findings([finding], ["foo.rs"], {"foo.rs": {1}})
    assert kept == []


def test_filter_findings_keeps_added_line() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=2,
        summary="test",
        detail="detail",
    )
    kept = mod.filter_findings([finding], ["foo.rs"], {"foo.rs": {2}})
    assert len(kept) == 1


def test_filter_findings_drops_line_finding_without_added_lines() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="deleted.rs",
        line=4,
        summary="test",
        detail="detail",
    )
    kept = mod.filter_findings([finding], ["deleted.rs"], {})
    assert kept == []


def test_filter_findings_drops_file_level_block_finding() -> None:
    mod = load_module()
    block = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=None,
        summary="test",
        detail="detail",
    )
    warn = mod.Finding(
        id="w",
        severity="warn",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=None,
        summary="test",
        detail="detail",
    )
    kept = mod.filter_findings([block, warn], ["foo.rs"], {"foo.rs": {1}})
    assert kept == [warn]


def test_filter_findings_drops_global_llm_finding_when_disallowed() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="",
        line=None,
        summary="test",
        detail="detail",
    )
    kept = mod.filter_findings(
        [finding],
        ["foo.rs"],
        {"foo.rs": {1}},
        allow_global=False,
    )
    assert kept == []


def test_reconcile_verdict_only_blocks_fail() -> None:
    mod = load_module()
    warn = mod.Finding("w", "warn", "s", "f", 1, "s", "d")
    assert mod.reconcile_verdict([warn]) == "pass"
    block = mod.Finding("b", "block", "s", "f", 1, "s", "d")
    assert mod.reconcile_verdict([warn, block]) == "fail"


def test_select_rule_sections_includes_ad_for_ad_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tenferro-internal-ops/src/ad/rules/foo.rs"])
    assert "AD Rule Coverage" in sections
    assert "Public Surface Discipline" in sections


def test_select_rule_sections_includes_ad_for_tenferro_ad_crate() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tenferro-ad/src/lib.rs"])
    assert "AD Rule Coverage" in sections
    assert "Rule Source Of Truth" in sections
    assert "Oracle Gate" in sections


def test_select_rule_sections_includes_performance_for_tensor_crates() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(
        [
            "crates/tenferro-tensor-core/src/layout.rs",
            "crates/tenferro-tensor/src/view.rs",
        ]
    )
    assert "Performance-Sensitive Safety Contracts" in sections
    assert "Materialization And Copies" in sections
    assert "Range Checks And Slicing" in sections
    assert "Tensor Core Data Model" in sections
    assert "Performance And Layout Rules" not in sections
    assert "Unsafe Code Boundary" in sections


def test_select_rule_sections_includes_gpu_contract_for_gpu_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tenferro-gpu/src/cuda/runtime.rs"])
    assert "GPU Backend Contract" in sections
    assert "Device Transfer And Backend Buffer Errors" in sections
    assert "Cache Ownership" in sections


def test_select_rule_sections_includes_benchmark_rules_for_bench_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tenferro-cpu/benches/map.rs"])
    assert "Performance-Sensitive Tests And Benchmarks" in sections
    assert "Performance-Gated Experiment Protocol" not in sections


def test_select_rule_sections_includes_public_boundary_audits() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tenferro-runtime/src/traced.rs"])
    assert "Public Boundary Safety Audits" in sections


def test_final_cross_phase_multi_agent_audit_contract_is_present() -> None:
    mod = load_module()
    section_title = "Final Cross-Phase Multi-Agent Audit"
    text = mod.RULES_PATH.read_text(encoding="utf-8")
    section_marker = f"\n## {section_title}\n"
    expected_lanes = [
        ("1", "Specification and architecture"),
        ("2", "Rust safety and resource lifecycle"),
        ("3", "Performance and parallelism"),
        ("4", "Public API and documentation"),
        ("5", "CPU and NUMA"),
        ("6", "GPU, XLA, and multi-GPU"),
    ]
    expected_links = [
        ("Public Boundary Safety Audits", "#public-boundary-safety-audits"),
        ("Unsafe Code Boundary", "#unsafe-code-boundary"),
        (
            "Performance-Sensitive Safety Contracts",
            "#performance-sensitive-safety-contracts",
        ),
        ("Materialization And Copies", "#materialization-and-copies"),
        (
            "Performance-Gated Experiment Protocol",
            "#performance-gated-experiment-protocol",
        ),
        ("Cache Ownership", "#cache-ownership"),
        ("CPU Threading Contract", "#cpu-threading-contract"),
        ("GPU Backend Contract", "#gpu-backend-contract"),
        ("Documentation Policy", "#documentation-policy"),
        ("Work Logs And Design Records", "#work-logs-and-design-records"),
    ]
    required_clauses = [
        (
            "Repository-scale, multi-phase implementation programs require one final "
            "audit after every phase and its task-local reviews are complete, but "
            "before the umbrella issue or implementation branch is declared ready for "
            "integration."
        ),
        (
            "Audit one exact candidate commit. Every report must name that commit, "
            "and an auditor must not audit a lane whose implementation or task-local "
            "review it performed."
        ),
        "The lanes may run in batches when agent concurrency is limited.",
        "Assign a distinct independent auditor to each required lane:",
        (
            "**Specification and architecture:** accepted issues, phase acceptance "
            "criteria, eager/graph semantic parity, extension lowering, and migration "
            "compatibility."
        ),
        (
            "**Rust safety and resource lifecycle:** aliasing, unsafe boundaries, "
            "lifetimes, permits, locks, buffers, caches, identifiers, and cleanup on "
            "success, error, cancellation, and unwind."
        ),
        (
            "**Performance and parallelism:** current-main baseline, eager fast path, "
            "allocations and request/container overhead, nested fan-out, provider "
            "worker ownership, thread-count and placement control, and CPU/GPU "
            "synchronization."
        ),
        (
            "**Public API and documentation:** facade boundaries, operation-family "
            "traits, typed errors, feature combinations, runnable examples, online "
            "parallelism documentation, and source/checker consistency."
        ),
        (
            "**CPU and NUMA:** managed and external domains, strict versus advisory "
            "placement, resource arbitration, faer/BLAS/strided behavior, multiple "
            "sockets, re-entry, fairness, and failure recovery."
        ),
        (
            "**GPU, XLA, and multi-GPU:** context/stream/event ownership, "
            "backend-neutral artifacts, compiler and prepared-operation caches, device "
            "placement, independent devices, and cross-device failure handling."
        ),
        (
            "After all lane reports, a separate integration auditor must check "
            "cross-phase invariants, duplicated or contradictory findings, and the "
            "closure evidence."
        ),
        (
            "Each lane report must record the candidate commit; relevant feature, "
            "toolchain, and hardware configuration; inspected files, public contracts, "
            "and issue acceptance criteria; fresh commands and complete result "
            "classifications; findings classified as `Critical`, `Important`, or "
            "`Minor`; and explicit limitations or skipped hardware paths."
        ),
        (
            "Performance results must be classified as `PASS`, `FAIL`, or "
            "`INCONCLUSIVE`."
        ),
        "Do not infer a pass from an implementer's earlier run.",
        (
            "Source scanners and mutation tests support, but do not replace, call-path "
            "review and runtime tests."
        ),
        (
            "every `Critical` and `Important` finding is fixed and independently "
            "re-reviewed"
        ),
        (
            "every `Minor` finding is fixed or has a written rationale and accepted "
            "tracking issue"
        ),
        "every required performance gate is `PASS`",
        (
            "`INCONCLUSIVE` blocks promotion until a valid rerun or explicit accepted "
            "scope decision is recorded"
        ),
        "integration auditor reports no unresolved cross-phase contradiction",
        (
            "Environment-limited CPU, GPU, XLA, or multi-device paths must retain "
            "reproducible diagnostics and an identified verification owner."
        ),
        (
            "The final worklog must link every lane report, the integration report, the "
            "exact candidate commit, and the final verification commands."
        ),
        (
            "This gate supplements rather than replaces task-local TDD, specification "
            "review, code-quality review, CI, and required performance gates."
        ),
        (
            "Auditing is read-only: audit agents must not modify the candidate while "
            "reviewing it."
        ),
        "A finding fix creates a new exact candidate revision.",
        (
            "Before the audit can pass, every lane report must be refreshed to name and "
            "validate that final revision: each auditor reviews the intervening diff, "
            "every affected lane reruns its relevant evidence, and an unaffected lane "
            "may carry earlier runtime evidence forward only with a recorded diff-impact "
            "rationale."
        ),
        (
            "The separate integration auditor runs last against the same final "
            "revision."
        ),
    ]

    def assert_contract(rules_text: str) -> None:
        assert rules_text.count(section_marker) == 1
        section_start = rules_text.index(section_marker) + len(section_marker)
        next_section = re.search(r"(?m)^##[ \t]+", rules_text[section_start:])
        section_end = (
            section_start + next_section.start()
            if next_section is not None
            else len(rules_text)
        )
        section = rules_text[section_start:section_end]

        numbered_lanes = re.findall(
            r"(?m)^[ \t]+(\d+)\.[ \t]+\*\*(.+?):\*\*", section
        )
        assert numbered_lanes == expected_lanes

        heading_titles = re.findall(r"(?m)^#{2,6}[ \t]+(.+?)[ \t]*$", rules_text)
        for heading_title, anchor in expected_links:
            assert f"[{heading_title}]({anchor})" in section
            assert heading_titles.count(heading_title) == 1

        normalized = " ".join(section.split())
        for required_clause in required_clauses:
            assert required_clause in normalized, required_clause

    def without_bullet(rules_text: str, opening: str) -> str:
        pattern = rf"(?ms)^- {re.escape(opening)}.*?(?=^- |\n## |\Z)"
        mutated, count = re.subn(pattern, "", rules_text, count=1)
        assert count == 1
        return mutated

    def without_paragraph(rules_text: str, opening: str) -> str:
        pattern = rf"(?ms)^{re.escape(opening)}.*?\n\n"
        mutated, count = re.subn(pattern, "", rules_text, count=1)
        assert count == 1
        return mutated

    def without_sentence(rules_text: str, sentence: str) -> str:
        assert rules_text.count(sentence) == 1
        return rules_text.replace(sentence, "", 1)

    def without_first_lane_scope(rules_text: str) -> str:
        pattern = (
            r"(?ms)^(?P<title>[ \t]+1\.[ \t]+\*\*Specification and architecture:"
            r"\*\*).*?(?=^[ \t]+2\.[ \t]+\*\*)"
        )
        mutated, count = re.subn(pattern, r"\g<title>\n", rules_text, count=1)
        assert count == 1
        return mutated

    def assert_rejected(mutated_text: str, mutation: str) -> None:
        try:
            assert_contract(mutated_text)
        except AssertionError:
            return
        raise AssertionError(f"contract accepted mutation: {mutation}")

    assert_contract(text)
    assert section_title in mod.parse_repository_rules_sections()
    assert section_title not in mod.ALWAYS_SECTIONS
    assert section_title not in mod.select_rule_sections(["README.md"])

    assert_rejected(
        without_paragraph(
            text,
            "Repository-scale, multi-phase implementation programs require one final audit",
        ),
        "deleted final-audit trigger paragraph",
    )
    assert_rejected(
        without_sentence(
            text,
            "The lanes may run in batches when agent concurrency is limited.",
        ),
        "deleted batching sentence",
    )
    assert_rejected(
        without_first_lane_scope(text),
        "stripped specification lane scope but retained numbered bold title",
    )
    assert_rejected(
        without_bullet(text, "Audit one exact candidate commit."),
        "deleted exact-commit and independence bullet",
    )
    assert_rejected(
        without_bullet(
            text,
            "Environment-limited CPU, GPU, XLA, or multi-device paths must retain",
        ),
        "deleted reproducible-diagnostics and owner bullet",
    )
    integration_bullet = "- After all lane reports, a separate integration auditor"
    assert text.count(integration_bullet) == 1
    assert_rejected(
        text.replace(
            integration_bullet,
            "   7. **Unexpected lane:** mutation proof.\n" + integration_bullet,
            1,
        ),
        "inserted seventh lane with three-space indentation",
    )


def test_extract_json_payload_strips_fence() -> None:
    mod = load_module()
    parsed = mod.extract_json_payload(
        '```json\n{"verdict": "pass", "findings": []}\n```'
    )
    assert parsed["verdict"] == "pass"


def test_extract_json_payload_reports_malformed_embedded_object() -> None:
    mod = load_module()
    try:
        mod.extract_json_payload(
            'prefix {"verdict": "fail", "findings": [{"summary": "unterminated]} suffix'
        )
    except ValueError as err:
        assert "model response was not valid JSON" in str(err)
    else:
        raise AssertionError("malformed embedded JSON should raise ValueError")


def test_parse_findings_caps_model_output() -> None:
    mod = load_module()
    raw = {
        "verdict": "fail",
        "findings": [
            {
                "id": f"finding-{index}",
                "severity": "warn",
                "rule_section": "Public Surface Discipline",
                "file": "foo.rs",
                "line": 1,
                "summary": "test",
                "detail": "detail",
            }
            for index in range(mod.MAX_FINDINGS_PER_CHUNK + 3)
        ],
    }
    _, findings = mod.parse_findings(raw)
    assert len(findings) == mod.MAX_FINDINGS_PER_CHUNK


def test_parse_findings_normalizes_common_severity_aliases() -> None:
    mod = load_module()
    raw = {
        "verdict": "fail",
        "findings": [
            {
                "id": "a",
                "severity": "warning",
                "rule_section": "Public Surface Discipline",
                "file": "foo.rs",
                "line": 1,
                "summary": "test",
                "detail": "detail",
            },
            {
                "id": "b",
                "severity": "error",
                "rule_section": "Public Surface Discipline",
                "file": "foo.rs",
                "line": 2,
                "summary": "test",
                "detail": "detail",
            },
        ],
    }
    _, findings = mod.parse_findings(raw)
    assert [finding.severity for finding in findings] == ["warn", "block"]


def test_llm_response_error_finding_blocks_with_diagnostic() -> None:
    mod = load_module()
    finding = mod.llm_response_error_finding(ValueError("bad json"))

    assert finding.severity == "block"
    assert finding.id == "llm-review-unusable"
    assert "ValueError" in finding.detail


def test_split_diff_chunks_respects_limit() -> None:
    mod = load_module()
    big = "x" * (mod.MAX_DIFF_CHARS + 1)
    chunks = mod.split_diff_chunks({"a.rs": big, "b.rs": "small"})
    assert len(chunks) >= 2


def test_split_large_file_diff_preserves_file_header() -> None:
    mod = load_module()
    original_limit = mod.MAX_FILE_DIFF_CHARS
    try:
        mod.MAX_FILE_DIFF_CHARS = 170
        diff = "\n".join(
            [
                "diff --git a/foo.rs b/foo.rs",
                "index abc..def 100644",
                "--- a/foo.rs",
                "+++ b/foo.rs",
                "@@ -1,2 +1,3 @@",
                " context",
                "+added one",
                "+added two",
                "@@ -20,2 +21,3 @@",
                " context",
                "+added three",
                "+added four",
            ]
        )
        chunks = mod.split_diff_chunks({"foo.rs": diff})
    finally:
        mod.MAX_FILE_DIFF_CHARS = original_limit

    assert len(chunks) == 2
    assert all(chunk.startswith("diff --git a/foo.rs b/foo.rs") for chunk in chunks)
    assert all("--- a/foo.rs" in chunk and "+++ b/foo.rs" in chunk for chunk in chunks)
    assert "@@ -20,2 +21,3 @@" in chunks[1]


def test_split_large_file_diff_splits_oversized_single_hunk() -> None:
    mod = load_module()
    original_limit = mod.MAX_FILE_DIFF_CHARS
    try:
        mod.MAX_FILE_DIFF_CHARS = 115
        diff = "\n".join(
            [
                "diff --git a/foo.rs b/foo.rs",
                "index abc..def 100644",
                "--- a/foo.rs",
                "+++ b/foo.rs",
                "@@ -1,1 +1,8 @@",
                "+aaaaaaaaaa",
                "+bbbbbbbbbb",
                "+cccccccccc",
                "+dddddddddd",
                "+eeeeeeeeee",
                "+ffffffffff",
                "+gggggggggg",
                "+hhhhhhhhhh",
            ]
        )
        chunks = mod.split_diff_chunks({"foo.rs": diff})
    finally:
        mod.MAX_FILE_DIFF_CHARS = original_limit

    assert len(chunks) > 1
    assert all(chunk.startswith("diff --git a/foo.rs b/foo.rs") for chunk in chunks)
    # Each chunk is renumbered to its own offsets rather than repeating the
    # original header, so the model reports usable line numbers throughout.
    starts = []
    for chunk in chunks:
        hunk_line = [l for l in chunk.splitlines() if l.startswith("@@")][0]
        parsed = mod.HUNK_HEADER.match(hunk_line)
        assert parsed is not None, hunk_line
        starts.append((int(parsed.group(3)), int(parsed.group(4))))
    assert starts[0][0] == 1
    for (start, count), (next_start, _) in zip(starts, starts[1:]):
        assert start + count == next_start
    assert sum(count for _, count in starts) == 8
    assert all(len(chunk) <= 115 for chunk in chunks)


def test_split_large_file_diff_splits_single_overlong_diff_line() -> None:
    mod = load_module()
    original_limit = mod.MAX_FILE_DIFF_CHARS
    try:
        mod.MAX_FILE_DIFF_CHARS = 130
        diff = "\n".join(
            [
                "diff --git a/minified.js b/minified.js",
                "index abc..def 100644",
                "--- a/minified.js",
                "+++ b/minified.js",
                "@@ -0,0 +1 @@",
                "+" + ("x" * 220),
            ]
        )
        chunks = mod.split_diff_chunks({"minified.js": diff})
    finally:
        mod.MAX_FILE_DIFF_CHARS = original_limit

    assert len(chunks) > 1
    assert all(chunk.startswith("diff --git a/minified.js b/minified.js") for chunk in chunks)
    # One source line split across chunks: every chunk describes that same
    # single added line, so the header names a one-line span.
    assert all("@@ -0,0 +1,1 @@" in chunk for chunk in chunks)
    assert all(len(chunk) <= 130 for chunk in chunks)


def test_scan_runtime_boundary_text_reports_forbidden_symbol() -> None:
    mod = load_module()
    violations = mod.scan_runtime_boundary_text(
        "crates/tenferro-runtime/src/lib.rs",
        "pub struct Safe;\npub struct EagerTensor;\n",
    )
    assert violations == [
        "crates/tenferro-runtime/src/lib.rs:2: pub struct EagerTensor;"
    ]


def test_scan_runtime_boundary_text_can_limit_to_changed_lines() -> None:
    mod = load_module()
    violations = mod.scan_runtime_boundary_text(
        "crates/tenferro-runtime/src/lib.rs",
        "pub struct Safe;\n//! autodiff docs\npub struct EagerTensor;\n",
        {3},
    )
    assert violations == [
        "crates/tenferro-runtime/src/lib.rs:3: pub struct EagerTensor;"
    ]


def test_scan_runtime_boundary_text_ignores_comments() -> None:
    mod = load_module()
    violations = mod.scan_runtime_boundary_text(
        "crates/tenferro-runtime/src/lib.rs",
        "\n".join(
            [
                "//! use tenferro-ad for autodiff",
                "// must not depend on tidu",
                "/* EagerTensor */",
                "pub struct Safe;",
                "pub struct EagerTensor;",
            ]
        ),
    )
    assert violations == [
        "crates/tenferro-runtime/src/lib.rs:5: pub struct EagerTensor;"
    ]


def test_scan_runtime_boundary_text_tracks_block_comments_across_context() -> None:
    mod = load_module()
    violations = mod.scan_runtime_boundary_text(
        "crates/tenferro-runtime/src/lib.rs",
        "/*\nEagerTensor\n*/\npub struct Safe;\n",
        {2},
    )
    assert violations == []


def test_deterministic_checks_passes_added_lines_to_runtime_scan() -> None:
    mod = load_module()
    captured: dict[str, object] = {}
    original = mod.runtime_ad_boundary_violations

    def fake_runtime_ad_boundary_violations(
        *,
        ref: str | None,
        worktree: bool,
        changed_lines: dict[str, set[int]] | None = None,
    ) -> list[str]:
        captured["ref"] = ref
        captured["worktree"] = worktree
        captured["changed_lines"] = changed_lines
        return []

    try:
        mod.runtime_ad_boundary_violations = fake_runtime_ad_boundary_violations
        mod.deterministic_checks(
            [
                "crates/tenferro-runtime/src/lib.rs",
            ],
            head="HEAD",
            worktree=False,
            added_lines={"crates/tenferro-runtime/src/lib.rs": {9}},
        )
    finally:
        mod.runtime_ad_boundary_violations = original

    assert captured == {
        "ref": "HEAD",
        "worktree": False,
        "changed_lines": {"crates/tenferro-runtime/src/lib.rs": {9}},
    }


def test_redact_sensitive_text_masks_common_secret_forms() -> None:
    mod = load_module()
    api_value = "sk-" + "live-secret-abcdefghijklmnopqrstuvwxyz"
    github_value = "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"
    aws_value = "AKIA" + "1234567890ABCDEF"
    text = "\n".join(
        [
            "+DEEPSEEK_" + "API_" + "KEY=" + api_value,
            "+to" + "ken: " + github_value,
            "+aws_access_" + "key_id = " + aws_value,
        ]
    )
    redacted = mod.redact_sensitive_text(text)
    assert api_value not in redacted
    assert github_value not in redacted
    assert aws_value not in redacted
    assert redacted.count("[REDACTED_SECRET]") == 3


def test_sensitive_diff_finding_checks_added_lines_only() -> None:
    mod = load_module()
    old_secret = "sk-" + "old-secret-abcdefghijklmnopqrstuvwxyz"
    added_secret = "sk-" + "new-secret-abcdefghijklmnopqrstuvwxyz"
    unchanged_secret = "sk-" + "dummy-secret-abcdefghijklmnopqrstuvwxyz"

    removed_or_context_only = "\n".join(
        [
            "diff --git a/secrets.txt b/secrets.txt",
            "--- a/secrets.txt",
            "+++ b/secrets.txt",
            "@@ -1,3 +1,3 @@",
            f" unchanged = {unchanged_secret}",
            f"-removed = {old_secret}",
            "+replacement = safe-placeholder",
        ]
    )
    assert mod.sensitive_diff_finding(removed_or_context_only) is None

    added = removed_or_context_only + f"\n+added = {added_secret}"
    assert mod.sensitive_diff_finding(added) is not None


def test_contains_sensitive_text_ignores_env_lookup_code() -> None:
    mod = load_module()
    text = "\n".join(
        [
            '+        api_key = os.environ.get("DEEPSEEK_API_KEY")',
            '+            print("DEEPSEEK_API_KEY is not set", file=sys.stderr)',
        ]
    )
    assert not mod.contains_sensitive_text(text)


def test_contains_sensitive_text_distinguishes_identifiers_from_quoted_credentials() -> None:
    mod = load_module()
    quoted_credential = '"' + "live-token-value-1234567890" + '"'

    assert not mod.contains_sensitive_text("+        self.candidate_token = candidate_token")
    assert not mod.contains_sensitive_text(
        '+        token_type: "WebGPU event token from another queue"'
    )
    assert mod.contains_sensitive_text(f"+        api_token = {quoted_credential}")


def test_sensitive_diff_finding_reports_added_match_location() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/crates/tenferro-runtime/src/runtime/engine_registration.rs "
            "b/crates/tenferro-runtime/src/runtime/engine_registration.rs",
            "--- a/crates/tenferro-runtime/src/runtime/engine_registration.rs",
            "+++ b/crates/tenferro-runtime/src/runtime/engine_registration.rs",
            "@@ -658,0 +659,1 @@",
            "+        api_" + 'token = "' + "fixture-credential-value-1234567890" + '"',
        ]
    )

    finding = mod.sensitive_diff_finding(diff)

    assert finding is not None
    assert finding.file == "crates/tenferro-runtime/src/runtime/engine_registration.rs"
    assert finding.line == 659


def test_summarize_llm_review_computes_dropped_count() -> None:
    mod = load_module()
    summary = mod.summarize_llm_review(
        chunk_sizes=[118923, 46296],
        elapsed_seconds=4.56,
        returned_count=3,
        kept_count=1,
    )
    assert "2 dropped" in summary


def test_format_report_includes_llm_summary_line() -> None:
    mod = load_module()
    report = mod.format_report(
        base="base",
        head="head",
        verdict="pass",
        findings=[],
        waived=False,
        llm_summary="LLM review: 1 chunk(s) (10 chars) in 0.5s; "
        "0 finding(s) returned, 0 kept, 0 dropped by diff-anchor filtering.",
    )
    lines = report.splitlines()
    assert lines[1] == "Verdict: pass"
    assert lines[2].startswith("LLM review: 1 chunk(s)")
    assert lines[3] == "No findings."


def test_format_report_omits_llm_summary_when_absent() -> None:
    mod = load_module()
    report = mod.format_report(
        base="base",
        head="head",
        verdict="pass",
        findings=[],
        waived=False,
    )
    assert "LLM review:" not in report


INLINE_TEST_SOURCE = "\n".join(
    ["fn production() {}"]
    + [f"fn helper_{i}() {{}}" for i in range(160)]
    + [
        "#[cfg(test)]",
        "mod tests {",
        "    #[test]",
        "    fn works() {",
        "        assert!(true);",
        "    }",
        "}",
    ]
)


def test_inline_test_module_findings_ignores_a_shrinking_block() -> None:
    """Extraction work that shrinks an oversized block must not warn.

    The check read only the head revision, so any edit inside what remained of
    a block classified the change as "added or grown" — penalizing exactly the
    cleanup the rule asks for.
    """
    mod = load_module()
    bigger = "\n".join(
        ["fn production() {}"]
        + [f"fn helper_{i}() {{}}" for i in range(160)]
        + ["#[cfg(test)]", "mod tests {"]
        + [f"    // case {i}" for i in range(40)]
        + ["}"]
    )
    _with_fake_text(
        mod,
        {"crates/x/src/error.rs": INLINE_TEST_SOURCE},
        base_mapping={"crates/x/src/error.rs": bigger},
    )
    findings = mod.inline_test_module_findings(
        ["crates/x/src/error.rs"],
        ref="HEAD",
        base="origin/main",
        worktree=False,
        added_lines={"crates/x/src/error.rs": {165}},
    )
    assert findings == []


def test_inline_test_module_findings_still_flags_real_growth() -> None:
    """A block that is larger than at base still warns."""
    mod = load_module()
    smaller = "\n".join(
        ["fn production() {}"] + [f"fn helper_{i}() {{}}" for i in range(160)]
    )
    _with_fake_text(
        mod,
        {"crates/x/src/error.rs": INLINE_TEST_SOURCE},
        base_mapping={"crates/x/src/error.rs": smaller},
    )
    findings = mod.inline_test_module_findings(
        ["crates/x/src/error.rs"],
        ref="HEAD",
        base="origin/main",
        worktree=False,
        added_lines={"crates/x/src/error.rs": {165}},
    )
    assert [item.id for item in findings] == ["inline-test-module"]


def test_dependency_diagram_findings_rejects_a_crate_without_a_manifest() -> None:
    """A diagram node with no manifest never entered the manifest loop.

    Enumerating the manifests only covers `manifest_crates - diagram`; the
    opposite direction let an invented or long-stale crate entry pass.
    """
    mod = load_module()
    doc = "\n".join(
        [
            "## IV. Dependency Direction",
            "",
            "```text",
            "tenferro-fft              -> tenferro-runtime",
            "tenferro-phantom          -> tenferro-runtime",
            "```",
        ]
    )
    cargo = "\n".join(["[dependencies]", "tenferro-runtime.workspace = true"])
    _with_fake_text(
        mod,
        {
            mod.DEPENDENCY_DIAGRAM_DOC: doc,
            "crates/tenferro-fft/Cargo.toml": cargo,
        },
    )
    mod.list_crate_manifests = lambda *, ref, worktree: [
        "crates/tenferro-fft/Cargo.toml"
    ]
    findings = mod.dependency_diagram_findings(
        [mod.DEPENDENCY_DIAGRAM_DOC],
        ref="HEAD",
        worktree=False,
    )
    assert [item.id for item in findings] == ["dependency-diagram-drift"]
    assert "tenferro-phantom" in findings[0].summary


def test_prompt_does_not_promise_pr_text_disclosure() -> None:
    """The payload carries paths, routed rules and the diff — never the PR body.

    Listing `PR text` as a disclosure source asked the model for a
    classification it has no way to make.
    """
    mod = load_module()
    prompt = mod.PROMPT_PATH.read_text(encoding="utf-8")
    assert "disclosed-in-worklog:" in prompt
    assert "PR text" not in prompt


def test_rust_inline_test_blocks_reports_span() -> None:
    mod = load_module()
    blocks = mod.rust_inline_test_blocks(INLINE_TEST_SOURCE)
    assert len(blocks) == 1
    start, end = blocks[0]
    assert start == 162
    assert end == 168


def test_rust_inline_test_blocks_ignores_mod_declaration() -> None:
    mod = load_module()
    text = "fn production() {}\n#[cfg(test)]\nmod tests;\n"
    assert mod.rust_inline_test_blocks(text) == []


def _with_fake_text(mod, mapping, base_mapping=None):
    """Stub `changed_file_text`; `base_mapping` serves any ref other than HEAD."""

    def fake(path, *, ref, worktree):
        if base_mapping is not None and ref != "HEAD":
            return base_mapping.get(path)
        return mapping.get(path)

    mod.changed_file_text = fake


def test_inline_test_module_findings_flags_grown_block() -> None:
    mod = load_module()
    _with_fake_text(mod, {"crates/x/src/error.rs": INLINE_TEST_SOURCE})
    findings = mod.inline_test_module_findings(
        ["crates/x/src/error.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/error.rs": {165}},
    )
    assert [item.id for item in findings] == ["inline-test-module"]
    assert findings[0].severity == "warn"
    assert findings[0].rule_section == "Unit Test Organization"
    assert findings[0].line == 162


def test_inline_test_module_findings_exempts_tiny_leaf_module() -> None:
    mod = load_module()
    tiny = "\n".join(
        [
            "fn leaf() {}",
            "#[cfg(test)]",
            "mod tests {",
            "    #[test]",
            "    fn works() {}",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/leaf.rs": tiny})
    findings = mod.inline_test_module_findings(
        ["crates/x/src/leaf.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/leaf.rs": {4}},
    )
    assert findings == []


def test_inline_test_module_findings_skips_untouched_block() -> None:
    mod = load_module()
    _with_fake_text(mod, {"crates/x/src/error.rs": INLINE_TEST_SOURCE})
    findings = mod.inline_test_module_findings(
        ["crates/x/src/error.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/error.rs": {1}},
    )
    assert findings == []


DOC_EXAMPLE_SOURCE = "\n".join(
    [
        "/// Summary.",
        "///",
        "/// # Examples",
        "///",
        "/// ```",
        "/// documented::call();",
        "/// ```",
        "pub fn documented() {}",
        "",
        "/// Only errors.",
        "///",
        "/// # Errors",
        "///",
        "/// Never.",
        "pub fn undocumented() {}",
        "",
        "#[doc(hidden)]",
        "pub fn hidden_hook() {}",
        "",
        "mod private {",
        "    pub trait Sealed {}",
        "}",
    ]
)


def test_missing_doc_example_findings_flags_only_real_gaps() -> None:
    mod = load_module()
    _with_fake_text(mod, {"crates/x/src/lib.rs": DOC_EXAMPLE_SOURCE})
    total = len(DOC_EXAMPLE_SOURCE.splitlines())
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": set(range(1, total + 1))},
    )
    assert len(findings) == 1
    assert findings[0].id == "missing-doc-examples"
    assert "fn undocumented" in findings[0].detail
    assert "hidden_hook" not in findings[0].detail
    assert "Sealed" not in findings[0].detail
    assert "fn documented" not in findings[0].detail


def test_missing_doc_example_findings_ignores_unchanged_items() -> None:
    mod = load_module()
    _with_fake_text(mod, {"crates/x/src/lib.rs": DOC_EXAMPLE_SOURCE})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {1}},
    )
    assert findings == []


def test_vacuous_doc_example_findings_flags_path_only_example() -> None:
    mod = load_module()
    vacuous = "\n".join(
        [
            "/// # Examples",
            "///",
            "/// ```rust",
            "/// use crate::Widget;",
            "/// let _method = Widget::spin;",
            "/// ```",
            "pub fn spin() {}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": vacuous})
    findings = mod.vacuous_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {5}},
    )
    assert [item.id for item in findings] == ["vacuous-doc-example"]


def test_vacuous_doc_example_findings_ignores_comment_lines() -> None:
    """Prose in a comment does not turn an assignment into real API usage.

    A comment line stayed in the classified set, so the `all(...)` condition
    failed and an assignment-only example escaped the audit entirely.
    """
    mod = load_module()
    vacuous = "\n".join(
        [
            "/// # Examples",
            "///",
            "/// ```rust",
            "/// use crate::Widget;",
            "/// // Obtain the method",
            "/// let _method = Widget::spin;",
            "/// ```",
            "pub fn spin() {}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": vacuous})
    findings = mod.vacuous_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {6}},
    )
    assert [item.id for item in findings] == ["vacuous-doc-example"]


def test_vacuous_doc_example_findings_accepts_comment_only_example() -> None:
    """An example that is nothing but comments has no classified code left."""
    mod = load_module()
    comments_only = "\n".join(
        [
            "/// # Examples",
            "///",
            "/// ```rust",
            "/// // See the integration tests for a runnable example.",
            "/// ```",
            "pub fn spin() {}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": comments_only})
    findings = mod.vacuous_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {4}},
    )
    assert findings == []


def test_vacuous_doc_example_findings_accepts_real_usage() -> None:
    mod = load_module()
    real = "\n".join(
        [
            "/// # Examples",
            "///",
            "/// ```rust",
            "/// use crate::Widget;",
            "/// let widget = Widget::new();",
            "/// assert!(widget.spin().is_ok());",
            "/// ```",
            "pub fn spin() {}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": real})
    findings = mod.vacuous_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {5}},
    )
    assert findings == []


def test_ai_report_file_findings_flags_reports_outside_worklogs() -> None:
    mod = load_module()
    _with_fake_text(
        mod,
        {
            ".superpowers/sdd/task-1-report.md": "report",
            "notes/session-report.md": "report",
            "docs/worklogs/2026-08-01-task-report.md": "worklog",
            "crates/x/src/lib.rs": "code",
        },
    )
    findings = mod.ai_report_file_findings(
        [
            ".superpowers/sdd/task-1-report.md",
            "notes/session-report.md",
            "docs/worklogs/2026-08-01-task-report.md",
            "crates/x/src/lib.rs",
        ],
        ref="HEAD",
        worktree=False,
    )
    flagged = sorted(item.file for item in findings)
    assert flagged == [
        ".superpowers/sdd/task-1-report.md",
        "notes/session-report.md",
    ]
    assert all(item.rule_section == "PR Content Hygiene" for item in findings)


def test_parse_cargo_tenferro_dependencies_skips_optional_and_dev() -> None:
    mod = load_module()
    cargo = "\n".join(
        [
            "[package]",
            'name = "tenferro-demo"',
            "",
            "[dependencies]",
            "tenferro-tensor.workspace = true",
            'tenferro-runtime = { workspace = true }',
            'tenferro-ad = { workspace = true, optional = true }',
            "serde.workspace = true",
            "",
            "[dependencies.tenferro-cpu]",
            "workspace = true",
            "",
            "[dependencies.tenferro-gpu]",
            "workspace = true",
            "optional = true",
            "",
            "[dev-dependencies]",
            "tenferro-fft.workspace = true",
        ]
    )
    deps = mod.parse_cargo_tenferro_dependencies(cargo)
    assert deps == {"tenferro-tensor", "tenferro-runtime", "tenferro-cpu"}


def test_parse_cargo_tenferro_dependencies_accepts_compact_optional_syntax() -> None:
    """TOML allows any spacing around `=`; compact optional entries are optional.

    Matching the literal `optional = true` string recorded
    `{workspace=true,optional=true}` as a production edge, which then produced
    a false `dependency-diagram-drift` warning.
    """
    mod = load_module()
    cargo = "\n".join(
        [
            "[dependencies]",
            "tenferro-ad={workspace=true,optional=true}",
            "tenferro-gpu.optional=true",
            "tenferro-runtime = { workspace = true }",
            "",
            "[dependencies.tenferro-cpu]",
            "workspace = true",
            "optional=true",
        ]
    )
    assert mod.parse_cargo_tenferro_dependencies(cargo) == {"tenferro-runtime"}


def test_parse_dependency_diagram_handles_continuation_lines() -> None:
    mod = load_module()
    doc = "\n".join(
        [
            "## IV. Dependency Direction",
            "",
            "```text",
            "tenferro-tensor           -> tenferro-tensor-core, tenferro-core-ops",
            "tenferro-runtime          -> tenferro-tensor,",
            "                              tenferro-internal-ops",
            "tenferro-internal-cpu-kernels",
            "                           -> tenferro-tensor",
            "```",
        ]
    )
    edges = mod.parse_dependency_diagram(doc)
    assert edges is not None
    assert edges["tenferro-tensor"] == {"tenferro-tensor-core", "tenferro-core-ops"}
    assert edges["tenferro-runtime"] == {"tenferro-tensor", "tenferro-internal-ops"}
    assert edges["tenferro-internal-cpu-kernels"] == {"tenferro-tensor"}


def test_dependency_diagram_findings_reports_missing_edge() -> None:
    mod = load_module()
    doc = "\n".join(
        [
            "## IV. Dependency Direction",
            "",
            "```text",
            "tenferro-fft              -> tenferro-runtime",
            "```",
        ]
    )
    cargo = "\n".join(
        [
            "[dependencies]",
            "tenferro-runtime.workspace = true",
            "tenferro-cpu.workspace = true",
        ]
    )
    _with_fake_text(
        mod,
        {
            mod.DEPENDENCY_DIAGRAM_DOC: doc,
            "crates/tenferro-fft/Cargo.toml": cargo,
        },
    )
    findings = mod.dependency_diagram_findings(
        ["crates/tenferro-fft/Cargo.toml"],
        ref="HEAD",
        worktree=False,
    )
    assert [item.id for item in findings] == ["dependency-diagram-drift"]
    assert "tenferro-cpu" in findings[0].detail


def test_dependency_diagram_findings_accepts_matching_edges() -> None:
    mod = load_module()
    doc = "\n".join(
        [
            "## IV. Dependency Direction",
            "",
            "```text",
            "tenferro-fft              -> tenferro-runtime, tenferro-cpu",
            "```",
        ]
    )
    cargo = "\n".join(
        [
            "[dependencies]",
            "tenferro-runtime.workspace = true",
            "tenferro-cpu.workspace = true",
            "tenferro-core-ops.workspace = true",
        ]
    )
    _with_fake_text(
        mod,
        {
            mod.DEPENDENCY_DIAGRAM_DOC: doc,
            "crates/tenferro-fft/Cargo.toml": cargo,
        },
    )
    findings = mod.dependency_diagram_findings(
        ["crates/tenferro-fft/Cargo.toml"],
        ref="HEAD",
        worktree=False,
    )
    assert findings == []


def test_select_rule_sections_routes_unit_test_rules_for_src_rust() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tenferro-cpu/src/lib.rs"])
    assert "Unit Test Organization" in sections
    assert "Documentation Policy" in sections


def test_select_rule_sections_falls_back_for_unrouted_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections([".superpowers/sdd/plan.md"])
    assert mod.FALLBACK_SECTION in sections


def test_pr_content_hygiene_section_is_parseable() -> None:
    mod = load_module()
    sections = mod.parse_repository_rules_sections()
    assert "PR Content Hygiene" in sections
    assert "AI-generated" in sections["PR Content Hygiene"]


def test_ai_report_file_findings_skips_deleted_reports() -> None:
    mod = load_module()
    _with_fake_text(mod, {})
    findings = mod.ai_report_file_findings(
        [".superpowers/sdd/task-1-report.md"],
        ref="HEAD",
        worktree=False,
    )
    assert findings == []


def test_rust_inline_test_blocks_recognizes_cfg_all_test() -> None:
    mod = load_module()
    text = "\n".join(
        [
            "fn production() {}",
            '#[cfg(all(test, feature = "cuda"))]',
            "mod tests {",
            "    #[test]",
            "    fn works() {}",
            "}",
        ]
    )
    assert mod.rust_inline_test_blocks(text) == [(2, 6)]


def test_rust_inline_test_blocks_ignores_cfg_not_test() -> None:
    mod = load_module()
    text = "\n".join(
        [
            "#[cfg(not(test))]",
            "mod production {",
            "    fn run() {}",
            "}",
        ]
    )
    assert mod.rust_inline_test_blocks(text) == []


def test_dependency_diagram_findings_checks_all_crates_on_doc_only_change() -> None:
    mod = load_module()
    doc = "\n".join(
        [
            "## IV. Dependency Direction",
            "",
            "```text",
            "tenferro-fft              -> tenferro-runtime",
            "```",
        ]
    )
    cargo = "\n".join(
        [
            "[dependencies]",
            "tenferro-runtime.workspace = true",
            "tenferro-cpu.workspace = true",
        ]
    )
    _with_fake_text(
        mod,
        {
            mod.DEPENDENCY_DIAGRAM_DOC: doc,
            "crates/tenferro-fft/Cargo.toml": cargo,
        },
    )
    mod.list_crate_manifests = lambda *, ref, worktree: [
        "crates/tenferro-fft/Cargo.toml"
    ]
    findings = mod.dependency_diagram_findings(
        [mod.DEPENDENCY_DIAGRAM_DOC],
        ref="HEAD",
        worktree=False,
    )
    assert [item.id for item in findings] == ["dependency-diagram-drift"]
    assert "tenferro-cpu" in findings[0].detail


def test_is_cfg_test_attr_matches_any_operand_order() -> None:
    mod = load_module()
    assert mod.is_cfg_test_attr("#[cfg(test)]")
    assert mod.is_cfg_test_attr('#[cfg(all(test, feature = "cuda"))]')
    assert mod.is_cfg_test_attr('#[cfg(all(feature = "cuda", test))]')
    assert not mod.is_cfg_test_attr("#[cfg(not(test))]")
    assert not mod.is_cfg_test_attr('#[cfg(all(feature = "cuda", not(test)))]')
    assert not mod.is_cfg_test_attr('#[cfg(feature = "test-utils")]')


def test_is_cfg_test_attr_tracks_nested_not_polarity() -> None:
    """A `test` operand nested under `not` gates the item OFF during tests.

    Deleting a literal `not(test)` substring left the inner token behind, so a
    production-only module read as an inline test module and the audit fired a
    false positive whenever that module changed.
    """
    mod = load_module()
    assert not mod.is_cfg_test_attr('#[cfg(not(any(test, feature = "cuda")))]')
    assert not mod.is_cfg_test_attr('#[cfg(not(all(test, feature = "cuda")))]')
    assert not mod.is_cfg_test_attr('#[cfg(all(unix, not(any(test, feature = "x"))))]')
    # Even nesting restores positive polarity.
    assert mod.is_cfg_test_attr("#[cfg(not(not(test)))]")
    # A positive operand elsewhere still gates on tests.
    assert mod.is_cfg_test_attr('#[cfg(any(test, feature = "cuda"))]')


def test_rust_inline_test_blocks_recognizes_trailing_test_operand() -> None:
    mod = load_module()
    text = "\n".join(
        [
            "fn production() {}",
            '#[cfg(all(feature = "cuda", test))]',
            "mod tests {",
            "    #[test]",
            "    fn works() {}",
            "}",
        ]
    )
    assert mod.rust_inline_test_blocks(text) == [(2, 6)]


def test_module_publicly_reachable_respects_private_parent_declaration() -> None:
    mod = load_module()
    _with_fake_text(
        mod,
        {
            "crates/x/src/cubecl/memory.rs": "pub fn helper() {}",
            "crates/x/src/cubecl/mod.rs": "mod memory;\npub fn api() {}",
            "crates/x/src/lib.rs": "pub mod cubecl;",
        },
    )
    assert not mod.module_publicly_reachable(
        "crates/x/src/cubecl/memory.rs", ref="HEAD", worktree=False
    )
    assert mod.module_publicly_reachable(
        "crates/x/src/cubecl/mod.rs", ref="HEAD", worktree=False
    )


def test_missing_doc_example_findings_skips_privately_declared_module() -> None:
    mod = load_module()
    _with_fake_text(
        mod,
        {
            "crates/x/src/cubecl/memory.rs": "pub fn helper() {}",
            "crates/x/src/cubecl/mod.rs": "mod memory;",
            "crates/x/src/lib.rs": "pub mod cubecl;",
        },
    )
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/cubecl/memory.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/cubecl/memory.rs": {1}},
    )
    assert findings == []


def test_dependency_diagram_findings_reports_deleted_crate_entry() -> None:
    mod = load_module()
    doc = "\n".join(
        [
            "## IV. Dependency Direction",
            "",
            "```text",
            "tenferro-foo              -> tenferro-runtime",
            "```",
        ]
    )
    _with_fake_text(mod, {mod.DEPENDENCY_DIAGRAM_DOC: doc})
    findings = mod.dependency_diagram_findings(
        ["crates/tenferro-foo/Cargo.toml"],
        ref="HEAD",
        worktree=False,
    )
    assert [item.id for item in findings] == ["dependency-diagram-drift"]
    assert "Deleted crate" in findings[0].summary


def test_rust_inline_test_blocks_handles_multiline_cfg_attribute() -> None:
    mod = load_module()
    text = "\n".join(
        [
            "fn production() {}",
            "#[cfg(all(",
            '    feature = "cuda",',
            "    test",
            "))]",
            "mod tests {",
            "    #[test]",
            "    fn works() {}",
            "}",
        ]
    )
    assert mod.rust_inline_test_blocks(text) == [(2, 9)]


def test_rust_inline_test_blocks_skips_multiline_non_cfg_attribute() -> None:
    mod = load_module()
    text = "\n".join(
        [
            "#[cfg(test)]",
            "#[allow(",
            "    dead_code",
            ")]",
            "mod tests {",
            "    #[test]",
            "    fn works() {}",
            "}",
        ]
    )
    assert mod.rust_inline_test_blocks(text) == [(1, 8)]


def test_module_publicly_reachable_follows_pub_use_reexport() -> None:
    mod = load_module()
    _with_fake_text(
        mod,
        {
            "crates/x/src/concrete.rs": "pub fn helper() {}",
            "crates/x/src/lib.rs": "mod concrete;\npub use concrete::helper;",
        },
    )
    assert mod.module_publicly_reachable(
        "crates/x/src/concrete.rs", ref="HEAD", worktree=False
    )
    _with_fake_text(
        mod,
        {
            "crates/x/src/concrete.rs": "pub fn helper() {}",
            "crates/x/src/lib.rs": "mod concrete;",
        },
    )
    assert not mod.module_publicly_reachable(
        "crates/x/src/concrete.rs", ref="HEAD", worktree=False
    )


def test_select_rule_sections_routes_hygiene_for_unknown_top_level_rust() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["new-crate/src/lib.rs"])
    assert mod.FALLBACK_SECTION in sections
    sections = mod.select_rule_sections([".audit/check.rs"])
    assert mod.FALLBACK_SECTION in sections
    sections = mod.select_rule_sections(["crates/tenferro-cpu/src/lib.rs"])
    assert mod.FALLBACK_SECTION not in sections


def test_dependency_diagram_findings_reports_new_leaf_crate() -> None:
    mod = load_module()
    doc = "\n".join(
        [
            "## IV. Dependency Direction",
            "",
            "```text",
            "tenferro-fft              -> tenferro-runtime",
            "```",
            "",
            "Additional internal dependencies: `tenferro-core-ops` prose.",
        ]
    )
    leaf_cargo = "[dependencies]\nserde.workspace = true\n"
    _with_fake_text(
        mod,
        {
            mod.DEPENDENCY_DIAGRAM_DOC: doc,
            "crates/tenferro-newleaf/Cargo.toml": leaf_cargo,
            "crates/tenferro-core-ops/Cargo.toml": leaf_cargo,
        },
    )
    findings = mod.dependency_diagram_findings(
        ["crates/tenferro-newleaf/Cargo.toml"],
        ref="HEAD",
        worktree=False,
    )
    assert [item.id for item in findings] == ["dependency-diagram-drift"]
    assert "tenferro-newleaf" in findings[0].summary
    findings = mod.dependency_diagram_findings(
        ["crates/tenferro-core-ops/Cargo.toml"],
        ref="HEAD",
        worktree=False,
    )
    assert findings == []


def test_pub_use_exports_parses_selective_and_glob_forms() -> None:
    mod = load_module()
    assert mod.pub_use_exports("pub use concrete::{einsum, plan as p};", "concrete") == {
        "einsum",
        "plan",
    }
    assert mod.pub_use_exports("pub use concrete::*;", "concrete") == "all"
    assert mod.pub_use_exports("pub use concrete::single;", "concrete") == {"single"}
    assert mod.pub_use_exports("pub use other::thing;", "concrete") is None
    assert (
        mod.pub_use_exports("pub use concrete::{\n    a,\n    b,\n};", "concrete")
        == {"a", "b"}
    )


def test_missing_doc_example_findings_respects_selective_reexport() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "pub fn exported() {}",
            "",
            "pub fn internal_only() {}",
        ]
    )
    _with_fake_text(
        mod,
        {
            "crates/x/src/concrete.rs": source,
            "crates/x/src/lib.rs": "mod concrete;\npub use concrete::{exported};",
        },
    )
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/concrete.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/concrete.rs": {1, 3}},
    )
    assert len(findings) == 1
    assert "exported" in findings[0].detail
    assert "internal_only" not in findings[0].detail
def test_transport_errors_cover_every_below_json_failure() -> None:
    """socket.timeout only aliases TimeoutError from Python 3.10 on."""
    import http.client
    import socket
    import urllib.error

    mod = load_module()
    for exc_type in (
        socket.timeout,
        TimeoutError,
        ConnectionResetError,
        urllib.error.URLError,
        http.client.IncompleteRead,
    ):
        assert issubclass(exc_type, mod.TRANSPORT_ERRORS), exc_type


def test_call_deepseek_retries_transient_network_errors() -> None:
    import socket
    import urllib.request

    mod = load_module()
    calls = {"n": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"verdict\\":\\"pass\\"}"}}]}'

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout("The read operation timed out")
        return FakeResponse()

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = fake_urlopen
    mod.time.sleep = lambda _seconds: None
    try:
        payload = mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
        )
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep

    assert calls["n"] == 2
    assert payload == {"verdict": "pass"}


def test_call_deepseek_reraises_after_retries_exhausted() -> None:
    import socket
    import urllib.request

    mod = load_module()

    def always_timeout(request, timeout=None):
        raise socket.timeout("nope")

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = always_timeout
    mod.time.sleep = lambda _seconds: None
    try:
        mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
        )
    except mod.TRANSPORT_ERRORS:
        pass
    else:
        raise AssertionError("expected the timeout to propagate")
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep


def test_contains_sensitive_text_flags_typed_declaration() -> None:
    """A type annotation used to hide the literal from the pre-upload guard."""
    mod = load_module()
    for line in (
        f'const API_KEY: &str = "{VALUE}";',
        f'let api_key: String = "{VALUE}".into();',
        f'client_secret : &\'static str = "{VALUE}"',
        f'PASSWORD: str = "{VALUE}"',
    ):
        assert mod.contains_sensitive_text(line), line


def test_redact_sensitive_text_masks_typed_declaration() -> None:
    mod = load_module()
    redacted = mod.redact_sensitive_text(f'const API_KEY: &str = "{VALUE}";')
    assert VALUE not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_typed_declaration_guard_keeps_env_lookups_quiet() -> None:
    mod = load_module()
    for line in (
        'let key = std::env::var("DEEPSEEK_API_KEY")?;',
        "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}",
        "api_key: Option<String>,",
    ):
        assert not mod.contains_sensitive_text(line), line


# --- hunk header renumbering --------------------------------------------------


def test_split_oversized_hunk_renumbers_each_chunk() -> None:
    mod = load_module()
    header = ["diff --git a/big.rs b/big.rs", "--- a/big.rs", "+++ b/big.rs"]
    body = [f"+line {index}" + "y" * 900 for index in range(120)]
    chunks = mod.split_oversized_hunk(header, ["@@ -1,0 +1,120 @@ fn ctx()", *body])
    assert len(chunks) > 1

    starts = []
    for chunk in chunks:
        assert len(chunk) <= mod.MAX_FILE_DIFF_CHARS
        hunk_line = [line for line in chunk.splitlines() if line.startswith("@@")][0]
        parsed = mod.HUNK_HEADER.match(hunk_line)
        assert parsed is not None
        assert parsed.group(5) == " fn ctx()"
        starts.append((int(parsed.group(3)), int(parsed.group(4))))

    # Every chunk starts where the previous one ended, and the counts sum to
    # the original 120 added lines.
    assert starts[0][0] == 1
    for (start, count), (next_start, _) in zip(starts, starts[1:]):
        assert start + count == next_start
    assert sum(count for _, count in starts) == 120


def test_split_oversized_hunk_counts_context_and_removals() -> None:
    mod = load_module()
    header = ["diff --git a/a.rs b/a.rs", "--- a/a.rs", "+++ b/a.rs"]
    hunk = ["@@ -10,3 +20,3 @@", " ctx", "-gone", "+added"]
    chunks = mod.split_oversized_hunk(header, hunk)
    assert len(chunks) == 1
    hunk_line = [line for line in chunks[0].splitlines() if line.startswith("@@")][0]
    parsed = mod.HUNK_HEADER.match(hunk_line)
    # context + removal advance old; context + addition advance new.
    assert (int(parsed.group(1)), int(parsed.group(2))) == (10, 2)
    assert (int(parsed.group(3)), int(parsed.group(4))) == (20, 2)


def test_split_oversized_hunk_falls_back_on_unparseable_header() -> None:
    mod = load_module()
    header = ["diff --git a/a.rs b/a.rs", "--- a/a.rs", "+++ b/a.rs"]
    chunks = mod.split_oversized_hunk(header, ["@@ garbage @@", "+one"])
    assert len(chunks) == 1
    assert "@@ garbage @@" in chunks[0]


def test_line_deltas_classifies_diff_lines() -> None:
    mod = load_module()
    assert mod.line_deltas("+added") == (0, 1)
    assert mod.line_deltas("-removed") == (1, 0)
    assert mod.line_deltas(" context") == (1, 1)
    assert mod.line_deltas("\\ No newline at end of file") == (0, 0)


# --- API key validation -------------------------------------------------------


def test_api_key_problem_detects_non_ascii_without_echoing_it() -> None:
    mod = load_module()
    key = SK[:29] + "\u2026" + "tail"
    problem = mod.api_key_problem(key)
    assert problem is not None
    assert "non-ASCII" in problem
    assert "29" in problem
    assert key not in problem


def test_api_key_problem_detects_empty_and_whitespace() -> None:
    mod = load_module()
    assert "empty" in mod.api_key_problem("")
    assert "whitespace" in mod.api_key_problem("sk-abc def")


def test_api_key_problem_accepts_a_normal_key() -> None:
    mod = load_module()
    assert mod.api_key_problem(SK) is None


def test_api_key_error_finding_blocks_and_names_the_secret() -> None:
    mod = load_module()
    finding = mod.api_key_error_finding("The secret is empty.")
    assert finding.severity == "block"
    assert finding.id == "llm-api-key-invalid"
    assert "DEEPSEEK_API_KEY" in finding.summary


def test_run_git_disables_pathname_quoting() -> None:
    """git C-quotes non-ASCII paths by default, and the quoted form matches none."""
    mod = load_module()
    import subprocess

    captured = {}
    original = subprocess.run

    def fake_run(args, **kwargs):
        captured["args"] = args
        return original(["true"], capture_output=True, text=True)

    subprocess.run = fake_run
    try:
        mod.run_git(["diff", "--name-only"])
    finally:
        subprocess.run = original
    assert captured["args"][:3] == ["git", "-c", "core.quotePath=false"]


def test_contains_sensitive_text_flags_passphrase_with_spaces() -> None:
    mod = load_module()
    assert mod.contains_sensitive_text(f'password = "{PW}"')


def test_redact_sensitive_text_masks_whole_quoted_value() -> None:
    mod = load_module()
    redacted = mod.redact_sensitive_text(f'password = "{PW}"')
    assert "horse" not in redacted and "battery" not in redacted
    assert redacted == "password = [REDACTED_SECRET]"


def test_contains_sensitive_text_flags_unterminated_quote() -> None:
    mod = load_module()
    assert mod.contains_sensitive_text('secret = "opens here')


def test_metadata_names_are_not_credentials() -> None:
    """Allowing spaces in values means the name must carry the discrimination."""
    mod = load_module()
    assert not mod.is_credential_name("token_type")
    assert not mod.is_credential_name("secret_name")
    assert not mod.is_credential_name("private_key_path")
    assert mod.is_credential_name("api_token")
    assert mod.is_credential_name("password")
    assert not mod.contains_sensitive_text(
        'token_type: "WebGPU event token from another queue"'
    )
    assert not mod.redact_sensitive_text(
        'token_type: "an ordinary description"'
    ).count("[REDACTED_SECRET]")


def test_select_rule_sections_routes_on_changed_content() -> None:
    mod = load_module()
    path = "crates/tenferro-runtime/src/lib.rs"
    assert "Unsafe Code Boundary" not in mod.select_rule_sections([path])
    added = {path: [(10, "    unsafe { ptr.read() }")]}
    assert "Unsafe Code Boundary" in mod.select_rule_sections([path], added)


def test_content_triggers_name_only_documented_sections() -> None:
    mod = load_module()
    documented = set(mod.parse_repository_rules_sections())
    for _pattern, names in mod.CONTENT_TRIGGERS:
        assert set(names) <= documented, names


def test_content_triggers_never_select_human_only_sections() -> None:
    mod = load_module()
    path = "crates/tenferro-runtime/src/lib.rs"
    added = {path: [(1, "unsafe { }"), (2, "rayon::join(|| (), || ())")]}
    assert set(mod.select_rule_sections([path], added)).isdisjoint(
        mod.HUMAN_ONLY_SECTIONS
    )


def test_budget_is_smaller_than_the_workflow_timeout() -> None:
    """The script must finish before the job is killed, or no report is posted."""
    mod = load_module()
    workflow = (mod.ROOT / ".github" / "workflows" / "review_bot.yml").read_text()
    minutes = [
        int(line.split(":")[1].strip())
        for line in workflow.splitlines()
        if line.strip().startswith("timeout-minutes:")
    ]
    assert minutes, "review_bot.yml lost its job timeout"
    assert mod.DEFAULT_BUDGET_SECONDS < min(minutes) * 60


def test_call_deepseek_does_not_retry_past_the_deadline() -> None:
    import socket
    import urllib.request

    mod = load_module()
    calls = {"n": 0}

    def always_timeout(request, timeout=None):
        calls["n"] += 1
        raise socket.timeout("nope")

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = always_timeout
    mod.time.sleep = lambda _seconds: None
    try:
        mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
            deadline=mod.time.monotonic(),
        )
    except mod.TRANSPORT_ERRORS:
        pass
    else:
        raise AssertionError("expected the timeout to propagate")
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep
    assert calls["n"] == 1


def test_budget_exhausted_finding_warns_without_blocking() -> None:
    mod = load_module()
    finding = mod.budget_exhausted_finding(2, 5, 30.0)
    assert finding.severity == "warn"
    assert "2 of 5" in finding.detail
    # The configured budget, not the default, or the diagnostic misleads
    # whoever is trying to work out why the review was incomplete.
    assert "30s budget" in finding.detail
    assert "900s" not in finding.detail


def test_sensitive_diff_blocks_a_value_on_a_continuation_line() -> None:
    """The assignment can stay unchanged while only the value line is replaced."""
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/src/x.rs b/src/x.rs",
            "--- a/src/x.rs",
            "+++ b/src/x.rs",
            "@@ -1,2 +1,2 @@",
            f" const {KEYNAME}: &str =",
            '-    "old";',
            f'+    "{PW}";',
        ]
    )
    finding = mod.sensitive_diff_finding(diff)
    assert finding is not None
    assert finding.severity == "block"


def test_sensitive_diff_ignores_an_ordinary_continuation_value() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/src/x.rs b/src/x.rs",
            "--- a/src/x.rs",
            "+++ b/src/x.rs",
            "@@ -1,2 +1,2 @@",
            " let message =",
            '+    "hello world there";',
        ]
    )
    assert mod.sensitive_diff_finding(diff) is None


def test_sensitive_diff_ignores_an_unchanged_continuation_value() -> None:
    """Only added lines may be reported; a context value is pre-existing."""
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/src/x.rs b/src/x.rs",
            "--- a/src/x.rs",
            "+++ b/src/x.rs",
            "@@ -1,3 +1,3 @@",
            f" const {KEYNAME}: &str =",
            f'     "{PW}";',
            "+let unrelated = 1;",
        ]
    )
    assert mod.sensitive_diff_finding(diff) is None


def test_redactor_does_not_consume_a_deletion_marker_as_the_value() -> None:
    mod = load_module()
    text = 'const API_KEY: &str =\n-    "old";'
    # The separator must not cross the newline and swallow the `-` marker,
    # which used to leave the following line's literal untouched.
    assert mod.redact_sensitive_text(text).splitlines()[1] == '-    "old";'


def main() -> int:
    for test in [
        test_added_lines_by_file,
        test_default_deepseek_model_uses_current_v4_name,
        test_filter_findings_drops_unchanged_files,
        test_filter_findings_keeps_added_line,
        test_filter_findings_drops_line_finding_without_added_lines,
        test_filter_findings_drops_file_level_block_finding,
        test_filter_findings_drops_global_llm_finding_when_disallowed,
        test_reconcile_verdict_only_blocks_fail,
        test_select_rule_sections_includes_ad_for_ad_paths,
        test_select_rule_sections_includes_ad_for_tenferro_ad_crate,
        test_select_rule_sections_includes_performance_for_tensor_crates,
        test_select_rule_sections_includes_gpu_contract_for_gpu_paths,
        test_select_rule_sections_includes_benchmark_rules_for_bench_paths,
        test_select_rule_sections_includes_public_boundary_audits,
        test_final_cross_phase_multi_agent_audit_contract_is_present,
        test_extract_json_payload_strips_fence,
        test_extract_json_payload_reports_malformed_embedded_object,
        test_parse_findings_caps_model_output,
        test_parse_findings_normalizes_common_severity_aliases,
        test_llm_response_error_finding_blocks_with_diagnostic,
        test_transport_errors_cover_every_below_json_failure,
        test_call_deepseek_retries_transient_network_errors,
        test_call_deepseek_reraises_after_retries_exhausted,
        test_split_diff_chunks_respects_limit,
        test_split_large_file_diff_preserves_file_header,
        test_split_large_file_diff_splits_oversized_single_hunk,
        test_split_large_file_diff_splits_single_overlong_diff_line,
        test_scan_runtime_boundary_text_reports_forbidden_symbol,
        test_scan_runtime_boundary_text_can_limit_to_changed_lines,
        test_scan_runtime_boundary_text_ignores_comments,
        test_scan_runtime_boundary_text_tracks_block_comments_across_context,
        test_deterministic_checks_passes_added_lines_to_runtime_scan,
        test_redact_sensitive_text_masks_common_secret_forms,
        test_sensitive_diff_finding_checks_added_lines_only,
        test_contains_sensitive_text_ignores_env_lookup_code,
        test_contains_sensitive_text_distinguishes_identifiers_from_quoted_credentials,
        test_sensitive_diff_finding_reports_added_match_location,
        test_summarize_llm_review_computes_dropped_count,
        test_format_report_includes_llm_summary_line,
        test_format_report_omits_llm_summary_when_absent,
        test_rust_inline_test_blocks_reports_span,
        test_rust_inline_test_blocks_ignores_mod_declaration,
        test_inline_test_module_findings_flags_grown_block,
        test_inline_test_module_findings_ignores_a_shrinking_block,
        test_inline_test_module_findings_still_flags_real_growth,
        test_dependency_diagram_findings_rejects_a_crate_without_a_manifest,
        test_prompt_does_not_promise_pr_text_disclosure,
        test_inline_test_module_findings_exempts_tiny_leaf_module,
        test_inline_test_module_findings_skips_untouched_block,
        test_missing_doc_example_findings_flags_only_real_gaps,
        test_missing_doc_example_findings_ignores_unchanged_items,
        test_vacuous_doc_example_findings_flags_path_only_example,
        test_vacuous_doc_example_findings_ignores_comment_lines,
        test_vacuous_doc_example_findings_accepts_comment_only_example,
        test_vacuous_doc_example_findings_accepts_real_usage,
        test_ai_report_file_findings_flags_reports_outside_worklogs,
        test_parse_cargo_tenferro_dependencies_skips_optional_and_dev,
        test_parse_cargo_tenferro_dependencies_accepts_compact_optional_syntax,
        test_parse_dependency_diagram_handles_continuation_lines,
        test_dependency_diagram_findings_reports_missing_edge,
        test_dependency_diagram_findings_accepts_matching_edges,
        test_select_rule_sections_routes_unit_test_rules_for_src_rust,
        test_select_rule_sections_falls_back_for_unrouted_paths,
        test_pr_content_hygiene_section_is_parseable,
        test_ai_report_file_findings_skips_deleted_reports,
        test_rust_inline_test_blocks_recognizes_cfg_all_test,
        test_rust_inline_test_blocks_ignores_cfg_not_test,
        test_dependency_diagram_findings_checks_all_crates_on_doc_only_change,
        test_is_cfg_test_attr_matches_any_operand_order,
        test_is_cfg_test_attr_tracks_nested_not_polarity,
        test_rust_inline_test_blocks_recognizes_trailing_test_operand,
        test_module_publicly_reachable_respects_private_parent_declaration,
        test_missing_doc_example_findings_skips_privately_declared_module,
        test_dependency_diagram_findings_reports_deleted_crate_entry,
        test_rust_inline_test_blocks_handles_multiline_cfg_attribute,
        test_rust_inline_test_blocks_skips_multiline_non_cfg_attribute,
        test_module_publicly_reachable_follows_pub_use_reexport,
        test_select_rule_sections_routes_hygiene_for_unknown_top_level_rust,
        test_dependency_diagram_findings_reports_new_leaf_crate,
        test_pub_use_exports_parses_selective_and_glob_forms,
        test_missing_doc_example_findings_respects_selective_reexport,
        test_run_git_disables_pathname_quoting,
        test_contains_sensitive_text_flags_passphrase_with_spaces,
        test_redact_sensitive_text_masks_whole_quoted_value,
        test_contains_sensitive_text_flags_unterminated_quote,
        test_metadata_names_are_not_credentials,
        test_select_rule_sections_routes_on_changed_content,
        test_content_triggers_name_only_documented_sections,
        test_content_triggers_never_select_human_only_sections,
        test_budget_is_smaller_than_the_workflow_timeout,
        test_call_deepseek_does_not_retry_past_the_deadline,
        test_budget_exhausted_finding_warns_without_blocking,
        test_sensitive_diff_blocks_a_value_on_a_continuation_line,
        test_sensitive_diff_ignores_an_ordinary_continuation_value,
        test_sensitive_diff_ignores_an_unchanged_continuation_value,
        test_redactor_does_not_consume_a_deletion_marker_as_the_value,
        test_contains_sensitive_text_flags_typed_declaration,
        test_redact_sensitive_text_masks_typed_declaration,
        test_typed_declaration_guard_keeps_env_lookups_quiet,
        test_split_oversized_hunk_renumbers_each_chunk,
        test_split_oversized_hunk_counts_context_and_removals,
        test_split_oversized_hunk_falls_back_on_unparseable_header,
        test_line_deltas_classifies_diff_lines,
        test_api_key_problem_detects_non_ascii_without_echoing_it,
        test_api_key_problem_detects_empty_and_whitespace,
        test_api_key_problem_accepts_a_normal_key,
        test_api_key_error_finding_blocks_and_names_the_secret,
    ]:
        test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
