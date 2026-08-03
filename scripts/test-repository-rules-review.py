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
    assert all("@@ -1,1 +1,8 @@" in chunk for chunk in chunks)
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
    assert all("@@ -0,0 +1 @@" in chunk for chunk in chunks)
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
            '+        api_token = "live-token-value-1234567890"',
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
    ]:
        test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
