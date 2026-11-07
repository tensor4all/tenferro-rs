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

    assert mod.DEFAULT_MODEL == "deepseek-flash"
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


def test_inline_test_module_findings_flags_a_new_block_during_extraction() -> None:
    """A shrink elsewhere must not hide a newly added block.

    The file-wide total can fall while a PR adds a fresh oversized inline
    module, so the net-size exemption is applied per block: a block whose
    OPENER is itself an added line is judged on its own.
    """
    mod = load_module()
    base = "\n".join(
        ["fn p() {}"]
        + [f"fn h{i}() {{}}" for i in range(160)]
        + ["#[cfg(test)]", "mod tests {"]
        + [f"    // c{i}" for i in range(60)]
        + ["}"]
    )
    head = "\n".join(
        ["fn p() {}"]
        + [f"fn h{i}() {{}}" for i in range(160)]
        + ["#[cfg(test)]", "mod tests {", "    // c0", "}"]
        + ["#[cfg(test)]", "mod more {"]
        + [f"    // n{i}" for i in range(30)]
        + ["}"]
    )
    _with_fake_text(
        mod,
        {"crates/x/src/a.rs": head},
        base_mapping={"crates/x/src/a.rs": base},
    )
    findings = mod.inline_test_module_findings(
        ["crates/x/src/a.rs"],
        ref="HEAD",
        base="origin/main",
        worktree=False,
        added_lines={"crates/x/src/a.rs": set(range(163, 200))},
    )
    assert [item.id for item in findings] == ["inline-test-module"]
    assert findings[0].line == 166


def test_is_rust_doc_fence_accepts_only_rust_attributes() -> None:
    mod = load_module()
    assert mod.is_rust_doc_fence("")
    assert mod.is_rust_doc_fence("rust")
    assert mod.is_rust_doc_fence("rust,no_run")
    assert mod.is_rust_doc_fence("ignore")
    assert not mod.is_rust_doc_fence("text")
    assert not mod.is_rust_doc_fence("bash")
    assert not mod.is_rust_doc_fence("toml")


def test_vacuous_doc_example_findings_skips_non_rust_fences() -> None:
    """A ```text grammar block is prose, not a doctest.

    Treating every fence as a doctest reported `vacuous-doc-example` on
    ordinary syntax documentation.
    """
    mod = load_module()
    doc = "\n".join(
        [
            "/// Grammar:",
            "///",
            "/// ```text",
            "/// Widget;",
            "/// ```",
            "pub fn spin() {}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": doc})
    findings = mod.vacuous_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {4}},
    )
    assert findings == []


def test_parse_cargo_tenferro_dependencies_reads_target_tables() -> None:
    """`[target.'cfg(unix)'.dependencies]` is a real production dependency.

    Matching the section name exactly ignored it, so a matching diagram edge
    was reported stale and a missing one passed.
    """
    mod = load_module()
    cargo = "\n".join(
        [
            "[dependencies]",
            "tenferro-runtime.workspace = true",
            "[target.'cfg(unix)'.dependencies]",
            "tenferro-cpu.workspace = true",
            "[target.'cfg(windows)'.dependencies.tenferro-gpu]",
            "workspace = true",
            "[target.'cfg(unix)'.dev-dependencies]",
            "tenferro-fft.workspace = true",
            "[target.'cfg(unix)'.dependencies.tenferro-ad]",
            "workspace = true",
            "optional = true",
        ]
    )
    assert mod.parse_cargo_tenferro_dependencies(cargo) == {
        "tenferro-runtime",
        "tenferro-cpu",
        "tenferro-gpu",
    }


def test_pub_item_matches_a_public_union() -> None:
    """A public union is a documented public type like a struct or enum."""
    mod = load_module()
    match = mod.PUB_ITEM.match("pub union Slot { a: u32 }")
    assert match is not None
    assert match.group(2) == "Slot"


def test_rust_inline_test_blocks_ignores_braces_in_literals() -> None:
    """A brace inside a literal is not structural.

    `let expected = "}";` ended the detected block at that line, so a test
    added below fell outside the span and evaded the audit entirely.
    """
    mod = load_module()
    src = "\n".join(
        [
            "#[cfg(test)]",
            "mod tests {",
            "    #[test]",
            "    fn a() {",
            '        let expected = "}";',
            "    }",
            "    #[test]",
            "    fn b() { assert!(true); }",
            "}",
        ]
    )
    assert mod.rust_inline_test_blocks(src) == [(1, 9)]


def test_rust_inline_test_blocks_handles_literal_edge_cases() -> None:
    """Char literals, raw strings, escapes and multi-line strings.

    Lifetimes (`&'a str`) must NOT be read as an unterminated char literal.
    """
    mod = load_module()
    cases = [
        (["    let c = '}';", "    let d = '{';"], 5),
        (["    fn f<'a>(x: &'a str) -> &'a str { x }"], 4),
        (['    let s = r#"} { "#;'], 4),
        ([r'    let s = "a\"} b";'], 4),
        (['    let s = "start', '        } still string";'], 5),
    ]
    for body, expected_end in cases:
        src = "\n".join(["#[cfg(test)]", "mod tests {"] + body + ["}"])
        assert mod.rust_inline_test_blocks(src) == [(1, expected_end)], body


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


def test_missing_doc_example_findings_flags_added_public_trait_method() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "pub trait PublicApi {",
            "    fn old(&self);",
            "    fn method(&self);",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    diff = "\n".join(
        [
            "diff --git a/crates/x/src/lib.rs b/crates/x/src/lib.rs",
            "--- a/crates/x/src/lib.rs",
            "+++ b/crates/x/src/lib.rs",
            "@@ -1,3 +1,4 @@",
            " pub trait PublicApi {",
            "     fn old(&self);",
            "+    fn method(&self);",
            " }",
        ]
    )
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines=mod.added_lines_by_file(diff),
    )
    assert len(findings) == 1
    assert findings[0].id == "missing-doc-examples"
    assert findings[0].severity == "warn"
    assert "PublicApi::method" in findings[0].detail


def test_missing_doc_example_findings_accepts_trait_method_examples() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "/// Public API.",
            "///",
            "/// # Examples",
            "///",
            "/// ```",
            "/// let _ = 1;",
            "/// ```",
            "pub trait PublicApi {",
            "    /// Calls the API.",
            "    ///",
            "    /// # Examples",
            "    ///",
            "    /// ```",
            "    /// let _ = 1;",
            "    /// ```",
            "    fn method(&self);",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {16}},
    )
    assert findings == []


def test_missing_doc_example_findings_skips_hidden_trait_methods() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "/// Public API.",
            "///",
            "/// # Examples",
            "///",
            "/// ```",
            "/// let _ = 1;",
            "/// ```",
            "pub trait PublicApi {",
            "    #[doc(hidden)]",
            "    fn hidden_method(&self);",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {10}},
    )
    assert findings == []


def test_missing_doc_example_findings_skips_private_traits() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "trait PrivateApi {",
            "    fn method(&self);",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {1, 2, 3}},
    )
    assert findings == []


def test_missing_doc_example_findings_recognizes_trait_method_forms() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "/// Public API.",
            "///",
            "/// # Examples",
            "///",
            "/// ```",
            "/// let _ = 1;",
            "/// ```",
            "pub trait PublicApi {",
            "    fn required(&self);",
            "    fn provided(&self) {",
            "        // } in a comment",
            '        let _string = "}";',
            "        /* { in a block comment */",
            "        let _value = 1;",
            "    }",
            "    fn default_method(&self) {",
            '        let _raw = r#"} {"#;',
            "    }",
            "    async fn async_method(&self);",
            "    unsafe fn unsafe_method(&self);",
            "    const fn const_method(&self);",
            '    extern "C" fn extern_method(&self);',
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": set(range(1, len(source.splitlines()) + 1))},
    )
    assert len(findings) == 1
    detail = findings[0].detail
    for method in (
        "required",
        "provided",
        "default_method",
        "async_method",
        "unsafe_method",
        "const_method",
        "extern_method",
    ):
        assert f"PublicApi::{method}" in detail


def test_missing_doc_example_findings_ignores_unchanged_trait_methods() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "pub trait PublicApi {",
            "    fn old(&self);",
            "    // unrelated changed line",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    diff = "\n".join(
        [
            "diff --git a/crates/x/src/lib.rs b/crates/x/src/lib.rs",
            "--- a/crates/x/src/lib.rs",
            "+++ b/crates/x/src/lib.rs",
            "@@ -1,3 +1,4 @@",
            " pub trait PublicApi {",
            "     fn old(&self);",
            "+    // unrelated changed line",
            " }",
        ]
    )
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines=mod.added_lines_by_file(diff),
    )
    assert findings == []


def test_missing_doc_example_findings_uses_trait_name_for_selective_reexports() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "/// Public API.",
            "///",
            "/// # Examples",
            "///",
            "/// ```",
            "/// let _ = 1;",
            "/// ```",
            "pub trait PublicApi {",
            "    fn method(&self);",
            "}",
            "",
            "/// Private API.",
            "///",
            "/// # Examples",
            "///",
            "/// ```",
            "/// let _ = 1;",
            "/// ```",
            "pub trait PrivateApi {",
            "    fn hidden_method(&self);",
            "}",
        ]
    )
    _with_fake_text(
        mod,
        {
            "crates/x/src/concrete.rs": source,
            "crates/x/src/lib.rs": "mod concrete;\npub use concrete::PublicApi;",
        },
    )
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/concrete.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={
            "crates/x/src/concrete.rs": set(range(1, len(source.splitlines()) + 1))
        },
    )
    assert len(findings) == 1
    assert "PublicApi::method" in findings[0].detail
    assert "PrivateApi::hidden_method" not in findings[0].detail


def test_missing_doc_example_findings_ignores_nested_trait_helpers() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "pub trait PublicApi {",
            "    fn provided(&self) {",
            "        fn helper() {}",
            "    }",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {3}},
    )
    assert findings == []


def test_missing_doc_example_findings_ignores_trait_comments_and_literals() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "pub trait PublicApi {",
            "    /*",
            "        fn block_fake(&self);",
            "    */",
            "    const NORMAL: &str = \"\\",
            "fn normal_fake(&self);\\",
            "\";",
            "    const RAW: &str = r#\"",
            "fn raw_fake(&self);",
            "\"#;",
            "    // fn line_fake(&self);",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {3, 6, 9, 11}},
    )
    assert findings == []


def test_missing_doc_example_findings_ignores_traits_in_outer_comments() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "/*",
            "pub trait Fake {",
            "    fn fake(&self);",
            "}",
            "*/",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": set(range(1, 6))},
    )
    assert findings == []


def test_missing_doc_example_findings_tracks_nested_block_comments() -> None:
    mod = load_module()
    source = "\n".join(
        [
            "pub trait PublicApi {",
            "    /*",
            "        /*",
            "        */",
            "        }",
            "        fn fake(&self);",
            "    */",
            "    fn real(&self);",
            "}",
        ]
    )
    _with_fake_text(mod, {"crates/x/src/lib.rs": source})
    findings = mod.missing_doc_example_findings(
        ["crates/x/src/lib.rs"],
        ref="HEAD",
        worktree=False,
        added_lines={"crates/x/src/lib.rs": {6, 8}},
    )
    assert len(findings) == 1
    assert "PublicApi::real" in findings[0].detail
    assert "PublicApi::fake" not in findings[0].detail


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


APPLE_PRE_EDIT_FIXTURE = """\
/// Explicit Apple context.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
///
/// // Constructor type signature.
/// let _constructor: fn() -> tenferro_tensor::Result<AppleContext> = AppleContext::new;
/// ```
///
/// Create a context with [`AppleContext::new`].
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
///
/// // The constructor is fallible.
/// let _constructor: fn() -> tenferro_tensor::Result<AppleContext> = AppleContext::new;
/// ```
///
/// Return this context's allocation-domain identity.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
/// use tenferro_tensor::AllocationDomainId;
///
/// // Inspect the method signature.
/// let _method: fn(&AppleContext) -> AllocationDomainId = AppleContext::domain_id;
/// ```
///
/// Return the paired CPU backend.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
///
/// // The endpoint is available from the context.
/// let _method = AppleContext::cpu_backend;
/// ```
///
/// Return the paired Metal WebGPU backend.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
///
/// // The endpoint is available from the context.
/// let _method = AppleContext::metal_backend;
/// ```
///
/// Return transfer statistics.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
///
/// // Read the current counters.
/// let _method = AppleContext::transfer_stats;
/// ```
///
/// Upload a host tensor.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
///
/// // Explicitly create managed storage.
/// let _method = AppleContext::upload_tensor;
/// ```
///
/// Download a managed tensor.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::apple::AppleContext;
///
/// // Explicitly copy managed storage back to the host.
/// let _method = AppleContext::download_tensor;
/// ```
"""


def test_vacuous_doc_example_fixture_audits_pre_edit_apple_examples() -> None:
    mod = load_module()
    fixture_path = "crates/tenferro-gpu/src/webgpu/apple.rs"
    _with_fake_text(mod, {fixture_path: APPLE_PRE_EDIT_FIXTURE})
    findings = mod.vacuous_doc_example_findings(
        [fixture_path],
        ref="HEAD",
        worktree=False,
        added_lines={fixture_path: set(range(1, len(APPLE_PRE_EDIT_FIXTURE.splitlines()) + 1))},
    )
    assert len(findings) == 8
    assert all(item.id == "vacuous-doc-example" for item in findings)


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


def test_filter_findings_keeps_file_level_block_for_deletions() -> None:
    """A deletion-only violation has no new-file line the model can anchor to."""
    mod = load_module()
    block = mod.Finding("x", "block", "S", "a.rs", None, "removed validation", "d")
    assert mod.filter_findings([block], ["a.rs"], {}) == []
    kept = mod.filter_findings([block], ["a.rs"], {}, files_with_deletions={"a.rs"})
    assert kept == [block]


def test_files_with_unanchorable_deletions_reads_the_new_side_path() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/a.rs b/a.rs",
            "--- a/a.rs",
            "+++ b/a.rs",
            "@@ -1,2 +1,1 @@",
            " keep",
            "-gone",
            "diff --git a/b.rs b/b.rs",
            "--- a/b.rs",
            "+++ b/b.rs",
            "@@ -1,1 +1,2 @@",
            " keep",
            "+added",
        ]
    )
    assert mod.files_with_unanchorable_deletions(diff) == {"a.rs"}


def test_files_with_unanchorable_deletions_keeps_a_fully_deleted_file() -> None:
    """A whole-file deletion has `+++ /dev/null`; keep its old-side path.

    Clearing `current_file` on the `/dev/null` header omitted the deleted file
    from the very set that exists to retain unanchored blocks about deletions.
    """
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/crates/x/src/old.rs b/crates/x/src/old.rs",
            "deleted file mode 100644",
            "--- a/crates/x/src/old.rs",
            "+++ /dev/null",
            "@@ -1,2 +0,0 @@",
            "-fn validate() {}",
            "-// SAFETY: checked above",
        ]
    )
    assert mod.files_with_unanchorable_deletions(diff) == {"crates/x/src/old.rs"}


def test_files_with_unanchorable_deletions_skips_replacement_edits() -> None:
    """A replacement edit adds lines, so a finding about it can be anchored.

    Treating every patch that removes a line as unanchorable disabled the
    anti-generalization filter for most modified files.
    """
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/a.rs b/a.rs",
            "--- a/a.rs",
            "+++ b/a.rs",
            "@@ -1,1 +1,1 @@",
            "-let a = 1;",
            "+let a = 2;",
            "diff --git a/b.rs b/b.rs",
            "--- a/b.rs",
            "+++ b/b.rs",
            "@@ -1,2 +1,1 @@",
            " keep",
            "-// SAFETY: checked above",
        ]
    )
    assert mod.files_with_unanchorable_deletions(diff) == {"b.rs"}


def test_sensitive_diff_ignores_an_expression_continuation() -> None:
    """A credential-LOADING expression is not a credential value.

    The bare continuation alternative matched the whole expression, so
    `let api_key =` followed by `std::env::var("API_KEY")?;` blocked the
    required gate even though no credential appears in the diff.
    """
    mod = load_module()
    for continuation in (
        '+    std::env::var("API_KEY")?;',
        "+    env::var(name).unwrap();",
        "+    settings.api_key();",
    ):
        diff = "\n".join(
            [
                "diff --git a/src/x.rs b/src/x.rs",
                "--- a/src/x.rs",
                "+++ b/src/x.rs",
                "@@ -1,2 +1,2 @@",
                f" {KEYNAME} =",
                continuation,
            ]
        )
        assert mod.sensitive_diff_finding(diff) is None, continuation


def test_files_with_unanchorable_deletions_keeps_a_mixed_hunk_deletion() -> None:
    """An unrelated addition elsewhere in the file is not a valid anchor.

    Aggregating additions and deletions per FILE removed a deletion-only hunk
    from the set as soon as any other hunk added a line, so `filter_findings`
    dropped a real block about removed validation or a `// SAFETY:` comment.
    """
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/a.rs b/a.rs",
            "--- a/a.rs",
            "+++ b/a.rs",
            "@@ -1,2 +1,1 @@",
            " keep",
            "-// SAFETY: checked above",
            "@@ -20,1 +19,2 @@",
            " ctx",
            "+let extra = 1;",
        ]
    )
    assert mod.files_with_unanchorable_deletions(diff) == {"a.rs"}


def test_sensitive_diff_ignores_a_field_access_continuation() -> None:
    """A dotted member expression is not a credential value.

    `.` in the bare class let `let api_key =` continued by `settings.api_key;`
    read as a leaked secret, so credential-LOADING code still could not pass
    the required gate.
    """
    mod = load_module()
    for continuation in ("+    settings.api_key;", "+    self.token;"):
        diff = "\n".join(
            [
                "diff --git a/src/x.rs b/src/x.rs",
                "--- a/src/x.rs",
                "+++ b/src/x.rs",
                "@@ -1,2 +1,2 @@",
                f" {KEYNAME} =",
                continuation,
            ]
        )
        assert mod.sensitive_diff_finding(diff) is None, continuation


def test_sensitive_diff_ignores_an_identifier_continuation() -> None:
    """A plain identifier is spelled from the same alphabet as a credential.

    Length and word shape carry the discrimination: `configured_token` (16)
    and `ENV_API_KEY` (11) are below the 20-character floor the file already
    uses for bare credentials, and snake_case / SCREAMING_SNAKE_CASE is an
    identifier convention that base64/hex credentials do not follow.
    """
    mod = load_module()
    for continuation in (
        "+    configured_token;",
        "+    ENV_API_KEY;",
        "+    configured_authentication_token;",
    ):
        diff = "\n".join(
            [
                "diff --git a/src/x.rs b/src/x.rs",
                "--- a/src/x.rs",
                "+++ b/src/x.rs",
                "@@ -1,2 +1,2 @@",
                f" {KEYNAME} =",
                continuation,
            ]
        )
        assert mod.sensitive_diff_finding(diff) is None, continuation


def test_is_standalone_secret_value_keeps_credential_shapes() -> None:
    """The positive cases the guard exists for must survive the narrowing."""
    mod = load_module()
    assert mod.is_standalone_secret_value("    abcdefghijklmnopqrstuvwx")
    assert mod.is_standalone_secret_value("    A1b2C3d4E5f6G7h8J9k0;")
    assert mod.is_standalone_secret_value('    "quoted-secret-value"')
    assert not mod.is_standalone_secret_value("    short_id;")


def test_files_with_unanchorable_deletions_reads_headers_outside_hunks() -> None:
    """`--- validation` inside a hunk is a deleted body line, not a header.

    Treating it as an old-file header reset the hunk flags, so the file was
    lost from the set and a real unanchored block was still discarded.
    """
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/a.rs b/a.rs",
            "--- a/a.rs",
            "+++ b/a.rs",
            "@@ -1,2 +1,1 @@",
            " keep",
            "--- validation",
        ]
    )
    assert mod.files_with_unanchorable_deletions(diff) == {"a.rs"}
    # The `+++` twin: deleting nothing, adding a line that reads `++ added`.
    added = "\n".join(
        [
            "diff --git a/b.rs b/b.rs",
            "--- a/b.rs",
            "+++ b/b.rs",
            "@@ -1,2 +1,3 @@",
            " keep",
            "+++ added",
        ]
    )
    assert mod.files_with_unanchorable_deletions(added) == set()


def test_sensitive_diff_blocks_a_bare_continuation_value() -> None:
    """The continuation value need not be quoted."""
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/src/x.rs b/src/x.rs",
            "--- a/src/x.rs",
            "+++ b/src/x.rs",
            "@@ -1,2 +1,2 @@",
            f" {KEYNAME} =",
            "-old",
            "+abcdefghijklmnopqrstuvwx",
        ]
    )
    assert mod.sensitive_diff_finding(diff) is not None


def test_sensitive_diff_ignores_an_ordinary_bare_continuation() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/src/x.rs b/src/x.rs",
            "--- a/src/x.rs",
            "+++ b/src/x.rs",
            "@@ -1,2 +1,2 @@",
            " let total =",
            "+    compute_sum(values)",
        ]
    )
    assert mod.sensitive_diff_finding(diff) is None


# ---------------------------------------------------------------------------
# Review gate #88: a hunk body line must not be able to name a file.
#
# `git diff` is a state machine and `@@ -a,b +c,d @@` states exactly how many
# old-side and new-side lines follow. Recognising `+++ ` as a file header
# without consulting that state let an ADDED line whose content starts with
# `++ ` -- rendered as `+++ b/whatever` -- become the current file, so every
# later addition was attributed to a path the pull-request author chose. The
# fixtures are real `git diff` output of a block comment that carries the line
# `++ b/review.txt`.
#
# The three fixtures are deliberately paired: `forged-header.diff` is the attack,
# `normal.diff` is the same edit without the payload (attribution must not move
# at all), and `devnull.diff` is a create plus a delete (the `/dev/null`
# contract).
# ---------------------------------------------------------------------------

REVIEW_GATE_FIXTURES = ROOT / "scripts" / "fixtures" / "review-gate"
FROZEN_FILE = "crates/tenferro-tensor/src/dispatch.rs"
FORGED_FILE = "review.txt"


def review_gate_fixture(name: str) -> str:
    return (REVIEW_GATE_FIXTURES / name).read_text(encoding="utf-8")


def legacy_added_lines_with_text(diff_text: str) -> dict[str, list[tuple[int, str]]]:
    """The pre-fix loop, verbatim from this script before the #88 fix.

    Kept so the vulnerability stays pinned: if an edit reintroduces "a file
    header is any line starting with `+++ `", this records exactly what that
    buys an attacker instead of letting the regression pass silently.
    """

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


def test_the_pre_fix_parser_really_was_forgeable() -> None:
    """Pin the defect: the old loop did believe the payload."""

    legacy = legacy_added_lines_with_text(review_gate_fixture("forged-header.diff"))

    assert FORGED_FILE in legacy, "fixture no longer exercises the forged header"
    # The payload line was consumed as a file header, so it is not an addition
    # anywhere; everything after it landed on the author's chosen path.
    assert legacy[FROZEN_FILE] == [(6, "    /*")]
    assert legacy[FORGED_FILE] == [(7, "*/")]


def test_a_hunk_body_line_cannot_reassign_the_file() -> None:
    mod = load_module()

    added = mod.added_lines_with_text(review_gate_fixture("forged-header.diff"))

    assert FORGED_FILE not in added
    assert sorted(added) == [FROZEN_FILE]


def test_every_addition_stays_with_the_real_file() -> None:
    mod = load_module()

    added = mod.added_lines_with_text(review_gate_fixture("forged-header.diff"))

    assert added[FROZEN_FILE] == [
        (6, "    /*"),
        (7, "++ b/review.txt"),
        (8, "*/"),
    ]


def test_added_line_numbers_stay_with_the_real_file() -> None:
    mod = load_module()

    added = mod.added_lines_by_file(review_gate_fixture("forged-header.diff"))

    assert added == {FROZEN_FILE: {6, 7, 8}}


def test_a_forged_payload_does_not_become_the_deleted_from_file() -> None:
    mod = load_module()

    deleted = mod.files_with_unanchorable_deletions(
        review_gate_fixture("forged-header.diff")
    )

    assert FORGED_FILE not in deleted
    # The forged hunk is a replacement edit (it removes and adds), so the file
    # is correctly not in the deletion-only set.
    assert deleted == set()


def test_a_sensitive_line_that_renders_as_a_header_is_still_inspected() -> None:
    """The same missing state hid a secret from the upload guard."""

    mod = load_module()
    # Assembled at runtime: this file must not contain a secret-shaped literal.
    payload = "++ " + PAT
    # The pre-fix loop's first test was `line.startswith("+++ ")` and it
    # `continue`d on a match, so this line never reached the secret patterns.
    assert ("+" + payload).startswith("+++ ")

    diff = "\n".join(
        [
            f"diff --git a/{FROZEN_FILE} b/{FROZEN_FILE}",
            f"--- a/{FROZEN_FILE}",
            f"+++ b/{FROZEN_FILE}",
            "@@ -1,2 +1,3 @@",
            " fn main() {}",
            "+" + payload,
        ]
    )

    assert mod.sensitive_diff_location(diff) == (FROZEN_FILE, 2)


def test_an_honest_diff_keeps_its_attribution() -> None:
    mod = load_module()

    added = mod.added_lines_with_text(review_gate_fixture("normal.diff"))

    assert added == {
        FROZEN_FILE: [
            (6, "    let added_marker = total * 2;"),
            (7, "    added_marker"),
        ]
    }
    assert mod.added_lines_by_file(review_gate_fixture("normal.diff")) == {
        FROZEN_FILE: {6, 7}
    }


def test_dev_null_handling_is_unchanged() -> None:
    """A create and a delete keep the meaning the call sites already rely on."""

    mod = load_module()
    diff = review_gate_fixture("devnull.diff")
    fresh = "crates/tenferro-tensor/src/fresh.rs"

    # The new side of a delete is `/dev/null`, so there is no new-side path and
    # no bucket -- exactly what the pre-fix helpers reported.
    assert mod.added_lines_with_text(diff) == {
        fresh: [
            (1, "pub fn fresh() -> u8 {"),
            (2, "    9"),
            (3, "}"),
        ]
    }
    assert mod.added_lines_by_file(diff) == {fresh: {1, 2, 3}}
    # A fully deleted file falls back to its old-side path, which is the path a
    # finding about the removal names.
    assert mod.files_with_unanchorable_deletions(diff) == {FROZEN_FILE}


def test_a_real_header_after_a_spent_hunk_is_honoured_again() -> None:
    """Do not over-correct: headers resume once the hunk budget is spent."""

    mod = load_module()
    combined = (
        review_gate_fixture("forged-header.diff")
        + "diff --git a/src/other.rs b/src/other.rs\n"
        "--- a/src/other.rs\n"
        "+++ b/src/other.rs\n"
        "@@ -1 +1,2 @@\n"
        " fn other() {}\n"
        "+fn added() {}\n"
    )

    added = mod.added_lines_with_text(combined)

    assert sorted(added) == [FROZEN_FILE, "src/other.rs"]
    assert added["src/other.rs"] == [(2, "fn added() {}")]


def test_omitted_counts_and_the_no_newline_marker() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/x.rs b/x.rs",
            "--- a/x.rs",
            "+++ b/x.rs",
            "@@ -1 +1 @@",
            "-old",
            "+new",
            "\\ No newline at end of file",
        ]
    )

    assert mod.added_lines_with_text(diff) == {"x.rs": [(1, "new")]}


def test_a_rename_reports_both_paths() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/old.rs b/new.rs",
            "similarity index 100%",
            "rename from old.rs",
            "rename to new.rs",
        ]
    )

    assert sorted(mod.changed_paths(diff)) == ["new.rs", "old.rs"]


def test_name_status_cross_check_flags_a_path_git_does_not_list() -> None:
    """Second layer: a diff naming a path git does not list must fail closed."""

    mod = load_module()
    forged_as_headers = "".join(
        [
            f"diff --git a/{FROZEN_FILE} b/{FROZEN_FILE}\n",
            f"--- a/{FROZEN_FILE}\n",
            f"+++ b/{FORGED_FILE}\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
        ]
    )

    consistency = mod.paths_consistent(forged_as_headers, f"M\0{FROZEN_FILE}\0")

    assert not consistency.ok
    assert consistency.unexpected == (FORGED_FILE,)
    assert consistency.missing == ()


def test_name_status_cross_check_accepts_an_honest_diff() -> None:
    mod = load_module()

    for name in ("forged-header.diff", "normal.diff"):
        consistency = mod.paths_consistent(
            review_gate_fixture(name), f"M\0{FROZEN_FILE}\0"
        )
        assert consistency.ok, (name, consistency)


def test_name_status_cross_check_understands_renames_and_deletes() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/old.rs b/new.rs",
            "similarity index 100%",
            "rename from old.rs",
            "rename to new.rs",
            "diff --git a/gone.rs b/gone.rs",
            "deleted file mode 100644",
            "--- a/gone.rs",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-fn gone() {}",
        ]
    )
    name_status = "R100\0old.rs\0new.rs\0D\0gone.rs\0"

    consistency = mod.paths_consistent(diff, name_status)

    assert consistency.ok, consistency


def test_path_consistency_finding_blocks_only_on_a_mismatch() -> None:
    mod = load_module()

    honest = mod.path_consistency_finding(
        review_gate_fixture("normal.diff"), f"M\0{FROZEN_FILE}\0"
    )
    assert honest is None

    mismatch = mod.path_consistency_finding(
        f"diff --git a/{FROZEN_FILE} b/{FROZEN_FILE}\n"
        f"--- a/{FROZEN_FILE}\n"
        f"+++ b/{FORGED_FILE}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n",
        f"M\0{FROZEN_FILE}\0",
    )
    assert mismatch is not None
    assert mismatch.severity == "block"
    assert FORGED_FILE in mismatch.detail


def test_binary_mode_and_quoted_paths_do_not_look_like_tampering() -> None:
    """Fail-closed must not fire on the shapes git itself emits."""

    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/blob.bin b/blob.bin",
            "index 5d27a55..1f09903 100644",
            "Binary files a/blob.bin and b/blob.bin differ",
            "diff --git a/mode.sh b/mode.sh",
            "old mode 100644",
            "new mode 100755",
            "index 587be6b..975fbec",
            "--- a/mode.sh",
            "+++ b/mode.sh",
            "@@ -1 +1 @@",
            "-x",
            "+y",
            "diff --git a/src/\u00fc\u00f1\u00ee/na\u00efve file.rs b/src/\u00fc\u00f1\u00ee/na\u00efve file.rs",
            "index 7898192..422c2b7 100644",
            "--- a/src/\u00fc\u00f1\u00ee/na\u00efve file.rs\t",
            "+++ b/src/\u00fc\u00f1\u00ee/na\u00efve file.rs\t",
            "@@ -1 +1,2 @@",
            " a",
            "+b",
        ]
    )
    name_status = (
        "M\0blob.bin\0M\0mode.sh\0M\0src/\u00fc\u00f1\u00ee/na\u00efve file.rs\0"
    )

    assert mod.paths_consistent(diff, name_status).ok
    assert mod.added_lines_by_file(diff) == {
        # A binary or mode-only file names its path but adds no line; the empty
        # bucket is inert at every call site (`if not touched: continue`).
        "blob.bin": set(),
        "mode.sh": {1},
        "src/\u00fc\u00f1\u00ee/na\u00efve file.rs": {2},
    }


# git's quoted forms, byte-for-byte as `git -c core.quotePath=true diff` emits
# them: non-ASCII becomes three-digit octal BYTES, and `"`, tab and backslash
# are quoted even with `core.quotePath=false` because the quoting has to stay
# unambiguous. Spelling these out (rather than building them) is the point:
# they are the shapes the gate actually receives.
QUOTED_NAIVE = '"src/na\\303\\257ve file.rs"'
QUOTED_JAPANESE = '"src/\\346\\227\\245\\346\\234\\254\\350\\252\\236.rs"'
QUOTED_QUOTE = '"q\\"uote.rs"'
REAL_NAIVE = "src/na\u00efve file.rs"
REAL_JAPANESE = "src/\u65e5\u672c\u8a9e.rs"
REAL_QUOTE = 'q"uote.rs'


def test_unquote_diff_path_decodes_git_octal_bytes() -> None:
    """git quotes non-ASCII as octal BYTES, which `json.loads` cannot read.

    `"\\303"` is not a JSON escape, so the old implementation raised, fell back
    to `path[1:-1]`, and returned the escape sequence as if it were the name.
    """

    mod = load_module()

    assert mod._unquote_diff_path(QUOTED_NAIVE) == REAL_NAIVE
    assert mod._unquote_diff_path(QUOTED_JAPANESE) == REAL_JAPANESE
    # The regression itself: the escaped literal must never come back.
    assert mod._unquote_diff_path(QUOTED_NAIVE) != QUOTED_NAIVE[1:-1]
    assert "\\303" not in mod._unquote_diff_path(QUOTED_NAIVE)


def test_unquote_diff_path_decodes_single_character_escapes() -> None:
    mod = load_module()

    assert mod._unquote_diff_path(QUOTED_QUOTE) == REAL_QUOTE
    assert mod._unquote_diff_path('"tab\\tx.rs"') == "tab\tx.rs"
    assert mod._unquote_diff_path('"back\\\\slash.rs"') == "back\\slash.rs"
    assert mod._unquote_diff_path('"new\\nline.rs"') == "new\nline.rs"
    assert mod._unquote_diff_path('"bell\\a.rs"') == "bell\a.rs"


def test_unquote_diff_path_leaves_unquoted_paths_alone() -> None:
    mod = load_module()

    assert mod._unquote_diff_path("src/plain file.rs") == "src/plain file.rs"
    assert mod._unquote_diff_path("") == ""
    assert mod._unquote_diff_path('"') == '"'


def test_unquote_diff_path_keeps_a_truncated_sequence_unmatchable() -> None:
    """A trailing byte fraction is not repaired into a plausible path."""

    mod = load_module()

    # `\303` alone is the first byte of a two-byte sequence: no character.
    assert mod._unquote_diff_path('"a\\303"') == "a\udcc3"
    # A lone trailing backslash is literal, not a dropped escape.
    assert mod._unquote_diff_path('"odd\\"') == "odd\\"


def test_quoted_diff_paths_match_unquoted_name_status() -> None:
    """The diff header quotes, `--name-status -z` does not: both must match.

    This is the production shape under `core.quotePath=true`: one side of the
    gate reads quoted headers, the other reads raw NUL-separated paths.
    """

    mod = load_module()
    diff = "\n".join(
        [
            f"diff --git {QUOTED_NAIVE} {QUOTED_NAIVE}",
            "index 1111111..2222222 100644",
            f"--- {QUOTED_NAIVE}\t",
            f"+++ {QUOTED_NAIVE}\t",
            "@@ -1 +1,2 @@",
            " a",
            "+b",
            f"diff --git {QUOTED_QUOTE} {QUOTED_QUOTE}",
            "index 3333333..4444444 100644",
            f"--- {QUOTED_QUOTE}",
            f"+++ {QUOTED_QUOTE}",
            "@@ -1 +1,2 @@",
            " a",
            "+b",
        ]
    )
    name_status = f"M\0{REAL_NAIVE}\0M\0{REAL_QUOTE}\0"

    consistency = mod.paths_consistent(diff, name_status)

    assert consistency.ok, consistency
    assert mod.changed_paths(diff) == {REAL_NAIVE, REAL_QUOTE}
    assert mod.added_lines_by_file(diff) == {REAL_NAIVE: {2}, REAL_QUOTE: {2}}


def test_changed_files_unquotes_every_listing_shape() -> None:
    """The file list is the third path source, and it quotes too.

    With `core.quotePath=false` a non-ASCII path is printed raw, but a path
    containing `"` or a tab is still quoted; with the flag on both are. Every
    per-path rule keys on this list, so neither shape may survive as a literal.
    """

    mod = load_module()
    listing = "\n".join([REAL_NAIVE, QUOTED_QUOTE, '"tab\\tx.rs"'])
    untracked = "\n".join(['"untracked \\"quote\\".md"', "untracked naïve.md"])

    def fake_run_git(args, cwd=mod.ROOT, *, check=True):
        if args[:1] == ["ls-files"]:
            return untracked
        if args[:2] == ["diff", "--name-only"]:
            return listing
        return ""

    original = mod.run_git
    mod.run_git = fake_run_git
    try:
        files = mod.changed_files("origin/main", "HEAD")
        worktree_files = mod.changed_files("origin/main", "HEAD", worktree=True)
        newly_untracked = mod.untracked_files()
    finally:
        mod.run_git = original

    assert files == [REAL_NAIVE, REAL_QUOTE, "tab\tx.rs"]
    assert newly_untracked == ['untracked "quote".md', "untracked naïve.md"]
    assert worktree_files == sorted({*files, *newly_untracked})


def test_main_path_matching_accepts_git_quoted_headers() -> None:
    """main()'s path matching must not depend on `core.quotePath`.

    End to end over the real reporting path: the file list and the diff headers
    carry git's quoted form while `--name-status -z` lists the raw path, and no
    `diff-path-mismatch` block may appear. Before the fix the quoted literal
    became the claimed path, so the gate blocked an honest diff.
    """

    import json as json_module
    import tempfile

    mod = load_module()
    naive_path = "docs/na\u00efve \u65e5\u672c\u8a9e note.md"
    quote_path = 'docs/q"uote note.md'
    quoted = [
        '"docs/na\\303\\257ve \\346\\227\\245\\346\\234\\254\\350\\252\\236 note.md"',
        '"docs/q\\"uote note.md"',
    ]
    diff = "\n".join(
        [
            f"diff --git {quoted[0]} {quoted[0]}",
            "index 1111111..2222222 100644",
            f"--- {quoted[0]}\t",
            f"+++ {quoted[0]}\t",
            "@@ -1 +1,2 @@",
            " a",
            "+b",
            f"diff --git {quoted[1]} {quoted[1]}",
            "index 3333333..4444444 100644",
            f"--- {quoted[1]}",
            f"+++ {quoted[1]}",
            "@@ -1 +1,2 @@",
            " a",
            "+b",
        ]
    )
    name_status = f"M\0{naive_path}\0M\0{quote_path}\0"

    def fake_run_git(args, cwd=mod.ROOT, *, check=True):
        if args[:2] == ["diff", "--name-only"]:
            # What the gate sees if `core.quotePath` were on: the listing quotes
            # non-ASCII too, and `--name-status -z` stays raw.
            return f"{quoted[0]}\n{quoted[1]}\n"
        if args[:2] == ["diff", "--name-status"]:
            return name_status
        if args[:2] == ["diff", "--unified=3"]:
            return diff
        return ""

    original = mod.run_git
    mod.run_git = fake_run_git
    try:
        with tempfile.TemporaryDirectory() as directory:
            report = pathlib.Path(directory) / "report.json"
            code = mod.main(
                [
                    "--base",
                    "origin/main",
                    "--head",
                    "HEAD",
                    "--dry-run",
                    "--no-dotenv",
                    "--llm-skipped-reason",
                    "local deterministic review",
                    "--output-json",
                    str(report),
                ]
            )
            payload = json_module.loads(report.read_text(encoding="utf-8"))
    finally:
        mod.run_git = original

    ids = [finding["id"] for finding in payload["findings"]]
    assert "diff-path-mismatch" not in ids, payload["findings"]
    assert payload["changed_files"] == [naive_path, quote_path]
    assert payload["verdict"] == "pass", payload["block_findings"]
    assert code == 0


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
        test_rust_inline_test_blocks_ignores_braces_in_literals,
        test_rust_inline_test_blocks_handles_literal_edge_cases,
        test_rust_inline_test_blocks_ignores_mod_declaration,
        test_inline_test_module_findings_flags_grown_block,
        test_inline_test_module_findings_ignores_a_shrinking_block,
        test_inline_test_module_findings_still_flags_real_growth,
        test_inline_test_module_findings_flags_a_new_block_during_extraction,
        test_is_rust_doc_fence_accepts_only_rust_attributes,
        test_vacuous_doc_example_findings_skips_non_rust_fences,
        test_parse_cargo_tenferro_dependencies_reads_target_tables,
        test_pub_item_matches_a_public_union,
        test_dependency_diagram_findings_rejects_a_crate_without_a_manifest,
        test_prompt_does_not_promise_pr_text_disclosure,
        test_inline_test_module_findings_exempts_tiny_leaf_module,
        test_inline_test_module_findings_skips_untouched_block,
        test_missing_doc_example_findings_flags_only_real_gaps,
        test_missing_doc_example_findings_ignores_unchanged_items,
        test_missing_doc_example_findings_flags_added_public_trait_method,
        test_missing_doc_example_findings_accepts_trait_method_examples,
        test_missing_doc_example_findings_skips_hidden_trait_methods,
        test_missing_doc_example_findings_skips_private_traits,
        test_missing_doc_example_findings_recognizes_trait_method_forms,
        test_missing_doc_example_findings_ignores_unchanged_trait_methods,
        test_missing_doc_example_findings_uses_trait_name_for_selective_reexports,
        test_missing_doc_example_findings_ignores_nested_trait_helpers,
        test_missing_doc_example_findings_ignores_trait_comments_and_literals,
        test_missing_doc_example_findings_ignores_traits_in_outer_comments,
        test_missing_doc_example_findings_tracks_nested_block_comments,
        test_vacuous_doc_example_findings_flags_path_only_example,
        test_vacuous_doc_example_findings_ignores_comment_lines,
        test_vacuous_doc_example_findings_accepts_comment_only_example,
        test_vacuous_doc_example_findings_accepts_real_usage,
        test_vacuous_doc_example_fixture_audits_pre_edit_apple_examples,
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
        test_filter_findings_keeps_file_level_block_for_deletions,
        test_files_with_unanchorable_deletions_reads_the_new_side_path,
        test_files_with_unanchorable_deletions_keeps_a_fully_deleted_file,
        test_files_with_unanchorable_deletions_skips_replacement_edits,
        test_sensitive_diff_ignores_an_expression_continuation,
        test_sensitive_diff_ignores_a_field_access_continuation,
        test_sensitive_diff_ignores_an_identifier_continuation,
        test_is_standalone_secret_value_keeps_credential_shapes,
        test_files_with_unanchorable_deletions_reads_headers_outside_hunks,
        test_files_with_unanchorable_deletions_keeps_a_mixed_hunk_deletion,
        test_sensitive_diff_blocks_a_bare_continuation_value,
        test_sensitive_diff_ignores_an_ordinary_bare_continuation,
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
        test_the_pre_fix_parser_really_was_forgeable,
        test_a_hunk_body_line_cannot_reassign_the_file,
        test_every_addition_stays_with_the_real_file,
        test_added_line_numbers_stay_with_the_real_file,
        test_a_forged_payload_does_not_become_the_deleted_from_file,
        test_a_sensitive_line_that_renders_as_a_header_is_still_inspected,
        test_an_honest_diff_keeps_its_attribution,
        test_dev_null_handling_is_unchanged,
        test_a_real_header_after_a_spent_hunk_is_honoured_again,
        test_omitted_counts_and_the_no_newline_marker,
        test_a_rename_reports_both_paths,
        test_name_status_cross_check_flags_a_path_git_does_not_list,
        test_name_status_cross_check_accepts_an_honest_diff,
        test_name_status_cross_check_understands_renames_and_deletes,
        test_path_consistency_finding_blocks_only_on_a_mismatch,
        test_binary_mode_and_quoted_paths_do_not_look_like_tampering,
        test_unquote_diff_path_decodes_git_octal_bytes,
        test_unquote_diff_path_decodes_single_character_escapes,
        test_unquote_diff_path_leaves_unquoted_paths_alone,
        test_unquote_diff_path_keeps_a_truncated_sequence_unmatchable,
        test_quoted_diff_paths_match_unquoted_name_status,
        test_changed_files_unquotes_every_listing_shape,
        test_main_path_matching_accepts_git_quoted_headers,
    ]:
        test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
