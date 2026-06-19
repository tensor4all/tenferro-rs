#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
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
    assert "Performance And Layout Rules" in sections
    assert "Unsafe Code Boundary" in sections


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


def main() -> int:
    for test in [
        test_added_lines_by_file,
        test_filter_findings_drops_unchanged_files,
        test_filter_findings_keeps_added_line,
        test_filter_findings_drops_line_finding_without_added_lines,
        test_filter_findings_drops_file_level_block_finding,
        test_filter_findings_drops_global_llm_finding_when_disallowed,
        test_reconcile_verdict_only_blocks_fail,
        test_select_rule_sections_includes_ad_for_ad_paths,
        test_select_rule_sections_includes_ad_for_tenferro_ad_crate,
        test_select_rule_sections_includes_performance_for_tensor_crates,
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
    ]:
        test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
