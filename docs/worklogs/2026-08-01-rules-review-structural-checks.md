# 2026-08-01: Repository-rules-review structural blind-spot fixes

## Session Summary

Implemented issue #1581: closed the structural blind spots in
`scripts/repository-rules-review.py` that a false-negative audit of the
2026-07-18..2026-08-01 PR window identified. The audit re-reviewed 67 PRs
whose bot report was "pass, No findings" and found 16 with missed findings;
most misses traced to routing and scoping defects in the script, not to LLM
judgment.

## Context Read

- `REPOSITORY_RULES.md` (Unit Test Organization, Documentation Policy,
  Invariant Markers, No Ad Hoc Fixes)
- `scripts/repository-rules-review.py`, `scripts/test-repository-rules-review.py`
- `ai/prompts/repository-rules-review.md`
- Audit remediation issues #1574..#1581

## Chosen Design

1. **Deterministic checks over LLM judgment.** New pure `diff -> Finding`
   functions run alongside the existing `sensitive_diff_finding` /
   AD-boundary checks, so they work without `DEEPSEEK_API_KEY` and are unit
   testable:
   - `inline-test-module` (warn): diff-added or -grown inline
     `#[cfg(test)] mod` blocks in non-tests `.rs` files, with a tiny-leaf
     exemption (`file < 150 lines` and `block <= 60 lines`).
   - `missing-doc-examples` (warn): diff-added `pub fn/struct/enum/trait/type`
     whose doc block lacks `# Examples`; skips `#[doc(hidden)]` items and
     items inside inline test modules or non-`pub` inline modules (sealed
     traits). One finding per file.
   - `vacuous-doc-example` (warn): doc examples whose code consists only of
     path/assignment statements (`let _m = Type::method;`), ignoring `use`
     and hidden lines.
   - `ai-report-file` (warn): `*-report.md` or `.superpowers/**` paths
     outside `docs/worklogs/`.
   - `dependency-diagram-drift` (warn): compares `tenferro-*`
     `[dependencies]` edges of changed `crates/*/Cargo.toml` against the
     Dependency Direction diagram in `docs/architecture/tenferro-crates.md`;
     edges to prose-documented targets (`tenferro-core-ops`,
     `tenferro-internal-extension-macros`) are skipped, optional and dev
     dependencies are ignored.
2. **Routing repairs.** `Unit Test Organization` and `Documentation Policy`
   now route on any `.rs$` change (previously the unit-test rule loaded only
   for `/tests/` paths — exactly where inline-test violations cannot occur).
   Paths matching no trigger now route the new `PR Content Hygiene` section
   instead of being invisible.
3. **Rules text.** `REPOSITORY_RULES.md` Doc Examples now carries the
   per-item `# Examples` mandate (previously AGENTS.md-only, hence never in
   the LLM payload) and bans vacuous examples. New `## PR Content Hygiene`
   section covers AI report files, undecided top-level directories, and
   commit-hash pins in guides.
4. **Prompt.** System prompt now instructs the model to report
   worklog-disclosed rule deviations as `warn` findings tagged
   `disclosed-in-worklog:` instead of suppressing them (PROMPT_VERSION 2).
   Motivated by PR #1500's disclosed `ExecContext::serial()` regression that
   PR #1501 immediately had to fix.
5. Removed the dead `MAX_ROUTED_SECTION_LINES` constant.

## Rejected Alternatives

- **Per-method example enforcement inside `pub trait` blocks**: trait-level
  `# Examples` currently satisfies the deterministic check; per-method
  enforcement needs item-nesting analysis with a high false-positive risk, so
  it stays with the LLM rule text.
- **Full git-range regression corpus in CI**: replaying historic PR diffs in
  CI would depend on history availability in shallow clones; synthetic
  fixtures in `scripts/test-repository-rules-review.py` cover the same logic.
- **Co-chunking surface pairs and a final cross-pass** (audit blind spot 4):
  deferred; it changes LLM cost/latency and needs its own design pass. The
  deterministic diagram-sync check covers the highest-value cross-file case.

## Verification

- `python3 scripts/test-repository-rules-review.py` (17 new tests) and
  `python3.11 scripts/test-doc-consistency.py` pass.
- Replay against audited false negatives: PR #1408
  (`a863e053~1..a863e053`) now yields 9 `inline-test-module` and several
  `missing-doc-examples` warns; PR #1428 (`32563562^1..32563562`) yields 4
  `ai-report-file`, 9 `vacuous-doc-example`, and `dependency-diagram-drift`
  for the undocumented fft->cpu and gpu->cpu edges.
- Negative controls: PRs #1517, #1533, #1501 (audit-confirmed justified clean
  passes) replay with zero findings.

## Residual Risks

- Brace matching in the inline-test/private-mod scanners is line-based and
  can mis-span blocks containing unbalanced braces inside string literals;
  all affected checks are warn-severity and waivable.
- The diagram-sync check only runs when a `crates/*/Cargo.toml` or the
  architecture doc changes, so pre-existing drift (issue #1578) surfaces on
  the next touching PR rather than immediately.
- `missing-doc-examples` treats a trait-level example as satisfying the
  mandate for the trait item; per-method gaps inside traits remain LLM-only
  (issue #1575 tracks the current backlog).
