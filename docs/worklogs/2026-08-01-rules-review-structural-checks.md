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
- Issue #1605 is resolved: `missing-doc-examples` now checks diff-added
  methods inside reachable public traits. The check remains diff-scoped, so
  unchanged methods and non-public traits remain outside the deterministic
  audit.

## Review Follow-ups (Codex on PR #1582)

Three P2 findings, each reproduced against the pre-fix script before changing
it:

- **cfg polarity** — `is_cfg_test_attr` deleted a literal `not(test)`
  substring, so `#[cfg(not(any(test, feature = "cuda")))]` still matched the
  bare `test` token and a production-only module read as an inline test
  module (false positive on every change to it). Replaced with
  `cfg_expression_enables_test`, a structural walk of the nested
  `all`/`any`/`not` grammar that tracks polarity; `test` under an odd number
  of `not`s no longer counts as a test gate.
- **compact optional dependencies** — the inline-entry check matched the
  literal string `optional = true`, so the equally valid
  `tenferro-ad={workspace=true,optional=true}` was recorded as a production
  edge and produced a false `dependency-diagram-drift` warning. Both the
  inline and `[dependencies.x]` table paths now use one whitespace-independent
  `OPTIONAL_TRUE` pattern.
- **comments in vacuous examples** — a comment line stayed in the classified
  set, so `all(VACUOUS_EXAMPLE_LINE...)` failed and an assignment-only doc
  example with a comment above it escaped the audit (false negative).
  `VACUOUS_IGNORE_LINE` now also skips `//` lines; an example that is nothing
  but comments still reports nothing, because no classified code remains.

Coverage: `test_is_cfg_test_attr_tracks_nested_not_polarity`,
`test_parse_cargo_tenferro_dependencies_accepts_compact_optional_syntax`,
`test_vacuous_doc_example_findings_ignores_comment_lines`, and
`test_vacuous_doc_example_findings_accepts_comment_only_example`.

## Review Follow-ups, Round 2 (Codex on PR #1582)

Three more P2 findings, each reproduced against the previous revision
(`f23b3c1f`) before changing anything.

- **Diagram entries without manifests** — enumerating every crate manifest on
  a doc-only change only covers `manifest_crates - diagram`. A diagram node
  with no manifest never entered the loop, so an invented crate
  (`tenferro-phantom`) or a long-stale entry passed the audit. The opposite
  direction is now compared, but only where the enumeration makes the crate
  set authoritative (`doc_changed`); with just the PR's own manifests every
  untouched crate would look invented. Scoped to diagram *sources*: a node
  appearing solely as an edge target may legitimately be a prose-documented
  crate or a non-crate box, and judging those needs its own rule. The current
  diagram has no orphan in either direction.
- **Shrinking inline test blocks warned** — growth was inferred from "a line
  inside the block was touched", never from a comparison, so extraction work
  that shrinks an oversized block while editing what remains was reported as
  "added or grown" — penalizing exactly the cleanup the rule asks for. The
  check now reads the BASE revision and compares the file's net inline-test
  line count; blocks cannot be matched one-to-one across revisions (they move,
  split and merge), so net size is the comparison. `base` is threaded from
  `main` through `deterministic_checks`; with no base supplied the previous
  behavior is unchanged.
- **`PR text` promised a classification the model cannot make** — the prompt
  listed the pull-request description as a disclosure source, but
  `review_chunk()` supplies only changed paths, routed rules and the unified
  diff, and nothing passes the PR body to the script. Removed `PR text` from
  the list and stated the payload's contents explicitly, rather than growing
  the payload: the disclosure sources that remain (worklog, design doc, code
  comment) are all in-diff and therefore actually visible.

Coverage: `test_dependency_diagram_findings_rejects_a_crate_without_a_manifest`,
`test_inline_test_module_findings_ignores_a_shrinking_block`,
`test_inline_test_module_findings_still_flags_real_growth`, and
`test_prompt_does_not_promise_pr_text_disclosure`.

## Review Follow-ups, Round 3 (Codex on PR #1582)

Four findings, all verified by running the previous revision (`14e831ff`)
before changing anything. Two could already misfire on the current tree; two
were latent (the repo has no `[target.*]` dependency table and no public
`union` today).

- **Non-Rust fences classified as doctests** — the scanner treated every
  ```` ``` ```` fence in a doc comment as a doctest, so a ```` ```text ````
  grammar block containing `Widget;` produced `vacuous-doc-example` on
  ordinary syntax documentation. `is_rust_doc_fence` now gates the classifier
  on the fence info string: empty, or every token a Rust attribute
  (`rust`, `no_run`, `ignore`, `should_panic`, `compile_fail`, `edition*`).
  Deliberately NOT also restricting the check to the `# Examples` section: a
  vacuous Rust doctest outside `# Examples` is equally vacuous, and gating on
  the section would weaken the audit beyond what the false positive requires.
- **A new inline test block hidden by a simultaneous shrink** — round 2's
  net-size exemption is file-wide, so a PR that shrinks one block while adding
  another can lower the total and skip the fresh violation. The exemption is
  now applied per block: a block whose OPENER is itself an added line is judged
  on its own, while an edit inside a surviving block of a non-growing file
  stays exempt.
- **Target-specific dependency tables ignored** — `[target.'cfg(unix)'.dependencies]`
  is an ordinary production dependency table, but the section name was matched
  exactly, so a real edge read as absent (matching diagram edge reported stale,
  omitted edge passed). `TARGET_TABLE_PREFIX` strips the target spec, including
  the quoted form, before the section is classified;
  `[target.<spec>.dev-dependencies]` still falls through like `[dev-dependencies]`.
- **`pub union` invisible to the doc audit** — `PUB_ITEM` listed
  `fn|struct|enum|trait|type`, so a public union never reached the
  `missing-doc-examples` check. Added `union`.

Coverage: `test_vacuous_doc_example_findings_skips_non_rust_fences`,
`test_is_rust_doc_fence_accepts_only_rust_attributes`,
`test_inline_test_module_findings_flags_a_new_block_during_extraction`,
`test_parse_cargo_tenferro_dependencies_reads_target_tables`, and
`test_pub_item_matches_a_public_union`. The round-2 shrink-exemption test still
passes, pinning that the per-block refinement did not undo it.

## Review Follow-ups, Round 4 (Codex on PR #1582)

Both findings restate limitations already recorded under §Residual Risks. One
is a repair and is fixed here; one is a scope expansion and is now tracked.

- **Fixed — braces inside Rust literals mis-span a block.** `let expected =
  "}";` inside an inline test module ended the detected block at that line
  (measured: a 9-line block reported as 1..6), so a test added below fell
  outside the span and evaded `inline-test-module` entirely — a silent false
  negative, not a waivable warning. `strip_code_comments_and_literals` now
  blanks string, raw-string and char-literal contents as well as comments, with
  carried state for multi-line strings. Lifetimes (`&'a str`) are explicitly
  distinguished from char literals. Only the two brace-COUNTING scanners use
  it; `scan_runtime_boundary_text` keeps `strip_code_comments`, because a
  forbidden symbol inside a string is still worth reporting there. This retires
  the first Residual Risk entry.
- **Deferred — methods added to public traits.** A trait-body `fn spin(&self);`
  has no `pub` token, so `PUB_ITEM` never matches it and
  `missing-doc-examples` skips public API. Confirmed, and it is the
  checker-side counterpart of the second Residual Risk entry. Not folded in:
  it needs public-trait body spans, a method matcher, and an `item_filter`
  decision (a trait method is not a module-level re-export name), and it makes
  the audit materially stricter repo-wide while #1575 shows an existing
  documentation backlog. Filed as #1605 with the proposed approach and a
  before/after sizing step.

Coverage: `test_rust_inline_test_blocks_ignores_braces_in_literals` and
`test_rust_inline_test_blocks_handles_literal_edge_cases` (char literals, raw
strings, escaped quotes, multi-line strings, and the lifetime negative case).
