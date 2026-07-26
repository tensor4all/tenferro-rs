# Repository Rules Routing Restructure

## Context

Issue #1491 identified that `REPOSITORY_RULES.md` had become both a human
policy document and the machine-consumed routing source for
`scripts/repository-rules-review.py`. The review script splits only on `##`
headings, so the former `Performance And Layout Rules` section routed roughly
four hundred lines of CPU, GPU, cache, benchmark, and error-contract text for
many small diffs.

## Decisions

- Keep one `REPOSITORY_RULES.md` file, but treat every `##` heading as a
  routing unit.
- Promote the performance/layout subsections to `##` sections so CPU, tensor,
  GPU, linalg, cache, and benchmark diffs can select narrower rule text.
- Mark process-only rules through `HUMAN_ONLY_SECTIONS` rather than sending
  them to the diff-scoped review bot.
- Route `Invariant Markers` globally because other routed rules refer to that
  marker and reviewers need the false-positive policy with the code contracts.
- Replace the retired `register_runtime` extension-registration example with
  the current `Runtime::builder().install_extension_module(...)` model.
- Record the accepted post-U8 policies once: local uniqueness proof for
  `Arc`-reachable in-place writes, and the need-before-implementation gate for
  static-audit-derived performance candidates.

## Verification

- Added a doc-consistency check that every `##` rules section is either routed
  or explicitly human-only, no routed section exceeds the configured line
  limit, stale `register_runtime` wording is absent from `REPOSITORY_RULES.md`,
  and the two post-U8 policies appear once.
- Updated `scripts/test-repository-rules-review.py` for the split section
  names and added coverage for GPU and benchmark routing.
