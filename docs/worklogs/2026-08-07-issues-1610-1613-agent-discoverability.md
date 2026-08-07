# Work log: downstream skill and llms.txt discoverability (#1610 + #1613)

## Summary

Implemented the accepted combined batch after #1609: a canonical
`tenferro-compute` downstream-usage skill with Claude/Kimi mirrors and an
OpenCode entry point, five executable named snippet regions, a freshness
checker, and a curated `docs/llms.txt` index published by the Quarto site.

## Context read

- Accepted implementation plans and audit comments on #1610 and #1613.
- Existing skills under `.agents/skills/`, `.claude/skills/`, `.kimi/skills/`,
  and `.opencode/commands/`.
- `scripts/check-doc-snippets.py`, `scripts/check-docs-site.py`,
  `scripts/build_docs_site.sh`, and the tutorial binary manifest/test.
- Existing API-choice, PyTorch/JAX mapping, einsum, parallelism/caching,
  troubleshooting, and CPU-provider documentation.
- The updated `antheducation/tenferro-rs:add-llms-txt` prototype at commit
  `561a6d1346845b1661715f3bfe81bb47f523eb44`, preserving its curated-index
  attribution while retaining the current repository's complete Quarto render
  and sidebar configuration.

## Decisions

- `.agents/skills/tenferro-compute/` is authoritative. Portable Markdown
  files are copied byte-for-byte to Claude and Kimi mirrors; the OpenCode entry
  points at all four canonical references. The OpenAI metadata remains in the
  canonical `agents/` directory because it is adapter-specific.
- The existing snippet synchronizer now discovers only the new canonical skill
  Markdown and checks its Rust fences alongside guides and tutorials. All five
  runnable examples come from named regions in one tutorial binary; no second
  compiler or documentation parser was added.
- `docs/llms.txt` is a tracked, curated 14-entry source. The build copies it to
  the site root, while `check-docs-site.py` validates its Quarto resource,
  descriptions, unique URLs, repository-relative documentation targets, and
  canonical skill target. No Sourcey, crawler, YAML parser, or `llms-full.txt`
  was added.
- The skill teaches current direct crates and extension traits, column-major
  input, backend/runtime reuse, compile-once/run-many, explicit extension
  registration, einsum syntax, scratch-workspace setup, and CPU/BLAS feature
  choices without adding an API layer.

## Verification

- RED: skill mirror test initially failed because the checker and canonical
  skill did not exist; snippet discovery initially failed to include skill
  Markdown.
- `python3 scripts/test-check-agent-skills.py` → passed.
- `python3 scripts/test-check-doc-snippets.py` → passed.
- `python3 scripts/check-agent-skills.py` → passed.
- `python3 scripts/check-doc-snippets.py --check` → passed.
- `cargo check --manifest-path docs/tutorial-code/Cargo.toml
  --no-default-features --features cpu-faer,doc-snippets --bin
  tenferro_compute_skill` → passed.
- `cargo run --manifest-path docs/tutorial-code/Cargo.toml
  --no-default-features --features cpu-faer,doc-snippets --bin
  tenferro_compute_skill` → passed.
- `python3 scripts/test-check-docs-site.py` → passed, including missing-target
  and duplicate-URL regressions.

## Remaining validation

Run the full tutorial-binary release test, docs-site build and comparison,
`python3 scripts/check-docs-site.py`, docs CI profile, repository-rules review,
and hosted CI before merging the combined PR.
