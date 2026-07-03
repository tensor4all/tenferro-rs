# Recent Bug/Perf Remediation Batch

Date: 2026-07-03

## Summary

This batch addresses recent bug and audit-performance issues #1258, #1259,
#1260, #1261, #1262, #1263, and #1270 in one branch. The common theme is
removing avoidable repeated work in graph construction, runtime caches,
compiler passes, LAPACK batching, and FFT execution, plus signposting verified
audit false-positive hotspots.

## Context Read

- `AGENTS.md`
- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- Shared tensor4all common, Rust, performance, and docs rules
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- Issue bodies for #1258 through #1263 and #1270

## Decisions

- Use one batch branch because the issues are all already-filed bug or
  repository-rule remediation items, and none require a new public API.
- Keep source-contract tests for audit-style findings where synthetic runtime
  reproducers would be brittle or too broad.
- Replace traced input-map merge clones with a shared merge helper that reuses
  an existing `Arc<HashMap>` when it already matches ordered merge semantics.
- Replace traced metadata scope list rebuilds with `MetadataScopeChain`, a
  parent-linked lifetime holder that materializes a deduplicated slice only at
  support boundaries.
- Store checkpoint `old_inputs` behind `Arc` so checkpoint chain relinking
  reuses the captured map instead of deep-cloning it per merge.
- Remove the duplicate `layout_chain_transpose_folding` flag and bump
  `OptimizerConfig::VERSION` because optimizer fingerprints include the config
  shape.
- Keep einsum cache discriminators as hashes but store exact key data inside
  cache entries and verify equality after lookup before reuse.
- Cache RustFFT plans per scalar type, length, and direction. Keep lane
  execution sequential for now and document the scratch-buffer invariant.
- Use existing LAPACK pooled batch helpers for `eigh_values` and `svd_values`;
  use a reusable pooled batch tensor for `lu_factor`.
- Add local `// INVARIANT:` markers instead of broad audit allowlists.

## Verification

- `cargo fmt --all --check`
- `git diff --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check -p tenferro-runtime -p tenferro-ad`
- `cargo test -p tenferro-runtime --lib`
- `cargo test -p tenferro-ad --lib`
- `cargo test -p tenferro-ad --test ad -- jvp_elementwise_add_y_tangent vjp_matmul grad_matmul_sum`
- `cargo test -p tenferro-einsum --lib`
- `cargo test -p tenferro-linalg --lib`
- `cargo test -p tenferro-fft`
- `cargo check -p tenferro-einsum -p tenferro-linalg -p tenferro-fft`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`

## Skipped Or Blocked Verification

- Full `cargo test --workspace` was attempted before this work log was written,
  but the local environment repeatedly failed while linking doctests or large
  test binaries with `rust-lld` Bus error. A later attempt also filled the
  filesystem through `target/` growth. The worktree `target/` was cleaned and
  verification was narrowed to the touched crates.
- Coverage and full docs were not run locally for this batch because of the
  same local disk/linker constraints.

## Residual Risks

- `MetadataScopeChain` keeps graph-construction cost low, but materializing a
  scope slice at AD support boundaries is still proportional to the chain size.
  This preserves existing support API shape while avoiding per-op scope vector
  rebuilds on ordinary traced graph construction.
- Checkpoint collection still materializes a merged input map for AD execution
  boundaries. The per-merge deep clone from chain relinking is removed.
- The FFT lane loop is still sequential by design. Parallelizing it would need
  disjoint output splitting and per-worker scratch storage.
