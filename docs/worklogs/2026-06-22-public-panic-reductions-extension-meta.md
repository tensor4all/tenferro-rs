# 2026-06-22 Public Panic, Reduction, and Extension Metadata Fixes

## Summary

Addressed issues #1139 and #1140 together because both were public-boundary
validation failures:

- CPU reductions now reject zero-length reduced axes consistently for
  `reduce_sum`, `reduce_prod`, `reduce_max`, and `reduce_min`.
- Public panic helpers in tensor element counts, checkpointing, and CPU GEMM
  borrowed-input paths were removed or converted to fallible propagation.
- `ExtensionOp::infer_output_meta` now returns `tenferro_tensor::Result` so
  extension metadata validation can report typed errors instead of empty
  sentinels or panics.
- Linalg extension metadata helpers now validate rank before indexing shapes.
- Follow-up CI coverage failure fixed: rank-1 linalg solve tests now assert the
  traced operation returns a typed `RankMismatch` without panicking, and the
  metadata error `op` names use the public linalg operation names.

## Context Read

- Issue scope: #1139 reduction zero-length axis inconsistency and #1140
  public `.expect()` calls in `n_elements()` / `checkpoint_tensor()`.
- Source audited around CPU reduction owned/read paths, runtime reduction shape
  inference, tensor element-count APIs, checkpoint graph data attachment, CPU
  GEMM borrowed reads, and extension metadata inference.
- Additional mini-agent audit findings reviewed: reduction same-root parity,
  public panic risk in linalg extension metadata helpers, and source-contract
  coverage gaps for GEMM and reductions.

## Decisions

- Kept empty reduction axes as a no-op, but moved validation into shared helpers
  before dtype dispatch so owned/read paths and all four reductions share the
  same boundary behavior.
- Treated zero-length reduced axes as invalid for all reduction kinds, matching
  existing max/min behavior and avoiding identity-value special casing for
  sum/prod on empty domains.
- Changed the canonical `ExtensionOp::infer_output_meta` trait method to return
  `Result` rather than adding a parallel `try_*` API. This keeps malformed
  extension metadata in the normal typed-error path and removes the old empty
  vector sentinel.
- Added repository guidance requiring repeated public-boundary input checks to
  be helperized when sibling surfaces need the same validation.

## Verification

- `cargo fmt --all`
- `cargo fmt --all --check`
- `cargo test -p tenferro-linalg traced_linalg_metadata_helpers_reject_rank_less_than_two_without_panicking -- --nocapture`
- `cargo test -p tenferro-tensor -p tenferro-internal-ops -p tenferro-runtime -p tenferro-ad -p tenferro-cpu -p tenferro-einsum -p tenferro-fft -p tenferro-linalg`
- `cargo test -p tenferro-xla --no-run`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --manifest-path ext/tropical/Cargo.toml --no-run`
- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit --release`

## Remaining Risks

- `n_elements()` still relies on construction-time tensor shape validation for
  the impossible overflow case; source-contract tests now prevent public
  `.expect()` / `.unwrap()` regressions in the touched public paths.
- Full workspace CI may include feature combinations not exercised locally,
  especially GPU/XLA runtime environments.
