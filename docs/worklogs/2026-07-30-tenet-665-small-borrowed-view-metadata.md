# Small Borrowed-View Metadata For TeNeT #665

## Summary

Dynamic borrowed tensor views now collect shape and stride metadata into
tenferro's existing inline `ShapeVec` and `StrideVec` storage. This removes the
small-rank heap allocations observed while TeNeT rebuilds warmed grouped-GEMM
views without changing validation or the rank-generic API.

## Context Read

- `AGENTS.md` and `REPOSITORY_RULES.md`
- shared tensor4all repository, Rust, performance, numerical, documentation,
  and test rules
- `tenferro-tensor/src/types.rs` borrowed-view constructors
- `tenferro-tensor/src/backend.rs` grouped-GEMM validation
- `tenferro-cpu/src/dot_runtime.rs` grouped-GEMM dispatch
- TeNeT's `tenferro_adapter.rs` and prepared fusion replay allocation probe

## Decisions

- Reuse the existing rank-eight inline metadata types instead of adding a new
  view type, public prepared API, cache, or provider contract.
- Preserve the dynamic-rank constructors and all layout validation. Ranks above
  inline capacity continue to spill to the heap.
- Leave grouped-GEMM output-range validation unchanged. On current upstream it
  executes on a backend worker, while TeNeT #665 measures caller-thread
  allocations.

## Rejected Or Deferred

- A grouped-GEMM validation workspace was rejected because process-wide and
  backend-worker allocation removal is outside TeNeT #665.
- TeNeT-only stack metadata was insufficient on its own because restoring
  constructor-local `.to_vec()` calls would silently reintroduce allocations.

## Verification

- `cargo test --offline -p tenferro-tensor small_dynamic_borrowed_view_metadata_stays_inline`
- `cargo test --offline -p tenferro-tensor inline_metadata_collection_keeps_small_shapes_and_strides_inline`
- `cargo test --offline -p tenferro-cpu grouped_gemm_rejects_overlapping_output_ranges`
- `cargo clippy --offline -p tenferro-tensor -p tenferro-cpu --all-targets -- -D warnings`
- TeNeT rank-four prepared replay: zero caller-thread allocations and
  reallocations for U(1), fZ2, SU(2), and U(1) x fZ2 across compose, source
  transform, and output-transform workloads.

## Remaining Risks

- Rank greater than eight intentionally retains heap-spill behavior.
- Backend-worker grouped-GEMM validation allocation remains outside this
  caller-thread acceptance boundary.
