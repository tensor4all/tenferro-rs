# Strided Materialization API Convergence

## Summary

Issue [#1393](https://github.com/tensor4all/tenferro-rs/issues/1393)
identified a CPU API ownership split: context-free view materialization used a
serial logical traversal, while backend structural operations reached pooled,
parallel `strided-rs` kernels. This change records one durable ownership rule,
updates user guidance to require backend-owned canonicalization, and adds an
allocation-inclusive Criterion benchmark for the converged API.

## Context Read

- Root and repository `AGENTS.md`, workspace `CODING_RULES.md`, and
  `REPOSITORY_RULES.md`.
- The approved strided-materialization design and Task 5 implementation plan.
- Issue #1393 and its root-cause follow-up, including the external benchmark's
  exact scattered 24D shape, permutation, and source strides.
- CPU context, buffer-pool, canonicalization, structural-kernel, test, and
  Criterion benchmark patterns.
- Shared Tensor4all Rust, performance, documentation, test, and benchmark
  rules.

## Historical Reason

`TypedTensorView::to_contiguous` descended from a generic host-view adapter
requiring only `T: Clone`. The later backend canonicalization trait initially
delegated CPU work back to that adapter. `strided-kernel` subsequently gained
parallel structural kernels, but the context-free canonicalization route did
not migrate, leaving equivalent public semantics with different performance
and no access to the backend's execution resources.

The accepted design treats that divergence as an ownership defect. A fast CPU
copy is not only a loop choice: it needs persistent reusable storage,
full-overwrite uninitialized allocation, the configured Rayon pool,
nested-execution safety, and serial/parallel threshold policy. These belong to
`CpuBackend`/`CpuContext`, not tensor layout metadata.

## Overlap Inventory And Decision

| Area | `strided-rs` | tenferro |
|---|---|---|
| Copy/permutation | Affine traversal and kernel parallelism | Metadata validation, placement, allocation, context, and errors |
| Broadcast | Zero-stride views and bulk traversal | Dimension semantics and output ownership |
| Map/zip-map | Strided element traversal | Operation and dtype semantics |
| Axis reduction | Per-axis kernel | Axis validation and multi-axis orchestration |
| Indirect indexing | No matching general primitive today | Gather/scatter and related dedicated kernels |
| Einsum/dot-general | Reusable lower-level primitives where applicable | Benchmark-backed exception for planning, preparation, and providers |

The priority rule is to express an operation as tenferro metadata/semantic
preparation followed by an existing `strided-rs` primitive. A missing generally
useful primitive should be added to `strided-rs` first. New tenferro traversal
exceptions require an accepted issue, comparative benchmarks, and a recorded
rationale.

## Implementation

- Added a source contract requiring the ownership rationale and rejecting the
  removed context-free tensor methods/helper.
- Expanded repository rules with the ownership/overlap table, delegation
  priority, explicit einsum exception, and backend execution-resource contract.
- Updated eager and parallelism guides with backend-owned materialization
  examples and the same-placement/no-hidden-transfer boundary.
- Added `view_materialization`, a Criterion `harness = false` benchmark of
  `CpuBackend::to_contiguous` at one and four threads. It covers compact and
  permuted 3D inputs, the issue's contiguous and scattered 24D layouts, and a
  tiny transpose. Every timed iteration creates the materialized result, and
  each case is checked exactly once before timing against independently
  computed physical offsets.

## TDD And Verification

- RED: `cargo test -p tenferro-cpu --test backend_capability_contracts
  --release strided_kernel_ownership_requires_backend_execution_resources --
  --exact --nocapture` first failed at
  `REPOSITORY_RULES.md must contain ownership contract text: CPU
  affine-strided copy, permutation, broadcast, map, zip-map, and axis
  reduction`.
- After adding the ownership policy, the same command advanced to the
  concurrent Task 4B boundary and failed solely at `context-free tensor
  materialization surface/helper remains: pub fn to_contiguous(&self)`. At
  that point tensor source also still contained
  `TypedTensorViewMut::copy_from_contiguous`, context-free `to_tensor` methods,
  and `materialize_typed_view_col_major`; Task 5 did not edit or revert them.
- The complete `backend_capability_contracts` release target reported 10 passed
  and this one Task 4B-dependent source-contract failure.
- `cargo bench -p tenferro-cpu --bench view_materialization --no-run`: passed.
- `cargo check -p tenferro-cpu --benches`: passed.
- The two scoped Rust files were formatted with `rustfmt --edition 2021`.
  `cargo fmt --all --check` remained blocked solely by concurrent, out-of-scope
  formatting changes in
  `crates/tenferro-runtime/src/graph/executor/tests/preflight.rs`; Task 5 did
  not modify that file.

## Remaining Risks And Follow-Up

- This repository benchmark records representative behavior but does not make
  unstable timing assertions. The external `tenferro-benchmark` permutation
  suite remains the acceptance evidence for Apple M4 scaling and peer gaps.
- Einsum remains an intentional ownership exception; future exceptions must
  meet the repository's issue, benchmark, and rationale gate.

## Task 5 Quality Follow-Up

- Replaced exact prose assertions with the versioned
  `TENFERRO_CPU_STRIDED_OWNERSHIP_CONTRACT` key/value block. The source test
  verifies stable owner/resource fields and only the vocabulary necessary to
  preserve the policy; surrounding explanatory prose may be edited freely.
- Replaced the single-file formatting-sensitive API check with a recursive scan
  of every Rust file under `tenferro-tensor/src`. A small lexer discards line
  comments, nested block comments, string literals, and character literals,
  then identifies `pub fn` signatures independent of whitespace and modifiers.
  It rejects the removed public method names wherever they relocate while
  allowing backend trait methods (which have no inherent `pub fn` signature),
  and rejects the named private serial helpers anywhere in the source tree.
  A fixture guards formatting, comment/string decoys, restricted visibility,
  public modifiers, and private-helper detection. Compile-fail coverage is not
  used because standard integration tests cannot reliably depend on a missing
  method.
- Benchmark cases now retain canonical source shape, source strides, and
  permutation. Exact expected offsets are computed directly from those inputs,
  independently of the already-permuted view metadata used by the operation.
- Pre-timing materialization is confined to `verify_case_once`; its checked
  output is dropped when the helper returns, before Criterion begins timing the
  allocation-inclusive operation.
- Follow-up verification passed for the structured ownership test, the complete
  recursive public-surface guard, its lexer fixture, and
  `cargo bench -p tenferro-cpu --bench view_materialization --no-run`. Scoped
  `rustfmt --check` and `git diff --check` also passed; no broader checks were
  run while Task 4B shared-tree edits were active.
