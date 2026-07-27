# 2026-07-27 Strided Delegation For CPU Hot Kernels

## Summary

Updated tenferro-rs to consume `tensor4all/strided-rs@20b2def9bf9554605d12fe13762f8770ba732efb`
and made `tenferro-cpu` thinner where strided-rs now owns reusable prepared
kernel replay:

- CPU elementwise fusion delegates each output replay through
  `strided_kernel::ErasedFusedPlan`.
- CPU sum/product reductions delegate owned and read-view axis reductions
  through `strided_kernel::ErasedReducePlan::compile_axes`.
- CPU additive scatter and fixed-window dynamic slice/update delegate replay
  through `ErasedScatterPlan`, `ErasedDynamicSlicePlan`, and
  `ErasedDynamicUpdateSlicePlan`.

Tenferro still owns public tensor semantics, validation, dtype dispatch,
placement checks, `CpuContext` entry, buffer-pool allocation, and error
translation.

## Context Read

- `AGENTS.md` and `REPOSITORY_RULES.md`
- shared tensor4all common/Rust performance, docs/tests, and numerical rules
- `docs/worklogs/2026-07-27-release-codegen-strided-upstream.md`
- `docs/worklogs/2026-07-27-strided-gather-adoption.md`
- `crates/tenferro-internal-cpu-kernels/src/elementwise.rs`
- `crates/tenferro-cpu/src/indexing.rs`
- `crates/tenferro-cpu/src/reduction.rs`
- strided-rs `strided-kernel/src/erased.rs` and erased reduce/indexed plan tests

## Decisions

- Delegate only where strided-rs has a matching prepared erased plan. This
  removes tenferro-local replay classifiers and tensor-sized indexed loops
  without moving tenferro validation or resource ownership out of `CpuBackend`.
- Keep direct eager/broadcast elementwise helpers on existing strided map/zip
  calls. The change is specifically fused replay, not all elementwise wrapper
  plumbing.
- Execute strided plans with `ExecContext::serial()` inside the existing
  `CpuContext::install` boundary. This preserves the repository rule that
  `CpuBackend` owns Rayon/thread policy and avoids selecting ambient Rayon from
  helper code.
- Keep max/min reductions local for now. Strided `ReduceOp` only covers
  sum/product, and tenferro's max/min NaN policy must not be changed silently.
- Initialize bool destinations before constructing erased mutable descriptors
  where strided validates bool byte values before replay. Non-bool full
  overwrite outputs continue to use uninitialized pooled storage.
- Fix integer sum/product overflow in strided-rs instead of adding a tenferro
  fallback. The upstream bug was that erased integer reductions used normal
  `+`/`*`; strided-rs #158 changed them to wrapping arithmetic for `i32` and
  `i64`.

## Rejected Or Deferred

- No C ABI work in this PR. The strided C ABI remains a separate follow-up.
- No TBLIS dependency or provider change in `tenferro-cpu`.
- No static slice, pad, concatenate, reverse, triangular mask, or max/min
  migration. These either lack a matching strided plan or have tenferro-specific
  semantics that still need an accepted upstream design.
- No performance claim. This PR is an ownership and correctness cleanup; formal
  runtime/build benchmark work remains separate.

## Verification

- RED before implementation:
  - `cargo test -p tenferro-internal-cpu-kernels elementwise_fusion_executes_i32_add_multiply_plan`
  - `cargo test -p tenferro-cpu cpu_hot_kernels_delegate_to_erased_strided_replay`
  - strided-rs `cargo test -p strided-kernel erased_reduce_plan_integer -- --nocapture`
- strided-rs #158:
  - `cargo fmt --check`
  - `CARGO_BUILD_JOBS=64 cargo test -p strided-kernel`
  - `CARGO_BUILD_JOBS=64 cargo test`
  - `CARGO_BUILD_JOBS=64 cargo llvm-cov --workspace --json --output-path /tmp/strided-coverage-158.json`
  - GitHub checks passed after rerunning a non-reproducing coverage allocation-count failure.
- tenferro-rs:
  - `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo check -p tenferro-internal-cpu-kernels -p tenferro-cpu`
  - `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-cpu indexing`
  - `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-cpu test_integer_reduce`
  - `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-internal-cpu-kernels`
  - `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-cpu`
  - `cargo fmt --all --check`
  - `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-cpu'`
  - `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`
  - `python3 -m unittest discover -s scripts/ci/tests -v`
  - `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true python3 scripts/ci/run_profile.py coverage`

## Residual Risk

- Multi-output fused plans are replayed as one strided single-output plan per
  output. This keeps semantics simple but can recompute intermediates.
- The ownership contract now assumes strided-rs keeps wrapping integer
  sum/product semantics for erased reductions. The strided regression tests in
  #158 cover both full and axis reductions.
- Some dedicated tenferro loops remain by design. Future migrations should add
  the general primitive to strided-rs first, then consume it from tenferro.
