# 2026-07-27 Strided Gather Adoption

## Summary

Updated tenferro-rs to consume the merged strided-rs gather plan API from
`tensor4all/strided-rs@cfe90bd2fba38452f32fcdc9df120bb4dd28d151`.
CPU gather now keeps tenferro-owned validation, dtype dispatch, index
normalization, output allocation, and error translation, but delegates the
bulk indexed-read traversal to `strided_kernel::ErasedGatherPlan`.

## Context Read

- `AGENTS.md` and `REPOSITORY_RULES.md`
- shared tensor4all Rust, performance, numerical, and docs/test rules
- strided-rs PR #154, which merged explicit strided execution policy and the
  erased gather/reduce plan APIs into strided main
- `crates/tenferro-cpu/src/indexing.rs`
- CPU backend/provider source-contract tests for strided ownership and native
  threading policy

## Decisions

- Keep gather as a CPU backend operation, not a tensor-core helper. The backend
  still owns the `BufferPool`, host placement checks, and `CpuContext` entry
  boundary.
- Use `ErasedGatherPlan` with compact column-major descriptors. Current
  tenferro owned tensors are compact column-major, so no extra layout
  canonicalization is introduced in this PR.
- Keep index tensors normalized to `i64` through tenferro's existing
  `try_index_tensor` path, and compile the strided gather plan with
  `KernelDType::I64` indices.
- Initialize bool output before constructing the erased mutable descriptor.
  This is only for byte-validity validation at the erased bool boundary; other
  dtypes still use uninitialized full-overwrite output allocation.
- Restore `CpuExecutionContext::with_native_parallelism` as the single source
  of strided execution policy. Native modules inherit this policy and cannot
  select ambient Rayon or ad hoc policy locally.

## Rejected Or Deferred

- No i32 index zero-copy path yet. It would need a second compile/execute path
  and should be benchmarked only if index normalization is shown to matter.
- No new gather parallelism tuning in tenferro. The execution policy boundary
  is preserved, but the erased plan is invoked with an explicit serial
  `ExecContext` until strided's gather planner owns a parallel strategy.
- Scatter, slice, pad, concatenate, and reverse remain tenferro-owned dedicated
  loops because there is not yet a matching strided primitive for their current
  semantics.

## Verification

- `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-cpu gather`
- `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-cpu native`
- `CARGO_BUILD_JOBS=64 CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p tenferro-cpu --test integration strided_kernel_ownership_requires_backend_execution_resources`

## Residual Risk

This PR changes CPU gather's implementation owner but not its public semantics.
The main residual risk is parity between tenferro's validation and strided's
plan compiler if either side changes independently. The new source-contract
test makes the ownership boundary explicit, while value tests continue to cover
representative gather behavior.
