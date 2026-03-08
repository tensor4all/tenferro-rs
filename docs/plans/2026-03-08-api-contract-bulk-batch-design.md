# API Contract Bulk Batch Design

## Scope

This batch fixes four non-GPU, non-stub contract bugs:

- `#377` `tenferro-dyadtensor` scalar operator overloads panic on mixed reverse tapes
- `#380` `tenferro-prims` contract fallback fails to contract input-only modes
- `#386` `tenferro-prims` batched GEMM accepts unsupported scalar types until execute time
- `#387` `tenferro-prims` exposes `ReduceOp::Max/Min` but rejects them at execute time

Excluded:

- GPU runtime issues: `#397`, `#398`
- `todo!`/runtime-stub surfaces: `#375`, `#378`, `#384`, `#385`, `#388`
- `tenferro-burn` downstream resolution issue `#376` because another contributor already claimed it in the issue thread

## Goals

- Make `tenferro-prims` reject unsupported contracts at plan time rather than after planning or during execution.
- Make `tenferro-prims` contract fallback numerically correct when axes appear only in the inputs.
- Remove panic-on-validation-failure behavior from `AdScalar` and `DynAdScalar` binary operator sugar.
- Keep the change set reviewable as one PR by centering it on "public contract matches implementation."

## Non-Goals

- Implement GPU backends or runtime loading.
- Implement missing `burn`/`mdarray`/`capi` stubs.
- Generalize GEMM to all `ScalarBase` types.
- Redesign dyadtensor AD semantics beyond mixed-tape scalar binary operators.

## Design

### 1. Dyadtensor scalar operators become fallible sugar

Root cause for `#377` is structural: `try_add`/`try_mul` already return `Result`, but `std::ops::{Add, Sub, Mul, Div}` currently unwrap those errors and panic. For mixed reverse tapes, the checked API reports `Error::MixedReverseTape`, while `x + y` and `x * y` abort.

The only panic-free way to keep operator syntax and preserve the same validation result is to make the operator overloads return `Result<...>` instead of raw values:

- `impl Add for AdScalar<T> { type Output = Result<Self>; }`
- `impl Sub/Mul/Div for AdScalar<T>` likewise
- `impl Add/Sub/Mul/Div for DynAdScalar` likewise
- mixed scalar overloads in `dyn_types.rs` also return `Result<DynAdScalar>`

This is a source-breaking API change, but the crate is still `0.1.0`, and it aligns operator syntax with the checked API instead of preserving panic behavior. `Neg` stays infallible.

### 2. Contract fallback sums over every non-output input mode

Root cause for `#380` is the fallback axis mapping in `execute_contract`: axes that appear only in `A` or only in `B` are currently mapped to constant index `0` instead of entering the reduction space.

The fix is to redefine fallback contraction space as:

- every mode present in inputs but absent from `modes_c`

That reduction space is shared across:

- modes in both `A` and `B` but not in `C`
- modes only in `A`
- modes only in `B`

Implementation-wise, the fallback keeps the same generic strided loop structure, but builds one reduction-mode table covering all non-output input modes and maps both operand indices from that table where relevant.

### 3. BatchedGemm rejects unsupported scalar types at plan time

For `#386`, the bug is not that generic `ScalarBase` GEMM is missing; the bug is that the public path allows planning and only fails in execute dispatch.

The fix is to move the scalar-type contract boundary earlier:

- add a helper that checks whether `T` is one of `f32`, `f64`, `Complex32`, `Complex64`
- call it from `CpuBackend::build_plan` for `PrimDescriptor::BatchedGemm`
- keep execute-side dispatch defensive, but it should no longer be reachable from a successful plan

Docs for `BatchedGemm` and crate-level CPU capability text will be updated to state the supported scalar set explicitly.

### 4. ReduceOp::Max/Min become real CPU operations for ordered real scalars

For `#387`, shrinking the enum would be unnecessarily regressive. The CPU backend can support `Max`/`Min` for ordered real scalars without invasive design changes.

Chosen contract:

- implement `ReduceOp::Max` and `ReduceOp::Min` for `f32` and `f64`
- reject unsupported scalar types at plan time with a clear error
- continue to support `ReduceOp::Sum` for any `Scalar`

This keeps `Max/Min` useful for the practical real-valued cases that motivated the issue, while avoiding undefined ordering for complex types.

## Files

Primary files:

- `extension/tenferro-dyadtensor/src/ad_value.rs`
- `extension/tenferro-dyadtensor/src/dyn_types.rs`
- `extension/tenferro-dyadtensor/src/ad_value/tests/mod.rs`
- `extension/tenferro-dyadtensor/src/dyn_types/tests/mod.rs`
- `extension/tenferro-dyadtensor/tests/dyn_ad_scalar_reverse_tests.rs`
- `tenferro-prims/src/cpu.rs`
- `tenferro-prims/src/lib.rs`
- `tenferro-prims/tests/prims_tests.rs`

Possible docs touch points:

- `extension/tenferro-dyadtensor/src/lib.rs`
- `extension/tenferro-dyadtensor/src/dyn_types.rs`
- `tenferro-prims/src/lib.rs`

## Testing Strategy

Targeted tests first:

- dyadtensor unit/integration tests proving `x + y` and `x * y` return `Err(Error::MixedReverseTape { .. })` instead of panicking
- prims regression for contract fallback with input-only modes in `A` and in `B`
- prims regression that `ReduceOp::Max/Min` run correctly for `f32/f64`
- prims regression that unsupported `Max/Min` and unsupported `BatchedGemm` scalar types fail during `plan`, not execute

Then full repo verification required for PR:

- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

## Risks

- `#377` changes operator trait output types, so downstream compile breakage is possible. This is intentional contract alignment and should be called out clearly in the PR.
- `#387` needs explicit plan-time rejection for complex `Max/Min`; otherwise the bug just moves from execute to a later panic or nonsense ordering.
- `#380` touches generic contraction fallback indexing, so nearby fallback cases need regression coverage beyond the single reported reproducer.
