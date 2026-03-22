# Linalg Ex Status Contracts Replan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `lu_factor_ex`, `solve_ex`, `inv_ex`, and `cholesky_ex` report truthful per-batch status while preserving successful batch payloads, without breaking the CPU/GPU-generic layering of `tenferro-linalg`.

**Architecture:** The failed attempt showed that Layer 4 cannot safely synthesize truthful `_ex` results from coarse backend ops: once `info` becomes per-batch, payload validity must be coupled to the same per-batch backend execution. The fix is to move status-bearing `_ex` contracts into `tenferro-linalg-prims`, implement them in the CPU tensor backend, and make `tenferro-linalg` a thin public wrapper again. Successful batches must keep valid payloads; failed batches may remain zero-filled, but only for the failing batches.

**Tech Stack:** Rust, `tenferro-linalg-prims` tensor backend traits, shared CPU tensor backend (`tenferro-linalg/src/backend/cpu_tensor_impl.rs` via `#[path]` reuse), `faer` / LAPACK CPU providers, existing `Tensor` view ops (`select`, `narrow`, `broadcast`, `unsqueeze`), TDD with focused unit tests.

---

## Problem Statement

The reverted Task 4 attempt exposed two real blockers:

1. `tenferro-linalg` cannot synthesize truthful per-batch `_ex` status while also preserving successful batch payloads.
   - Returning `info = [0, 2]` with an all-zero `solution` / `l` tensor is a broken contract.
   - Reconstructing only successful batches at Layer 4 would require tensor combine/update substrate that is still CPU-main-memory-only (`Tensor::stack` / `Tensor::cat`).

2. `lu_factor_impl()` is not a valid source of backend-truthful `solve_ex` status.
   - It currently uses an epsilon heuristic on `U`'s diagonal.
   - `_ex` semantics must instead be defined explicitly and implemented at the backend contract layer.

## New Contract Semantics

Define `_ex` status semantically rather than as “whatever the backend happened to return”:

- `lu_factor_ex.info[b] == 0`: batch `b` factorized successfully.
- `lu_factor_ex.info[b] == i > 0`: the first 1-based zero pivot in the LU factorization of batch `b` is at pivot `i`.
- `solve_ex.info[b]` uses the same convention as `lu_factor_ex`.
- `cholesky_ex.info[b] == 0`: batch `b` is positive definite and `l` is valid for that batch.
- `cholesky_ex.info[b] == i > 0`: the leading principal minor `i` of batch `b` is the first one that is not positive definite.

Payload rule:

- successful batches must preserve valid payloads in the returned tensor
- failed batches may be left zero-filled
- `info[b] == 0` must imply that the corresponding batch slice of the payload tensor is valid

This contract is what CUDA should implement later via cuSOLVER `info` as well.

## Important Constraint

Do **not** add ad hoc Layer 4 reconstruction using:

- `extract_slice(...)`
- `backend::slice_bridge::...`
- `Tensor::stack(...)`
- `Tensor::cat(...)`

If a step cannot preserve payload validity without such hacks, stop and move the missing substrate or backend contract downward instead.

### Task 1: Add Public Regression Tests for Mixed-Success `_ex` Semantics

**Files:**
- Modify: `tenferro-linalg/src/tests/batch_a_contracts.rs`

**Step 1: Write the failing tests**

Add these regressions:

```rust
#[test]
fn solve_ex_mixed_batches_preserve_successful_solution_and_report_zero_pivot() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0,
            1.0, 2.0, 2.0, 4.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::from_slice(
        &[3.0_f64, -1.0, 1.0, 1.0],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = solve_ex(&mut ctx, &a, &b).unwrap();
    assert_eq!(result.info, vec![0, 2]);

    let payload = tensor_data(&result.solution);
    assert_eq!(&payload[..2], &[3.0, -1.0]);
    assert_eq!(&payload[2..], &[0.0, 0.0]);
}

#[test]
fn cholesky_ex_mixed_batches_preserve_successful_factor_and_report_failing_minor() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            4.0_f64, 2.0, 2.0, 3.0,
            1.0, 2.0, 2.0, 1.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = cholesky_ex(&mut ctx, &a).unwrap();
    assert_eq!(result.info, vec![0, 2]);

    let payload = tensor_data(&result.l);
    assert_eq!(&payload[..4], &[2.0, 1.0, 0.0, (2.0_f64).sqrt()]);
    assert_eq!(&payload[4..], &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn cholesky_ex_multi_axis_batches_follow_column_major_batch_order() {
    // Shape [2, 2, 2, 2]; assert info order matches trailing batch dims in col-major order.
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg tests::batch_a_contracts::solve_ex_mixed_batches_preserve_successful_solution_and_report_zero_pivot -- --exact
cargo test -p tenferro-linalg tests::batch_a_contracts::cholesky_ex_mixed_batches_preserve_successful_factor_and_report_failing_minor -- --exact
cargo test -p tenferro-linalg tests::batch_a_contracts::cholesky_ex_multi_axis_batches_follow_column_major_batch_order -- --exact
```

Expected: FAIL because current Layer 4 implementations still collapse status and/or zero the entire payload.

**Step 3: Commit**

```bash
git add tenferro-linalg/src/tests/batch_a_contracts.rs
git commit -m "test: pin mixed-batch ex payload semantics"
```

### Task 2: Add Status-Bearing `_ex` Contracts to `tenferro-linalg-prims`

**Files:**
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg-prims/src/backend/tensor_api.rs`
- Modify: `tenferro-linalg-prims/src/backend/cpu.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda.rs`
- Modify: `tenferro-linalg-prims/src/backend/hip.rs`
- Modify: `tenferro-linalg-prims/src/backend/tests/mod.rs`

**Step 1: Write the failing compile-time/runtime tests**

Add backend-level regressions:

```rust
#[test]
fn cpu_backend_solve_ex_preserves_successful_batches_and_reports_zero_pivot() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    // same input as Task 1
    let result =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::solve_ex(
            &mut ctx, &a, &b,
        )
        .unwrap();
    assert_eq!(result.info, vec![0, 2]);
}

#[test]
fn cpu_backend_cholesky_ex_preserves_successful_batches_and_reports_minor() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let result =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::cholesky_ex(
            &mut ctx, &a,
        )
        .unwrap();
    assert_eq!(result.info, vec![0, 2]);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg-prims cpu_backend_solve_ex_preserves_successful_batches_and_reports_zero_pivot -- --exact
cargo test -p tenferro-linalg-prims cpu_backend_cholesky_ex_preserves_successful_batches_and_reports_minor -- --exact
```

Expected: FAIL to compile because `TensorLinalgPrims` does not yet expose `_ex` methods or result types.

**Step 3: Write minimal contract changes**

Add new result structs to `tenferro-linalg-prims/src/lib.rs`:

```rust
#[derive(Clone)]
pub struct LuTensorExResult<T: LinalgScalar> {
    pub l: Tensor<T>,
    pub u: Tensor<T>,
    pub pivots: Vec<i32>,
    pub info: Vec<i32>,
}

#[derive(Clone)]
pub struct SolveTensorExResult<T: LinalgScalar> {
    pub solution: Tensor<T>,
    pub info: Vec<i32>,
}

#[derive(Clone)]
pub struct CholeskyTensorExResult<T: LinalgScalar> {
    pub l: Tensor<T>,
    pub info: Vec<i32>,
}
```

Extend the trait:

```rust
fn lu_factor_ex(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorExResult<T>>;
fn solve_ex(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<SolveTensorExResult<T>>;
fn cholesky_ex(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<CholeskyTensorExResult<T>>;
```

Stub policy:

- `CpuTensorLinalgBackend`: implement
- `CudaTensorLinalgBackend`: return unsupported for now; capability remains false
- `HipTensorLinalgBackend`: return unsupported for now; capability remains false

**Step 4: Run tests to verify they compile and still fail on missing CPU implementation details if needed**

Run:

```bash
cargo test -p tenferro-linalg-prims cpu_backend_solve_ex_preserves_successful_batches_and_reports_zero_pivot -- --exact
```

Expected: either runtime FAIL or provider-path FAIL, but no longer “method not found”.

**Step 5: Commit**

```bash
git add tenferro-linalg-prims/src/lib.rs tenferro-linalg-prims/src/backend/tensor_api.rs tenferro-linalg-prims/src/backend/cpu.rs tenferro-linalg-prims/src/backend/cuda.rs tenferro-linalg-prims/src/backend/hip.rs tenferro-linalg-prims/src/backend/tests/mod.rs
git commit -m "feat: add linalg prims ex status contracts"
```

### Task 3: Implement CPU `lu_factor_ex` and `solve_ex` at Layer 3

**Files:**
- Modify: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg-prims/src/backend/cpu.rs`
- Modify: `tenferro-linalg-prims/src/backend/tests/mod.rs`

**Important note:** `tenferro-linalg-prims` reuses `tenferro-linalg/src/backend/cpu_tensor_impl.rs` via `#[path = "../../../tenferro-linalg/src/backend/cpu_tensor_impl.rs"]`. Editing that file changes the CPU tensor backend for both crates.

**Step 1: Write/extend the failing backend tests**

Add payload assertions, not just `info`:

```rust
let payload = tensor_data(&result.solution);
assert_eq!(&payload[..2], &[3.0, -1.0]);
assert_eq!(&payload[2..], &[0.0, 0.0]);
```

Add a `lu_factor_ex` regression:

```rust
#[test]
fn cpu_backend_lu_factor_ex_reports_zero_pivot_without_epsilon_heuristic() {
    // exact singular matrix only; assert info == vec![2]
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg-prims cpu_backend_lu_factor_ex_reports_zero_pivot_without_epsilon_heuristic -- --exact
cargo test -p tenferro-linalg-prims cpu_backend_solve_ex_preserves_successful_batches_and_reports_zero_pivot -- --exact
```

Expected: FAIL because CPU backend still only has coarse `solve` / heuristic LU status.

**Step 3: Write minimal implementation**

In `cpu_tensor_impl.rs` add tensor-level helpers:

```rust
pub(crate) fn lu_factor_ex<T>(...) -> Result<LuTensorExResult<T>>;
pub(crate) fn solve_ex<T>(...) -> Result<SolveTensorExResult<T>>;
```

Rules:

- loop over batches once
- preserve successful batch payloads in output buffers
- leave only failing batches zero-filled
- use exact zero-pivot semantics, not `real_epsilon()`

For `solve_ex`:

- factor each batch once
- if `info[b] == 0`, solve that batch and write it into `solution`
- if `info[b] > 0`, do not solve that batch; keep that batch zero-filled

Do **not** route `solve_ex` back through Layer 4 `solve_ex`.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg-prims cpu_backend_lu_factor_ex_reports_zero_pivot_without_epsilon_heuristic -- --exact
cargo test -p tenferro-linalg-prims cpu_backend_solve_ex_preserves_successful_batches_and_reports_zero_pivot -- --exact
cargo test -p tenferro-linalg-prims --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/cpu_tensor_impl.rs tenferro-linalg-prims/src/backend/cpu.rs tenferro-linalg-prims/src/backend/tests/mod.rs
git commit -m "feat: implement cpu lu and solve ex contracts"
```

### Task 4: Implement CPU `cholesky_ex` at Layer 3

**Files:**
- Modify: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg-prims/src/backend/cpu.rs`
- Modify: `tenferro-linalg-prims/src/backend/tests/mod.rs`

**Step 1: Write the failing backend tests**

Add:

```rust
#[test]
fn cpu_backend_cholesky_ex_preserves_successful_batches_and_reports_minor() {
    // assert info == vec![0, 2]
    // assert first batch lower factor is preserved
    // assert second batch is zero-filled
}

#[test]
fn cpu_backend_cholesky_ex_multi_axis_batches_follow_column_major_batch_order() {
    // shape [2, 2, 2, 2], assert info order over trailing batch dims
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg-prims cpu_backend_cholesky_ex_preserves_successful_batches_and_reports_minor -- --exact
cargo test -p tenferro-linalg-prims cpu_backend_cholesky_ex_multi_axis_batches_follow_column_major_batch_order -- --exact
```

Expected: FAIL because CPU backend still collapses all failures.

**Step 3: Write minimal implementation**

Add:

```rust
pub(crate) fn cholesky_ex<T>(...) -> Result<CholeskyTensorExResult<T>>;
```

Rules:

- successful batches must keep their factor in `l`
- failing batches must remain zero-filled
- `info[b]` must be the first failing leading principal minor, 1-based

Provider guidance:

- if the provider can surface the exact minor directly in a future lower-layer API, use it
- until then, a batch-local leading-principal-minor probe inside the CPU backend is acceptable
- this probe belongs in Layer 3, not in `tenferro-linalg`

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg-prims cpu_backend_cholesky_ex_preserves_successful_batches_and_reports_minor -- --exact
cargo test -p tenferro-linalg-prims cpu_backend_cholesky_ex_multi_axis_batches_follow_column_major_batch_order -- --exact
cargo test -p tenferro-linalg-prims --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/cpu_tensor_impl.rs tenferro-linalg-prims/src/backend/cpu.rs tenferro-linalg-prims/src/backend/tests/mod.rs
git commit -m "feat: implement cpu cholesky ex contract"
```

### Task 5: Switch `tenferro-linalg` Public `_ex` APIs to Thin Wrappers

**Files:**
- Modify: `tenferro-linalg/src/backend/tensor_api.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `tenferro-linalg/src/result_types/status.rs`
- Modify: `tenferro-linalg/src/tests/batch_a_contracts.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing thin-wrapper/runtime tests**

Add/adjust tests so they assert:

- `solve_ex` and `inv_ex` no longer synthesize `vec![1; bc]`
- `cholesky_ex` no longer synthesizes `vec![1; bc]`
- mixed-success payloads are preserved for successful batches
- source-level runtime tests still forbid `extract_slice(...)` and `backend::slice_bridge::...` in these wrapper sections

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg tests::batch_a_contracts::solve_ex_mixed_batches_preserve_successful_solution_and_report_zero_pivot -- --exact
cargo test -p tenferro-linalg tests::batch_a_contracts::cholesky_ex_mixed_batches_preserve_successful_factor_and_report_failing_minor -- --exact
```

Expected: FAIL until the wrappers delegate to Layer 3 `_ex`.

**Step 3: Write minimal implementation**

Replace Layer 4 synthesis with direct delegation:

```rust
let result = <C::Backend as backend::TensorLinalgBackend<T>>::solve_ex(ctx, a, b)?;
Ok(SolveExResult {
    solution: result.solution,
    info: result.info,
})
```

Similarly for:

- `lu_factor_ex`
- `cholesky_ex`
- `inv_ex` via `solve_ex` on broadcast identity RHS

Update public docs in `result_types/status.rs` to reflect the semantic contract:

- successful batches preserve valid payload slices
- positive `info` is 1-based zero-pivot / failing-minor status

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg tests::batch_a_contracts::solve_ex_mixed_batches_preserve_successful_solution_and_report_zero_pivot -- --exact
cargo test -p tenferro-linalg tests::batch_a_contracts::cholesky_ex_mixed_batches_preserve_successful_factor_and_report_failing_minor -- --exact
cargo test -p tenferro-linalg tests::batch_a_contracts::cholesky_ex_multi_axis_batches_follow_column_major_batch_order -- --exact
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/tensor_api.rs tenferro-linalg/src/primal/decompositions.rs tenferro-linalg/src/primal/linear_systems.rs tenferro-linalg/src/primal/least_squares.rs tenferro-linalg/src/result_types/status.rs tenferro-linalg/src/tests/batch_a_contracts.rs tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "fix: delegate linalg ex APIs to status-bearing backend contracts"
```

### Task 6: Keep CUDA/HIP Capability Reporting Truthful

**Files:**
- Modify: `tenferro-linalg-prims/src/backend/cuda.rs`
- Modify: `tenferro-linalg-prims/src/backend/hip.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/tests/mod.rs`

**Step 1: Write the failing tests**

Add capability/stub regressions:

```rust
#[test]
fn cuda_backend_reports_ex_capabilities_only_when_wired() {
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::SolveEx
        )
    );
    assert!(
        !<super::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            LinalgCapabilityOp::CholeskyEx
        )
    );
}
```

**Step 2: Run tests to verify they fail if capability flags drift**

Run:

```bash
cargo test -p tenferro-linalg-prims cuda_backend_reports_ex_capabilities_only_when_wired -- --exact
```

**Step 3: Write minimal implementation**

- keep `SolveEx`, `CholeskyEx`, and `LuFactorEx` unsupported in CUDA/HIP until native implementations exist
- ensure trait methods return unsupported stubs, not CPU fallback

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg-prims cuda_backend_reports_ex_capabilities_only_when_wired -- --exact
cargo test -p tenferro-linalg-prims --features cuda --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg-prims/src/backend/cuda.rs tenferro-linalg-prims/src/backend/hip.rs tenferro-linalg-prims/src/backend/cuda/tests/mod.rs
git commit -m "test: keep ex capability reporting truthful on gpu stubs"
```

## Final Verification

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-linalg-prims --lib
cargo test -p tenferro-linalg --lib
cargo build -p tenferro-linalg-prims --features cuda
cargo build -p tenferro-linalg --features cuda
```

If the branch is intended for PR, also run the repository-wide checklist from `AGENTS.md` before pushing.

## Expected End State

- `tenferro-linalg` no longer synthesizes coarse `_ex` status
- `_ex` payload validity and per-batch `info` are coupled at Layer 3
- successful batches keep valid payloads even when sibling batches fail
- failed batches are isolated to their own payload slices
- CPU implementation is exact under the explicit semantic contract
- CUDA/HIP remain capability-truthful until native `_ex` kernels exist
