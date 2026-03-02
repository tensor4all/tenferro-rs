# Complex Prims GEMM Dispatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optimized complex GEMM dispatch to `tenferro-prims`, remove the CPU naive GEMM fallback, and route complex `tenferro-linalg` GEMM helpers through prims so issue `#245` is fully closed for real and complex scalars.

**Architecture:** Tighten `tenferro-prims` feature policy so exactly one CPU GEMM backend is enabled, then teach both backend implementations (`faer` and `openblas`) to execute complex GEMM through optimized paths. Once prims can do complex GEMM efficiently, simplify `tenferro-linalg`'s private bridge so both helper entry points always call prims and no longer fall back to `backend.mat_mul`.

**Tech Stack:** Rust, `tenferro-prims`, `tenferro-linalg`, `faer`, `cblas-sys`, `num-complex`, `cargo test`, `cargo fmt`

---

### Task 1: Lock Feature Policy and Remove the Unsupported Backend State

**Files:**
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/Cargo.toml`
- Test: `tenferro-prims/src/lib.rs`

**Step 1: Write the failing compile-time checks**

At the top of `tenferro-prims/src/lib.rs`, add mutually exclusive feature guards:

```rust
#[cfg(all(feature = "gemm-faer", feature = "gemm-openblas"))]
compile_error!("enable exactly one GEMM backend: gemm-faer or gemm-openblas");

#[cfg(all(not(feature = "gemm-faer"), not(feature = "gemm-openblas")))]
compile_error!("enable exactly one GEMM backend: gemm-faer or gemm-openblas");
```

Also update the crate-level GEMM backend docs to remove the "neither feature" fallback description.

**Step 2: Run a compile check to verify the guard matters**

Run: `cargo check -p tenferro-prims --no-default-features`

Expected: FAIL with the new compile-time error because no backend is enabled.

**Step 3: Keep the feature surface consistent**

Leave the existing feature names in `tenferro-prims/Cargo.toml`, but treat them as an exclusive choice in docs and compile guards. Do not add a third fallback feature.

**Step 4: Run a supported compile check**

Run: `cargo check -p tenferro-prims --features gemm-faer`

Expected: PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/lib.rs tenferro-prims/Cargo.toml
git commit -m "build: require exactly one prims gemm backend"
```

### Task 2: Add Complex `faer` GEMM Dispatch

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`
- Test: `tenferro-prims/src/cpu.rs`

**Step 1: Write the failing test**

Add a unit test in `tenferro-prims/src/cpu.rs` that exercises complex `BatchedGemm` under default `gemm-faer`:

```rust
#[test]
fn batched_gemm_complex64_uses_faer_dispatch() {
    use num_complex::Complex64;
    use tenferro_algebra::Standard;
    use tenferro_tensor::{MemoryOrder, Tensor};
    use tenferro_device::LogicalMemorySpace;

    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_vec(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 0.5),
        ],
        &[2, 2],
        &[1, 2],
        0,
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![
            Complex64::new(0.5, -1.0),
            Complex64::new(1.5, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.25),
        ],
        &[2, 2],
        &[1, 2],
        0,
    )
    .unwrap();
    let mut c = Tensor::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };

    let plan = <CpuBackend as TensorPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap();

    <CpuBackend as TensorPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a, &b],
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    let out = c.try_into_data_vec().unwrap();
    assert_eq!(out.len(), 4);
    assert!((out[0] - Complex64::new(5.0, 0.0)).norm() < 1e-12);
}
```

**Step 2: Run the test to verify it fails for the right reason**

Run: `cargo test -p tenferro-prims batched_gemm_complex64_uses_faer_dispatch --lib -q`

Expected: FAIL before the complex fast path is added, or fail to compile if the current `FaerGemm` macro cannot accept complex types.

**Step 3: Implement complex `FaerGemm` support**

In `tenferro-prims/src/cpu.rs`:

- Generalize `impl_faer_gemm!` to compare `beta` against typed `zero` / `one`
- Replace `0.0` / `1.0` literals with values constructed in a type-correct way
- Add:

```rust
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(Complex64);
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(Complex32);
```

- Extend `execute_batched_gemm` runtime dispatch with `Complex64` / `Complex32` branches

**Step 4: Run the tests to verify they pass**

Run: `cargo test -p tenferro-prims batched_gemm_complex64_uses_faer_dispatch --lib -q`

Run: `cargo test -p tenferro-prims --lib -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "feat: add complex faer gemm dispatch"
```

### Task 3: Add Complex OpenBLAS GEMM Helpers and Delete the Naive Fallback

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`
- Test: `tenferro-prims/src/cpu.rs`

**Step 1: Write the failing backend-selection check**

Add a small internal test or assertion path proving the naive fallback symbol is no longer reachable. The simplest check is to prepare the edit, then use ripgrep:

Run: `rg -n "execute_batched_gemm_naive|not\\(feature = \\\"gemm-openblas\\\"\\)" tenferro-prims/src/cpu.rs`

Expected before the edit: matches for the naive fallback helper and the no-backend `gemm_f64` / `gemm_f32` implementations.

**Step 2: Add complex OpenBLAS helpers**

In `tenferro-prims/src/cpu.rs`, add:

- `gemm_c64` under `#[cfg(all(not(feature = "gemm-faer"), feature = "gemm-openblas"))]`
- `gemm_c32` under the same cfg

These should use `cblas_sys::cblas_zgemm` / `cblas_sys::cblas_cgemm` with column-major arguments and reinterpret the `Complex64` / `Complex32` buffers as CBLAS-compatible interleaved complex buffers.

Then extend the non-faer `execute_batched_gemm` dispatch so complex types use `execute_batched_gemm_contiguous(..., gemm_c64)` / `gemm_c32`.

**Step 3: Delete the unsupported fallback state**

Remove:

- `execute_batched_gemm_naive`
- the `gemm_f64` / `gemm_f32` implementations compiled when both backends are disabled
- the final `execute_batched_gemm_naive(...)` tail call

After this change, every `execute_batched_gemm` branch should either:

- dispatch to `FaerGemm`, or
- dispatch to `execute_batched_gemm_contiguous` with a concrete GEMM function

and unsupported feature states should already be stopped by `compile_error!`.

**Step 4: Run verification**

Run: `rg -n "execute_batched_gemm_naive|not\\(feature = \\\"gemm-openblas\\\"\\)" tenferro-prims/src/cpu.rs`

Run: `cargo test -p tenferro-prims -q`

Expected:

- `rg` shows no remaining naive fallback helper
- tests pass

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "refactor: remove naive prims gemm fallback"
```

### Task 4: Generalize the `tenferro-linalg` Prims Bridge for Complex Scalars

**Files:**
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Test: `tenferro-linalg/src/prims_bridge.rs`
- Test: `tenferro-linalg/src/lib.rs`

**Step 1: Write the failing complex helper test**

Extend the existing `RejectingMatMulBackend` tests in `tenferro-linalg/src/lib.rs`:

```rust
#[test]
fn backend_mat_mul_nn_uses_prims_for_complex_scalars() {
    use num_complex::Complex64;

    let mut backend = RejectingComplexMatMulBackend;
    let a = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.0, 2.0),
        Complex64::new(3.0, 0.5),
    ];
    let b = vec![
        Complex64::new(0.5, -1.0),
        Complex64::new(1.5, 0.0),
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.0, 0.25),
    ];

    let c = backend_mat_mul_nn(&mut backend, &a, &b, 2).unwrap();

    assert_eq!(c.len(), 4);
}
```

Use the same strategy as the real test: `mat_mul` should return an error if called, so this test proves the helper does not fall back.

**Step 2: Run the test to verify it fails**

Run: `cargo test -p tenferro-linalg backend_mat_mul_nn_uses_prims_for_complex_scalars --lib -q`

Expected: FAIL because the current complex path still falls back to `backend.mat_mul`.

**Step 3: Replace the temporary real-only bridge**

In `tenferro-linalg/src/prims_bridge.rs`:

- Rename `batched_gemm_real` to a generic `batched_gemm_via_prims<T>`
- Change the bound from `T: LinalgScalar<Real = T> + Float` to `T: LinalgScalar`
- Remove the `TypeId`-based `maybe_batched_gemm_square` helper entirely
- Build `Tensor<T>` values directly for all scalar types

In `tenferro-linalg/src/lib.rs`:

- Make `backend_mat_mul` call the new generic bridge
- Make `backend_mat_mul_nn` call the new generic bridge directly
- Delete the last `backend.mat_mul(...)` fallback in `backend_mat_mul_nn`

**Step 4: Run the tests to verify they pass**

Run: `cargo test -p tenferro-linalg backend_mat_mul_nn_uses_prims_for_complex_scalars --lib -q`

Run: `cargo test -p tenferro-linalg matrix_exp_complex64 -q`

Run: `cargo test -p tenferro-linalg -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tenferro-linalg/src/prims_bridge.rs tenferro-linalg/src/lib.rs
git commit -m "refactor: route complex linalg gemm through prims"
```

### Task 5: Full Verification and Documentation Cleanup

**Files:**
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `docs/plans/2026-03-02-complex-prims-gemm-design.md`

**Step 1: Remove stale docs references**

Make sure all docs that mention the naive fallback are updated, including the crate-level backend-selection section in `tenferro-prims/src/lib.rs`.

**Step 2: Run the full verification suite**

Run: `cargo fmt --all`

Run: `cargo test -p tenferro-prims`

Run: `cargo test -p tenferro-linalg`

Run: `cargo test --workspace`

Expected: all commands succeed

**Step 3: Record the final scope check**

Append a short completion note to `docs/plans/2026-03-02-complex-prims-gemm-design.md` noting that:

- complex optimized dispatch exists for both supported CPU GEMM backends
- naive fallback is removed
- `tenferro-linalg` helper GEMM now always routes through prims

**Step 4: Re-run a narrow sanity check**

Run: `rg -n "backend\\.mat_mul\\(" tenferro-linalg/src/lib.rs`

Run: `rg -n "execute_batched_gemm_naive" tenferro-prims/src/cpu.rs`

Expected:

- no helper-path `backend.mat_mul` remains in `tenferro-linalg`
- no naive GEMM fallback symbol remains in `tenferro-prims`

**Step 5: Commit**

```bash
git add tenferro-prims/src/lib.rs tenferro-prims/src/cpu.rs tenferro-linalg/src/prims_bridge.rs tenferro-linalg/src/lib.rs docs/plans/2026-03-02-complex-prims-gemm-design.md docs/plans/2026-03-02-complex-prims-gemm-impl.md
git commit -m "feat: complete complex prims gemm routing"
```
