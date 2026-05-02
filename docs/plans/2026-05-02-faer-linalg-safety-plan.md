# Faer Linalg Safety Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert faer-backed SVD/EIGH/eig decomposition failures from panic paths into `Result` errors and make complex slice reinterpretation invariants release-visible.

**Architecture:** Keep the existing faer linalg module layout. Add result-returning batched helpers only for multi-output operations that can fail (`svd`, `eigh`, `eig`) so successful operations such as `qr`, `lu`, and `full_piv_lu` do not churn. Flatten the new internal `Result` at `CpuBackend` and `CpuExecSession` boundaries.

**Tech Stack:** Rust, `tenferro-tensor`, faer CPU backend, module-local CPU tests.

---

## Scope For This Step

Implement:

- release-visible layout checks in `complex64_to_faer_slice` and `complex64_to_faer_slice_mut`,
- `crate::Result` propagation for faer `svd`, `eigh`, and `eig`,
- focused tests that call faer linalg internals and assert invalid decomposition inputs return `Err`.

Defer:

- broad linalg validation refactors for all `assert!` call paths,
- LAPACK backend panic conversion,
- full batched helper assertion cleanup outside the failing decomposition paths.

## Task 1: Add Failing Tests

**Files:**

- Modify: `tenferro-tensor/src/tests/cpu_tests.rs`

**Steps:**

1. Add `#[cfg(feature = "cpu-faer")]` tests that call `crate::cpu::linalg::faer_linalg::{svd, eigh, eig}` directly with a NaN-containing matrix.
2. Assert each call returns `Err` and the error text contains the operation name.
3. Run the focused tests and verify they fail to compile or fail because the current helpers return `Vec` and panic internally.

## Task 2: Result-Returning Faer Decompositions

**Files:**

- Modify: `tenferro-tensor/src/cpu/linalg/faer_linalg.rs`

**Steps:**

1. Change `FaerLinalg::svd_2d` and `FaerLinalg::eigh_2d` to return `crate::Result<Vec<TypedTensor<Self>>>`.
2. Add `batched_multi_result` and `batched_multi_convert_result` alongside existing helpers.
3. Change public faer `svd` and `eigh` helpers to return `crate::Result<Vec<TypedTensor<T>>>`.
4. Change `eig_real_2d`, `eig_complex_2d`, and faer `eig` to return `crate::Result`.
5. Replace each faer decomposition `unwrap_or_else(|_| panic!(...))` with `map_err` returning `crate::Error::BackendFailure { op, message }`.

## Task 3: Boundary Flattening

**Files:**

- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-tensor/src/cpu/exec_session.rs`

**Steps:**

1. In faer `CpuBackend::svd`, `CpuBackend::eigh`, and `CpuBackend::eig`, flatten `catch_backend_panic(...).and_then(|r| r)`.
2. Add a faer-only `linalg_multi_result!` macro for `CpuExecSession::svd` and `CpuExecSession::eigh`.
3. Flatten faer `CpuExecSession::eig` similarly.
4. Keep cpu-blas branches unchanged.

## Task 4: Complex Layout Checks

**Files:**

- Modify: `tenferro-tensor/src/cpu/linalg/faer_linalg.rs`

**Steps:**

1. Replace `debug_assert_eq!` layout checks before complex slice reinterpretation with always-on `assert_eq!`.
2. Add one short safety comment before the unsafe conversion explaining that size and alignment are checked in release builds.

## Task 5: Verify And Commit

**Commands:**

```bash
cargo test -p tenferro-tensor --lib -- faer
cargo test -p tenferro-tensor --lib -- linalg
cargo fmt --all --check
cargo check -p tenferro-tensor
```

Commit:

```bash
git add docs/plans/2026-05-02-faer-linalg-safety-plan.md tenferro-tensor/src
git commit -m "fix: return faer linalg decomposition errors"
```
