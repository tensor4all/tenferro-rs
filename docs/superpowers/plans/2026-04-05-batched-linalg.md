# Batched Linalg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the 5 linalg ops (cholesky, svd, qr, eigh, solve) to support trailing-batch dims and Complex64 dtype, using batch loop over 2D faer kernels.

**Architecture:** A `FaerLinalg` trait abstracts over f64/Complex64. Batch loop helpers slice contiguous trailing-batch data, call the 2D kernel per slice, and reassemble. CpuBackend dispatches by dtype. tensor4all-meta gets the batch convention documented.

**Tech Stack:** faer 0.24 (`llt`, `thin_svd`, `qr`, `self_adjoint_eigen`, `partial_piv_lu`), `faer::complex_native::c64`, `num_complex::Complex64`

**Spec:** `docs/superpowers/specs/2026-04-05-batched-linalg-design.md`

---

## File Structure

```
tenferro-tensor/src/cpu/linalg/
  faer_linalg.rs        Rewrite: FaerLinalg trait, batch helpers, f64 + Complex64 impls
  mod.rs                Unchanged (dispatch by feature)
  lapack_linalg.rs      Unchanged (stub)

tenferro-tensor/src/cpu/backend.rs    Add C64 dispatch for linalg methods
tenferro-tensor/src/tests/cpu_tests.rs    Add batched + complex linalg tests

../tensor4all-meta/docs/design-v2/spec/tensor-semantics.md    Add batch convention section
```

---

## Task 1: Add FaerLinalg trait and refactor existing f64 impls

**Files:**
- Modify: `tenferro-tensor/src/cpu/linalg/faer_linalg.rs`

- [ ] Introduce a `FaerLinalg` trait that abstracts over scalar type:
  ```rust
  pub(crate) trait FaerLinalg: Copy + Clone + Send + Sync + 'static {
      fn cholesky_2d(input: &TypedTensor<Self>) -> TypedTensor<Self>;
      fn svd_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
      fn qr_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
      fn eigh_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
      fn solve_2d(a: &TypedTensor<Self>, b: &TypedTensor<Self>) -> TypedTensor<Self>;
  }
  ```
- [ ] Move existing `cholesky`, `svd`, `qr`, `eigh`, `solve` bodies into `impl FaerLinalg for f64`
- [ ] Replace the 5 public functions with generic dispatchers that call `T::cholesky_2d(input)` etc (still 2D only for now)
- [ ] `cargo test --workspace` — existing linalg tests pass unchanged
- [ ] Commit: `refactor: introduce FaerLinalg trait for f64 linalg`

## Task 2: Add batch loop helpers

**Files:**
- Modify: `tenferro-tensor/src/cpu/linalg/faer_linalg.rs`

- [ ] Add batch loop helper for single-output ops:
  ```rust
  fn batched_single<T: FaerLinalg>(
      input: &TypedTensor<T>,
      core_rank: usize,
      kernel: fn(&TypedTensor<T>) -> TypedTensor<T>,
  ) -> TypedTensor<T> {
      if input.shape.len() <= core_rank {
          return kernel(input);
      }
      let core_shape = &input.shape[..core_rank];
      let batch_shape = &input.shape[core_rank..];
      let slice_size: usize = core_shape.iter().product();
      let batch_total: usize = batch_shape.iter().product();
      let data = input.host_data();
      let first = kernel(&TypedTensor::from_vec(core_shape.to_vec(), data[..slice_size].to_vec()));
      let out_core_shape = first.shape.clone();
      let out_slice_size: usize = out_core_shape.iter().product();
      let mut result_data = Vec::with_capacity(out_slice_size * batch_total);
      result_data.extend_from_slice(first.host_data());
      for b in 1..batch_total {
          let slice = &data[b * slice_size..(b + 1) * slice_size];
          let out = kernel(&TypedTensor::from_vec(core_shape.to_vec(), slice.to_vec()));
          result_data.extend_from_slice(out.host_data());
      }
      let mut out_shape = out_core_shape;
      out_shape.extend_from_slice(batch_shape);
      TypedTensor::from_vec(out_shape, result_data)
  }
  ```
- [ ] Add `batched_multi` for multi-output ops (svd, qr, eigh):
  ```rust
  fn batched_multi<T: FaerLinalg>(
      input: &TypedTensor<T>,
      core_rank: usize,
      kernel: fn(&TypedTensor<T>) -> Vec<TypedTensor<T>>,
  ) -> Vec<TypedTensor<T>> {
      if input.shape.len() <= core_rank {
          return kernel(input);
      }
      let core_shape = &input.shape[..core_rank];
      let batch_shape = &input.shape[core_rank..];
      let slice_size: usize = core_shape.iter().product();
      let batch_total: usize = batch_shape.iter().product();
      let data = input.host_data();
      let first = kernel(&TypedTensor::from_vec(core_shape.to_vec(), data[..slice_size].to_vec()));
      let n_outputs = first.len();
      let mut accumulators: Vec<Vec<T>> = first.iter().map(|t| {
          let mut v = Vec::with_capacity(t.host_data().len() * batch_total);
          v.extend_from_slice(t.host_data());
          v
      }).collect();
      let out_core_shapes: Vec<Vec<usize>> = first.iter().map(|t| t.shape.clone()).collect();
      for b in 1..batch_total {
          let slice = &data[b * slice_size..(b + 1) * slice_size];
          let outs = kernel(&TypedTensor::from_vec(core_shape.to_vec(), slice.to_vec()));
          for (acc, out) in accumulators.iter_mut().zip(outs.iter()) {
              acc.extend_from_slice(out.host_data());
          }
      }
      (0..n_outputs).map(|i| {
          let mut shape = out_core_shapes[i].clone();
          shape.extend_from_slice(batch_shape);
          TypedTensor::from_vec(shape, std::mem::take(&mut accumulators[i]))
      }).collect()
  }
  ```
- [ ] Add `batched_binary` for solve (two inputs with matching batch dims):
  ```rust
  fn batched_binary<T: FaerLinalg>(
      a: &TypedTensor<T>,
      b: &TypedTensor<T>,
      core_rank: usize,
      kernel: fn(&TypedTensor<T>, &TypedTensor<T>) -> TypedTensor<T>,
  ) -> TypedTensor<T> {
      if a.shape.len() <= core_rank {
          return kernel(a, b);
      }
      let a_core = &a.shape[..core_rank];
      let b_core = &b.shape[..core_rank];
      let batch_shape = &a.shape[core_rank..];
      assert_eq!(batch_shape, &b.shape[core_rank..], "solve: batch dims must match");
      let a_slice_size: usize = a_core.iter().product();
      let b_slice_size: usize = b_core.iter().product();
      let batch_total: usize = batch_shape.iter().product();
      let a_data = a.host_data();
      let b_data = b.host_data();
      let first = kernel(
          &TypedTensor::from_vec(a_core.to_vec(), a_data[..a_slice_size].to_vec()),
          &TypedTensor::from_vec(b_core.to_vec(), b_data[..b_slice_size].to_vec()),
      );
      let out_core_shape = first.shape.clone();
      let out_slice_size: usize = out_core_shape.iter().product();
      let mut result_data = Vec::with_capacity(out_slice_size * batch_total);
      result_data.extend_from_slice(first.host_data());
      for bi in 1..batch_total {
          let out = kernel(
              &TypedTensor::from_vec(a_core.to_vec(), a_data[bi*a_slice_size..(bi+1)*a_slice_size].to_vec()),
              &TypedTensor::from_vec(b_core.to_vec(), b_data[bi*b_slice_size..(bi+1)*b_slice_size].to_vec()),
          );
          result_data.extend_from_slice(out.host_data());
      }
      let mut out_shape = out_core_shape;
      out_shape.extend_from_slice(batch_shape);
      TypedTensor::from_vec(out_shape, result_data)
  }
  ```
- [ ] Wire the public functions through batch helpers:
  ```rust
  pub(crate) fn cholesky<T: FaerLinalg>(input: &TypedTensor<T>) -> TypedTensor<T> {
      batched_single(input, 2, T::cholesky_2d)
  }
  pub(crate) fn svd<T: FaerLinalg>(input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
      batched_multi(input, 2, T::svd_2d)
  }
  pub(crate) fn qr<T: FaerLinalg>(input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
      batched_multi(input, 2, T::qr_2d)
  }
  pub(crate) fn eigh<T: FaerLinalg>(input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
      batched_multi(input, 2, T::eigh_2d)
  }
  pub(crate) fn solve<T: FaerLinalg>(a: &TypedTensor<T>, b: &TypedTensor<T>) -> TypedTensor<T> {
      batched_binary(a, b, 2, T::solve_2d)
  }
  ```
- [ ] `cargo test --workspace` — existing 2D tests still pass
- [ ] Commit: `feat: add batch loop helpers for linalg ops`

## Task 3: Implement FaerLinalg for Complex64

**Files:**
- Modify: `tenferro-tensor/src/cpu/linalg/faer_linalg.rs`

- [ ] Add `impl FaerLinalg for Complex64` with all 5 methods. Key differences from f64:
  - Use `faer::complex_native::c64` for faer API. `Complex64` and `c64` have identical memory layout, so use transmute or pointer cast for zero-copy conversion.
  - `MatRef::from_column_major_slice` works with `c64` slices.
  - `llt(Side::Lower)` works for Hermitian PD matrices.
  - `thin_svd()`, `qr()`, `self_adjoint_eigen(Side::Lower)`, `partial_piv_lu()` all support `c64`.
  - For SVD, `V` conjugate-transpose gives `Vt` (Hermitian adjoint).
  - `vec_from_diag` for complex eigenvalues returns `Complex64` values (eigenvalues of Hermitian matrix are real, but faer returns complex).
- [ ] `cargo check -p tenferro-tensor`
- [ ] Commit: `feat: implement FaerLinalg for Complex64`

## Task 4: Update CpuBackend dispatch for Complex64

**Files:**
- Modify: `tenferro-tensor/src/cpu/backend.rs`

- [ ] Add `Tensor::C64` arms to all 5 linalg methods in `impl TensorBackend for CpuBackend`:
  ```rust
  fn cholesky(&mut self, input: &Tensor) -> Tensor {
      match input {
          Tensor::F64(t) => Tensor::F64(linalg::cholesky(t)),
          Tensor::C64(t) => Tensor::C64(linalg::cholesky(t)),
          _ => todo!("cholesky: unsupported dtype"),
      }
  }
  ```
  Same pattern for svd, qr, eigh, solve.
- [ ] `cargo check -p tenferro-tensor`
- [ ] Commit: `feat: CpuBackend linalg dispatch for Complex64`

## Task 5: Add batched and complex linalg tests

**Files:**
- Modify: `tenferro-tensor/src/tests/cpu_tests.rs`

- [ ] Add batched f64 cholesky test:
  ```rust
  #[test]
  fn test_batched_cholesky() {
      // Two 3x3 SPD matrices stacked as [3, 3, 2]
      // Batch 0: [[4,2,0],[2,5,1],[0,1,3]] (col-major)
      // Batch 1: [[9,3,0],[3,5,1],[0,1,2]] (col-major)
      // Verify L @ L^T = A for each batch slice
  }
  ```
- [ ] Add batched f64 SVD test:
  ```rust
  #[test]
  fn test_batched_svd() {
      // Two 4x3 matrices stacked as [4, 3, 2]
      // Verify U @ diag(S) @ Vt ≈ A for each batch
  }
  ```
- [ ] Add batched f64 solve test:
  ```rust
  #[test]
  fn test_batched_solve() {
      // Two 3x3 systems stacked as A=[3,3,2], b=[3,1,2]
      // Verify A @ x ≈ b for each batch
  }
  ```
- [ ] Add Complex64 cholesky test:
  ```rust
  #[test]
  fn test_complex_cholesky() {
      // Hermitian PD 2x2 complex matrix
      // Verify L @ L^H = A
  }
  ```
- [ ] Add Complex64 SVD test:
  ```rust
  #[test]
  fn test_complex_svd() {
      // 3x2 complex matrix, verify reconstruction
  }
  ```
- [ ] `cargo test --workspace`
- [ ] Commit: `test: batched and Complex64 linalg tests`

## Task 6: Update tensor4all-meta batch convention

**Files:**
- Modify: `../tensor4all-meta/docs/design-v2/spec/tensor-semantics.md`

- [ ] Add section "Linalg Batch Convention" to tensor-semantics.md:
  ```markdown
  ## Linalg Batch Convention

  Linalg ops follow trailing-batch convention: core matrix dims are leftmost,
  batch dims are rightmost. Shape `[M, N, B1, B2, ...]` means `B1*B2*...`
  independent M x N matrices. Each batch slice is contiguous in col-major memory.

  This differs from JAX/NumPy/PyTorch which use leading-batch `[B, M, N]`.
  The choice matches tenferro's col-major storage for zero-copy batch slicing.

  | Op | Input shape | Output shape(s) |
  |---|---|---|
  | cholesky | `[N, N, B...]` | `[N, N, B...]` |
  | svd | `[M, N, B...]` | U `[M, K, B...]`, S `[K, B...]`, Vt `[K, N, B...]` |
  | qr | `[M, N, B...]` | Q `[M, K, B...]`, R `[K, N, B...]` |
  | eigh | `[N, N, B...]` | vals `[N, B...]`, vecs `[N, N, B...]` |
  | solve | A `[N, N, B...]`, b `[N, M, B...]` | `[N, M, B...]` |
  ```
- [ ] Commit in tensor4all-meta: `docs: add linalg trailing-batch convention`

## Task 7: Verification

- [ ] `cargo fmt --all --check`
- [ ] `cargo test --workspace`
- [ ] `cargo check -p tenferro-tensor --features cuda` (stubs still compile)
- [ ] Commit if needed
