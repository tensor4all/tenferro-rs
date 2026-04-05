# Execution Layer Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure tenferro-rs so that tenferro-tensor owns all execution (types + kernels + backends), tenferro-ops is metadata-only, and tenferro is pipeline-only. Validate with tropical algebra tests and GPU stubs.

**Architecture:** Two backend traits (`TensorBackend` for standard algebra on `Tensor`, `SemiringBackend<Alg>` for custom algebra on `TypedTensor<Alg::Scalar>`) replace the current `SemiringCore`. Config types move from tenferro-ops to tenferro-tensor to avoid dependency cycles. Structural ops are algebra-independent free functions. Both standard and custom algebra go through the same compilation pipeline.

**Tech Stack:** Rust, strided-kernel (follow-up), faer, computegraph-rs, tenferro-algebra (Semiring trait)

**Spec:** `docs/superpowers/specs/2026-04-05-execution-layer-separation-design.md`

---

## File Structure

### tenferro-tensor (major restructure)

```
tenferro-tensor/src/
  lib.rs                    module declarations, re-exports, compile_error! gates
  types.rs                  TypedTensor<T> (strides removed), Tensor, Buffer<T>, DType, etc.
  config.rs                 DotGeneralConfig, CompareDir, GatherConfig, etc. (from tenferro-ops)
  backend.rs                TensorBackend trait, SemiringBackend<Alg> trait
  cpu/
    mod.rs                  CpuBackend struct + re-exports
    backend.rs              impl TensorBackend for CpuBackend
    elementwise.rs          typed_add, typed_mul, typed_neg, typed_conj, etc.
    structural.rs           typed_transpose, typed_reshape, typed_broadcast_in_dim,
                            typed_extract_diagonal, typed_embed_diagonal (free functions)
    reduction.rs            typed_reduce_sum (+ stubs for prod/max/min)
    indexing.rs             stubs: gather, scatter, slice, pad, concatenate, reverse
    gemm/
      mod.rs                feature-gate dispatch
      faer_gemm.rs          FaerGemm trait (moved from tenferro)
      blas_gemm.rs          BlasGemm trait (moved from tenferro)
    linalg/
      mod.rs                stubs
      faer_linalg.rs        stubs
      lapack_linalg.rs      stubs
  cuda/
    mod.rs                  CudaBackend stub (TensorBackend + SemiringBackend<Alg>)
  rocm/
    mod.rs                  RocmBackend stub
```

### tenferro-ops (cleanup)

- Delete: `config.rs` (moved to tenferro-tensor)
- Modify: `std_tensor_op.rs` (remove eval), `semiring_op.rs` (SemiringOp\<Alg\>), imports

### tenferro (pipeline only)

- Delete: `backend.rs`, `cpu_backend.rs`, `structural.rs`, `standard.rs`, `gemm/`, `indexing.rs`, `reduction.rs`, `linalg.rs`
- Modify: `exec.rs` (eval_exec_ir uses TensorBackend), `engine.rs`, `traced.rs`, imports

---

## Phase 1: tenferro-tensor Foundation

### Task 1: Move config types to tenferro-tensor

**Files:**
- Create: `tenferro-tensor/src/config.rs`
- Modify: `tenferro-tensor/src/lib.rs`
- Modify: `tenferro-tensor/Cargo.toml`
- Modify: `tenferro-ops/src/lib.rs` and all files importing `crate::config`
- Modify: `tenferro/src/stablehlo.rs`, `tenferro/src/exec.rs`, `tenferro/src/backend.rs`, `tenferro/src/compiler.rs`

- [ ] Copy `tenferro-ops/src/config.rs` contents to `tenferro-tensor/src/config.rs`
- [ ] Add `pub mod config;` and `pub use config::*;` to `tenferro-tensor/src/lib.rs`
- [ ] Remove `computegraph` dep from `tenferro-tensor/Cargo.toml`
- [ ] In `tenferro-ops`: delete `config.rs`, add `pub use tenferro_tensor::config;` in `lib.rs`. Update all `crate::config::*` imports to `tenferro_tensor::*`
- [ ] In `tenferro/src/`: update config imports from `tenferro_ops::config` to `tenferro_tensor`
- [ ] `cargo build --workspace` passes
- [ ] Commit: `refactor: move config types from tenferro-ops to tenferro-tensor`

### Task 2: Remove strides from TypedTensor, delete operand.rs and tensor_data.rs

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Delete: `tenferro-tensor/src/operand.rs`, `tenferro-tensor/src/tensor_data.rs`
- Modify: `tenferro-tensor/src/lib.rs`

- [ ] Remove `strides` and `preferred_compute_device` fields from `TypedTensor<T>`
- [ ] Update `zeros`, `ones`, `from_vec` constructors (remove strides init)
- [ ] Update `linear_offset` to compute col-major offset from shape:
  ```rust
  pub fn linear_offset(&self, indices: &[usize]) -> usize {
      let mut offset = 0usize;
      let mut stride = 1usize;
      for i in 0..self.shape.len() {
          offset += indices[i] * stride;
          stride *= self.shape[i];
      }
      offset
  }
  ```
- [ ] Remove compute methods from `impl Tensor` (keep only `shape()`, `dtype()`)
- [ ] Delete `operand.rs`, `tensor_data.rs`, update `lib.rs`
- [ ] `cargo check -p tenferro-tensor` passes (dependent crates will break, that's ok)
- [ ] Commit: `refactor: remove strides from TypedTensor, delete operand.rs`

### Task 3: Create TensorBackend and SemiringBackend traits

**Files:**
- Create: `tenferro-tensor/src/backend.rs`
- Modify: `tenferro-tensor/Cargo.toml` (add tenferro-algebra dep)
- Modify: `tenferro-tensor/src/lib.rs`

- [ ] Add `tenferro-algebra = { path = "../tenferro-algebra" }` to Cargo.toml
- [ ] Create `backend.rs` with:
  - `pub trait TensorBackend` — all ops on `&Tensor` (full signature from spec)
  - `pub trait SemiringBackend<Alg: Semiring>` — `batched_gemm` required, `add`/`mul`/`reduce_sum` with default impls using `Alg::add`/`Alg::mul`
- [ ] Default impls for `add`/`mul`/`reduce_sum` use simple loops with `Alg::add`/`Alg::mul` (strided-kernel upgrade is follow-up)
- [ ] Add `pub mod backend;` and `pub use backend::*;` to `lib.rs`
- [ ] `cargo check -p tenferro-tensor` passes
- [ ] Commit: `feat: add TensorBackend and SemiringBackend<Alg> traits`

### Task 4: Create cpu/ module with CpuBackend and kernel functions

**Files:**
- Create: `tenferro-tensor/src/cpu/mod.rs`, `backend.rs`, `structural.rs`, `elementwise.rs`, `reduction.rs`, `indexing.rs`
- Create: `tenferro-tensor/src/cpu/gemm/mod.rs`, `faer_gemm.rs`
- Modify: `tenferro-tensor/Cargo.toml` (add faer optional dep)
- Modify: `tenferro-tensor/src/lib.rs`

- [ ] Create `cpu/structural.rs`: move `typed_transpose`, `typed_reshape`, `typed_broadcast_in_dim`, `typed_extract_diagonal`, `typed_embed_diagonal`, `typed_neg` from deleted `types.rs`/`operand.rs` as public free functions on `TypedTensor<T>`. Add `Tensor`-level dispatch wrappers.
- [ ] Create `cpu/elementwise.rs`: move `typed_add`, `typed_mul`, `typed_conj`, `typed_dot_general` from deleted `operand.rs`. Add `Tensor`-level dispatch wrappers.
- [ ] Create `cpu/reduction.rs`: move `typed_reduce_sum` from deleted `operand.rs`. Add stubs for `reduce_prod`, `reduce_max`, `reduce_min`.
- [ ] Create `cpu/indexing.rs`: stubs (`todo!()`) for gather, scatter, slice, etc.
- [ ] Move `tenferro/src/gemm/` to `tenferro-tensor/src/cpu/gemm/`. Update imports.
- [ ] Add `faer = { workspace = true, optional = true }` and feature flags to Cargo.toml:
  ```toml
  [features]
  default = ["cpu-faer"]
  cpu-faer = ["dep:faer"]
  cpu-blas = ["dep:cblas-sys"]
  ```
- [ ] Create `cpu/backend.rs`: `impl TensorBackend for CpuBackend` delegating to elementwise/structural/reduction functions. GEMM dispatch: try faer → fallback to `typed_dot_general`.
- [ ] Create `cpu/mod.rs` with `pub use backend::CpuBackend;`
- [ ] Add `pub mod cpu;` to `lib.rs`
- [ ] `cargo check -p tenferro-tensor` passes
- [ ] Commit: `feat: add cpu/ module with CpuBackend, kernels, GEMM`

---

## Phase 2: tenferro-ops Cleanup

### Task 5: Remove eval from StdTensorOp, change SemiringOp\<T\> to SemiringOp\<Alg\>

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Modify: `tenferro-ops/src/semiring_op.rs`
- Modify: `tenferro-ops/src/semiring_op_kind.rs`
- Modify: `tenferro-ops/src/semiring_ops.rs`
- Modify: `tenferro-ops/src/lib.rs`
- Modify: `tenferro-ops/Cargo.toml` (add tenferro-algebra dep)

- [ ] In `std_tensor_op.rs`: remove `eval` method from `impl GraphOp for StdTensorOp`. Remove `use computegraph::Operand`. (computegraph `GraphOp` no longer has `eval`)
- [ ] In `semiring_op.rs`: change `SemiringOp<T>` to `SemiringOp<Alg: Algebra>`. Change `type Operand = TypedTensor<Alg::Scalar>`. Remove `T: Operand` bound, `eval`, `use computegraph::Operand`. Add `n_inputs` impl delegating to `SemiringOpKind::n_inputs()`.
- [ ] In `semiring_op_kind.rs`: add `pub fn n_inputs(&self) -> usize` method
- [ ] Update `SemiringOps` impl to use `Alg: Algebra` bound
- [ ] Add `tenferro-algebra` dep to Cargo.toml
- [ ] `cargo check -p tenferro-ops` passes
- [ ] Commit: `refactor: remove eval, change SemiringOp<T> to SemiringOp<Alg>`

---

## Phase 3: tenferro Pipeline Update

### Task 6: Update tenferro to use TensorBackend, delete moved code

**Files:**
- Delete: `tenferro/src/backend.rs`, `cpu_backend.rs`, `structural.rs`, `standard.rs`, `gemm/`, `indexing.rs`, `reduction.rs`, `linalg.rs`
- Modify: `tenferro/src/lib.rs`, `exec.rs`, `engine.rs`, `traced.rs`, `stablehlo.rs`, `compiler.rs`
- Modify: `tenferro/Cargo.toml`

- [ ] Delete the 8 files/directories listed above
- [ ] Update `lib.rs`: remove deleted modules
- [ ] In `exec.rs`: change `eval_exec_ir` to take `&mut B` where `B: TensorBackend`. All dispatch goes through `backend.method()`.
- [ ] In `engine.rs`: change `Engine<B: SemiringCore>` to `Engine<B: TensorBackend>`
- [ ] In `traced.rs`: change `SemiringCore` to `TensorBackend`, update `eval` and `eval_all` imports
- [ ] In `stablehlo.rs`, `compiler.rs`: update config imports to `tenferro_tensor::*`
- [ ] In `Cargo.toml`: remove `faer`, `cblas-sys`, `blas-src` deps and `cpu-*` features (now on tenferro-tensor). Ensure `tenferro-tensor` is listed with appropriate features forwarded if needed.
- [ ] `cargo build --workspace` passes
- [ ] `cargo test --workspace` — existing tests pass
- [ ] Commit: `refactor: tenferro uses TensorBackend, execution code moved to tenferro-tensor`

---

## Phase 4: Tropical Algebra End-to-End Test

### Task 7: Tropical algebra test with naive GEMM

**Files:**
- Create: `tenferro/tests/tropical.rs`

- [ ] Define `TropicalAlgebra`, impl `Algebra` and `Semiring` (zero=-inf, one=0, add=max, mul=+)
- [ ] Implement `SemiringBackend<TropicalAlgebra> for CpuBackend` with naive batched GEMM (triple loop using `Alg::add`/`Alg::mul`)
- [ ] Test: tropical 2x2 matrix multiply `C[i,k] = max_j (A[i,j] + B[j,k])`
- [ ] Test: tropical elementwise add (=max), mul (=+), reduce_sum (=max over axis) using default impls
- [ ] `cargo test -p tenferro --test tropical` passes
- [ ] Commit: `test: tropical algebra end-to-end validation`

---

## Phase 5: GPU Stubs

### Task 8: CUDA and ROCm backend stubs

**Files:**
- Create: `tenferro-tensor/src/cuda/mod.rs`
- Create: `tenferro-tensor/src/rocm/mod.rs`
- Modify: `tenferro-tensor/src/lib.rs`

- [ ] Create `cuda/mod.rs`: `CudaBackend` struct with `todo!()` for all `TensorBackend` methods. Blanket `impl<Alg: Semiring> SemiringBackend<Alg> for CudaBackend` with `todo!()` batched_gemm.
- [ ] Create `rocm/mod.rs`: same pattern with `RocmBackend`
- [ ] Add feature-gated modules to `lib.rs`: `#[cfg(feature = "cuda")] pub mod cuda;`
- [ ] Write compile-time assertion test: verify CpuBackend, CudaBackend both impl TensorBackend + SemiringBackend
- [ ] `cargo check -p tenferro-tensor` and `cargo check -p tenferro-tensor --features cuda` both pass
- [ ] `cargo test --workspace` passes
- [ ] Commit: `feat: CUDA/ROCm backend stubs, 4-quadrant design validation`

---

## Phase 6: Final Cleanup

### Task 9: Verification and docs

- [ ] `cargo fmt --all --check` passes (fix if needed)
- [ ] `cargo test --workspace --release` passes
- [ ] `cargo doc --workspace --no-deps` passes
- [ ] Verify AGENTS.md CPU kernel rule is present
- [ ] Commit if any cleanup needed: `chore: final cleanup`
