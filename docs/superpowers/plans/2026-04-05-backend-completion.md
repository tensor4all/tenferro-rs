# Backend Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all remaining `todo!()` methods in `CpuBackend`'s `TensorBackend` impl: analytic ops, Tier-2 elementwise, additional reductions, indexing, and linalg. All CPU kernels must use strided-kernel or faer (no naive loops per AGENTS.md).

**Architecture:** Each category maps to a file in `tenferro-tensor/src/cpu/`. Elementwise and analytic ops use strided-kernel `map_into` / `zip_map2_into`. Reductions use `reduce_axis`. Indexing ops get dedicated implementations. Linalg dispatches to faer (cpu-faer) or LAPACK (cpu-blas). CpuBackend delegates to these module functions.

**Tech Stack:** strided-kernel (`map_into`, `zip_map2_into`, `zip_map3_into`, `reduce_axis`), faer (linalg), num-traits

**Reference:**
- strided-kernel API: `strided-kernel/src/lib.rs` (map_view, reduce_view, ops_view)
- Existing elementwise pattern: `tenferro-tensor/src/cpu/elementwise.rs`
- StableHLO op semantics: `tenferro/src/stablehlo.rs` enum variants
- AGENTS.md CPU kernel rules

---

## File Structure

All changes in `tenferro-tensor/src/cpu/`:

```
tenferro-tensor/src/cpu/
  elementwise.rs     ADD: div, abs, sign, maximum, minimum, compare, select, clamp
  analytic.rs        NEW: exp, log, sin, cos, tanh, sqrt, rsqrt, pow, expm1, log1p
  reduction.rs       ADD: reduce_prod, reduce_max, reduce_min
  indexing.rs        IMPLEMENT: gather, scatter, slice, dynamic_slice, pad, concatenate, reverse
  linalg/
    mod.rs           UPDATE: dispatch to faer or lapack
    faer_linalg.rs   IMPLEMENT: cholesky, svd, qr, eigh, solve via faer
    lapack_linalg.rs IMPLEMENT: cholesky, svd, qr, eigh, solve via lapack (stubs ok for now)
  backend.rs         UPDATE: delegate new methods to modules
  mod.rs             ADD: pub mod analytic
```

---

## Phase 1: Analytic Ops (strided-kernel map_into)

### Task 1: Implement analytic unary ops

**Files:**
- Create: `tenferro-tensor/src/cpu/analytic.rs`
- Modify: `tenferro-tensor/src/cpu/mod.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

All analytic ops are unary element-wise. Pattern:

```rust
use crate::types::{Tensor, TypedTensor, Buffer};
use crate::cpu::{typed_view, typed_array, tensor_from_array};
use strided_kernel::{map_into, StridedArray};

fn typed_exp<T: Copy + num_traits::Float>(input: &TypedTensor<T>) -> TypedTensor<T> {
    let mut out = typed_array(&input.shape, T::zero());
    map_into(&mut out.view_mut(), &typed_view(input), |x| x.exp())
        .expect("exp");
    tensor_from_array(out)
}

macro_rules! dispatch_unary {
    ($input:expr, $f:ident) => {
        match $input {
            Tensor::F32(t) => Tensor::F32($f(t)),
            Tensor::F64(t) => Tensor::F64($f(t)),
            Tensor::C32(t) => Tensor::C32($f(t)),
            Tensor::C64(t) => Tensor::C64($f(t)),
        }
    };
}
```

Implement all 10 analytic ops: exp, log, sin, cos, tanh, sqrt, rsqrt, pow, expm1, log1p.

Note: `pow` is binary (base, exponent). Use `zip_map2_into`.
Note: `rsqrt(x) = 1/sqrt(x)`. Use `map_into` with `|x| x.sqrt().recip()`.
Note: Complex types need `num_complex` methods. `Complex::exp()`, `Complex::ln()`, etc.

- [ ] Create `analytic.rs` with all 10 ops as public functions
- [ ] Add `pub mod analytic;` to `cpu/mod.rs`
- [ ] Update `backend.rs`: delegate `exp`, `log`, etc. to `analytic::exp`, `analytic::log`, etc.
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement analytic ops (exp, log, sin, cos, tanh, sqrt, rsqrt, pow, expm1, log1p)`

---

## Phase 2: Tier-2 Elementwise Ops

### Task 2: Implement div, abs, sign

**Files:**
- Modify: `tenferro-tensor/src/cpu/elementwise.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

```rust
// div: element-wise division
fn typed_div<T: Copy + std::ops::Div<Output = T> + Zero>(
    lhs: &TypedTensor<T>, rhs: &TypedTensor<T>
) -> TypedTensor<T> {
    let mut out = typed_array(&lhs.shape, T::zero());
    zip_map2_into(&mut out.view_mut(), &typed_view(lhs), &typed_view(rhs), |a, b| a / b)
        .expect("div");
    tensor_from_array(out)
}

// abs: element-wise absolute value
// sign: element-wise sign (-1, 0, or 1)
```

For complex types: `abs` returns real magnitude, `sign` returns `z / |z|`.

- [ ] Implement `div`, `abs`, `sign` in `elementwise.rs`
- [ ] Update `backend.rs` delegation
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement div, abs, sign elementwise ops`

### Task 3: Implement maximum, minimum, compare, select, clamp

**Files:**
- Modify: `tenferro-tensor/src/cpu/elementwise.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

```rust
// maximum(a, b): element-wise max
// minimum(a, b): element-wise min
// compare(a, b, dir): element-wise comparison, returns 0.0 or 1.0
// select(pred, on_true, on_false): ternary, use zip_map3_into
// clamp(input, lower, upper): ternary, use zip_map3_into
```

For `compare`: returns `1.0` if condition true, `0.0` otherwise. Real types only (complex comparison is ill-defined for ordering).
For `select`: `pred[i] != 0.0 ? on_true[i] : on_false[i]`
For `clamp`: `max(lower[i], min(upper[i], input[i]))`

- [ ] Implement all 5 ops
- [ ] Update `backend.rs` delegation
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement maximum, minimum, compare, select, clamp`

---

## Phase 3: Additional Reductions

### Task 4: Implement reduce_prod, reduce_max, reduce_min

**Files:**
- Modify: `tenferro-tensor/src/cpu/reduction.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

Pattern using strided-kernel `reduce_axis`:

```rust
pub fn reduce_prod(input: &Tensor, axes: &[usize]) -> Tensor {
    dispatch_tensor!(input, t => typed_reduce_prod(t, axes))
}

fn typed_reduce_prod<T>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T>
where T: Copy + num_traits::One + std::ops::Mul<Output = T> + strided_traits::ScalarBase
{
    // reduce_axis with |a, b| a * b, init = T::one()
    // Apply axes from highest to lowest to avoid index shifts
}
```

Same for reduce_max (init = T::min_value or NEG_INFINITY, op = max) and reduce_min.

- [ ] Implement `reduce_prod`, `reduce_max`, `reduce_min`
- [ ] Update `backend.rs` delegation
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement reduce_prod, reduce_max, reduce_min`

---

## Phase 4: Indexing Ops

### Task 5: Implement slice, reverse, concatenate

**Files:**
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

These are the simpler indexing ops:

```rust
// slice: extract sub-tensor with starts, limits, strides
// reverse: reverse elements along specified axes
// concatenate: join tensors along an axis
```

`slice` config has `starts`, `limits`, `strides` (step). Iterate output elements, compute source indices.
`reverse` flips index along each specified axis: `new_idx[a] = shape[a] - 1 - idx[a]`.
`concatenate` copies each input into the right offset along the concat axis.

- [ ] Implement `slice`, `reverse`, `concatenate`
- [ ] Update `backend.rs` delegation
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement slice, reverse, concatenate`

### Task 6: Implement pad, dynamic_slice

**Files:**
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

`pad`: surround tensor with padding value. Config has edge_padding_low, edge_padding_high, interior_padding.
`dynamic_slice`: slice with runtime start indices (from a tensor).

- [ ] Implement `pad`, `dynamic_slice`
- [ ] Update `backend.rs` delegation
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement pad, dynamic_slice`

### Task 7: Implement gather, scatter (StableHLO semantics)

**Files:**
- Modify: `tenferro-tensor/src/config.rs` (fill in GatherConfig, ScatterConfig fields)
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

GatherConfig needs StableHLO dimension numbers:
```rust
pub struct GatherConfig {
    pub offset_dims: Vec<usize>,
    pub collapsed_slice_dims: Vec<usize>,
    pub start_index_map: Vec<usize>,
    pub index_vector_dim: usize,
    pub slice_sizes: Vec<usize>,
}
```

ScatterConfig similarly:
```rust
pub struct ScatterConfig {
    pub update_window_dims: Vec<usize>,
    pub inserted_window_dims: Vec<usize>,
    pub scatter_dims_to_operand_dims: Vec<usize>,
    pub index_vector_dim: usize,
}
```

Implement according to StableHLO spec semantics. Start with the cases needed by einsum diagonal patterns (simpler subset), then generalize.

- [ ] Update GatherConfig and ScatterConfig with proper fields
- [ ] Implement `gather` and `scatter`
- [ ] Add tests for diagonal extraction/embedding via gather/scatter configs
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement gather, scatter with StableHLO semantics`

---

## Phase 5: Linalg (faer)

### Task 8: Implement linalg ops via faer

**Files:**
- Modify: `tenferro-tensor/src/cpu/linalg/faer_linalg.rs`
- Modify: `tenferro-tensor/src/cpu/linalg/mod.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`

Implement using faer's linalg API:
- `cholesky`: `faer::linalg::cholesky`
- `svd`: `faer::linalg::svd` → returns (U, S, Vt) as 3 tensors
- `qr`: `faer::linalg::qr` → returns (Q, R) as 2 tensors
- `eigh`: `faer::linalg::eigh` → returns (eigenvalues, eigenvectors) as 2 tensors
- `solve`: `faer::linalg::solve` or LU-based solve

Each function takes `TypedTensor<T>` and returns `TypedTensor<T>` or `Vec<TypedTensor<T>>`.
Dispatch on f32/f64 (complex support later).

The `TensorBackend::svd/qr/eigh` methods return `Vec<Tensor>` for multi-output ops.

- [ ] Implement `cholesky_faer`, `svd_faer`, `qr_faer`, `eigh_faer`, `solve_faer` in `faer_linalg.rs`
- [ ] Update `linalg/mod.rs` to dispatch based on feature flag
- [ ] Update `backend.rs` to delegate linalg methods
- [ ] Add tests: cholesky of known SPD matrix, SVD of known matrix
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement linalg ops via faer (cholesky, svd, qr, eigh, solve)`

---

## Phase 6: strided-kernel Migration

### Task 9: Replace remaining naive loops with strided-kernel

**Files:**
- Modify: `tenferro-tensor/src/cpu/elementwise.rs`
- Modify: `tenferro-tensor/src/cpu/structural.rs`
- Modify: `tenferro-tensor/src/cpu/reduction.rs`

Check existing elementwise/structural/reduction code for naive element-by-element loops (flat_to_multi + get/set pattern). Replace with strided-kernel equivalents:

- `typed_add`, `typed_mul` → `zip_map2_into`
- `typed_neg`, `typed_conj` → `map_into`
- `typed_reduce_sum` → `reduce_axis`
- `typed_transpose` → `permute` + `copy_into`
- `typed_broadcast_in_dim` → `broadcast` + `copy_into`
- `typed_extract_diagonal` → `diagonal_view` + `copy_into`

Some may already use strided-kernel (check before changing).

- [ ] Audit each function for naive loops
- [ ] Replace with strided-kernel equivalents
- [ ] `cargo test --workspace` (behavior must be identical)
- [ ] Commit: `refactor: migrate remaining CPU kernels to strided-kernel`

---

## Phase 7: Verification

### Task 10: Full pre-push verification

- [ ] `cargo fmt --all --check`
- [ ] `cargo test --workspace --release`
- [ ] `cargo doc --workspace --no-deps`
- [ ] Verify no `todo!()` remains in `cpu/backend.rs` TensorBackend impl
- [ ] Verify no `todo!()` remains in `cpu/elementwise.rs`, `cpu/analytic.rs`, `cpu/reduction.rs`
- [ ] Commit if needed: `chore: backend completion cleanup`
