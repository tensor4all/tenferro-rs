# Tensor-Level Linalg Backend API

> Historical proposal: this document captures the design discussion that led to
> the eventual `tenferro-linalg-prims` split. It is not the source of truth for
> the current workspace layout. See `linalg-prims.md`, `linalg.md`, and
> `architecture.md` for the current contracts.

## Summary

This document proposes a new tensor-level backend layer for `tenferro-linalg`.

The goal is to make decomposition and solve operations device-aware without
overloading the primitive-family layer with responsibilities it is not
designed to carry.

This is the design follow-up for issue #246.

## Problem

The current stack has a clean primitive layer but an awkward linalg boundary.

- the primitive-family layer is designed for primitive tensor ops:
  - descriptor -> plan -> execute
  - one output tensor
  - `output = alpha * op(inputs) + beta * output`
  - homogeneous `Tensor<Alg::Scalar>` inputs and outputs
- `tenferro-linalg::backend::LinalgBackend` is designed for CPU slice kernels:
  - `&[T]` / `&mut [T]` I/O
  - multiple outputs for factorization ops
  - mixed dtypes for values vs vectors
  - pivot metadata for LU

This is why `tenferro-linalg` still contains CPU-slice extraction and local
helper paths in places where a future GPU path should operate directly on
`Tensor<T>`.

## Design Principle

Keep the layers coherent.

- `tenferro-prims` remains the primitive tensor execution layer.
- `tenferro-linalg` gets a separate tensor-level solver/factorization layer.

The key point is that decomposition and solve ops are not just “bigger prims”.
They have different output structure, state, and dispatch needs.

## What Stays in `tenferro-prims`

The following remain on the primitive side:

- `BatchedGemm`
- `Reduce`
- `MakeContiguous`
- elementwise ops

These map naturally to the current primitive-family contracts and can be shared by
CPU and future GPU backends without changing the trait model.

## What Moves to the Linalg Backend Layer

The following should be modeled as tensor-level linalg backend operations:

- `solve`
- `solve_triangular`
- `qr`
- `thin_svd`
- `lu_factor`
- `cholesky`
- `eigen_sym`
- `eig`

`lstsq` can be implemented either as a first-class backend op or as a composed
operation built from lower-level linalg kernels. The initial design should not
force it to be first-class.

## Proposed API Shape

The backend surface should be operation-specific, not a single generic
`execute_linalg(...)` API.

The recommended trait name is `TensorLinalgBackend`.

- `tenferro-prims` is already the established primitive-layer crate.
- `LinalgBackend` is already the existing slice-based backend trait.
- `TensorLinalgBackend` makes the intended boundary explicit: this is the
  tensor-level backend surface for linalg operations.

An illustrative trait shape is:

```rust
pub trait TensorLinalgBackend<T: LinalgScalar> {
    type Context;

    fn solve(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<Tensor<T>>;

    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>>;

    fn qr(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
    ) -> Result<QrTensorResult<T>>;

    fn thin_svd(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
    ) -> Result<SvdTensorResult<T>>;

    fn lu_factor(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
    ) -> Result<LuTensorResult<T>>;

    fn cholesky(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
    ) -> Result<Tensor<T>>;

    fn eigen_sym(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
    ) -> Result<EigenTensorResult<T>>;

    fn eig(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
    ) -> Result<EigTensorResult<T>>;
}
```

This shape is intentionally explicit:

- each op has its own method
- each op can have its own flags and future extensions
- results are strongly typed instead of being packed into a heterogeneous
  “output list”

## Result Types

Multiple outputs and mixed dtypes should be modeled by dedicated result types.

Illustrative examples:

```rust
pub struct QrTensorResult<T: LinalgScalar> {
    pub q: Tensor<T>,
    pub r: Tensor<T>,
}

pub struct SvdTensorResult<T: LinalgScalar> {
    pub u: Tensor<T>,
    pub s: Tensor<T::Real>,
    pub vt: Tensor<T>,
}

pub struct LuTensorResult<T: LinalgScalar> {
    pub l: Tensor<T>,
    pub u: Tensor<T>,
    pub pivots: Vec<i32>,
}

pub struct EigenTensorResult<T: LinalgScalar> {
    pub values: Tensor<T::Real>,
    pub vectors: Tensor<T>,
}
```

For `eig`, the design needs one additional dtype mapping beyond `Real`.

The recommended direction is to extend `LinalgScalar` with a canonical complex
associated type:

```rust
pub trait LinalgScalar {
    type Real: LinalgScalar<Real = Self::Real>;
    type Complex: LinalgScalar<Real = Self::Real, Complex = Self::Complex>;
}
```

with the intended mapping:

- `f64 -> Complex64`
- `f32 -> Complex32`
- `Complex64 -> Complex64`
- `Complex32 -> Complex32`

With that, `eig` can use a stable result type:

```rust
pub struct EigTensorResult<T: LinalgScalar> {
    pub values: Tensor<T::Complex>,
    pub vectors: Tensor<T::Complex>,
}
```

This is the preferred API because it removes backend-specific packed layouts
from the public tensor-level linalg boundary.

### Why this is preferable

- It matches the mathematical contract of general eigendecomposition.
- It avoids leaking “interleaved real/imaginary” storage conventions into the
  new backend interface.
- It keeps the API stable across CPU and GPU backends.
- It mirrors PyTorch’s public behavior more closely: real input may still
  produce complex eigenpairs.

The packed representation can still exist internally as a CPU-only private
representation while the public backend API remains canonical.

## Context and Workspace

The linalg backend context should be able to own backend execution resources.

- CPU: reusable workspace, scratch buffers, optional plan caches
- CUDA/ROCm: stream, handle, workspace, algorithm choice caches

This is the main reason to keep a dedicated linalg layer rather than trying to
encode everything as stateless slices or to overload the primitive-family
execution context.

The concrete backend type and the execution context should also stay distinct.

- `CpuTensorLinalgBackend` is the device-oriented backend type that implements
  the trait. The name reflects the device (CPU), not the provider (faer).
- The execution context is `tenferro_prims::CpuContext` — shared with the
  prims layer. There are no linalg-specific context types.

## Pivot and `info` Policy

The first version should keep this simple.

- LU pivots should remain host-side `Vec<i32>` in the first implementation.
- Backend failures that are terminal can continue to use `Result`.
- If a backend exposes LAPACK/cuSOLVER style `info`, it can be surfaced either:
  - as a field on the result type, or
  - as a backend-internal detail that is converted to `Result`

The design should not require a generic “metadata tensor” abstraction before it
is needed by multiple concrete backends.

## Migration Plan

The migration should be incremental.

1. Keep the primitive-family layer unchanged except for GEMM-path cleanup
   handled elsewhere.
2. Fix the public design around `TensorLinalgBackend` and extend `LinalgScalar`
   with `type Complex`.
3. Define the new tensor-level linalg trait and result structs.
4. Implement a CPU adapter that wraps the current `FaerBackend`.
   The adapter uses `tenferro_prims::CpuContext` as execution context.
5. Migrate `solve` and `solve_triangular` first.
6. Migrate `qr`, `lu_factor`, and `lu_solve`.
7. Migrate `thin_svd`, `eigen_sym`, and `eig`.
8. Remove direct CPU-slice extraction assumptions from `tenferro-linalg`.

`solve` and `solve_triangular` are the right first targets because they already
have the simplest shape: `Tensor<T> -> Tensor<T>`.

## Relationship to PyTorch

The current plan is conceptually aligned with PyTorch.

### Where it aligns

PyTorch does not force decomposition and solve ops into a single generic tensor
primitive API. Instead, it uses operation-specific dispatch stubs such as:

- `linalg_eig_stub`
- `geqrf_stub`
- `orgqr_stub`
- `ormqr_stub`
- `lstsq_stub`
- `triangular_solve_stub`
- `lu_factor_stub`
- `lu_solve_stub`
- `svd_stub`

This is the same high-level direction proposed here:

- primitive tensor ops stay separate
- solver/factorization ops get their own API surface
- each op uses a dedicated signature matching its real outputs

PyTorch also keeps “composed” linalg ops possible. For example, `lstsq` on GPU
can be built from lower-level QR-related kernels rather than requiring one giant
monolithic primitive.

### Where it intentionally differs

PyTorch is based on:

- function pointer dispatch stubs
- preallocated out tensors
- a dynamic runtime type system

tenferro-rs is better served by:

- Rust traits instead of global dispatch tables
- typed result structs instead of only out-parameter style APIs
- canonical complex result types for `eig` instead of exposing packed re/im
  buffers at the tensor-level backend boundary
- a smaller initial surface that can be adapted from the current CPU backend

So the plan is not a byte-for-byte copy of PyTorch. It is structurally
consistent with PyTorch’s layering and dispatch model, while using Rust-native
types and ownership patterns.

## Recommendation

The first concrete deliverable for issue #246 should be:

- a decision to use `TensorLinalgBackend` as the trait name
- a decision to extend `LinalgScalar` with `type Complex`
- a trait definition for the tensor-level linalg backend
- result structs for the multi-output operations
- a decision that initial LU pivots remain `Vec<i32>`
- a decision that linalg reuses `tenferro_prims` contexts (no linalg-specific context types)
- an explicit statement of which ops remain on `tenferro-prims`
- a CPU adapter sketch that maps the new API onto the current `FaerBackend`

This is enough to stabilize the abstraction boundary before any GPU
implementation work starts.
