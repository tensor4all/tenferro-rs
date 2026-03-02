# Issue 246 Linalg Backend Cutover Design

**Issue:** `#246` Define a tensor-level linalg backend layer for decompositions and solves

**Decision date:** 2026-03-02

## Goal

Replace the current slice-based `tenferro-linalg` execution boundary with a tensor-level backend boundary that:

- matches `tenferro-prims` in structure (`Backend + associated Context`)
- accepts `Tensor<T>` at the backend boundary instead of raw slices
- uses explicit execution contexts only
- removes the old slice-based backend API instead of keeping compatibility shims
- establishes CPU/CUDA/HIP backend types now, with CPU implemented first

## Core design

The canonical backend trait remains a backend-marker trait with an associated context:

```rust
pub trait TensorLinalgBackend<T: LinalgScalar> {
    type Context;

    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;
    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>>;
    fn qr(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<QrTensorResult<T>>;
    fn thin_svd(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<SvdTensorResult<T>>;
    fn lu_factor(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorResult<T>>;
    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>>;
    fn eigen_sym(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigenTensorResult<T>>;
    fn eig(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigTensorResult<T>>;
}
```

This intentionally mirrors `tenferro-prims::TensorPrims`:

- the backend type defines capability
- the context owns execution resources
- callers pass `&mut Context` explicitly

The trait remains op-specific rather than adopting a descriptor/plan API. That is the correct asymmetry versus `TensorPrims`, because linalg ops need multi-output results, mixed dtypes, and pivot metadata.

## Explicit context only

There is no `solve_auto`-style API in the new backend layer.

- backend execution always requires an explicit context
- there is no global mutable execution context
- there is no hidden provider/device dispatch

This is a deliberate divergence from PyTorch's global/thread-local dispatch style. In Rust, explicit context ownership is preferable because it keeps workspace reuse, mutable state, and execution resources visible in the API.

## Backend and context naming

The public backend/context names are device-oriented, not provider-oriented:

- `CpuTensorLinalgBackend`
- `CpuTensorLinalgContext`
- `CudaTensorLinalgBackend`
- `CudaTensorLinalgContext`
- `HipTensorLinalgBackend`
- `HipTensorLinalgContext`

Provider names such as `faer` or `lapack` do not appear in the public tensor-level backend API.

## CPU provider selection

CPU provider selection is compile-time only.

`tenferro-linalg` introduces two mutually exclusive CPU linalg features:

- `linalg-faer`
- `linalg-lapack`

Policy:

- enabling both is a compile error
- enabling neither is a compile error
- exactly one must be enabled

This matches the CPU GEMM feature policy already used by `tenferro-prims`.

The implementation target for this cutover is:

- `linalg-faer`: fully implemented
- `linalg-lapack`: module boundary and feature policy established now, implementation may be stubbed unless the same change also supplies a real LAPACK path

## Public API direction

The crate-root APIs in `tenferro-linalg` move to explicit context as well.

Representative shape:

```rust
pub fn solve<T, C>(ctx: &mut C, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorLinalgContextFor<T>,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve(ctx, a, b)
}
```

The public entrypoints remain the user-facing policy layer:

- `svd` continues to apply `SvdOptions`
- `lu` continues to handle `LuPivot`
- result structs exposed at crate root remain the final user-facing result types

But all actual execution moves behind the tensor-level backend boundary.

## Helper trait for context-to-backend dispatch

To preserve ergonomic `solve(&mut ctx, ...)` calls while keeping `TensorLinalgBackend` in the same structural shape as `TensorPrims`, `tenferro-linalg` adds a small bridge trait:

```rust
pub trait TensorLinalgContextFor<T: LinalgScalar> {
    type Backend: backend::TensorLinalgBackend<T, Context = Self>;
}
```

Each context type binds itself to its backend. This keeps the public APIs generic over context while preserving the backend-marker pattern.

## File layout

The previous monolithic `backend/tensor_backend.rs` is split into focused files:

- `tenferro-linalg/src/backend/mod.rs`
- `tenferro-linalg/src/backend/tensor_api.rs`
- `tenferro-linalg/src/backend/tensor_context.rs`
- `tenferro-linalg/src/backend/tensor_helpers.rs`
- `tenferro-linalg/src/backend/cpu.rs`
- `tenferro-linalg/src/backend/cpu_faer.rs`
- `tenferro-linalg/src/backend/cpu_lapack.rs`
- `tenferro-linalg/src/backend/cuda.rs`
- `tenferro-linalg/src/backend/hip.rs`

Responsibilities:

- `tensor_api.rs`: trait and tensor-level result structs
- `tensor_context.rs`: `TensorLinalgContextFor`
- `tensor_helpers.rs`: shared tensor validation, contiguous packing, output allocation
- `cpu.rs`: CPU exports, feature policy, context shell
- `cpu_faer.rs`: `linalg-faer` implementation
- `cpu_lapack.rs`: `linalg-lapack` implementation or stub boundary
- `cuda.rs` / `hip.rs`: future-facing backend/context stubs

## Removal of the old slice boundary

This cutover removes the old slice-based linalg backend instead of retaining a compatibility layer.

Delete:

- `backend::LinalgBackend<T>`
- `backend::FaerBackend`
- `tenferro-linalg/src/backend/faer_backend.rs`
- slice-oriented helper wrappers that exist only to bridge public APIs to slice backends

After this change, slices may still appear inside provider-specific implementation files as an internal detail, but not in the public backend boundary.

## Scope of the cutover

This is a full API boundary cutover, not a narrow adapter implementation.

It includes:

- backend module redesign
- public API signature changes from `&mut backend` to `&mut ctx`
- reverse-mode and forward-mode AD API signature changes to use contexts
- tests and docs updated to call the new context-first API
- `tenferro-capi` updates wherever it currently constructs or passes the old backend type

## Non-goals

- no global or thread-local auto-dispatch layer
- no runtime CPU provider switching
- no retention of deprecated `Faer*` names for compatibility
- no attempt to force linalg ops into the `TensorPrims` descriptor/plan model

## Known risk

This is intentionally a breaking API change. The advantage is that the linalg execution boundary is corrected once, instead of requiring a second public migration later.
