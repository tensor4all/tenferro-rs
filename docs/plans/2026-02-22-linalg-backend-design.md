# Linalg Backend Architecture Design

**Issue**: #123 — Design freeze: structured primal backend architecture
**Date**: 2026-02-22
**Status**: Approved

## Problem

`tenferro-linalg` hard-wires algorithm dispatch to faer via a private `FaerOps`
trait bound on `LinalgScalar`. This blocks LAPACK and GPU backend integration.

## Decisions

1. **Compile-time feature flags** for backend selection (`faer` default, `lapack` opt-in)
2. **Separate `LinalgBackend<T>` trait** (not extending `TensorPrims`)
3. **Raw slices interface** with pre-allocated output buffers (zero-allocation at trait boundary)
4. **`&mut self`** on backend methods (handle/context pattern, enables workspace reuse)
5. **GPU linalg deferred** to a future issue (trait designed to be extensible)
6. **Backend-explicit API only** — public functions always take `&mut B: LinalgBackend<T>`

## LinalgBackend Trait

```rust
/// Backend trait for linear algebra decompositions.
///
/// Methods take pre-allocated output slices. All matrix data is column-major.
/// `&mut self` allows backends to hold reusable workspace buffers.
pub trait LinalgBackend<T: Copy + 'static> {
    /// Real scalar type (f64 for both f64 and Complex64).
    type Real: Copy + 'static;

    /// Thin SVD: A = U diag(S) V^H.
    /// - a: m*n, u: m*k, s: k, vt: k*n (k = min(m,n))
    fn thin_svd(
        &mut self, a: &[T], m: usize, n: usize,
        u: &mut [T], s: &mut [Self::Real], vt: &mut [T],
    ) -> Result<()>;

    /// QR: A = Q R.
    /// - q: m*k, r: k*n (k = min(m,n))
    fn qr(
        &mut self, a: &[T], m: usize, n: usize,
        q: &mut [T], r: &mut [T],
    ) -> Result<()>;

    /// LU with partial pivoting: P A = L U.
    /// - perm: m, l: m*k, u: k*n (k = min(m,n))
    fn lu(
        &mut self, a: &[T], m: usize, n: usize,
        perm: &mut [usize], l: &mut [T], u: &mut [T],
    ) -> Result<()>;

    /// Cholesky: A = L L^H.
    /// - l: n*n
    fn cholesky(
        &mut self, a: &[T], n: usize,
        l: &mut [T],
    ) -> Result<()>;

    /// Symmetric eigendecomposition: A = V diag(lambda) V^H.
    /// - values: n (ascending), vectors: n*n
    fn eigen_sym(
        &mut self, a: &[T], n: usize,
        values: &mut [Self::Real], vectors: &mut [T],
    ) -> Result<()>;

    /// Matrix multiply: C = A * B (m*k @ k*n -> m*n).
    fn mat_mul(
        &mut self,
        a: &[T], m: usize, k: usize,
        b: &[T], n: usize,
        c: &mut [T],
    ) -> Result<()>;

    /// Solve A x = b. A: n*n, b: n*nrhs, x: n*nrhs.
    fn solve(
        &mut self, a: &[T], b: &[T], n: usize, nrhs: usize,
        x: &mut [T],
    ) -> Result<()>;

    /// Solve triangular system. A: n*n, b: n*nrhs, x: n*nrhs.
    fn solve_triangular(
        &mut self, a: &[T], b: &[T], n: usize, nrhs: usize,
        upper: bool, x: &mut [T],
    ) -> Result<()>;
}
```

## LinalgScalar

Simplified to a marker trait (no associated backend type):

```rust
pub trait LinalgScalar:
    Scalar + Float + Sub<Output = Self> + Debug + 'static
{
}

impl LinalgScalar for f64 {}
impl LinalgScalar for f32 {}
// Future: impl LinalgScalar for Complex64 {}
```

## Public API

All public functions take an explicit backend reference:

```rust
pub fn svd<T: LinalgScalar, B: LinalgBackend<T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T>> { ... }

pub fn qr<T: LinalgScalar, B: LinalgBackend<T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<QrResult<T>> { ... }

// etc. for lu, cholesky, eigen_sym, solve, solve_triangular, ...
```

Usage:

```rust
let mut backend = FaerBackend::new();
let svd_result = svd(&mut backend, &tensor, None)?;
let qr_result = qr(&mut backend, &tensor2)?;  // workspace reuse
```

## File Structure

```
tenferro-linalg/
  Cargo.toml              -- features = ["faer" (default), "lapack"]
  src/
    lib.rs                 -- public API + LinalgScalar
    backend/
      mod.rs               -- LinalgBackend<T> trait definition + re-export
      faer.rs              -- #[cfg(feature = "faer")]  FaerBackend
      lapack.rs            -- #[cfg(feature = "lapack")] LapackBackend (future)
```

## Feature Flags

```toml
[features]
default = ["faer"]
faer = ["dep:faer"]
lapack = ["dep:lapack"]  # future

[dependencies]
faer = { workspace = true, optional = true }
```

## Backend Implementations

### FaerBackend

```rust
#[cfg(feature = "faer")]
pub struct FaerBackend {
    // Future: workspace cache for repeated same-size decompositions
}

impl LinalgBackend<f64> for FaerBackend {
    type Real = f64;
    // ... calls faer API, copies results into output slices
}

impl LinalgBackend<f32> for FaerBackend {
    type Real = f32;
    // ...
}
```

Note: faer internally allocates `Mat` types. The trait interface is zero-allocation,
but the faer backend will copy results from faer's owned matrices into the provided
output slices. LAPACK backend can write directly into output buffers.

### LapackBackend (future)

```rust
#[cfg(feature = "lapack")]
pub struct LapackBackend {
    work: Vec<f64>,  // reusable workspace
}
```

## Capability Matrix

| Operation     | faer f64 | faer f32 | faer c64 | LAPACK f64 | LAPACK c64 | AD rrule | AD frule |
|---------------|----------|----------|----------|------------|------------|----------|----------|
| thin_svd      | Yes      | Yes      | Future   | Future     | Future     | Yes      | Yes      |
| qr            | Yes      | Yes      | Future   | Future     | Future     | Yes      | Yes      |
| lu            | Yes      | Yes      | Future   | Future     | Future     | Yes      | Yes      |
| cholesky      | Yes      | Yes      | Future   | Future     | Future     | Yes      | Yes      |
| eigen_sym     | Yes      | Yes      | Future   | Future     | Future     | Yes      | Yes      |
| mat_mul       | Yes      | Yes      | Future   | Future     | Future     | N/A      | N/A      |
| solve         | Yes      | Yes      | Future   | Future     | Future     | Yes      | Yes      |
| solve_tri     | Yes      | Yes      | Future   | Future     | Future     | Yes      | Yes      |

## AD Rules

AD rules (`svd_rrule`, `qr_rrule`, etc.) are stateless functions that call the
public API. They are unaffected by the backend abstraction — they continue to work
through the same `svd()`, `qr()` functions, now with an explicit backend parameter.

## Error Handling

- Backend methods return `tenferro_device::Result<()>`
- Backend-specific errors use `Error::InvalidArgument(String)`
  (e.g., "matrix is not positive definite" for Cholesky)
- Output buffer size validation at the public API layer before calling backend

## Migration from Current Code

1. Replace `FaerOps` trait with `LinalgBackend<T>` trait
2. Move faer impl into `backend/faer.rs` behind `#[cfg(feature = "faer")]`
3. Remove `FaerOps` bound from `LinalgScalar`
4. Add `backend: &mut B` parameter to all public functions
5. Update all tests to pass `FaerBackend::new()` explicitly
6. Update `tenferro-capi` calls to create and pass backend
