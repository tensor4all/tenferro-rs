# Complex64/Complex32 Linalg Support Design

**Issue**: #132 — Add Complex64/Complex32 support to LinalgBackend
**Date**: 2026-02-23
**Status**: Approved
**Scope**: Primal operations only. AD rules for complex types deferred to future issue.

## Problem

`LinalgScalar` trait requires `Float`, which excludes complex types.
Public API functions require `Real = T`, preventing complex scalars.
Result types (`SvdResult<T>`, `EigenResult<T>`) store real-valued outputs
(singular values, eigenvalues) as `Tensor<T>`, which is incorrect for complex types.

## Decisions

1. **Remove `Float` from `LinalgScalar`** — replace with `NumCast + Neg<Output = Self>`
2. **Add `type Real` to `LinalgScalar`** — mirrors `LinalgBackend::Real`
3. **Parameterize result types** — `SvdResult<T, R = T>`, `EigenResult<T, R = T>`
4. **Remove `Real = T` from public API bounds** — use `B::Real` in return types
5. **Safe copy conversion** for `Complex64 ↔ faer::c64` (no unsafe transmute)
6. **AD rules unchanged** — complex AD deferred to separate issue

## LinalgScalar Trait

```rust
pub trait LinalgScalar:
    Scalar + std::ops::Sub<Output = Self> + std::ops::Neg<Output = Self>
    + std::fmt::Debug + num_traits::NumCast + 'static
{
    /// The real scalar type. For real T, this is T itself.
    /// For Complex64, this is f64.
    type Real: LinalgScalar<Real = Self::Real> + num_traits::Float;
}

impl LinalgScalar for f64 { type Real = f64; }
impl LinalgScalar for f32 { type Real = f32; }
impl LinalgScalar for Complex64 { type Real = f64; }
impl LinalgScalar for Complex32 { type Real = f32; }
```

## Result Types

```rust
pub struct SvdResult<T: Scalar, R: Scalar = T> {
    pub u: Tensor<T>,
    pub s: Tensor<R>,   // singular values are always real
    pub vt: Tensor<T>,
}

pub struct EigenResult<T: Scalar, R: Scalar = T> {
    pub values: Tensor<R>,   // eigenvalues of Hermitian matrix are real
    pub vectors: Tensor<T>,
}
```

Default `R = T` preserves backward compatibility for `f64`/`f32`.

## Public API

```rust
pub fn svd<T: LinalgScalar, B: backend::LinalgBackend<T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, B::Real>>
```

Similarly for `eigen`, `slogdet`, `norm`, and other functions that return real values.

## FaerBackend Complex Implementation

```rust
impl LinalgBackend<Complex64> for FaerBackend {
    type Real = f64;

    fn thin_svd(&mut self, a: &[Complex64], m: usize, n: usize,
                u: &mut [Complex64], s: &mut [f64], vt: &mut [Complex64]) -> Result<()> {
        // Convert Complex64 slice → faer::c64 slice (safe copy)
        // Run faer thin_svd
        // Copy results back to output slices
    }
}
```

Type conversion: `Complex64 { re, im }` ↔ `faer::c64 { re, im }` via element-wise copy.

## Scope Exclusions

- AD rules (`svd_rrule`, `qr_rrule`, etc.) remain real-only for now
- AD rules will need `transpose` → conjugate transpose changes (future issue)
- Public AD functions may add `T: LinalgScalar<Real = T>` bound to restrict to real types
