//! CPU fallback kernels for tropical matrix products.
//!
//! The routines in this module operate on compact column-major buffers. They
//! are intended as small, generic fallbacks for extension planning and lowering
//! work that needs both tropical values and the first winning contracted index.
//! NaN products are ignored in the same spirit as tenferro CPU `reduce_max` and
//! `reduce_min`; if every product for an output cell is NaN, the cell receives
//! the semiring additive identity. A zero contracted dimension is accepted only
//! when the output is empty.
//!
//! # Examples
//!
//! ```
//! use tenferro_ext_tropical::cpu::{tropical_gemm_with_argmax, TropicalGemmKind};
//!
//! let a = vec![10.0_f64, 0.0, 1.0, 5.0]; // shape [2, 2], column-major
//! let b = vec![1.0_f64, 10.0, 0.0, 1.0]; // shape [2, 2], column-major
//! let out = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 2, 2, &b, 2)?;
//!
//! assert_eq!(out.values, vec![11.0, 15.0, 10.0, 6.0]);
//! assert_eq!(out.argmax, vec![0, 1, 0, 1]);
//! # Ok::<(), tenferro_tensor::Error>(())
//! ```

use num_traits::Float;
#[cfg(feature = "tropical-gemm")]
use num_traits::NumCast;

const OP: &str = "tropical_gemm_with_argmax";

/// Tropical GEMM semiring flavor.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::cpu::TropicalGemmKind;
///
/// assert_eq!(TropicalGemmKind::MaxPlus, TropicalGemmKind::MaxPlus);
/// assert_ne!(TropicalGemmKind::MaxPlus, TropicalGemmKind::MinPlus);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TropicalGemmKind {
    /// Max-plus product: `out[i, j] = max_kk(a[i, kk] + b[kk, j])`.
    MaxPlus,
    /// Min-plus product: `out[i, j] = min_kk(a[i, kk] + b[kk, j])`.
    MinPlus,
}

impl From<crate::TropicalKind> for TropicalGemmKind {
    fn from(kind: crate::TropicalKind) -> Self {
        match kind {
            crate::TropicalKind::MaxPlus => Self::MaxPlus,
            crate::TropicalKind::MinPlus => Self::MinPlus,
        }
    }
}

/// Tropical GEMM values and first-winning contracted indices.
///
/// Both buffers use column-major output order for shape `[m, n]`, with flat
/// index `i + j * m`.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::cpu::{tropical_gemm_with_argmax, TropicalGemmKind};
///
/// let out = tropical_gemm_with_argmax(
///     TropicalGemmKind::MinPlus,
///     &[1.0_f64, 4.0, 3.0, 2.0],
///     2,
///     2,
///     &[5.0_f64, 6.0, 7.0, 1.0],
///     2,
/// )?;
///
/// assert_eq!(out.values, vec![6.0, 8.0, 4.0, 3.0]);
/// assert_eq!(out.argmax, vec![0, 1, 1, 1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TropicalGemmArgmax<T> {
    /// Column-major tropical GEMM values.
    pub values: Vec<T>,
    /// First contracted index that produced each output value.
    pub argmax: Vec<u32>,
}

/// Compute a column-major tropical GEMM and first-winning contracted indices.
///
/// Inputs are compact column-major floating-point matrices: `a` has shape
/// `[m, k]`, `b` has shape `[k, n]`, and the output has shape `[m, n]` with
/// flat index `i + j * m`. Ties keep the first contracted index. NaN products
/// are ignored; if every product for an output cell is NaN, the value is
/// `-inf` for max-plus or `inf` for min-plus and the argmax placeholder is `0`.
///
/// With the `tropical-gemm` feature enabled, finite supported `f32`/`f64`
/// inputs may dispatch through the external `tropical-gemm` crate. Inputs that
/// contain NaN or infinities use [`tropical_gemm_with_argmax_generic`] so the
/// NaN-skipping semantics remain exact.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::Validation`] when input lengths do not
/// exactly match the provided dimensions, when `k == 0` with a non-empty
/// output, or when `k` is too large to represent a winning contracted index as
/// `u32`.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::cpu::{tropical_gemm_with_argmax, TropicalGemmKind};
///
/// let a = vec![10.0_f64, 0.0, 1.0, 5.0];
/// let b = vec![1.0_f64, 10.0, 0.0, 1.0];
/// let out = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 2, 2, &b, 2)?;
///
/// assert_eq!(out.values, vec![11.0, 15.0, 10.0, 6.0]);
/// assert_eq!(out.argmax, vec![0, 1, 0, 1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn tropical_gemm_with_argmax<T>(
    kind: TropicalGemmKind,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> tenferro_tensor::Result<TropicalGemmArgmax<T>>
where
    T: Float + 'static,
{
    validate_inputs(a, m, k, b, n)?;

    #[cfg(feature = "tropical-gemm")]
    {
        if let Some(out) = try_tropical_gemm_external(kind, a, m, k, b, n) {
            return Ok(out);
        }
    }

    Ok(tropical_gemm_with_argmax_generic_validated(
        kind, a, m, k, b, n,
    ))
}

/// Compute a column-major tropical GEMM using the generic reference fallback.
///
/// Inputs are compact column-major floating-point matrices: `a` has shape
/// `[m, k]`, `b` has shape `[k, n]`, and the output has shape `[m, n]` with
/// flat index `i + j * m`. Ties keep the first contracted index. NaN products
/// are ignored; if every product for an output cell is NaN, the value is
/// `-inf` for max-plus or `inf` for min-plus and the argmax placeholder is `0`.
///
/// # Errors
///
/// Returns a structured shape or argument validation error when input lengths
/// do not exactly match the provided dimensions, when `k == 0` with a
/// [`tenferro_tensor::Error::Validation`] when input lengths do not exactly
/// match the provided dimensions, when `k == 0` with a non-empty output, or
/// when `k` is too large to represent a winning contracted index as `u32`.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::cpu::{
///     tropical_gemm_with_argmax_generic, TropicalGemmKind,
/// };
///
/// let a = vec![10.0_f64, 0.0, 1.0, 5.0];
/// let b = vec![1.0_f64, 10.0, 0.0, 1.0];
/// let out = tropical_gemm_with_argmax_generic(TropicalGemmKind::MaxPlus, &a, 2, 2, &b, 2)?;
///
/// assert_eq!(out.values, vec![11.0, 15.0, 10.0, 6.0]);
/// assert_eq!(out.argmax, vec![0, 1, 0, 1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn tropical_gemm_with_argmax_generic<T>(
    kind: TropicalGemmKind,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> tenferro_tensor::Result<TropicalGemmArgmax<T>>
where
    T: Float,
{
    validate_inputs(a, m, k, b, n)?;
    Ok(tropical_gemm_with_argmax_generic_validated(
        kind, a, m, k, b, n,
    ))
}

fn tropical_gemm_with_argmax_generic_validated<T>(
    kind: TropicalGemmKind,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> TropicalGemmArgmax<T>
where
    T: Float,
{
    let out_len = m * n;

    let mut values = Vec::with_capacity(out_len);
    let mut argmax = Vec::with_capacity(out_len);

    for j in 0..n {
        let b_col = j * k;
        for i in 0..m {
            let mut best = identity(kind);
            let mut winner = 0_u32;
            let mut has_ordered_candidate = false;

            for kk in 0..k {
                let candidate = a[i + kk * m] + b[kk + b_col];
                if candidate.is_nan() {
                    continue;
                }
                let is_better = match kind {
                    TropicalGemmKind::MaxPlus => !has_ordered_candidate || candidate > best,
                    TropicalGemmKind::MinPlus => !has_ordered_candidate || candidate < best,
                };
                if is_better {
                    best = candidate;
                    winner = kk as u32;
                    has_ordered_candidate = true;
                }
            }

            values.push(best);
            argmax.push(winner);
        }
    }

    TropicalGemmArgmax { values, argmax }
}

#[cfg(feature = "tropical-gemm")]
fn try_tropical_gemm_external<T>(
    kind: TropicalGemmKind,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Option<TropicalGemmArgmax<T>>
where
    T: Float + 'static,
{
    use std::any::TypeId;

    // `tropical-gemm` keeps NaN products in its argmax path. Requiring every
    // input to be finite is conservative, but guarantees no contracted sum can
    // become NaN through either an input NaN or opposite-signed infinities.
    if !a.iter().chain(b).all(|value| value.is_finite()) {
        return None;
    }

    // This is the only scalar-type dispatch in this crate. Stable Rust cannot
    // specialize the existing generic public API for f32/f64 without changing
    // generic callers, while `tropical-gemm` exposes kernel dispatch for a
    // closed subset of semiring/scalar pairs.
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let a = to_f32_vec(a)?;
        let b = to_f32_vec(b)?;
        let out = match kind {
            TropicalGemmKind::MaxPlus => {
                tropical_gemm_external::<tropical_gemm::MaxPlus<f32>>(&a, m, k, &b, n)
            }
            TropicalGemmKind::MinPlus => {
                tropical_gemm_external::<tropical_gemm::MinPlus<f32>>(&a, m, k, &b, n)
            }
        };
        return convert_output(out);
    }

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let TropicalGemmKind::MaxPlus = kind else {
            return None;
        };
        let a = to_f64_vec(a)?;
        let b = to_f64_vec(b)?;
        let out = tropical_gemm_external::<tropical_gemm::MaxPlus<f64>>(&a, m, k, &b, n);
        return convert_output(out);
    }

    None
}

#[cfg(feature = "tropical-gemm")]
fn to_f32_vec<T: Float>(values: &[T]) -> Option<Vec<f32>> {
    values.iter().map(|value| NumCast::from(*value)).collect()
}

#[cfg(feature = "tropical-gemm")]
fn to_f64_vec<T: Float>(values: &[T]) -> Option<Vec<f64>> {
    values.iter().map(|value| NumCast::from(*value)).collect()
}

#[cfg(feature = "tropical-gemm")]
fn convert_output<T, U>(out: TropicalGemmArgmax<U>) -> Option<TropicalGemmArgmax<T>>
where
    T: Float,
    U: Float,
{
    let values = out
        .values
        .into_iter()
        .map(NumCast::from)
        .collect::<Option<Vec<T>>>()?;
    Some(TropicalGemmArgmax {
        values,
        argmax: out.argmax,
    })
}

#[cfg(feature = "tropical-gemm")]
fn tropical_gemm_external<S>(
    a: &[S::Scalar],
    m: usize,
    k: usize,
    b: &[S::Scalar],
    n: usize,
) -> TropicalGemmArgmax<S::Scalar>
where
    S: tropical_gemm::TropicalWithArgmax<Index = u32> + tropical_gemm::KernelDispatch,
    S::Scalar: Float,
{
    use tropical_gemm::TropicalSemiring;

    let a_ref = tropical_gemm::MatRef::<S>::from_slice(a, m, k);
    let b_ref = tropical_gemm::MatRef::<S>::from_slice(b, k, n);
    let result = a_ref.matmul_argmax(&b_ref);
    let values = result
        .values
        .as_slice()
        .iter()
        .map(TropicalSemiring::value)
        .collect();

    TropicalGemmArgmax {
        values,
        argmax: result.argmax,
    }
}

fn identity<T: Float>(kind: TropicalGemmKind) -> T {
    match kind {
        TropicalGemmKind::MaxPlus => T::neg_infinity(),
        TropicalGemmKind::MinPlus => T::infinity(),
    }
}

fn validate_inputs<T>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> tenferro_tensor::Result<()> {
    let expected_a = checked_len(m, k, "a")?;
    if a.len() != expected_a {
        return Err(tenferro_tensor::Error::validation(
            OP,
            tenferro_tensor::ValidationError::ShapeDataLengthMismatch {
                expected: expected_a,
                actual: a.len(),
            },
        ));
    }

    let expected_b = checked_len(k, n, "b")?;
    if b.len() != expected_b {
        return Err(tenferro_tensor::Error::validation(
            OP,
            tenferro_tensor::ValidationError::ShapeDataLengthMismatch {
                expected: expected_b,
                actual: b.len(),
            },
        ));
    }

    let out_len = checked_len(m, n, "output")?;
    if k == 0 {
        if out_len == 0 {
            return Ok(());
        }
        return Err(invalid_argument(
            "contracting dimension k must be nonzero for non-empty outputs",
        ));
    }

    let max_argmax_len = (u32::MAX as usize).saturating_add(1);
    if k > max_argmax_len {
        return Err(invalid_argument(format!(
            "contracting dimension k={k} cannot be represented as u32 argmax indices"
        )));
    }

    Ok(())
}

fn checked_len(lhs: usize, rhs: usize, _label: &str) -> tenferro_tensor::Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        tenferro_tensor::Error::validation(
            OP,
            tenferro_tensor::ValidationError::IntegerOverflow,
        )
    })
}

fn invalid_argument(message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(OP, "configuration", message)
}
