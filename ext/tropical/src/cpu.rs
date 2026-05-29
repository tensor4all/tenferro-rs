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
/// # Errors
///
/// Returns [`tenferro_tensor::Error::InvalidConfig`] when input lengths do not
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
    T: Float,
{
    validate_inputs(a, m, k, b, n)?;

    let out_len = checked_len(m, n, "output")?;
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

    Ok(TropicalGemmArgmax { values, argmax })
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
        return Err(invalid_config(format!(
            "a length mismatch: expected {expected_a} elements for shape [{m}, {k}], got {}",
            a.len()
        )));
    }

    let expected_b = checked_len(k, n, "b")?;
    if b.len() != expected_b {
        return Err(invalid_config(format!(
            "b length mismatch: expected {expected_b} elements for shape [{k}, {n}], got {}",
            b.len()
        )));
    }

    let out_len = checked_len(m, n, "output")?;
    if k == 0 {
        if out_len == 0 {
            return Ok(());
        }
        return Err(invalid_config(
            "contracting dimension k must be nonzero for non-empty outputs",
        ));
    }

    let max_argmax_len = (u32::MAX as usize).saturating_add(1);
    if k > max_argmax_len {
        return Err(invalid_config(format!(
            "contracting dimension k={k} cannot be represented as u32 argmax indices"
        )));
    }

    Ok(())
}

fn checked_len(lhs: usize, rhs: usize, label: &str) -> tenferro_tensor::Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        invalid_config(format!(
            "{label} element count overflows usize for dimensions {lhs} and {rhs}"
        ))
    })
}

fn invalid_config(message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::InvalidConfig {
        op: OP,
        message: message.into(),
    }
}
