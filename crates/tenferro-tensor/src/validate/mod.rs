//! Validation helpers shared across backends and exec layers.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::validate::validate_nonsingular_u;
//! use tenferro_tensor::{Tensor, TypedTensor};
//!
//! let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap());
//! assert!(validate_nonsingular_u(&t).is_ok());
//! ```

use num_complex::{Complex32, Complex64};

use crate::{DType, Error, Result, Tensor, TypedTensor};

/// Promote two dtypes according to tenferro's public dtype-promotion lattice.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::promote_dtype;
/// use tenferro_tensor::DType;
///
/// assert_eq!(promote_dtype(DType::I32, DType::F32), DType::F64);
/// ```
pub fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    use DType::*;
    match (lhs, rhs) {
        (Bool, Bool) => Bool,
        (Bool, other) | (other, Bool) => other,
        (I32, I32) => I32,
        (I32, I64) | (I64, I32) | (I64, I64) => I64,
        (I32 | I64, F32 | F64) | (F32 | F64, I32 | I64) => F64,
        (I32 | I64, C32 | C64) | (C32 | C64, I32 | I64) => C64,
        (F32, F32) => F32,
        (F32, F64) | (F64, F32) | (F64, F64) => F64,
        (F32, C32) | (C32, F32) | (C32, C32) => C32,
        (F32, C64) | (C64, F32) => C64,
        (F64, C32 | C64) | (C32 | C64, F64) => C64,
        (C32, C64) | (C64, C32) | (C64, C64) => C64,
    }
}

/// Return whether public `convert` may change `from` into `to`.
///
/// Checked conversion follows the same dtype lattice as implicit promotion.
/// Use explicit `cast` for value-changing projections outside this lattice.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::can_convert_dtype;
/// use tenferro_tensor::DType;
///
/// assert!(can_convert_dtype(DType::F32, DType::F64));
/// assert!(!can_convert_dtype(DType::F64, DType::I32));
/// ```
pub fn can_convert_dtype(from: DType, to: DType) -> bool {
    promote_dtype(from, to) == to
}

/// Validate a public checked dtype conversion.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::validate_convert_dtype;
/// use tenferro_tensor::DType;
///
/// assert!(validate_convert_dtype("convert", DType::F32, DType::F64).is_ok());
/// assert!(validate_convert_dtype("convert", DType::C64, DType::F64).is_err());
/// ```
pub fn validate_convert_dtype(op: &'static str, from: DType, to: DType) -> Result<()> {
    if can_convert_dtype(from, to) {
        return Ok(());
    }

    Err(Error::UnsupportedDTypeConversion {
        op,
        from,
        to,
        message: "checked convert only accepts conversions allowed by dtype promotion; use explicit cast for lossy dtype projection".to_string(),
    })
}

/// Trait for detecting singular or non-finite diagonal entries.
///
/// Implemented for `f32`, `f64`, `Complex32`, and `Complex64`.
/// A value is considered singular if it is zero, NaN, infinite,
/// or (for complex types) if either component is non-finite.
pub trait DiagSingularity {
    /// Returns `true` if the value is singular or non-finite.
    fn is_singular_or_nonfinite(&self) -> bool;
}

macro_rules! impl_diag_singularity_float {
    ($($t:ty),* $(,)?) => {
        $(
            impl DiagSingularity for $t {
                fn is_singular_or_nonfinite(&self) -> bool {
                    !self.is_finite() || *self == 0.0
                }
            }
        )*
    };
}

impl_diag_singularity_float!(f64, f32);

macro_rules! impl_diag_singularity_complex {
    ($($t:ty),* $(,)?) => {
        $(
            impl DiagSingularity for $t {
                fn is_singular_or_nonfinite(&self) -> bool {
                    !self.re.is_finite() || !self.im.is_finite() || self.norm_sqr() == 0.0
                }
            }
        )*
    };
}

impl_diag_singularity_complex!(Complex64, Complex32);

/// Checks that every diagonal element of a (possibly batched) upper-triangular
/// factor is non-singular and finite.
///
/// Iterates over all batch slices and inspects the diagonal entries
/// `data[i + i * rows]` for `i` in `0..min(rows, cols)`. Returns
/// [`Error::BackendFailure`] with `op: "solve"` on the first offending entry,
/// or [`Error::RankMismatch`] when `t` has rank less than two.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::check_singular_diagonal;
/// use tenferro_tensor::TypedTensor;
///
/// let t = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0f32, 0.0, 0.0, 2.0]).unwrap();
/// assert!(check_singular_diagonal(&t).is_ok());
/// ```
pub fn check_singular_diagonal<T: DiagSingularity + Copy + std::fmt::Debug>(
    t: &TypedTensor<T>,
) -> Result<()> {
    if t.shape().len() < 2 {
        return Err(Error::RankMismatch {
            op: "solve",
            expected: 2,
            actual: t.shape().len(),
        });
    }
    let rows = t.shape()[0];
    let cols = t.shape()[1];
    let n = rows.min(cols);
    let batch_total: usize = t.shape()[2..].iter().product();
    let batch_total = batch_total.max(1);
    let slice_size = rows * cols;
    let data = t.host_data()?;
    for batch_idx in 0..batch_total {
        let batch = &data[batch_idx * slice_size..(batch_idx + 1) * slice_size];
        for i in 0..n {
            let diag = batch[i + i * rows];
            if diag.is_singular_or_nonfinite() {
                return Err(Error::backend_failure(
                    "solve",
                    if batch_total > 1 {
                        format!(
                            "singular matrix: non-finite or zero diagonal at batch {}, position [{},{}] = {:?}",
                            batch_idx, i, i, diag
                        )
                    } else {
                        format!(
                            "singular matrix: non-finite or zero diagonal at position [{},{}] = {:?}",
                            i, i, diag
                        )
                    },
                ));
            }
        }
    }
    Ok(())
}

/// Validates that the upper-triangular factor `u` of a matrix decomposition
/// has no singular (zero) or non-finite diagonal entries.
///
/// Dispatches to [`check_singular_diagonal`] after unpacking the concrete
/// tensor variant. Returns `Ok(())` when all diagonal entries are valid.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::validate_nonsingular_u;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap());
/// assert!(validate_nonsingular_u(&t).is_ok());
/// ```
pub fn validate_nonsingular_u(u: &Tensor) -> Result<()> {
    match u {
        Tensor::F64(t) => check_singular_diagonal(t),
        Tensor::F32(t) => check_singular_diagonal(t),
        Tensor::C64(t) => check_singular_diagonal(t),
        Tensor::C32(t) => check_singular_diagonal(t),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(Error::backend_failure(
            "solve",
            format!("unsupported dtype {:?}", u.dtype()),
        )),
    }
}

#[cfg(test)]
mod tests;
