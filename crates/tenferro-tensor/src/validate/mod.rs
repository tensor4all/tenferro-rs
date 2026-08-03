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

use crate::{
    DType, DotGeneralConfig, Error, ErrorKind, Result, ShapeMismatch, Tensor, TensorScalar,
    TypedTensor, ValidationError,
};

/// Domain-specific reasons reported by triangular-factor validation.
///
/// The outer tensor error classifies these failures as numerical or unsupported
/// while retaining this value as its typed source.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::DiagonalError;
///
/// let error = DiagonalError::SingularOrNonFinite {
///     index: 1,
/// };
/// assert!(error.to_string().contains("position [1,1]"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum DiagonalError {
    #[error("singular or non-finite diagonal at position [{index},{index}]")]
    SingularOrNonFinite { index: usize },
    #[error("singular or non-finite diagonal at batch {batch}, position [{index},{index}]")]
    BatchedSingularOrNonFinite { batch: usize, index: usize },
    #[error("triangular solve does not support dtype {dtype:?}")]
    UnsupportedDType { dtype: DType },
}

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
/// # Errors
///
/// Returns [`crate::Error::UnsupportedDTypeConversion`] when the requested
/// conversion is outside the checked promotion lattice. Use an explicit cast
/// for lossy projections.
pub fn validate_convert_dtype(op: &'static str, from: DType, to: DType) -> Result<()> {
    if can_convert_dtype(from, to) {
        return Ok(());
    }

    Err(Error::unsupported_dtype_conversion(
        op,
        from,
        to,
        "checked convert only accepts conversions allowed by dtype promotion; use explicit cast for lossy dtype projection",
    ))
}

/// Compute a shape product with overflow reported as a typed tensor error.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::checked_shape_product;
///
/// assert_eq!(checked_shape_product("zeros", "shape", &[2, 3])?, 6);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
/// # Errors
///
/// Returns [`crate::Error::Validation`] containing
/// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
/// product of `shape` exceeds `usize::MAX`; `role` identifies the shape-like
/// argument in the diagnostic.
pub fn checked_shape_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            Error::invalid_argument(op, role, format!("product overflows for shape {shape:?}"))
        })
}

/// Validate a full permutation for a tensor rank.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::validate_permutation_axes;
///
/// validate_permutation_axes("transpose", 2, &[1, 0])?;
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
/// # Errors
///
/// Returns [`crate::Error::Validation`] with `RankMismatch`,
/// `AxisOutOfBounds`, or `DuplicateAxis` as appropriate.
pub fn validate_permutation_axes(op: &'static str, rank: usize, perm: &[usize]) -> Result<()> {
    if perm.len() != rank {
        return Err(Error::validation(
            op,
            ValidationError::RankMismatch {
                expected: rank,
                actual: perm.len(),
            },
        ));
    }

    let mut seen = vec![false; rank];
    for &axis in perm {
        if axis >= rank {
            return Err(Error::validation(
                op,
                ValidationError::AxisOutOfBounds { axis, rank },
            ));
        }
        if seen[axis] {
            return Err(Error::validation(
                op,
                ValidationError::DuplicateAxis {
                    axis,
                    role: "permutation",
                },
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

/// Validate a subset of axes for a tensor rank.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::validate_unique_axes;
///
/// validate_unique_axes("reduce_sum", "axis", 3, &[0, 2])?;
/// assert!(validate_unique_axes("reduce_sum", "axis", 2, &[2]).is_err());
/// assert!(validate_unique_axes("reduce_sum", "axis", 2, &[0, 0]).is_err());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds` for an invalid
/// axis or `DuplicateAxis` when an axis occurs more than once.
pub fn validate_unique_axes(
    op: &'static str,
    role: &'static str,
    rank: usize,
    axes: &[usize],
) -> Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::validation(
                op,
                ValidationError::AxisOutOfBounds { axis, rank },
            ));
        }
        if seen[axis] {
            return Err(Error::validation(
                op,
                ValidationError::DuplicateAxis { axis, role },
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

/// Validate rank-2 matrix multiplication shapes and return its dot-general config.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::validate::matmul_config_for_shapes;
///
/// let config = matmul_config_for_shapes("matmul", &[2, 3], &[3, 4])?;
/// assert_eq!(config.lhs_contracting_dims, vec![1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
/// # Errors
///
/// Returns [`crate::Error::Validation`] with `RankMismatch` for a non-matrix
/// input or `ShapeMismatch` when the contracting dimensions differ.
pub fn matmul_config_for_shapes(
    op: &'static str,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<DotGeneralConfig> {
    if lhs_shape.len() != 2 {
        return Err(Error::validation(
            op,
            ValidationError::RankMismatch {
                expected: 2,
                actual: lhs_shape.len(),
            },
        ));
    }
    if rhs_shape.len() != 2 {
        return Err(Error::validation(
            op,
            ValidationError::RankMismatch {
                expected: 2,
                actual: rhs_shape.len(),
            },
        ));
    }
    if lhs_shape[1] != rhs_shape[0] {
        return Err(Error::validation(
            op,
            ShapeMismatch::IncompatibleShapes {
                lhs: lhs_shape.to_vec().into(),
                rhs: rhs_shape.to_vec().into(),
            }
            .into(),
        ));
    }

    Ok(DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
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
                    // Why not `norm_sqr() == 0`: squaring a representable tiny
                    // component can underflow and relabel a nonzero pivot as zero.
                    !self.re.is_finite()
                        || !self.im.is_finite()
                        || (self.re == 0.0 && self.im == 0.0)
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
/// a numerical [`crate::Error::Extension`] carrying a typed
/// [`DiagonalError::SingularOrNonFinite`] source for the first offending entry,
/// or [`ValidationError::RankMismatch`] wrapped in [`Error::Validation`] when
/// `t` has rank less than two.
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
/// # Errors
///
/// Returns [`crate::Error::Validation`] with `RankMismatch` for a non-matrix
/// tensor, a numerical [`crate::Error::Extension`] with a typed
/// [`DiagonalError::SingularOrNonFinite`] source for a singular or non-finite
/// diagonal, or an unsupported [`crate::Error::Extension`] with a typed
/// [`DiagonalError::UnsupportedDType`] source for integer and boolean inputs.
pub fn check_singular_diagonal<T: DiagSingularity + TensorScalar + std::fmt::Debug>(
    t: &TypedTensor<T>,
) -> Result<()> {
    if t.shape().len() < 2 {
        return Err(Error::validation(
            "solve",
            ValidationError::RankMismatch {
                expected: 2,
                actual: t.shape().len(),
            },
        ));
    }
    let rows = t.shape()[0];
    let cols = t.shape()[1];
    let n = rows.min(cols);
    let batch_total = checked_shape_product("solve", "batch shape", &t.shape()[2..])?;
    let slice_size = checked_shape_product("solve", "matrix shape", &t.shape()[..2])?;
    let data = t.host_data()?;
    for batch_idx in 0..batch_total {
        let batch = &data[batch_idx * slice_size..(batch_idx + 1) * slice_size];
        for i in 0..n {
            let diag = batch[i + i * rows];
            if diag.is_singular_or_nonfinite() {
                return Err(Error::extension(
                    "solve",
                    "tensor-validation",
                    ErrorKind::NumericalFailure,
                    if batch_total > 1 {
                        DiagonalError::BatchedSingularOrNonFinite {
                            batch: batch_idx,
                            index: i,
                        }
                    } else {
                        DiagonalError::SingularOrNonFinite { index: i }
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
/// # Errors
///
/// Returns [`crate::Error::Validation`] with the applicable typed shape, rank,
/// axis, dtype, or argument source when validation fails. Singular or
/// non-finite diagonal checks return [`crate::Error::BackendFailure`].
pub fn validate_nonsingular_u(u: &Tensor) -> Result<()> {
    match u {
        Tensor::F64(t) => check_singular_diagonal(t),
        Tensor::F32(t) => check_singular_diagonal(t),
        Tensor::C64(t) => check_singular_diagonal(t),
        Tensor::C32(t) => check_singular_diagonal(t),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(Error::extension(
            "solve",
            "tensor-validation",
            ErrorKind::Unsupported,
            DiagonalError::UnsupportedDType { dtype: u.dtype() },
        )),
    }
}

#[cfg(test)]
mod tests;
