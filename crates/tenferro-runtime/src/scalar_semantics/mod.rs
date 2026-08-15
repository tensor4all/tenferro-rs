use tenferro_tensor::{DType, Tensor};

use crate::error::{Error, ErrorPhase, Result};

fn finite_real_scalar(op: &'static str, value: f64) -> Result<f64> {
    finite_real_scalar_at(op, ErrorPhase::GraphBuild, value)
}

fn finite_real_scalar_at(op: &'static str, phase: ErrorPhase, value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(Error::invalid_argument(
            op,
            phase,
            "value",
            format!("real scalar value must be finite, got {value}"),
        ));
    }
    Ok(value)
}

pub(crate) fn round_real_to_i64(value: f64) -> Result<i64> {
    round_real_to_i64_for_op("scale_real", value)
}

pub(crate) fn round_real_to_i64_for_op(op: &'static str, value: f64) -> Result<i64> {
    let rounded = finite_real_scalar(op, value)?.round();
    // `i64::MAX as f64` rounds up to 2^63, so `rounded > i64::MAX as f64`
    // lets exactly 2^63 pass and `as i64` silently saturates to i64::MAX.
    // Reject `rounded >= 2^63` instead. f64 spacing at 2^63 is 1024, so no
    // value in (2^63 - 1024, 2^63) is representable except 2^63 - 1024,
    // which is the largest valid f64 below 2^63; -2^63 == i64::MIN stays
    // valid (exactly representable, in range).
    const I64_EXCLUSIVE_UPPER_F64: f64 = 9_223_372_036_854_775_808.0; // 2^63
    if rounded < i64::MIN as f64 || rounded >= I64_EXCLUSIVE_UPPER_F64 {
        return Err(Error::invalid_argument(
            op,
            ErrorPhase::GraphBuild,
            "value",
            format!("rounded real scalar {rounded} is out of i64 range"),
        ));
    }
    Ok(rounded as i64)
}

pub(crate) fn round_real_to_i32_for_op(op: &'static str, value: f64) -> Result<i32> {
    let rounded = round_real_to_i64_for_op(op, value)?;
    i32::try_from(rounded).map_err(|_| {
        Error::invalid_argument(
            op,
            ErrorPhase::GraphBuild,
            "value",
            format!("rounded real scalar {rounded} is out of i32 range"),
        )
    })
}

pub(crate) fn bool_from_real_for_op(op: &'static str, value: f64) -> Result<bool> {
    Ok(finite_real_scalar(op, value)? != 0.0)
}

/// Evaluate the scalar size supplied to a dynamic truncation operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::Tensor;
/// use tenferro_runtime::scalar_semantics::dynamic_truncate_size;
///
/// let size = Tensor::from_vec_col_major(vec![], vec![3_i64]).unwrap();
/// assert_eq!(dynamic_truncate_size(&size, 10).unwrap(), 3);
/// ```
///
/// # Errors
///
/// Returns `Validation(InvalidArgument)` when `size_tensor` is not scalar,
/// `Unsupported` when its dtype is not `f32`, `f64`, or `i64`,
/// `Validation(InvalidArgument)` for a non-finite floating value, and
/// `RuntimeState` when the scalar's backend storage is unexpectedly empty.
pub fn dynamic_truncate_size(size_tensor: &Tensor, axis_extent: usize) -> Result<usize> {
    if !size_tensor.shape().is_empty() {
        return Err(Error::invalid_argument(
            "dynamic_truncate",
            ErrorPhase::Execution,
            "size",
            format!("size must be scalar, got shape {:?}", size_tensor.shape()),
        ));
    }
    if let Tensor::I64(inner) = size_tensor {
        let value = scalar_host_value(inner.host_data()?, DType::I64)?;
        return Ok(truncate_i64_size(value, axis_extent));
    }
    let value = scalar_size_value(size_tensor)?;
    let rounded = finite_real_scalar_at("dynamic_truncate", ErrorPhase::Execution, value)?.round();
    Ok(rounded.max(0.0).min(axis_extent as f64) as usize)
}

fn truncate_i64_size(value: i64, axis_extent: usize) -> usize {
    if value <= 0 {
        return 0;
    }
    usize::try_from(value).map_or(axis_extent, |value| value.min(axis_extent))
}

fn scalar_size_value(size_tensor: &Tensor) -> Result<f64> {
    if !size_tensor.shape().is_empty() {
        return Err(Error::invalid_argument(
            "dynamic_truncate",
            ErrorPhase::Execution,
            "size",
            format!("size must be scalar, got shape {:?}", size_tensor.shape()),
        ));
    }

    match size_tensor {
        Tensor::F64(inner) => scalar_host_value(inner.host_data()?, DType::F64),
        Tensor::F32(inner) => Ok(scalar_host_value(inner.host_data()?, DType::F32)? as f64),
        _ => Err(Error::unsupported(
            "dynamic_truncate",
            ErrorPhase::Execution,
            format!(
                "dtype {:?} is not accepted for the size scalar",
                size_tensor.dtype()
            ),
        )),
    }
}

fn scalar_host_value<T: Copy>(data: &[T], dtype: DType) -> Result<T> {
    data.first().copied().ok_or_else(|| {
        Error::runtime_state(
            "dynamic_truncate",
            ErrorPhase::Execution,
            format!("dynamic_truncate size scalar had empty {dtype:?} host buffer"),
        )
    })
}

#[cfg(test)]
mod tests;
