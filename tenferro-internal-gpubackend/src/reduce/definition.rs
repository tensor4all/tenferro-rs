use crate::{CubeclKernelError, Result};

/// Reduction operation supported by the CubeCL launch layer.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::ReduceOp;
///
/// let op = ReduceOp::Sum;
/// assert_eq!(format!("{op:?}"), "Sum");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all values along the reduced axis.
    Sum,
    /// Multiply all values along the reduced axis.
    Prod,
    /// Select the maximum value along the reduced axis.
    Max,
    /// Select the minimum value along the reduced axis.
    Min,
}

/// Scalar dtype accepted by the reduction launch helpers.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::ReduceDType;
///
/// let dtype = ReduceDType::F64;
/// assert_eq!(format!("{dtype:?}"), "F64");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceDType {
    /// 32-bit floating point values.
    F32,
    /// 64-bit floating point values.
    F64,
    /// 64-bit signed integer values.
    I64,
    /// 32-bit complex values.
    Complex32,
    /// 64-bit complex values.
    Complex64,
}

/// Validate that a reduction axis is in bounds for a tensor rank.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::validate_axis;
///
/// assert!(validate_axis(3, 2).is_ok());
/// assert!(validate_axis(3, 3).is_err());
/// ```
pub fn validate_axis(rank: usize, axis: usize) -> Result<()> {
    if axis >= rank {
        return Err(CubeclKernelError::InvalidAxis { axis, rank });
    }
    Ok(())
}

/// Return the keepdims output shape for reducing a single axis.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::keepdims_output_shape;
///
/// assert_eq!(keepdims_output_shape(&[2, 3, 4], 1).unwrap(), vec![2, 1, 4]);
/// ```
pub fn keepdims_output_shape(input_shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    validate_axis(input_shape.len(), axis)?;
    let mut output = input_shape.to_vec();
    output[axis] = 1;
    Ok(output)
}

/// Validate that an output shape matches a single-axis keepdims reduction.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::validate_keepdims_output_shape;
///
/// assert!(validate_keepdims_output_shape(&[2, 3, 4], &[2, 1, 4], 1).is_ok());
/// assert!(validate_keepdims_output_shape(&[2, 3, 4], &[2, 3, 1], 1).is_err());
/// ```
pub fn validate_keepdims_output_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
) -> Result<()> {
    let expected = keepdims_output_shape(input_shape, axis)?;
    if output_shape != expected {
        return Err(CubeclKernelError::MismatchOutputShape {
            expected,
            actual: output_shape.to_vec(),
        });
    }
    Ok(())
}

/// Return the length of the axis being reduced.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::axis_reduce_len;
///
/// assert_eq!(axis_reduce_len(&[2, 3, 4], 1).unwrap(), 3);
/// assert!(axis_reduce_len(&[2, 3, 4], 3).is_err());
/// ```
pub fn axis_reduce_len(input_shape: &[usize], axis: usize) -> Result<usize> {
    validate_axis(input_shape.len(), axis)?;
    Ok(input_shape[axis])
}

/// Return the number of elements produced by a single-axis keepdims reduction.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::reduced_output_len;
///
/// assert_eq!(reduced_output_len(&[2, 3, 4], 1).unwrap(), 8);
/// ```
pub fn reduced_output_len(input_shape: &[usize], axis: usize) -> Result<usize> {
    Ok(keepdims_output_shape(input_shape, axis)?
        .iter()
        .product::<usize>())
}

/// Return whether a reduction operation supports a scalar dtype.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::{supports_dtype, ReduceDType, ReduceOp};
///
/// assert!(supports_dtype(ReduceOp::Sum, ReduceDType::I64));
/// assert!(!supports_dtype(ReduceOp::Max, ReduceDType::Complex64));
/// ```
pub fn supports_dtype(op: ReduceOp, dtype: ReduceDType) -> bool {
    match op {
        ReduceOp::Sum | ReduceOp::Prod => matches!(
            dtype,
            ReduceDType::F32
                | ReduceDType::F64
                | ReduceDType::I64
                | ReduceDType::Complex32
                | ReduceDType::Complex64
        ),
        ReduceOp::Max | ReduceOp::Min => matches!(dtype, ReduceDType::F32 | ReduceDType::F64),
    }
}
