use crate::{CubeclKernelError, Result};

/// Reduction operation supported by the CubeCL launch layer.
///
/// # Examples
///
/// ```
/// use tenferro_cubecl::reduce::ReduceOp;
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
/// use tenferro_cubecl::reduce::ReduceDType;
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
/// use tenferro_cubecl::reduce::validate_axis;
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
/// use tenferro_cubecl::reduce::keepdims_output_shape;
///
/// assert_eq!(keepdims_output_shape(&[2, 3, 4], 1).unwrap(), vec![2, 1, 4]);
/// ```
pub fn keepdims_output_shape(input_shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    validate_axis(input_shape.len(), axis)?;
    let mut output = input_shape.to_vec();
    output[axis] = 1;
    Ok(output)
}
