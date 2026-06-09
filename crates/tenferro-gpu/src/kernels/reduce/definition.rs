use crate::kernels::{CubeclKernelError, Result};

/// Validate that a reduction axis is in bounds for a tensor rank.
pub(crate) fn validate_axis(rank: usize, axis: usize) -> Result<()> {
    if axis >= rank {
        return Err(CubeclKernelError::InvalidAxis { axis, rank });
    }
    Ok(())
}

/// Return the keepdims output shape for reducing a single axis.
pub(crate) fn keepdims_output_shape(input_shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    validate_axis(input_shape.len(), axis)?;
    let mut output = input_shape.to_vec();
    output[axis] = 1;
    Ok(output)
}

/// Validate that an output shape matches a single-axis keepdims reduction.
pub(crate) fn validate_keepdims_output_shape(
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
