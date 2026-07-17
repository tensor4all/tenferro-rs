use thiserror::Error;

/// Error returned when computing public reduction output shapes.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReductionShapeError {
    /// Axis is outside the input rank.
    #[error("axis {axis} is out of bounds for rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    /// Axis appears more than once.
    #[error("duplicate axis {axis}")]
    DuplicateAxis { axis: usize },
}

/// Compute the output shape for a reduction.
///
/// # Examples
///
/// ```
/// use tenferro_ops::reduction::reduced_shape;
///
/// assert_eq!(reduced_shape(&[2, 3, 4], &[1], false).unwrap(), vec![2, 4]);
/// assert_eq!(reduced_shape(&[2, 3, 4], &[1], true).unwrap(), vec![2, 1, 4]);
/// ```
///
/// # Errors
///
/// Returns [`ReductionShapeError::AxisOutOfBounds`] for an invalid axis or
/// [`ReductionShapeError::DuplicateAxis`] when an axis is repeated.
pub fn reduced_shape(
    input_shape: &[usize],
    axes: &[usize],
    keepdims: bool,
) -> Result<Vec<usize>, ReductionShapeError> {
    let mut reduced = vec![false; input_shape.len()];
    for &axis in axes {
        if axis >= input_shape.len() {
            return Err(ReductionShapeError::AxisOutOfBounds {
                axis,
                rank: input_shape.len(),
            });
        }
        if reduced[axis] {
            return Err(ReductionShapeError::DuplicateAxis { axis });
        }
        reduced[axis] = true;
    }
    let mut out = Vec::with_capacity(input_shape.len());
    for (axis, &dim) in input_shape.iter().enumerate() {
        if reduced[axis] {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    Ok(out)
}
