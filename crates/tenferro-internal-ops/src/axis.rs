use thiserror::Error;

/// Error returned when normalizing user-facing axis arguments.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AxisError {
    /// Axis is outside `[-rank, rank)`.
    #[error("axis {axis} is out of bounds for rank {rank}")]
    OutOfBounds { axis: isize, rank: usize },
    /// Axis appears more than once after negative-axis normalization.
    #[error("duplicate axis {axis}")]
    Duplicate { axis: usize },
}

/// Normalize a possibly-negative axis against `rank`.
///
/// # Examples
///
/// ```
/// use tenferro_ops::axis::normalize_axis;
///
/// assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
/// assert!(normalize_axis(3, 3).is_err());
/// ```
///
/// # Errors
///
/// Returns [`AxisError::OutOfBounds`] when `axis` is outside the normalized
/// range for `rank`.
pub fn normalize_axis(axis: isize, rank: usize) -> Result<usize, AxisError> {
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        rank.checked_sub(axis.unsigned_abs())
            .ok_or(AxisError::OutOfBounds { axis, rank })?
    };
    if normalized >= rank {
        return Err(AxisError::OutOfBounds { axis, rank });
    }
    Ok(normalized)
}

/// Normalize a list of possibly-negative axes and reject duplicates.
///
/// # Examples
///
/// ```
/// use tenferro_ops::axis::normalize_axes;
///
/// assert_eq!(normalize_axes(&[0, -1], 3).unwrap(), vec![0, 2]);
/// assert!(normalize_axes(&[1, -2], 3).is_err());
/// ```
///
/// # Errors
///
/// Returns [`AxisError::OutOfBounds`] for an invalid axis or
/// [`AxisError::Duplicate`] when two axes normalize to the same position.
pub fn normalize_axes(axes: &[isize], rank: usize) -> Result<Vec<usize>, AxisError> {
    let mut out = Vec::with_capacity(axes.len());
    let mut seen = vec![false; rank];
    for &axis in axes {
        let normalized = normalize_axis(axis, rank)?;
        if seen[normalized] {
            return Err(AxisError::Duplicate { axis: normalized });
        }
        seen[normalized] = true;
        out.push(normalized);
    }
    Ok(out)
}
