//! Signed public axes and their normalization (#2552).
//!
//! Every public argument that names an axis is an `isize`: a non-negative
//! value counts from the front and a negative value counts from the end
//! (`-1` is the last axis), as in NumPy. Lengths, extents, shapes and IR
//! identifiers stay `usize`. Operations normalize a signed axis to an index
//! with [`normalize_axis`] or [`normalize_axes`] and nowhere else.

use tinyvec::TinyVec;

use crate::{ValidationError, INLINE_RANK};

/// Small vector of signed public axes with inline capacity for common ranks.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::AxisVec;
///
/// let axes: AxisVec = [0, -1].into_iter().collect();
/// assert_eq!(axes.as_slice(), &[0, -1]);
/// ```
pub type AxisVec = TinyVec<[isize; INLINE_RANK]>;

/// Normalize one signed axis against `rank`.
///
/// `axis` in `0..rank` is returned unchanged; `axis` in `-rank..0` counts
/// from the end.
///
/// # Errors
///
/// Returns [`ValidationError::AxisOutOfBounds`] carrying the caller's original
/// signed `axis` when it is outside `-rank..rank`.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{normalize_axis, ValidationError};
///
/// assert_eq!(normalize_axis(1, 3).unwrap(), 1);
/// assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
/// assert_eq!(
///     normalize_axis(-4, 3),
///     Err(ValidationError::AxisOutOfBounds { axis: -4, rank: 3 })
/// );
/// ```
pub fn normalize_axis(axis: isize, rank: usize) -> Result<usize, ValidationError> {
    let out_of_bounds = ValidationError::AxisOutOfBounds { axis, rank };
    if axis >= 0 {
        let index = axis.unsigned_abs();
        if index < rank {
            Ok(index)
        } else {
            Err(out_of_bounds)
        }
    } else {
        let from_end = axis.unsigned_abs();
        if from_end <= rank {
            Ok(rank - from_end)
        } else {
            Err(out_of_bounds)
        }
    }
}

/// Normalize a list of signed axes against `rank`, rejecting duplicates.
///
/// The result keeps the order of `axes`. Two entries that name the same axis
/// (for example `1` and `-2` at rank 3) are a duplicate.
///
/// # Errors
///
/// Returns [`ValidationError::AxisOutOfBounds`] for the first entry outside
/// `-rank..rank`, and [`ValidationError::DuplicateAxis`] (role `"axes"`,
/// normalized index) for the first repeated axis.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{normalize_axes, ValidationError};
///
/// assert_eq!(normalize_axes(&[0, -1], 3).unwrap().as_slice(), &[0, 2]);
/// assert_eq!(
///     normalize_axes(&[1, -2], 3),
///     Err(ValidationError::DuplicateAxis { axis: 1, role: "axes" })
/// );
/// ```
pub fn normalize_axes(
    axes: &[isize],
    rank: usize,
) -> Result<TinyVec<[usize; INLINE_RANK]>, ValidationError> {
    let mut normalized = TinyVec::<[usize; INLINE_RANK]>::new();
    for &axis in axes {
        let index = normalize_axis(axis, rank)?;
        // INVARIANT: `normalized.len() <= rank`, and public axis lists are
        // bounded by the tensor rank, so this scan is rank-bounded quadratic.
        if normalized.contains(&index) {
            return Err(ValidationError::DuplicateAxis {
                axis: index,
                role: "axes",
            });
        }
        normalized.push(index);
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests;
