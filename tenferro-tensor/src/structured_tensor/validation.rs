use std::collections::HashMap;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::Tensor;

/// Canonicalize arbitrary axis class IDs to first-appearance order.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::structured_tensor::canonicalize_axis_classes;
///
/// assert_eq!(
///     canonicalize_axis_classes(&[4, 9, 4, 7, 9]),
///     vec![0, 1, 0, 2, 1],
/// );
/// ```
pub fn canonicalize_axis_classes(classes: &[usize]) -> Vec<usize> {
    let mut map = HashMap::new();
    let mut next = 0usize;
    classes
        .iter()
        .map(|&class_id| {
            if let Some(&mapped) = map.get(&class_id) {
                mapped
            } else {
                let mapped = next;
                next += 1;
                map.insert(class_id, mapped);
                mapped
            }
        })
        .collect()
}

/// Validate structured-tensor metadata against a compressed payload.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{structured_tensor::validate_layout, MemoryOrder, Tensor};
///
/// let payload =
///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// validate_layout(&[2, 2], &[0, 0], &payload).unwrap();
/// ```
pub fn validate_layout<T: Scalar>(
    logical_dims: &[usize],
    axis_classes: &[usize],
    payload: &Tensor<T>,
) -> Result<()> {
    if logical_dims.len() != axis_classes.len() {
        return Err(Error::InvalidArgument(format!(
            "logical_dims length ({}) must match axis_classes length ({})",
            logical_dims.len(),
            axis_classes.len(),
        )));
    }
    if logical_dims.is_empty() && payload.dims().is_empty() {
        return Ok(());
    }

    let class_count = axis_classes
        .iter()
        .copied()
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    if payload.dims().len() != class_count {
        return Err(Error::InvalidArgument(format!(
            "payload rank {} must equal number of classes {}",
            payload.dims().len(),
            class_count,
        )));
    }

    let mut class_dims = vec![None; class_count];
    for (&dim, &class_id) in logical_dims.iter().zip(axis_classes.iter()) {
        if let Some(existing) = class_dims[class_id] {
            if existing != dim {
                return Err(Error::InvalidArgument(format!(
                    "axis class {class_id} has inconsistent logical dims: {existing} vs {dim}",
                )));
            }
        } else {
            class_dims[class_id] = Some(dim);
        }
    }

    for (class_id, maybe_dim) in class_dims.iter().enumerate() {
        let expected = maybe_dim.unwrap_or(0);
        let got = payload.dims()[class_id];
        if expected != got {
            return Err(Error::InvalidArgument(format!(
                "payload dim mismatch for class {class_id}: expected {expected}, got {got}",
            )));
        }
    }

    Ok(())
}

/// Validate that `perm` is a complete permutation of `0..rank`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::structured_tensor::validate_permutation;
///
/// validate_permutation(&[1, 0], 2, "example").unwrap();
/// ```
pub(crate) fn validate_permutation(
    perm: &[usize],
    rank: usize,
    op_name: &'static str,
) -> Result<()> {
    if perm.len() != rank {
        return Err(Error::InvalidArgument(format!(
            "{op_name} requires permutation length {rank}, got {}",
            perm.len()
        )));
    }

    let mut seen = vec![false; rank];
    for &axis in perm {
        if axis >= rank {
            return Err(Error::InvalidArgument(format!(
                "{op_name} permutation index {axis} out of range for rank {rank}",
            )));
        }
        if seen[axis] {
            return Err(Error::InvalidArgument(format!(
                "{op_name} permutation contains duplicate axis {axis}",
            )));
        }
        seen[axis] = true;
    }

    Ok(())
}
