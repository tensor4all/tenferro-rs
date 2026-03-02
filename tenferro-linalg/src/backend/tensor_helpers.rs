//! Shared tensor validation and helper utilities for backend implementations.
//!
//! This module provides common tensor operations used by CPU (and future GPU)
//! backend implementations: shape validation, contiguous packing, and output
//! tensor allocation.

use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::LinalgScalar;

/// Validate that a tensor has at least 2 dimensions and return `(m, n, batch_dims)`.
pub(crate) fn validate_matrix_shape<T: LinalgScalar>(
    a: &Tensor<T>,
) -> Result<(usize, usize, &[usize])> {
    let dims = a.dims();
    if dims.len() < 2 {
        return Err(Error::InvalidArgument(format!(
            "expected at least 2D tensor, got {}D",
            dims.len()
        )));
    }
    let m = dims[0];
    let n = dims[1];
    let batch_dims = &dims[2..];
    Ok((m, n, batch_dims))
}

/// Validate that a tensor is square (m == n) and return `(n, batch_dims)`.
pub(crate) fn validate_square<T: LinalgScalar>(a: &Tensor<T>) -> Result<(usize, &[usize])> {
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    if m != n {
        return Err(Error::InvalidArgument(format!(
            "expected square matrix, got {}x{}",
            m, n
        )));
    }
    Ok((n, batch_dims))
}

/// Ensure a tensor is column-major contiguous. Returns the tensor as-is if
/// already contiguous, otherwise creates a contiguous copy.
pub(crate) fn ensure_col_major<T: LinalgScalar>(a: &Tensor<T>) -> Tensor<T> {
    if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous(MemoryOrder::ColumnMajor)
    }
}

/// Compute the total number of elements in batch dimensions.
pub(crate) fn batch_count(batch_dims: &[usize]) -> usize {
    batch_dims.iter().product::<usize>().max(1)
}

/// Extract the underlying buffer slice from a tensor. Returns an error if
/// the buffer is not CPU-accessible.
pub(crate) fn extract_contiguous_slice<T: LinalgScalar>(a: &Tensor<T>) -> Result<&[T]> {
    a.buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidArgument("tensor buffer is not a contiguous CPU slice".into()))
}
