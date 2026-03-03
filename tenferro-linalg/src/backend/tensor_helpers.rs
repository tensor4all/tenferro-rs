//! Shared tensor validation and helper utilities for backend implementations.
//!
//! This module provides common tensor operations used by CPU (and future GPU)
//! backend implementations: shape validation, contiguous packing, and output
//! tensor allocation.

use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::LinalgScalar;

/// Normalized RHS metadata for solve-style operations.
pub(crate) struct SolveRhsLayout {
    /// Number of right-hand sides.
    pub nrhs: usize,
    /// Output shape to preserve the original vector-vs-matrix rank.
    pub output_dims: Vec<usize>,
}

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

/// Validate solve RHS shape against a square matrix `(n, n, batch...)`.
///
/// Accepted RHS shapes:
/// - `(n, batch...)`
/// - `(n, nrhs, batch...)`
pub(crate) fn validate_solve_rhs_shape<T: LinalgScalar>(
    b: &Tensor<T>,
    n: usize,
    batch_dims: &[usize],
    op_name: &str,
) -> Result<SolveRhsLayout> {
    let dims = b.dims();
    if dims.len() == 1 + batch_dims.len() {
        if dims[0] != n {
            return Err(Error::InvalidArgument(format!(
                "{op_name} expects b dim[0] == n ({n}), got {}",
                dims[0]
            )));
        }
        if &dims[1..] != batch_dims {
            return Err(Error::InvalidArgument(format!(
                "{op_name} batch dims mismatch: expected {:?}, got {:?}",
                batch_dims,
                &dims[1..]
            )));
        }
        return Ok(SolveRhsLayout {
            nrhs: 1,
            output_dims: dims.to_vec(),
        });
    }

    if dims.len() == 2 + batch_dims.len() {
        if dims[0] != n {
            return Err(Error::InvalidArgument(format!(
                "{op_name} expects b dim[0] == n ({n}), got {}",
                dims[0]
            )));
        }
        if dims[1] == 0 {
            return Err(Error::InvalidArgument(format!(
                "{op_name} requires b dim[1] (nrhs) > 0"
            )));
        }
        if &dims[2..] != batch_dims {
            return Err(Error::InvalidArgument(format!(
                "{op_name} batch dims mismatch: expected {:?}, got {:?}",
                batch_dims,
                &dims[2..]
            )));
        }
        return Ok(SolveRhsLayout {
            nrhs: dims[1],
            output_dims: dims.to_vec(),
        });
    }

    Err(Error::InvalidArgument(format!(
        "{op_name} expects b shape (n, *) or (n, k, *), got {:?}",
        dims
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::MemoryOrder;

    fn make(data: &[f64], dims: &[usize]) -> Tensor<f64> {
        Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
    }

    #[test]
    fn validate_matrix_shape_2d() {
        let a = make(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let (m, n, batch) = validate_matrix_shape(&a).unwrap();
        assert_eq!((m, n), (2, 3));
        assert!(batch.is_empty());
    }

    #[test]
    fn validate_matrix_shape_1d_fails() {
        let a = make(&[1.0, 2.0], &[2]);
        assert!(validate_matrix_shape(&a).is_err());
    }

    #[test]
    fn validate_square_ok() {
        let a = make(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let (n, batch) = validate_square(&a).unwrap();
        assert_eq!(n, 2);
        assert!(batch.is_empty());
    }

    #[test]
    fn validate_square_nonsquare_fails() {
        let a = make(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert!(validate_square(&a).is_err());
    }

    #[test]
    fn batch_count_empty() {
        assert_eq!(batch_count(&[]), 1);
    }

    #[test]
    fn batch_count_nonempty() {
        assert_eq!(batch_count(&[2, 3]), 6);
    }

    #[test]
    fn ensure_col_major_contiguous() {
        let a = make(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = ensure_col_major(&a);
        assert!(b.is_contiguous());
    }

    #[test]
    fn extract_contiguous_slice_ok() {
        let a = make(&[1.0, 2.0], &[2]);
        let s = extract_contiguous_slice(&a).unwrap();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn validate_solve_rhs_shape_vector() {
        let b = make(&[1.0, 2.0], &[2]);
        let layout = validate_solve_rhs_shape(&b, 2, &[], "solve").unwrap();
        assert_eq!(layout.nrhs, 1);
        assert_eq!(layout.output_dims, vec![2]);
    }

    #[test]
    fn validate_solve_rhs_shape_scalar_fails() {
        let b = make(&[1.0], &[]);
        assert!(validate_solve_rhs_shape(&b, 2, &[], "solve").is_err());
    }
}
