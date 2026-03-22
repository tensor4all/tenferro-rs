//! Shared tensor validation and helper utilities for backend implementations.

use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::LinalgScalar;

pub(crate) struct SolveRhsLayout {
    pub nrhs: usize,
    pub output_dims: Vec<usize>,
}

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

pub(crate) fn ensure_col_major<T: LinalgScalar>(a: &Tensor<T>) -> Tensor<T> {
    a.contiguous(MemoryOrder::ColumnMajor)
}

pub(crate) fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

pub(crate) fn extract_contiguous_slice<T: LinalgScalar>(a: &Tensor<T>) -> Result<&[T]> {
    a.buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidArgument("tensor buffer is not a contiguous CPU slice".into()))
}

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
