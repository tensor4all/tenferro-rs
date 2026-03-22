//! Shared tensor validation and helper utilities for backend implementations.
//!
//! This module provides common tensor operations used by CPU (and future GPU)
//! backend implementations: shape validation, contiguous packing, and output
//! tensor allocation.

use tenferro_device::{Error, Result};
use tenferro_tensor::{KeepCountScalar, MemoryOrder, Tensor};

use crate::LinalgScalar;

/// Normalized RHS metadata for solve-style operations.
pub(crate) struct SolveRhsLayout {
    /// Number of right-hand sides.
    pub nrhs: usize,
    /// Output shape to preserve the original vector-vs-matrix rank.
    pub output_dims: Vec<usize>,
    /// Output batch shape after broadcasting `A` and `b`.
    pub output_batch_dims: Vec<usize>,
    /// Number of leading structural dimensions in the RHS tensor.
    pub structural_rank: usize,
    /// Broadcast mapping from output batches back to the source RHS batches.
    pub rhs_batch_indexer: BroadcastBatchIndexer,
}

#[allow(dead_code)]
pub(crate) struct BroadcastBatchIndexer {
    output_batch_dims: Vec<usize>,
    normalized_source_batch_dims: Vec<usize>,
    source_strides: Vec<usize>,
    identity: bool,
}

impl BroadcastBatchIndexer {
    pub(crate) fn new(
        source_batch_dims: &[usize],
        output_batch_dims: &[usize],
        op_name: &str,
        arg_name: &str,
    ) -> Result<Self> {
        if source_batch_dims.len() > output_batch_dims.len() {
            return Err(Error::InvalidArgument(format!(
                "{op_name} {arg_name} batch rank {} exceeds target batch rank {}",
                source_batch_dims.len(),
                output_batch_dims.len()
            )));
        }

        let missing = output_batch_dims.len() - source_batch_dims.len();
        let mut normalized_source_batch_dims = vec![1; output_batch_dims.len()];
        normalized_source_batch_dims[missing..].copy_from_slice(source_batch_dims);

        for (axis, (&source_dim, &output_dim)) in normalized_source_batch_dims
            .iter()
            .zip(output_batch_dims.iter())
            .enumerate()
        {
            if source_dim != 1 && source_dim != output_dim {
                return Err(Error::InvalidArgument(format!(
                    "{op_name} {arg_name} batch dims are not broadcastable to {:?}: source axis {axis} has {source_dim}, target has {output_dim}",
                    output_batch_dims
                )));
            }
        }

        let identity = normalized_source_batch_dims == output_batch_dims;
        Ok(Self {
            output_batch_dims: output_batch_dims.to_vec(),
            source_strides: col_major_batch_strides(&normalized_source_batch_dims),
            normalized_source_batch_dims,
            identity,
        })
    }

    pub(crate) fn output_batch_dims(&self) -> &[usize] {
        &self.output_batch_dims
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.identity
    }

    #[allow(dead_code)]
    pub(crate) fn source_linear_batch_index(&self, mut output_linear_batch_index: usize) -> usize {
        if self.output_batch_dims.is_empty() {
            return 0;
        }

        let mut source_linear_batch_index = 0usize;
        for axis in 0..self.output_batch_dims.len() {
            let output_dim = self.output_batch_dims[axis];
            let coord = output_linear_batch_index % output_dim;
            output_linear_batch_index /= output_dim;
            if self.normalized_source_batch_dims[axis] != 1 {
                source_linear_batch_index += coord * self.source_strides[axis];
            }
        }
        source_linear_batch_index
    }
}

fn col_major_batch_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    strides[0] = 1;
    for axis in 1..dims.len() {
        strides[axis] = strides[axis - 1] * dims[axis - 1];
    }
    strides
}

pub(crate) fn broadcast_batch_dims(
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    op_name: &str,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<Vec<usize>> {
    let rank = lhs_batch_dims.len().max(rhs_batch_dims.len());
    let lhs_pad = rank - lhs_batch_dims.len();
    let rhs_pad = rank - rhs_batch_dims.len();
    let mut output_batch_dims = Vec::with_capacity(rank);

    for axis in 0..rank {
        let lhs_dim = if axis < lhs_pad {
            1
        } else {
            lhs_batch_dims[axis - lhs_pad]
        };
        let rhs_dim = if axis < rhs_pad {
            1
        } else {
            rhs_batch_dims[axis - rhs_pad]
        };
        if lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1 {
            output_batch_dims.push(lhs_dim.max(rhs_dim));
        } else {
            return Err(Error::InvalidArgument(format!(
                "{op_name} batch dims are not broadcastable: {lhs_name} has {:?}, {rhs_name} has {:?}",
                lhs_batch_dims, rhs_batch_dims
            )));
        }
    }

    Ok(output_batch_dims)
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

/// Ensure a tensor is column-major contiguous.
pub(crate) fn ensure_col_major<T: LinalgScalar>(a: &Tensor<T>) -> Tensor<T> {
    a.contiguous(MemoryOrder::ColumnMajor)
}

/// Compute the total number of elements in batch dimensions.
pub(crate) fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

/// Extract the underlying buffer slice from a tensor. Returns an error if
/// the buffer is not CPU-accessible.
pub(crate) fn extract_contiguous_slice<T: LinalgScalar>(a: &Tensor<T>) -> Result<&[T]> {
    a.buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidArgument("tensor buffer is not a contiguous CPU slice".into()))
}

/// Thin wrapper over the tensor-level keep-count trailing zero-fill helper.
pub(crate) fn zero_trailing_by_counts<T, R>(
    input: &Tensor<T>,
    keep_counts: &Tensor<R>,
    axis: usize,
    structural_rank: usize,
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    R: KeepCountScalar,
{
    input.zero_trailing_by_counts(keep_counts, axis, structural_rank)
}

pub(crate) fn materialize_broadcasted_batches<T: LinalgScalar>(
    src: &Tensor<T>,
    structural_rank: usize,
    batch_indexer: &BroadcastBatchIndexer,
    op_name: &str,
    arg_name: &str,
) -> Result<Tensor<T>> {
    if src.is_conjugated() {
        return Err(Error::InvalidArgument(format!(
            "{op_name} requires resolved (non-conjugated) {arg_name}"
        )));
    }
    if structural_rank > src.ndim() {
        return Err(Error::InvalidArgument(format!(
            "{op_name} structural rank {structural_rank} exceeds {arg_name} ndim {}",
            src.ndim()
        )));
    }
    if batch_indexer.is_identity() {
        return Ok(ensure_col_major(src));
    }

    let source_batch_rank = src.ndim() - structural_rank;
    if source_batch_rank > batch_indexer.output_batch_dims().len() {
        return Err(Error::InvalidArgument(format!(
            "{op_name} {arg_name} batch rank {source_batch_rank} exceeds target batch rank {}",
            batch_indexer.output_batch_dims().len()
        )));
    }

    let mut expanded = src.clone();
    for _ in 0..(batch_indexer.output_batch_dims().len() - source_batch_rank) {
        expanded = expanded.unsqueeze(structural_rank as isize)?;
    }

    let mut target_dims = expanded.dims()[..structural_rank].to_vec();
    target_dims.extend_from_slice(batch_indexer.output_batch_dims());
    let broadcasted = expanded.broadcast(&target_dims)?;
    Ok(broadcasted.contiguous(MemoryOrder::ColumnMajor))
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
    if dims.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects b shape (n, *) or (n, k, *), got {:?}",
            dims
        )));
    }

    if dims.len() <= 1 + batch_dims.len() {
        if dims[0] != n {
            return Err(Error::InvalidArgument(format!(
                "{op_name} expects b dim[0] == n ({n}), got {}",
                dims[0]
            )));
        }
        let output_batch_dims = broadcast_batch_dims(batch_dims, &dims[1..], op_name, "a", "b")?;
        let rhs_batch_indexer =
            BroadcastBatchIndexer::new(&dims[1..], &output_batch_dims, op_name, "b")?;
        let mut output_dims = vec![n];
        output_dims.extend_from_slice(&output_batch_dims);
        return Ok(SolveRhsLayout {
            nrhs: 1,
            output_dims,
            output_batch_dims,
            structural_rank: 1,
            rhs_batch_indexer,
        });
    }

    if dims.len() <= 2 + batch_dims.len() {
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
        let output_batch_dims = broadcast_batch_dims(batch_dims, &dims[2..], op_name, "a", "b")?;
        let rhs_batch_indexer =
            BroadcastBatchIndexer::new(&dims[2..], &output_batch_dims, op_name, "b")?;
        let mut output_dims = vec![n, dims[1]];
        output_dims.extend_from_slice(&output_batch_dims);
        return Ok(SolveRhsLayout {
            nrhs: dims[1],
            output_dims,
            output_batch_dims,
            structural_rank: 2,
            rhs_batch_indexer,
        });
    }

    Err(Error::InvalidArgument(format!(
        "{op_name} expects b shape (n, *) or (n, k, *), got {:?}",
        dims
    )))
}

#[cfg(test)]
mod tests;
