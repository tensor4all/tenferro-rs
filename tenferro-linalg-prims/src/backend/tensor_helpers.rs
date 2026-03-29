//! Shared tensor validation and helper utilities for backend implementations.

#[cfg(any(feature = "cuda", test))]
use tenferro_algebra::Conjugate;
#[cfg(feature = "cuda")]
use tenferro_algebra::Scalar;
#[cfg(feature = "cuda")]
use tenferro_device::LogicalMemorySpace;
use tenferro_device::{Error, Result};
#[cfg(any(feature = "cuda", test))]
use tenferro_prims::TensorResolveConjContextFor;
use tenferro_tensor::{KeepCountScalar, MemoryOrder, Tensor};

pub use tenferro_device::{broadcast_batch_dims, BroadcastBatchIndexer};

use crate::LinalgScalar;

#[doc(hidden)]
pub struct SolveRhsLayout {
    pub nrhs: usize,
    pub output_dims: Vec<usize>,
    pub output_batch_dims: Vec<usize>,
    pub structural_rank: usize,
    pub rhs_batch_indexer: BroadcastBatchIndexer,
}

#[doc(hidden)]
pub fn materialize_broadcasted_batches<T: LinalgScalar>(
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
    materialize_broadcasted_batches_impl(src, structural_rank, batch_indexer, op_name, arg_name)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn materialize_broadcasted_batches_resolving_conj<T, C>(
    ctx: &mut C,
    src: &Tensor<T>,
    structural_rank: usize,
    batch_indexer: &BroadcastBatchIndexer,
    op_name: &str,
    arg_name: &str,
) -> Result<Tensor<T>>
where
    T: LinalgScalar + Conjugate,
    C: TensorResolveConjContextFor<T>,
{
    let resolved = if src.is_conjugated() {
        <C as TensorResolveConjContextFor<T>>::resolve_conj(ctx, src)
    } else {
        src.clone()
    };
    materialize_broadcasted_batches_impl(
        &resolved,
        structural_rank,
        batch_indexer,
        op_name,
        arg_name,
    )
}

fn materialize_broadcasted_batches_impl<T: LinalgScalar>(
    src: &Tensor<T>,
    structural_rank: usize,
    batch_indexer: &BroadcastBatchIndexer,
    op_name: &str,
    arg_name: &str,
) -> Result<Tensor<T>> {
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

#[doc(hidden)]
pub fn materialize_broadcasted_pivot_batches(
    pivots: &Tensor<i32>,
    step_count: usize,
    source_batch_dims: &[usize],
    output_batch_dims: &[usize],
    op_name: &str,
) -> Result<Tensor<i32>> {
    validate_lu_pivot_shape(pivots, step_count, source_batch_dims, op_name)?;
    let batch_indexer =
        BroadcastBatchIndexer::new(source_batch_dims, output_batch_dims, op_name, "pivots")?;
    if batch_indexer.is_identity() {
        return Ok(pivots.contiguous(MemoryOrder::ColumnMajor));
    }

    let mut expanded = pivots.clone();
    for _ in 0..(output_batch_dims.len() - source_batch_dims.len()) {
        expanded = expanded.unsqueeze(1)?;
    }
    let mut target_dims = vec![step_count];
    target_dims.extend_from_slice(output_batch_dims);
    Ok(expanded
        .broadcast(&target_dims)?
        .contiguous(MemoryOrder::ColumnMajor))
}

#[doc(hidden)]
pub fn validate_lu_pivot_shape(
    pivots: &Tensor<i32>,
    step_count: usize,
    batch_dims: &[usize],
    op_name: &str,
) -> Result<()> {
    let dims = pivots.dims();
    if dims.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects pivot tensor shape ({step_count}, *batch), got scalar pivots"
        )));
    }
    if dims[0] != step_count {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects pivots dim[0] == {step_count}, got {}",
            dims[0]
        )));
    }
    if dims[1..] != *batch_dims {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects pivots batch dims {:?} to match factor batch dims {:?}",
            &dims[1..],
            batch_dims
        )));
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_matrix_shape<T: LinalgScalar>(a: &Tensor<T>) -> Result<(usize, usize, &[usize])> {
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

#[doc(hidden)]
pub fn validate_square<T: LinalgScalar>(a: &Tensor<T>) -> Result<(usize, &[usize])> {
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    if m != n {
        return Err(Error::InvalidArgument(format!(
            "expected square matrix, got {}x{}",
            m, n
        )));
    }
    Ok((n, batch_dims))
}

#[doc(hidden)]
pub fn ensure_col_major<T: LinalgScalar>(a: &Tensor<T>) -> Tensor<T> {
    a.contiguous(MemoryOrder::ColumnMajor)
}

#[doc(hidden)]
pub fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

#[doc(hidden)]
pub fn extract_contiguous_slice<T: tenferro_algebra::Scalar>(a: &Tensor<T>) -> Result<&[T]> {
    a.buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidArgument("tensor buffer is not a contiguous CPU slice".into()))
}

#[doc(hidden)]
pub fn validate_solve_rhs_shape<T: LinalgScalar>(
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
#[doc(hidden)]
pub fn zero_trailing_by_counts<T, R>(
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

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub(crate) fn tensor_from_data_on_space<T: Scalar>(
    data: Vec<T>,
    dims: &[usize],
    memory_space: LogicalMemorySpace,
) -> Result<Tensor<T>> {
    let tensor = Tensor::from_slice(&data, dims, MemoryOrder::ColumnMajor)?;
    if tensor.logical_memory_space() == memory_space {
        Ok(tensor)
    } else {
        tensor.to_memory_space_async(memory_space)
    }
}
