//! Thin re-export layer for shared tensor helper substrate.
//!
//! The actual broadcast/solve/layout helpers live in `tenferro-linalg-prims`
//! so the tensor-level linalg crate can reuse the same hidden substrate
//! without maintaining a local copy.

#[doc(hidden)]
pub(crate) use tenferro_linalg_prims::backend::{
    batch_count, ensure_col_major, extract_contiguous_slice, materialize_broadcasted_batches,
    materialize_broadcasted_pivot_batches, validate_lu_pivot_shape, validate_matrix_shape,
    validate_solve_rhs_shape, validate_square, zero_trailing_by_counts, BroadcastBatchIndexer,
};

use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

#[doc(hidden)]
pub(crate) fn info_tensor_from_vec_on_space(
    info: Vec<i32>,
    batch_dims: &[usize],
    memory_space: LogicalMemorySpace,
) -> Result<Tensor<i32>> {
    let shape = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };
    let tensor = Tensor::from_slice(&info, &shape, tenferro_tensor::MemoryOrder::ColumnMajor)?;
    if memory_space == LogicalMemorySpace::MainMemory {
        Ok(tensor)
    } else {
        tensor.to_memory_space_async(memory_space)
    }
}

#[doc(hidden)]
pub(crate) fn backend_pivots_to_forward_perm(
    pivots: &Tensor<i32>,
    row_count: usize,
) -> Result<Vec<usize>> {
    let cpu = pivots.to_memory_space_async(LogicalMemorySpace::MainMemory)?;
    let contiguous = cpu.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    let dims = contiguous.dims();
    let step_count = dims.first().copied().unwrap_or(0);
    let batch_count = if dims.len() <= 1 {
        1
    } else {
        dims[1..].iter().product()
    };
    let slice = contiguous.buffer().as_slice().ok_or_else(|| {
        Error::InvalidArgument("backend LU pivot tensor is not CPU accessible".into())
    })?;
    let flat = &slice[offset..offset + len];

    if step_count == 0 {
        let identity: Vec<usize> = (0..row_count).collect();
        let mut out = Vec::with_capacity(row_count * batch_count);
        for _ in 0..batch_count {
            out.extend_from_slice(&identity);
        }
        return Ok(out);
    }
    if flat.len() % step_count != 0 {
        return Err(Error::InvalidArgument(format!(
            "backend LU pivot tensor length {} is not divisible by pivot length {step_count}",
            flat.len(),
        )));
    }

    let actual_batches = flat.len() / step_count;
    if actual_batches != batch_count {
        return Err(Error::InvalidArgument(format!(
            "backend LU pivot tensor batch count {actual_batches} does not match expected {batch_count}",
        )));
    }

    let mut out = Vec::with_capacity(flat.len());
    for step_pivots in flat.chunks(step_count) {
        let mut perm: Vec<usize> = (0..row_count).collect();
        for (i, &pivot) in step_pivots.iter().enumerate() {
            if pivot <= 0 {
                return Err(Error::InvalidArgument(format!(
                    "backend LU pivot {pivot} is not 1-indexed positive"
                )));
            }
            let j = usize::try_from(pivot - 1).map_err(|_| {
                Error::InvalidArgument(format!(
                    "backend LU pivot {pivot} underflowed during usize conversion"
                ))
            })?;
            if j < i || j >= perm.len() {
                return Err(Error::InvalidArgument(format!(
                    "backend LU pivot {pivot} is invalid for step {i} and len {}",
                    perm.len()
                )));
            }
            perm.swap(i, j);
        }
        out.extend_from_slice(&perm);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
