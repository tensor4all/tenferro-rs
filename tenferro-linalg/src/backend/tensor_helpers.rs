//! Thin re-export layer for shared tensor helper substrate.
//!
//! The actual broadcast/solve/layout helpers live in `tenferro-linalg-prims`
//! so the tensor-level linalg crate can reuse the same hidden substrate
//! without maintaining a local copy.

#[doc(hidden)]
pub(crate) use tenferro_linalg_prims::backend::{
    batch_count, ensure_col_major, extract_contiguous_slice, materialize_broadcasted_batches,
    validate_matrix_shape, validate_solve_rhs_shape, validate_square, zero_trailing_by_counts,
    BroadcastBatchIndexer,
};

use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

#[doc(hidden)]
pub(crate) fn backend_info_to_vec(info: &Tensor<i32>) -> Result<Vec<i32>> {
    let cpu = info.to_memory_space_async(LogicalMemorySpace::MainMemory)?;
    let contiguous = cpu.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    let slice = contiguous.buffer().as_slice().ok_or_else(|| {
        Error::InvalidArgument("backend info tensor is not CPU accessible".into())
    })?;
    Ok(slice[offset..offset + len].to_vec())
}

#[doc(hidden)]
pub(crate) fn backend_pivots_to_usize(pivots: &Tensor<i32>) -> Result<Vec<usize>> {
    let cpu = pivots.to_memory_space_async(LogicalMemorySpace::MainMemory)?;
    let contiguous = cpu.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    let dims = contiguous.dims();
    let pivot_len = dims.first().copied().unwrap_or(0);
    let slice = contiguous.buffer().as_slice().ok_or_else(|| {
        Error::InvalidArgument("backend LU pivot tensor is not CPU accessible".into())
    })?;
    let flat = &slice[offset..offset + len];

    if pivot_len == 0 {
        return Ok(Vec::new());
    }
    if flat.len() % pivot_len != 0 {
        return Err(Error::InvalidArgument(format!(
            "backend LU pivot tensor length {} is not divisible by pivot length {pivot_len}",
            flat.len()
        )));
    }

    let mut out = Vec::with_capacity(flat.len());
    for step_pivots in flat.chunks(pivot_len) {
        let mut perm: Vec<usize> = (0..pivot_len).collect();
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
