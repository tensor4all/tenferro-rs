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

#[cfg(test)]
mod tests;
