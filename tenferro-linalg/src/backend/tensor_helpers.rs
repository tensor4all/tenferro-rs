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

use tenferro_device::{LogicalMemorySpace, Result};
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

#[cfg(test)]
mod tests;
