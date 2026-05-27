pub mod affinity;
pub mod analytic;
pub mod backend;
pub mod context;
pub mod elementwise;
mod exec_session;
pub mod gemm;
pub mod indexing;
mod indexing_alloc;
pub mod reduction;
pub mod structural;

use strided_kernel::{col_major_strides, StridedArray, StridedView};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::{Buffer, TypedTensor};

pub use affinity::{available_parallelism, process_cpu_affinity_count};
pub use backend::CpuBackend;
pub use context::CpuContext;
pub use elementwise::{
    abs, add, clamp, compare, conj, div, maximum, minimum, mul, neg, select, sign,
};
pub use indexing::{dynamic_slice, dynamic_update_slice, gather, pad, scatter};
pub use reduction::{reduce_max, reduce_min, reduce_prod, reduce_sum};
pub use structural::{
    broadcast_in_dim, convert, embed_diagonal, extract_diagonal, reshape, transpose, tril, triu,
};

pub(crate) fn typed_view<T: Copy>(tensor: &TypedTensor<T>) -> StridedView<'_, T> {
    match &tensor.buffer {
        Buffer::Host(data) => {
            let strides = col_major_strides(&tensor.shape);
            StridedView::new(data, &tensor.shape, &strides, 0).expect("contiguous host tensor")
        }
        Buffer::Backend(_) => todo!("typed_view for backend buffers"),
    }
}

/// Create an output array WITHOUT initializing element values.
///
/// # Safety
/// Caller must write every element before reading. The returned array
/// contains uninitialized data.
#[allow(clippy::uninit_vec)]
pub(crate) unsafe fn typed_array_uninit<T>(shape: &[usize]) -> StridedArray<T> {
    let total: usize = shape.iter().product();
    let strides = col_major_strides(shape);
    let mut data = Vec::with_capacity(total);
    // SAFETY: caller guarantees every element is written before any read.
    unsafe { data.set_len(total) };
    StridedArray::from_parts(data, shape, &strides, 0).expect("column-major output array")
}

/// Create an output array from the CPU buffer pool WITHOUT initializing values.
///
/// # Safety
/// Caller must write every element before reading. The returned array contains
/// uninitialized data acquired from `buffers`.
#[allow(clippy::uninit_vec)]
pub(crate) unsafe fn typed_array_uninit_from_pool<T>(
    buffers: &mut BufferPool,
    shape: &[usize],
) -> StridedArray<T>
where
    T: PoolScalar,
{
    let total: usize = shape.iter().product();
    let strides = col_major_strides(shape);
    // SAFETY: caller guarantees every element is written before any read.
    let data = unsafe { T::pool_acquire(buffers, total) };
    StridedArray::from_parts(data, shape, &strides, 0).expect("column-major output array")
}

pub(crate) fn tensor_from_array<T: Clone>(array: StridedArray<T>) -> TypedTensor<T> {
    TypedTensor::from_vec_col_major(array.dims().to_vec(), array.into_data())
}
