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
use crate::{Buffer, Tensor, TensorRank, TensorRead, TensorView, TypedTensor, TypedTensorView};

pub use affinity::{available_parallelism, process_cpu_affinity_count};
pub use backend::{CpuBackend, CpuBackendKind};
pub use context::CpuContext;
pub use elementwise::{
    abs, add, clamp, compare, conj, div, maximum, minimum, mul, neg, select, sign,
};
pub use indexing::{dynamic_slice, dynamic_update_slice, gather, pad, scatter};
pub use reduction::{reduce_max, reduce_min, reduce_prod, reduce_sum};
pub use structural::{
    broadcast_in_dim, convert, embed_diagonal, extract_diagonal, reshape, transpose, tril, triu,
};

pub(crate) fn cpu_backend_buffer_error(op: &'static str) -> crate::Error {
    crate::Error::backend_failure(
        op,
        "CPU backend received backend buffer; download to host before CPU execution",
    )
}

pub(crate) fn typed_host_data<'a, T>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> crate::Result<&'a [T]> {
    match &tensor.buffer {
        Buffer::Host(data) => Ok(data),
        Buffer::Backend(_) => Err(cpu_backend_buffer_error(op)),
    }
}

pub(crate) fn typed_view<'a, T: Copy>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> crate::Result<StridedView<'a, T>> {
    match &tensor.buffer {
        Buffer::Host(data) => {
            let strides = col_major_strides(tensor.shape());
            StridedView::new(data, tensor.shape(), &strides, 0)
                .map_err(|err| crate::Error::backend_failure(op, err))
        }
        Buffer::Backend(_) => Err(cpu_backend_buffer_error(op)),
    }
}

pub(crate) fn typed_view_from_view<'a, T: Copy + 'static, R: TensorRank>(
    op: &'static str,
    view: &TypedTensorView<'a, T, R>,
) -> crate::Result<StridedView<'a, T>> {
    if view.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    StridedView::new(
        view.as_physical_slice(),
        view.shape(),
        view.strides(),
        view.offset(),
    )
    .map_err(|err| crate::Error::backend_failure(op, err))
}

pub(crate) fn materialize_tensor_read(
    op: &'static str,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(tensor) => clone_host_tensor_read(op, tensor),
        TensorRead::View(view) => materialize_tensor_view(op, view),
    }
}

fn clone_host_tensor_read(op: &'static str, tensor: &Tensor) -> crate::Result<Tensor> {
    macro_rules! clone_host {
        ($variant:ident, $tensor:expr) => {{
            typed_host_data(op, $tensor)?;
            Ok(Tensor::$variant($tensor.clone()))
        }};
    }

    match tensor {
        Tensor::F32(tensor) => clone_host!(F32, tensor),
        Tensor::F64(tensor) => clone_host!(F64, tensor),
        Tensor::I32(tensor) => clone_host!(I32, tensor),
        Tensor::I64(tensor) => clone_host!(I64, tensor),
        Tensor::Bool(tensor) => clone_host!(Bool, tensor),
        Tensor::C32(tensor) => clone_host!(C32, tensor),
        Tensor::C64(tensor) => clone_host!(C64, tensor),
    }
}

fn materialize_tensor_view(op: &'static str, view: TensorView<'_>) -> crate::Result<Tensor> {
    macro_rules! materialize {
        ($variant:ident, $view:expr) => {{
            if $view.backend_buffer().is_some() {
                return Err(cpu_backend_buffer_error(op));
            }
            Ok(Tensor::$variant($view.to_contiguous()?))
        }};
    }

    match view {
        TensorView::F32(view) => materialize!(F32, view),
        TensorView::F64(view) => materialize!(F64, view),
        TensorView::I32(view) => materialize!(I32, view),
        TensorView::I64(view) => materialize!(I64, view),
        TensorView::Bool(view) => materialize!(Bool, view),
        TensorView::C32(view) => materialize!(C32, view),
        TensorView::C64(view) => materialize!(C64, view),
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
