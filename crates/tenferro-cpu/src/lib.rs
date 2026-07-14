//! CPU backend, kernels, provider selection, and CPU resource pools.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_cpu::{add, CpuBackend};
//! use tenferro_tensor::{Tensor, TensorBackend, TensorElementwise};
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
//! let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
//! let c = backend.add(&a, &b)?;
//! assert_eq!(c.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
//! let direct = add(&a, &b)?;
//! assert_eq!(direct.shape(), &[2]);
//! # Ok::<(), tenferro_tensor::Error>(())
//! ```

#[cfg(not(any(feature = "cpu-faer", feature = "cpu-blas")))]
compile_error!("enable at least one CPU backend: cpu-faer or cpu-blas");

#[cfg(all(feature = "provider-inject", not(feature = "cpu-blas")))]
compile_error!("provider-inject requires cpu-blas");

#[cfg(any(
    all(feature = "blas-openblas", feature = "blas-accelerate"),
    all(feature = "blas-openblas", feature = "blas-mkl"),
    all(feature = "blas-accelerate", feature = "blas-mkl"),
))]
compile_error!(
    "enable at most one explicit BLAS provider feature: blas-openblas, blas-accelerate, or blas-mkl"
);

#[cfg(all(
    feature = "provider-inject",
    any(
        feature = "blas-openblas",
        feature = "blas-accelerate",
        feature = "blas-mkl"
    )
))]
compile_error!("provider-inject cannot be combined with explicit BLAS provider features");

pub mod affinity;
mod analytic;
mod arbiter;
pub mod backend;
mod buffer_pool;
mod capability;
pub mod context;
mod elementwise;
mod engine;
mod exec_session;
mod gemm;
mod indexing;
mod indexing_alloc;
#[cfg(feature = "provider-inject")]
pub mod inject;
mod placement;
mod reduction;
mod structural;
mod topology;

use strided_kernel::{col_major_strides as kernel_col_major_strides, StridedArray, StridedView};

use crate::buffer_pool::{BufferPool, PoolScalar};
pub(crate) use tenferro_tensor::*;

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;
#[cfg(feature = "provider-inject")]
extern crate lapack_inject as _;
#[cfg(feature = "provider-src")]
extern crate lapack_src as _;

pub use affinity::{available_parallelism, process_cpu_affinity, process_cpu_affinity_count};
pub use analytic::pow;
pub use backend::{CpuBackend, CpuBackendKind, CpuExecutionInfo};
pub use buffer_pool::BufferPoolStats;
pub use capability::cpu_capabilities;
pub use context::{CpuContext, CpuContextError};
pub use elementwise::{
    abs, add, clamp, compare, conj, div, maximum, minimum, mul, neg, rem, select, sign, sub,
};
pub use indexing::{dynamic_slice, dynamic_update_slice, gather, pad, scatter};
pub use placement::{CpuPlacement, CpuPlacementError, ResolvedCpuPlacement};
pub use reduction::{reduce_max, reduce_min, reduce_prod, reduce_sum};
pub use structural::{
    broadcast_in_dim, convert, embed_diagonal, extract_diagonal, reshape, transpose, tril, triu,
};
pub use topology::{
    discover_cpu_topology, CpuId, CpuNode, CpuSet, CpuSetError, CpuTopology, CpuTopologyError,
    NumaNodeId,
};

/// Owner-scoped CPU scratch-pool API for operation-family crates.
///
/// This module is not an application-facing tensor API. It exists so
/// operation crates that implement CPU kernels can share `CpuBackend`'s
/// allocation pool without exposing the pool as a general public contract.
#[doc(hidden)]
pub mod linalg_interop {
    pub use crate::buffer_pool::{BufferPool, PoolScalar};
}

pub(crate) fn cpu_backend_buffer_error(op: &'static str) -> crate::Error {
    crate::Error::backend_failure(
        op,
        "CPU backend received backend buffer; download to host before CPU execution",
    )
}

pub(crate) trait ConjElem {
    fn conj_elem(self) -> Self;
}

impl ConjElem for f32 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for f64 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for num_complex::Complex32 {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

impl ConjElem for num_complex::Complex64 {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

pub(crate) fn typed_host_data<'a, T>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> crate::Result<&'a [T]> {
    match tensor.buffer() {
        Buffer::Host(data) => Ok(data.as_slice()),
        Buffer::Backend(_) => Err(cpu_backend_buffer_error(op)),
    }
}

pub(crate) fn typed_view<'a, T: Copy>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> crate::Result<StridedView<'a, T>> {
    match tensor.buffer() {
        Buffer::Host(data) => {
            let strides = kernel_col_major_strides(tensor.shape());
            StridedView::new(data.as_slice(), tensor.shape(), &strides, 0)
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
        view.host_storage()?,
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
#[cfg(test)]
pub(crate) unsafe fn typed_array_uninit<T>(shape: &[usize]) -> StridedArray<T> {
    let total: usize = shape.iter().product();
    let strides = kernel_col_major_strides(shape);
    let mut data = Vec::with_capacity(total);
    // SAFETY: test-only helper is used for outputs whose elements are fully overwritten.
    unsafe { data.set_len(total) };
    // Invariant: `kernel_col_major_strides(shape)` and `total` describe the
    // compact column-major array for this validated test output shape.
    StridedArray::from_parts(data, shape, &strides, 0).expect("column-major output array")
}

/// Create an output array from the CPU buffer pool WITHOUT initializing values.
///
/// # Safety
/// Caller must write every element before reading. The returned array contains
/// uninitialized or stale data acquired from `buffers`.
pub(crate) unsafe fn typed_array_uninit_from_pool<T>(
    buffers: &mut BufferPool,
    shape: &[usize],
) -> crate::Result<StridedArray<T>>
where
    T: PoolScalar,
{
    let total = tenferro_tensor::validate::checked_shape_product(
        "typed_array_uninit_from_pool",
        "shape",
        shape,
    )?;
    let strides = kernel_col_major_strides(shape);
    // SAFETY: callers use this only for operation outputs that fully overwrite every element.
    let data = unsafe { T::pool_acquire(buffers, total) };
    // Invariant: callers pass validated tensor-derived or prechecked output
    // shapes, and `strides` is their compact column-major layout.
    StridedArray::from_parts(data, shape, &strides, 0)
        .map_err(|err| crate::Error::backend_failure("typed_array_uninit_from_pool", err))
}

pub(crate) fn tensor_from_array<T: Clone>(array: StridedArray<T>) -> TypedTensor<T> {
    // Invariant: `StridedArray` owns data whose length matches its validated dimensions.
    TypedTensor::from_vec_col_major(array.dims().to_vec(), array.into_data())
        .expect("strided array dimensions match owned data length")
}

pub(crate) fn default_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
    }
}

pub(crate) fn flat_to_multi(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    assert_eq!(shape.len(), out.len());
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            out[axis] = 0;
        } else {
            out[axis] = flat % dim;
            flat /= dim;
        }
    }
}

#[cfg(test)]
mod tests;
