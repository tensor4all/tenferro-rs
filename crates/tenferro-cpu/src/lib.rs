//! CPU backend, kernels, provider selection, and CPU resource pools.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::{Tensor, TensorBackend, TensorElementwise};
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
//! let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
//! let c = backend.add(&a, &b)?;
//! assert_eq!(c.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
//! # Ok::<(), tenferro_tensor::Error>(())
//! ```

// `provider-inject` unit tests deliberately omit the broad default-backend
// suite below because no fixture has registered its FFI symbols. That makes
// private helpers referenced only by the broad suite appear unused in this one
// test build; call-through coverage lives in the registered integration test.
#![cfg_attr(
    all(test, feature = "provider-inject"),
    allow(dead_code, unused_imports)
)]

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
mod affinity_policy;
mod analytic;
mod arbiter;
pub mod backend;
pub(crate) mod buffer_pool {
    pub use tenferro_internal_cpu_kernels::buffer_pool::*;
}
mod capability;
pub mod context;
// INVARIANT: Task 2 stages crate-private stack adapters here before Task 3 wires
// them into CpuContext.
#[allow(dead_code)]
mod domain_executor;
#[allow(dead_code)]
mod dot_runtime;
pub(crate) use tenferro_internal_cpu_kernels::elementwise;
pub(crate) use tenferro_internal_cpu_kernels::PooledUninitOutput;
mod engine;
mod exec_session;
mod gemm;
mod indexed_plan_cache;
mod indexing;
#[cfg(feature = "provider-inject")]
pub mod inject;
mod placement;
pub mod provider;
mod provider_capability;
mod reduction;
mod resource_domain;
mod runtime_adapter;
mod structural;
mod topology;

use std::mem::MaybeUninit;
use std::ptr::NonNull;
#[cfg(test)]
use strided_kernel::StridedArray;
use strided_kernel::{col_major_strides as kernel_col_major_strides, StridedView};

use crate::buffer_pool::BufferPool;
pub(crate) use tenferro_tensor::*;

pub(crate) fn erased_raw_strided_ref<'a>(
    dtype: strided_kernel::KernelDType,
    data: &'a [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_kernel::Result<strided_kernel::ErasedRawStridedRef<'a>> {
    let data_ptr = NonNull::new(data.as_ptr().cast_mut()).unwrap_or_else(NonNull::dangling);
    // SAFETY: callers derive `data` from initialized typed host storage and
    // keep that storage alive for the returned descriptor lifetime.
    unsafe {
        strided_kernel::ErasedRawStridedRef::from_raw_parts(
            dtype,
            data_ptr,
            data.len(),
            dims,
            strides,
            offset,
        )
    }
}

pub(crate) fn erased_raw_strided_mut<'a>(
    dtype: strided_kernel::KernelDType,
    data: &'a mut [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_kernel::Result<strided_kernel::ErasedRawStridedMut<'a>> {
    let data_ptr = NonNull::new(data.as_mut_ptr()).unwrap_or_else(NonNull::dangling);
    // SAFETY: callers derive `data` from a uniquely borrowed initialized host
    // destination and retain that borrow for the returned descriptor lifetime.
    unsafe {
        strided_kernel::ErasedRawStridedMut::from_raw_parts(
            dtype,
            data_ptr,
            data.len(),
            dims,
            strides,
            offset,
        )
    }
}

pub(crate) fn erased_raw_strided_uninit_mut<'a>(
    dtype: strided_kernel::KernelDType,
    data: &'a mut [MaybeUninit<u8>],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_kernel::Result<strided_kernel::ErasedRawStridedUninitMut<'a>> {
    let data_ptr = NonNull::new(data.as_mut_ptr().cast::<u8>()).unwrap_or_else(NonNull::dangling);
    // SAFETY: the guard owns the allocation, and the caller proves that every
    // reachable destination element is overwritten before typed exposure.
    unsafe {
        strided_kernel::ErasedRawStridedUninitMut::from_raw_parts(
            dtype,
            data_ptr,
            data.len(),
            dims,
            strides,
            offset,
        )
    }
}

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

pub use affinity::{
    available_parallelism, process_cpu_affinity, process_cpu_affinity_count, CpuAffinityError,
};
pub use affinity_policy::{
    resolve_cpu_affinity, resolve_cpu_affinity_with_override, CpuAffinityInput,
    CpuAffinityInputError, CpuAffinityPolicy, CpuAffinityResolutionError, CpuAffinitySelection,
    CpuAffinitySelectionReason,
};
pub use backend::{
    CpuBackend, CpuBackendError, CpuBackendKind, CpuExecutionInfo, CpuExecutionMode,
    ExternalCpuDomainRegistryError,
};
pub use buffer_pool::BufferPoolStats;
pub use capability::cpu_capabilities;
pub use context::{CpuContext, CpuContextError};
pub use domain_executor::{
    CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuExecutorAffinity,
    CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism, ScopedCpuJob, ScopedCpuJobs,
};
pub use dot_runtime::{
    CpuProviderBundle, CpuProviderBundleBuildError, CpuProviderBundleBuilder,
    CpuProviderBundleInstallError, CpuProviderSlot, GeneralContractionPolicy,
};
#[doc(hidden)]
pub use exec_session::CpuExecSession;
pub use indexed_plan_cache::IndexedPlanCacheLimits;
pub use placement::{
    CpuEngineConstructionError, CpuPlacement, CpuPlacementError, CpuPlacementGuarantee,
    ResolvedCpuPlacement,
};
pub use provider::{CpuExecutionContext, ParallelMode};
pub use provider_capability::{
    CpuPlacementControl, CpuProviderDomainError, CpuProviderExecutionCapabilities,
    CpuThreadCountControl,
};
pub use resource_domain::{CpuDomainOwnership, ExternalCpuDomain, ExternalCpuDomainError};
pub use runtime_adapter::{
    runtime_engine_id, runtime_engine_registration, runtime_engine_registration_with_id,
    runtime_hardware_class,
};
pub use topology::{
    discover_cpu_topology, CpuId, CpuNode, CpuSet, CpuSetError, CpuTopology, CpuTopologyError,
    NumaNodeId,
};

/// Visit a CPU execution session carried by a type-erased backend session.
///
/// This is a backend-leaf capability bridge. The type-name check is performed
/// before the erased pointer is reconstructed, and the callback cannot return
/// a borrow of the session, so the borrowed resource lease remains scoped to
/// the caller's session closure.
#[doc(hidden)]
pub fn with_cpu_exec_session<R>(
    session: &mut dyn tenferro_tensor::BackendSession,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R,
) -> Option<R> {
    if session.session_type_name() != std::any::type_name::<CpuExecSession<'static>>() {
        return None;
    }
    let data = unsafe { session.session_data_mut() };
    // SAFETY: `session_type_name` is supplied by the same blanket
    // `BackendSession` implementation that produced `session_data_mut`, and
    // the equality above proves that the erased value is `CpuExecSession`.
    // The callback is higher-ranked and returns no session borrow, so the
    // reconstructed reference cannot escape the original session borrow.
    Some(unsafe { f(&mut *(data.cast::<CpuExecSession<'static>>())) })
}

// Unit tests exercise the pool-aware kernels through the former convenience
// names without restoring those names to the production crate surface.
#[cfg(test)]
pub(crate) use analytic::pow;
#[cfg(test)]
macro_rules! test_elementwise_wrapper {
    ($name:ident($($arg:ident: $ty:ty),*) => $with_pool:ident) => {
        pub(crate) fn $name($($arg: $ty),*) -> crate::Result<Tensor> {
            let mut buffers = BufferPool::new();
            elementwise::$with_pool(&mut buffers, $($arg),*)
        }
    };
}
#[cfg(test)]
test_elementwise_wrapper!(abs(input: &Tensor) => abs_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(add(lhs: &Tensor, rhs: &Tensor) => add_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) => clamp_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) => compare_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(conj(input: &Tensor) => conj_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(div(lhs: &Tensor, rhs: &Tensor) => div_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(maximum(lhs: &Tensor, rhs: &Tensor) => maximum_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(minimum(lhs: &Tensor, rhs: &Tensor) => minimum_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(mul(lhs: &Tensor, rhs: &Tensor) => mul_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(neg(input: &Tensor) => neg_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(rem(lhs: &Tensor, rhs: &Tensor) => rem_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) => select_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(sign(input: &Tensor) => sign_with_pool);
#[cfg(test)]
test_elementwise_wrapper!(sub(lhs: &Tensor, rhs: &Tensor) => sub_with_pool);
#[cfg(test)]
pub(crate) use indexing::{dynamic_slice, dynamic_update_slice, gather, pad, scatter};
#[cfg(test)]
pub(crate) use reduction::{reduce_max, reduce_min, reduce_prod, reduce_sum, reduce_sum_squares};
#[cfg(test)]
pub(crate) use structural::{
    broadcast_in_dim, embed_diagonal, extract_diagonal, reshape, transpose, tril, triu,
};

/// Owner-scoped CPU scratch-pool API for operation-family crates.
///
/// This module is not an application-facing tensor API. It exists so
/// operation crates that implement CPU kernels can share `CpuBackend`'s
/// allocation pool without exposing the pool as a general public contract.
#[doc(hidden)]
pub mod linalg_interop {
    pub use crate::buffer_pool::{BufferPool, PoolScalar};
    pub use tenferro_internal_cpu_kernels::PooledUninitOutput;
}

pub(crate) fn cpu_backend_buffer_error(op: &'static str) -> crate::Error {
    crate::Error::runtime_state(
        op,
        "CPU backend received backend buffer; download to host before CPU execution",
    )
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CpuNumericalError {
    #[error("{op} received a negative integer exponent for dtype {dtype:?}")]
    NegativeIntegerExponent { op: &'static str, dtype: DType },
}

pub(crate) fn cpu_negative_integer_exponent(op: &'static str, dtype: DType) -> crate::Error {
    crate::Error::extension(
        op,
        "cpu",
        ErrorKind::NumericalFailure,
        CpuNumericalError::NegativeIntegerExponent { op, dtype },
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
                .map_err(|err| crate::Error::backend_source(op, err))
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
    .map_err(|err| crate::Error::backend_source(op, err))
}

pub(crate) fn materialize_tensor_read(
    buffers: &mut BufferPool,
    op: &'static str,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(tensor) => clone_host_tensor_read(op, tensor),
        TensorRead::View(view) => materialize_tensor_view(buffers, op, view),
    }
}

pub(crate) fn copy_tensor_read_into(
    op: &'static str,
    src: TensorRead<'_>,
    dst: TensorWrite<'_>,
) -> crate::Result<()> {
    let src_dtype = src.dtype();
    let dst_dtype = dst.dtype();
    macro_rules! copy_source {
        ($variant:ident, $src:expr) => {{
            let src = $src;
            match dst {
                TensorWrite::Tensor(Tensor::$variant(dst)) => {
                    let mut dst = dst.as_view_mut();
                    structural::typed_copy_view_into(&src, &mut dst, op)
                }
                TensorWrite::View(TensorViewMut::$variant(mut dst)) => {
                    structural::typed_copy_view_into(&src, &mut dst, op)
                }
                _ => Err(crate::Error::dtype_mismatch(op, src_dtype, dst_dtype)),
            }
        }};
    }

    match src {
        TensorRead::Tensor(Tensor::F32(src)) => copy_source!(F32, src.as_view()),
        TensorRead::Tensor(Tensor::F64(src)) => copy_source!(F64, src.as_view()),
        TensorRead::Tensor(Tensor::I32(src)) => copy_source!(I32, src.as_view()),
        TensorRead::Tensor(Tensor::I64(src)) => copy_source!(I64, src.as_view()),
        TensorRead::Tensor(Tensor::Bool(src)) => copy_source!(Bool, src.as_view()),
        TensorRead::Tensor(Tensor::C32(src)) => copy_source!(C32, src.as_view()),
        TensorRead::Tensor(Tensor::C64(src)) => copy_source!(C64, src.as_view()),
        TensorRead::View(TensorView::F32(src)) => copy_source!(F32, src),
        TensorRead::View(TensorView::F64(src)) => copy_source!(F64, src),
        TensorRead::View(TensorView::I32(src)) => copy_source!(I32, src),
        TensorRead::View(TensorView::I64(src)) => copy_source!(I64, src),
        TensorRead::View(TensorView::Bool(src)) => copy_source!(Bool, src),
        TensorRead::View(TensorView::C32(src)) => copy_source!(C32, src),
        TensorRead::View(TensorView::C64(src)) => copy_source!(C64, src),
    }
}

fn clone_host_tensor_read(op: &'static str, tensor: &Tensor) -> crate::Result<Tensor> {
    macro_rules! clone_host {
        ($variant:ident, $tensor:expr) => {{
            structural::validate_cpu_host_placement(op, "source", $tensor.placement())?;
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

fn materialize_tensor_view(
    buffers: &mut BufferPool,
    op: &'static str,
    view: TensorView<'_>,
) -> crate::Result<Tensor> {
    macro_rules! materialize {
        ($variant:ident, $view:expr) => {{
            Ok(Tensor::$variant(
                structural::typed_materialize_view_with_pool(buffers, &$view, op)?,
            ))
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

#[cfg(test)]
pub(crate) fn tensor_from_array<T: Clone>(array: StridedArray<T>) -> TypedTensor<T> {
    // Invariant: `StridedArray` owns data whose length matches its validated dimensions.
    TypedTensor::from_vec_col_major(array.dims().to_vec(), array.into_data())
        .expect("strided array dimensions match owned data length")
}

pub(crate) fn default_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: None,
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

// `provider-inject` owns call-through coverage in the serialized integration
// fixture, which registers every BLAS symbol before the first operation.  The
// broad unit suite selects the compiled default backend and therefore must not
// call an intentionally unregistered injected symbol.
#[cfg(all(test, not(feature = "provider-inject")))]
mod tests;
