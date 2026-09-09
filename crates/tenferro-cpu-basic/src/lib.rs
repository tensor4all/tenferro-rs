#![doc(hidden)]

//! Shared host CPU resources and low-level adapters.

pub type Result<T> = tenferro_tensor::Result<T>;
use tenferro_tensor::CompareDir;
pub use tenferro_tensor::{
    CacheStats, DType, Error, ErrorKind, TensorRank, TensorScalar, TypedTensor, TypedTensorView,
};

pub mod buffer_pool;
pub use buffer_pool::{BufferPool, PoolScalar};
mod pooled_uninit_output;
pub use pooled_uninit_output::PooledUninitOutput;

use num_complex::{Complex32, Complex64};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use strided_basic::{
    col_major_strides as kernel_col_major_strides, ErasedRawStridedMut, ErasedRawStridedPtr,
    ErasedRawStridedRef, ErasedRawStridedUninitMut, KernelDType, StridedView,
};

/// Error returned when a CPU-only operation receives backend storage.
#[doc(hidden)]
pub fn cpu_backend_buffer_error(op: &'static str) -> Error {
    Error::runtime_state(
        op,
        "CPU backend received backend buffer; download to host before CPU execution",
    )
}

#[derive(Debug, thiserror::Error)]
#[doc(hidden)]
pub enum CpuNumericalError {
    #[error("{op} detected division by zero for dtype {dtype:?}")]
    DivisionByZero { op: &'static str, dtype: DType },
}

#[doc(hidden)]
pub fn cpu_division_by_zero(op: &'static str, dtype: DType) -> Error {
    Error::extension(
        op,
        "cpu",
        ErrorKind::NumericalFailure,
        CpuNumericalError::DivisionByZero { op, dtype },
    )
}

#[doc(hidden)]
pub trait ConjElem {
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
impl ConjElem for Complex32 {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}
impl ConjElem for Complex64 {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

#[doc(hidden)]
pub fn is_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::C32 | DType::C64)
}

#[doc(hidden)]
pub fn ordered_complex_error(op: &'static str) -> Error {
    Error::unsupported(
        op,
        "complex tensors do not have a total order; compute abs/norm explicitly before ordered operations",
    )
}

#[doc(hidden)]
pub fn reject_complex_ordered_dtypes(op: &'static str, dtypes: &[DType]) -> Result<()> {
    if dtypes.iter().copied().any(is_complex_dtype) {
        return Err(ordered_complex_error(op));
    }
    Ok(())
}

#[doc(hidden)]
pub fn reject_complex_unsupported_compare_dtypes(dir: &CompareDir, dtypes: &[DType]) -> Result<()> {
    if *dir != CompareDir::Eq {
        reject_complex_ordered_dtypes("compare", dtypes)?;
    }
    Ok(())
}

#[doc(hidden)]
pub fn typed_host_data<'a, T: TensorScalar>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> Result<&'a [T]> {
    if tensor.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    tensor.host_data()
}

#[doc(hidden)]
pub fn typed_view<'a, T: Copy + TensorScalar>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> Result<StridedView<'a, T>> {
    if tensor.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    let data = tensor.host_data()?;
    let strides = kernel_col_major_strides(tensor.shape());
    StridedView::new(data, tensor.shape(), &strides, 0)
        .map_err(|err| Error::backend_source(op, err))
}

#[doc(hidden)]
pub fn typed_view_from_view<'a, T: Copy + 'static, R: TensorRank>(
    op: &'static str,
    view: &TypedTensorView<'a, T, R>,
) -> Result<StridedView<'a, T>> {
    if view.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    StridedView::new(
        view.host_storage()?,
        view.shape(),
        view.strides(),
        view.offset(),
    )
    .map_err(|err| Error::backend_source(op, err))
}

/// Construct an erased read-only strided view over initialized typed storage.
///
/// # Safety
/// `dtype` must match the aligned initialized storage. All reachable offsets
/// must be in bounds and the storage/metadata must remain valid for `'a`.
#[doc(hidden)]
pub unsafe fn erased_raw_strided_ref<'a>(
    dtype: KernelDType,
    data: &'a [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_basic::Result<ErasedRawStridedRef<'a>> {
    let data_ptr = NonNull::new(data.as_ptr().cast_mut()).unwrap_or_else(NonNull::dangling);
    // SAFETY: the caller supplies the documented dtype, bounds, alignment and lifetime invariants.
    unsafe {
        ErasedRawStridedRef::from_raw_parts(dtype, data_ptr, data.len(), dims, strides, offset)
    }
}

/// Construct an erased pointer view over initialized storage.
///
/// # Safety
/// The caller must uphold the dtype, alignment, bounds, lifetime and
/// initialization requirements of the returned descriptor.
#[doc(hidden)]
pub unsafe fn erased_raw_strided_ptr<'a>(
    dtype: KernelDType,
    data: &'a [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_basic::Result<ErasedRawStridedPtr<'a>> {
    let data_ptr = NonNull::new(data.as_ptr().cast_mut()).unwrap_or_else(NonNull::dangling);
    // SAFETY: the caller supplies the documented dtype, bounds, alignment and lifetime invariants.
    unsafe {
        ErasedRawStridedPtr::from_raw_parts(dtype, data_ptr, data.len(), dims, strides, offset)
    }
}

/// Construct an erased writable view over exclusively owned initialized storage.
///
/// # Safety
/// The caller must uphold the dtype, alignment, bounds, lifetime and exclusive
/// access requirements of the returned descriptor.
#[doc(hidden)]
pub unsafe fn erased_raw_strided_mut<'a>(
    dtype: KernelDType,
    data: &'a mut [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_basic::Result<ErasedRawStridedMut<'a>> {
    let data_ptr = NonNull::new(data.as_mut_ptr()).unwrap_or_else(NonNull::dangling);
    // SAFETY: the caller supplies the documented dtype, bounds, alignment and exclusive-lifetime invariants.
    unsafe {
        ErasedRawStridedMut::from_raw_parts(dtype, data_ptr, data.len(), dims, strides, offset)
    }
}

/// Construct an erased writable strided view over exclusively owned uninitialized storage.
///
/// # Safety
/// `dtype` must match the aligned storage; reachable offsets must be in bounds;
/// the borrow must remain exclusive; and storage must stay uninitialized until
/// every reachable element has been written.
#[doc(hidden)]
pub unsafe fn erased_raw_strided_uninit_mut<'a>(
    dtype: KernelDType,
    data: &'a mut [MaybeUninit<u8>],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_basic::Result<ErasedRawStridedUninitMut<'a>> {
    let data_ptr = NonNull::new(data.as_mut_ptr().cast::<u8>()).unwrap_or_else(NonNull::dangling);
    // SAFETY: the caller supplies the documented dtype, bounds, alignment and exclusive-lifetime invariants.
    unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(
            dtype,
            data_ptr,
            data.len(),
            dims,
            strides,
            offset,
        )
    }
}
