#![doc(hidden)]

//! Shared host CPU resources and low-level adapters.

/// Internal result alias for shared CPU adapters.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu_basic::Result;
/// let result: Result<()> = Ok(());
/// assert!(result.is_ok());
/// ```
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
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    KernelDType, StridedView,
};

// Both sides own the same inline-capacity contract: tenferro's `ShapeVec`/
// `StrideVec` and strided-view's metadata must spill at the same rank, or a
// view would allocate where the metadata budget assumes it cannot.
const _: () = assert!(tenferro_tensor::INLINE_RANK == strided_basic::INLINE_RANK);

/// Error returned when a CPU-only operation receives backend storage.
#[doc(hidden)]
pub fn cpu_backend_buffer_error(op: &'static str) -> Error {
    Error::runtime_state(
        op,
        "CPU backend received backend buffer; download to host before CPU execution",
    )
}

/// Error returned when a CPU operand's payload is not host memory.
///
/// `role` names the operand (`lhs`, `destination`, ...). The verdict is the
/// payload predicate, so this error never accuses a host tensor of being a
/// device buffer because a label said so.
#[doc(hidden)]
pub fn cpu_non_host_operand_error(op: &'static str, role: &'static str) -> Error {
    Error::runtime_state(
        op,
        format!(
            "CPU backend requires {role} host placement for {op}: the payload is a backend buffer; download to host before CPU execution"
        ),
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
    /// Apply complex conjugation, or leave a real scalar unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu_basic::ConjElem;
    /// assert_eq!(ConjElem::conj_elem(2.0_f64), 2.0);
    /// ```
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
    if let Some(buffer) = tensor.backend_buffer() {
        let expected =
            tenferro_tensor::validate::checked_shape_product(op, "shape", tensor.shape())?;
        return shared_read_slice(op, expected, buffer.shared_host_slice());
    }
    tensor.host_data()
}

/// Admit one backend allocation the CPU can read, or keep the CPU refusal.
///
/// A shared allocation domain hands out allocations the CPU can read at the
/// same address as the device, so the provider's borrow (`shared_host_slice`,
/// which waits for pending device writes) is the admission. Every other
/// backend payload keeps the historical "download to host" refusal, and a
/// provider's own typed failure survives as [`Error::host_access`].
#[doc(hidden)]
pub fn shared_read_slice<'a, T>(
    op: &'static str,
    expected: usize,
    slice: std::result::Result<&'a [T], tenferro_tensor::HostAccessError>,
) -> Result<&'a [T]> {
    let slice = admit_shared_slice(op, slice)?;
    slice
        .get(..expected)
        .ok_or_else(|| Error::validation(op, tenferro_tensor::ValidationError::ViewOutOfBounds))
}

/// Turn the provider's borrow into a slice, keeping every non-admission a typed refusal.
fn admit_shared_slice<'a, T>(
    op: &'static str,
    slice: std::result::Result<&'a [T], tenferro_tensor::HostAccessError>,
) -> Result<&'a [T]> {
    match slice {
        Ok(slice) => Ok(slice),
        Err(tenferro_tensor::HostAccessError::Unsupported { .. }) => {
            Err(cpu_backend_buffer_error(op))
        }
        Err(error) => Err(Error::host_access(op, error)),
    }
}

#[doc(hidden)]
pub fn typed_view<'a, T: Copy + TensorScalar>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> Result<StridedView<'a, T>> {
    let data = typed_host_data(op, tensor)?;
    // The tensor caches its compact layout, so the view borrows both metadata
    // vectors instead of allocating a pair per call.
    StridedView::new(data, tensor.shape(), tensor.strides(), 0)
        .map_err(|err| Error::backend_source(op, err))
}

/// Admit one borrowed view the CPU can read, keeping its strides and offset.
///
/// The allocation, not the view, bounds the borrow: a sliced or strided view
/// reaches outside its own `n_elements`, so the extent check belongs to the
/// caller's `StridedView::new`, as it already does for a host view.
#[doc(hidden)]
pub fn shared_view_read_slice<'a, T>(
    op: &'static str,
    buffer: &'a dyn tenferro_tensor::BackendStorage<T>,
) -> Result<&'a [T]>
where
    T: 'static,
{
    admit_shared_slice(op, buffer.shared_host_slice())
}

#[doc(hidden)]
pub fn typed_view_from_view<'a, T: Copy + 'static, R: TensorRank>(
    op: &'static str,
    view: &TypedTensorView<'a, T, R>,
) -> Result<StridedView<'a, T>> {
    let data = match view.shared_backend_storage() {
        Some(buffer) => shared_view_read_slice(op, buffer)?,
        None => view.host_storage()?,
    };
    StridedView::new(data, view.shape(), view.strides(), view.offset())
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
