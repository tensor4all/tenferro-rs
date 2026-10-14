use num_complex::{Complex, Complex32, Complex64};
use num_traits::{One, Zero};
use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::{align_of, needs_drop, offset_of, size_of};
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::SliceConfig;
use crate::error::ReinterpretError;
use smallvec::SmallVec;
pub use tenferro_tensor_core::{DType, DynRank, Rank, TensorLayout, TensorRank};
use tenferro_tensor_core::{ShapeVec, StrideVec};
use tenferro_tensor_core::{SliceSpec as CoreSliceSpec, ValidationError};

use crate::storage::{
    AllocationGroup, AllocationSlot, BackendAllocation, DescriptorSlot, GroupError, GroupReadView,
    GroupWriteView,
};

mod accessors;
mod col_major;
mod shape_packing;
mod strided_view;
#[cfg(test)]
mod tests;

pub use col_major::{ColMajorView, ColMajorViewMut};
pub use strided_view::StridedSliceSpec;

fn shape_vec(shape: &[usize]) -> ShapeVec {
    shape.iter().copied().collect()
}

fn stride_vec(strides: &[isize]) -> StrideVec {
    strides.iter().copied().collect()
}

fn representation_pair_error(
    op: &'static str,
    from: DType,
    to: DType,
    message: impl Into<String>,
) -> crate::Error {
    crate::Error::unsupported_dtype_conversion(op, from, to, message)
}

fn validate_representation_pair(op: &'static str, from: DType, to: DType) -> crate::Result<()> {
    let valid = match (from, to) {
        (DType::C32, DType::F32) | (DType::F32, DType::C32) => {
            size_of::<Complex32>() == 2 * size_of::<f32>()
                && align_of::<Complex32>() == align_of::<f32>()
                && offset_of!(Complex32, re) == 0
                && offset_of!(Complex32, im) == size_of::<f32>()
                && !needs_drop::<Complex32>()
                && !needs_drop::<f32>()
        }
        (DType::C64, DType::F64) | (DType::F64, DType::C64) => {
            size_of::<Complex64>() == 2 * size_of::<f64>()
                && align_of::<Complex64>() == align_of::<f64>()
                && offset_of!(Complex64, re) == 0
                && offset_of!(Complex64, im) == size_of::<f64>()
                && !needs_drop::<Complex64>()
                && !needs_drop::<f64>()
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(representation_pair_error(
            op,
            from,
            to,
            "only the sealed Complex<f32><->f32 and Complex<f64><->f64 representations are supported",
        ))
    }
}

fn reinterpret_complex_to_real_layout(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    complex_buffer_len: usize,
    op: &'static str,
) -> crate::Result<TensorLayout<DynRank>> {
    let real_buffer_len = complex_buffer_len
        .checked_mul(2)
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    let rank = shape
        .len()
        .checked_add(1)
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    let mut real_shape = ShapeVec::with_capacity(rank);
    real_shape.push(2);
    real_shape.extend_from_slice(shape);
    let mut real_strides = StrideVec::with_capacity(rank);
    real_strides.push(1);
    for &stride in strides {
        real_strides.push(
            stride
                .checked_mul(2)
                .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?,
        );
    }
    let real_offset = offset
        .checked_mul(2)
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    TensorLayout::from_parts(real_shape, real_strides, real_offset, real_buffer_len)
        .map_err(|err| tensor_layout_error(op, err))
}

fn reinterpret_real_to_complex_layout(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    real_buffer_len: usize,
    op: &'static str,
) -> crate::Result<TensorLayout<DynRank>> {
    if shape.first().copied() != Some(2) {
        return Err(crate::Error::invalid_argument(
            op,
            "shape",
            "the leading extent must be 2 for a complex reinterpretation",
        ));
    }
    if strides.first().copied() != Some(1) {
        return Err(crate::Error::invalid_argument(
            op,
            "strides",
            "the leading stride must be 1 for a complex reinterpretation",
        ));
    }
    if offset % 2 != 0 {
        return Err(crate::Error::invalid_argument(
            op,
            "offset",
            "the offset must be divisible by 2 for a complex reinterpretation",
        ));
    }
    let mut complex_strides = StrideVec::with_capacity(strides.len() - 1);
    for &stride in &strides[1..] {
        if stride % 2 != 0 {
            return Err(crate::Error::invalid_argument(
                op,
                "strides",
                "all non-leading strides must be divisible by 2",
            ));
        }
        complex_strides.push(stride / 2);
    }
    let complex_buffer_len = real_buffer_len / 2;
    TensorLayout::from_parts(
        shape[1..].iter().copied().collect(),
        complex_strides,
        offset / 2,
        complex_buffer_len,
    )
    .map_err(|err| tensor_layout_error(op, err))
}

fn reinterpret_host_slice<'a, T: TensorScalar, U: TensorScalar>(
    data: &'a [T],
    op: &'static str,
) -> crate::Result<&'a [U]> {
    let byte_len = data
        .len()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    if !byte_len.is_multiple_of(size_of::<U>()) {
        return Err(crate::Error::validation(
            op,
            ValidationError::ViewOutOfBounds,
        ));
    }
    if data.as_ptr().align_offset(align_of::<U>()) != 0 {
        return Err(crate::Error::invalid_argument(
            op,
            "alignment",
            "the source allocation is not aligned for the target representation",
        ));
    }
    // SAFETY: `validate_representation_pair` seals the only supported pairs;
    // sizes, alignment, field order, and drop properties are checked before
    // exposing the borrowed target slice.
    Ok(unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<U>(), byte_len / size_of::<U>()) })
}

fn reinterpret_host_slice_mut<'a, T: TensorScalar, U: TensorScalar>(
    data: &'a mut [T],
    op: &'static str,
) -> crate::Result<&'a mut [U]> {
    let byte_len = data
        .len()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    if !byte_len.is_multiple_of(size_of::<U>()) {
        return Err(crate::Error::validation(
            op,
            ValidationError::ViewOutOfBounds,
        ));
    }
    if data.as_mut_ptr().align_offset(align_of::<U>()) != 0 {
        return Err(crate::Error::invalid_argument(
            op,
            "alignment",
            "the source allocation is not aligned for the target representation",
        ));
    }
    // SAFETY: the mutable source borrow is unique and the sealed pair has no
    // padding or drop glue, so the target slice covers the same bytes exactly.
    Ok(unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<U>(), byte_len / size_of::<U>())
    })
}

/// Memory location for tensor storage.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::MemoryKind;
///
/// let kind = MemoryKind::UnpinnedHost;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Managed,
    Other(String),
}

/// Compute device family.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DeviceKind;
///
/// let kind = DeviceKind::Cpu;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu(GpuBackendKind),
    Other(String),
}

/// GPU backend family used by placement metadata.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::GpuBackendKind;
///
/// let kind = GpuBackendKind::Cuda;
/// let webgpu = GpuBackendKind::WebGpu;
/// assert_ne!(kind, webgpu);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuBackendKind {
    Cuda,
    WebGpu,
    Rocm,
    Other(String),
}

/// Concrete compute device identifier.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DeviceId, DeviceKind, GpuBackendKind};
///
/// let device = DeviceId {
///     kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
///     ordinal: 0,
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub kind: DeviceKind,
    pub ordinal: usize,
}

/// Caller-stable identity for a CPU execution domain.
///
/// Domain IDs are metadata supplied by the caller or execution coordinator;
/// creating an ID does not allocate a process-global identity.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::CpuDomainId;
///
/// let domain = CpuDomainId::new(17);
/// assert_eq!(domain.as_u64(), 17);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CpuDomainId(u64);

impl CpuDomainId {
    /// Create a caller-stable CPU domain identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::CpuDomainId;
    ///
    /// assert_eq!(CpuDomainId::new(3), CpuDomainId::new(3));
    /// ```
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the caller-supplied integer identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::CpuDomainId;
    ///
    /// assert_eq!(CpuDomainId::new(9).as_u64(), 9);
    /// ```
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Placement metadata for a tensor buffer.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement};
///
/// let placement = Placement {
///     memory_kind: MemoryKind::Device,
///     device: Some(DeviceId {
///         kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
///         ordinal: 0,
///     }),
///     cpu_affinity: None,
/// };
/// assert!(placement.cpu_affinity.is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    /// Storage memory class, independent of execution routing metadata.
    pub memory_kind: MemoryKind,
    /// Device that owns or addresses the storage, when applicable.
    pub device: Option<DeviceId>,
    /// Preferred or producing CPU execution domain for routing and locality.
    ///
    /// This tag is not proof of allocation ownership, page residency, NUMA
    /// pinning, or worker-affinity enforcement. Backend allocation-domain
    /// metadata remains attached to the buffer independently.
    pub cpu_affinity: Option<CpuDomainId>,
}

impl Default for Placement {
    fn default() -> Self {
        default_placement()
    }
}

/// Backend-owned buffer handle.
///
/// `BackendStorageHandle::new` creates an empty opaque handle. Use
/// [`BackendStorageHandle::new_with_len`] when test or adapter code needs to model a
/// non-empty backend allocation.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BackendStorageHandle;
///
/// let handle = BackendStorageHandle::<f64>::new(7);
/// ```
pub struct BackendStorageHandle<T> {
    id: u64,
    len: usize,
    allocation_domain: AllocationDomainId,
    _phantom: std::marker::PhantomData<T>,
}

/// Identity of a backend-owned allocation domain.
///
/// Domains let cooperating backends accept shared allocations without treating
/// another context's physically similar buffer as compatible.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::AllocationDomainId;
///
/// assert_ne!(AllocationDomainId::fresh(), AllocationDomainId::fresh());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationDomainId(u64);

impl AllocationDomainId {
    /// Create a process-unique allocation-domain identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::AllocationDomainId;
    ///
    /// let domain = AllocationDomainId::fresh();
    /// assert_eq!(domain, domain);
    /// ```
    pub fn fresh() -> Self {
        static NEXT_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_DOMAIN_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Stable physical identity of one backend allocation.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::AllocationId;
///
/// assert_eq!(AllocationId::from_backend_id(7), AllocationId::from_backend_id(7));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId(u64);

impl AllocationId {
    /// Wrap an allocation identity supplied by the owning backend.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::AllocationId;
    ///
    /// let id = AllocationId::from_backend_id(3);
    /// assert_eq!(id, AllocationId::from_backend_id(3));
    /// ```
    pub const fn from_backend_id(id: u64) -> Self {
        Self(id)
    }
}

/// Typed failure returned by guarded backend host access.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::HostAccessError;
///
/// let error = HostAccessError::Unsupported { backend: "opaque" };
/// assert!(error.to_string().contains("opaque"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HostAccessError {
    /// The backend does not expose guarded host access for this allocation.
    #[error("backend `{backend}` does not support guarded host access")]
    Unsupported { backend: &'static str },
    /// The allocation belongs to another shared-allocation domain.
    #[error("allocation belongs to domain {actual:?}, expected {expected:?}")]
    ForeignDomain {
        expected: AllocationDomainId,
        actual: AllocationDomainId,
    },
    /// Another host mapping overlaps this allocation.
    #[error("the allocation already has an active host mapping")]
    OverlappingHostMapping,
    /// GPU work currently owns or has reserved the allocation.
    #[error("GPU access is in progress for the allocation")]
    GpuAccessInProgress,
    /// Host mapping is active while GPU access was requested.
    #[error("the allocation is mapped for host access")]
    MappedForHost,
    /// The backend failed to complete the map operation.
    #[error("backend host mapping failed: {message}")]
    BackendFailure { message: String },
    /// The source did not cover the full write-only mapping.
    #[error("host write length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
}

/// Metadata sealed at the tensor/root boundary before a provider launch.
///
/// Providers receive this request exactly once for a prepared access. Binding
/// code consumes the resulting opaque state and does not receive replacement
/// storage, ranges, or raw pointers.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct DeviceAccessRequest<'a> {
    allocation_domain: AllocationDomainId,
    allocation_id: AllocationId,
    byte_len: usize,
    element_size: usize,
    shape: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

impl<'a> DeviceAccessRequest<'a> {
    pub(crate) fn new(
        allocation_domain: AllocationDomainId,
        allocation_id: AllocationId,
        byte_len: usize,
        element_size: usize,
        shape: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Self {
        Self {
            allocation_domain,
            allocation_id,
            byte_len,
            element_size,
            shape,
            strides,
            offset,
        }
    }

    pub fn allocation_domain(&self) -> AllocationDomainId {
        self.allocation_domain
    }

    pub fn allocation_id(&self) -> AllocationId {
        self.allocation_id
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn element_size(&self) -> usize {
        self.element_size
    }

    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    pub fn strides(&self) -> &[isize] {
        self.strides
    }

    pub fn offset(&self) -> isize {
        self.offset
    }
}

/// Typed failure returned while preparing a provider-native device access.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DeviceAccessError {
    #[error("backend `{backend}` does not support prepared device access")]
    Unsupported { backend: &'static str },
    #[error("prepared device access request is invalid: {message}")]
    InvalidRequest { message: String },
    #[error("provider device preparation failed: {message}")]
    ProviderFailure { message: String },
}

/// Opaque provider-prepared state retained for one device binding.
#[doc(hidden)]
pub trait PreparedDeviceAccess: Debug {
    fn as_any(&self) -> &dyn Any;

    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

trait ReadGuardAccess<T> {
    fn as_slice(&self) -> &[T];
}

impl<T, G> ReadGuardAccess<T> for G
where
    G: Deref,
    G::Target: AsRef<[T]>,
{
    fn as_slice(&self) -> &[T] {
        self.deref().as_ref()
    }
}

/// Closure-scoped read mapping of a backend allocation.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::HostReadGuard;
///
/// let guard = HostReadGuard::new(vec![1_u32, 2]);
/// assert_eq!(&*guard, &[1, 2]);
/// ```
pub struct HostReadGuard<'a, T> {
    access: Box<dyn ReadGuardAccess<T> + 'a>,
}

impl<T> Debug for HostReadGuard<'_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostReadGuard")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl<'a, T> HostReadGuard<'a, T> {
    /// Wrap a backend-native read guard without exposing its concrete type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::HostReadGuard;
    ///
    /// let guard = HostReadGuard::new(vec![3_i32]);
    /// assert_eq!(guard[0], 3);
    /// ```
    pub fn new<G>(guard: G) -> Self
    where
        G: Deref + 'a,
        G::Target: AsRef<[T]>,
        T: 'a,
    {
        Self {
            access: Box::new(guard),
        }
    }

    /// Borrow the mapped elements as a rank-1 host view.
    ///
    /// The view lends the mapping's shared borrow, so the guard stays alive for
    /// as long as the view is used and no copy or transfer happens.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::HostReadGuard;
    ///
    /// let guard = HostReadGuard::new(vec![3_i32, 4]);
    /// let view = guard.as_view()?;
    /// assert_eq!(view.shape(), &[2]);
    /// assert_eq!(view.get(&[1]), Some(&4));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// mapped length cannot be represented as a rank-1 layout.
    pub fn as_view(&self) -> crate::Result<TypedTensorView<'_, T, DynRank>>
    where
        T: 'static,
    {
        let data: &[T] = self;
        TypedTensorView::from_slice(vec![data.len()], vec![1], 0, data)
    }
}

impl<T> Deref for HostReadGuard<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.access.as_slice()
    }
}

/// Backend-neutral owner of one shared tensor allocation domain.
///
/// CPU operation crates use this object-safe boundary to allocate results in
/// the same managed domain without depending on a GPU provider crate.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use tenferro_tensor::SharedTensorAllocationDomain;
///
/// let _domain: Option<Arc<dyn SharedTensorAllocationDomain>> = None;
/// ```
pub trait SharedTensorAllocationDomain: Debug + Send + Sync + 'static {
    /// Return the stable identity shared by every allocation from this owner.
    fn id(&self) -> AllocationDomainId;

    /// Allocate an uninitialized compact column-major tensor in this domain.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, unsupported-dtype, or backend allocation error.
    fn allocate(&self, dtype: DType, shape: &[usize]) -> crate::Result<Tensor>;
}

type HostWriteCopy<'a, T> = dyn FnMut(&[T]) -> Result<(), HostAccessError> + 'a;

/// Closure-scoped write-only mapping of a backend allocation.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{HostAccessError, HostWriteGuard};
///
/// let mut written = Vec::new();
/// {
///     let mut guard = HostWriteGuard::new(2, |source: &[u32]| {
///         written.extend_from_slice(source);
///         Ok::<(), HostAccessError>(())
///     });
///     guard.copy_from_slice(&[4, 5]).unwrap();
/// }
/// assert_eq!(written, [4, 5]);
/// ```
pub struct HostWriteGuard<'a, T> {
    len: usize,
    copy: Box<HostWriteCopy<'a, T>>,
}

impl<T> Debug for HostWriteGuard<'_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostWriteGuard")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<'a, T> HostWriteGuard<'a, T> {
    /// Wrap a backend-native write guard without exposing its concrete type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{HostAccessError, HostWriteGuard};
    ///
    /// let guard = HostWriteGuard::new(0, |_source: &[f32]| Ok::<(), HostAccessError>(()));
    /// assert!(guard.is_empty());
    /// ```
    ///
    /// # Errors
    ///
    /// Construction is infallible. A callback failure such as
    /// [`HostAccessError::BackendFailure`] is returned later by
    /// [`Self::copy_from_slice`].
    pub fn new<F>(len: usize, copy: F) -> Self
    where
        F: FnMut(&[T]) -> Result<(), HostAccessError> + 'a,
        T: 'a,
    {
        Self {
            len,
            copy: Box::new(copy),
        }
    }

    /// Number of elements covered by this write-only mapping.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{HostAccessError, HostWriteGuard};
    ///
    /// let guard = HostWriteGuard::new(2, |_source: &[f32]| Ok::<(), HostAccessError>(()));
    /// assert_eq!(guard.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when this mapping covers no elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{HostAccessError, HostWriteGuard};
    ///
    /// let guard = HostWriteGuard::new(0, |_source: &[f32]| Ok::<(), HostAccessError>(()));
    /// assert!(guard.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Replace the full mapped allocation contents.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{HostAccessError, HostWriteGuard};
    ///
    /// let mut guard = HostWriteGuard::new(1, |_source: &[f32]| Ok::<(), HostAccessError>(()));
    /// guard.copy_from_slice(&[1.0]).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`HostAccessError::LengthMismatch`] when `source` does not cover
    /// the complete mapping, or the typed backend error returned by the owning
    /// write guard.
    pub fn copy_from_slice(&mut self, source: &[T]) -> Result<(), HostAccessError> {
        if source.len() != self.len {
            return Err(HostAccessError::LengthMismatch {
                expected: self.len,
                actual: source.len(),
            });
        }
        (self.copy)(source)
    }
}

impl<T> Debug for BackendStorageHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendStorageHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl<T> BackendStorageHandle<T> {
    /// Create a new backend buffer handle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::BackendStorageHandle;
    ///
    /// let handle = BackendStorageHandle::<f64>::new(1);
    /// assert_eq!(tenferro_tensor::BackendStorage::len(&handle), 0);
    /// ```
    pub fn new(id: u64) -> Self {
        Self::new_with_len(id, 0)
    }

    /// Create a new backend buffer handle with a logical element count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{BackendStorage, BackendStorageHandle};
    ///
    /// let handle = BackendStorageHandle::<f64>::new_with_len(1, 4);
    /// assert_eq!(BackendStorage::len(&handle), 4);
    /// ```
    pub fn new_with_len(id: u64, len: usize) -> Self {
        Self {
            id,
            len,
            // Synthetic opaque handles are test/adapter allocations. Give
            // each one an explicit domain so root import never fabricates
            // identity from a missing provider field.
            allocation_domain: AllocationDomainId::fresh(),
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Opaque backend-owned tensor buffer.
///
/// Tensor core never inspects backend-native allocations directly. Backend
/// crates store their own concrete handle types behind this trait and
/// downcast inside the owning backend only.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use tenferro_tensor::{BackendStorage, BackendStorageHandle};
///
/// let buffer: Arc<dyn BackendStorage<f64>> = Arc::new(BackendStorageHandle::<f64>::new_with_len(7, 2));
/// assert_eq!(buffer.backend_family(), "opaque");
/// assert_eq!(buffer.len(), 2);
/// ```
pub trait BackendStorage<T>: Debug + Send + Sync + 'static {
    /// Stable backend family identifier.
    fn backend_family(&self) -> &'static str;

    /// Number of logical elements in the backend allocation.
    fn len(&self) -> usize;

    /// Returns `true` when the backend allocation is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the shared-allocation domain, when this buffer belongs to one.
    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        None
    }

    /// Return the stable physical allocation identity, when available.
    fn allocation_id(&self) -> Option<AllocationId> {
        None
    }

    /// Prepare one provider-native device access from the root-owned buffer.
    ///
    /// The returned state is consumed by the provider binding path. Providers
    /// that do not expose device launches return [`DeviceAccessError::Unsupported`].
    #[doc(hidden)]
    fn prepare_device_access(
        &self,
        _request: DeviceAccessRequest<'_>,
    ) -> Result<Box<dyn PreparedDeviceAccess>, DeviceAccessError> {
        Err(DeviceAccessError::Unsupported {
            backend: self.backend_family(),
        })
    }

    /// Map the allocation for closure-scoped host reads.
    ///
    /// # Errors
    ///
    /// The default returns [`HostAccessError::Unsupported`]. Host-visible
    /// backends return typed overlap, pending-GPU, or backend mapping failures.
    fn map_read(&self) -> Result<HostReadGuard<'_, T>, HostAccessError> {
        Err(HostAccessError::Unsupported {
            backend: self.backend_family(),
        })
    }

    /// Map the allocation for closure-scoped host writes.
    ///
    /// The mutable receiver keeps write authority with the owning tensor or
    /// provider object; borrowed views do not clone or share that authority.
    ///
    /// # Errors
    ///
    /// The default returns [`HostAccessError::Unsupported`]. Host-visible
    /// backends return typed overlap, pending-GPU, or backend mapping failures.
    fn map_write(&mut self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
        Err(HostAccessError::Unsupported {
            backend: self.backend_family(),
        })
    }

    /// Type-erased access for the backend crate that owns the concrete handle.
    fn as_any(&self) -> &dyn Any;
}

impl<T: Send + Sync + 'static> BackendStorage<T> for BackendStorageHandle<T> {
    fn backend_family(&self) -> &'static str {
        "opaque"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.allocation_domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(AllocationId::from_backend_id(self.id))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Tensor storage.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::StorageBuffer;
///
/// let host = StorageBuffer::Host(vec![1.0_f64, 2.0]);
/// ```
#[derive(Debug)]
pub enum StorageBuffer<T> {
    Host(Vec<T>),
    Backend(Box<dyn BackendStorage<T>>),
}

impl<T: 'static> StorageBuffer<T> {
    /// Return the physical element count in this buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::StorageBuffer;
    ///
    /// assert_eq!(StorageBuffer::Host(vec![1_i32, 2]).len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
        }
    }

    /// Return whether this buffer has no physical elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::StorageBuffer;
    ///
    /// assert!(StorageBuffer::<i32>::Host(Vec::new()).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return whether the storage is backend-owned rather than host-owned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::StorageBuffer;
    ///
    /// assert!(!StorageBuffer::Host(vec![1_i32]).is_backend());
    /// ```
    pub fn is_backend(&self) -> bool {
        matches!(self, Self::Backend(_))
    }
}

/// Sealed marker for the plain and pooled host payload of a [`TypedTensor`].
///
/// `TypedTensor<T, R, Host>` owns host elements directly, so it offers
/// infallible host access (`as_slice`, `get`, `get_mut`, `Index`/`IndexMut`)
/// and [`Clone`] without a runtime device check. `T` needs no `Copy`,
/// [`TensorScalar`] or arithmetic bound for that.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let tensor: TypedTensor<f64, DynRank, Host> =
///     TypedTensor::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
/// assert_eq!(tensor.as_slice(), &[1.0, 2.0]);
/// assert_eq!(tensor[&[1]], 2.0);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct Host;

/// Sealed marker for the group-backed payload of a [`TypedTensor`].
///
/// `TypedTensor<T, R, Gpu>` carries provider allocation authority, identity
/// and retirement state through its allocation group, so host access stays
/// fallible and must be prepared or mapped explicitly. The marker is not proof
/// of a particular device ordinal or provider.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DynRank, Gpu, Host, TypedTensor};
///
/// let host = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
/// let gpu: TypedTensor<f64, DynRank, Gpu> = host.promote()?;
/// assert_eq!(gpu.host_data()?, &[1.0, 2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct Gpu;

/// Sealed marker for the runtime union of the host and group-backed payloads.
///
/// This is the default representation: one type for ordinary tensors that may
/// be host-resident or backend-resident at runtime.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Dynamic, DynRank, Host, TypedTensor};
///
/// let host = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
/// let dynamic: TypedTensor<i32, DynRank, Dynamic> = host.into_dynamic();
/// assert_eq!(dynamic.host_data()?, &[1, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct Dynamic;

mod representation_sealed {
    pub trait Sealed {}
    impl Sealed for super::Host {}
    impl Sealed for super::Gpu {}
    impl Sealed for super::Dynamic {}
}

/// Sealed selector for the owned payload representation of a [`TypedTensor`].
///
/// The three markers are [`Host`], [`Gpu`] and [`Dynamic`]. The trait is
/// sealed: downstream crates select an existing representation, they do not
/// define one.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Dynamic, DynRank, Host, Representation, TypedTensor};
///
/// fn extent<D: Representation>(tensor: &TypedTensor<f64, DynRank, D>) -> usize {
///     tensor.shape()[0]
/// }
/// let host = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![3], vec![0.0; 3])?;
/// assert_eq!(extent(&host), 3);
/// let dynamic: TypedTensor<f64, DynRank, Dynamic> = host.into_dynamic();
/// assert_eq!(extent(&dynamic), 3);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait Representation: representation_sealed::Sealed + 'static {
    /// Owned payload stored for this representation.
    #[doc(hidden)]
    type Storage<T, R: TensorRank>;
}

impl Representation for Host {
    #[doc(hidden)]
    type Storage<T, R: TensorRank> = HostStorage<T>;
}

impl Representation for Gpu {
    #[doc(hidden)]
    type Storage<T, R: TensorRank> = GroupStorage<R>;
}

impl Representation for Dynamic {
    #[doc(hidden)]
    type Storage<T, R: TensorRank> = DynamicStorage<T, R>;
}

/// Directly owned plain or pooled host elements.
#[doc(hidden)]
pub struct HostStorage<T> {
    data: HostData<T>,
}

/// Group-backed payload: provider allocation authority, identity and retirement.
#[doc(hidden)]
pub struct GroupStorage<R: TensorRank> {
    group: Box<OwnedTensorGroup<R>>,
}

/// Runtime union of the [`Host`] and [`Gpu`] owned payloads.
#[doc(hidden)]
pub enum DynamicStorage<T, R: TensorRank> {
    Host(HostStorage<T>),
    Group(GroupStorage<R>),
}

impl<T> std::fmt::Debug for HostStorage<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(formatter)
    }
}

impl<R: TensorRank> std::fmt::Debug for GroupStorage<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroupStorage")
            .field("group", &self.group)
            .finish()
    }
}

impl<T, R: TensorRank> std::fmt::Debug for DynamicStorage<T, R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host(host) => formatter.debug_tuple("Host").field(host).finish(),
            Self::Group(group) => formatter.debug_tuple("Group").field(group).finish(),
        }
    }
}

/// Owned compact column-major typed tensor.
///
/// `T` is the element type, `R` the rank metadata (default [`DynRank`]) and `D`
/// the owned representation (default [`Dynamic`]). Shape and placement live on
/// the tensor itself, once, never inside a storage arm.
///
/// No bound is imposed on `T` by the type: plain host adoption only needs the
/// constructor's own requirements, and `T: Clone` is required only by the
/// copying constructors. A preset [`TensorScalar`] appears only where a dtype
/// identity or a numerical capability is actually used, never for plain host
/// ownership.
///
/// Owned tensors are compact column-major. Arbitrary strides and metadata-only
/// layout changes are represented by [`TypedTensorView`] and
/// [`TypedTensorViewMut`].
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DynRank, Host, Rank, Tensor, TypedTensor};
///
/// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(t.shape(), &[2, 2]);
///
/// let static_rank = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
/// assert_eq!(static_rank.rank(), 2);
///
/// let host: TypedTensor<i32, DynRank, Host> =
///     TypedTensor::from_host_vec_col_major(vec![2], vec![1, 2]).unwrap();
/// assert_eq!(host.get(&[1])?, &2);
///
/// let dynamic = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
/// assert_eq!(dynamic.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub struct TypedTensor<T, R: TensorRank = DynRank, D: Representation = Dynamic> {
    shape: R::Shape,
    placement: Placement,
    /// Column-major element strides, cached at construction so a strided view of
    /// this tensor borrows its layout instead of recomputing it per operation.
    layout: StrideVec,
    storage: D::Storage<T, R>,
}

/// Validate and adopt a column-major host vector as a statically host-owned tensor.
fn typed_host_tensor_from_vec_col_major<T, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    data: Vec<T>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R, Host>> {
    let shape = shape
        .into_rank_shape()
        .map_err(|err| tensor_layout_error(op, err))?;
    let layout = tenferro_tensor_core::col_major_strides(shape.as_ref())
        .map_err(|err| tensor_layout_error(op, err))?;
    try_checked_shape_len(shape.as_ref(), data.len(), op)?;
    Ok(TypedTensor {
        shape,
        placement: default_placement(),
        layout,
        storage: HostStorage {
            data: HostData::new(data),
        },
    })
}

/// Reorder explicit row-major host values into column-major order.
fn row_major_reorder<T: Clone, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    data: Vec<T>,
    op: &'static str,
) -> crate::Result<(R::Shape, Vec<T>)> {
    let shape = shape
        .into_rank_shape()
        .map_err(|err| tensor_layout_error(op, err))?;
    tenferro_tensor_core::col_major_strides(shape.as_ref())
        .map_err(|err| tensor_layout_error(op, err))?;
    try_checked_shape_len(shape.as_ref(), data.len(), op)?;
    let mut row_strides = ShapeVec::from_elem(0, shape.as_ref().len());
    let mut stride = 1usize;
    for axis in (0..row_strides.len()).rev() {
        row_strides[axis] = stride;
        stride = stride
            .checked_mul(shape.as_ref()[axis])
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    }
    let mut coordinates = ShapeVec::from_elem(0, row_strides.len());
    let mut source_offset = 0usize;
    let mut reordered = Vec::with_capacity(data.len());
    for _ in 0..data.len() {
        reordered.push(data[source_offset].clone());
        for axis in 0..coordinates.len() {
            coordinates[axis] += 1;
            if coordinates[axis] < shape.as_ref()[axis] {
                source_offset += row_strides[axis];
                break;
            }
            coordinates[axis] = 0;
            source_offset -= row_strides[axis] * (shape.as_ref()[axis] - 1);
        }
    }
    Ok((shape, reordered))
}

impl<T, R: TensorRank> TypedTensor<T, R, Host> {
    /// Adopt a column-major host `Vec<T>` as a statically host-owned tensor.
    ///
    /// The payload is the caller's vector itself; `T` needs no `Copy`,
    /// [`TensorScalar`] or arithmetic bound, and no group, session or device is
    /// involved.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{Host, Rank, TypedTensor};
    /// struct Custom(String);
    /// let tensor = TypedTensor::<Custom, Rank<2>, Host>::from_host_vec_col_major(
    ///     [1, 2], vec![Custom("a".into()), Custom("b".into())],
    /// )?;
    /// assert_eq!(tensor[&[0, 1]].0.as_str(), "b");
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`crate::Error::Validation`] for a rank mismatch, shape/data
    /// length mismatch or shape/stride arithmetic overflow.
    pub fn from_host_vec_col_major(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        data: Vec<T>,
    ) -> crate::Result<Self> {
        typed_host_tensor_from_vec_col_major(shape, data, "from_host_vec_col_major")
    }

    /// Explicitly import row-major host values into column-major storage.
    ///
    /// Clones each input element once.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{Host, Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<2>, Host>::from_host_vec_row_major(
    ///     [2, 3], vec![1, 2, 3, 4, 5, 6],
    /// )?;
    /// assert_eq!(tensor[&[1, 0]], 4);
    /// assert_eq!(tensor[&[0, 2]], 3);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`crate::Error::Validation`] for a rank, shape-length or stride
    /// overflow, or when the shape product disagrees with the input length.
    pub fn from_host_vec_row_major(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        data: Vec<T>,
    ) -> crate::Result<Self>
    where
        T: Clone,
    {
        let (shape, data) = row_major_reorder(shape, data, "from_host_vec_row_major")?;
        typed_host_tensor_from_vec_col_major(shape, data, "from_host_vec_row_major")
    }

    /// Borrow the owned host elements.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// assert_eq!(tensor.as_slice(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.storage.data.as_slice()
    }

    /// Alias of [`Self::as_slice`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// assert_eq!(tensor.host_data(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn host_data(&self) -> &[T] {
        self.as_slice()
    }

    /// Exclusively borrow the owned host elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// tensor.host_data_mut()[0] = 5;
    /// assert_eq!(tensor.as_slice(), &[5, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn host_data_mut(&mut self) -> &mut [T] {
        self.storage.data.as_mut_slice()
    }

    /// Borrow one element by checked column-major multi-index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1, 2, 3, 4])?;
    /// assert_eq!(tensor.get(&[1, 1])?, &4);
    /// assert!(tensor.get(&[2, 0]).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::Validation`] for a wrong rank, an out-of-range
    /// coordinate or offset arithmetic overflow.
    pub fn get(&self, indices: &[usize]) -> crate::Result<&T> {
        let offset = self.linear_offset(indices)?;
        self.as_slice().get(offset).ok_or_else(|| {
            crate::Error::validation("TypedTensor::get", ValidationError::ViewOutOfBounds)
        })
    }

    /// Exclusively borrow one element by checked column-major multi-index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// *tensor.get_mut(&[1])? = 9;
    /// assert_eq!(tensor.as_slice(), &[1, 9]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::Validation`] for a wrong rank, an out-of-range
    /// coordinate or offset arithmetic overflow.
    pub fn get_mut(&mut self, indices: &[usize]) -> crate::Result<&mut T> {
        let offset = self.linear_offset(indices)?;
        self.host_data_mut().get_mut(offset).ok_or_else(|| {
            crate::Error::validation("TypedTensor::get_mut", ValidationError::ViewOutOfBounds)
        })
    }

    /// Consume this tensor and return the original host vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// assert_eq!(tensor.into_host_vec(), vec![1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_host_vec(self) -> Vec<T> {
        self.storage.data.into_vec()
    }

    /// Consume this tensor and return its shape and column-major host vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2, 1], vec![1, 2])?;
    /// let (shape, data) = tensor.into_vec_col_major();
    /// assert_eq!(shape, vec![2, 1]);
    /// assert_eq!(data, vec![1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_vec_col_major(self) -> (Vec<usize>, Vec<T>) {
        let shape = self.shape().to_vec();
        (shape, self.into_host_vec())
    }

    /// Make an explicit independent host copy with the same placement.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<String, DynRank, Host>::from_host_vec_col_major(vec![1], vec!["a".into()])?;
    /// let copy = tensor.clone();
    /// assert_eq!(copy[&[0]], "a");
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn duplicate(&self) -> Self
    where
        T: Clone,
    {
        let mut copy = typed_host_tensor_from_vec_col_major::<T, R>(
            self.shape.clone(),
            self.as_slice().to_vec(),
            "TypedTensor::duplicate",
        )
        .unwrap_or_else(|err| unreachable!("a validated host owner re-copies: {err}"));
        copy.placement = self.placement.clone();
        copy
    }

    /// Move this host owner into the runtime union without copying or regrouping.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{Dynamic, DynRank, Host, TypedTensor};
    /// let host = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![1], vec![7])?;
    /// let dynamic: TypedTensor<i32, DynRank, Dynamic> = host.into_dynamic();
    /// assert_eq!(dynamic.host_data()?, &[7]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_dynamic(self) -> TypedTensor<T, R, Dynamic> {
        TypedTensor {
            shape: self.shape,
            placement: self.placement,
            layout: self.layout,
            storage: DynamicStorage::Host(self.storage),
        }
    }

    /// Borrow this host owner as a typed view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// let view = tensor.as_view();
    /// assert_eq!(view.shape(), &[2]);
    /// assert_eq!(view.as_host_slice(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_view(&self) -> TypedTensorView<'_, T, R, Host> {
        TypedTensorView {
            buffer: TensorStorageRef::Host(self.as_slice()),
            root: None,
            layout: self.layout(),
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        }
    }

    /// Exclusively borrow this host owner as a mutable typed view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// tensor.as_view_mut().as_host_slice_mut()[1] = 4;
    /// assert_eq!(tensor.as_slice(), &[1, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_view_mut(&mut self) -> TypedTensorViewMut<'_, T, R, Host> {
        let layout = self.layout();
        let placement = self.placement.clone();
        TypedTensorViewMut {
            buffer: TensorStorageRefMut::Host(self.storage.data.as_mut_slice()),
            root: None,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        }
    }
}

impl<T: Clone, R: TensorRank> Clone for TypedTensor<T, R, Host> {
    fn clone(&self) -> Self {
        self.duplicate()
    }
}

impl<T, R: TensorRank> std::ops::Index<&[usize]> for TypedTensor<T, R, Host> {
    type Output = T;

    /// # Panics
    ///
    /// Panics when the index has the wrong rank or is out of bounds, like any
    /// other slice indexing. Use [`TypedTensor::get`] for the checked form.
    fn index(&self, indices: &[usize]) -> &T {
        match self.get(indices) {
            Ok(value) => value,
            Err(err) => panic!("TypedTensor host index {indices:?} is invalid: {err}"),
        }
    }
}

impl<T, R: TensorRank> std::ops::IndexMut<&[usize]> for TypedTensor<T, R, Host> {
    /// # Panics
    ///
    /// Panics when the index has the wrong rank or is out of bounds, like any
    /// other slice indexing. Use [`TypedTensor::get_mut`] for the checked form.
    fn index_mut(&mut self, indices: &[usize]) -> &mut T {
        match self.get_mut(indices) {
            Ok(value) => value,
            Err(err) => panic!("TypedTensor host index {indices:?} is invalid: {err}"),
        }
    }
}

/// Read mapping over a host slice, as one owning guard.
fn host_read_guard<T>(data: &[T]) -> HostReadGuard<'_, T> {
    HostReadGuard::new(data)
}

/// Write mapping over an exclusive host slice, as one owning guard.
fn host_write_guard<T: Clone>(data: &mut [T]) -> HostWriteGuard<'_, T> {
    let len = data.len();
    HostWriteGuard::new(len, move |source| {
        data.clone_from_slice(source);
        Ok(())
    })
}

/// Promote a plain host payload into a group-backed root without copying elements.
fn promote_host_group<T: TensorScalar, R: TensorRank>(
    shape: R::Shape,
    host: HostData<T>,
) -> crate::Result<OwnedTensorGroup<R>> {
    let recycler = host.recycler.clone();
    let mut group = OwnedTensorGroup::from_host_vec(shape, host.into_vec())?;
    if let Some(recycler) = recycler {
        group
            .group
            .set_host_recycler(group.allocation_index.index(), recycler)
            .map_err(|error| crate::Error::runtime_state_source("TypedTensor::promote", error))?;
    }
    Ok(group)
}

impl<T, R: TensorRank> TypedTensor<T, R, Host> {
    /// Borrow the owned host elements through one owning read mapping.
    ///
    /// The guard retains the exclusive borrow of this owner, so no other access
    /// can overlap it while the mapping is alive.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// let guard = tensor.map_read();
    /// assert_eq!(&guard[..], &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn map_read(&self) -> HostReadGuard<'_, T> {
        host_read_guard(self.as_slice())
    }

    /// Exclusively borrow the owned host elements through one owning write mapping.
    ///
    /// Publish data with [`HostWriteGuard::copy_from_slice`]; the mapping is
    /// released when the guard is dropped.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2])?;
    /// tensor.map_write().copy_from_slice(&[3, 4]).unwrap();
    /// assert_eq!(tensor.as_slice(), &[3, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn map_write(&mut self) -> HostWriteGuard<'_, T>
    where
        T: Clone,
    {
        host_write_guard(self.host_data_mut())
    }
}

impl<T: TensorScalar, R: TensorRank> TypedTensor<T, R, Host> {
    /// Promote this plain host owner into a group-backed owner.
    ///
    /// The payload is adopted as a provider root: no element is copied and a
    /// pooled return target survives the promotion.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let host = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let gpu = host.promote()?;
    /// assert_eq!(gpu.host_data()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`crate::Error::RuntimeState`] when the new allocation group
    /// rejects the promoted root or its recycler.
    pub fn promote(self) -> crate::Result<TypedTensor<T, R, Gpu>> {
        let TypedTensor {
            shape,
            placement,
            layout,
            storage,
        } = self;
        let group = promote_host_group(shape.clone(), storage.data)?;
        Ok(TypedTensor {
            shape,
            placement,
            layout,
            storage: GroupStorage {
                group: Box::new(group),
            },
        })
    }
}

impl<T, R: TensorRank> TypedTensor<T, R, Dynamic> {
    /// Borrow host elements through one owning read mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2])?;
    /// let guard = tensor.map_read()?;
    /// assert_eq!(&guard[..], &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::HostAccess`] with
    /// [`HostAccessError::Unsupported`] when the group's allocation is not
    /// host-accessible.
    pub fn map_read(&self) -> crate::Result<HostReadGuard<'_, T>> {
        match &self.storage {
            DynamicStorage::Host(host) => Ok(host_read_guard(host.data.as_slice())),
            DynamicStorage::Group(core) => core.group.host_slice::<T>().map(host_read_guard),
        }
    }

    /// Exclusively borrow host elements through one owning write mapping.
    ///
    /// Publish data with [`HostWriteGuard::copy_from_slice`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Error, TypedTensor};
    /// let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2])?;
    /// tensor
    ///     .map_write()?
    ///     .copy_from_slice(&[3, 4])
    ///     .map_err(|err| Error::host_access("map_write", err))?;
    /// assert_eq!(tensor.host_data()?, &[3, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::HostAccess`] with
    /// [`HostAccessError::Unsupported`] when the group's allocation is not
    /// host-writable.
    pub fn map_write(&mut self) -> crate::Result<HostWriteGuard<'_, T>>
    where
        T: Clone,
    {
        match &mut self.storage {
            DynamicStorage::Host(host) => Ok(host_write_guard(host.data.as_mut_slice())),
            DynamicStorage::Group(core) => core.group.host_slice_mut::<T>().map(host_write_guard),
        }
    }
}

impl<T, R: TensorRank> TypedTensor<T, R, Gpu> {
    /// Adopt a backend-owned buffer as a statically group-backed owner.
    ///
    /// A host buffer belongs to the [`Host`] representation; pass it to
    /// `TypedTensor::<T, R, Host>` instead of silently regrouping it here.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{BackendStorageHandle, Placement, StorageBuffer, TypedTensor};
    /// let handle = BackendStorageHandle::<f32>::new_with_len(1, 2);
    /// let gpu = TypedTensor::<f32, tenferro_tensor::DynRank, tenferro_tensor::Gpu>::from_backend_buffer_col_major(
    ///     vec![2],
    ///     StorageBuffer::Backend(Box::new(handle)),
    ///     Placement::default(),
    /// )?;
    /// assert!(gpu.is_backend_buffer());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`crate::Error::Validation`] when the compact shape disagrees
    /// with the buffer length, [`crate::Error::RuntimeState`] for a host buffer,
    /// or a runtime-state error when the allocation group cannot be built.
    pub fn from_backend_buffer_col_major(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        buffer: StorageBuffer<T>,
        placement: Placement,
    ) -> crate::Result<Self>
    where
        T: TensorScalar + Send + Sync + 'static,
    {
        let op = "TypedTensor::from_backend_buffer_col_major";
        let layout = try_compact_layout(shape, op)?;
        try_checked_shape_len(layout.shape(), buffer.len(), op)?;
        let group_shape = R::shape_from_vec(shape_vec(layout.shape()))
            .map_err(|err| tensor_layout_error(op, err))?;
        match buffer {
            StorageBuffer::Host(_) => Err(crate::Error::runtime_state(
                op,
                "a host buffer is plain host storage; use TypedTensor::<T, R, Host>::from_vec_col_major",
            )),
            StorageBuffer::Backend(buffer) => {
                let group = OwnedTensorGroup::from_backend_buffer(
                    group_shape.clone(),
                    StorageBuffer::Backend(buffer),
                    placement.clone(),
                )?;
                Ok(TypedTensor {
                    shape: group_shape,
                    placement,
                    layout: stride_vec(layout.strides()),
                    storage: GroupStorage {
                        group: Box::new(group),
                    },
                })
            }
        }
    }

    /// Move this group-backed owner into the runtime union without rewriting it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Dynamic, DynRank, Host, TypedTensor};
    /// let gpu = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![7.0])?.promote()?;
    /// let dynamic: TypedTensor<f64, DynRank, Dynamic> = gpu.into_dynamic();
    /// assert_eq!(dynamic.host_data()?, &[7.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_dynamic(self) -> TypedTensor<T, R, Dynamic> {
        TypedTensor {
            shape: self.shape,
            placement: self.placement,
            layout: self.layout,
            storage: DynamicStorage::Group(self.storage),
        }
    }

    /// Whether this group's descriptor names a non-CPU provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let gpu = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![7.0])?.promote()?;
    /// // A promoted host payload stays on the CPU provider.
    /// assert!(!gpu.is_backend_buffer());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_backend_buffer(&self) -> bool {
        self.storage.group.is_backend_buffer()
    }

    /// Borrow host elements when the group's allocation is host-accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let gpu = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?.promote()?;
    /// assert_eq!(gpu.host_data()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::RuntimeState`] for a device-only allocation.
    pub fn host_data(&self) -> crate::Result<&[T]> {
        self.storage.group.host_slice::<T>()
    }

    /// Exclusively borrow host elements when the group's allocation is host-accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensor};
    /// let mut gpu = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?.promote()?;
    /// gpu.host_data_mut()?[0] = 3.0;
    /// assert_eq!(gpu.host_data()?, &[3.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::RuntimeState`] for a device-only allocation.
    pub fn host_data_mut(&mut self) -> crate::Result<&mut [T]> {
        self.storage.group.host_slice_mut::<T>()
    }

    /// Prepare this group-backed owner for one provider-native read binding.
    ///
    /// # Errors
    /// Returns a provider preparation error when the group has no device-read
    /// path for this layout.
    #[doc(hidden)]
    pub fn prepare_device_read(
        &self,
        op: &'static str,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>>
    where
        T: TensorScalar + 'static,
    {
        let layout = self.layout();
        self.storage
            .group
            .prepare_device_read_for_layout::<T>(&layout)
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }

    /// Prepare this group-backed owner for one provider-native write binding.
    ///
    /// # Errors
    /// Returns a provider preparation error when the group has no device-write
    /// path for this layout.
    #[doc(hidden)]
    pub fn prepare_device_write(
        &mut self,
        op: &'static str,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>>
    where
        T: TensorScalar + 'static,
    {
        let layout = self.layout();
        self.storage
            .group
            .prepare_device_write_for_layout::<T>(&layout)
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }
}

impl<T, R: TensorRank> TypedTensor<T, R, Dynamic> {
    /// Checked narrowing to the statically host-owned representation.
    ///
    /// Fails without consuming ownership of the source tensor.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let dynamic = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![7])?;
    /// let host = dynamic.into_host().unwrap();
    /// assert_eq!(host.as_slice(), &[7]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when it is
    /// group-backed rather than plain host storage.
    #[allow(clippy::result_large_err)]
    pub fn into_host(self) -> Result<TypedTensor<T, R, Host>, ReinterpretError<Self>> {
        let TypedTensor {
            shape,
            placement,
            layout,
            storage,
        } = self;
        match storage {
            DynamicStorage::Host(host) => Ok(TypedTensor {
                shape,
                placement,
                layout,
                storage: host,
            }),
            DynamicStorage::Group(group) => Err(ReinterpretError::new(
                TypedTensor {
                    shape,
                    placement,
                    layout,
                    storage: DynamicStorage::Group(group),
                },
                crate::Error::runtime_state(
                    "TypedTensor::into_host",
                    "the tensor is group-backed rather than plain host storage",
                ),
            )),
        }
    }

    /// Checked narrowing to the group-backed representation.
    ///
    /// Fails without consuming ownership of the source tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Gpu, Host, TypedTensor};
    /// let host = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![2.0])?;
    /// let dynamic = host.promote()?.into_dynamic();
    /// let gpu: TypedTensor<f64, DynRank, Gpu> = dynamic.into_gpu().map_err(|f| f.into_parts().1)?;
    /// assert_eq!(gpu.host_data()?, &[2.0]);
    /// let plain = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0])?;
    /// assert!(plain.into_gpu().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when it is a
    /// plain host owner rather than a group-backed one.
    #[allow(clippy::result_large_err)]
    pub fn into_gpu(self) -> Result<TypedTensor<T, R, Gpu>, ReinterpretError<Self>> {
        let TypedTensor {
            shape,
            placement,
            layout,
            storage,
        } = self;
        match storage {
            DynamicStorage::Group(group) => Ok(TypedTensor {
                shape,
                placement,
                layout,
                storage: group,
            }),
            DynamicStorage::Host(host) => Err(ReinterpretError::new(
                TypedTensor {
                    shape,
                    placement,
                    layout,
                    storage: DynamicStorage::Host(host),
                },
                crate::Error::runtime_state(
                    "TypedTensor::into_gpu",
                    "the tensor is a plain host owner rather than group-backed storage",
                ),
            )),
        }
    }
}

impl<T, R: TensorRank> TypedTensor<T, R, Dynamic> {
    /// Layout that reflects the owned storage rather than the compact shape.
    ///
    /// [`Self::layout`] reports the compact construction layout. A
    /// root-preserving extraction may instead carry a nonzero descriptor offset
    /// and fewer logical elements than the backing root, so callers that
    /// canonicalize an owned tensor must use this accessor.
    pub(crate) fn physical_layout(&self) -> TensorLayout<R> {
        match &self.storage {
            DynamicStorage::Host(_) => TensorLayout::<R>::compact(self.shape.clone())
                .unwrap_or_else(|err| unreachable!("validated owned shape: {err}")),
            DynamicStorage::Group(core) => {
                let descriptor = core
                    .group
                    .group
                    .descriptor_layout(core.group.slot)
                    .unwrap_or_else(|| unreachable!("a live group carries its descriptor layout"));
                let shape = R::shape_from_vec(descriptor.shape().to_vec().into())
                    .unwrap_or_else(|err| unreachable!("validated descriptor shape: {err:?}"));
                let strides = R::strides_from_vec(descriptor.strides().to_vec().into())
                    .unwrap_or_else(|err| unreachable!("validated descriptor strides: {err:?}"));
                let buffer_len = core
                    .group
                    .group
                    .descriptor_backing_len(core.group.slot)
                    .unwrap_or_else(|| unreachable!("a live group carries its root length"));
                TensorLayout::<R>::from_parts(shape, strides, descriptor.offset(), buffer_len)
                    .unwrap_or_else(|err| unreachable!("validated descriptor layout: {err:?}"))
            }
        }
    }
}

impl<T, R: TensorRank, D: Representation> std::fmt::Debug for TypedTensor<T, R, D> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TypedTensor")
            .field("shape", &self.shape.as_ref())
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

impl<T, R: TensorRank, D: Representation> TypedTensor<T, R, D> {
    /// Number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    pub fn n_elements(&self) -> usize {
        // Invariant: owned tensor constructors validate compact shape length against buffer length.
        match try_shape_product(self.shape(), "TypedTensor::n_elements") {
            Ok(n) => n,
            Err(err) => {
                unreachable!("TypedTensor compact shape is validated at construction: {err}")
            }
        }
    }

    /// Tensor shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }

    /// Column-major element strides of this compact tensor.
    ///
    /// The layout is cached at construction, so views of a tensor borrow it
    /// rather than computing (and allocating) it once per operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.strides(), &[1, 2]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.layout
    }

    /// Tensor rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Tensor layout metadata.
    ///
    /// Owned typed tensors are always compact column-major layouts.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.layout().strides(), &[1, 2]);
    /// ```
    pub fn layout(&self) -> TensorLayout<R> {
        TensorLayout::compact(self.shape.clone())
            .unwrap_or_else(|err| unreachable!("validated owned shape: {err}"))
    }

    /// Return placement metadata for this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryKind, TypedTensor};
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    /// assert_eq!(t.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Replace placement metadata without changing the storage buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryKind, Placement, TypedTensor};
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    /// t.set_placement(Placement {
    ///     memory_kind: MemoryKind::PinnedHost,
    ///     device: None,
    ///     cpu_affinity: None,
    /// });
    /// assert_eq!(t.placement().memory_kind, MemoryKind::PinnedHost);
    /// ```
    pub fn set_placement(&mut self, placement: Placement) {
        self.placement = placement;
    }

    /// Replace only CPU routing/locality metadata without changing storage.
    ///
    /// Device, memory kind, backend allocation domain, and allocation identity
    /// remain unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{CpuDomainId, TypedTensor};
    ///
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0])?;
    /// tensor.set_cpu_affinity(Some(CpuDomainId::new(4)));
    /// assert_eq!(tensor.placement().cpu_affinity, Some(CpuDomainId::new(4)));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn set_cpu_affinity(&mut self, cpu_affinity: Option<CpuDomainId>) {
        self.placement.cpu_affinity = cpu_affinity;
    }

    /// Compute the checked compact column-major offset of an element.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<String>::from_vec_col_major([2, 3], vec![String::new(); 6])?;
    /// assert_eq!(tensor.linear_offset(&[1, 2])?, 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Wrong rank, out-of-range coordinates or arithmetic overflow return
    /// [`crate::Error::Validation`].
    pub fn linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), indices, "TypedTensor::linear_offset")
    }

    /// Consume this tensor and return its layout metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert!(t.into_layout().is_compact_col_major().unwrap());
    /// ```
    pub fn into_layout(self) -> TensorLayout<R> {
        self.layout()
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]).unwrap();
    /// assert_eq!(t.layout_linear_offset(&[1, 2])?, 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), indices, "TypedTensor::layout_linear_offset")
    }

    /// Return whether this owned tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2]).unwrap();
    /// assert!(t.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        self.layout()
            .is_compact_col_major()
            .map_err(|err| tensor_layout_error("TypedTensor::is_col_major_contiguous", err))
    }

    /// Return a compact string summary of this tensor's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2]).unwrap();
    /// assert!(t.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        let layout = self.layout();
        layout_summary(self.shape(), layout.strides(), layout.offset())
    }

    /// Assert this tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2]).unwrap();
    /// t.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// tensor is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        let layout = self.layout();
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            layout.strides(),
            layout.offset(),
            "TypedTensor::assert_col_major_contiguous",
        )
    }
}

struct HostData<T> {
    buffer: StorageBuffer<T>,
    recycler: Option<std::sync::Weak<dyn crate::HostBufferRecycler<T>>>,
}

impl<T> std::fmt::Debug for HostData<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostData")
            .field("pooled", &self.recycler.is_some())
            .finish_non_exhaustive()
    }
}

impl<T> HostData<T> {
    fn new(data: Vec<T>) -> Self {
        Self {
            buffer: StorageBuffer::Host(data),
            recycler: None,
        }
    }

    fn into_vec(mut self) -> Vec<T> {
        self.recycler = None;
        match std::mem::replace(&mut self.buffer, StorageBuffer::Host(Vec::new())) {
            StorageBuffer::Host(data) => data,
            StorageBuffer::Backend(_) => unreachable!("plain host storage is host-owned"),
        }
    }

    fn as_slice(&self) -> &[T] {
        match &self.buffer {
            StorageBuffer::Host(data) => data,
            StorageBuffer::Backend(_) => unreachable!("plain host storage is host-owned"),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.buffer {
            StorageBuffer::Host(data) => data,
            StorageBuffer::Backend(_) => unreachable!("plain host storage is host-owned"),
        }
    }
}

impl<T> Drop for HostData<T> {
    fn drop(&mut self) {
        let Some(recycler) = self.recycler.take().and_then(|weak| weak.upgrade()) else {
            return;
        };
        if let StorageBuffer::Host(data) = &mut self.buffer {
            recycler.recycle(std::mem::take(data));
        }
    }
}

/// The sole owner handle for host tensors. The allocation group owns the
/// provider root; the descriptor slot carries only the logical view metadata.
pub(crate) struct OwnedTensorGroup<R: TensorRank> {
    group: AllocationGroup,
    slot: DescriptorSlot,
    allocation_index: AllocationSlot,
    // INVARIANT: this non-owning address points into the group root, whose host
    // vector cannot resize while the owning tensor is borrowed. It is stored as
    // `NonZeroUsize` rather than `NonNull<u8>`: the niche keeps the field at one
    // word, and unlike `NonNull` it preserves the `Send`/`Sync` auto traits the
    // typed tensor contract requires.
    host_ptr: Option<NonZeroUsize>,
    host_byte_len: usize,
    _rank: PhantomData<R>,
}

impl<R: TensorRank> Debug for OwnedTensorGroup<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedTensorGroup")
            .field("slot", &self.slot)
            .finish_non_exhaustive()
    }
}

impl<R: TensorRank> OwnedTensorGroup<R> {
    fn from_host_vec<T: TensorScalar>(shape: R::Shape, data: Vec<T>) -> crate::Result<Self> {
        let (group, slot) = AllocationGroup::from_host_vec::<T, R>(shape, data)
            .map_err(|error| group_error("TypedTensor::from_host_vec", error))?;
        let allocation_index = group
            .allocation_index(slot)
            .map_err(|error| group_error("TypedTensor::from_host_vec", error))?;
        let (host_ptr, host_byte_len) = host_metadata::<T>(&group, slot);
        Ok(Self {
            group,
            slot,
            allocation_index,
            host_ptr,
            host_byte_len,
            _rank: PhantomData,
        })
    }

    fn from_backend_buffer<T: TensorScalar + Send + Sync + 'static>(
        shape: R::Shape,
        buffer: StorageBuffer<T>,
        placement: Placement,
    ) -> crate::Result<Self> {
        let (mut group, slot) = AllocationGroup::from_backend_buffer::<T, R>(shape, buffer)
            .map_err(|error| group_error("TypedTensor::from_backend_buffer", error))?;
        group
            .set_descriptor_placement(slot, placement)
            .map_err(|error| group_error("TypedTensor::from_backend_buffer", error))?;
        let allocation_index = group
            .allocation_index(slot)
            .map_err(|error| group_error("TypedTensor::from_backend_buffer", error))?;
        let (host_ptr, host_byte_len) = host_metadata::<T>(&group, slot);
        Ok(Self {
            group,
            slot,
            allocation_index,
            host_ptr,
            host_byte_len,
            _rank: PhantomData,
        })
    }

    fn view<T: TensorScalar>(&self) -> crate::Result<GroupReadView<'_, T, R>> {
        self.group
            .view(self.slot)
            .map_err(|error| group_error("TypedTensor::group_view", error))
    }

    fn view_dyn<T: TensorScalar>(&self) -> crate::Result<GroupReadView<'_, T, DynRank>> {
        self.group
            .view(self.slot)
            .map_err(|error| group_error("TypedTensor::group_view", error))
    }

    fn view_mut<T: TensorScalar>(&mut self) -> crate::Result<GroupWriteView<'_, T, R>> {
        self.group
            .view_mut(self.slot)
            .map_err(|error| group_error("TypedTensor::group_view_mut", error))
    }

    fn view_mut_dyn<T: TensorScalar>(&mut self) -> crate::Result<GroupWriteView<'_, T, DynRank>> {
        self.group
            .view_mut(self.slot)
            .map_err(|error| group_error("TypedTensor::group_view_mut", error))
    }

    fn prepare_device_read_for_layout<T: TensorScalar>(
        &self,
        layout: &TensorLayout<R>,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>> {
        self.group
            .prepare_device_read_for_layout::<T, R>(self.slot, layout)
            .map_err(|error| {
                crate::Error::runtime_state_source("TypedTensor::prepare_device_read", error)
            })
    }

    fn prepare_device_write_for_layout<T: TensorScalar>(
        &mut self,
        layout: &TensorLayout<R>,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>> {
        self.group
            .prepare_device_write_for_layout::<T, R>(self.slot, layout)
            .map_err(|error| {
                crate::Error::runtime_state_source("TypedTensor::prepare_device_write", error)
            })
    }

    fn host_buffer<T: 'static>(&self) -> Option<&StorageBuffer<T>> {
        self.group.host_buffer_at::<T>(self.allocation_index)
    }

    fn host_slice<T>(&self) -> crate::Result<&[T]> {
        let Some(pointer) = self.host_ptr else {
            return Err(crate::Error::runtime_state(
                "TypedTensor::host_data",
                "backend storage cannot be borrowed as host data; download explicitly first",
            ));
        };
        let element_size = size_of::<T>();
        let Some(element_count) = self.host_byte_len.checked_div(element_size) else {
            return Err(crate::Error::runtime_state(
                "TypedTensor::host_data",
                "host allocation byte length is not aligned to the requested dtype",
            ));
        };
        // SAFETY: the pointer and byte length were captured from the unique
        // root's full host allocation; the root cannot resize while borrowed.
        Ok(unsafe { std::slice::from_raw_parts(pointer.get() as *const T, element_count) })
    }

    fn host_slice_mut<T>(&mut self) -> crate::Result<&mut [T]> {
        let Some(pointer) = self.host_ptr else {
            return Err(crate::Error::runtime_state(
                "TypedTensor::host_data_mut",
                "backend storage cannot be borrowed as host data; download explicitly first",
            ));
        };
        let element_size = size_of::<T>();
        let Some(element_count) = self.host_byte_len.checked_div(element_size) else {
            return Err(crate::Error::runtime_state(
                "TypedTensor::host_data_mut",
                "host allocation byte length is not aligned to the requested dtype",
            ));
        };
        // SAFETY: the pointer and byte length were captured from the unique
        // root; this method has the only mutable borrow of that root.
        Ok(unsafe { std::slice::from_raw_parts_mut(pointer.get() as *mut T, element_count) })
    }

    fn backend_buffer<T: 'static>(&self) -> Option<&StorageBuffer<T>> {
        self.group.backend_buffer::<T>(self.slot)
    }

    fn backend_buffer_mut<T: 'static>(&mut self) -> Option<&mut StorageBuffer<T>> {
        self.group.backend_buffer_mut::<T>(self.slot)
    }

    // INVARIANT: the failure carrier must return the unchanged group owner, so
    // the wide `(Self, Error)` pair is deliberate rather than a boxing bug.
    #[allow(clippy::result_large_err)]
    fn into_host_vec<T: 'static>(self) -> std::result::Result<Vec<T>, (Self, crate::Error)> {
        if self.backend_buffer::<T>().is_some() {
            return Err((
                self,
                crate::Error::runtime_state(
                    "TypedTensor::into_host_vec",
                    "backend buffers cannot be exported as host Vec; download the tensor first",
                ),
            ));
        }
        let OwnedTensorGroup {
            group,
            slot,
            allocation_index,
            host_ptr,
            host_byte_len,
            ..
        } = self;
        match group.into_host_vec::<T>(slot) {
            Ok(data) => Ok(data),
            Err((group, error)) => Err((
                OwnedTensorGroup {
                    group,
                    slot,
                    allocation_index,
                    host_ptr,
                    host_byte_len,
                    _rank: PhantomData,
                },
                crate::Error::runtime_state("TypedTensor::into_host_vec", error),
            )),
        }
    }

    fn into_parts(self) -> (AllocationGroup, DescriptorSlot) {
        (self.group, self.slot)
    }

    #[allow(clippy::result_large_err)]
    fn reinterpret<T: TensorScalar, U: TensorScalar>(
        self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> Result<OwnedTensorGroup<DynRank>, (Self, crate::Error)> {
        let OwnedTensorGroup {
            group,
            slot,
            allocation_index,
            host_ptr,
            host_byte_len,
            _rank: _,
        } = self;
        match group.reinterpret_descriptor::<T, U>(slot, shape, strides, offset) {
            Ok(group) => Ok(OwnedTensorGroup {
                group,
                slot,
                allocation_index,
                host_ptr,
                host_byte_len,
                _rank: PhantomData,
            }),
            Err((group, error)) => Err((
                OwnedTensorGroup {
                    group,
                    slot,
                    allocation_index,
                    host_ptr,
                    host_byte_len,
                    _rank: PhantomData,
                },
                group_error("TypedTensor::reinterpret", error),
            )),
        }
    }
}

fn host_metadata<T: 'static>(
    group: &AllocationGroup,
    slot: DescriptorSlot,
) -> (Option<NonZeroUsize>, usize) {
    group
        .host_root_metadata::<T>(slot)
        .map_or((None, 0), |(pointer, byte_len)| {
            // INVARIANT: `host_root_metadata` reports the base address of the group's live
            // host allocation, which is non-null for any slice, empty included.
            (NonZeroUsize::new(pointer), byte_len)
        })
}

fn group_error(op: &'static str, error: GroupError) -> crate::Error {
    crate::Error::runtime_state(op, error.to_string())
}

/// Borrowed tensor buffer reference used by read-only typed views.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorStorageRef;
///
/// let data = [1_i32, 2];
/// let buffer = TensorStorageRef::Host(&data);
/// assert_eq!(buffer.len(), 2);
/// ```
#[derive(Debug)]
pub enum TensorStorageRef<'a, T> {
    Host(&'a [T]),
    Backend(&'a dyn BackendStorage<T>),
    #[doc(hidden)]
    Root(&'a dyn BackendAllocation),
}

impl<T> Clone for TensorStorageRef<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Host(data) => Self::Host(data),
            Self::Backend(buffer) => Self::Backend(*buffer),
            Self::Root(allocation) => Self::Root(*allocation),
        }
    }
}

impl<T: 'static> TensorStorageRef<'_, T> {
    /// Return the logical length of the backing allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorStorageRef;
    ///
    /// let data = [1_i32, 2, 3];
    /// assert_eq!(TensorStorageRef::Host(&data).len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
            Self::Root(allocation) => allocation
                .root_extent()
                .byte_len()
                .checked_div(std::mem::size_of::<T>())
                .unwrap_or(0),
        }
    }

    /// Return whether the backing allocation is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorStorageRef;
    ///
    /// let data: [f64; 0] = [];
    /// assert!(TensorStorageRef::Host(&data).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed tensor buffer reference used by mutable typed views.
///
/// Backend buffers can be represented for residency metadata, but this crate
/// does not expose host mutation for backend-native allocations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorStorageRefMut;
///
/// let mut data = [1_i32, 2];
/// let buffer = TensorStorageRefMut::Host(&mut data);
/// assert_eq!(buffer.len(), 2);
/// ```
#[derive(Debug)]
pub enum TensorStorageRefMut<'a, T> {
    Host(&'a mut [T]),
    Backend(&'a mut dyn BackendStorage<T>),
}

impl<T: 'static> TensorStorageRefMut<'_, T> {
    /// Return the logical length of the backing allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorStorageRefMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// assert_eq!(TensorStorageRefMut::Host(&mut data).len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
        }
    }

    /// Return whether the backing allocation is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorStorageRefMut;
    ///
    /// let mut data: [f64; 0] = [];
    /// assert!(TensorStorageRefMut::Host(&mut data).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Read-only borrowed view of typed tensor storage with arbitrary strides.
///
/// `TypedTensorView` is the typed representation for layout-only tensor
/// transformations. It borrows an existing host or backend allocation and
/// carries a logical shape, strides, and an offset. Slicing, reshaping when
/// stride-compatible, and [`transpose_view`](TypedTensorView::transpose_view)
/// update only metadata and do not copy storage.
///
/// Materialize through [`TensorStructural::to_contiguous_read`](crate::TensorStructural::to_contiguous_read)
/// on the active backend session when a compact owned [`TypedTensor`] is
/// required. Use [`TypedTensorView::as_slice`] only when the current view is
/// contiguous in the requested layout.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Rank, TypedTensorView};
///
/// let data = [1_i32, 2, 3, 4];
/// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data)?;
/// assert_eq!(view.get(&[1, 1]), Some(&4));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
/// Borrowed typed view of one tensor representation.
///
/// The buffer stays concrete on purpose: a generic-associated buffer would make
/// the projection invariant in `'a`, and the borrowed-read surface relies on
/// `TensorRead<'long>` shortening to `TensorRead<'short>`. `D` therefore marks
/// which representation produced the view - `Host` is only ever built from host
/// slices, with no retained region.
pub struct TypedTensorView<'a, T, R: TensorRank = DynRank, D: Representation = Dynamic> {
    buffer: TensorStorageRef<'a, T>,
    root: Option<GroupReadView<'a, T, R>>,
    layout: TensorLayout<R>,
    placement: Placement,
    _representation: std::marker::PhantomData<D>,
}

impl<'a, T, R: TensorRank, D: Representation> Clone for TypedTensorView<'a, T, R, D> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            root: self.root.clone(),
            layout: self.layout.clone(),
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        }
    }
}

impl<'a, T, R: TensorRank, D: Representation> std::fmt::Debug for TypedTensorView<'a, T, R, D> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TypedTensorView")
            .field("shape", &self.layout.shape())
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

impl<'a, T: 'static> TypedTensorView<'a, T, DynRank> {
    /// Create a borrowed dynamic-rank view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_col_major(&[2, 2], &data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when compact
    /// strides or reachable bounds overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// requested shape reaches beyond `data`.
    pub fn from_col_major(shape: &[usize], data: &'a [T]) -> crate::Result<Self> {
        let layout = TensorLayout::<DynRank>::compact(shape_vec(shape))
            .map_err(|err| tensor_layout_error("TypedTensorView::from_col_major", err))?;
        Self::from_buffer_ref(
            shape_vec(layout.shape()),
            stride_vec(layout.strides()),
            layout.offset(),
            TensorStorageRef::Host(data),
            default_placement(),
            "TypedTensorView::from_col_major",
        )
    }

    /// Create a borrowed host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![-1], 2, &data)?;
    /// assert_eq!(view.get(&[2]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `shape` and
    /// `strides` have different ranks,
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// reachable layout exceeds `data`, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when layout
    /// arithmetic overflows.
    pub fn from_slice(
        shape: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: isize,
        data: &'a [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref(
            shape_vec(shape.as_ref()),
            stride_vec(strides.as_ref()),
            offset,
            TensorStorageRef::Host(data),
            default_placement(),
            "TypedTensorView::from_slice",
        )
    }
}

impl<'a, T: 'static, R: TensorRank, D: Representation> TypedTensorView<'a, T, R, D> {
    /// Create a rank-generic borrowed host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data)?;
    /// assert_eq!(view.get(&[1, 1]), Some(&4));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when the typed
    /// rank does not match `shape` or `strides`,
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// reachable layout exceeds `data`, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when layout
    /// arithmetic overflows.
    pub fn from_slice_ranked(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref(
            shape,
            strides,
            offset,
            TensorStorageRef::Host(data),
            default_placement(),
            "TypedTensorView::from_slice_ranked",
        )
    }
}

impl<'a, T: 'static, R: TensorRank, D: Representation> TypedTensorView<'a, T, R, D> {
    fn from_buffer_ref(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        buffer: TensorStorageRef<'a, T>,
        placement: Placement,
        op: &'static str,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            buffer,
            root: None,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, T: 'static, R: TensorRank> TypedTensorView<'a, T, R, Host> {
    /// Create a representation-marked view over an explicit host layout.
    ///
    /// A `Host`-marked view is only ever built from a host slice, so it carries
    /// no retained region and its host slice needs no runtime check.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DynRank, Host, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view: TypedTensorView<'_, i32, DynRank, Host> =
    ///     TypedTensorView::from_host_slice(vec![2, 2], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.as_host_slice(), &[1, 2, 3, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `strides`
    /// has a different rank, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] /
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when the
    /// reachable layout leaves `data` or overflows.
    pub fn from_host_slice(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a [T],
    ) -> crate::Result<Self> {
        // Built here rather than through the shared helper so that the Host
        // marker can only ever be paired with a host slice.
        let buffer = TensorStorageRef::Host(data);
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error("TypedTensorView::from_host_slice", err))?;
        Ok(Self {
            buffer,
            root: None,
            layout,
            placement: default_placement(),
            _representation: std::marker::PhantomData,
        })
    }

    /// Borrow the host elements without a runtime representation check.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DynRank, Host, TypedTensorView};
    ///
    /// let data = [1_i32, 2];
    /// let view: TypedTensorView<'_, i32, DynRank, Host> =
    ///     TypedTensorView::from_host_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.as_host_slice(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Panics only if a `Host`-marked view was built from non-host storage,
    /// which no constructor in this crate does.
    pub fn as_host_slice(&self) -> &'a [T] {
        match &self.buffer {
            TensorStorageRef::Host(data) => data,
            // INVARIANT: `Host`-marked views are constructed only by
            // `from_host_slice` and by `TypedTensor<_, _, Host>::as_view`.
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => {
                unreachable!("a Host-marked view always borrows host storage")
            }
        }
    }
}

impl<'a, T: 'static, R: TensorRank, D: Representation> TypedTensorView<'a, T, R, D> {
    /// Erase a borrowed view's rank and dtype for session dispatch without
    /// allocating tensor storage or promoting it into an allocation group.
    /// Common ranks use the existing inline shape and stride metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    /// let data = [1.0_f64, 2.0];
    /// let view = TypedTensorView::<_, Rank<1>>::from_slice_ranked([2], [1], 0, &data)?;
    /// let read = view.into_tensor_read()?;
    /// assert_eq!(read.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`crate::Error::Validation`] with
    /// [`ValidationError::IntegerOverflow`] for unrepresentable metadata or
    /// [`ValidationError::ViewOutOfBounds`] if the layout leaves its allocation.
    pub fn into_tensor_read(self) -> crate::Result<TensorRead<'a>>
    where
        T: TensorScalar,
    {
        let layout = TensorLayout::<DynRank>::from_parts(
            shape_vec(self.layout.shape()),
            stride_vec(self.layout.strides()),
            self.layout.offset(),
            self.buffer.len(),
        )
        .map_err(|err| tensor_layout_error("TypedTensorView::into_tensor_read", err))?;
        let view = TypedTensorView {
            buffer: self.buffer,
            root: self.root.map(GroupReadView::into_dyn),
            layout,
            placement: self.placement,
            _representation: std::marker::PhantomData,
        };
        Ok(TensorRead::from_view(T::tensor_view(view)))
    }

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Return the logical rank carried by this view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 6];
    /// let view = TypedTensorView::from_slice(vec![2, 3], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.rank(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![-1], 1, &data)?;
    /// assert_eq!(view.strides(), &[-1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Return the physical element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![1], vec![1], 1, &data)?;
    /// assert_eq!(view.offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    /// Return the borrowed host storage backing this view.
    ///
    /// This exposes the entire backing host allocation, not just the logical
    /// slice covered by this view. Use [`TypedTensorView::as_slice`] when the
    /// caller needs the contiguous logical region instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.host_storage()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when this view wraps a backend
    /// buffer; backend storage must be downloaded before host inspection.
    pub fn host_storage(&self) -> crate::Result<&'a [T]> {
        match &self.buffer {
            TensorStorageRef::Host(data) => Ok(data),
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => {
                Err(crate::Error::runtime_state(
                    "TypedTensorView::host_storage",
                    "backend buffers cannot expose host storage; download explicitly first",
                ))
            }
        }
    }

    /// Return the number of logical elements in this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 6];
    /// let view = TypedTensorView::from_slice(vec![2, 3], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.n_elements(), 6);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn n_elements(&self) -> usize {
        // Invariant: public view constructors validate logical element count.
        match checked_view_element_count(self.shape(), "TypedTensorView::n_elements") {
            Ok(n) => n,
            Err(err) => {
                unreachable!("TypedTensorView layout shape is validated at construction: {err}")
            }
        }
    }

    /// Return layout metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert!(view.layout().is_compact_col_major().unwrap());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Return placement metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, TypedTensorView};
    ///
    /// let data = [1_i32];
    /// let view = TypedTensorView::from_slice(vec![1], vec![1], 0, &data)?;
    /// assert_eq!(view.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Return the backend allocation for backend integrations.
    #[doc(hidden)]
    pub fn backing_len(&self) -> usize {
        self.buffer.len()
    }

    /// Return the backend allocation for backend integrations.
    #[doc(hidden)]
    pub fn backend_buffer(&self) -> Option<&dyn BackendStorage<T>> {
        match &self.buffer {
            TensorStorageRef::Host(_) => None,
            TensorStorageRef::Backend(buffer) => Some(*buffer),
            TensorStorageRef::Root(_) => {
                self.root
                    .as_ref()?
                    .backend_buffer()
                    .and_then(|buffer| match buffer {
                        StorageBuffer::Host(_) => None,
                        StorageBuffer::Backend(buffer) => Some(buffer.as_ref()),
                    })
            }
        }
    }

    /// Return the provider family for this view when it is backend-owned.
    #[doc(hidden)]
    pub fn backend_family(&self) -> Option<&'static str>
    where
        T: TensorScalar + 'static,
    {
        self.root
            .as_ref()
            .and_then(|root| {
                root.backend_allocation()
                    .map(|_| root.provider_kind().as_str())
            })
            .or_else(|| self.backend_buffer().map(|buffer| buffer.backend_family()))
    }

    /// Return the shared allocation domain for this view when backend-owned.
    #[doc(hidden)]
    pub fn allocation_domain(&self) -> Option<AllocationDomainId>
    where
        T: TensorScalar + 'static,
    {
        self.root
            .as_ref()
            .and_then(|root| root.backend_identity().map(|(domain, _)| domain))
            .or_else(|| {
                self.backend_buffer()
                    .and_then(|buffer| buffer.allocation_domain())
            })
    }

    /// Return the physical allocation identity for this view when backend-owned.
    #[doc(hidden)]
    pub fn allocation_id(&self) -> Option<AllocationId>
    where
        T: TensorScalar + 'static,
    {
        self.root
            .as_ref()
            .and_then(|root| root.backend_identity().map(|(_, id)| id))
            .or_else(|| {
                self.backend_buffer()
                    .and_then(|buffer| buffer.allocation_id())
            })
    }

    /// Prepare this backend view for one provider-native read binding.
    #[doc(hidden)]
    pub fn prepare_device_read(
        &self,
        op: &'static str,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>>
    where
        T: TensorScalar + 'static,
    {
        if let Some(root) = &self.root {
            return root
                .prepare_device_read_for_layout(&self.layout)
                .map_err(|error| crate::Error::runtime_state_source(op, error));
        }
        let buffer = self.backend_buffer().ok_or_else(|| {
            crate::Error::runtime_state_source(
                op,
                crate::AccessError::Unsupported { backend: "host" },
            )
        })?;
        prepare_backend_access(buffer, &self.layout, op)
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![-1], 2, &data)?;
    /// assert_eq!(view.linear_offset(&[2]), Some(0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_view_offset(self.shape(), self.strides(), self.offset(), indices)
    }

    /// Compute the physical element offset for a logical index, returning a typed error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice([3], [-1], 2, &data)?;
    /// assert_eq!(view.layout_linear_offset(&[2])?, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        checked_view_offset_result(
            self.shape(),
            self.strides(),
            self.offset(),
            indices,
            "TypedTensorView::layout_linear_offset",
        )
    }

    /// Return whether this view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice([2], [1], 0, &data)?;
    /// assert!(view.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] if compactness
    /// arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        self.layout
            .is_compact_col_major()
            .map_err(|err| tensor_layout_error("TypedTensorView::is_col_major_contiguous", err))
    }

    /// Return a compact string summary of this view's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice([2], [1], 0, &data)?;
    /// assert!(view.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    /// Assert this view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice([2], [1], 0, &data)?;
    /// view.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// view is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TypedTensorView::assert_col_major_contiguous",
        )
    }

    /// Borrow one host element by logical index.
    ///
    /// Returns `None` for out-of-bounds indices and backend buffers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.linear_offset(indices)?;
        match &self.buffer {
            TensorStorageRef::Host(data) => data.get(offset),
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => None,
        }
    }

    /// Borrow the contiguous host slice covered by this view.
    ///
    /// Returns an explicit error for backend buffers and for non-contiguous
    /// layouts. This method never downloads or materializes backend data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 1, &data)?;
    /// assert_eq!(view.as_slice()?, &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when this view wraps a backend
    /// buffer, [`tenferro_tensor_core::ValidationError::NonContiguousViewAsSlice`]
    /// when the layout is not compact column-major,
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the view
    /// offset is negative, or [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when the
    /// requested host range is invalid.
    pub fn as_slice(&self) -> crate::Result<&'a [T]> {
        let data = match &self.buffer {
            TensorStorageRef::Host(data) => data,
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => {
                return Err(crate::Error::runtime_state(
                    "TypedTensorView::as_slice",
                    "backend buffers cannot be inspected as host slices; download explicitly first",
                ))
            }
        };
        contiguous_layout_slice(self.layout(), data, "TypedTensorView::as_slice")
    }

    /// Borrow the compact logical region through a scoped host mapping.
    ///
    /// Backend integrations must validate placement and allocation-domain ownership
    /// before mapping. The guard cannot escape the callback; no transfer is performed.
    ///
    /// # Errors
    ///
    /// Returns a host-access or provider-preparation error when mapping fails,
    /// or a validation/unsupported error for a non-contiguous or invalid layout.
    #[doc(hidden)]
    pub fn with_host_read<U>(&self, f: impl FnOnce(&[T]) -> U) -> crate::Result<U>
    where
        T: TensorScalar + 'static,
    {
        const OP: &str = "TypedTensorView::with_host_read";
        if !self.is_col_major_contiguous()? {
            return Err(crate::Error::unsupported(
                OP,
                "host guard access requires a compact descriptor",
            ));
        }
        if let Some(buffer) = self.backend_buffer() {
            let guard = buffer
                .map_read()
                .map_err(|error| crate::Error::host_access(OP, error))?;
            return Ok(f(contiguous_layout_slice(self.layout(), &guard, OP)?));
        }
        if let Some(root) = &self.root {
            let prepared = root
                .prepare_host_read_for_layout(self.layout())
                .map_err(|error| crate::Error::runtime_state_source(OP, error))?;
            let slice = prepared.as_slice().ok_or_else(|| {
                crate::Error::unsupported(OP, "host guard access requires a compact descriptor")
            })?;
            return Ok(f(slice));
        }
        Ok(f(self.as_slice()?))
    }

    /// Explicitly duplicate a compact host view into a new owner.
    ///
    /// Backend views require an explicit provider canonicalization or download
    /// boundary; this method never transfers or materializes them implicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// let copy = view.duplicate()?;
    /// assert_eq!(copy.as_slice()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::HostAccess`] or
    /// [`ValidationError::NonContiguousViewAsSlice`] when the view is backend
    /// owned or not contiguous, and [`ValidationError::InvalidArgument`] when
    /// the static-rank shape cannot be reconstructed.
    pub fn duplicate(&self) -> crate::Result<TypedTensor<T, R>>
    where
        T: Clone,
    {
        let data = self.as_slice()?.to_vec();
        let shape = R::shape_from_vec(shape_vec(self.shape()))
            .map_err(|err| tensor_layout_error("TypedTensorView::duplicate", err))?;
        let mut tensor = TypedTensor::from_vec_col_major(shape, data)?;
        tensor.set_placement(self.placement.clone());
        Ok(tensor)
    }

    /// Explicitly copy a host view, including strided/offset views, to a compact owner.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{Rank, TypedTensorView};
    /// let storage = [1, 4, 2, 5, 3, 6];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked(
    ///     [2, 3], [1, 2], 0, &storage,
    /// )?;
    /// let transposed = view.transpose_view([1, 0])?;
    /// assert_eq!(transposed.to_col_major()?.as_slice()?, &[1, 2, 3, 4, 5, 6]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Backend-only storage returns [`crate::Error::RuntimeState`]; invalid
    /// layout or rank conversion returns [`crate::Error::Validation`].
    pub fn to_col_major(&self) -> crate::Result<TypedTensor<T, R>>
    where
        T: Clone,
    {
        let data = self.host_storage()?;
        let mut coordinates = ShapeVec::from_elem(0, self.shape().len());
        let element_count = self.n_elements();
        let mut copied = Vec::with_capacity(element_count);
        for _ in 0..element_count {
            let offset = self.layout_linear_offset(&coordinates)?;
            let value = data.get(offset).ok_or_else(|| {
                crate::Error::validation(
                    "TypedTensorView::to_col_major",
                    ValidationError::ViewOutOfBounds,
                )
            })?;
            copied.push(value.clone());
            for axis in 0..coordinates.len() {
                coordinates[axis] += 1;
                if coordinates[axis] < self.shape()[axis] {
                    break;
                }
                coordinates[axis] = 0;
            }
        }
        let shape = R::shape_from_vec(shape_vec(self.shape()))
            .map_err(|err| tensor_layout_error("TypedTensorView::to_col_major", err))?;
        let mut tensor = TypedTensor::from_vec_col_major(shape, copied)?;
        tensor.set_placement(self.placement.clone());
        Ok(tensor)
    }

    /// Return a metadata-only axis permutation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4, 5, 6];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 3], [1, 2], 0, &data)?;
    /// let transposed = view.transpose_view([1, 0])?;
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::InvalidPermutationLength`],
    /// [`tenferro_tensor_core::ValidationError::AxisOutOfBounds`], or
    /// [`tenferro_tensor_core::ValidationError::DuplicateAxis`] when `axes` is
    /// not a valid permutation of the view rank.
    pub fn transpose_view(&self, axes: impl AsRef<[usize]>) -> crate::Result<Self> {
        let layout = self
            .layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TypedTensorView::transpose_view", err))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            root: self.root.clone(),
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }

    /// Return a metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![1], 0, &data)?;
    /// let reversed = view.slice_view(&[StridedSliceSpec::reverse()])?;
    /// assert_eq!(reversed.get(&[0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when the slice
    /// count differs from the view rank,
    /// [`tenferro_tensor_core::ValidationError::InvalidSliceStep`] or
    /// [`tenferro_tensor_core::ValidationError::InvalidSliceBounds`] for an
    /// invalid slice, [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// for slice arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// resulting layout exceeds the backing buffer.
    pub fn slice_view(&self, slices: &[StridedSliceSpec]) -> crate::Result<Self> {
        let specs = core_slice_specs(slices, self.shape(), "TypedTensorView::slice_view")?;
        let layout = self
            .layout
            .slice_view(specs, self.buffer.len())
            .map_err(|err| tensor_layout_error("TypedTensorView::slice_view", err))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            root: self.root.clone(),
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }

    /// Return a metadata-only slice along one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_slice(vec![2, 2], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.slice_axis_view(1, StridedSliceSpec::reverse())?.get(&[0, 0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::AxisOutOfBounds`] when `axis`
    /// is outside the view rank, [`tenferro_tensor_core::ValidationError::InvalidSliceStep`]
    /// or [`tenferro_tensor_core::ValidationError::InvalidSliceBounds`] for an
    /// invalid slice, [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// for slice arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// resulting layout exceeds the backing buffer.
    pub fn slice_axis_view(&self, axis: usize, slice: StridedSliceSpec) -> crate::Result<Self> {
        let slices = slice_axis_specs(
            self.shape().len(),
            axis,
            slice,
            "TypedTensorView::slice_axis_view",
        )?;
        self.slice_view(&slices)
    }

    /// Return a metadata-only dynamic-rank reshape for contiguous column-major views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_slice(vec![2, 2], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.reshape_view(&[4])?.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::NonContiguousViewAsSlice`] when
    /// the source is not compact column-major,
    /// [`tenferro_tensor_core::ValidationError::ShapeMismatch`] (whose
    /// [`tenferro_tensor_core::ShapeMismatch::ReshapeElementCount`] source
    /// records the counts) when element counts differ,
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for shape
    /// arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// reshaped view exceeds the backing buffer.
    pub fn reshape_view(
        &self,
        shape: &[usize],
    ) -> crate::Result<TypedTensorView<'a, T, DynRank, D>> {
        let layout = reshape_layout_dyn(
            &self.layout,
            shape,
            self.buffer.len(),
            "TypedTensorView::reshape_view",
        )?;
        Ok(TypedTensorView {
            buffer: self.buffer.clone(),
            root: self.root.as_ref().map(GroupReadView::clone_dyn),
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, R: TensorRank> TypedTensorView<'a, Complex32, R> {
    /// Borrow this complex view as an interleaved real view without copying.
    ///
    /// The result has dynamic rank because reinterpretation prepends the
    /// component axis `[2, ...]`. Only `Complex32 <-> f32` is sealed in this
    /// API; this is representation reinterpretation, not numeric conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex32, TypedTensorView};
    ///
    /// let data = [Complex32::new(1.0, 2.0)];
    /// let view = TypedTensorView::from_col_major(&[1], &data)?;
    /// let real = view.as_real_view()?;
    /// assert_eq!(real.shape(), &[2, 1]);
    /// assert_eq!(real.as_slice()?, &[1.0_f32, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not a valid sealed
    /// representation or when backend reinterpretation is unsupported.
    pub fn as_real_view(&self) -> crate::Result<TypedTensorView<'a, f32, DynRank>> {
        let op = "TypedTensorView::as_real_view";
        validate_representation_pair(op, DType::C32, DType::F32)?;
        let layout = reinterpret_complex_to_real_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        let buffer = match &self.buffer {
            TensorStorageRef::Host(data) => {
                TensorStorageRef::Host(reinterpret_host_slice::<Complex32, f32>(data, op)?)
            }
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorView {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, R: TensorRank> TypedTensorView<'a, Complex64, R> {
    /// Borrow this complex view as an interleaved real view without copying.
    ///
    /// The result has dynamic rank because reinterpretation prepends the
    /// component axis `[2, ...]`. Only `Complex64 <-> f64` is sealed in this
    /// API; this is representation reinterpretation, not numeric conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex64, TypedTensorView};
    ///
    /// let data = [Complex64::new(1.0, 2.0)];
    /// let view = TypedTensorView::from_col_major(&[1], &data)?;
    /// let real = view.as_real_view()?;
    /// assert_eq!(real.shape(), &[2, 1]);
    /// assert_eq!(real.as_slice()?, &[1.0_f64, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not a valid sealed
    /// representation or when backend reinterpretation is unsupported.
    pub fn as_real_view(&self) -> crate::Result<TypedTensorView<'a, f64, DynRank>> {
        let op = "TypedTensorView::as_real_view";
        validate_representation_pair(op, DType::C64, DType::F64)?;
        let layout = reinterpret_complex_to_real_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        let buffer = match &self.buffer {
            TensorStorageRef::Host(data) => {
                TensorStorageRef::Host(reinterpret_host_slice::<Complex64, f64>(data, op)?)
            }
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorView {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, R: TensorRank> TypedTensorView<'a, f32, R> {
    /// Borrow this interleaved real view as a complex view without copying.
    ///
    /// The source must have a leading extent and stride of `2` and `1`, and
    /// every remaining stride plus the offset must be divisible by `2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex32, TypedTensorView};
    ///
    /// let data = [1.0_f32, 2.0];
    /// let view = TypedTensorView::from_col_major(&[2, 1], &data)?;
    /// let complex = view.as_complex_view()?;
    /// assert_eq!(complex.shape(), &[1]);
    /// assert_eq!(complex.as_slice()?, &[Complex32::new(1.0, 2.0)]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not a valid sealed
    /// representation or when backend reinterpretation is unsupported.
    pub fn as_complex_view(&self) -> crate::Result<TypedTensorView<'a, Complex32, DynRank>> {
        let op = "TypedTensorView::as_complex_view";
        validate_representation_pair(op, DType::F32, DType::C32)?;
        let layout = reinterpret_real_to_complex_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        let buffer = match &self.buffer {
            TensorStorageRef::Host(data) => {
                TensorStorageRef::Host(reinterpret_host_slice::<f32, Complex32>(data, op)?)
            }
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorView {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, R: TensorRank> TypedTensorView<'a, f64, R> {
    /// Borrow this interleaved real view as a complex view without copying.
    ///
    /// The source must have a leading extent and stride of `2` and `1`, and
    /// every remaining stride plus the offset must be divisible by `2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex64, TypedTensorView};
    ///
    /// let data = [1.0_f64, 2.0];
    /// let view = TypedTensorView::from_col_major(&[2, 1], &data)?;
    /// let complex = view.as_complex_view()?;
    /// assert_eq!(complex.shape(), &[1]);
    /// assert_eq!(complex.as_slice()?, &[Complex64::new(1.0, 2.0)]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not a valid sealed
    /// representation or when backend reinterpretation is unsupported.
    pub fn as_complex_view(&self) -> crate::Result<TypedTensorView<'a, Complex64, DynRank>> {
        let op = "TypedTensorView::as_complex_view";
        validate_representation_pair(op, DType::F64, DType::C64)?;
        let layout = reinterpret_real_to_complex_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        let buffer = match &self.buffer {
            TensorStorageRef::Host(data) => {
                TensorStorageRef::Host(reinterpret_host_slice::<f64, Complex64>(data, op)?)
            }
            TensorStorageRef::Backend(_) | TensorStorageRef::Root(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorView {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

/// Mutable borrowed view of typed tensor storage with arbitrary strides.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TypedTensorViewMut;
///
/// let mut data = [1_i32, 2, 3];
/// let mut view = TypedTensorViewMut::from_slice(vec![3], vec![-1], 2, &mut data)?;
/// *view.get_mut(&[2]).unwrap() = 10;
/// assert_eq!(view.as_read_only().get(&[2]), Some(&10));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
/// Exclusive typed view of one tensor representation.
///
/// As with [`TypedTensorView`], the buffer stays concrete so the view keeps its
/// lifetime covariance, and `D` marks which representation produced the view.
pub struct TypedTensorViewMut<'a, T, R: TensorRank = DynRank, D: Representation = Dynamic> {
    buffer: TensorStorageRefMut<'a, T>,
    root: Option<GroupWriteView<'a, T, R>>,
    layout: TensorLayout<R>,
    placement: Placement,
    _representation: std::marker::PhantomData<D>,
}

impl<'a, T, R: TensorRank, D: Representation> std::fmt::Debug for TypedTensorViewMut<'a, T, R, D> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TypedTensorViewMut")
            .field("shape", &self.layout.shape())
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

/// Pair of mutable tensor views returned by disjoint multi-slice operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut, TypedTensorViewMutSplit};
///
/// let mut data = [1_i32, 2, 3, 4];
/// let mut view = TypedTensorViewMut::from_slice(vec![4], vec![1], 0, &mut data)?;
/// let pair: TypedTensorViewMutSplit<'_, i32> = view
///     .try_multi_slice_mut(
///         &[StridedSliceSpec::new(0, Some(2), 1)],
///         &[StridedSliceSpec::new(2, Some(4), 1)],
///     )
///     ?
///     .unwrap();
/// assert_eq!(pair.0.shape(), &[2]);
/// assert_eq!(pair.1.shape(), &[2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub type TypedTensorViewMutSplit<'a, T, R = DynRank> =
    (TypedTensorViewMut<'a, T, R>, TypedTensorViewMut<'a, T, R>);

impl<'a, T: 'static> TypedTensorViewMut<'a, T, DynRank> {
    /// Create a mutable dynamic-rank view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::from_col_major(&[2, 2], &mut data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn from_col_major(shape: &[usize], data: &'a mut [T]) -> crate::Result<Self> {
        let layout = TensorLayout::<DynRank>::compact(shape_vec(shape))
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::from_col_major", err))?;
        Self::from_buffer_ref_mut(
            shape_vec(layout.shape()),
            stride_vec(layout.strides()),
            layout.offset(),
            TensorStorageRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_col_major",
        )
    }

    /// Create a mutable host view from explicit layout metadata.
    ///
    /// Layouts where distinct logical elements can alias the same physical
    /// element are rejected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// assert!(TypedTensorViewMut::from_slice(vec![2], vec![0], 0, &mut data).is_err());
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `shape` and
    /// `strides` have different ranks,
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// layout reaches beyond `data`,
    /// [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`] when
    /// logical elements alias, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    pub fn from_slice(
        shape: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: isize,
        data: &'a mut [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref_mut(
            shape_vec(shape.as_ref()),
            stride_vec(strides.as_ref()),
            offset,
            TensorStorageRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_slice",
        )
    }
}

impl<'a, T: 'static, R: TensorRank, D: Representation> TypedTensorViewMut<'a, T, R, D> {
    /// Create a rank-generic mutable host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &mut data)?;
    /// assert_eq!(view.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when the typed
    /// rank does not match `shape` or `strides`,
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// layout reaches beyond `data`,
    /// [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`] when
    /// logical elements alias, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    pub fn from_slice_ranked(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a mut [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref_mut(
            shape,
            strides,
            offset,
            TensorStorageRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_slice_ranked",
        )
    }
}

impl<'a, T: 'static, R: TensorRank, D: Representation> TypedTensorViewMut<'a, T, R, D> {
    fn from_buffer_ref_mut(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        buffer: TensorStorageRefMut<'a, T>,
        placement: Placement,
        op: &'static str,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            buffer,
            root: None,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, T: 'static, R: TensorRank> TypedTensorViewMut<'a, T, R, Host> {
    /// Create a representation-marked mutable view over an explicit host layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DynRank, Host, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view: TypedTensorViewMut<'_, i32, DynRank, Host> =
    ///     TypedTensorViewMut::from_host_slice(vec![2], vec![1], 0, &mut data)?;
    /// view.slice_view(&[tenferro_tensor::StridedSliceSpec::all()])?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `strides`
    /// has a different rank, [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`]
    /// when logical elements alias, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] /
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when the
    /// reachable layout leaves `data` or overflows.
    pub fn from_host_slice(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a mut [T],
    ) -> crate::Result<Self> {
        // Built here rather than through the shared helper so that the Host
        // marker can only ever be paired with a host slice.
        let buffer = TensorStorageRefMut::Host(data);
        let layout =
            TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
                .map_err(|err| tensor_layout_error("TypedTensorViewMut::from_host_slice", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::from_host_slice", err))?;
        Ok(Self {
            buffer,
            root: None,
            layout,
            placement: default_placement(),
            _representation: std::marker::PhantomData,
        })
    }

    /// Exclusively borrow the host elements without a runtime check.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DynRank, Host, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view: TypedTensorViewMut<'_, i32, DynRank, Host> =
    ///     TypedTensorViewMut::from_host_slice(vec![2], vec![1], 0, &mut data)?;
    /// view.as_host_slice_mut()[0] = 5;
    /// assert_eq!(data, [5, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Panics only if a `Host`-marked view was built from non-host storage,
    /// which no constructor in this crate does.
    pub fn as_host_slice_mut(&mut self) -> &mut [T] {
        match &mut self.buffer {
            TensorStorageRefMut::Host(data) => data,
            // INVARIANT: `Host`-marked views are constructed only by
            // `from_host_slice` and by `TypedTensor<_, _, Host>::as_view_mut`.
            TensorStorageRefMut::Backend(_) => {
                unreachable!("a Host-marked view always borrows host storage")
            }
        }
    }
}

impl<'a, T: 'static, R: TensorRank, D: Representation> TypedTensorViewMut<'a, T, R, D> {
    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Return the logical rank carried by this mutable view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 6];
    /// let view = TypedTensorViewMut::from_slice(vec![2, 3], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.rank(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![-1], 1, &mut data)?;
    /// assert_eq!(view.strides(), &[-1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Return the physical element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 1, &mut data)?;
    /// assert_eq!(view.offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    /// Return the borrowed host storage backing this view.
    ///
    /// This exposes the entire backing host allocation, not just the logical
    /// slice covered by this view. Use [`TypedTensorViewMut::as_read_only`]
    /// with [`TypedTensorView::as_slice`] when the caller needs the contiguous
    /// logical region instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.host_storage()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when this view wraps a backend
    /// buffer; backend storage must be downloaded before host inspection.
    pub fn host_storage(&self) -> crate::Result<&[T]> {
        match &self.buffer {
            TensorStorageRefMut::Host(data) => Ok(data),
            TensorStorageRefMut::Backend(_) => Err(crate::Error::runtime_state(
                "TypedTensorViewMut::host_storage",
                "backend buffers cannot expose host storage; download explicitly first",
            )),
        }
    }

    /// Mutably borrow the host storage backing this view.
    ///
    /// This exposes the entire backing host allocation, not just the logical
    /// slice covered by this view. Prefer scalar element accessors when mutating
    /// a logical region; tensor-sized copies belong to an active backend.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// view.host_storage_mut()?[0] = 3;
    /// assert_eq!(view.get(&[0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when this view wraps a backend
    /// buffer; backend storage must be downloaded before host inspection.
    pub fn host_storage_mut(&mut self) -> crate::Result<&mut [T]> {
        match &mut self.buffer {
            TensorStorageRefMut::Host(data) => Ok(data),
            TensorStorageRefMut::Backend(_) => Err(crate::Error::runtime_state(
                "TypedTensorViewMut::host_storage_mut",
                "backend buffers cannot expose mutable host storage; download explicitly first",
            )),
        }
    }

    /// Return the number of logical elements in this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 6];
    /// let view = TypedTensorViewMut::from_slice(vec![2, 3], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.n_elements(), 6);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn n_elements(&self) -> usize {
        // Invariant: public mutable view constructors validate logical element count.
        match checked_view_element_count(self.shape(), "TypedTensorViewMut::n_elements") {
            Ok(n) => n,
            Err(err) => {
                unreachable!("TypedTensorViewMut layout shape is validated at construction: {err}")
            }
        }
    }

    /// Return layout metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert!(view.layout().is_compact_col_major().unwrap());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Return placement metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Return the backend allocation for backend integrations.
    #[doc(hidden)]
    pub fn backend_buffer(&self) -> Option<&dyn BackendStorage<T>> {
        match &self.buffer {
            TensorStorageRefMut::Host(_) => None,
            TensorStorageRefMut::Backend(buffer) => Some(&**buffer),
        }
    }

    /// Prepare this backend view for one provider-native write binding.
    #[doc(hidden)]
    pub fn prepare_device_write(
        &mut self,
        op: &'static str,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>>
    where
        T: TensorScalar + 'static,
    {
        let layout = self.layout.clone();
        if self.root.is_none() {
            let buffer = self.backend_buffer().ok_or_else(|| {
                crate::Error::runtime_state_source(
                    op,
                    crate::AccessError::Unsupported { backend: "host" },
                )
            })?;
            return prepare_backend_access(buffer, &self.layout, op);
        }
        let root = self
            .root
            .as_mut()
            .ok_or_else(|| crate::Error::runtime_state(op, "expected a root-backed tensor view"))?;
        root.prepare_device_write_for_layout(&layout)
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let view = TypedTensorViewMut::from_slice(vec![3], vec![-1], 2, &mut data)?;
    /// assert_eq!(view.linear_offset(&[2]), Some(0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_view_offset(self.shape(), self.strides(), self.offset(), indices)
    }

    /// Compute the physical element offset for a logical index, returning a typed error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let view = TypedTensorViewMut::from_slice([3], [-1], 2, &mut data)?;
    /// assert_eq!(view.layout_linear_offset(&[2])?, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        checked_view_offset_result(
            self.shape(),
            self.strides(),
            self.offset(),
            indices,
            "TypedTensorViewMut::layout_linear_offset",
        )
    }

    /// Return whether this mutable view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?;
    /// assert!(view.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        self.layout
            .is_compact_col_major()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::is_col_major_contiguous", err))
    }

    /// Return a compact string summary of this mutable view's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?;
    /// assert!(view.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    /// Assert this mutable view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?;
    /// view.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// view is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TypedTensorViewMut::assert_col_major_contiguous",
        )
    }

    /// Borrow one host element by logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.linear_offset(indices)?;
        match &self.buffer {
            TensorStorageRefMut::Host(data) => data.get(offset),
            TensorStorageRefMut::Backend(_) => None,
        }
    }

    /// Mutably borrow one host element by logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// *view.get_mut(&[1]).unwrap() = 20;
    /// assert_eq!(view.get(&[1]), Some(&20));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let offset = self.linear_offset(indices)?;
        match &mut self.buffer {
            TensorStorageRefMut::Host(data) => data.get_mut(offset),
            TensorStorageRefMut::Backend(_) => None,
        }
    }

    /// Borrow this mutable view as a read-only view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.as_read_only().get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// Explicitly duplicate the compact host data visible through this
    /// mutable view into a new owner.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::HostAccess`] or
    /// [`ValidationError::NonContiguousViewAsSlice`] when the view is backend
    /// owned or not contiguous, and [`ValidationError::InvalidArgument`] when
    /// the static-rank shape cannot be reconstructed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// let copy = view.duplicate()?;
    /// assert_eq!(copy.as_slice()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn duplicate(&self) -> crate::Result<TypedTensor<T, R>>
    where
        T: Clone,
    {
        self.as_read_only().duplicate()
    }

    /// Explicitly copy this host view to a compact column-major owner.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::TypedTensorViewMut;
    /// let mut values = ["a".to_owned(), "b".to_owned()];
    /// let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut values)?;
    /// assert_eq!(view.to_col_major()?.as_slice()?, &["a", "b"]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Backend-only storage returns [`crate::Error::RuntimeState`]; invalid
    /// layout or rank conversion returns [`crate::Error::Validation`].
    pub fn to_col_major(&self) -> crate::Result<TypedTensor<T, R>>
    where
        T: Clone,
    {
        self.as_read_only().to_col_major()
    }

    /// Borrow this mutable view as a read-only typed view over the same storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.as_read_only().as_slice()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_read_only(&self) -> TypedTensorView<'_, T, R, D> {
        let buffer = match &self.buffer {
            TensorStorageRefMut::Host(data) => TensorStorageRef::Host(data),
            TensorStorageRefMut::Backend(buffer) => TensorStorageRef::Backend(&**buffer),
        };
        TypedTensorView {
            buffer,
            root: None,
            layout: self.layout.clone(),
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        }
    }

    /// Convert this mutable view into a read-only view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.into_read_only().get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_read_only(self) -> TypedTensorView<'a, T, R, D> {
        let buffer = match self.buffer {
            TensorStorageRefMut::Host(data) => TensorStorageRef::Host(data),
            TensorStorageRefMut::Backend(buffer) => TensorStorageRef::Backend(buffer),
        };
        TypedTensorView {
            buffer,
            root: None,
            layout: self.layout,
            placement: self.placement,
            _representation: std::marker::PhantomData,
        }
    }

    /// Consume this mutable view and return a metadata-only axis permutation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &mut data)?;
    /// let transposed = view.transpose_view([1, 0])?;
    /// assert_eq!(transposed.strides(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::InvalidPermutationLength`],
    /// [`tenferro_tensor_core::ValidationError::AxisOutOfBounds`], or
    /// [`tenferro_tensor_core::ValidationError::DuplicateAxis`] when `axes` is
    /// not a valid permutation, [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`]
    /// when the permutation creates aliases, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    pub fn transpose_view(
        self,
        axes: impl AsRef<[usize]>,
    ) -> crate::Result<TypedTensorViewMut<'a, T, R>> {
        let Self {
            buffer,
            root,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        } = self;
        let layout = layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        match buffer {
            TensorStorageRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorStorageRefMut::Host(data),
                root,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            }),
            TensorStorageRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorStorageRefMut::Backend(buffer),
                root,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            }),
        }
    }

    /// Return a mutable metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let mut view = TypedTensorViewMut::from_slice(vec![3], vec![1], 0, &mut data)?;
    /// *view.slice_view(&[StridedSliceSpec::reverse()])?.get_mut(&[0]).unwrap() = 30;
    /// assert_eq!(view.get(&[2]), Some(&30));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when the slice
    /// count differs from the view rank,
    /// [`tenferro_tensor_core::ValidationError::InvalidSliceStep`] or
    /// [`tenferro_tensor_core::ValidationError::InvalidSliceBounds`] for an
    /// invalid slice, [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`]
    /// when the result exceeds the backing buffer,
    /// [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`] when
    /// logical elements alias, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    pub fn slice_view(
        &mut self,
        slices: &[StridedSliceSpec],
    ) -> crate::Result<TypedTensorViewMut<'_, T, R>> {
        let specs = core_slice_specs(slices, self.shape(), "TypedTensorViewMut::slice_view")?;
        let layout = self
            .layout
            .slice_view(specs, self.buffer.len())
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::slice_view", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::slice_view", err))?;
        let placement = self.placement.clone();
        match &mut self.buffer {
            TensorStorageRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorStorageRefMut::Host(data),
                root: None,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            }),
            TensorStorageRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorStorageRefMut::Backend(*buffer),
                root: None,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            }),
        }
    }

    /// Return a mutable metadata-only slice along one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.slice_axis_view(1, StridedSliceSpec::reverse())?.get(&[0, 0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::AxisOutOfBounds`] when `axis`
    /// is outside the view rank, [`tenferro_tensor_core::ValidationError::InvalidSliceStep`]
    /// or [`tenferro_tensor_core::ValidationError::InvalidSliceBounds`] for an
    /// invalid slice, [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`]
    /// when the result exceeds the backing buffer,
    /// [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`] when
    /// logical elements alias, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    pub fn slice_axis_view(
        &mut self,
        axis: usize,
        slice: StridedSliceSpec,
    ) -> crate::Result<TypedTensorViewMut<'_, T, R>> {
        let slices = slice_axis_specs(
            self.shape().len(),
            axis,
            slice,
            "TypedTensorViewMut::slice_axis_view",
        )?;
        self.slice_view(&slices)
    }

    /// Return two mutable metadata-only slices when their physical ranges are disjoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![4], vec![1], 0, &mut data)?;
    /// let (left, right) = view
    ///     .try_multi_slice_mut(
    ///         &[StridedSliceSpec::new(0, Some(2), 1)],
    ///         &[StridedSliceSpec::new(2, Some(4), 1)],
    ///     )
    ///     ?
    ///     .unwrap();
    /// assert_eq!(left.shape(), &[2]);
    /// assert_eq!(right.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] for either
    /// slice count, [`tenferro_tensor_core::ValidationError::InvalidSliceStep`]
    /// or [`tenferro_tensor_core::ValidationError::InvalidSliceBounds`] for
    /// invalid parameters, [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`]
    /// when a result exceeds the backing buffer,
    /// [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`] when
    /// a result aliases, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// for a negative reachable offset, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow. The method returns `Ok(None)` when the two ranges
    /// overlap or the view uses backend storage.
    pub fn try_multi_slice_mut(
        &mut self,
        first: &[StridedSliceSpec],
        second: &[StridedSliceSpec],
    ) -> crate::Result<Option<TypedTensorViewMutSplit<'_, T, R>>> {
        let op = "TypedTensorViewMut::try_multi_slice_mut";
        let first_specs = core_slice_specs(first, self.shape(), op)?;
        let second_specs = core_slice_specs(second, self.shape(), op)?;
        let buffer_len = self.buffer.len();
        let first_layout = self
            .layout
            .slice_view(first_specs, buffer_len)
            .map_err(|err| tensor_layout_error(op, err))?;
        let second_layout = self
            .layout
            .slice_view(second_specs, buffer_len)
            .map_err(|err| tensor_layout_error(op, err))?;
        first_layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        second_layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;

        match (
            reachable_layout_span(
                first_layout.shape(),
                first_layout.strides(),
                first_layout.offset(),
            )?,
            reachable_layout_span(
                second_layout.shape(),
                second_layout.strides(),
                second_layout.offset(),
            )?,
        ) {
            (Some(first_span), Some(second_span)) => {
                let first_offset = adjusted_view_offset(first_layout.offset(), first_span.0)?;
                let second_offset = adjusted_view_offset(second_layout.offset(), second_span.0)?;
                let (first_data, second_data) = match &mut self.buffer {
                    TensorStorageRefMut::Host(data) => {
                        match split_two_mut_ranges(data, first_span, second_span) {
                            Some(ranges) => ranges,
                            None => return Ok(None),
                        }
                    }
                    TensorStorageRefMut::Backend(_) => return Ok(None),
                };
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    first_offset,
                    first_data,
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    second_offset,
                    second_data,
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
            (None, Some(second_span)) => {
                let second_offset = adjusted_view_offset(second_layout.offset(), second_span.0)?;
                let (_, after_start) = match &mut self.buffer {
                    TensorStorageRefMut::Host(data) => data.split_at_mut(second_span.0),
                    TensorStorageRefMut::Backend(_) => return Ok(None),
                };
                let (second_data, _) = after_start.split_at_mut(second_span.1 - second_span.0 + 1);
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    second_offset,
                    second_data,
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
            (Some(first_span), None) => {
                let first_offset = adjusted_view_offset(first_layout.offset(), first_span.0)?;
                let (_, after_start) = match &mut self.buffer {
                    TensorStorageRefMut::Host(data) => data.split_at_mut(first_span.0),
                    TensorStorageRefMut::Backend(_) => return Ok(None),
                };
                let (first_data, _) = after_start.split_at_mut(first_span.1 - first_span.0 + 1);
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    first_offset,
                    first_data,
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
            (None, None) => {
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
        }
    }

    /// Return a mutable metadata-only dynamic-rank reshape for contiguous views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.reshape_view(&[4])?.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::NonContiguousViewAsSlice`] when
    /// the source is not compact column-major,
    /// [`tenferro_tensor_core::ValidationError::ShapeMismatch`] (whose
    /// [`tenferro_tensor_core::ShapeMismatch::ReshapeElementCount`] source
    /// records the counts) when element counts differ,
    /// [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`] when
    /// the reshaped layout aliases, [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// for shape or layout arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// reshaped view exceeds the backing buffer.
    pub fn reshape_view(
        &mut self,
        shape: &[usize],
    ) -> crate::Result<TypedTensorViewMut<'_, T, DynRank, D>> {
        let layout = reshape_layout_dyn(
            &self.layout,
            shape,
            self.buffer.len(),
            "TypedTensorViewMut::reshape_view",
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::reshape_view", err))?;
        let placement = self.placement.clone();
        match &mut self.buffer {
            TensorStorageRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorStorageRefMut::Host(data),
                root: None,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            }),
            TensorStorageRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorStorageRefMut::Backend(*buffer),
                root: None,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            }),
        }
    }
}

impl<'a, R: TensorRank> TypedTensorViewMut<'a, Complex32, R> {
    /// Borrow this mutable complex view as an interleaved real view.
    ///
    /// This changes only the typed descriptor and borrows the same host
    /// allocation. The result has dynamic rank because the component axis is
    /// prepended. Backend-native buffers are rejected until their provider
    /// phase supplies the corresponding mapping capability.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex32, TypedTensorViewMut};
    ///
    /// let mut data = [Complex32::new(1.0, 2.0)];
    /// let mut view = TypedTensorViewMut::from_col_major(&[1], &mut data)?;
    /// let real = view.as_real_view_mut()?;
    /// assert_eq!(real.shape(), &[2, 1]);
    /// assert_eq!(real.as_read_only().as_slice()?, &[1.0_f32, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not injective, the sealed
    /// representation is invalid, or backend reinterpretation is unsupported.
    pub fn as_real_view_mut(&mut self) -> crate::Result<TypedTensorViewMut<'_, f32, DynRank>> {
        let op = "TypedTensorViewMut::as_real_view_mut";
        validate_representation_pair(op, DType::C32, DType::F32)?;
        let layout = reinterpret_complex_to_real_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        let buffer = match &mut self.buffer {
            TensorStorageRefMut::Host(data) => {
                TensorStorageRefMut::Host(reinterpret_host_slice_mut::<Complex32, f32>(data, op)?)
            }
            TensorStorageRefMut::Backend(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, R: TensorRank> TypedTensorViewMut<'a, Complex64, R> {
    /// Borrow this mutable complex view as an interleaved real view.
    ///
    /// This changes only the typed descriptor and borrows the same host
    /// allocation. The result has dynamic rank because the component axis is
    /// prepended. Backend-native buffers are rejected until their provider
    /// phase supplies the corresponding mapping capability.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex64, TypedTensorViewMut};
    ///
    /// let mut data = [Complex64::new(1.0, 2.0)];
    /// let mut view = TypedTensorViewMut::from_col_major(&[1], &mut data)?;
    /// let real = view.as_real_view_mut()?;
    /// assert_eq!(real.shape(), &[2, 1]);
    /// assert_eq!(real.as_read_only().as_slice()?, &[1.0_f64, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not injective, the sealed
    /// representation is invalid, or backend reinterpretation is unsupported.
    pub fn as_real_view_mut(&mut self) -> crate::Result<TypedTensorViewMut<'_, f64, DynRank>> {
        let op = "TypedTensorViewMut::as_real_view_mut";
        validate_representation_pair(op, DType::C64, DType::F64)?;
        let layout = reinterpret_complex_to_real_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        let buffer = match &mut self.buffer {
            TensorStorageRefMut::Host(data) => {
                TensorStorageRefMut::Host(reinterpret_host_slice_mut::<Complex64, f64>(data, op)?)
            }
            TensorStorageRefMut::Backend(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, R: TensorRank> TypedTensorViewMut<'a, f32, R> {
    /// Borrow this mutable interleaved real view as a complex view.
    ///
    /// The source must have a leading extent and stride of `2` and `1`, and
    /// all remaining strides plus the offset must be divisible by `2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex32, TypedTensorViewMut};
    ///
    /// let mut data = [1.0_f32, 2.0];
    /// let mut view = TypedTensorViewMut::from_col_major(&[2, 1], &mut data)?;
    /// let complex = view.as_complex_view_mut()?;
    /// assert_eq!(complex.shape(), &[1]);
    /// assert_eq!(complex.as_read_only().as_slice()?, &[Complex32::new(1.0, 2.0)]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not injective, the sealed
    /// representation is invalid, or backend reinterpretation is unsupported.
    pub fn as_complex_view_mut(
        &mut self,
    ) -> crate::Result<TypedTensorViewMut<'_, Complex32, DynRank>> {
        let op = "TypedTensorViewMut::as_complex_view_mut";
        validate_representation_pair(op, DType::F32, DType::C32)?;
        let layout = reinterpret_real_to_complex_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        let buffer = match &mut self.buffer {
            TensorStorageRefMut::Host(data) => {
                TensorStorageRefMut::Host(reinterpret_host_slice_mut::<f32, Complex32>(data, op)?)
            }
            TensorStorageRefMut::Backend(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

impl<'a, R: TensorRank> TypedTensorViewMut<'a, f64, R> {
    /// Borrow this mutable interleaved real view as a complex view.
    ///
    /// The source must have a leading extent and stride of `2` and `1`, and
    /// all remaining strides plus the offset must be divisible by `2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex64, TypedTensorViewMut};
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let mut view = TypedTensorViewMut::from_col_major(&[2, 1], &mut data)?;
    /// let complex = view.as_complex_view_mut()?;
    /// assert_eq!(complex.shape(), &[1]);
    /// assert_eq!(complex.as_read_only().as_slice()?, &[Complex64::new(1.0, 2.0)]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the view layout is not injective, the sealed
    /// representation is invalid, or backend reinterpretation is unsupported.
    pub fn as_complex_view_mut(
        &mut self,
    ) -> crate::Result<TypedTensorViewMut<'_, Complex64, DynRank>> {
        let op = "TypedTensorViewMut::as_complex_view_mut";
        validate_representation_pair(op, DType::F64, DType::C64)?;
        let layout = reinterpret_real_to_complex_layout(
            self.shape(),
            self.strides(),
            self.offset(),
            self.buffer.len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        let buffer = match &mut self.buffer {
            TensorStorageRefMut::Host(data) => {
                TensorStorageRefMut::Host(reinterpret_host_slice_mut::<f64, Complex64>(data, op)?)
            }
            TensorStorageRefMut::Backend(_) => {
                return Err(crate::Error::unsupported(
                    op,
                    "backend representation reinterpretation is enabled by the provider phases",
                ))
            }
        };
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }
}

/// Sealed trait for scalar types that can be stored in a [`Tensor`].
///
/// This trait is implemented for `f64`, `f32`, `i32`, `i64`, `bool`,
/// [`Complex64`], and [`Complex32`].
///
/// # Examples
///
/// ```
/// use tenferro_tensor::TensorScalar;
///
/// let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0])?;
/// assert_eq!(tensor.as_slice::<f64>()?, [1.0, 2.0].as_slice());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorScalar: Copy + Clone + Send + Sync + 'static + private::Sealed {
    /// Real-valued counterpart of this scalar type.
    type Real: TensorScalar;

    /// The [`DType`] tag corresponding to this scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar};
    ///
    /// assert_eq!(f64::dtype(), DType::F64);
    /// assert_eq!(f32::dtype(), DType::F32);
    /// ```
    fn dtype() -> DType;

    /// Build the crate's default scalar set from validated column-major data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar};
    ///
    /// let set = <f64 as TensorScalar>::into_default_scalars(vec![2], vec![1.0, 2.0])?;
    /// assert_eq!(set.dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error carrying
    /// [`tenferro_tensor_core::ValidationError::ShapeDataLengthMismatch`] when the shape product
    /// differs from `data.len()`, or [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// when shape arithmetic overflows.
    fn into_default_scalars(
        shape: Vec<usize>,
        data: Vec<Self>,
    ) -> crate::Result<crate::DefaultScalars>;

    /// Borrow the default scalar set's values when it holds this scalar type.
    ///
    /// Returns `None` when the set currently holds another member.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DefaultScalars, TensorScalar};
    ///
    /// let set = DefaultScalars::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(<f64 as TensorScalar>::default_scalars_slice(&set), Some(&[1.0, 2.0][..]));
    /// assert!(<f32 as TensorScalar>::default_scalars_slice(&set).is_none());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn default_scalars_slice(set: &crate::DefaultScalars) -> Option<&[Self]>;

    /// Exclusively borrow the default scalar set's values when it holds this scalar type.
    ///
    /// Returns `None` when the set currently holds another member.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DefaultScalars, TensorScalar};
    ///
    /// let mut set = DefaultScalars::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// if let Some(values) = <f64 as TensorScalar>::default_scalars_slice_mut(&mut set) {
    ///     values[0] = 5.0;
    /// }
    /// assert_eq!(set.as_slice::<f64>()?, &[5.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn default_scalars_slice_mut(set: &mut crate::DefaultScalars) -> Option<&mut [Self]>;

    /// Move the host tensor out of the default scalar set when it holds this scalar type.
    ///
    /// Returns `None` when the set currently holds another member.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DefaultScalars, TensorScalar};
    ///
    /// let set = DefaultScalars::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let host = <f64 as TensorScalar>::from_default_scalars(set);
    /// assert_eq!(host.as_ref().map(|t| t.shape()), Some(&[2][..]));
    /// assert_eq!(host.as_ref().map(|t| t.as_slice()), Some(&[1.0, 2.0][..]));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn from_default_scalars(set: crate::DefaultScalars)
        -> Option<TypedTensor<Self, DynRank, Host>>;

    /// Wrap typed column-major data into a [`Tensor`] enum variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar};
    ///
    /// let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0])?;
    /// assert_eq!(tensor.dtype(), DType::F64);
    /// assert_eq!(tensor.shape(), &[2]);
    /// assert!(<f64 as TensorScalar>::into_tensor(vec![3], vec![1.0]).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::ShapeDataLengthMismatch`] when
    /// the shape product differs from `data.len()`, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when shape
    /// arithmetic overflows.
    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> crate::Result<Tensor>;

    /// Wrap a typed tensor into its dynamic [`Tensor`] enum variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, Tensor, TensorScalar, TypedTensor};
    ///
    /// let typed = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0])?;
    /// let tensor = <f64 as TensorScalar>::typed_tensor_into_tensor(typed);
    /// assert!(matches!(tensor.dtype(), DType::F64));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn typed_tensor_into_tensor(tensor: TypedTensor<Self>) -> Tensor;

    /// Borrow a typed tensor as a dtype-erased [`TensorRead`] view.
    ///
    /// This keeps the typed tensor borrowed instead of copying host data into
    /// a new dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let read = f64::tensor_read(&tensor);
    /// assert_eq!(read.dtype(), DType::F64);
    /// assert_eq!(read.shape(), &[2]);
    /// ```
    fn tensor_read(tensor: &TypedTensor<Self>) -> TensorRead<'_>;

    /// Wrap a typed borrowed view as a dtype-erased [`TensorView`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensorView};
    ///
    /// let data = [1.0_f64];
    /// let view = TypedTensorView::from_col_major(&[1], &data)?;
    /// assert_eq!(f64::tensor_view(view).dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn tensor_view<'a>(view: TypedTensorView<'a, Self>) -> TensorView<'a>;

    /// Wrap a typed mutable borrowed view as a dtype-erased [`TensorViewMut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensorViewMut};
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let view = TypedTensorViewMut::from_col_major(&[2], &mut data)?;
    /// let erased = f64::tensor_view_mut(view);
    /// assert_eq!(erased.dtype(), DType::F64);
    /// assert_eq!(erased.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn tensor_view_mut<'a>(view: TypedTensorViewMut<'a, Self>) -> TensorViewMut<'a>;

    /// Mutably borrow a typed tensor as a dtype-erased [`TensorWrite`] view.
    ///
    /// This keeps the typed output borrowed instead of wrapping it in a
    /// temporary dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensor};
    ///
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![0.0]).unwrap();
    /// let write = f64::tensor_write(&mut tensor);
    /// assert_eq!(write.dtype(), DType::F64);
    /// ```
    fn tensor_write(tensor: &mut TypedTensor<Self>) -> TensorWrite<'_>;

    /// Borrow the host data from a [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(<f64 as TensorScalar>::as_slice(&tensor)?, &[1.0, 2.0]);
    /// assert!(<f32 as TensorScalar>::as_slice(&tensor).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::DTypeMismatch`] when `tensor`
    /// is not the scalar type represented by this implementation, or
    /// [`crate::Error::RuntimeState`] when the matching tensor uses backend
    /// storage that has not been downloaded.
    fn as_slice(tensor: &Tensor) -> crate::Result<&[Self]>;

    /// Mutably borrow the host data from a [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// <f64 as TensorScalar>::as_slice_mut(&mut tensor)?[0] = 3.0;
    ///
    /// assert_eq!(tensor.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::DTypeMismatch`] when `tensor`
    /// is not the scalar type represented by this implementation, or
    /// [`crate::Error::RuntimeState`] when the matching tensor uses backend
    /// storage that has not been downloaded.
    fn as_slice_mut(tensor: &mut Tensor) -> crate::Result<&mut [Self]>;

    /// Extract a [`TypedTensor<Self>`] from a dynamic [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let Ok(typed) = <f64 as TensorScalar>::into_typed(tensor) else {
    ///     panic!("the dtype matches by construction")
    /// };
    ///
    /// assert_eq!(typed.as_slice()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when it is
    /// not the scalar type represented by this implementation, with
    /// [`crate::Error::Validation`] and
    /// [`tenferro_tensor_core::ValidationError::DTypeMismatch`] as the cause.
    fn into_typed(
        tensor: Tensor,
    ) -> std::result::Result<TypedTensor<Self>, ReinterpretError<Tensor>>;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for bool {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}

macro_rules! impl_tensor_scalar {
    ($ty:ty, $real:ty, $dtype:ident, $variant:ident) => {
        impl TensorScalar for $ty {
            type Real = $real;

            #[inline]
            fn dtype() -> DType {
                DType::$dtype
            }

            fn into_default_scalars(
                shape: Vec<usize>,
                data: Vec<Self>,
            ) -> crate::Result<crate::DefaultScalars> {
                TypedTensor::<Self, DynRank, Host>::from_host_vec_col_major(shape, data).map(
                    |tensor| {
                        crate::DefaultScalars::from_payload(
                            crate::default_scalars::DefaultScalarsValue::$variant(tensor),
                        )
                    },
                )
            }

            fn default_scalars_slice(set: &crate::DefaultScalars) -> Option<&[Self]> {
                match set.payload() {
                    crate::default_scalars::DefaultScalarsValue::$variant(tensor) => {
                        Some(tensor.as_slice())
                    }
                    _ => None,
                }
            }

            fn default_scalars_slice_mut(set: &mut crate::DefaultScalars) -> Option<&mut [Self]> {
                match set.payload_mut() {
                    crate::default_scalars::DefaultScalarsValue::$variant(tensor) => {
                        Some(tensor.host_data_mut())
                    }
                    _ => None,
                }
            }

            fn from_default_scalars(
                set: crate::DefaultScalars,
            ) -> Option<TypedTensor<Self, DynRank, Host>> {
                match set.into_payload() {
                    crate::default_scalars::DefaultScalarsValue::$variant(tensor) => Some(tensor),
                    _ => None,
                }
            }

            fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> crate::Result<Tensor> {
                TypedTensor::from_vec_col_major(shape, data).map(Self::typed_tensor_into_tensor)
            }

            fn typed_tensor_into_tensor(tensor: TypedTensor<Self>) -> Tensor {
                Tensor {
                    payload: TensorPayload::Native(PresetTensor::$variant(tensor)),
                }
            }

            fn tensor_read(tensor: &TypedTensor<Self>) -> TensorRead<'_> {
                TensorRead::from_view(TensorView::$variant(tensor.as_view()))
            }

            #[inline]
            fn tensor_view<'a>(view: TypedTensorView<'a, Self>) -> TensorView<'a> {
                TensorView::$variant(view)
            }

            #[inline]
            fn tensor_view_mut<'a>(view: TypedTensorViewMut<'a, Self>) -> TensorViewMut<'a> {
                TensorViewMut::$variant(view)
            }

            fn tensor_write(tensor: &mut TypedTensor<Self>) -> TensorWrite<'_> {
                TensorWrite::from_view(TensorViewMut::$variant(tensor.as_view_mut()))
            }

            fn as_slice(tensor: &Tensor) -> crate::Result<&[Self]> {
                tensor
                    .as_typed::<Self>()
                    .ok_or_else(|| {
                        crate::Error::validation(
                            "Tensor::as_slice",
                            ValidationError::DTypeMismatch {
                                expected: Self::dtype(),
                                actual: tensor.dtype(),
                            },
                        )
                    })?
                    .host_data()
            }

            fn as_slice_mut(tensor: &mut Tensor) -> crate::Result<&mut [Self]> {
                let actual = tensor.dtype();
                let typed = tensor.as_typed_mut::<Self>().ok_or_else(|| {
                    crate::Error::validation(
                        "Tensor::as_slice_mut",
                        ValidationError::DTypeMismatch {
                            expected: Self::dtype(),
                            actual,
                        },
                    )
                })?;
                typed.host_data_mut()
            }

            fn into_typed(
                tensor: Tensor,
            ) -> std::result::Result<TypedTensor<Self>, ReinterpretError<Tensor>> {
                let actual = tensor.dtype();
                match tensor.payload {
                    TensorPayload::Native(PresetTensor::$variant(typed)) => Ok(typed),
                    payload => Err(ReinterpretError::new(
                        Tensor { payload },
                        crate::Error::validation(
                            "TensorScalar::into_typed",
                            ValidationError::DTypeMismatch {
                                expected: Self::dtype(),
                                actual,
                            },
                        ),
                    )),
                }
            }
        }
    };
}

impl_tensor_scalar!(f64, f64, F64, F64);
impl_tensor_scalar!(f32, f32, F32, F32);
impl_tensor_scalar!(i64, i64, I64, I64);
impl_tensor_scalar!(i32, i32, I32, I32);
impl_tensor_scalar!(bool, bool, Bool, Bool);
impl_tensor_scalar!(Complex64, f64, C64, C64);
impl_tensor_scalar!(Complex32, f32, C32, C32);

/// Dynamic tensor enum over the supported scalar types.
///
/// The enum keeps dtype dynamic and rank dynamic. Use
/// [`TypedTensor<T, R>`](TypedTensor) directly when the scalar type or rank
/// should be represented in Rust's type system.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let t = Tensor::from_typed(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
/// assert_eq!(t.shape(), &[2]);
///
/// let erased = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0]).unwrap();
/// assert_eq!(erased.shape().len(), 2);
/// ```
/// The owning payload behind the erased [`Tensor`].
///
/// `Native` stores a preset scalar owner directly; `External` retains its
/// existing explicit caller-owned host payload.
#[derive(Debug)]
enum TensorPayload {
    Native(PresetTensor),
    External(crate::ErasedHostTensor, Placement),
}

#[derive(Debug)]
enum PresetTensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    I32(TypedTensor<i32>),
    I64(TypedTensor<i64>),
    Bool(TypedTensor<bool>),
    C32(TypedTensor<Complex32>),
    C64(TypedTensor<Complex64>),
}

macro_rules! with_preset {
    ($preset:expr, |$value:ident| $body:expr) => {
        match $preset {
            PresetTensor::F32($value) => $body,
            PresetTensor::F64($value) => $body,
            PresetTensor::I32($value) => $body,
            PresetTensor::I64($value) => $body,
            PresetTensor::Bool($value) => $body,
            PresetTensor::C32($value) => $body,
            PresetTensor::C64($value) => $body,
        }
    };
}

/// Dynamic tensor over the supported scalar types.
///
/// The erased tensor keeps dtype and rank dynamic: each preset scalar retains its
/// typed owner without constructing a group, while caller-owned scalars remain
/// `External` payloads recovered by their own type. Use [`TypedTensor<T, R>`](TypedTensor)
/// directly when the scalar type or rank should be represented in Rust's type system.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// assert_eq!(tensor.dtype(), DType::F64);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct Tensor {
    payload: TensorPayload,
}

impl<R: TensorRank> OwnedTensorGroup<R> {
    /// Whether this group's descriptor names a non-CPU provider.
    fn is_backend_buffer(&self) -> bool {
        !matches!(
            self.group.provider_kind(self.slot),
            None | Some(crate::storage::ProviderKind::Cpu)
        )
    }
}

impl Tensor {
    /// Carry an externally defined scalar as a caller-owned payload.
    ///
    /// The payload keeps its own element type and is recovered by that type, so no
    /// bytes are reinterpreted. Placement defaults to unpinned host memory, which
    /// is where a caller-owned payload lives.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor};
    /// use tenferro_tensor::{DynRank, ErasedHostTensor, Host, TypedTensor};
    ///
    /// let payload = ErasedHostTensor::new(
    ///     TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![1.0_f64])?,
    /// );
    /// let element = payload.element_type_id();
    /// let tensor = Tensor::external(payload);
    /// assert_eq!(tensor.dtype(), DType::External(element));
    /// assert_eq!(tensor.shape(), &[1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn external(payload: crate::ErasedHostTensor) -> Self {
        Self {
            payload: TensorPayload::External(payload, Placement::default()),
        }
    }

    /// Build a tensor from a typed one, without naming its variant.
    ///
    /// A call site that constructs a tensor from a typed tensor should use this rather than a variant, so
    /// that changing how the erased representation is stored changes this function and not its 1290 call
    /// sites. The variants remain until the removal's last step, so both forms currently produce the same
    /// value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let typed = tensor.into_typed::<f64>().unwrap();
    /// let rebuilt = Tensor::from_typed(typed);
    /// assert_eq!(rebuilt.as_typed::<f64>().unwrap().shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn from_typed<T: TensorScalar>(typed: TypedTensor<T>) -> Self {
        T::typed_tensor_into_tensor(typed)
    }

    /// Borrow the erased payload of an externally defined tensor.
    ///
    /// This is the counterpart of [`Tensor::external`] for dispatch: a table that matches on
    /// [`Tensor::dtype`] reaches the externally defined tag and needs the payload that tag stands
    /// for, just as the typed tags reach theirs through [`Tensor::as_typed`]. Every other tag
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    /// use tenferro_tensor::{DynRank, ErasedHostTensor, Host, TypedTensor};
    ///
    /// let payload = ErasedHostTensor::new(
    ///     TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![1.0_f64])?,
    /// );
    /// let tensor = Tensor::external(payload);
    /// assert!(tensor.external_payload().is_some());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn external_payload(&self) -> Option<&crate::ErasedHostTensor> {
        match &self.payload {
            TensorPayload::External(payload, _) => Some(payload),
            TensorPayload::Native(_) => None,
        }
    }

    /// Carry an externally defined payload with an explicit placement.
    ///
    /// [`Tensor::external`] defaults the placement to unpinned host memory, which is where
    /// a caller-owned payload normally lives; this entry point is for a caller that knows
    /// the placement it wants.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Placement, Tensor};
    /// use tenferro_tensor::{DynRank, ErasedHostTensor, Host, TypedTensor};
    ///
    /// let payload = ErasedHostTensor::new(
    ///     TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![1.0_f64])?,
    /// );
    /// let tensor = Tensor::external_with_placement(payload, Placement::default());
    /// assert!(tensor.external_payload().is_some());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn external_with_placement(payload: crate::ErasedHostTensor, placement: Placement) -> Self {
        Self {
            payload: TensorPayload::External(payload, placement),
        }
    }

    /// Mutably borrow the erased payload of an externally defined tensor.
    ///
    /// The counterpart of [`Tensor::external_payload`] for callers that update the payload
    /// in place, such as a mutation test that checks the copy boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    /// use tenferro_tensor::{DynRank, ErasedHostTensor, Host, TypedTensor};
    ///
    /// let mut tensor = Tensor::external(ErasedHostTensor::new(
    ///     TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![1.0_f64])?,
    /// ));
    /// assert!(tensor.external_payload_mut().is_some());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn external_payload_mut(&mut self) -> Option<&mut crate::ErasedHostTensor> {
        match &mut self.payload {
            TensorPayload::External(payload, _) => Some(payload),
            TensorPayload::Native(_) => None,
        }
    }

    pub(crate) fn into_group_parts(self) -> (AllocationGroup, DescriptorSlot) {
        match self.payload {
            TensorPayload::Native(preset) => with_preset!(preset, |typed| typed.into_group_parts()),
            // INVARIANT: a caller-owned payload has no allocation group.
            TensorPayload::External(..) => {
                unreachable!("an externally defined payload has no allocation group")
            }
        }
    }
}

/// Dynamic read-only borrowed tensor view.
///
/// `TensorView` keeps dtype erased while borrowing typed view metadata and
/// storage. Use [`TypedTensorView`] directly when the scalar type is statically
/// known.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, TensorView, TypedTensorView};
///
/// let data = [1_i32, 2, 3, 4];
/// let typed = TypedTensorView::from_slice([2, 2], [1, 2], 0, &data)?;
/// let view = TensorView::I32(typed);
///
/// assert_eq!(view.dtype(), DType::I32);
/// assert_eq!(view.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub enum TensorView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool(TypedTensorView<'a, bool>),
    C32(TypedTensorView<'a, Complex<f32>>),
    C64(TypedTensorView<'a, Complex<f64>>),
}

/// Dynamic mutable borrowed tensor view.
///
/// `TensorViewMut` is the mutable counterpart to [`TensorView`]. It keeps the
/// dtype erased while preserving the typed mutable view's shape, strides, and
/// offset metadata.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, TensorViewMut, TypedTensorViewMut};
///
/// let mut data = [1.0_f64, 2.0];
/// let view = TensorViewMut::F64(TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?);
/// assert_eq!(view.dtype(), DType::F64);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum TensorViewMut<'a> {
    F32(TypedTensorViewMut<'a, f32>),
    F64(TypedTensorViewMut<'a, f64>),
    I32(TypedTensorViewMut<'a, i32>),
    I64(TypedTensorViewMut<'a, i64>),
    Bool(TypedTensorViewMut<'a, bool>),
    C32(TypedTensorViewMut<'a, Complex<f32>>),
    C64(TypedTensorViewMut<'a, Complex<f64>>),
}

/// Read-only tensor input accepted by synchronous eager kernels.
///
/// `TensorRead` lets kernels accept either an owned tensor reference or a
/// borrowed [`TensorView`] without forcing callers to materialize first.
/// The `View` variant preserves arbitrary strides and offsets, so kernels that
/// support strided reads can consume transposes, slices, and broadcasts directly.
///
/// `TensorRead` is intentionally borrowed. It is an input-dispatch type, not an
/// owned lazy tensor value. APIs that need to store a lazy layout result should
/// keep an owned base tensor plus layout metadata, then expose a `TensorRead`
/// only for the duration of kernel dispatch.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TensorRead};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let read = TensorRead::from_tensor(&tensor);
///
/// assert_eq!(read.dtype(), DType::F64);
/// assert_eq!(read.shape(), &[2]);
/// ```
// Keep borrowed views inline to avoid allocation on read-only tensor dispatch paths.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum TensorRead<'a> {
    Tensor(&'a Tensor),
    View(TensorView<'a>),
}

/// Mutable typed tensor output accepted by synchronous eager kernels.
///
/// `TypedTensorWrite` is the typed counterpart to [`TensorWrite`]. It accepts
/// either an owned compact [`TypedTensor`] or an arbitrary-strided mutable
/// [`TypedTensorViewMut`] without erasing the scalar type at the public API
/// boundary.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{TypedTensorViewMut, TypedTensorWrite};
///
/// let mut data = [0.0_f64, 1.0, 0.0, 2.0];
/// let view = TypedTensorViewMut::from_slice([2], [2], 1, &mut data)?;
/// let write = TypedTensorWrite::from_view(view).into_tensor_write();
/// assert_eq!(write.shape(), &[2]);
/// assert_eq!(write.strides()?, [2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum TypedTensorWrite<'a, T> {
    /// An owned compact typed tensor borrowed mutably for the write.
    Tensor(&'a mut TypedTensor<T>),
    /// An arbitrary-strided mutable typed tensor view.
    View(TypedTensorViewMut<'a, T>),
}

impl<'a, T> TypedTensorWrite<'a, T> {
    /// Create a writable target from an owned typed tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TypedTensor, TypedTensorWrite};
    ///
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![0.0])?;
    /// let write = TypedTensorWrite::from_tensor(&mut tensor);
    /// assert!(matches!(write, TypedTensorWrite::Tensor(_)));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_tensor(tensor: &'a mut TypedTensor<T>) -> Self {
        Self::Tensor(tensor)
    }

    /// Create a writable target from a mutable typed tensor view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TypedTensorViewMut, TypedTensorWrite};
    ///
    /// let mut data = [0.0_f64, 1.0];
    /// let view = TypedTensorViewMut::from_col_major(&[2], &mut data)?;
    /// let write = TypedTensorWrite::from_view(view);
    /// assert!(matches!(write, TypedTensorWrite::View(_)));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_view(view: TypedTensorViewMut<'a, T>) -> Self {
        Self::View(view)
    }
}

impl<'a, T: TensorScalar> TypedTensorWrite<'a, T> {
    /// Erase the scalar type while preserving the output layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TypedTensor, TypedTensorWrite};
    ///
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0; 2])?;
    /// let write = TypedTensorWrite::from_tensor(&mut tensor).into_tensor_write();
    /// assert_eq!(write.dtype(), DType::F64);
    /// assert_eq!(write.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_tensor_write(self) -> TensorWrite<'a> {
        match self {
            Self::Tensor(tensor) => T::tensor_write(tensor),
            Self::View(view) => TensorWrite::from_view(T::tensor_view_mut(view)),
        }
    }
}

impl<'a, T> From<&'a mut TypedTensor<T>> for TypedTensorWrite<'a, T> {
    fn from(tensor: &'a mut TypedTensor<T>) -> Self {
        Self::from_tensor(tensor)
    }
}

impl<'a, T> From<TypedTensorViewMut<'a, T>> for TypedTensorWrite<'a, T> {
    fn from(view: TypedTensorViewMut<'a, T>) -> Self {
        Self::from_view(view)
    }
}

/// Mutable tensor output accepted by synchronous eager kernels.
///
/// `TensorWrite` mirrors [`TensorRead`] for output dispatch: it can target an
/// owned compact [`Tensor`] or a borrowed mutable [`TensorViewMut`]. The target
/// is never resized.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TensorWrite};
///
/// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// let write = TensorWrite::from_tensor(&mut tensor);
/// assert_eq!(write.shape(), &[1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum TensorWrite<'a> {
    Tensor(&'a mut Tensor),
    View(TensorViewMut<'a>),
}

/// Owned tensor value with one move-only physical owner and metadata-only layout.
///
/// `TensorValue` is intentionally not cloneable. View transformations consume
/// the value and move its existing owner; [`TensorValue::duplicate`] is the
/// explicit boundary for creating another physical allocation.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TensorValue};
/// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
/// let view = value.transpose_view([1, 0])?;
/// assert_eq!(view.strides(), &[2, 1]);
/// assert!(view.is_view());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct TensorValue {
    owner: Tensor,
    layout: TensorLayout<DynRank>,
}

/// A consuming view transformation failed while retaining its unchanged owner.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TensorValue};
/// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2], vec![3., 4.])?);
/// let failure = value.try_reshape_view([3]).unwrap_err();
/// let (recovered, cause) = failure.into_parts();
/// assert_eq!(recovered.into_tensor()?.as_slice::<f64>()?, &[3., 4.]);
/// assert!(matches!(cause, tenferro_tensor::Error::Validation { .. }));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct TensorValueViewError {
    value: TensorValue,
    source: crate::Error,
}

impl TensorValueViewError {
    /// Return the unchanged value and the typed validation/backend error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([1], vec![7.])?);
    /// let (value, error) = value.try_reshape_view([2]).unwrap_err().into_parts();
    /// assert_eq!(value.into_tensor()?.as_slice::<f64>()?, &[7.]);
    /// assert!(matches!(error, tenferro_tensor::Error::Validation { .. }));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_parts(self) -> (TensorValue, crate::Error) {
        (self.value, self.source)
    }

    fn new(value: TensorValue, source: crate::Error) -> Self {
        Self { value, source }
    }
}

impl std::fmt::Display for TensorValueViewError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.source, formatter)
    }
}

impl std::error::Error for TensorValueViewError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl TensorValue {
    /// Explicitly duplicate the physical owner represented by this value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2], vec![3., 4.])?);
    /// let copy = value.duplicate()?.into_tensor()?;
    /// assert_eq!(copy.as_slice::<f64>()?, &[3., 4.]);
    /// assert_eq!(value.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] or [`crate::Error::Unsupported`]
    /// when the backend/storage owner cannot be duplicated.
    pub fn duplicate(&self) -> crate::Result<Self> {
        let tensor = self.owner.duplicate()?;
        Self::from_parts(
            tensor,
            self.shape().to_vec(),
            self.strides().to_vec(),
            self.offset(),
        )
    }

    /// Retain a compact tensor as a move-only value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2], vec![3., 4.])?);
    /// assert!(!value.is_view());
    /// assert_eq!(value.into_tensor()?.as_slice::<f64>()?, &[3., 4.]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_tensor(tensor: Tensor) -> Self {
        let layout = tensor_layout(&tensor);
        Self {
            owner: tensor,
            layout,
        }
    }

    /// # Errors
    ///
    /// Returns [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::IntegerOverflow`] when the supplied layout is
    /// invalid for the tensor's physical buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let tensor = Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?;
    /// let view = TensorValue::from_parts(tensor, vec![2], vec![1], 2)?;
    /// assert_eq!(view.shape(), &[2]);
    /// assert_eq!(view.offset(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_parts(
        tensor: Tensor,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(
            shape.into(),
            strides.into(),
            offset,
            tensor_buffer_len(&tensor),
        )
        .map_err(|err| tensor_layout_error("TensorValue::from_parts", err))?;
        Ok(Self {
            owner: tensor,
            layout,
        })
    }

    // INVARIANT: unchanged-owner recovery is part of the consuming ownership
    // contract, so this intentionally carries the large move-only error.
    #[doc(hidden)]
    /// Move the value's sole physical owner into an allocation group.
    ///
    /// This preserves metadata-only views without copying. The consumed value
    /// always contains one unique owner, so no compatibility fallback exists.
    ///
    /// # Errors
    ///
    /// Returns the unchanged value when descriptor publication fails because
    /// of [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::IntegerOverflow`].
    #[allow(clippy::result_large_err)]
    pub fn try_into_group_parts(
        self,
    ) -> std::result::Result<(AllocationGroup, DescriptorSlot, DType, Vec<usize>), Self> {
        let Self { owner, layout } = self;
        if owner.external_payload().is_some() {
            // A caller-owned payload owns no allocation group, so it is returned
            // unchanged instead of being forced into one.
            return Err(Self { owner, layout });
        }
        let dtype = owner.dtype();
        let shape = layout.shape().to_vec();
        let strides = layout.strides().to_vec();
        let offset = layout.offset();
        let (group, slot) = owner.into_group_parts();
        match group.update_descriptor_layout(slot, shape, strides, offset) {
            Ok(group) => Ok((group, slot, dtype, layout.shape().to_vec())),
            Err((_group, _error)) => {
                unreachable!("TensorValue layout was validated before group ownership transfer")
            }
        }
    }

    /// Consume a value with its owner's original layout and return that owner.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2], vec![3., 4.])?);
    /// assert_eq!(value.into_tensor()?.as_slice::<f64>()?, &[3., 4.]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] when the value's metadata-only
    /// view differs from its physical owner's layout.
    pub fn into_tensor(self) -> crate::Result<Tensor> {
        if self.layout != tensor_layout(&self.owner) {
            return Err(crate::Error::unsupported(
                "TensorValue::into_tensor",
                "a metadata-only view has no compact tensor owner",
            ));
        }
        Ok(self.owner)
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// assert_eq!(value.as_tensor().unwrap().as_slice::<f64>()?, &[1., 2., 3., 4.]);
    /// assert!(value.transpose_view([1, 0])?.as_tensor().is_none());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_tensor(&self) -> Option<&Tensor> {
        (self.layout == tensor_layout(&self.owner)).then_some(&self.owner)
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// assert!(!value.is_view());
    /// assert!(value.transpose_view([1, 0])?.is_view());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_view(&self) -> bool {
        self.as_tensor().is_none()
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// assert_eq!(value.dtype(), tenferro_tensor::DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        self.owner.dtype()
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// assert_eq!(value.reshape_view([4])?.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// assert_eq!(value.transpose_view([1, 0])?.strides(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// assert_eq!(value.offset(), 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// let view = value.tensor_view();
    /// assert_eq!(view.shape(), &[2, 2]);
    /// assert_eq!(view.dtype(), tenferro_tensor::DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn tensor_view(&self) -> TensorView<'_> {
        tensor_view_with_layout(&self.owner, self.layout.clone())
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// let read = value.tensor_read();
    /// assert_eq!(read.shape(), &[2, 2]);
    /// assert_eq!(read.dtype(), tenferro_tensor::DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn tensor_read(&self) -> TensorRead<'_> {
        self.as_tensor()
            .map(TensorRead::from_tensor)
            .unwrap_or_else(|| TensorRead::from_view(self.tensor_view()))
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::InvalidPermutationLength`],
    /// [`tenferro_tensor_core::ValidationError::AxisOutOfBounds`], or
    /// [`tenferro_tensor_core::ValidationError::DuplicateAxis`] when `axes` is
    /// not a valid permutation of the value rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// let view = value.transpose_view([1, 0])?;
    /// assert_eq!(view.strides(), &[2, 1]);
    /// assert!(view.is_view());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn transpose_view(self, axes: impl AsRef<[usize]>) -> crate::Result<Self> {
        let layout = self
            .layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TensorValue::transpose_view", err))?;
        Ok(Self {
            owner: self.owner,
            layout,
        })
    }

    // INVARIANT: unchanged-owner recovery is part of the consuming view
    // contract, so this intentionally carries the large move-only error.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::NonContiguousViewAsSlice`] when
    /// the source is not compact column-major,
    /// [`tenferro_tensor_core::ValidationError::ShapeMismatch`] (whose
    /// [`tenferro_tensor_core::ShapeMismatch::ReshapeElementCount`] source
    /// records the counts) when element counts differ,
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for shape
    /// arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// reshaped view exceeds the backing buffer.
    #[allow(clippy::result_large_err)]
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// let (recovered, error) = value.try_reshape_view([3]).unwrap_err().into_parts();
    /// assert_eq!(recovered.into_tensor()?.as_slice::<f64>()?, &[1., 2., 3., 4.]);
    /// assert!(matches!(error, tenferro_tensor::Error::Validation { .. }));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_reshape_view(
        self,
        shape: impl tenferro_tensor_core::IntoShapeVec,
    ) -> std::result::Result<Self, TensorValueViewError> {
        let shape = shape.into_shape_vec();
        let layout = match reshape_layout_dyn(
            &self.layout,
            &shape,
            tensor_buffer_len(&self.owner),
            "TensorValue::reshape_view",
        ) {
            Ok(layout) => layout,
            Err(error) => return Err(TensorValueViewError::new(self, error)),
        };
        Ok(Self {
            owner: self.owner,
            layout,
        })
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::NonContiguousViewAsSlice`]
    /// when the source is not compact, [`tenferro_tensor_core::ValidationError::ShapeMismatch`]
    /// when element counts differ, or [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// / [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] for invalid
    /// target-shape arithmetic or bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// let view = value.reshape_view([4])?;
    /// assert_eq!(view.shape(), &[4]);
    /// assert_eq!(view.strides(), &[1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn reshape_view(
        self,
        shape: impl tenferro_tensor_core::IntoShapeVec,
    ) -> crate::Result<Self> {
        self.try_reshape_view(shape).map_err(|error| error.source)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when a slice
    /// vector does not match the value rank,
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when a bound
    /// or stride cannot be represented or is invalid,
    /// [`tenferro_tensor_core::ValidationError::InvalidSliceStep`] or
    /// [`tenferro_tensor_core::ValidationError::InvalidSliceBounds`] for slice
    /// parameters, [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// for slice arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// result exceeds the backing buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2, 2], vec![1., 2., 3., 4.])?);
    /// let view = value.slice_view(&tenferro_tensor::SliceConfig {
    ///    starts: vec![0, 1], limits: vec![2, 2], strides: vec![1, 1],
    /// })?;
    /// assert_eq!(view.shape(), &[2, 1]);
    /// assert_eq!(view.offset(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn slice_view(self, config: &SliceConfig) -> crate::Result<Self> {
        let op = "TensorValue::slice_view";
        if config.starts.len() != self.shape().len()
            || config.limits.len() != self.shape().len()
            || config.strides.len() != self.shape().len()
        {
            return Err(crate::Error::validation(
                op,
                ValidationError::RankMismatch {
                    expected: self.shape().len(),
                    actual: config.starts.len(),
                },
            ));
        }
        // The per-axis specs stay inline: a metadata-only slice must not
        // allocate its own scratch at rank <= 8 (#2040).
        let mut specs = SmallVec::<[CoreSliceSpec; 8]>::with_capacity(self.shape().len());
        for (((&start, &limit), &stride), &axis_len) in config
            .starts
            .iter()
            .zip(config.limits.iter())
            .zip(config.strides.iter())
            .zip(self.shape())
        {
            let start = isize::try_from(start).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "slice start",
                    "slice start does not fit in isize",
                )
            })?;
            let limit = isize::try_from(limit).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "slice limit",
                    "slice limit does not fit in isize",
                )
            })?;
            let stride = isize::try_from(stride).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "slice stride",
                    "slice stride does not fit in isize",
                )
            })?;
            specs.push(core_slice_spec(
                StridedSliceSpec::new(start, Some(limit), stride),
                axis_len,
                op,
            )?);
        }
        let layout = self
            .layout
            .slice_view(&specs, tensor_buffer_len(&self.owner))
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            owner: self.owner,
            layout,
        })
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`],
    /// [`tenferro_tensor_core::ValidationError::AxisOutOfBounds`], or
    /// [`tenferro_tensor_core::ValidationError::DuplicateAxis`] for invalid
    /// dimension mappings,
    /// [`tenferro_tensor_core::ValidationError::ShapeDataLengthMismatch`] for
    /// incompatible extents,
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// result exceeds the backing buffer, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorValue};
    /// let value = TensorValue::from_tensor(Tensor::from_vec_col_major([2], vec![3., 4.])?);
    /// let view = value.broadcast_in_dim_view([2, 3], [0])?;
    /// assert_eq!(view.shape(), &[2, 3]);
    /// assert_eq!(view.strides(), &[1, 0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn broadcast_in_dim_view(
        self,
        shape: impl tenferro_tensor_core::IntoShapeVec,
        dims: impl AsRef<[usize]>,
    ) -> crate::Result<Self> {
        let shape = shape.into_shape_vec();
        let layout = self
            .layout
            .broadcast_in_dim_view::<DynRank>(shape, dims, tensor_buffer_len(&self.owner))
            .map_err(|err| tensor_layout_error("TensorValue::broadcast_in_dim_view", err))?;
        Ok(Self {
            owner: self.owner,
            layout,
        })
    }
}

fn tensor_layout(tensor: &Tensor) -> TensorLayout<DynRank> {
    match &tensor.payload {
        // A caller-owned payload carries its own view layout, so strides and offset
        // come from the payload rather than from a compact assumption.
        TensorPayload::External(payload, _) => TensorLayout::<DynRank>::from_parts(
            tenferro_tensor_core::ShapeVec::from_slice(payload.shape()),
            tenferro_tensor_core::StrideVec::from_slice(payload.strides()),
            payload.offset(),
            payload.payload_element_count(),
        )
        .unwrap_or_else(|_| {
            // INVARIANT: a payload is built from a validated host tensor, and the
            // only layout change is a permutation, so its strides stay inside the
            // payload it was created from and this construction cannot fail.
            unreachable!("a validated payload yields a layout inside its own storage")
        }),
        TensorPayload::Native(preset) => with_preset!(preset, |typed| typed.physical_layout()),
    }
}

fn tensor_buffer_len(tensor: &Tensor) -> usize {
    match &tensor.payload {
        // A caller-owned payload stores its element count directly.
        TensorPayload::External(payload, _) => payload.element_count(),
        TensorPayload::Native(preset) => with_preset!(preset, |typed| typed.buffer_len()),
    }
}

fn prepare_backend_access<'a, T: 'static, R: TensorRank>(
    buffer: &'a dyn BackendStorage<T>,
    layout: &'a TensorLayout<R>,
    op: &'static str,
) -> crate::Result<Box<dyn PreparedDeviceAccess + 'a>> {
    let domain = buffer.allocation_domain().ok_or_else(|| {
        crate::Error::runtime_state(op, "backend buffer is missing an allocation domain")
    })?;
    let allocation_id = buffer.allocation_id().ok_or_else(|| {
        crate::Error::runtime_state(op, "backend buffer is missing an allocation identity")
    })?;
    let byte_len = buffer
        .len()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    let request = DeviceAccessRequest::new(
        domain,
        allocation_id,
        byte_len,
        size_of::<T>(),
        layout.shape(),
        layout.strides(),
        layout.offset(),
    );
    buffer
        .prepare_device_access(request)
        .map_err(|error| crate::Error::runtime_state_source(op, error))
}

fn cast_view_slice<S: 'static, T: TensorScalar>(source: &[S]) -> crate::Result<&[T]> {
    if size_of::<S>() != size_of::<T>() || align_of::<S>() != align_of::<T>() {
        return Err(crate::Error::invalid_argument(
            "TensorView::as_slice",
            "dtype",
            "matching dtypes must have identical scalar layout",
        ));
    }
    // SAFETY: the dtype check above is exhaustive over the sealed scalar set;
    // equal size/alignment preserve the element boundaries and the source
    // slice remains borrowed for the returned lifetime.
    Ok(unsafe { std::slice::from_raw_parts(source.as_ptr().cast::<T>(), source.len()) })
}

/// Re-layout a borrowed typed view while keeping its buffer, retained root and
/// placement.
///
/// #2143 item 1: an eager implicit broadcast is a metadata change, so the view
/// keeps reading the same storage through zero strides instead of copying the
/// broadcast tensor into a fresh allocation.
fn broadcast_typed_view<'a, T: 'static, R: TensorRank>(
    view: TypedTensorView<'a, T, R>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensorView<'a, T, R>> {
    let shape = R::shape_from_vec(ShapeVec::from_slice(shape))
        .map_err(|err| tensor_layout_error("TensorRead::broadcast_in_dim_view", err))?;
    let layout = view
        .layout()
        .broadcast_in_dim_view::<R>(shape, dims, view.backing_len())
        .map_err(|err| tensor_layout_error("TensorRead::broadcast_in_dim_view", err))?;
    Ok(TypedTensorView {
        buffer: view.buffer,
        root: view.root,
        layout,
        placement: view.placement,
        _representation: std::marker::PhantomData,
    })
}

fn tensor_view_with_layout(tensor: &Tensor, layout: TensorLayout<DynRank>) -> TensorView<'_> {
    match tensor.dtype() {
        DType::F32 => TensorView::F32(typed_view_with_layout(
            tensor
                .as_typed::<f32>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm")),
            layout,
        )),
        DType::F64 => TensorView::F64(typed_view_with_layout(
            tensor
                .as_typed::<f64>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm")),
            layout,
        )),
        DType::I32 => TensorView::I32(typed_view_with_layout(
            tensor
                .as_typed::<i32>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm")),
            layout,
        )),
        DType::I64 => TensorView::I64(typed_view_with_layout(
            tensor
                .as_typed::<i64>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm")),
            layout,
        )),
        DType::Bool => TensorView::Bool(typed_view_with_layout(
            tensor
                .as_typed::<bool>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm")),
            layout,
        )),
        DType::C32 => TensorView::C32(typed_view_with_layout(
            tensor
                .as_typed::<Complex<f32>>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm")),
            layout,
        )),
        DType::C64 => TensorView::C64(typed_view_with_layout(
            tensor
                .as_typed::<Complex<f64>>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm")),
            layout,
        )),
        // INVARIANT: `TensorView` has no externally defined variant, and a view is
        // never requested for a caller-owned payload.
        DType::External(_) => unreachable!("views cover the preset scalars"),
    }
}

fn typed_view_with_layout<T: TensorScalar + 'static>(
    tensor: &TypedTensor<T>,
    layout: TensorLayout<DynRank>,
) -> TypedTensorView<'_, T> {
    match &tensor.storage {
        DynamicStorage::Host(HostStorage { data }) => TypedTensorView {
            buffer: TensorStorageRef::Host(data.as_slice()),
            root: None,
            layout,
            placement: tensor.placement.clone(),
            _representation: std::marker::PhantomData,
        },
        DynamicStorage::Group(core) => {
            let root = core.group.view::<T>().unwrap_or_else(|error| {
                unreachable!("typed tensor group descriptor mismatch: {error}")
            });
            let buffer = if let Some(allocation) = root.backend_allocation() {
                TensorStorageRef::Root(allocation)
            } else {
                TensorStorageRef::Host(tensor.group_host_slice())
            };
            TypedTensorView {
                buffer,
                root: Some(root),
                layout,
                placement: tensor.placement.clone(),
                _representation: std::marker::PhantomData,
            }
        }
    }
}

pub(crate) fn tensor_view_from_group<'a, T: TensorScalar>(
    view: GroupReadView<'a, T, DynRank>,
) -> crate::Result<TensorView<'a>> {
    let buffer = if let Some(allocation) = view.backend_allocation() {
        TensorStorageRef::Root(allocation)
    } else {
        let storage = view.storage_buffer().ok_or_else(|| {
            crate::Error::runtime_state(
                "AllocationGroup::tensor_read",
                "group descriptor has no backing storage",
            )
        })?;
        match storage {
            StorageBuffer::Host(data) => TensorStorageRef::Host(data),
            StorageBuffer::Backend(buffer) => TensorStorageRef::Backend(buffer.as_ref()),
        }
    };
    let layout = view.descriptor().layout().clone();
    let placement = view.descriptor().placement().clone();
    let typed = TypedTensorView {
        buffer,
        root: Some(view.clone()),
        layout,
        placement,
        _representation: std::marker::PhantomData,
    };
    Ok(T::tensor_view(typed))
}

pub(crate) fn tensor_from_group(
    group: AllocationGroup,
    slot: DescriptorSlot,
    allocation_index: AllocationSlot,
    dtype: DType,
    layout: TensorLayout<DynRank>,
    placement: Placement,
) -> Tensor {
    fn typed<T: TensorScalar>(
        group: AllocationGroup,
        slot: DescriptorSlot,
        allocation_index: AllocationSlot,
        layout: TensorLayout<DynRank>,
        placement: Placement,
    ) -> TypedTensor<T> {
        let descriptor_dtype = group
            .descriptor_dtype(slot)
            .unwrap_or_else(|| unreachable!("tensor_from_group requires a live typed descriptor"));
        assert_eq!(
            descriptor_dtype,
            T::dtype(),
            "managed typed owner must match its descriptor dtype",
        );
        let (host_ptr, host_byte_len) = host_metadata::<T>(&group, slot);
        TypedTensor {
            shape: shape_vec(layout.shape()),
            placement,
            layout: stride_vec(layout.strides()),
            storage: DynamicStorage::Group(GroupStorage {
                group: Box::new(OwnedTensorGroup {
                    group,
                    slot,
                    allocation_index,
                    host_ptr,
                    host_byte_len,
                    _rank: PhantomData,
                }),
            }),
        }
    }

    match dtype {
        DType::F32 => <f32 as TensorScalar>::typed_tensor_into_tensor(typed::<f32>(
            group,
            slot,
            allocation_index,
            layout,
            placement,
        )),
        DType::F64 => <f64 as TensorScalar>::typed_tensor_into_tensor(typed::<f64>(
            group,
            slot,
            allocation_index,
            layout,
            placement,
        )),
        DType::I32 => <i32 as TensorScalar>::typed_tensor_into_tensor(typed::<i32>(
            group,
            slot,
            allocation_index,
            layout,
            placement,
        )),
        DType::I64 => <i64 as TensorScalar>::typed_tensor_into_tensor(typed::<i64>(
            group,
            slot,
            allocation_index,
            layout,
            placement,
        )),
        DType::Bool => <bool as TensorScalar>::typed_tensor_into_tensor(typed::<bool>(
            group,
            slot,
            allocation_index,
            layout,
            placement,
        )),
        DType::C32 => <Complex<f32> as TensorScalar>::typed_tensor_into_tensor(
            typed::<Complex<f32>>(group, slot, allocation_index, layout, placement),
        ),
        DType::C64 => <Complex<f64> as TensorScalar>::typed_tensor_into_tensor(
            typed::<Complex<f64>>(group, slot, allocation_index, layout, placement),
        ),
        // INVARIANT: an allocation group is created from a sealed preset scalar,
        // so it can never carry an externally defined one. External payloads are
        // caller-owned and do not enter a group.
        DType::External(_) => unreachable!("allocation groups are preset-typed"),
    }
}

/// Wrap an `f64` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<f64>> for Tensor {
    fn from(t: TypedTensor<f64>) -> Self {
        Tensor::from_typed(t)
    }
}

/// Wrap an `f32` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<f32>> for Tensor {
    fn from(t: TypedTensor<f32>) -> Self {
        Tensor::from_typed(t)
    }
}

/// Wrap an `i64` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::I64);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<i64>> for Tensor {
    fn from(t: TypedTensor<i64>) -> Self {
        Tensor::from_typed(t)
    }
}

/// Wrap an `i32` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::I32);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<i32>> for Tensor {
    fn from(t: TypedTensor<i32>) -> Self {
        Tensor::from_typed(t)
    }
}

/// Wrap a `bool` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::Bool);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<bool>> for Tensor {
    fn from(t: TypedTensor<bool>) -> Self {
        Tensor::from_typed(t)
    }
}

/// Wrap a [`Complex64`] [`TypedTensor`] into the corresponding [`Tensor`]
/// variant.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(
///     vec![1],
///     vec![Complex64::new(1.0, 2.0)],
/// ).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[1]);
/// ```
impl From<TypedTensor<Complex<f64>>> for Tensor {
    fn from(t: TypedTensor<Complex<f64>>) -> Self {
        Tensor::from_typed(t)
    }
}

/// Wrap a [`Complex32`] [`TypedTensor`] into the corresponding [`Tensor`]
/// variant.
///
/// # Examples
///
/// ```
/// use num_complex::Complex32;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(
///     vec![1],
///     vec![Complex32::new(1.0, 2.0)],
/// ).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[1]);
/// ```
impl From<TypedTensor<Complex<f32>>> for Tensor {
    fn from(t: TypedTensor<Complex<f32>>) -> Self {
        Tensor::from_typed(t)
    }
}

impl<'a> TensorView<'a> {
    /// Create a dynamic `f32` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1.0_f32, 2.0];
    /// let view = TensorView::f32(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::F32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn f32(shape: &'a [usize], data: &'a [f32]) -> crate::Result<Self> {
        Ok(Self::F32(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `f64` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1.0_f64, 2.0];
    /// let view = TensorView::f64(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn f64(shape: &'a [usize], data: &'a [f64]) -> crate::Result<Self> {
        Ok(Self::F64(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `i64` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1_i64, 2];
    /// let view = TensorView::i64(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::I64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn i64(shape: &'a [usize], data: &'a [i64]) -> crate::Result<Self> {
        Ok(Self::I64(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `i32` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1_i32, 2];
    /// let view = TensorView::i32(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::I32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn i32(shape: &'a [usize], data: &'a [i32]) -> crate::Result<Self> {
        Ok(Self::I32(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `bool` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [true, false];
    /// let view = TensorView::bool(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::Bool);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn bool(shape: &'a [usize], data: &'a [bool]) -> crate::Result<Self> {
        Ok(Self::Bool(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `Complex32` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [Complex32::new(1.0, 2.0)];
    /// let view = TensorView::c32(&[1], &data)?;
    /// assert_eq!(view.dtype(), DType::C32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn c32(shape: &'a [usize], data: &'a [Complex32]) -> crate::Result<Self> {
        Ok(Self::C32(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `Complex64` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [Complex64::new(1.0, 2.0)];
    /// let view = TensorView::c64(&[1], &data)?;
    /// assert_eq!(view.dtype(), DType::C64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for compact
    /// shape or offset arithmetic overflow, or
    /// [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`] when the
    /// compact shape reaches beyond `data`.
    pub fn c64(shape: &'a [usize], data: &'a [Complex64]) -> crate::Result<Self> {
        Ok(Self::C64(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Return the element dtype of this borrowed view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let view = TensorView::f64(&[2], &[1.0, 2.0])?;
    /// assert_eq!(view.dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    /// Return the logical shape of this borrowed view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TensorView;
    ///
    /// let view = TensorView::i32(&[2, 1], &[1, 2])?;
    /// assert_eq!(view.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
            Self::C32(t) => t.shape(),
            Self::C64(t) => t.shape(),
        }
    }

    /// Borrow a contiguous host slice when the requested scalar matches this view's dtype.
    ///
    /// Backend buffers and non-contiguous views return an explicit error. No
    /// download or materialization is performed.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::DTypeMismatch`] when `T` does not match the
    /// view dtype, [`ValidationError::NonContiguousViewAsSlice`] for a
    /// non-contiguous layout, or [`crate::Error::HostAccess`] for unavailable
    /// backend host access.
    pub fn as_slice<T: TensorScalar>(&self) -> crate::Result<&'a [T]> {
        if self.dtype() != T::dtype() {
            return Err(crate::Error::validation(
                "TensorView::as_slice",
                ValidationError::DTypeMismatch {
                    expected: T::dtype(),
                    actual: self.dtype(),
                },
            ));
        }
        match self {
            Self::F32(view) => cast_view_slice(view.as_slice()?),
            Self::F64(view) => cast_view_slice(view.as_slice()?),
            Self::I32(view) => cast_view_slice(view.as_slice()?),
            Self::I64(view) => cast_view_slice(view.as_slice()?),
            Self::Bool(view) => cast_view_slice(view.as_slice()?),
            Self::C32(view) => cast_view_slice(view.as_slice()?),
            Self::C64(view) => cast_view_slice(view.as_slice()?),
        }
    }

    /// Reinterpret a complex view as its sealed real representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex64, DType, TensorView};
    ///
    /// let data = [Complex64::new(1.0, 2.0)];
    /// let real = TensorView::c64(&[1], &data)?.as_real_view()?;
    /// assert_eq!(real.dtype(), DType::F64);
    /// assert_eq!(real.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for the wrong dtype pair and
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] for invalid layout metadata.
    pub fn as_real_view(&self) -> crate::Result<Self> {
        match self {
            Self::C32(t) => t.as_real_view().map(Self::F32),
            Self::C64(t) => t.as_real_view().map(Self::F64),
            _ => Err(crate::Error::unsupported(
                "TensorView::as_real_view",
                "only complex views have a sealed real representation",
            )),
        }
    }

    /// Reinterpret a real view as its sealed complex representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Complex64, DType, TensorView};
    ///
    /// let data = [1.0_f64, 2.0];
    /// let complex = TensorView::f64(&[2, 1], &data)?.as_complex_view()?;
    /// assert_eq!(complex.dtype(), DType::C64);
    /// assert_eq!(complex.as_slice::<Complex64>()?, &[Complex64::new(1.0, 2.0)]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for the wrong dtype pair and
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] for invalid layout metadata.
    pub fn as_complex_view(&self) -> crate::Result<Self> {
        match self {
            Self::F32(t) => t.as_complex_view().map(Self::C32),
            Self::F64(t) => t.as_complex_view().map(Self::C64),
            _ => Err(crate::Error::unsupported(
                "TensorView::as_complex_view",
                "only real views have a sealed complex representation",
            )),
        }
    }

    /// Return the placement metadata carried by this borrowed view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryKind, TensorView};
    ///
    /// let view = TensorView::f64(&[1], &[1.0])?;
    /// assert_eq!(view.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        match self {
            Self::F32(t) => t.placement(),
            Self::F64(t) => t.placement(),
            Self::I32(t) => t.placement(),
            Self::I64(t) => t.placement(),
            Self::Bool(t) => t.placement(),
            Self::C32(t) => t.placement(),
            Self::C64(t) => t.placement(),
        }
    }

    /// Return the physical backend family, when this view is backend-owned.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TensorView;
    ///
    /// let view = TensorView::f64(&[1], &[1.0])?;
    /// assert_eq!(view.backend_family(), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn backend_family(&self) -> Option<&'static str> {
        match self {
            Self::F32(t) => t.backend_family(),
            Self::F64(t) => t.backend_family(),
            Self::I32(t) => t.backend_family(),
            Self::I64(t) => t.backend_family(),
            Self::Bool(t) => t.backend_family(),
            Self::C32(t) => t.backend_family(),
            Self::C64(t) => t.backend_family(),
        }
    }

    /// Return the shared allocation domain, when this view has one.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TensorView;
    ///
    /// let view = TensorView::f64(&[1], &[1.0])?;
    /// assert_eq!(view.allocation_domain(), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn allocation_domain(&self) -> Option<AllocationDomainId> {
        match self {
            Self::F32(t) => t.allocation_domain(),
            Self::F64(t) => t.allocation_domain(),
            Self::I32(t) => t.allocation_domain(),
            Self::I64(t) => t.allocation_domain(),
            Self::Bool(t) => t.allocation_domain(),
            Self::C32(t) => t.allocation_domain(),
            Self::C64(t) => t.allocation_domain(),
        }
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TensorView;
    ///
    /// let data = [0.0_f64; 6];
    /// let view = TensorView::f64(&[2, 3], &data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        match self {
            Self::F32(t) => t.strides(),
            Self::F64(t) => t.strides(),
            Self::I32(t) => t.strides(),
            Self::I64(t) => t.strides(),
            Self::Bool(t) => t.strides(),
            Self::C32(t) => t.strides(),
            Self::C64(t) => t.strides(),
        }
    }

    /// Return the physical element offset.
    pub fn offset(&self) -> isize {
        match self {
            Self::F32(t) => t.offset(),
            Self::F64(t) => t.offset(),
            Self::I32(t) => t.offset(),
            Self::I64(t) => t.offset(),
            Self::Bool(t) => t.offset(),
            Self::C32(t) => t.offset(),
            Self::C64(t) => t.offset(),
        }
    }

    /// Compute the physical element offset for a logical index.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::F32(t) => t.layout_linear_offset(indices),
            Self::F64(t) => t.layout_linear_offset(indices),
            Self::I32(t) => t.layout_linear_offset(indices),
            Self::I64(t) => t.layout_linear_offset(indices),
            Self::Bool(t) => t.layout_linear_offset(indices),
            Self::C32(t) => t.layout_linear_offset(indices),
            Self::C64(t) => t.layout_linear_offset(indices),
        }
    }

    /// Return whether this view is compact column-major.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::F32(t) => t.is_col_major_contiguous(),
            Self::F64(t) => t.is_col_major_contiguous(),
            Self::I32(t) => t.is_col_major_contiguous(),
            Self::I64(t) => t.is_col_major_contiguous(),
            Self::Bool(t) => t.is_col_major_contiguous(),
            Self::C32(t) => t.is_col_major_contiguous(),
            Self::C64(t) => t.is_col_major_contiguous(),
        }
    }

    /// Return a compact string summary of this view's layout metadata.
    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    /// Assert this view is compact column-major.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// view is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TensorView::assert_col_major_contiguous",
        )
    }

    /// Explicitly duplicate a compact host view into a fresh tensor.
    ///
    /// Backend views and non-contiguous layouts return a typed error; this
    /// operation never downloads or silently canonicalizes a view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TensorView, TypedTensorView};
    ///
    /// let data = [1_i32, 2];
    /// let view = TensorView::I32(TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?);
    /// let copy = view.duplicate()?;
    /// assert_eq!(copy.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::HostAccess`] for backend-owned views,
    /// [`ValidationError::NonContiguousViewAsSlice`] for non-contiguous views,
    /// or [`ValidationError::InvalidArgument`] for invalid layout metadata.
    pub fn duplicate(&self) -> crate::Result<Tensor> {
        fn duplicate_typed<T: TensorScalar>(
            view: &TypedTensorView<'_, T>,
        ) -> crate::Result<TypedTensor<T>> {
            let mut tensor = TypedTensor::<T>::from_vec_col_major(
                view.shape().to_vec(),
                view.as_slice()?.to_vec(),
            )?;
            tensor.set_placement(view.placement().clone());
            Ok(tensor)
        }

        match self {
            Self::F32(view) => duplicate_typed(view).map(Tensor::from_typed),
            Self::F64(view) => duplicate_typed(view).map(Tensor::from_typed),
            Self::I32(view) => duplicate_typed(view).map(Tensor::from_typed),
            Self::I64(view) => duplicate_typed(view).map(Tensor::from_typed),
            Self::Bool(view) => duplicate_typed(view).map(Tensor::from_typed),
            Self::C32(view) => duplicate_typed(view).map(Tensor::from_typed),
            Self::C64(view) => duplicate_typed(view).map(Tensor::from_typed),
        }
    }
}

macro_rules! tensor_view_mut_constructor {
    ($name:ident, $variant:ident, $scalar:ty) => {
        /// Create a dynamic mutable view over compact column-major host data.
        ///
        /// # Errors
        ///
        /// Returns [`crate::Error::Validation`] with
        /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
        /// compact layout arithmetic overflows, or
        /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
        /// requested shape exceeds `data`.
        pub fn $name(shape: &'a [usize], data: &'a mut [$scalar]) -> crate::Result<Self> {
            Ok(Self::$variant(TypedTensorViewMut::from_col_major(
                shape, data,
            )?))
        }
    };
}

impl<'a> TensorViewMut<'a> {
    tensor_view_mut_constructor!(f32, F32, f32);
    tensor_view_mut_constructor!(f64, F64, f64);
    tensor_view_mut_constructor!(i32, I32, i32);
    tensor_view_mut_constructor!(i64, I64, i64);
    tensor_view_mut_constructor!(bool, Bool, bool);
    tensor_view_mut_constructor!(c32, C32, Complex32);
    tensor_view_mut_constructor!(c64, C64, Complex64);

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
            Self::C32(t) => t.shape(),
            Self::C64(t) => t.shape(),
        }
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TensorViewMut;
    ///
    /// let mut data = [0.0_f64; 6];
    /// let view = TensorViewMut::f64(&[2, 3], &mut data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        match self {
            Self::F32(t) => t.strides(),
            Self::F64(t) => t.strides(),
            Self::I32(t) => t.strides(),
            Self::I64(t) => t.strides(),
            Self::Bool(t) => t.strides(),
            Self::C32(t) => t.strides(),
            Self::C64(t) => t.strides(),
        }
    }

    pub fn offset(&self) -> isize {
        match self {
            Self::F32(t) => t.offset(),
            Self::F64(t) => t.offset(),
            Self::I32(t) => t.offset(),
            Self::I64(t) => t.offset(),
            Self::Bool(t) => t.offset(),
            Self::C32(t) => t.offset(),
            Self::C64(t) => t.offset(),
        }
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::F32(t) => t.layout_linear_offset(indices),
            Self::F64(t) => t.layout_linear_offset(indices),
            Self::I32(t) => t.layout_linear_offset(indices),
            Self::I64(t) => t.layout_linear_offset(indices),
            Self::Bool(t) => t.layout_linear_offset(indices),
            Self::C32(t) => t.layout_linear_offset(indices),
            Self::C64(t) => t.layout_linear_offset(indices),
        }
    }

    /// Return whether this view is compact column-major.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::F32(t) => t.is_col_major_contiguous(),
            Self::F64(t) => t.is_col_major_contiguous(),
            Self::I32(t) => t.is_col_major_contiguous(),
            Self::I64(t) => t.is_col_major_contiguous(),
            Self::Bool(t) => t.is_col_major_contiguous(),
            Self::C32(t) => t.is_col_major_contiguous(),
            Self::C64(t) => t.is_col_major_contiguous(),
        }
    }

    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    /// Assert this view is compact column-major.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// view is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TensorViewMut::assert_col_major_contiguous",
        )
    }

    /// Explicitly duplicate the compact host data visible through this
    /// mutable view into a new tensor owner.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TensorViewMut, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TensorViewMut::I32(TypedTensorViewMut::from_slice(
    ///     vec![2], vec![1], 0, &mut data,
    /// )?);
    /// let copy = view.duplicate()?;
    /// assert_eq!(copy.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::HostAccess`] for backend-owned views,
    /// [`ValidationError::NonContiguousViewAsSlice`] for non-contiguous views,
    /// or [`ValidationError::InvalidArgument`] for invalid layout metadata.
    pub fn duplicate(&self) -> crate::Result<Tensor> {
        self.as_read_only().duplicate()
    }

    /// Borrow this mutable view as a read-only dtype-erased view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TensorViewMut;
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let view = TensorViewMut::f64(&[2], &mut data)?;
    /// assert_eq!(view.as_read_only().as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_read_only(&self) -> TensorView<'_> {
        match self {
            Self::F32(t) => TensorView::F32(t.as_read_only()),
            Self::F64(t) => TensorView::F64(t.as_read_only()),
            Self::I32(t) => TensorView::I32(t.as_read_only()),
            Self::I64(t) => TensorView::I64(t.as_read_only()),
            Self::Bool(t) => TensorView::Bool(t.as_read_only()),
            Self::C32(t) => TensorView::C32(t.as_read_only()),
            Self::C64(t) => TensorView::C64(t.as_read_only()),
        }
    }
}

impl<'a> TensorRead<'a> {
    /// Borrow an owned tensor as a read target without copying it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let read = TensorRead::from_tensor(&tensor);
    /// assert_eq!(read.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        Self::Tensor(tensor)
    }

    /// Wrap a borrowed dtype-erased view as a read target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TensorRead, TensorView};
    ///
    /// let read = TensorRead::from_view(TensorView::i32(&[2], &[3, 4])?);
    /// assert_eq!(read.as_slice::<i32>()?, &[3, 4]);
    /// assert!(read.as_tensor().is_none());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline]
    pub fn from_view(view: TensorView<'a>) -> Self {
        Self::View(view)
    }

    /// Convert this read target into a dtype-erased tensor view.
    ///
    /// Owned tensors are borrowed without copying their storage. Existing
    /// views preserve their layout and placement metadata.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_tensor::{Tensor, TensorRead};
    /// # let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let view = TensorRead::from_tensor(&tensor).tensor_view();
    /// assert_eq!(view.shape(), &[2]);
    /// assert_eq!(view.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn tensor_view(self) -> TensorView<'a> {
        match self {
            Self::Tensor(tensor) => tensor_view_with_layout(tensor, tensor_layout(tensor)),
            Self::View(view) => view,
        }
    }

    /// Build a metadata-only stride-0 broadcast of this read.
    ///
    /// An owned preset is borrowed and an existing view keeps its buffer,
    /// retained root and placement, so the result copies no elements and
    /// allocates no storage. A caller that can consume a read-only operand
    /// should use this instead of the materializing session
    /// `broadcast_in_dim_read`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    /// let read = TensorRead::from_tensor(&tensor).broadcast_in_dim_view(&[2, 3], &[1])?;
    /// assert_eq!(read.shape(), &[2, 3]);
    /// assert_eq!(read.strides()?, vec![0, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedDType`] for a caller-owned payload,
    /// which has no borrowed view, or a validation error when `dims` does not
    /// map this read's layout into `shape`.
    #[doc(hidden)]
    pub fn broadcast_in_dim_view(self, shape: &[usize], dims: &[usize]) -> crate::Result<Self> {
        let dtype = self.dtype();
        // INVARIANT: `tensor_view()` has no arm for a caller-owned payload, so
        // the refusal must happen before the read is converted into a view.
        if let DType::External(_) = dtype {
            return Err(crate::Error::unsupported_dtype(
                "TensorRead::broadcast_in_dim_view",
                dtype,
                "a caller-owned payload has no borrowed broadcast view",
            ));
        }
        let view = self.tensor_view();
        Ok(Self::View(match view {
            TensorView::F32(view) => TensorView::F32(broadcast_typed_view(view, shape, dims)?),
            TensorView::F64(view) => TensorView::F64(broadcast_typed_view(view, shape, dims)?),
            TensorView::I32(view) => TensorView::I32(broadcast_typed_view(view, shape, dims)?),
            TensorView::I64(view) => TensorView::I64(broadcast_typed_view(view, shape, dims)?),
            TensorView::Bool(view) => TensorView::Bool(broadcast_typed_view(view, shape, dims)?),
            TensorView::C32(view) => TensorView::C32(broadcast_typed_view(view, shape, dims)?),
            TensorView::C64(view) => TensorView::C64(broadcast_typed_view(view, shape, dims)?),
        }))
    }

    /// Borrow this read target as a typed compact host slice without allocation.
    ///
    /// This delegates to [`TensorView::as_slice`]. Cloning a `TensorRead` is a
    /// shallow clone of its borrowed reference or view metadata, so the returned
    /// slice retains the original `'a` storage lifetime and never outlives the
    /// storage borrowed by this read target. This method does not materialize,
    /// transfer, or otherwise canonicalize the input.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::DTypeMismatch`] when `T` does not match the
    /// input dtype, [`ValidationError::NonContiguousViewAsSlice`] for a
    /// noncompact view, or a typed runtime-state host-access error for
    /// backend-owned storage.
    /// Backend-owned
    /// inputs are never downloaded implicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let read = TensorRead::from_tensor(&tensor);
    /// assert_eq!(read.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_slice<T: TensorScalar>(&self) -> crate::Result<&'a [T]> {
        self.clone().tensor_view().as_slice()
    }

    /// Return the element dtype of this read target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert_eq!(TensorRead::from_tensor(&tensor).dtype(), DType::I64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    /// Return the logical shape of this read target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
    /// assert_eq!(TensorRead::from_tensor(&tensor).shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    /// Return the placement metadata carried by this read target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryKind, Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let read = TensorRead::from_tensor(&tensor);
    /// assert_eq!(read.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        match self {
            Self::Tensor(tensor) => tensor.placement(),
            Self::View(view) => view.placement(),
        }
    }

    /// Return the physical backend family of this read target, when backend-owned.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert_eq!(TensorRead::from_tensor(&tensor).backend_family(), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn backend_family(&self) -> Option<&'static str> {
        match self {
            Self::Tensor(tensor) => match tensor.dtype() {
                // A caller-owned payload is host memory without backend family.
                DType::External(_) => None,
                DType::F32 => tensor.as_typed::<f32>().and_then(|t| t.backend_family()),
                DType::F64 => tensor.as_typed::<f64>().and_then(|t| t.backend_family()),
                DType::I32 => tensor.as_typed::<i32>().and_then(|t| t.backend_family()),
                DType::I64 => tensor.as_typed::<i64>().and_then(|t| t.backend_family()),
                DType::Bool => tensor.as_typed::<bool>().and_then(|t| t.backend_family()),
                DType::C32 => tensor
                    .as_typed::<Complex<f32>>()
                    .and_then(|t| t.backend_family()),
                DType::C64 => tensor
                    .as_typed::<Complex<f64>>()
                    .and_then(|t| t.backend_family()),
            },
            Self::View(view) => view.backend_family(),
        }
    }

    /// Return the shared allocation domain of this read target, when present.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert_eq!(TensorRead::from_tensor(&tensor).allocation_domain(), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn allocation_domain(&self) -> Option<AllocationDomainId> {
        match self {
            Self::Tensor(tensor) => match tensor.dtype() {
                // A caller-owned payload has no allocation domain.
                DType::External(_) => None,
                DType::F32 => tensor.as_typed::<f32>().and_then(|t| t.allocation_domain()),
                DType::F64 => tensor.as_typed::<f64>().and_then(|t| t.allocation_domain()),
                DType::I32 => tensor.as_typed::<i32>().and_then(|t| t.allocation_domain()),
                DType::I64 => tensor.as_typed::<i64>().and_then(|t| t.allocation_domain()),
                DType::Bool => tensor
                    .as_typed::<bool>()
                    .and_then(|t| t.allocation_domain()),
                DType::C32 => tensor
                    .as_typed::<Complex<f32>>()
                    .and_then(|t| t.allocation_domain()),
                DType::C64 => tensor
                    .as_typed::<Complex<f64>>()
                    .and_then(|t| t.allocation_domain()),
            },
            Self::View(view) => view.allocation_domain(),
        }
    }

    /// Return strides in element units; owned tensors report compact column-major strides.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?;
    /// assert_eq!(TensorRead::from_tensor(&tensor).strides()?, vec![1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// column-major stride arithmetic overflows.
    pub fn strides(&self) -> crate::Result<Vec<isize>> {
        match self {
            Self::Tensor(tensor) => col_major_strides(tensor.shape()),
            Self::View(view) => Ok(view.strides().to_vec()),
        }
    }

    /// Return the physical element offset; owned tensors always start at `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensorView};
    ///
    /// let data = [1.0_f64, 2.0, 3.0];
    /// let view = TensorView::F64(TypedTensorView::from_slice(vec![2], vec![1], 1, &data)?);
    /// assert_eq!(TensorRead::from_view(view).offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        match self {
            Self::Tensor(_) => 0,
            Self::View(view) => view.offset(),
        }
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?;
    /// let read = TensorRead::from_tensor(&tensor);
    /// assert_eq!(read.layout_linear_offset(&[1, 2])?, 5);
    /// assert!(read.layout_linear_offset(&[2, 0]).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::Tensor(tensor) => tensor.layout_linear_offset(indices),
            Self::View(view) => view.layout_linear_offset(indices),
        }
    }

    /// Return whether this read target is compact column-major.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensorView};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(TensorRead::from_tensor(&tensor).is_col_major_contiguous()?);
    /// let data = [1.0_f64, 2.0, 3.0];
    /// let strided = TensorView::F64(TypedTensorView::from_slice(vec![2], vec![2], 0, &data)?);
    /// assert!(!TensorRead::from_view(strided).is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::Tensor(tensor) => tensor.is_col_major_contiguous(),
            Self::View(view) => view.is_col_major_contiguous(),
        }
    }

    /// Return a compact string summary of this read target's layout metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(
    ///     TensorRead::from_tensor(&tensor).layout_summary(),
    ///     "shape=[2] strides=[1] offset=0"
    /// );
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        let strides = match self.strides() {
            Ok(strides) => strides,
            Err(err) => return format!("layout unavailable: {err}"),
        };
        layout_summary(self.shape(), &strides, self.offset())
    }

    /// Assert this read target is compact column-major.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensorView};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// TensorRead::from_tensor(&tensor).assert_col_major_contiguous()?;
    /// let data = [1.0_f64, 2.0, 3.0];
    /// let strided = TensorView::F64(TypedTensorView::from_slice(vec![2], vec![2], 0, &data)?);
    /// assert!(TensorRead::from_view(strided).assert_col_major_contiguous().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// view is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        let strides = self.strides()?;
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            &strides,
            self.offset(),
            "TensorRead::assert_col_major_contiguous",
        )
    }

    /// Return the borrowed owned tensor, or `None` when this read target is a view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorRead, TensorView};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert!(TensorRead::from_tensor(&tensor).as_tensor().is_some());
    /// assert!(TensorRead::from_view(TensorView::f64(&[1], &[1.0])?).as_tensor().is_none());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_tensor(&self) -> Option<&'a Tensor> {
        match self {
            Self::Tensor(tensor) => Some(*tensor),
            Self::View(_) => None,
        }
    }
}

impl<'a> TensorWrite<'a> {
    /// Borrow an owned tensor as a writable target without copying it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let write = TensorWrite::from_tensor(&mut tensor);
    /// assert_eq!(write.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_tensor(tensor: &'a mut Tensor) -> Self {
        Self::Tensor(tensor)
    }

    /// Wrap a borrowed mutable dtype-erased view as a writable target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorViewMut, TensorWrite};
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let write = TensorWrite::from_view(TensorViewMut::f64(&[2], &mut data)?);
    /// assert_eq!(write.dtype(), DType::F64);
    /// assert_eq!(write.as_read().as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_view(view: TensorViewMut<'a>) -> Self {
        Self::View(view)
    }

    /// Borrow this writable target as a read-only tensor input.
    ///
    /// This is useful for explicit read-modify-write kernels such as
    /// accumulation updates. The returned view borrows through `&self`, so it
    /// cannot outlive the current read-only borrow of the writable target.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let write = TensorWrite::from_tensor(&mut tensor);
    /// let read = write.as_read();
    /// assert_eq!(read.dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_read(&self) -> TensorRead<'_> {
        match self {
            Self::Tensor(tensor) => TensorRead::from_tensor(tensor),
            Self::View(view) => TensorRead::from_view(view.as_read_only()),
        }
    }

    /// Return the element dtype of this writable target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![1_i32])?;
    /// assert_eq!(TensorWrite::from_tensor(&mut tensor).dtype(), DType::I32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    /// Return the logical shape of this writable target.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
    /// assert_eq!(TensorWrite::from_tensor(&mut tensor).shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    /// Return strides in element units; owned tensors report compact column-major strides.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?;
    /// assert_eq!(TensorWrite::from_tensor(&mut tensor).strides()?, vec![1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// column-major stride arithmetic overflows.
    pub fn strides(&self) -> crate::Result<Vec<isize>> {
        match self {
            Self::Tensor(tensor) => col_major_strides(tensor.shape()),
            Self::View(view) => Ok(view.strides().to_vec()),
        }
    }

    /// Return the physical element offset; owned tensors always start at `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TensorViewMut, TensorWrite, TypedTensorViewMut};
    ///
    /// let mut data = [1.0_f64, 2.0, 3.0];
    /// let view = TensorViewMut::F64(TypedTensorViewMut::from_slice(vec![2], vec![1], 1, &mut data)?);
    /// assert_eq!(TensorWrite::from_view(view).offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        match self {
            Self::Tensor(_) => 0,
            Self::View(view) => view.offset(),
        }
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?;
    /// let write = TensorWrite::from_tensor(&mut tensor);
    /// assert_eq!(write.layout_linear_offset(&[1, 2])?, 5);
    /// assert!(write.layout_linear_offset(&[0]).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::Tensor(tensor) => tensor.layout_linear_offset(indices),
            Self::View(view) => view.layout_linear_offset(indices),
        }
    }

    /// Return whether this writable target is compact column-major.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorViewMut, TensorWrite, TypedTensorViewMut};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(TensorWrite::from_tensor(&mut tensor).is_col_major_contiguous()?);
    /// let mut data = [1.0_f64, 2.0, 3.0];
    /// let strided = TensorViewMut::F64(TypedTensorViewMut::from_slice(vec![2], vec![2], 0, &mut data)?);
    /// assert!(!TensorWrite::from_view(strided).is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::Tensor(tensor) => tensor.is_col_major_contiguous(),
            Self::View(view) => view.is_col_major_contiguous(),
        }
    }

    /// Return a compact string summary of this writable target's layout metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(
    ///     TensorWrite::from_tensor(&mut tensor).layout_summary(),
    ///     "shape=[2] strides=[1] offset=0"
    /// );
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        let strides = match self.strides() {
            Ok(strides) => strides,
            Err(err) => return format!("layout unavailable: {err}"),
        };
        layout_summary(self.shape(), &strides, self.offset())
    }

    /// Assert this writable target is compact column-major.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorViewMut, TensorWrite, TypedTensorViewMut};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// TensorWrite::from_tensor(&mut tensor).assert_col_major_contiguous()?;
    /// let mut data = [1.0_f64, 2.0, 3.0];
    /// let strided = TensorViewMut::F64(TypedTensorViewMut::from_slice(vec![2], vec![2], 0, &mut data)?);
    /// assert!(TensorWrite::from_view(strided).assert_col_major_contiguous().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// view is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        let strides = self.strides()?;
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            &strides,
            self.offset(),
            "TensorWrite::assert_col_major_contiguous",
        )
    }
}

/// Column-major strides derived from a shape.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::col_major_strides;
///
/// assert_eq!(col_major_strides(&[2, 3])?, vec![1, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
/// # Errors
///
/// Returns [`crate::Error::Validation`] with
/// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when a
/// column-major stride product overflows.
pub fn col_major_strides(shape: &[usize]) -> crate::Result<Vec<isize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        let extent = isize::try_from(extent).map_err(|_| {
            crate::Error::validation("col_major_strides", ValidationError::IntegerOverflow)
        })?;
        stride = stride.checked_mul(extent).ok_or_else(|| {
            crate::Error::validation("col_major_strides", ValidationError::IntegerOverflow)
        })?;
    }
    Ok(strides)
}

fn try_linear_offset_for_shape(
    shape: &[usize],
    indices: &[usize],
    op: &'static str,
) -> crate::Result<usize> {
    if indices.len() != shape.len() {
        return Err(crate::Error::validation(
            op,
            ValidationError::RankMismatch {
                expected: shape.len(),
                actual: indices.len(),
            },
        ));
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, (&idx, &extent)) in indices.iter().zip(shape).enumerate() {
        if idx >= extent {
            return Err(crate::Error::invalid_argument(
                op,
                "index",
                format!("index {idx} out of bounds for axis {axis} extent {extent}"),
            ));
        }
        offset =
            offset
                .checked_add(idx.checked_mul(stride).ok_or_else(|| {
                    crate::Error::validation(op, ValidationError::IntegerOverflow)
                })?)
                .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    }
    Ok(offset)
}

fn checked_view_offset_result(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    indices: &[usize],
    op: &'static str,
) -> crate::Result<usize> {
    if indices.len() != shape.len() {
        return Err(crate::Error::validation(
            op,
            ValidationError::RankMismatch {
                expected: shape.len(),
                actual: indices.len(),
            },
        ));
    }
    for (axis, (&index, &extent)) in indices.iter().zip(shape).enumerate() {
        if index >= extent {
            return Err(crate::Error::invalid_argument(
                op,
                "index",
                format!("index {index} out of bounds for axis {axis} extent {extent}"),
            ));
        }
    }
    checked_view_offset(shape, strides, base_offset, indices)
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))
}

fn layout_summary(shape: &[usize], strides: &[isize], offset: isize) -> String {
    format!("shape={shape:?} strides={strides:?} offset={offset}")
}

fn assert_layout_col_major_contiguous(
    is_contiguous: bool,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    op: &'static str,
) -> crate::Result<()> {
    if is_contiguous {
        Ok(())
    } else {
        Err(crate::Error::invalid_argument(
            op,
            "layout",
            format!(
                "expected compact column-major layout, got {}",
                layout_summary(shape, strides, offset)
            ),
        ))
    }
}

fn try_shape_product(shape: &[usize], op: &'static str) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))
    })
}

fn try_checked_shape_len(shape: &[usize], data_len: usize, op: &'static str) -> crate::Result<()> {
    let n = try_shape_product(shape, op)?;
    if data_len != n {
        return Err(crate::Error::validation(
            op,
            ValidationError::ShapeDataLengthMismatch {
                expected: n,
                actual: data_len,
            },
        ));
    }
    Ok(())
}

fn try_compact_layout<R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    op: &'static str,
) -> crate::Result<TensorLayout<R>> {
    let shape = shape
        .into_rank_shape()
        .map_err(|err| tensor_layout_error(op, err))?;
    TensorLayout::compact(shape).map_err(|err| tensor_layout_error(op, err))
}

fn tensor_layout_error(
    op: &'static str,
    err: tenferro_tensor_core::ValidationError,
) -> crate::Error {
    crate::Error::validation(op, err)
}

fn checked_view_element_count(shape: &[usize], op: &'static str) -> crate::Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))
    })
}

fn checked_view_offset(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    indices: &[usize],
) -> Option<usize> {
    if indices.len() != shape.len() {
        return None;
    }

    let mut offset = base_offset;
    for ((&index, &extent), &stride) in indices.iter().zip(shape).zip(strides) {
        if index >= extent {
            return None;
        }
        let index = isize::try_from(index).ok()?;
        let delta = index.checked_mul(stride)?;
        offset = offset.checked_add(delta)?;
    }

    usize::try_from(offset).ok()
}

fn reachable_layout_span(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> crate::Result<Option<(usize, usize)>> {
    if shape.contains(&0) {
        return Ok(None);
    }

    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let steps = isize::try_from(extent.saturating_sub(1)).map_err(|_| {
            crate::Error::validation(
                "TypedTensorViewMut::try_multi_slice_mut",
                ValidationError::IntegerOverflow,
            )
        })?;
        let end = stride.checked_mul(steps).ok_or_else(|| {
            crate::Error::validation(
                "TypedTensorViewMut::try_multi_slice_mut",
                ValidationError::IntegerOverflow,
            )
        })?;
        let (axis_min, axis_max) = if end < 0 { (end, 0) } else { (0, end) };
        min_offset = min_offset.checked_add(axis_min).ok_or_else(|| {
            crate::Error::validation(
                "TypedTensorViewMut::try_multi_slice_mut",
                ValidationError::IntegerOverflow,
            )
        })?;
        max_offset = max_offset.checked_add(axis_max).ok_or_else(|| {
            crate::Error::validation(
                "TypedTensorViewMut::try_multi_slice_mut",
                ValidationError::IntegerOverflow,
            )
        })?;
    }

    let min_offset = usize::try_from(min_offset).map_err(|_| {
        crate::Error::invalid_argument(
            "TypedTensorViewMut::try_multi_slice_mut",
            "layout",
            "minimum reachable offset is negative",
        )
    })?;
    let max_offset = usize::try_from(max_offset).map_err(|_| {
        crate::Error::invalid_argument(
            "TypedTensorViewMut::try_multi_slice_mut",
            "layout",
            "maximum reachable offset is negative",
        )
    })?;
    Ok(Some((min_offset, max_offset)))
}

fn split_two_mut_ranges<T>(
    data: &mut [T],
    first: (usize, usize),
    second: (usize, usize),
) -> Option<(&mut [T], &mut [T])> {
    if first.1 < second.0 {
        let (_, after_first_start) = data.split_at_mut(first.0);
        let (first_slice, after_first) = after_first_start.split_at_mut(first.1 - first.0 + 1);
        let (_, after_gap) = after_first.split_at_mut(second.0 - first.1 - 1);
        let (second_slice, _) = after_gap.split_at_mut(second.1 - second.0 + 1);
        Some((first_slice, second_slice))
    } else if second.1 < first.0 {
        let (_, after_second_start) = data.split_at_mut(second.0);
        let (second_slice, after_second) = after_second_start.split_at_mut(second.1 - second.0 + 1);
        let (_, after_gap) = after_second.split_at_mut(first.0 - second.1 - 1);
        let (first_slice, _) = after_gap.split_at_mut(first.1 - first.0 + 1);
        Some((first_slice, second_slice))
    } else {
        None
    }
}

fn adjusted_view_offset(offset: isize, span_start: usize) -> crate::Result<isize> {
    let span_start = isize::try_from(span_start).map_err(|_| {
        crate::Error::validation(
            "TypedTensorViewMut::try_multi_slice_mut",
            ValidationError::IntegerOverflow,
        )
    })?;
    offset.checked_sub(span_start).ok_or_else(|| {
        crate::Error::validation(
            "TypedTensorViewMut::try_multi_slice_mut",
            ValidationError::IntegerOverflow,
        )
    })
}

fn view_mut_from_layout_and_slice<'a, T: 'static, R: TensorRank>(
    layout: &TensorLayout<R>,
    offset: isize,
    data: &'a mut [T],
    placement: Placement,
) -> crate::Result<TypedTensorViewMut<'a, T, R>> {
    let shape = R::shape_from_vec(shape_vec(layout.shape()))
        .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_multi_slice_mut", err))?;
    let strides = R::strides_from_vec(stride_vec(layout.strides()))
        .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_multi_slice_mut", err))?;
    TypedTensorViewMut::from_buffer_ref_mut(
        shape,
        strides,
        offset,
        TensorStorageRefMut::Host(data),
        placement,
        "TypedTensorViewMut::try_multi_slice_mut",
    )
}

fn contiguous_layout_slice<'a, T, R: TensorRank>(
    layout: &TensorLayout<R>,
    data: &'a [T],
    op: &'static str,
) -> crate::Result<&'a [T]> {
    if !layout
        .is_compact_col_major()
        .map_err(|err| tensor_layout_error(op, err))?
    {
        return Err(crate::Error::validation(
            op,
            ValidationError::NonContiguousViewAsSlice,
        ));
    }
    let len = checked_view_element_count(layout.shape(), op)?;
    let start = usize::try_from(layout.offset())
        .map_err(|_| crate::Error::invalid_argument(op, "layout", "view offset is negative"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    data.get(start..end)
        .ok_or_else(|| crate::Error::validation(op, ValidationError::ViewOutOfBounds))
}

fn relaxed_col_major_contiguous(
    shape: &[usize],
    strides: &[isize],
    op: &'static str,
) -> crate::Result<bool> {
    if shape.contains(&0) {
        return Ok(true);
    }

    let mut expected = 1isize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent <= 1 {
            continue;
        }
        if stride != expected {
            return Ok(false);
        }
        let extent = isize::try_from(extent)
            .map_err(|_| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
        expected = expected
            .checked_mul(extent)
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    }
    Ok(true)
}

fn reshape_layout_dyn<R: TensorRank>(
    layout: &TensorLayout<R>,
    shape: &[usize],
    buffer_len: usize,
    op: &'static str,
) -> crate::Result<TensorLayout<DynRank>> {
    match layout.reshape_view_as::<DynRank>(shape_vec(shape), buffer_len) {
        Ok(layout) => Ok(layout),
        Err(err) => {
            if !relaxed_col_major_contiguous(layout.shape(), layout.strides(), op)? {
                return Err(tensor_layout_error(op, err));
            }
            let from = checked_view_element_count(layout.shape(), op)?;
            let to = checked_view_element_count(shape, op)?;
            if from != to {
                return Err(tensor_layout_error(
                    op,
                    tenferro_tensor_core::ShapeMismatch::ReshapeElementCount { from, to }.into(),
                ));
            }
            TensorLayout::<DynRank>::compact(shape_vec(shape))
                .and_then(|compact| {
                    TensorLayout::from_parts(
                        shape_vec(compact.shape()),
                        stride_vec(compact.strides()),
                        layout.offset(),
                        buffer_len,
                    )
                })
                .map_err(|err| tensor_layout_error(op, err))
        }
    }
}

fn core_slice_specs(
    slices: &[StridedSliceSpec],
    shape: &[usize],
    op: &'static str,
) -> crate::Result<SmallVec<[CoreSliceSpec; 8]>> {
    if slices.len() != shape.len() {
        return Err(crate::Error::validation(
            op,
            ValidationError::RankMismatch {
                expected: shape.len(),
                actual: slices.len(),
            },
        ));
    }

    let mut specs = SmallVec::<[CoreSliceSpec; 8]>::with_capacity(slices.len());
    for (slice, &axis_len) in slices.iter().zip(shape) {
        specs.push(core_slice_spec(*slice, axis_len, op)?);
    }
    Ok(specs)
}

fn core_slice_spec(
    slice: StridedSliceSpec,
    axis_len: usize,
    op: &'static str,
) -> crate::Result<CoreSliceSpec> {
    if slice.step() == 0 {
        return Err(crate::Error::validation(
            op,
            ValidationError::InvalidSliceStep { step: slice.step() },
        ));
    }

    let start = normalize_strided_bound(slice.start(), axis_len, op, "slice start")?;
    let end = match slice.end() {
        Some(end) => normalize_strided_bound(end, axis_len, op, "slice end")?,
        None => isize::try_from(axis_len)
            .map_err(|_| crate::Error::validation(op, ValidationError::IntegerOverflow))?,
    };

    if slice.step() > 0 {
        return Ok(CoreSliceSpec {
            start,
            end,
            step: slice.step(),
        });
    }

    if start >= end {
        // `core`'s negative-step contract requires the inclusive `start` to
        // point at an element (`start < axis_len`). An empty selection is
        // step-sign independent, so encode the `start == axis_len` boundary at
        // the last element instead of re-encoding it verbatim (which `core`
        // would reject as out of bounds).
        let empty_start = if axis_len == 0 {
            start
        } else {
            start.min(axis_len as isize - 1)
        };
        return Ok(CoreSliceSpec {
            start: empty_start,
            end: empty_start,
            step: slice.step(),
        });
    }

    Ok(CoreSliceSpec {
        start: end
            .checked_sub(1)
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?,
        end: start
            .checked_sub(1)
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?,
        step: slice.step(),
    })
}

fn normalize_strided_bound(
    bound: isize,
    axis_len: usize,
    op: &'static str,
    role: &'static str,
) -> crate::Result<isize> {
    let original_axis_len = axis_len;
    let axis_len = isize::try_from(axis_len)
        .map_err(|_| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
    let bound = if bound < 0 {
        axis_len
            .checked_add(bound)
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?
    } else {
        bound
    };
    if !(0..=axis_len).contains(&bound) {
        let (start, end) = if role == "slice start" {
            (bound, bound)
        } else {
            (0, bound)
        };
        return Err(crate::Error::validation(
            op,
            ValidationError::InvalidSliceBounds {
                start,
                end,
                axis_len: original_axis_len,
            },
        ));
    }
    Ok(bound)
}

fn slice_axis_specs(
    rank: usize,
    axis: usize,
    slice: StridedSliceSpec,
    op: &'static str,
) -> crate::Result<Vec<StridedSliceSpec>> {
    if axis >= rank {
        return Err(crate::Error::validation(
            op,
            ValidationError::AxisOutOfBounds { axis, rank },
        ));
    }

    let mut slices = vec![StridedSliceSpec::all(); rank];
    slices[axis] = slice;
    Ok(slices)
}

pub(crate) fn default_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: None,
    }
}

fn typed_tensor_from_vec_col_major<T, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    data: Vec<T>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_from_vec_col_major(shape, data, op)
}

fn try_typed_tensor_from_vec_col_major<T, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    data: Vec<T>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R>> {
    let shape = shape
        .into_rank_shape()
        .map_err(|err| tensor_layout_error(op, err))?;
    let layout = tenferro_tensor_core::col_major_strides(shape.as_ref())
        .map_err(|err| tensor_layout_error(op, err))?;
    try_checked_shape_len(shape.as_ref(), data.len(), op)?;
    Ok(TypedTensor {
        shape,
        placement: default_placement(),
        layout,
        storage: DynamicStorage::Host(HostStorage {
            data: HostData::new(data),
        }),
    })
}

fn typed_tensor_zeros<T: TensorScalar + Zero, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_zeros(shape)
}

fn try_typed_tensor_zeros<T: TensorScalar + Clone + Zero, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, "zeros")?;
    let n = try_shape_product(layout.shape(), "zeros")?;
    typed_tensor_from_vec_col_major(
        R::shape_from_vec(shape_vec(layout.shape()))
            .map_err(|err| tensor_layout_error("zeros", err))?,
        vec![T::zero(); n],
        "zeros",
    )
}

fn typed_tensor_ones<T: TensorScalar + One + Zero, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_ones(shape)
}

fn try_typed_tensor_ones<T: TensorScalar + Clone + One + Zero, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, "ones")?;
    let n = try_shape_product(layout.shape(), "ones")?;
    typed_tensor_from_vec_col_major(
        R::shape_from_vec(shape_vec(layout.shape()))
            .map_err(|err| tensor_layout_error("ones", err))?,
        vec![T::one(); n],
        "ones",
    )
}

fn typed_tensor_from_buffer_col_major<T: TensorScalar + Send + Sync + 'static, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    buffer: StorageBuffer<T>,
    placement: Placement,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_from_buffer_col_major(shape, buffer, placement)
}

#[doc(hidden)]
fn typed_tensor_from_backend_allocation<T: TensorScalar + Send + Sync + 'static, R: TensorRank>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    allocation: Box<dyn crate::BackendAllocation>,
    placement: Placement,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, "from_backend_allocation")?;
    let group_shape = R::shape_from_vec(shape_vec(layout.shape()))
        .map_err(|err| tensor_layout_error("from_backend_allocation", err))?;
    let shape = group_shape.clone();
    let (mut group, slot) =
        AllocationGroup::from_backend_allocation::<T, R>(group_shape, allocation)
            .map_err(|error| group_error("TypedTensor::from_backend_allocation", error))?;
    group
        .set_descriptor_placement(slot, placement.clone())
        .map_err(|error| group_error("TypedTensor::from_backend_allocation", error))?;
    let allocation_index = group
        .allocation_index(slot)
        .map_err(|error| group_error("TypedTensor::from_backend_allocation", error))?;
    let (host_ptr, host_byte_len) = host_metadata::<T>(&group, slot);
    Ok(TypedTensor {
        shape,
        placement,
        layout: stride_vec(layout.strides()),
        storage: DynamicStorage::Group(GroupStorage {
            group: Box::new(OwnedTensorGroup {
                group,
                slot,
                allocation_index,
                host_ptr,
                host_byte_len,
                _rank: PhantomData,
            }),
        }),
    })
}

fn try_typed_tensor_from_buffer_col_major<
    T: TensorScalar + Send + Sync + 'static,
    R: TensorRank,
>(
    shape: impl tenferro_tensor_core::IntoRankShape<R>,
    buffer: StorageBuffer<T>,
    placement: Placement,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, "from_buffer_col_major")?;
    let len = buffer.len();
    try_checked_shape_len(layout.shape(), len, "from_buffer_col_major")?;
    let group_shape = R::shape_from_vec(shape_vec(layout.shape()))
        .map_err(|err| tensor_layout_error("from_buffer_col_major", err))?;
    match buffer {
        StorageBuffer::Host(data) => Ok(TypedTensor {
            shape: group_shape,
            placement,
            layout: stride_vec(layout.strides()),
            storage: DynamicStorage::Host(HostStorage {
                data: HostData::new(data),
            }),
        }),
        StorageBuffer::Backend(buffer) => {
            let group = OwnedTensorGroup::from_backend_buffer(
                group_shape.clone(),
                StorageBuffer::Backend(buffer),
                placement.clone(),
            )?;
            Ok(TypedTensor {
                shape: group_shape,
                placement,
                layout: stride_vec(layout.strides()),
                storage: DynamicStorage::Group(GroupStorage {
                    group: Box::new(group),
                }),
            })
        }
    }
}

impl<T: TensorScalar + Zero, R: TensorRank> TypedTensor<T, R> {
    /// Allocate a zero-filled tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]).unwrap();
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when shape
    /// product or compact-stride arithmetic overflows.
    /// A vector or slice with a different static rank returns
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`].
    pub fn zeros(shape: impl tenferro_tensor_core::IntoRankShape<R>) -> crate::Result<Self> {
        typed_tensor_zeros(shape)
    }
}

impl<T: TensorScalar + One + Zero, R: TensorRank> TypedTensor<T, R> {
    /// Allocate a one-filled tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::ones(vec![2]).unwrap();
    /// assert_eq!(t.host_data().unwrap(), &[1.0, 1.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when shape
    /// product or compact-stride arithmetic overflows.
    /// A vector or slice with a different static rank returns
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`].
    pub fn ones(shape: impl tenferro_tensor_core::IntoRankShape<R>) -> crate::Result<Self> {
        typed_tensor_ones(shape)
    }
}

impl<T, R: TensorRank> TypedTensor<T, R> {
    /// Adopt a column-major host `Vec<T>` with no scalar, copy or thread-safety bound.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// struct Custom(String);
    /// let tensor = TypedTensor::<Custom, Rank<2>>::from_vec_col_major(
    ///     [1, 2], vec![Custom("a".into()), Custom("b".into())],
    /// )?;
    /// assert_eq!(tensor.get(&[0, 1])?.0.as_str(), "b");
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`crate::Error::Validation`] for a rank mismatch, shape/data
    /// length mismatch or shape/stride arithmetic overflow.
    pub fn from_vec_col_major(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        data: Vec<T>,
    ) -> crate::Result<Self> {
        typed_tensor_from_vec_col_major(shape, data, "from_vec_col_major")
    }

    /// Explicitly import row-major host values into column-major storage.
    ///
    /// Clones each input element once; no backend or scalar registration is used.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<2>>::from_vec_row_major(
    ///     [2, 3], vec![1, 2, 3, 4, 5, 6],
    /// )?;
    /// assert_eq!(tensor.get(&[1, 0])?, &4);
    /// assert_eq!(tensor.get(&[0, 2])?, &3);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Returns [`crate::Error::Validation`] for a rank, shape-length or stride overflow,
    /// or when the shape product disagrees with the input length.
    pub fn from_vec_row_major(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        data: Vec<T>,
    ) -> crate::Result<Self>
    where
        T: Clone,
    {
        let op = "from_vec_row_major";
        let (shape, reordered) = row_major_reorder(shape, data, op)?;
        typed_tensor_from_vec_col_major(shape, reordered, op)
    }

    /// Consume this compact tensor and return the original host `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2])?;
    /// let data = tensor.into_host_vec().map_err(|failure| failure.into_parts().1)?;
    /// assert_eq!(data, vec![1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when the
    /// storage is device-only or the managed root cannot export a host vector.
    pub fn into_host_vec(self) -> std::result::Result<Vec<T>, ReinterpretError<Self>>
    where
        T: 'static,
    {
        let TypedTensor {
            shape,
            placement,
            layout,
            storage,
        } = self;
        match storage {
            DynamicStorage::Host(HostStorage { data }) => Ok(data.into_vec()),
            DynamicStorage::Group(GroupStorage { group }) => match group.into_host_vec::<T>() {
                Ok(data) => Ok(data),
                Err((group, error)) => Err(ReinterpretError::new(
                    TypedTensor {
                        shape,
                        placement,
                        layout,
                        storage: DynamicStorage::Group(GroupStorage {
                            group: Box::new(group),
                        }),
                    },
                    error,
                )),
            },
        }
    }

    /// Consume the original host vector along with its column-major shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<i32>::from_vec_col_major(vec![2, 1], vec![1, 2])?;
    /// let (shape, data) = tensor.into_vec_col_major().map_err(|failure| failure.into_parts().1)?;
    /// assert_eq!(shape, vec![2, 1]);
    /// assert_eq!(data, vec![1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when the
    /// storage is device-only or the managed root cannot export a host vector.
    pub fn into_vec_col_major(
        self,
    ) -> std::result::Result<(Vec<usize>, Vec<T>), ReinterpretError<Self>>
    where
        T: 'static,
    {
        let shape = self.shape().to_vec();
        match self.into_host_vec() {
            Ok(data) => Ok((shape, data)),
            Err(failure) => Err(failure),
        }
    }

    /// Borrow the plain host values (or an existing managed host root).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// assert_eq!(tensor.host_data()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::RuntimeState`] for device-only storage.
    pub fn host_data(&self) -> crate::Result<&[T]> {
        match &self.storage {
            DynamicStorage::Host(HostStorage { data }) => Ok(data.as_slice()),
            DynamicStorage::Group(core) => core.group.host_slice::<T>(),
        }
    }

    /// Borrow compact host storage as a flat column-major slice without copying.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<String>::from_vec_col_major([2], vec!["a".into(), "b".into()])?;
    /// assert_eq!(tensor.as_slice()?, &["a", "b"]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Device-only storage returns [`crate::Error::RuntimeState`].
    pub fn as_slice(&self) -> crate::Result<&[T]> {
        self.host_data()
    }

    /// Mutably borrow the plain host values (or an existing managed host root).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// tensor.host_data_mut()?[0] = 5.0;
    /// assert_eq!(tensor.host_data()?, &[5.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`crate::Error::RuntimeState`] for device-only storage.
    pub fn host_data_mut(&mut self) -> crate::Result<&mut [T]> {
        match &mut self.storage {
            DynamicStorage::Host(HostStorage { data }) => Ok(data.as_mut_slice()),
            DynamicStorage::Group(core) => core.group.host_slice_mut::<T>(),
        }
    }

    /// Borrow an element by checked column-major multi-index.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<String>::from_vec_col_major([2], vec!["a".into(), "b".into()])?;
    /// assert_eq!(tensor.get(&[1])?, "b");
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Invalid rank, coordinates or offset return [`crate::Error::Validation`];
    /// device-only storage returns [`crate::Error::RuntimeState`].
    pub fn get(&self, indices: &[usize]) -> crate::Result<&T> {
        let offset = self.linear_offset(indices)?;
        self.host_data()?.get(offset).ok_or_else(|| {
            crate::Error::validation("TypedTensor::get", ValidationError::ViewOutOfBounds)
        })
    }

    /// Exclusively borrow an element by checked column-major multi-index.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let mut tensor = TypedTensor::<String>::from_vec_col_major([1], vec!["a".into()])?;
    /// *tensor.get_mut(&[0])? = "b".into();
    /// assert_eq!(tensor.get(&[0])?, "b");
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Invalid rank, coordinates or offset return [`crate::Error::Validation`];
    /// device-only storage returns [`crate::Error::RuntimeState`].
    pub fn get_mut(&mut self, indices: &[usize]) -> crate::Result<&mut T> {
        let offset = self.linear_offset(indices)?;
        self.host_data_mut()?.get_mut(offset).ok_or_else(|| {
            crate::Error::validation("TypedTensor::get_mut", ValidationError::ViewOutOfBounds)
        })
    }

    /// Make an explicit independent host copy with the same shape and placement.
    ///
    /// The dynamic `TypedTensor` does not implement [`Clone`] (only the
    /// host-only representation does): the copy reads host data, rejects
    /// device-only storage, and creates a new owner, so it is fallible. See
    /// [`Tensor::duplicate`] for the sharing alternatives.
    ///
    /// # Examples
    /// ```
    /// use tenferro_tensor::TypedTensor;
    /// let tensor = TypedTensor::<String>::from_vec_col_major([1], vec!["a".into()])?;
    /// assert_eq!(tensor.duplicate()?.get(&[0])?, "a");
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    /// Device-only storage returns [`crate::Error::RuntimeState`]; invalid
    /// host metadata returns [`crate::Error::Validation`].
    pub fn duplicate(&self) -> crate::Result<Self>
    where
        T: Clone,
    {
        let mut copy = Self::from_vec_col_major(
            R::shape_from_vec(shape_vec(self.shape()))
                .map_err(|err| tensor_layout_error("TypedTensor::duplicate", err))?,
            self.host_data()?.to_vec(),
        )?;
        copy.set_placement(self.placement().clone());
        Ok(copy)
    }

    /// Create a tensor from an existing buffer and compact column-major layout.
    ///
    /// This preserves the owned tensor invariant that layout metadata is
    /// compact column-major, including for backend-owned buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StorageBuffer, Placement, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64>::from_buffer_col_major(
    ///     vec![2],
    ///     StorageBuffer::Host(vec![1.0, 2.0]),
    ///     Placement {
    ///         memory_kind: tenferro_tensor::MemoryKind::UnpinnedHost,
    ///         device: None,
    ///         cpu_affinity: None,
    ///     },
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.shape(), &[2]);
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::ShapeDataLengthMismatch`] when
    /// the shape product differs from the buffer length,
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when shape or
    /// stride arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when a supplied
    /// rank-specific shape cannot be represented.
    pub fn from_buffer_col_major(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        buffer: StorageBuffer<T>,
        placement: Placement,
    ) -> crate::Result<Self>
    where
        T: TensorScalar + Send + Sync + 'static,
    {
        typed_tensor_from_buffer_col_major(shape, buffer, placement)
    }

    /// Consume a scalar-independent provider root into one compact tensor.
    #[doc(hidden)]
    pub fn from_backend_allocation(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        allocation: Box<dyn crate::BackendAllocation>,
        placement: Placement,
    ) -> crate::Result<Self>
    where
        T: TensorScalar + Send + Sync + 'static,
    {
        typed_tensor_from_backend_allocation(shape, allocation, placement)
    }

    /// Convert this tensor into static rank metadata after validating its rank.
    ///
    /// The buffer and placement are preserved. This method changes only the
    /// compile-time rank marker on the owned compact column-major tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Rank, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let Ok(ranked) = tensor.try_into_rank::<2>() else {
    ///     panic!("a rank-2 tensor converts to two axes")
    /// };
    /// assert_eq!(ranked.shape(), &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when the
    /// typed rank does not match the existing shape, with
    /// [`crate::Error::Validation`] and
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] as the cause,
    /// or when the compact rank layout overflows.
    pub fn try_into_rank<const N: usize>(
        self,
    ) -> std::result::Result<TypedTensor<T, Rank<N>>, ReinterpretError<Self>> {
        let op = "TypedTensor::try_into_rank";
        let actual = self.shape().len();
        let shape: [usize; N] = match self.shape().try_into() {
            Ok(shape) => shape,
            Err(_) => {
                return Err(ReinterpretError::new(
                    self,
                    tensor_layout_error(
                        op,
                        ValidationError::RankMismatch {
                            expected: N,
                            actual,
                        },
                    ),
                ))
            }
        };
        let layout = match TensorLayout::<Rank<N>>::compact(shape) {
            Ok(layout) => layout,
            Err(error) => return Err(ReinterpretError::new(self, tensor_layout_error(op, error))),
        };
        let TypedTensor {
            placement, storage, ..
        } = self;
        let storage = match storage {
            DynamicStorage::Host(host) => DynamicStorage::Host(host),
            DynamicStorage::Group(GroupStorage { group }) => DynamicStorage::Group(GroupStorage {
                group: Box::new(OwnedTensorGroup {
                    group: group.group,
                    slot: group.slot,
                    allocation_index: group.allocation_index,
                    host_ptr: group.host_ptr,
                    host_byte_len: group.host_byte_len,
                    _rank: PhantomData,
                }),
            }),
        };
        Ok(TypedTensor {
            shape,
            placement,
            layout: stride_vec(layout.strides()),
            storage,
        })
    }

    /// Return the storage backing this tensor.
    ///
    /// This is an explicit storage-inspection API for backend glue and tests.
    /// Host value inspection should prefer [`TypedTensor::host_data`] when the
    /// caller requires host storage.
    ///
    /// # Panics
    ///
    /// Panics only if the typed descriptor and its single group owner are
    /// internally inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StorageBuffer, TypedTensor};
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert!(matches!(t.buffer(), StorageBuffer::Host(_)));
    /// ```
    pub fn buffer(&self) -> &StorageBuffer<T>
    where
        T: 'static,
    {
        match &self.storage {
            DynamicStorage::Host(HostStorage { data }) => &data.buffer,
            DynamicStorage::Group(core) => core
                .group
                .host_buffer::<T>()
                .or_else(|| core.group.backend_buffer::<T>())
                .unwrap_or_else(|| unreachable!("typed tensor group storage mismatch")),
        }
    }

    /// Return the provider family for this tensor when backend-owned.
    #[doc(hidden)]
    pub fn backend_family(&self) -> Option<&'static str>
    where
        T: TensorScalar + 'static,
    {
        self.as_view().backend_family()
    }

    /// Return the opaque backend buffer for backend-owned tensors.
    #[doc(hidden)]
    pub fn backend_buffer(&self) -> Option<&dyn BackendStorage<T>>
    where
        T: 'static,
    {
        match &self.storage {
            DynamicStorage::Host(..) => None,
            DynamicStorage::Group(core) => match core.group.backend_buffer::<T>() {
                Some(StorageBuffer::Backend(buffer)) => Some(buffer.as_ref()),
                Some(StorageBuffer::Host(_)) | None => None,
            },
        }
    }

    /// Return the mutable backend buffer for an exclusive owner borrow.
    #[doc(hidden)]
    pub fn backend_buffer_mut(&mut self) -> Option<&mut dyn BackendStorage<T>>
    where
        T: 'static,
    {
        let DynamicStorage::Group(core) = &mut self.storage else {
            return None;
        };
        match core.group.backend_buffer_mut::<T>()? {
            StorageBuffer::Host(_) => None,
            StorageBuffer::Backend(buffer) => Some(buffer.as_mut()),
        }
    }

    /// Prepare this backend tensor for one provider-native read binding.
    #[doc(hidden)]
    pub fn prepare_device_read(
        &self,
        op: &'static str,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>>
    where
        T: TensorScalar + 'static,
    {
        let DynamicStorage::Group(core) = &self.storage else {
            return Err(crate::Error::runtime_state_source(
                op,
                crate::AccessError::Unsupported { backend: "host" },
            ));
        };
        let layout = self.layout();
        core.group
            .prepare_device_read_for_layout::<T>(&layout)
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }

    /// Prepare this backend tensor for one provider-native write binding.
    #[doc(hidden)]
    pub fn prepare_device_write(
        &mut self,
        op: &'static str,
    ) -> crate::Result<Box<dyn PreparedDeviceAccess + '_>>
    where
        T: TensorScalar + 'static,
    {
        let layout = self.layout();
        let DynamicStorage::Group(core) = &mut self.storage else {
            return Err(crate::Error::runtime_state_source(
                op,
                crate::AccessError::Unsupported { backend: "host" },
            ));
        };
        core.group
            .prepare_device_write_for_layout::<T>(&layout)
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }

    pub(crate) fn buffer_len(&self) -> usize
    where
        T: 'static,
    {
        match &self.storage {
            DynamicStorage::Host(HostStorage { data }) => data.as_slice().len(),
            DynamicStorage::Group(core) => core
                .group
                .group
                .descriptor_len(core.group.slot)
                .unwrap_or_else(|| unreachable!("typed tensor group descriptor mismatch")),
        }
    }

    /// Return the shared-allocation domain carried by the backend buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let tensor = TypedTensor::<f32>::from_vec_col_major(vec![1], vec![1.0])?;
    /// assert_eq!(tensor.allocation_domain(), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn allocation_domain(&self) -> Option<AllocationDomainId>
    where
        T: 'static,
    {
        match &self.storage {
            DynamicStorage::Host(..) => None,
            DynamicStorage::Group(core) => core
                .group
                .group
                .backend_identity(core.group.slot)
                .map(|(domain, _)| domain),
        }
    }

    /// Return the stable physical backend allocation identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let tensor = TypedTensor::<f32>::from_vec_col_major(vec![1], vec![1.0])?;
    /// assert_eq!(tensor.allocation_id(), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn allocation_id(&self) -> Option<AllocationId>
    where
        T: 'static,
    {
        match &self.storage {
            DynamicStorage::Host(..) => None,
            DynamicStorage::Group(core) => core
                .group
                .group
                .backend_identity(core.group.slot)
                .map(|(_, allocation)| allocation),
        }
    }

    /// Borrow this tensor as a typed view preserving rank and layout metadata.
    ///
    /// # Panics
    ///
    /// Panics only if the typed descriptor and its single group owner are
    /// internally inconsistent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    /// let view = tensor.as_view();
    /// assert_eq!(view.strides(), &[1, 2]);
    /// ```
    pub fn as_view(&self) -> TypedTensorView<'_, T, R>
    where
        T: 'static,
    {
        let layout = self.layout();
        let placement = self.placement.clone();
        match &self.storage {
            DynamicStorage::Host(HostStorage { data }) => TypedTensorView {
                buffer: TensorStorageRef::Host(data.as_slice()),
                root: None,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            },
            DynamicStorage::Group(core) => {
                // Managed owners are constructed only with their matching preset T.
                let root = core
                    .group
                    .group
                    .view_raw::<T, R>(core.group.slot)
                    .unwrap_or_else(|error| {
                        unreachable!("typed tensor group descriptor mismatch: {error}")
                    });
                let buffer = if let Some(allocation) = root.backend_allocation() {
                    TensorStorageRef::Root(allocation)
                } else {
                    TensorStorageRef::Host(core.group.host_slice::<T>().unwrap_or_default())
                };
                TypedTensorView {
                    buffer,
                    root: Some(root),
                    layout,
                    placement,
                    _representation: std::marker::PhantomData,
                }
            }
        }
    }

    /// Mutably borrow this tensor as a typed view preserving rank and layout metadata.
    ///
    /// # Panics
    ///
    /// Panics only if the typed descriptor and its single group owner are
    /// internally inconsistent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![1]).unwrap();
    /// *tensor.as_view_mut().get_mut(&[0]).unwrap() = 2;
    /// assert_eq!(tensor.as_slice().unwrap(), &[2]);
    /// ```
    pub fn as_view_mut(&mut self) -> TypedTensorViewMut<'_, T, R>
    where
        T: TensorScalar + 'static,
    {
        let layout = self.layout();
        let placement = self.placement.clone();
        match &mut self.storage {
            DynamicStorage::Host(HostStorage { data }) => TypedTensorViewMut {
                buffer: TensorStorageRefMut::Host(data.as_mut_slice()),
                root: None,
                layout,
                placement,
                _representation: std::marker::PhantomData,
            },
            DynamicStorage::Group(core) => {
                let mut root = core.group.view_mut::<T>().unwrap_or_else(|error| {
                    unreachable!("typed tensor group descriptor mismatch: {error}")
                });
                let buffer = if let Some(StorageBuffer::Backend(buffer)) = root.backend_buffer_mut()
                {
                    TensorStorageRefMut::Backend(buffer.as_mut())
                } else {
                    TensorStorageRefMut::Host(root.host_slice_mut().unwrap_or_else(|error| {
                        unreachable!("typed tensor group descriptor is not host-backed: {error}")
                    }))
                };
                TypedTensorViewMut {
                    buffer,
                    root: Some(root),
                    layout,
                    placement,
                    _representation: std::marker::PhantomData,
                }
            }
        }
    }

    /// Borrow a read-only strided region view over this tensor's backend
    /// (device) buffer from explicit layout metadata.
    ///
    /// This is a metadata-only view: no data is copied or transferred. The
    /// layout's reachable element span is validated against the backend
    /// buffer's physical length. Host-backed tensors are rejected with an
    /// explicit backend error; host regions are expressed with
    /// [`TypedTensorView::from_slice`] over host storage instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// // Host tensors are rejected: this constructor is for backend buffers.
    /// let host = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap();
    /// let err = host.backend_region_view(vec![2, 2], vec![1, 2], 0).unwrap_err();
    /// assert!(err.to_string().contains("backend"));
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when this tensor is host-backed;
    /// backend region views require a backend buffer. It returns
    /// [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] for incompatible
    /// shape/stride ranks, [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`]
    /// when the region exceeds the backend buffer, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    pub fn backend_region_view(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> crate::Result<TypedTensorView<'_, T, DynRank>>
    where
        T: TensorScalar + 'static,
    {
        let op = "TypedTensor::backend_region_view";
        let DynamicStorage::Group(core) = &self.storage else {
            return Err(crate::Error::runtime_state(op, "expected a backend (device) allocation; host tensors use TypedTensorView::from_slice over host storage"));
        };
        let root = core.group.view_dyn::<T>()?;
        let Some(allocation) = root.backend_allocation() else {
            return Err(crate::Error::runtime_state(
                op,
                "expected a backend (device) allocation; host tensors use \
                 TypedTensorView::from_slice over host storage",
            ));
        };
        let element_len = allocation
            .root_extent()
            .byte_len()
            .checked_div(size_of::<T>())
            .ok_or_else(|| crate::Error::validation(op, ValidationError::IntegerOverflow))?;
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, element_len)
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(TypedTensorView {
            buffer: TensorStorageRef::Root(allocation),
            root: Some(root),
            layout,
            placement: self.placement.clone(),
            _representation: std::marker::PhantomData,
        })
    }

    /// Borrow a mutable strided region view over this tensor's backend
    /// (device) buffer from explicit layout metadata.
    ///
    /// This is the mutable counterpart of
    /// [`TypedTensor::backend_region_view`]. The layout's reachable element
    /// span is validated against the backend buffer's physical length, and
    /// layouts whose logical elements alias the same physical element are
    /// rejected. Host-backed tensors are rejected with an explicit backend
    /// error; mutable host regions must go through
    /// [`TypedTensorViewMut::try_multi_slice_mut`] or host constructors.
    ///
    /// The returned view borrows the tensor's backend owner exclusively for its
    /// lifetime. This keeps write authority tied to the owner; a second mutable
    /// region view must be created only after the first borrow ends.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// // Host tensors are rejected: this constructor is for backend buffers.
    /// let mut host = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap();
    /// let err = host.backend_region_view_mut(vec![2, 2], vec![1, 2], 0).unwrap_err();
    /// assert!(err.to_string().contains("backend"));
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when this tensor is host-backed;
    /// mutable backend region views require a backend buffer. It returns
    /// [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] for incompatible
    /// shape/stride ranks, [`tenferro_tensor_core::ValidationError::ViewOutOfBounds`]
    /// when the region exceeds the backend buffer,
    /// [`tenferro_tensor_core::ValidationError::OverlappingMutableLayout`] when
    /// logical elements alias, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] for layout
    /// arithmetic overflow.
    pub fn backend_region_view_mut(
        &mut self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> crate::Result<TypedTensorViewMut<'_, T, DynRank>>
    where
        T: TensorScalar + 'static,
    {
        let op = "TypedTensor::backend_region_view_mut";
        let placement = self.placement.clone();
        let DynamicStorage::Group(core) = &mut self.storage else {
            return Err(crate::Error::runtime_state(op, "expected a backend (device) buffer; mutable host regions use TypedTensorViewMut host constructors or try_multi_slice_mut"));
        };
        let mut root = core.group.view_mut_dyn::<T>()?;
        let Some(StorageBuffer::Backend(buffer)) = root.backend_buffer_mut() else {
            return Err(crate::Error::runtime_state(
                op,
                "expected a backend (device) buffer; mutable host regions use \
                 TypedTensorViewMut host constructors or try_multi_slice_mut",
            ));
        };
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(TypedTensorViewMut {
            buffer: TensorStorageRefMut::Backend(buffer.as_mut()),
            root: Some(root),
            layout,
            placement,
            _representation: std::marker::PhantomData,
        })
    }

    /// Consume this tensor and return its storage, layout, and placement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StorageBuffer, TypedTensor};
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let Ok((buffer, layout, placement)) = t.into_parts() else {
    ///     panic!("a plain host owner extracts")
    /// };
    /// assert!(matches!(buffer, StorageBuffer::Host(_)));
    /// assert_eq!(layout.shape(), &[2]);
    /// assert!(placement.device.is_none());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when it uses
    /// backend storage; download it before extracting host storage.
    pub fn into_parts(
        self,
    ) -> std::result::Result<(StorageBuffer<T>, TensorLayout<R>, Placement), ReinterpretError<Self>>
    where
        T: TensorScalar,
    {
        let TypedTensor {
            shape,
            placement,
            layout: cached_layout,
            storage,
        } = self;
        let layout = TensorLayout::compact(shape.clone())
            .unwrap_or_else(|err| unreachable!("validated owned shape: {err}"));
        match storage {
            DynamicStorage::Host(HostStorage { data }) => {
                Ok((StorageBuffer::Host(data.into_vec()), layout, placement))
            }
            DynamicStorage::Group(GroupStorage { group }) => match group.into_host_vec::<T>() {
                Ok(data) => Ok((StorageBuffer::Host(data), layout, placement)),
                Err((group, error)) => Err(ReinterpretError::new(
                    TypedTensor {
                        shape,
                        placement,
                        layout: cached_layout,
                        storage: DynamicStorage::Group(GroupStorage {
                            group: Box::new(group),
                        }),
                    },
                    error,
                )),
            },
        }
    }
}

impl<T: TensorScalar, R: TensorRank> TypedTensor<T, R> {
    fn into_managed(self) -> Self {
        let TypedTensor {
            shape,
            placement,
            layout,
            storage,
        } = self;
        match storage {
            DynamicStorage::Host(HostStorage { data }) => {
                let group = promote_host_group::<T, R>(shape.clone(), data)
                    .unwrap_or_else(|err| unreachable!("a validated host owner promotes: {err}"));
                Self {
                    shape,
                    placement,
                    layout,
                    storage: DynamicStorage::Group(GroupStorage {
                        group: Box::new(group),
                    }),
                }
            }
            DynamicStorage::Group(core) => Self {
                shape,
                placement,
                layout,
                storage: DynamicStorage::Group(core),
            },
        }
    }

    fn into_group_parts(self) -> (AllocationGroup, DescriptorSlot) {
        let TypedTensor {
            placement, storage, ..
        } = self.into_managed();
        let DynamicStorage::Group(GroupStorage { group }) = storage else {
            unreachable!("explicit group promotion produces managed storage")
        };
        let (mut group, slot) = group.into_parts();
        group.publish_live_descriptor_placement(slot, placement);
        (group, slot)
    }

    /// Construct a backend-pooled host tensor with weak final-owner reclamation.
    /// Explicit Vec extraction disarms reclamation; aliases and retained groups
    /// keep the original root alive. The recycler must not acquire session locks.
    ///
    /// # Errors
    /// Returns validation errors for invalid shape/length and runtime-state errors
    /// if the freshly created host root cannot attach its recycler.
    #[doc(hidden)]
    pub fn from_vec_col_major_with_recycler(
        shape: impl tenferro_tensor_core::IntoRankShape<R>,
        data: Vec<T>,
        recycler: std::sync::Weak<dyn crate::HostBufferRecycler<T>>,
    ) -> crate::Result<Self> {
        let mut tensor = Self::from_vec_col_major(shape, data)?;
        if let DynamicStorage::Host(HostStorage { data }) = &mut tensor.storage {
            data.recycler = Some(recycler);
        }
        Ok(tensor)
    }

    /// Borrow compact host-visible storage through one synchronization guard.
    #[doc(hidden)]
    pub fn with_host_read<U>(&self, f: impl FnOnce(&[T]) -> U) -> crate::Result<U>
    where
        T: TensorScalar + 'static,
    {
        self.as_view().with_host_read(f)
    }

    /// Borrow compact host-visible storage through one exclusive write guard.
    #[doc(hidden)]
    pub fn with_host_write<U>(&mut self, f: impl FnOnce(&mut [T]) -> U) -> crate::Result<U>
    where
        T: TensorScalar + 'static,
    {
        match &mut self.storage {
            DynamicStorage::Host(HostStorage { data }) => Ok(f(data.as_mut_slice())),
            DynamicStorage::Group(core) => {
                let slot = core.group.slot;
                let mut view = core
                    .group
                    .group
                    .view_mut::<T, R>(slot)
                    .map_err(|error| group_error("TypedTensor::with_host_write", error))?;
                let mut prepared = view.prepare_host_write().map_err(|error| {
                    crate::Error::runtime_state("TypedTensor::with_host_write", error.to_string())
                })?;
                let slice = prepared.as_slice_mut().ok_or_else(|| {
                    crate::Error::unsupported(
                        "TypedTensor::with_host_write",
                        "host guard access requires a compact descriptor",
                    )
                })?;
                Ok(f(slice))
            }
        }
    }

    fn group_host_slice(&self) -> &[T] {
        let DynamicStorage::Group(core) = &self.storage else {
            return &[];
        };
        core.group
            .view::<T>()
            .ok()
            .and_then(|view| view.host_slice().ok())
            .unwrap_or_default()
    }
}

/// Layout builder used by an owning representation reinterpretation.
type ReinterpretLayoutFn =
    fn(&[usize], &[isize], isize, usize, &'static str) -> crate::Result<TensorLayout<DynRank>>;

fn reinterpret_owned<S: TensorScalar, U: TensorScalar, R: TensorRank>(
    owner: TypedTensor<S, R>,
    from: DType,
    to: DType,
    make_layout: ReinterpretLayoutFn,
    op: &'static str,
) -> Result<TypedTensor<U>, ReinterpretError<TypedTensor<S, R>>> {
    if let Err(error) = validate_representation_pair(op, from, to) {
        return Err(ReinterpretError::new(owner, error));
    }
    if matches!(to, DType::C32 | DType::C64) && !owner.buffer_len().is_multiple_of(2) {
        return Err(ReinterpretError::new(
            owner,
            crate::Error::invalid_argument(
                op,
                "buffer",
                "the owned real buffer must contain an even number of elements",
            ),
        ));
    }
    let source_layout = owner.layout();
    let target_layout = match make_layout(
        source_layout.shape(),
        source_layout.strides(),
        source_layout.offset(),
        owner.buffer_len(),
        op,
    ) {
        Ok(layout) => layout,
        Err(error) => return Err(ReinterpretError::new(owner, error)),
    };
    // Only an explicit owning representation conversion promotes a plain owner;
    // borrowed session dispatch never passes through a group.
    let TypedTensor {
        shape,
        placement,
        layout,
        storage,
    } = owner.into_managed();
    let DynamicStorage::Group(GroupStorage { group }) = storage else {
        unreachable!("explicit group promotion produces managed storage")
    };
    match group.reinterpret::<S, U>(
        target_layout.shape().to_vec(),
        target_layout.strides().to_vec(),
        target_layout.offset(),
    ) {
        Ok(group) => Ok(TypedTensor {
            shape: shape_vec(target_layout.shape()),
            placement,
            layout: stride_vec(target_layout.strides()),
            storage: DynamicStorage::Group(GroupStorage {
                group: Box::new(group),
            }),
        }),
        Err((group, error)) => Err(ReinterpretError::new(
            TypedTensor {
                shape,
                placement,
                layout,
                storage: DynamicStorage::Group(GroupStorage {
                    group: Box::new(group),
                }),
            },
            error,
        )),
    }
}

impl<R: TensorRank> TypedTensor<Complex32, R> {
    /// Borrow this tensor as an interleaved `f32` view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, or [`ValidationError::ViewOutOfBounds`] for an
    /// invalid tensor layout.
    pub fn as_real_view(&self) -> crate::Result<TypedTensorView<'_, f32, DynRank>> {
        self.as_view().as_real_view()
    }

    /// Borrow this tensor mutably as an interleaved `f32` view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, [`ValidationError::OverlappingMutableLayout`]
    /// for a non-injective layout, or [`ValidationError::ViewOutOfBounds`] for
    /// invalid representation metadata.
    pub fn as_real_view_mut(&mut self) -> crate::Result<TypedTensorViewMut<'_, f32, DynRank>> {
        let op = "TypedTensor::as_real_view_mut";
        validate_representation_pair(op, DType::C32, DType::F32)?;
        let source_layout = self.layout();
        let layout = reinterpret_complex_to_real_layout(
            self.shape(),
            source_layout.strides(),
            source_layout.offset(),
            self.buffer_len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        if self.backend_buffer().is_some() {
            return Err(crate::Error::unsupported(
                op,
                "backend representation reinterpretation is enabled by the provider phases",
            ));
        }
        let placement = self.placement().clone();
        let buffer = TensorStorageRefMut::Host(reinterpret_host_slice_mut::<Complex32, f32>(
            self.host_data_mut()?,
            op,
        )?);
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        })
    }

    /// Consume this tensor and reinterpret its owner as `f32` without copying.
    ///
    /// A failed operation returns the unchanged owner through
    /// [`ReinterpretError::into_owner`].
    ///
    /// # Errors
    ///
    /// Returns [`ReinterpretError::error`] containing
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] while retaining the unchanged
    /// owner.
    pub fn into_real(self) -> Result<TypedTensor<f32, DynRank>, ReinterpretError<Self>> {
        reinterpret_owned(
            self,
            DType::C32,
            DType::F32,
            reinterpret_complex_to_real_layout,
            "TypedTensor::into_real",
        )
    }
}

impl<R: TensorRank> TypedTensor<Complex64, R> {
    /// Borrow this tensor as an interleaved `f64` view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, or [`ValidationError::ViewOutOfBounds`] for an
    /// invalid tensor layout.
    pub fn as_real_view(&self) -> crate::Result<TypedTensorView<'_, f64, DynRank>> {
        self.as_view().as_real_view()
    }

    /// Borrow this tensor mutably as an interleaved `f64` view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, [`ValidationError::OverlappingMutableLayout`]
    /// for a non-injective layout, or [`ValidationError::ViewOutOfBounds`] for
    /// invalid representation metadata.
    pub fn as_real_view_mut(&mut self) -> crate::Result<TypedTensorViewMut<'_, f64, DynRank>> {
        let op = "TypedTensor::as_real_view_mut";
        validate_representation_pair(op, DType::C64, DType::F64)?;
        let source_layout = self.layout();
        let layout = reinterpret_complex_to_real_layout(
            self.shape(),
            source_layout.strides(),
            source_layout.offset(),
            self.buffer_len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        if self.backend_buffer().is_some() {
            return Err(crate::Error::unsupported(
                op,
                "backend representation reinterpretation is enabled by the provider phases",
            ));
        }
        let placement = self.placement().clone();
        let buffer = TensorStorageRefMut::Host(reinterpret_host_slice_mut::<Complex64, f64>(
            self.host_data_mut()?,
            op,
        )?);
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        })
    }

    /// Consume this tensor and reinterpret its owner as `f64` without copying.
    ///
    /// A failed operation returns the unchanged owner through
    /// [`ReinterpretError::into_owner`].
    ///
    /// # Errors
    ///
    /// Returns [`ReinterpretError::error`] containing
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] while retaining the unchanged
    /// owner.
    pub fn into_real(self) -> Result<TypedTensor<f64, DynRank>, ReinterpretError<Self>> {
        reinterpret_owned(
            self,
            DType::C64,
            DType::F64,
            reinterpret_complex_to_real_layout,
            "TypedTensor::into_real",
        )
    }
}

impl<R: TensorRank> TypedTensor<f32, R> {
    /// Borrow this tensor as a complex view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, or [`ValidationError::ViewOutOfBounds`] for an
    /// invalid tensor layout.
    pub fn as_complex_view(&self) -> crate::Result<TypedTensorView<'_, Complex32, DynRank>> {
        self.as_view().as_complex_view()
    }

    /// Borrow this tensor mutably as a complex view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, [`ValidationError::OverlappingMutableLayout`]
    /// for a non-injective layout, or [`ValidationError::ViewOutOfBounds`] for
    /// invalid representation metadata.
    pub fn as_complex_view_mut(
        &mut self,
    ) -> crate::Result<TypedTensorViewMut<'_, Complex32, DynRank>> {
        let op = "TypedTensor::as_complex_view_mut";
        validate_representation_pair(op, DType::F32, DType::C32)?;
        let source_layout = self.layout();
        let layout = reinterpret_real_to_complex_layout(
            self.shape(),
            source_layout.strides(),
            source_layout.offset(),
            self.buffer_len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        if self.backend_buffer().is_some() {
            return Err(crate::Error::unsupported(
                op,
                "backend representation reinterpretation is enabled by the provider phases",
            ));
        }
        let placement = self.placement().clone();
        let buffer = TensorStorageRefMut::Host(reinterpret_host_slice_mut::<f32, Complex32>(
            self.host_data_mut()?,
            op,
        )?);
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        })
    }

    /// Consume this tensor and reinterpret its owner as `Complex32` without copying.
    ///
    /// The compact source must have an even physical element count. A failed
    /// operation returns the unchanged owner through
    /// [`ReinterpretError::into_owner`].
    ///
    /// # Errors
    ///
    /// Returns [`ReinterpretError::error`] containing
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] while retaining the unchanged
    /// owner.
    pub fn into_complex(self) -> Result<TypedTensor<Complex32, DynRank>, ReinterpretError<Self>> {
        reinterpret_owned(
            self,
            DType::F32,
            DType::C32,
            reinterpret_real_to_complex_layout,
            "TypedTensor::into_complex",
        )
    }
}

impl<R: TensorRank> TypedTensor<f64, R> {
    /// Borrow this tensor as a complex view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, or [`ValidationError::ViewOutOfBounds`] for an
    /// invalid tensor layout.
    pub fn as_complex_view(&self) -> crate::Result<TypedTensorView<'_, Complex64, DynRank>> {
        self.as_view().as_complex_view()
    }

    /// Borrow this tensor mutably as a complex view without copying.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for backend reinterpretation
    /// that is not supported, [`ValidationError::OverlappingMutableLayout`]
    /// for a non-injective layout, or [`ValidationError::ViewOutOfBounds`] for
    /// invalid representation metadata.
    pub fn as_complex_view_mut(
        &mut self,
    ) -> crate::Result<TypedTensorViewMut<'_, Complex64, DynRank>> {
        let op = "TypedTensor::as_complex_view_mut";
        validate_representation_pair(op, DType::F64, DType::C64)?;
        let source_layout = self.layout();
        let layout = reinterpret_real_to_complex_layout(
            self.shape(),
            source_layout.strides(),
            source_layout.offset(),
            self.buffer_len(),
            op,
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        if self.backend_buffer().is_some() {
            return Err(crate::Error::unsupported(
                op,
                "backend representation reinterpretation is enabled by the provider phases",
            ));
        }
        let placement = self.placement().clone();
        let buffer = TensorStorageRefMut::Host(reinterpret_host_slice_mut::<f64, Complex64>(
            self.host_data_mut()?,
            op,
        )?);
        Ok(TypedTensorViewMut {
            buffer,
            root: None,
            layout,
            placement,
            _representation: std::marker::PhantomData,
        })
    }

    /// Consume this tensor and reinterpret its owner as `Complex64` without copying.
    ///
    /// The compact source must have an even physical element count. A failed
    /// operation returns the unchanged owner through
    /// [`ReinterpretError::into_owner`].
    ///
    /// # Errors
    ///
    /// Returns [`ReinterpretError::error`] containing
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] while retaining the unchanged
    /// owner.
    pub fn into_complex(self) -> Result<TypedTensor<Complex64, DynRank>, ReinterpretError<Self>> {
        reinterpret_owned(
            self,
            DType::F64,
            DType::C64,
            reinterpret_real_to_complex_layout,
            "TypedTensor::into_complex",
        )
    }
}

impl Tensor {
    /// Borrow a complex tensor as its sealed interleaved real representation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for a non-complex dtype and
    /// [`ValidationError::ViewOutOfBounds`] or
    /// [`ValidationError::InvalidArgument`] for invalid layout metadata.
    pub fn as_real_view(&self) -> crate::Result<TensorView<'_>> {
        match self.dtype() {
            DType::C32 => self
                .as_typed::<Complex<f32>>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_real_view()
                .map(TensorView::F32),
            DType::C64 => self
                .as_typed::<Complex<f64>>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_real_view()
                .map(TensorView::F64),
            other => Err(crate::Error::unsupported_dtype_conversion(
                "Tensor::as_real_view",
                other,
                DType::F32,
                "only complex tensors have a sealed real representation view",
            )),
        }
    }

    /// Borrow a complex tensor mutably as its sealed interleaved real representation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for a non-complex dtype and
    /// [`ValidationError::ViewOutOfBounds`] or
    /// [`ValidationError::InvalidArgument`] for invalid layout metadata.
    pub fn as_real_view_mut(&mut self) -> crate::Result<TensorViewMut<'_>> {
        match self.dtype() {
            DType::C32 => self
                .as_typed_mut::<Complex<f32>>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_real_view_mut()
                .map(TensorViewMut::F32),
            DType::C64 => self
                .as_typed_mut::<Complex<f64>>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_real_view_mut()
                .map(TensorViewMut::F64),
            other => Err(crate::Error::unsupported_dtype_conversion(
                "Tensor::as_real_view_mut",
                other,
                DType::F32,
                "only complex tensors have a sealed real representation view",
            )),
        }
    }

    /// Consume a complex tensor and reinterpret its owner as real without copying.
    ///
    /// # Errors
    ///
    /// Returns [`ReinterpretError::error`] containing
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] while retaining the unchanged
    /// owner.
    pub fn into_real(self) -> Result<Self, ReinterpretError<Self>> {
        let dtype = self.dtype();
        match dtype {
            DType::C32 => {
                let tensor = <Complex<f32> as TensorScalar>::into_typed(self)
                    .unwrap_or_else(|_| unreachable!("the dtype guard selects this arm"));
                tensor
                    .into_real()
                    .map(Tensor::from_typed::<f32>)
                    .map_err(|error| {
                        let (owner, error) = error.into_parts();
                        ReinterpretError::new(Tensor::from_typed::<Complex<f32>>(owner), error)
                    })
            }
            DType::C64 => {
                let tensor = <Complex<f64> as TensorScalar>::into_typed(self)
                    .unwrap_or_else(|_| unreachable!("the dtype guard selects this arm"));
                tensor
                    .into_real()
                    .map(Tensor::from_typed::<f64>)
                    .map_err(|error| {
                        let (owner, error) = error.into_parts();
                        ReinterpretError::new(Tensor::from_typed::<Complex<f64>>(owner), error)
                    })
            }
            _ => Err(ReinterpretError::new(
                self,
                crate::Error::unsupported(
                    "Tensor::into_real",
                    "only complex tensors have a sealed real representation",
                ),
            )),
        }
    }

    /// Borrow an interleaved real tensor as its sealed complex representation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for a non-real dtype and
    /// [`ValidationError::ViewOutOfBounds`] or
    /// [`ValidationError::InvalidArgument`] for invalid layout metadata.
    pub fn as_complex_view(&self) -> crate::Result<TensorView<'_>> {
        match self.dtype() {
            DType::F32 => self
                .as_typed::<f32>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_complex_view()
                .map(TensorView::C32),
            DType::F64 => self
                .as_typed::<f64>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_complex_view()
                .map(TensorView::C64),
            other => Err(crate::Error::unsupported_dtype_conversion(
                "Tensor::as_complex_view",
                other,
                DType::C32,
                "only real tensors can have a sealed complex representation view",
            )),
        }
    }

    /// Borrow an interleaved real tensor mutably as its sealed complex representation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] for a non-real dtype and
    /// [`ValidationError::ViewOutOfBounds`] or
    /// [`ValidationError::InvalidArgument`] for invalid layout metadata.
    pub fn as_complex_view_mut(&mut self) -> crate::Result<TensorViewMut<'_>> {
        match self.dtype() {
            DType::F32 => self
                .as_typed_mut::<f32>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_complex_view_mut()
                .map(TensorViewMut::C32),
            DType::F64 => self
                .as_typed_mut::<f64>()
                .unwrap_or_else(|| unreachable!("the dtype guard selects this arm"))
                .as_complex_view_mut()
                .map(TensorViewMut::C64),
            other => Err(crate::Error::unsupported_dtype_conversion(
                "Tensor::as_complex_view_mut",
                other,
                DType::C32,
                "only real tensors can have a sealed complex representation view",
            )),
        }
    }

    /// Consume a real tensor and reinterpret its owner as complex without copying.
    ///
    /// # Errors
    ///
    /// Returns [`ReinterpretError::error`] containing
    /// [`ValidationError::InvalidArgument`] or
    /// [`ValidationError::ViewOutOfBounds`] while retaining the unchanged
    /// owner.
    pub fn into_complex(self) -> Result<Self, ReinterpretError<Self>> {
        let dtype = self.dtype();
        match dtype {
            DType::F32 => {
                let tensor = <f32 as TensorScalar>::into_typed(self)
                    .unwrap_or_else(|_| unreachable!("the dtype guard selects this arm"));
                tensor
                    .into_complex()
                    .map(Tensor::from_typed::<Complex<f32>>)
                    .map_err(|error| {
                        let (owner, error) = error.into_parts();
                        ReinterpretError::new(Tensor::from_typed::<f32>(owner), error)
                    })
            }
            DType::F64 => {
                let tensor = <f64 as TensorScalar>::into_typed(self)
                    .unwrap_or_else(|_| unreachable!("the dtype guard selects this arm"));
                tensor
                    .into_complex()
                    .map(Tensor::from_typed::<Complex<f64>>)
                    .map_err(|error| {
                        let (owner, error) = error.into_parts();
                        ReinterpretError::new(Tensor::from_typed::<f64>(owner), error)
                    })
            }
            _ => Err(ReinterpretError::new(
                self,
                crate::Error::unsupported(
                    "Tensor::into_complex",
                    "only real tensors have a sealed complex representation",
                ),
            )),
        }
    }

    /// Make an explicit owning copy of this dtype-erased tensor.
    ///
    /// `Tensor` deliberately does not implement [`Clone`]: copying can fail
    /// (device-only storage is rejected) and allocates a new, independent
    /// owner, so it is an explicit fallible call rather than an infallible
    /// `clone()`. The copy shares nothing with `self`; mutating one never
    /// affects the other. To share one tensor between several users without
    /// copying, wrap it in [`std::sync::Arc`] or borrow views of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_tensor::Tensor;
    ///
    /// let weights = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// // Two independent owners of the same values.
    /// let copy = weights.duplicate()?;
    /// assert_eq!(copy.as_slice::<f64>()?, weights.as_slice::<f64>()?);
    ///
    /// // A shared, read-only handle instead of a copy.
    /// let shared = Arc::new(weights);
    /// let other = Arc::clone(&shared);
    /// assert_eq!(other.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] or [`crate::Error::Unsupported`]
    /// when the selected backend/storage owner cannot be duplicated.
    pub fn duplicate(&self) -> crate::Result<Self> {
        match &self.payload {
            // A caller-owned payload is copied through its own entry point, which
            // keeps its element type and its view; sharing the payload instead
            // would alias the caller's storage.
            TensorPayload::External(payload, placement) => Ok(Self {
                payload: TensorPayload::External(payload.duplicate(), placement.clone()),
            }),
            TensorPayload::Native(_) => match self.dtype() {
                DType::F32 => self.duplicate_typed::<f32>(),
                DType::F64 => self.duplicate_typed::<f64>(),
                DType::I32 => self.duplicate_typed::<i32>(),
                DType::I64 => self.duplicate_typed::<i64>(),
                DType::Bool => self.duplicate_typed::<bool>(),
                DType::C32 => self.duplicate_typed::<Complex<f32>>(),
                DType::C64 => self.duplicate_typed::<Complex<f64>>(),
                DType::External(_) => unreachable!("a native payload carries a preset dtype"),
            },
        }
    }

    /// Duplicate a preset tensor through its typed owner.
    fn duplicate_typed<T: TensorScalar>(&self) -> crate::Result<Self> {
        self.as_typed::<T>()
            .unwrap_or_else(|| unreachable!("the caller selected this arm from the dtype"))
            .duplicate()
            .map(Self::from_typed)
    }

    /// Create a tensor from a shape and column-major flat data.
    ///
    /// This is the `Tensor`-level equivalent of
    /// `TypedTensor::<T>::from_vec_col_major`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    /// assert_eq!(t.shape(), &[2, 2]);
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::ShapeDataLengthMismatch`] when
    /// the shape product differs from `data.len()`, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when shape
    /// arithmetic overflows.
    pub fn from_vec_col_major<T: TensorScalar>(
        shape: impl tenferro_tensor_core::IntoShapeVec,
        data: Vec<T>,
    ) -> crate::Result<Self> {
        T::into_tensor(shape.into_shape_vec().to_vec(), data)
    }

    /// Create a tensor from a shape and row-major (C-order) flat data.
    ///
    /// The values are reordered once into tenferro's column-major storage;
    /// no row-major owner is created. Use this for buffers authored in
    /// PyTorch/NumPy/C order instead of passing them to
    /// [`Self::from_vec_col_major`], which would reinterpret them silently.
    /// This is the `Tensor`-level equivalent of
    /// `TypedTensor::<T>::from_vec_row_major`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// // Row-major [[1, 2, 3], [4, 5, 6]].
    /// let t = Tensor::from_vec_row_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.as_slice::<f64>()?, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::ShapeDataLengthMismatch`] when
    /// the shape product differs from `data.len()`, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when shape
    /// arithmetic overflows.
    pub fn from_vec_row_major<T: TensorScalar>(
        shape: impl tenferro_tensor_core::IntoShapeVec,
        data: Vec<T>,
    ) -> crate::Result<Self> {
        let (shape, reordered) = row_major_reorder::<T, tenferro_tensor_core::DynRank>(
            shape.into_shape_vec().to_vec(),
            data,
            "Tensor::from_vec_row_major",
        )?;
        T::into_tensor(shape.as_ref().to_vec(), reordered)
    }

    /// Tensor shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::from_typed(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        match &self.payload {
            TensorPayload::Native(preset) => with_preset!(preset, |typed| typed.shape()),
            // The payload keeps its shape from construction.
            TensorPayload::External(payload, _) => payload.shape(),
        }
    }

    /// Tensor dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TypedTensor};
    ///
    /// let t = Tensor::from_typed(TypedTensor::from_vec_col_major(vec![], vec![1.0]).unwrap());
    /// assert_eq!(t.dtype(), DType::F64);
    /// ```
    pub fn dtype(&self) -> DType {
        match &self.payload {
            TensorPayload::Native(preset) => match preset {
                PresetTensor::F32(_) => DType::F32,
                PresetTensor::F64(_) => DType::F64,
                PresetTensor::I32(_) => DType::I32,
                PresetTensor::I64(_) => DType::I64,
                PresetTensor::Bool(_) => DType::Bool,
                PresetTensor::C32(_) => DType::C32,
                PresetTensor::C64(_) => DType::C64,
            },
            // The payload carries its own element identity.
            TensorPayload::External(payload, _) => DType::External(payload.element_type_id()),
        }
    }

    /// Return placement metadata for this dtype-erased tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, Tensor};
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// assert_eq!(t.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// ```
    pub fn placement(&self) -> &Placement {
        match &self.payload {
            TensorPayload::Native(preset) => with_preset!(preset, |typed| typed.placement()),
            // The placement is stored with the payload.
            TensorPayload::External(_, placement) => placement,
        }
    }

    /// Return whether this tensor is backed by backend-native storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// assert!(!t.is_backend_buffer());
    /// ```
    pub fn is_backend_buffer(&self) -> bool {
        match &self.payload {
            TensorPayload::Native(preset) => with_preset!(preset, |typed| {
                matches!(&typed.storage, DynamicStorage::Group(core) if core.group.is_backend_buffer())
            }),
            // A caller-owned payload is host memory, never a backend buffer.
            TensorPayload::External(..) => false,
        }
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(t.layout_linear_offset(&[1])?, 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::RankMismatch`] when `indices`
    /// has the wrong rank, [`tenferro_tensor_core::ValidationError::InvalidArgument`]
    /// when an index is outside its axis extent, or
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when offset
    /// arithmetic overflows.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match &self.payload {
            TensorPayload::Native(_) => {
                let layout = tensor_layout(self);
                checked_view_offset_result(
                    layout.shape(),
                    layout.strides(),
                    layout.offset(),
                    indices,
                    "Tensor::layout_linear_offset",
                )
            }
            // A caller-owned payload's owner resolves its own offsets.
            TensorPayload::External(..) => Err(crate::Error::unsupported_dtype(
                "layout_linear_offset",
                self.dtype(),
                "an externally defined payload resolves its own offsets",
            )),
        }
    }

    /// Return whether this tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(t.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match &self.payload {
            TensorPayload::Native(_) => tensor_layout(self)
                .is_compact_col_major()
                .map_err(|err| tensor_layout_error("Tensor::is_col_major_contiguous", err)),
            // The payload is a compact column-major host tensor by construction.
            TensorPayload::External(..) => Ok(true),
        }
    }

    /// Return a compact string summary of this tensor's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(t.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        let layout = tensor_layout(self);
        layout_summary(layout.shape(), layout.strides(), layout.offset())
    }

    /// Assert this tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// t.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::IntegerOverflow`] when
    /// compactness arithmetic overflows, or
    /// [`tenferro_tensor_core::ValidationError::InvalidArgument`] when the
    /// tensor is not compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        let layout = tensor_layout(self);
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            layout.shape(),
            layout.strides(),
            layout.offset(),
            "Tensor::assert_col_major_contiguous",
        )
    }

    /// Try to borrow the host data as a typed slice.
    ///
    /// Returns an error if the tensor dtype does not match `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::from_typed(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    /// assert_eq!(t.as_slice::<f64>().unwrap(), [1.0, 2.0, 3.0].as_slice());
    /// assert!(t.as_slice::<f32>().is_err());
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor_core::ValidationError::DTypeMismatch`] when `T` does
    /// not match the tensor dtype, or [`crate::Error::RuntimeState`] when the
    /// matching tensor uses backend storage that has not been downloaded.
    pub fn as_slice<T: TensorScalar>(&self) -> crate::Result<&[T]> {
        T::as_slice(self)
    }

    /// Borrow the typed tensor when the requested scalar matches this tensor's dtype.
    ///
    /// This is the accessor tag-based dispatch needs: it recovers the typed tensor — and with it the
    /// device buffer — from a value whose element type is only known at run time, so a caller can
    /// dispatch on [`Tensor::dtype`] instead of matching every variant. An externally defined scalar
    /// is not a typed tensor, so it returns `None` rather than guessing a representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(tensor.as_typed::<f64>().is_some());
    /// assert!(tensor.as_typed::<f32>().is_none());
    /// assert_eq!(tensor.as_typed::<f64>().unwrap().shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn as_typed<T: TensorScalar>(&self) -> Option<&TypedTensor<T>> {
        if self.dtype() != T::dtype() {
            return None;
        }
        match &self.payload {
            TensorPayload::Native(preset) => with_preset!(preset, |typed| {
                (typed as &dyn Any).downcast_ref::<TypedTensor<T>>()
            }),
            TensorPayload::External(..) => None,
        }
    }

    /// Mutably borrow the typed tensor when the requested scalar matches this tensor's dtype.
    ///
    /// This is the mutable half of [`Tensor::as_typed`], for the tables whose arm calls a method that
    /// needs `&mut`, such as marking a freshly allocated output with its placement. An externally
    /// defined scalar is not a typed tensor, so it returns `None` for the same reason.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(tensor.as_typed_mut::<f64>().is_some());
    /// assert!(tensor.as_typed_mut::<f32>().is_none());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn as_typed_mut<T: TensorScalar>(&mut self) -> Option<&mut TypedTensor<T>> {
        if self.dtype() != T::dtype() {
            return None;
        }
        match &mut self.payload {
            TensorPayload::Native(preset) => with_preset!(preset, |typed| {
                (typed as &mut dyn Any).downcast_mut::<TypedTensor<T>>()
            }),
            TensorPayload::External(..) => None,
        }
    }

    /// Consume this tensor and return the owned typed tensor when the dtype matches.
    ///
    /// This is the consuming counterpart of [`Tensor::as_typed`], for the tables whose arm hands the typed
    /// tensor to a function that takes it by value — reusing its buffer rather than copying it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(tensor.into_typed::<f64>().is_ok());
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0])?;
    /// assert!(tensor.into_typed::<f64>().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when `T` is
    /// not this tensor's dtype, with [`crate::Error::Validation`] and
    /// [`tenferro_tensor_core::ValidationError::DTypeMismatch`] as the cause.
    /// A matching tensor is handed over as it is, including one whose storage
    /// lives in a backend buffer.
    pub fn into_typed<T: TensorScalar>(
        self,
    ) -> std::result::Result<TypedTensor<T>, ReinterpretError<Self>> {
        T::into_typed(self)
    }

    /// Consume this tensor and return its owned column-major buffer when the
    /// dtype matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// assert_eq!(t.into_vec_col_major::<f64>().unwrap().1, vec![2.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`ReinterpretError`] carrying the unchanged tensor when `T` does
    /// not match the tensor dtype or when the matching tensor uses backend
    /// storage that has not been downloaded.
    pub fn into_vec_col_major<T: TensorScalar>(
        self,
    ) -> std::result::Result<(Vec<usize>, Vec<T>), ReinterpretError<Self>> {
        let typed = T::into_typed(self)?;
        match typed.into_vec_col_major() {
            Ok(parts) => Ok(parts),
            Err(failure) => {
                let (owner, error) = failure.into_parts();
                Err(ReinterpretError::new(Tensor::from_typed(owner), error))
            }
        }
    }
}

// INVARIANT: retained for crate-local layout tests while tensor indexing
// helpers remain split across the tensor and CPU crates.
#[allow(dead_code)]
pub(crate) fn flat_to_multi(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    for i in 0..shape.len() {
        if shape[i] == 0 {
            out[i] = 0;
        } else {
            out[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
}
