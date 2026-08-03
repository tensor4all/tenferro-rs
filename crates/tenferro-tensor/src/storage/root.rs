use std::any::Any;
use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut, Range};

use crate::BackendId;
use crate::{DType, TensorScalar};

use super::diagnostics::{
    RequestedIdentity, StorageOperation, StorageOperationContext, StorageOperationError,
};
use super::identity::{RootResourceIdentity, RootResourceIdentityError};
use super::prepared::{AccessError, ProviderReadMapping, ProviderWriteMapping};
use super::span::{ByteRange, RootBoundSpan, RootResourceExtent};

/// Provider family metadata retained by the private allocation boundary.
pub(crate) type ProviderKind = BackendId;

/// Metadata-only provider capability descriptor for the root boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct ProviderCapabilities {
    host_access: bool,
}

impl ProviderCapabilities {
    pub(crate) const fn none() -> Self {
        Self { host_access: false }
    }

    pub(crate) const fn host() -> Self {
        Self { host_access: true }
    }

    pub(crate) const fn host_access(self) -> bool {
        self.host_access
    }
}

/// The sole unsafe boundary for importing a uniquely owned provider root.
///
/// # Safety
///
/// Implementors must own exactly one provider allocation and its destructor,
/// report stable truthful metadata, and uphold `Send`/`Sync` for the boxed
/// value. The storage core does not recover from a violated provider contract.
/// Mapping hooks must return guards that remain valid for the complete borrow
/// of the imported root, retain the provider allocation for that borrow, and
/// expose initialized valid `TensorScalar` representations with stable exact
/// span length. Typed preparation checks the returned pointer alignment before
/// any typed access.
// INVARIANT: #1555/#1558/#1560 place the sole provider/mapping unsafe boundary
// in this private storage kernel; public tensor graph and AD paths remain safe.
pub(crate) unsafe trait BackendAllocation:
    std::fmt::Debug + Send + Sync + 'static
{
    fn root_extent(&self) -> RootResourceExtent;
    fn provider_kind(&self) -> ProviderKind;
    fn capabilities(&self) -> ProviderCapabilities;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Map the checked span to initialized bytes with valid `TensorScalar`
    /// representations for the requested dtype. The mapping must keep the
    /// exact span length stable and retain the provider allocation for the
    /// returned borrow; typed preparation checks the returned pointer's
    /// alignment before any typed access.
    fn map_read(
        &self,
        _span: RootBoundSpan,
        _dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError> {
        Err(AccessError::Unsupported {
            backend: "unimplemented",
        })
    }

    /// Map the checked span to writable bytes with valid `TensorScalar`
    /// representations for the requested dtype. The mapping must keep the
    /// exact span length stable and retain exclusive provider access for the
    /// returned borrow; typed preparation checks the returned pointer's
    /// alignment before any typed access.
    fn map_write(
        &self,
        _span: RootBoundSpan,
        _dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        Err(AccessError::Unsupported {
            backend: "unimplemented",
        })
    }
}

/// One non-`Clone` root-bound span authority.
pub(crate) struct OwnedSpanClaim {
    root: RootResourceIdentity,
    span: RootBoundSpan,
}

/// The sole private owner for one imported root span.
pub(crate) struct OwnedStorage {
    pin: RootResourcePin,
    claim: OwnedSpanClaim,
}

/// Read-only capability derived from a shared borrow of an owner.
pub(crate) struct StorageRef<'a> {
    owner: &'a OwnedStorage,
}

/// Exclusive capability derived from an exclusive borrow of an owner.
pub(crate) struct StorageMut<'a> {
    owner: &'a mut OwnedStorage,
}

/// Move-only host allocation used by the P3 owner boundary.
///
/// The mutex is only the provider mapping guard. Rust access authority still
/// comes from `StorageRef`/`StorageMut`; the provider trait takes `&self` so a
/// guard can retain the allocation for the complete prepared borrow.
pub(crate) struct HostAllocation<T> {
    extent: RootResourceExtent,
    data: UnsafeCell<crate::StorageBuffer<T>>,
}

/// Lifetime-only pin for one root resource. Host roots stay inline so the
/// common CPU owner path does not allocate a second box merely to erase a
/// scalar-specific host allocation. Backend roots retain their provider box.
pub(crate) enum RootResourcePin {
    HostF32(HostAllocation<f32>),
    HostF64(HostAllocation<f64>),
    HostI32(HostAllocation<i32>),
    HostI64(HostAllocation<i64>),
    HostBool(HostAllocation<bool>),
    HostC32(HostAllocation<num_complex::Complex32>),
    HostC64(HostAllocation<num_complex::Complex64>),
    Backend(Box<dyn BackendAllocation>),
}

struct HostByteReadGuard<'a, T: TensorScalar> {
    data: &'a [T],
    range: Range<usize>,
}

struct HostByteWriteGuard<'a, T: TensorScalar> {
    data: &'a mut [T],
    range: Range<usize>,
}

// SAFETY: the allocation is accessed through the group-owned shared/exclusive
// capabilities. `map_write` is called only for an exclusive group child; the
// provider trait's `&self` receiver is not itself an access capability.
unsafe impl<T: TensorScalar> Send for HostAllocation<T> {}
unsafe impl<T: TensorScalar> Sync for HostAllocation<T> {}

fn host_bytes<T: TensorScalar>(data: &[T], range: Range<usize>) -> &[u8] {
    let byte_len = data
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .unwrap_or(0);
    debug_assert!(range.end <= byte_len);
    // SAFETY: `TensorScalar` is initialized, plain-data storage with a stable
    // size/alignment. The checked range is within the Vec allocation.
    unsafe {
        std::slice::from_raw_parts(
            (data.as_ptr() as *const u8).add(range.start),
            range.end - range.start,
        )
    }
}

fn host_bytes_mut<T: TensorScalar>(data: &mut [T], range: Range<usize>) -> &mut [u8] {
    let byte_len = data
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .unwrap_or(0);
    debug_assert!(range.end <= byte_len);
    // SAFETY: `TensorScalar` is initialized, plain-data storage with a stable
    // size/alignment. The checked range is within the Vec allocation and the
    // mutex guard holds exclusive provider access for the borrow.
    unsafe {
        std::slice::from_raw_parts_mut(
            (data.as_mut_ptr() as *mut u8).add(range.start),
            range.end - range.start,
        )
    }
}

impl<T: TensorScalar> Deref for HostByteReadGuard<'_, T> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        host_bytes(self.data, self.range.clone())
    }
}

impl<T: TensorScalar> AsRef<[u8]> for HostByteReadGuard<'_, T> {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl<T: TensorScalar> Deref for HostByteWriteGuard<'_, T> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        host_bytes(self.data, self.range.clone())
    }
}

impl<T: TensorScalar> DerefMut for HostByteWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        host_bytes_mut(self.data, self.range.clone())
    }
}

impl<T: TensorScalar> AsRef<[u8]> for HostByteWriteGuard<'_, T> {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl<T: TensorScalar> AsMut<[u8]> for HostByteWriteGuard<'_, T> {
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}

impl<T: TensorScalar> std::fmt::Debug for HostAllocation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostAllocation")
            .field("extent", &self.extent)
            .field(
                "element_count",
                // SAFETY: debug inspection is read-only and the allocation's
                // length is immutable after import.
                unsafe { &(*self.data.get()).len() },
            )
            .finish()
    }
}

unsafe impl<T: TensorScalar> BackendAllocation for HostAllocation<T> {
    fn root_extent(&self) -> RootResourceExtent {
        self.extent
    }

    fn provider_kind(&self) -> ProviderKind {
        ProviderKind::Cpu
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::host()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn map_read(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError> {
        if !supported_representation(T::dtype(), dtype) {
            return Err(AccessError::DTypeMismatch {
                expected: T::dtype(),
                actual: dtype,
            });
        }
        // SAFETY: shared provider mappings are retained only under a shared
        // group borrow; the vector is never resized after import.
        let data = unsafe { &*self.data.get() };
        let crate::StorageBuffer::Host(data) = data else {
            return Err(AccessError::Unsupported { backend: "host" });
        };
        let range = host_byte_range(
            self.extent,
            span,
            data.len(),
            std::mem::size_of::<T>(),
            dtype_size(dtype),
        )?;
        Ok(ProviderReadMapping::from_guard(HostByteReadGuard {
            data,
            range,
        }))
    }

    fn map_write(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        if !supported_representation(T::dtype(), dtype) {
            return Err(AccessError::DTypeMismatch {
                expected: T::dtype(),
                actual: dtype,
            });
        }
        // SAFETY: `GroupWriteView` provides the exclusive capability before
        // calling this provider hook; the vector is never resized after import.
        let data = unsafe { &mut *self.data.get() };
        let crate::StorageBuffer::Host(data) = data else {
            return Err(AccessError::Unsupported { backend: "host" });
        };
        let range = host_byte_range(
            self.extent,
            span,
            data.len(),
            std::mem::size_of::<T>(),
            dtype_size(dtype),
        )?;
        Ok(ProviderWriteMapping::from_guard(HostByteWriteGuard {
            data,
            range,
        }))
    }
}

impl<T: TensorScalar> HostAllocation<T> {
    fn host_slice_as<U: TensorScalar>(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<&[U], AccessError> {
        if dtype != U::dtype() || !supported_representation(T::dtype(), dtype) {
            return Err(AccessError::DTypeMismatch {
                expected: T::dtype(),
                actual: dtype,
            });
        }
        let data = unsafe { &*self.data.get() };
        let crate::StorageBuffer::Host(data) = data else {
            return Err(AccessError::Unsupported { backend: "host" });
        };
        let range = host_byte_range(
            self.extent,
            span,
            data.len(),
            size_of::<T>(),
            size_of::<U>(),
        )?;
        let bytes = host_bytes(data, range);
        if bytes.as_ptr().align_offset(align_of::<U>()) != 0 {
            return Err(AccessError::Provider {
                message: format!("host reinterpretation is not aligned for {:?}", dtype),
            });
        }
        let count = bytes.len() / size_of::<U>();
        // SAFETY: the sealed representation pair has equal size/alignment,
        // the byte range is element-aligned, and the host root borrow outlives
        // the returned slice.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<U>(), count) })
    }

    #[allow(clippy::mut_from_ref)]
    fn host_slice_as_mut<U: TensorScalar>(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<&mut [U], AccessError> {
        if dtype != U::dtype() || !supported_representation(T::dtype(), dtype) {
            return Err(AccessError::DTypeMismatch {
                expected: T::dtype(),
                actual: dtype,
            });
        }
        let data = unsafe { &mut *self.data.get() };
        let crate::StorageBuffer::Host(data) = data else {
            return Err(AccessError::Unsupported { backend: "host" });
        };
        let range = host_byte_range(
            self.extent,
            span,
            data.len(),
            size_of::<T>(),
            size_of::<U>(),
        )?;
        let bytes = host_bytes_mut(data, range);
        if bytes.as_ptr().align_offset(align_of::<U>()) != 0 {
            return Err(AccessError::Provider {
                message: format!("host reinterpretation is not aligned for {:?}", dtype),
            });
        }
        let count = bytes.len() / size_of::<U>();
        // SAFETY: the sealed representation pair has equal size/alignment,
        // the byte range is element-aligned, and the caller holds the unique
        // group mutable capability.
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<U>(), count) })
    }
}

fn supported_representation(source: DType, target: DType) -> bool {
    source == target
        || matches!(
            (source, target),
            (DType::C32, DType::F32)
                | (DType::F32, DType::C32)
                | (DType::C64, DType::F64)
                | (DType::F64, DType::C64)
        )
}

fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::I32 => size_of::<f32>(),
        DType::F64 | DType::I64 => size_of::<f64>(),
        DType::Bool => size_of::<bool>(),
        DType::C32 => size_of::<num_complex::Complex32>(),
        DType::C64 => size_of::<num_complex::Complex64>(),
    }
}

pub(crate) trait HostRoot: TensorScalar {
    fn into_pin(extent: RootResourceExtent, data: Vec<Self>) -> RootResourcePin;
}

fn cast_host_vec<T: TensorScalar, U: TensorScalar>(data: Vec<T>) -> Vec<U> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    let mut data = std::mem::ManuallyDrop::new(data);
    // SAFETY: TensorScalar is sealed to the seven scalar types below. The
    // matching dtype branch preserves size, alignment, and representation.
    unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<U>(), data.len(), data.capacity()) }
}

impl<T: TensorScalar> HostRoot for T {
    fn into_pin(extent: RootResourceExtent, data: Vec<Self>) -> RootResourcePin {
        match T::dtype() {
            DType::F32 => RootResourcePin::HostF32(HostAllocation {
                extent,
                data: UnsafeCell::new(crate::StorageBuffer::Host(cast_host_vec(data))),
            }),
            DType::F64 => RootResourcePin::HostF64(HostAllocation {
                extent,
                data: UnsafeCell::new(crate::StorageBuffer::Host(cast_host_vec(data))),
            }),
            DType::I32 => RootResourcePin::HostI32(HostAllocation {
                extent,
                data: UnsafeCell::new(crate::StorageBuffer::Host(cast_host_vec(data))),
            }),
            DType::I64 => RootResourcePin::HostI64(HostAllocation {
                extent,
                data: UnsafeCell::new(crate::StorageBuffer::Host(cast_host_vec(data))),
            }),
            DType::Bool => RootResourcePin::HostBool(HostAllocation {
                extent,
                data: UnsafeCell::new(crate::StorageBuffer::Host(cast_host_vec(data))),
            }),
            DType::C32 => RootResourcePin::HostC32(HostAllocation {
                extent,
                data: UnsafeCell::new(crate::StorageBuffer::Host(cast_host_vec(data))),
            }),
            DType::C64 => RootResourcePin::HostC64(HostAllocation {
                extent,
                data: UnsafeCell::new(crate::StorageBuffer::Host(cast_host_vec(data))),
            }),
        }
    }
}

impl RootResourcePin {
    fn host_buffer<T: 'static>(&self) -> Option<&crate::StorageBuffer<T>> {
        let allocation = self.as_any().downcast_ref::<HostAllocation<T>>()?;
        // SAFETY: the returned reference is bounded by the shared root borrow;
        // host buffers never resize after import.
        Some(unsafe { &*allocation.data.get() })
    }

    fn root_extent(&self) -> RootResourceExtent {
        match self {
            Self::HostF32(root) => root.root_extent(),
            Self::HostF64(root) => root.root_extent(),
            Self::HostI32(root) => root.root_extent(),
            Self::HostI64(root) => root.root_extent(),
            Self::HostBool(root) => root.root_extent(),
            Self::HostC32(root) => root.root_extent(),
            Self::HostC64(root) => root.root_extent(),
            Self::Backend(root) => root.root_extent(),
        }
    }

    fn provider_kind(&self) -> ProviderKind {
        match self {
            Self::HostF32(root) => root.provider_kind(),
            Self::HostF64(root) => root.provider_kind(),
            Self::HostI32(root) => root.provider_kind(),
            Self::HostI64(root) => root.provider_kind(),
            Self::HostBool(root) => root.provider_kind(),
            Self::HostC32(root) => root.provider_kind(),
            Self::HostC64(root) => root.provider_kind(),
            Self::Backend(root) => root.provider_kind(),
        }
    }

    fn as_any(&self) -> &dyn Any {
        match self {
            Self::HostF32(root) => root,
            Self::HostF64(root) => root,
            Self::HostI32(root) => root,
            Self::HostI64(root) => root,
            Self::HostBool(root) => root,
            Self::HostC32(root) => root,
            Self::HostC64(root) => root,
            Self::Backend(root) => root.as_any(),
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        match self {
            Self::HostF32(root) => root,
            Self::HostF64(root) => root,
            Self::HostI32(root) => root,
            Self::HostI64(root) => root,
            Self::HostBool(root) => root,
            Self::HostC32(root) => root,
            Self::HostC64(root) => root,
            Self::Backend(root) => root.as_any_mut(),
        }
    }

    fn map_read(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError> {
        match self {
            Self::HostF32(root) => root.map_read(span, dtype),
            Self::HostF64(root) => root.map_read(span, dtype),
            Self::HostI32(root) => root.map_read(span, dtype),
            Self::HostI64(root) => root.map_read(span, dtype),
            Self::HostBool(root) => root.map_read(span, dtype),
            Self::HostC32(root) => root.map_read(span, dtype),
            Self::HostC64(root) => root.map_read(span, dtype),
            Self::Backend(root) => root.map_read(span, dtype),
        }
    }

    fn map_write(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        match self {
            Self::HostF32(root) => root.map_write(span, dtype),
            Self::HostF64(root) => root.map_write(span, dtype),
            Self::HostI32(root) => root.map_write(span, dtype),
            Self::HostI64(root) => root.map_write(span, dtype),
            Self::HostBool(root) => root.map_write(span, dtype),
            Self::HostC32(root) => root.map_write(span, dtype),
            Self::HostC64(root) => root.map_write(span, dtype),
            Self::Backend(root) => root.map_write(span, dtype),
        }
    }

    fn host_slice<T: TensorScalar>(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<&[T], AccessError> {
        match self {
            Self::HostF32(root) => root.host_slice_as(span, dtype),
            Self::HostF64(root) => root.host_slice_as(span, dtype),
            Self::HostI32(root) => root.host_slice_as(span, dtype),
            Self::HostI64(root) => root.host_slice_as(span, dtype),
            Self::HostBool(root) => root.host_slice_as(span, dtype),
            Self::HostC32(root) => root.host_slice_as(span, dtype),
            Self::HostC64(root) => root.host_slice_as(span, dtype),
            Self::Backend(_) => Err(AccessError::Unsupported { backend: "backend" }),
        }
    }

    fn host_slice_mut<T: TensorScalar>(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<&mut [T], AccessError> {
        match self {
            Self::HostF32(root) => root.host_slice_as_mut(span, dtype),
            Self::HostF64(root) => root.host_slice_as_mut(span, dtype),
            Self::HostI32(root) => root.host_slice_as_mut(span, dtype),
            Self::HostI64(root) => root.host_slice_as_mut(span, dtype),
            Self::HostBool(root) => root.host_slice_as_mut(span, dtype),
            Self::HostC32(root) => root.host_slice_as_mut(span, dtype),
            Self::HostC64(root) => root.host_slice_as_mut(span, dtype),
            Self::Backend(_) => Err(AccessError::Unsupported { backend: "backend" }),
        }
    }
}

fn host_byte_range(
    extent: RootResourceExtent,
    span: RootBoundSpan,
    element_count: usize,
    allocation_element_size: usize,
    requested_element_size: usize,
) -> Result<Range<usize>, AccessError> {
    let start = span
        .byte_offset()
        .checked_sub(extent.byte_offset())
        .ok_or_else(|| AccessError::Provider {
            message: "host mapping span precedes allocation".to_string(),
        })?;
    let end = start
        .checked_add(span.byte_len())
        .ok_or_else(|| AccessError::Provider {
            message: "host mapping span overflows".to_string(),
        })?;
    let total = element_count
        .checked_mul(allocation_element_size)
        .ok_or_else(|| AccessError::Provider {
            message: "host allocation byte length overflows".to_string(),
        })?;
    if end > total
        || !start.is_multiple_of(requested_element_size)
        || !span.byte_len().is_multiple_of(requested_element_size)
    {
        return Err(AccessError::Provider {
            message: "host mapping span is not an element-aligned subrange".to_string(),
        });
    }
    Ok(start..end)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum RootImportError {
    #[error("root-resource identity validation failed: {0}")]
    Identity(#[source] RootResourceIdentityError),
}

/// Import one uniquely owned provider root and create its single root claim.
pub(crate) fn import_unique_root(
    allocation: Box<dyn BackendAllocation>,
) -> Result<OwnedStorage, Box<StorageOperationError<RootImportError>>> {
    let extent = allocation.root_extent();
    let requested = RequestedIdentity::Keyed {
        key: extent.key(),
        range: ByteRange::new(extent.byte_offset(), extent.byte_len()),
    };
    let context =
        StorageOperationContext::unresolved(StorageOperation::ImportUniqueRoot, requested);
    let identity = RootResourceIdentity::try_new(extent).map_err(|source| {
        Box::new(StorageOperationError::new(
            context,
            RootImportError::Identity(source),
        ))
    })?;
    let span = identity.root_span();
    Ok(OwnedStorage {
        pin: RootResourcePin::Backend(allocation),
        claim: OwnedSpanClaim {
            root: identity,
            span,
        },
    })
}

/// Import one host vector as the unique root for a typed tensor owner.
pub(crate) fn import_host_vec<T: TensorScalar>(
    data: Vec<T>,
) -> Result<OwnedStorage, Box<StorageOperationError<RootImportError>>> {
    static NEXT_HOST_ALLOCATION: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(1);
    let element_size = std::mem::size_of::<T>();
    let byte_len = data.len().checked_mul(element_size).ok_or_else(|| {
        Box::new(StorageOperationError::new(
            StorageOperationContext::unresolved(
                StorageOperation::ImportUniqueRoot,
                RequestedIdentity::Raw(ByteRange::new(0, usize::MAX)),
            ),
            RootImportError::Identity(RootResourceIdentityError::Extent(
                super::span::SpanValidationError::RangeOverflow {
                    byte_offset: 0,
                    byte_len: usize::MAX,
                },
            )),
        ))
    })?;
    let key = super::identity::AllocationKey::new(
        crate::AllocationDomainId::fresh(),
        crate::AllocationId::from_backend_id(
            NEXT_HOST_ALLOCATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ),
    );
    let extent = RootResourceExtent::try_new(key, 0, byte_len, std::mem::align_of::<T>()).map_err(
        |source| {
            Box::new(StorageOperationError::new(
                StorageOperationContext::unresolved(
                    StorageOperation::ImportUniqueRoot,
                    RequestedIdentity::Keyed {
                        key,
                        range: ByteRange::new(0, byte_len),
                    },
                ),
                RootImportError::Identity(RootResourceIdentityError::Extent(source)),
            ))
        },
    )?;
    let identity = RootResourceIdentity::try_new(extent).map_err(|source| {
        Box::new(StorageOperationError::new(
            StorageOperationContext::unresolved(
                StorageOperation::ImportUniqueRoot,
                RequestedIdentity::Keyed {
                    key,
                    range: ByteRange::new(0, byte_len),
                },
            ),
            RootImportError::Identity(source),
        ))
    })?;
    let span = identity.root_span();
    Ok(OwnedStorage {
        pin: T::into_pin(extent, data),
        claim: OwnedSpanClaim {
            root: identity,
            span,
        },
    })
}

impl OwnedStorage {
    pub(crate) const fn as_ref(&self) -> StorageRef<'_> {
        StorageRef { owner: self }
    }

    pub(crate) fn as_mut(&mut self) -> StorageMut<'_> {
        StorageMut { owner: self }
    }

    pub(crate) fn into_root_pin(self) -> RootResourcePin {
        self.pin
    }

    pub(crate) const fn root_span(&self) -> RootBoundSpan {
        self.claim.span
    }

    pub(crate) fn host_buffer<T: 'static>(&self) -> Option<&crate::StorageBuffer<T>> {
        self.pin.host_buffer::<T>()
    }

    pub(crate) fn into_host_vec<T: TensorScalar>(mut self) -> Result<Vec<T>, AccessError> {
        let allocation = self
            .pin
            .as_any_mut()
            .downcast_mut::<HostAllocation<T>>()
            .ok_or(AccessError::Unsupported {
                backend: "non-host",
            })?;
        // SAFETY: the move-only root pin proves there are no other owners, so
        // taking the vector cannot race with a provider mapping.
        let crate::StorageBuffer::Host(data) = (unsafe { &mut *allocation.data.get() }) else {
            return Err(AccessError::Unsupported {
                backend: "non-host",
            });
        };
        Ok(std::mem::take(data))
    }
}

impl<'a> StorageRef<'a> {
    pub(crate) const fn root_identity(&self) -> RootResourceIdentity {
        self.owner.claim.root
    }

    pub(crate) const fn span(&self) -> RootBoundSpan {
        self.owner.claim.span
    }

    pub(crate) fn provider_kind(&self) -> ProviderKind {
        self.owner.pin.provider_kind()
    }

    pub(super) fn map_read(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderReadMapping<'a>, AccessError> {
        let mapping = self.owner.pin.map_read(span, dtype)?;
        if mapping.bytes().len() != span.byte_len() {
            return Err(AccessError::LengthMismatch {
                expected: span.byte_len(),
                actual: mapping.bytes().len(),
            });
        }
        // SAFETY: the shared owner borrow lasts for `'a`; the mapping cannot
        // outlive the allocation it borrows, and no mutable access is exposed.
        Ok(unsafe {
            std::mem::transmute::<ProviderReadMapping<'_>, ProviderReadMapping<'a>>(mapping)
        })
    }

    pub(super) fn host_slice<T: TensorScalar>(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<&'a [T], AccessError> {
        let slice = self.owner.pin.host_slice::<T>(span, dtype)?;
        // SAFETY: the root owner is borrowed for `'a`; the host allocation is
        // never resized after import and the representation helper returned a
        // slice bounded by that root.
        Ok(unsafe { std::mem::transmute::<&[T], &'a [T]>(slice) })
    }
}

impl<'a> StorageMut<'a> {
    pub(crate) const fn root_identity(&self) -> RootResourceIdentity {
        self.owner.claim.root
    }

    pub(crate) const fn span(&self) -> RootBoundSpan {
        self.owner.claim.span
    }

    pub(super) fn map_write(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderWriteMapping<'a>, AccessError> {
        let mapping = self.owner.pin.map_write(span, dtype)?;
        if mapping.len() != span.byte_len() {
            let actual = mapping.len();
            return Err(AccessError::LengthMismatch {
                expected: span.byte_len(),
                actual,
            });
        }
        // SAFETY: the checked write owns the exclusive `'a` borrow represented
        // by this `StorageMut`; callers do not use that reference while the
        // returned mapping is alive.
        Ok(unsafe {
            std::mem::transmute::<ProviderWriteMapping<'_>, ProviderWriteMapping<'a>>(mapping)
        })
    }

    pub(super) fn host_slice_mut<T: TensorScalar>(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<&'a mut [T], AccessError> {
        let slice = self.owner.pin.host_slice_mut::<T>(span, dtype)?;
        // SAFETY: the caller holds the group's exclusive borrow and the
        // representation helper bounded the slice by the imported root.
        Ok(unsafe { std::mem::transmute::<&mut [T], &'a mut [T]>(slice) })
    }
}
