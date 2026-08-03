use std::any::Any;
use std::ops::{Deref, DerefMut, Range};
use std::sync::Arc;
use std::sync::Mutex;

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

/// The physical root and its provider allocation. It is held only by the
/// lifetime pin; it grants no access capability by itself.
pub(crate) struct RootResource {
    identity: RootResourceIdentity,
    extent: RootResourceExtent,
    allocation: Box<dyn BackendAllocation>,
}

/// Lifetime-only pin for one root resource. It is intentionally not `Clone`.
pub(crate) struct RootResourcePin(Arc<RootResource>);

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
struct HostAllocation<T: TensorScalar> {
    extent: RootResourceExtent,
    data: Mutex<Vec<T>>,
}

struct HostByteReadGuard<'a, T: TensorScalar> {
    guard: std::sync::MutexGuard<'a, Vec<T>>,
    range: Range<usize>,
}

struct HostByteWriteGuard<'a, T: TensorScalar> {
    guard: std::sync::MutexGuard<'a, Vec<T>>,
    range: Range<usize>,
}

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
        host_bytes(&self.guard, self.range.clone())
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
        host_bytes(&self.guard, self.range.clone())
    }
}

impl<T: TensorScalar> DerefMut for HostByteWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        host_bytes_mut(&mut self.guard, self.range.clone())
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
                &self.data.lock().map_or(0, |data| data.len()),
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

    fn map_read(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError> {
        if dtype != T::dtype() {
            return Err(AccessError::DTypeMismatch {
                expected: T::dtype(),
                actual: dtype,
            });
        }
        let data = self.data.lock().map_err(|_| AccessError::Provider {
            message: "host allocation mapping lock poisoned".to_string(),
        })?;
        let range = host_byte_range(self.extent, span, data.len(), std::mem::size_of::<T>())?;
        Ok(ProviderReadMapping::from_guard(HostByteReadGuard {
            guard: data,
            range,
        }))
    }

    fn map_write(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError> {
        if dtype != T::dtype() {
            return Err(AccessError::DTypeMismatch {
                expected: T::dtype(),
                actual: dtype,
            });
        }
        let data = self.data.lock().map_err(|_| AccessError::Provider {
            message: "host allocation mapping lock poisoned".to_string(),
        })?;
        let range = host_byte_range(self.extent, span, data.len(), std::mem::size_of::<T>())?;
        Ok(ProviderWriteMapping::from_guard(HostByteWriteGuard {
            guard: data,
            range,
        }))
    }
}

fn host_byte_range(
    extent: RootResourceExtent,
    span: RootBoundSpan,
    element_count: usize,
    element_size: usize,
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
        .checked_mul(element_size)
        .ok_or_else(|| AccessError::Provider {
            message: "host allocation byte length overflows".to_string(),
        })?;
    if end > total
        || !start.is_multiple_of(element_size)
        || !span.byte_len().is_multiple_of(element_size)
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
    let resource = Arc::new(RootResource {
        identity,
        extent,
        allocation,
    });

    Ok(OwnedStorage {
        pin: RootResourcePin(resource),
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
    import_unique_root(Box::new(HostAllocation {
        extent,
        data: Mutex::new(data),
    }))
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
}

impl<'a> StorageRef<'a> {
    pub(crate) const fn root_identity(&self) -> RootResourceIdentity {
        self.owner.claim.root
    }

    pub(crate) const fn span(&self) -> RootBoundSpan {
        self.owner.claim.span
    }

    pub(crate) fn provider_kind(&self) -> ProviderKind {
        self.owner.pin.0.allocation.provider_kind()
    }

    pub(super) fn map_read(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderReadMapping<'a>, AccessError> {
        let mapping = self.owner.pin.0.allocation.map_read(span, dtype)?;
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
        let mapping = self.owner.pin.0.allocation.map_write(span, dtype)?;
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
}
