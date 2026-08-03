use std::any::Any;
use std::sync::Arc;

use crate::BackendId;
use crate::DType;

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
}

/// The sole unsafe boundary for importing a uniquely owned provider root.
///
/// # Safety
///
/// Implementors must own exactly one provider allocation and its destructor,
/// report stable truthful metadata, and uphold `Send`/`Sync` for the boxed
/// value. The storage core does not recover from a violated provider contract.
// INVARIANT: #1558 P2 names this private storage module as the sole provider
// boundary; higher-layer tensor graph and AD paths remain unsafe-free.
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
}

impl<'a> StorageRef<'a> {
    pub(crate) const fn root_identity(&self) -> RootResourceIdentity {
        self.owner.claim.root
    }

    pub(crate) const fn span(&self) -> RootBoundSpan {
        self.owner.claim.span
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
