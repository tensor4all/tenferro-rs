use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{AllocationDomainId, AllocationId};

use super::{ByteRange, RootBoundSpan, RootResourceExtent, SpanValidationError};

/// Domain-qualified diagnostic identity. It is not an ownership proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct AllocationKey {
    domain: AllocationDomainId,
    local: AllocationId,
}

impl AllocationKey {
    pub(crate) const fn new(domain: AllocationDomainId, local: AllocationId) -> Self {
        Self { domain, local }
    }

    pub(crate) const fn domain(self) -> AllocationDomainId {
        self.domain
    }

    pub(crate) const fn local(self) -> AllocationId {
        self.local
    }
}

/// Private non-reused provenance for one provider root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RootResourceId(NonZeroU64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RootResourceIdError {
    #[error("root-resource identity space exhausted")]
    Exhausted,
}

impl RootResourceId {
    fn fresh() -> Result<Self, RootResourceIdError> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        let value = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                if current == 0 {
                    None
                } else {
                    Some(if current == u64::MAX { 0 } else { current + 1 })
                }
            })
            .map_err(|_| RootResourceIdError::Exhausted)?;
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(RootResourceIdError::Exhausted)
    }

    pub(crate) const fn get(self) -> NonZeroU64 {
        self.0
    }
}

/// Root provenance paired with its exact checked extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RootResourceIdentity {
    root_resource: RootResourceId,
    extent: RootResourceExtent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RootResourceIdentityError {
    #[error("root-resource extent is invalid: {0}")]
    Extent(#[source] SpanValidationError),
    #[error("root-resource identity space is exhausted: {0}")]
    Id(#[source] RootResourceIdError),
}

impl RootResourceIdentity {
    pub(crate) fn try_new(extent: RootResourceExtent) -> Result<Self, RootResourceIdentityError> {
        extent
            .validate()
            .map_err(RootResourceIdentityError::Extent)?;
        let root_resource = RootResourceId::fresh().map_err(RootResourceIdentityError::Id)?;
        Ok(Self {
            root_resource,
            extent,
        })
    }

    pub(crate) const fn root_resource(self) -> RootResourceId {
        self.root_resource
    }

    pub(crate) const fn extent(self) -> RootResourceExtent {
        self.extent
    }

    pub(crate) fn bind_relative_range(
        self,
        relative: ByteRange,
    ) -> Result<RootBoundSpan, SpanValidationError> {
        let (byte_offset, byte_len, alignment) = self.extent.relative_parts(relative)?;
        Ok(RootBoundSpan::from_parts(
            self,
            byte_offset,
            byte_len,
            alignment,
        ))
    }

    pub(crate) fn validate_bound_span(
        self,
        span: &RootBoundSpan,
    ) -> Result<(), SpanValidationError> {
        let actual = span.root_identity().root_resource();
        if actual != self.root_resource {
            return Err(SpanValidationError::DifferentRoot {
                expected: self.root_resource,
                actual,
            });
        }
        Ok(())
    }
}
