// INVARIANT: these private P2 contracts are staged vocabulary; Task 3 will
// consume them after its phase gate, so dead-code linting is intentionally
// scoped to these modules rather than hidden behind compatibility machinery.
#[allow(dead_code)]
mod diagnostics;
#[allow(dead_code)]
mod group;
#[allow(dead_code)]
mod identity;
#[allow(dead_code)]
mod prepared;
#[allow(dead_code)]
mod retirement;
#[allow(dead_code)]
mod root;
#[allow(dead_code)]
mod span;

pub(crate) use group::{
    AllocationGroup, DescriptorSlot, GroupError, GroupReadView, GroupWriteView,
};

#[cfg(test)]
pub(crate) use diagnostics::{
    RequestedIdentity, StorageOperation, StorageOperationContext, StorageOperationError,
};
#[cfg(test)]
pub(crate) use identity::{AllocationKey, RootResourceIdentity};
#[cfg(test)]
pub(crate) use root::{import_unique_root, BackendAllocation, ProviderCapabilities, ProviderKind};
#[cfg(test)]
pub(crate) use span::{ByteRange, RootBoundSpan, RootResourceExtent, SpanValidationError};

#[cfg(test)]
mod tests;
