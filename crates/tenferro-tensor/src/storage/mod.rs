// INVARIANT: storage validation and provider-lifecycle helpers are exercised
// through feature-gated provider bindings and in-crate contract tests.
#![allow(dead_code)]

mod diagnostics;
mod group;
mod identity;
mod prepared;
mod root;
mod span;

pub use group::{AllocationGroup, DescriptorSlot, GroupError};
pub(crate) use group::{GroupReadView, GroupWriteView};

#[doc(hidden)]
pub use identity::{AllocationKey, RootResourceId};
#[doc(hidden)]
pub use prepared::{AccessError, ProviderReadMapping, ProviderWriteMapping};
#[doc(hidden)]
pub use root::{BackendAllocation, ProviderCapabilities, ProviderKind};
#[doc(hidden)]
pub use span::{RootBoundSpan, RootResourceExtent, SpanValidationError};

#[cfg(test)]
pub(crate) use diagnostics::{
    RequestedIdentity, StorageOperation, StorageOperationContext, StorageOperationError,
};
#[cfg(test)]
pub(crate) use identity::RootResourceIdentity;
#[cfg(test)]
pub(crate) use root::import_unique_root;
#[cfg(test)]
pub(crate) use span::ByteRange;

#[cfg(test)]
mod tests;
