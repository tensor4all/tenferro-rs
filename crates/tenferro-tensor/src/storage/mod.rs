// INVARIANT: these private P2 contracts are staged vocabulary; Task 3 will
// consume them after its phase gate, so dead-code linting is intentionally
// scoped to these modules rather than hidden behind compatibility machinery.
#[allow(dead_code)]
mod diagnostics;
#[allow(dead_code)]
mod identity;
#[allow(dead_code)]
mod span;

#[cfg(test)]
pub(crate) use diagnostics::{
    RequestedIdentity, StorageOperation, StorageOperationContext, StorageOperationError,
};
#[cfg(test)]
pub(crate) use identity::{AllocationKey, RootResourceIdentity};
#[cfg(test)]
pub(crate) use span::{ByteRange, RootBoundSpan, RootResourceExtent, SpanValidationError};

#[cfg(test)]
mod tests;
