mod diagnostics;
mod identity;
mod span;

pub(crate) use diagnostics::{
    RequestedIdentity, StorageOperation, StorageOperationContext, StorageOperationError,
};
pub(crate) use identity::{AllocationKey, RootResourceId, RootResourceIdentity};
pub(crate) use span::{ByteRange, RootBoundSpan, RootResourceExtent, SpanValidationError};

#[cfg(test)]
mod tests;
