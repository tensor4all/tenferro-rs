use std::fmt;

use super::{AllocationKey, ByteRange, RootBoundSpan, RootResourceId};

/// Explicit request identity used by operation diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RequestedIdentity {
    Raw(ByteRange),
    Keyed {
        key: AllocationKey,
        range: ByteRange,
    },
    Rooted {
        root: RootResourceId,
        key: AllocationKey,
        range: ByteRange,
    },
}

impl RequestedIdentity {
    pub(crate) const fn range(self) -> ByteRange {
        match self {
            Self::Raw(range) | Self::Keyed { range, .. } | Self::Rooted { range, .. } => range,
        }
    }

    pub(crate) const fn allocation_key(self) -> Option<AllocationKey> {
        match self {
            Self::Raw(_) => None,
            Self::Keyed { key, .. } | Self::Rooted { key, .. } => Some(key),
        }
    }

    pub(crate) const fn root_resource(self) -> Option<RootResourceId> {
        match self {
            Self::Raw(_) | Self::Keyed { .. } => None,
            Self::Rooted { root, .. } => Some(root),
        }
    }
}

/// Operation boundary represented in the current private storage core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StorageOperation {
    ImportUniqueRoot,
    ClaimSplit,
}

impl fmt::Display for StorageOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImportUniqueRoot => formatter.write_str("import_unique_root"),
            Self::ClaimSplit => formatter.write_str("claim_split"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum OperationResolution {
    Unresolved,
    Resolved(RootBoundSpan),
}

/// Operation metadata shared by future import and claim errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StorageOperationContext {
    operation: StorageOperation,
    requested: RequestedIdentity,
    resolution: OperationResolution,
}

impl StorageOperationContext {
    pub(crate) const fn unresolved(
        operation: StorageOperation,
        requested: RequestedIdentity,
    ) -> Self {
        Self {
            operation,
            requested,
            resolution: OperationResolution::Unresolved,
        }
    }

    pub(crate) const fn resolved(
        operation: StorageOperation,
        requested: RequestedIdentity,
        span: RootBoundSpan,
    ) -> Self {
        Self {
            operation,
            requested,
            resolution: OperationResolution::Resolved(span),
        }
    }

    pub(crate) const fn operation(self) -> StorageOperation {
        self.operation
    }

    pub(crate) const fn requested(self) -> RequestedIdentity {
        self.requested
    }

    pub(crate) const fn resolved_span(self) -> Option<RootBoundSpan> {
        match self.resolution {
            OperationResolution::Unresolved => None,
            OperationResolution::Resolved(span) => Some(span),
        }
    }
}

impl fmt::Display for StorageOperationContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.resolution {
            OperationResolution::Unresolved => {
                write!(
                    formatter,
                    "{} for {:?}, unresolved",
                    self.operation, self.requested
                )
            }
            OperationResolution::Resolved(span) => {
                write!(
                    formatter,
                    "{} for {:?}, resolved {:?}",
                    self.operation, self.requested, span
                )
            }
        }
    }
}

/// Typed operation envelope retaining the original storage cause.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{context}: {source}")]
pub(crate) struct StorageOperationError<E>
where
    E: std::error::Error + 'static,
{
    context: StorageOperationContext,
    #[source]
    source: E,
}

impl<E> StorageOperationError<E>
where
    E: std::error::Error + 'static,
{
    pub(crate) const fn new(context: StorageOperationContext, source: E) -> Self {
        Self { context, source }
    }

    pub(crate) const fn context(&self) -> StorageOperationContext {
        self.context
    }

    pub(crate) const fn source(&self) -> &E {
        &self.source
    }
}
