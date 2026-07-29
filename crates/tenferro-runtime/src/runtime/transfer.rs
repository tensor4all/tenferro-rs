use std::fmt;

use tenferro_tensor::{AllocationDomainId, DType, Placement, Tensor, TensorRead};

use super::schedule::{EventDomainId, ExecutionLocation};
use super::{EngineId, StorageClass};

/// Runtime-owned transfer provider between two execution locations.
///
/// Providers remain registered by source and destination storage class. Each
/// request also identifies the concrete engines and event domains at its
/// endpoints.
pub trait TransferProvider: fmt::Debug + Send + Sync + 'static {
    /// Complete one blocking transfer into the destination execution location.
    ///
    /// The returned tensor must be immediately readable by the destination
    /// executor. Providers must not return after merely enqueueing work on an
    /// asynchronous stream or queue. Native asynchronous transfers use the
    /// event-domain driver contract instead of this interface.
    ///
    /// # Errors
    ///
    /// Returns a [`tenferro_tensor::ErrorKind::RuntimeState`] or backend
    /// failure when the provider cannot materialize the destination tensor, or
    /// a validation error such as dtype, shape, placement, or buffer mismatch
    /// when the transfer request is unsupported.
    fn transfer_blocking(&self, request: TransferRequest<'_>) -> crate::Result<Tensor>;
}

/// Borrowed request passed to a [`TransferProvider`].
#[derive(Debug)]
pub struct TransferRequest<'a> {
    source_location: &'a ExecutionLocation,
    destination_location: &'a ExecutionLocation,
    input: TensorRead<'a>,
}

impl<'a> TransferRequest<'a> {
    pub(crate) fn new(
        source_location: &'a ExecutionLocation,
        destination_location: &'a ExecutionLocation,
        input: TensorRead<'a>,
    ) -> Self {
        Self {
            source_location,
            destination_location,
            input,
        }
    }

    /// Return the source engine for this transfer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, TransferRequest};
    ///
    /// # fn inspect(request: TransferRequest<'_>) {
    /// let source: &EngineId = request.source_engine_id();
    /// assert!(!source.as_str().is_empty());
    /// # }
    /// ```
    pub fn source_engine_id(&self) -> &'a EngineId {
        self.source_location.engine_id()
    }

    /// Return the source event domain for this transfer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EventDomainId, TransferRequest};
    ///
    /// # fn inspect(request: TransferRequest<'_>) {
    /// let _: EventDomainId = request.source_event_domain_id();
    /// # }
    /// ```
    pub fn source_event_domain_id(&self) -> EventDomainId {
        self.source_location.event_domain_id()
    }

    /// Return the source storage class for this transfer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{StorageClass, TransferRequest};
    ///
    /// # fn inspect(request: TransferRequest<'_>) {
    /// let source: &StorageClass = request.source_storage_class();
    /// assert!(!source.as_str().is_empty());
    /// # }
    /// ```
    pub fn source_storage_class(&self) -> &'a StorageClass {
        self.source_location.storage_class()
    }

    /// Return the destination engine for this transfer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, TransferRequest};
    ///
    /// # fn inspect(request: TransferRequest<'_>) {
    /// let destination: &EngineId = request.destination_engine_id();
    /// assert!(!destination.as_str().is_empty());
    /// # }
    /// ```
    pub fn destination_engine_id(&self) -> &'a EngineId {
        self.destination_location.engine_id()
    }

    /// Return the destination event domain for this transfer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EventDomainId, TransferRequest};
    ///
    /// # fn inspect(request: TransferRequest<'_>) {
    /// let _: EventDomainId = request.destination_event_domain_id();
    /// # }
    /// ```
    pub fn destination_event_domain_id(&self) -> EventDomainId {
        self.destination_location.event_domain_id()
    }

    /// Return the destination storage class for this transfer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{StorageClass, TransferRequest};
    ///
    /// # fn inspect(request: TransferRequest<'_>) {
    /// let destination: &StorageClass = request.destination_storage_class();
    /// assert!(!destination.as_str().is_empty());
    /// # }
    /// ```
    pub fn destination_storage_class(&self) -> &'a StorageClass {
        self.destination_location.storage_class()
    }

    /// Return the tensor read that must be transferred.
    pub fn input(&self) -> &TensorRead<'a> {
        &self.input
    }
}

/// Typed runtime transfer setup failure.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TransferError;
///
/// fn is_missing_provider(error: &TransferError) -> bool {
///     matches!(error, TransferError::MissingProvider { .. })
/// }
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TransferError {
    /// No provider was registered for the storage-class pair required by two
    /// concrete execution locations.
    #[error(
        "no transfer provider registered from {source_storage_class:?} on \
         {source_engine_id:?}/{source_event_domain_id:?} to \
         {destination_storage_class:?} on \
         {destination_engine_id:?}/{destination_event_domain_id:?}"
    )]
    MissingProvider {
        /// Source engine.
        source_engine_id: EngineId,
        /// Source event domain.
        source_event_domain_id: EventDomainId,
        /// Source storage class.
        source_storage_class: StorageClass,
        /// Destination engine.
        destination_engine_id: EngineId,
        /// Destination event domain.
        destination_event_domain_id: EventDomainId,
        /// Destination storage class.
        destination_storage_class: StorageClass,
    },
    /// A provider returned a tensor that violates the transfer request contract.
    #[error("transfer provider returned an invalid tensor")]
    ProviderContract {
        /// Typed provider-contract violation.
        #[source]
        source: TransferProviderContractError,
    },
}

/// Typed contract violation in a tensor returned by a transfer provider.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{DType, TransferProviderContractError};
///
/// let error = TransferProviderContractError::DTypeMismatch {
///     expected: DType::F64,
///     actual: DType::F32,
/// };
/// assert!(error.to_string().contains("dtype"));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TransferProviderContractError {
    /// The logical source shape cannot be represented as an element count.
    #[error("transfer source logical element count is invalid")]
    LogicalElementCount {
        /// Checked tensor shape-product failure.
        #[source]
        source: tenferro_tensor::Error,
    },
    /// The returned tensor changed the source dtype.
    #[error("transfer output dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch {
        /// Source dtype required by the request.
        expected: DType,
        /// Dtype returned by the provider.
        actual: DType,
    },
    /// The returned tensor changed the source shape.
    #[error("transfer output shape mismatch: expected {expected:?}, actual {actual:?}")]
    ShapeMismatch {
        /// Source shape required by the request.
        expected: Vec<usize>,
        /// Shape returned by the provider.
        actual: Vec<usize>,
    },
    /// The returned tensor placement is not accepted at the destination endpoint.
    #[error(
        "transfer output placement {actual:?} is incompatible with destination storage \
         {destination_storage_class:?} on engine {destination_engine_id:?}"
    )]
    DestinationPlacementMismatch {
        /// Destination engine from the transfer request.
        destination_engine_id: EngineId,
        /// Destination storage class from the transfer request.
        destination_storage_class: StorageClass,
        /// Placement returned by the provider.
        actual: Placement,
    },
    /// The returned tensor has compatible metadata but is not owned by the
    /// destination engine's backend or allocation domain.
    #[error(
        "transfer output storage family {actual_backend_family:?} and allocation domain \
         {actual_allocation_domain:?} are not resident in destination storage \
         {destination_storage_class:?} on engine {destination_engine_id:?}"
    )]
    DestinationResidencyMismatch {
        /// Destination engine from the transfer request.
        destination_engine_id: EngineId,
        /// Destination storage class from the transfer request.
        destination_storage_class: StorageClass,
        /// Physical backend family returned by the provider.
        actual_backend_family: Option<&'static str>,
        /// Shared allocation domain returned by the provider.
        actual_allocation_domain: Option<AllocationDomainId>,
    },
    /// The returned tensor's buffer length does not match its shape.
    #[error(
        "transfer output buffer length mismatch: shape requires {expected} elements, \
         buffer reports {actual}"
    )]
    InvalidBufferLength {
        /// Element count required by the returned shape.
        expected: usize,
        /// Element count reported by the returned buffer.
        actual: usize,
    },
}
