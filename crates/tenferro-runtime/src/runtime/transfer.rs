use std::fmt;

use tenferro_tensor::{Tensor, TensorRead};

use super::schedule::{EventDomainId, ExecutionLocation};
use super::{EngineId, StorageClass};

/// Runtime-owned transfer provider between two execution locations.
///
/// Providers remain registered by source and destination storage class. Each
/// request also identifies the concrete engines and event domains at its
/// endpoints.
pub trait TransferProvider: fmt::Debug + Send + Sync + 'static {
    /// Transfer one tensor read into the destination execution location.
    ///
    /// # Errors
    ///
    /// Returns a [`tenferro_tensor::ErrorKind::RuntimeState`] or backend
    /// failure when the provider cannot materialize the destination tensor, or
    /// a validation error such as dtype, shape, placement, or buffer mismatch
    /// when the transfer request is unsupported.
    fn transfer(&self, request: TransferRequest<'_>) -> crate::Result<Tensor>;
}

/// Borrowed request passed to a [`TransferProvider`].
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
}
