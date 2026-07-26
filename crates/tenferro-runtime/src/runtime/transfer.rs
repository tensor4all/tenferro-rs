use std::fmt;

use tenferro_tensor::{Tensor, TensorRead};

use super::StorageClass;

/// Runtime-owned transfer provider between two registered storage classes.
pub trait TransferProvider: fmt::Debug + Send + Sync + 'static {
    /// Transfer one tensor read into the destination storage class.
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
    source_storage_class: &'a StorageClass,
    destination_storage_class: &'a StorageClass,
    input: TensorRead<'a>,
}

impl<'a> TransferRequest<'a> {
    pub(crate) fn new(
        source_storage_class: &'a StorageClass,
        destination_storage_class: &'a StorageClass,
        input: TensorRead<'a>,
    ) -> Self {
        Self {
            source_storage_class,
            destination_storage_class,
            input,
        }
    }

    /// Return the source storage class for this transfer.
    pub fn source_storage_class(&self) -> &'a StorageClass {
        self.source_storage_class
    }

    /// Return the destination storage class for this transfer.
    pub fn destination_storage_class(&self) -> &'a StorageClass {
        self.destination_storage_class
    }

    /// Return the tensor read that must be transferred.
    pub fn input(&self) -> &TensorRead<'a> {
        &self.input
    }
}
