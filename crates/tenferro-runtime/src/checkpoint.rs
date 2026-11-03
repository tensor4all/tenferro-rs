use std::fmt;
use std::sync::Arc;

use tenferro_tensor::{AllocationGroup, DType, DescriptorSlot, Tensor, TensorRead, TensorValue};

use crate::error::{Error, ErrorPhase, Result};

/// A read-only retained tensor handle backed by one allocation group owner.
///
/// Cloning the handle retains the group container and descriptor metadata; it
/// never clones a physical allocation. Creating another physical value remains
/// an explicit tensor operation at the caller's boundary.
#[derive(Clone)]
pub struct RetainedValue {
    container: Arc<RetentionContainer>,
    dtype: DType,
    shape: Box<[usize]>,
}

/// What one retained handle holds.
// The pooled variant owns an allocation group inline; boxing it would add an
// allocation to every retained value on the hot path.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum RetentionContainer {
    /// Pooled storage owned through an allocation group and one descriptor slot.
    Pooled {
        group: AllocationGroup,
        slot: DescriptorSlot,
    },
    /// A caller-owned value the runtime retains without taking pool ownership.
    ///
    /// The value returns to its owner when the handle drops, and the runtime never
    /// substitutes it for pooled storage.
    CallerOwned {
        /// Boxed because a tensor value is much larger than the pooled variant.
        tensor: Box<Tensor>,
    },
}

impl fmt::Debug for RetainedValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedValue")
            .field("container", &self.container)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl RetainedValue {
    /// Move an owned tensor value into a retained group-backed handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the value's descriptor cannot be
    /// transferred into a retention allocation group.
    pub fn from_tensor_value(value: TensorValue) -> Result<Self> {
        match value.try_into_group_parts() {
            Ok((group, slot, dtype, shape)) => Ok(Self {
                container: Arc::new(RetentionContainer::Pooled { group, slot }),
                dtype,
                shape: shape.into_boxed_slice(),
            }),
            Err(value) => {
                let tensor = value.into_tensor().map_err(|_| {
                    Error::runtime_state(
                        "RetainedValue::from_tensor_value",
                        ErrorPhase::Execution,
                        "a TensorValue could not be transferred into its retention group",
                    )
                })?;
                if !matches!(tensor.dtype(), DType::External(_)) {
                    return Err(Error::runtime_state(
                        "RetainedValue::from_tensor_value",
                        ErrorPhase::Execution,
                        "a TensorValue could not be transferred into its retention group",
                    ));
                }
                // A caller-owned payload is retained directly: it owns no pooled
                // storage, so there is no group and nothing returns to a pool when
                // the handle drops.
                let dtype = tensor.dtype();
                let shape = tensor.shape().to_vec();
                Ok(Self {
                    container: Arc::new(RetentionContainer::CallerOwned {
                        tensor: Box::new(tensor),
                    }),
                    dtype,
                    shape: shape.into_boxed_slice(),
                })
            }
        }
    }

    /// Move an owned compact tensor into a retained group-backed handle.
    ///
    /// # Panics
    ///
    /// Panics only if the validated compact tensor cannot produce its valid
    /// allocation-group descriptor, which indicates an internal invariant
    /// violation.
    pub fn from_tensor(tensor: Tensor) -> Self {
        // A compact tensor always has a valid descriptor, so this conversion
        // cannot fail after the tensor owner has been constructed.
        Self::from_tensor_value(TensorValue::from_tensor(tensor))
            .expect("compact tensor retention must have a valid descriptor")
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the retained descriptor for one read-only execution boundary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained descriptor is vacant,
    /// out of bounds, or otherwise invalid in its allocation group.
    pub fn tensor_read(&self) -> Result<TensorRead<'_>> {
        match self.container.as_ref() {
            RetentionContainer::Pooled { group, slot } => group.read_view(*slot).map_err(|error| {
                Error::runtime_state(
                    "RetainedValue::tensor_read",
                    ErrorPhase::Execution,
                    error.to_string(),
                )
            }),
            RetentionContainer::CallerOwned { tensor } => Ok(TensorRead::from_tensor(tensor)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RetainedValue;
    use tenferro_tensor::{Tensor, TensorRead};

    #[test]
    fn a_retained_value_debugs_its_metadata_and_not_its_payload() {
        let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let retained = RetainedValue::from_tensor(tensor);

        assert_eq!(retained.shape(), &[2]);
        let text = format!("{retained:?}");
        assert!(text.contains("RetainedValue"), "{text}");
        assert!(text.contains("Pooled"), "{text}");
        assert!(
            !text.contains("1.0"),
            "the retained payload stays out of Debug: {text}"
        );

        let read: TensorRead<'_> = retained.tensor_read().unwrap();
        assert_eq!(read.shape(), &[2]);
    }
}
