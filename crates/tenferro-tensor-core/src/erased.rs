//! Host tensors whose element type is known only at run time.
//!
//! A closed variant list cannot carry a scalar the crate does not declare, so a
//! set with an externally defined member cannot use one. [`ErasedHostTensor`]
//! carries the element type inside the value instead: the payload keeps its own
//! concrete type and is recovered by identity, with no reinterpretation of bytes
//! and no assumption that the element type belongs to any particular set.

use core::any::{Any, TypeId};

use crate::{HostTensor, Scalar};

/// A host tensor whose element type is recovered at run time.
///
/// The payload is stored as its own concrete type. Type identity is the actual
/// Rust type, so equality of a tag, of a size, or of an alignment is never
/// treated as identity, and no byte reinterpretation happens.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
///
/// let erased = ErasedHostTensor::new(
///     HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?,
/// );
/// assert!(erased.is::<f64>());
/// assert!(!erased.is::<f32>());
/// assert_eq!(erased.downcast_ref::<f64>().unwrap().as_slice(), &[1.0, 2.0]);
/// assert!(erased.downcast_ref::<f32>().is_none());
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
pub struct ErasedHostTensor {
    payload: Box<dyn Any + Send + Sync>,
    type_id: TypeId,
}

impl core::fmt::Debug for ErasedHostTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ErasedHostTensor")
            .field("type_id", &self.type_id)
            .finish()
    }
}

impl ErasedHostTensor {
    /// Erase a host tensor's element type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![7_i32])?);
    /// assert!(erased.is::<i32>());
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub fn new<T: Scalar>(value: HostTensor<T>) -> Self {
        Self {
            payload: Box::new(value),
            type_id: TypeId::of::<HostTensor<T>>(),
        }
    }

    /// Identity of the stored element type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![1.0_f32])?);
    /// assert_eq!(erased.type_id(), core::any::TypeId::of::<HostTensor<f32>>());
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Whether the stored tensor has element type `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![1.0_f64])?);
    /// assert!(erased.is::<f64>());
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn is<T: Scalar>(&self) -> bool {
        self.payload.is::<HostTensor<T>>()
    }

    /// Borrow the tensor if the stored element type is `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![4.0_f64])?);
    /// assert_eq!(erased.downcast_ref::<f64>().unwrap().shape(), &[1]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn downcast_ref<T: Scalar>(&self) -> Option<&HostTensor<T>> {
        self.payload.downcast_ref::<HostTensor<T>>()
    }

    /// Mutably borrow the tensor if the stored element type is `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let mut erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![1_i64])?);
    /// erased.downcast_mut::<i64>().unwrap().as_mut_slice()[0] = 9;
    /// assert_eq!(erased.downcast_ref::<i64>().unwrap().as_slice(), &[9]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub fn downcast_mut<T: Scalar>(&mut self) -> Option<&mut HostTensor<T>> {
        self.payload.downcast_mut::<HostTensor<T>>()
    }

    /// Take the tensor if the stored element type is `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![2.0_f64])?);
    /// assert_eq!(erased.into_typed::<f64>().unwrap().as_slice(), &[2.0]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn into_typed<T: Scalar>(self) -> Option<HostTensor<T>> {
        self.payload
            .downcast::<HostTensor<T>>()
            .ok()
            .map(|boxed| *boxed)
    }
}

#[cfg(test)]
mod tests;
