//! Host tensors whose element type is known only at run time.
//!
//! A closed variant list cannot carry a scalar the crate does not declare, so a
//! set with an externally defined member cannot use one. [`ErasedHostTensor`]
//! carries the element type inside the value instead: the payload keeps its own
//! concrete type and is recovered by identity, with no reinterpretation of bytes
//! and no assumption that the element type belongs to any particular set.

use core::any::{Any, TypeId};

use crate::{HostTensor, Scalar, ShapeVec};

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
/// Object-safe clone for one erased payload.
///
/// `Box<dyn Any>` is not cloneable, so the payload carries its own duplication
/// entry point. A payload is only ever duplicated through this, which keeps the
/// concrete element type and therefore the representation intact.
trait ClonePayload: Send + Sync {
    fn clone_payload(&self) -> Box<dyn ClonePayload>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_payload(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T: Clone + Send + Sync + 'static> ClonePayload for T {
    fn clone_payload(&self) -> Box<dyn ClonePayload> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_payload(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// A host tensor whose element type is recovered at run time.
///
/// The payload keeps its own concrete type and is recovered by identity, so no
/// byte reinterpretation happens and a caller-owned payload is duplicated through
/// its own entry point rather than by copying bytes.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
///
/// let value = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?);
/// assert_eq!(value.downcast_ref::<f64>().unwrap().as_slice(), &[1.0, 2.0]);
/// assert_eq!(value.clone().element_count(), 2);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
pub struct ErasedHostTensor {
    payload: Box<dyn ClonePayload>,
    type_id: TypeId,
    element: TypeId,
    shape: ShapeVec,
    elements: usize,
}

impl Clone for ErasedHostTensor {
    /// Duplicate the payload while keeping its concrete element type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let value = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![7_i64])?);
    /// let copy = value.clone();
    /// assert_eq!(copy.downcast_ref::<i64>().unwrap().as_slice(), &[7]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    fn clone(&self) -> Self {
        Self {
            payload: self.payload.clone_payload(),
            type_id: self.type_id,
            element: self.element,
            shape: self.shape.clone(),
            elements: self.elements,
        }
    }
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
        let shape = value.shape().into();
        let elements = value.as_slice().len();
        Self {
            payload: Box::new(value),
            type_id: TypeId::of::<HostTensor<T>>(),
            element: TypeId::of::<T>(),
            shape,
            elements,
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

    /// Identity of the stored element type, without the tensor wrapper.
    ///
    /// This is what a runtime tag reports for an externally defined scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![1.0_f64])?);
    /// assert_eq!(erased.element_type_id(), core::any::TypeId::of::<f64>());
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn element_type_id(&self) -> TypeId {
        self.element
    }

    /// Shape of the stored tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?);
    /// assert_eq!(erased.shape(), &[2, 3]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of stored elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?);
    /// assert_eq!(erased.element_count(), 6);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements
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
        self.payload.as_any().is::<HostTensor<T>>()
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
        self.payload.as_any().downcast_ref::<HostTensor<T>>()
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
        self.payload.as_any_mut().downcast_mut::<HostTensor<T>>()
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
            .into_payload()
            .downcast::<HostTensor<T>>()
            .ok()
            .map(|boxed| *boxed)
    }
}

#[cfg(test)]
mod tests;
