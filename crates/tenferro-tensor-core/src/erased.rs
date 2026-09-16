//! Host tensors whose element type is known only at run time.
//!
//! A closed variant list cannot carry a scalar the crate does not declare, so a
//! set with an externally defined member cannot use one. [`ErasedHostTensor`]
//! carries the element type inside the value instead: the payload keeps its own
//! concrete type and is recovered by identity, with no reinterpretation of bytes
//! and no assumption that the element type belongs to any particular set.
//!
//! The erased value also carries the layout of the view it presents, so a
//! metadata-only permutation and an explicit contiguous materialization exist
//! without a variant in the typed view types. The payload is shared through an
//! [`Arc`], so a view is one reference count and `duplicate` is the explicit copy.

use core::any::{Any, TypeId};
use std::sync::Arc;

use crate::{HostTensor, Scalar, ShapeVec, ValidationError};

/// Shape, element strides, and element offset of one erased view.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    shape: ShapeVec,
    strides: Vec<isize>,
    offset: isize,
}

impl Layout {
    /// The dense column-major layout of `shape`.
    fn dense(shape: &[usize]) -> Self {
        let mut strides = Vec::with_capacity(shape.len());
        let mut running = 1isize;
        for extent in shape {
            strides.push(running);
            running = running.saturating_mul(*extent as isize);
        }
        Self {
            shape: shape.into(),
            strides,
            offset: 0,
        }
    }

    /// Number of logical elements.
    fn count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether this layout is the dense column-major layout of its own shape.
    fn is_dense(&self) -> bool {
        *self == Self::dense(&self.shape)
    }

    /// The layout of the same elements under `axes`.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::InvalidPermutationLength`] when `axes` has the
    /// wrong length, [`ValidationError::AxisOutOfBounds`] when an axis is out of
    /// range, and [`ValidationError::DuplicateAxis`] when an axis repeats.
    fn permuted(&self, axes: &[usize]) -> Result<Self, ValidationError> {
        let rank = self.shape.len();
        if axes.len() != rank {
            return Err(ValidationError::InvalidPermutationLength {
                expected: rank,
                actual: axes.len(),
            });
        }
        let mut seen = vec![false; rank];
        for axis in axes {
            if *axis >= rank {
                return Err(ValidationError::AxisOutOfBounds { axis: *axis, rank });
            }
            if seen[*axis] {
                return Err(ValidationError::DuplicateAxis {
                    axis: *axis,
                    role: "permute",
                });
            }
            seen[*axis] = true;
        }
        Ok(Self {
            shape: axes.iter().map(|axis| self.shape[*axis]).collect(),
            strides: axes.iter().map(|axis| self.strides[*axis]).collect(),
            offset: self.offset,
        })
    }

    /// The linear storage index of one logical index, when it is in range.
    fn linear_index(&self, index: &[usize]) -> Option<isize> {
        if index.len() != self.shape.len() {
            return None;
        }
        let mut linear = self.offset;
        for (position, extent) in self.shape.iter().enumerate() {
            if index[position] >= *extent {
                return None;
            }
            linear += self.strides[position] * index[position] as isize;
        }
        Some(linear)
    }
}

/// Object-safe operations the erased value needs from one payload.
///
/// The payload is only ever one concrete `HostTensor<T>`, so a payload answers
/// with its own element type and never by reinterpreting bytes.
trait ErasedPayload: Send + Sync {
    /// Copy the payload while keeping its concrete element type.
    fn clone_payload(&self) -> Box<dyn ErasedPayload>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Copy the elements named by `layout` into a new dense payload.
    fn gather(&self, layout: &Layout) -> Result<Box<dyn ErasedPayload>, ValidationError>;
}

impl<T: Scalar> ErasedPayload for HostTensor<T> {
    fn clone_payload(&self) -> Box<dyn ErasedPayload> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn gather(&self, layout: &Layout) -> Result<Box<dyn ErasedPayload>, ValidationError> {
        let source = self.as_slice();
        let expected = layout.count();
        let mut gathered = Vec::with_capacity(expected);
        let rank = layout.shape.len();
        let mut index = vec![0usize; rank];
        for _ in 0..expected {
            let linear = layout
                .linear_index(&index)
                .and_then(|linear| usize::try_from(linear).ok())
                .filter(|linear| *linear < source.len())
                .ok_or(ValidationError::ShapeDataLengthMismatch {
                    expected,
                    actual: source.len(),
                })?;
            gathered.push(source[linear]);
            for (position, current) in index.iter_mut().enumerate() {
                *current += 1;
                if *current < layout.shape[position] {
                    break;
                }
                *current = 0;
            }
        }
        Ok(Box::new(HostTensor::from_vec_col_major(
            layout.shape.clone(),
            gathered,
        )?))
    }
}

/// A host tensor whose element type is recovered at run time.
///
/// The payload keeps its own concrete type and is recovered by identity, so no
/// byte reinterpretation happens and a caller-owned payload is duplicated through
/// its own entry point rather than by copying bytes.
///
/// The value also carries the layout of the view it presents. [`Clone`] shares
/// that payload, so a metadata-only permutation is cheap; [`duplicate`] copies it,
/// so a caller that needs its own storage asks for one explicitly. A typed read
/// applies the layout, and the dense accessors refuse a strided view instead of
/// presenting the payload as if it were the view.
///
/// [`duplicate`]: ErasedHostTensor::duplicate
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
    payload: Arc<dyn ErasedPayload>,
    type_id: TypeId,
    element: TypeId,
    layout: Layout,
    payload_elements: usize,
}

impl Clone for ErasedHostTensor {
    /// Share the payload and copy only the layout metadata.
    ///
    /// This is a metadata-only operation, so it is what a permutation and a
    /// metadata view use. Use [`ErasedHostTensor::duplicate`] for an independent
    /// copy of the storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let value = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![7_i64])?);
    /// let view = value.clone();
    /// assert!(view.shares_payload_with(&value));
    /// assert!(!view.duplicate().shares_payload_with(&value));
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    fn clone(&self) -> Self {
        Self {
            payload: Arc::clone(&self.payload),
            type_id: self.type_id,
            element: self.element,
            layout: self.layout.clone(),
            payload_elements: self.payload_elements,
        }
    }
}

impl core::fmt::Debug for ErasedHostTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ErasedHostTensor")
            .field("type_id", &self.type_id)
            .field("shape", &self.layout.shape)
            .field("strides", &self.layout.strides)
            .field("offset", &self.layout.offset)
            .finish()
    }
}

impl ErasedHostTensor {
    /// Erase a host tensor's element type.
    ///
    /// The result presents the tensor's dense column-major layout.
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
        let layout = Layout::dense(value.shape());
        let payload_elements = layout.count();
        Self {
            payload: Arc::new(value),
            type_id: TypeId::of::<HostTensor<T>>(),
            element: TypeId::of::<T>(),
            layout,
            payload_elements,
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

    /// Shape of the presented view.
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
        &self.layout.shape
    }

    /// Element strides of the presented view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?);
    /// assert_eq!(erased.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn strides(&self) -> &[isize] {
        &self.layout.strides
    }

    /// Element offset of the presented view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2], vec![0.0_f64; 2])?);
    /// assert_eq!(erased.offset(), 0);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn offset(&self) -> isize {
        self.layout.offset
    }

    /// Whether the presented view is the dense column-major layout of its shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?);
    /// assert!(erased.is_contiguous());
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_dense()
    }

    /// Number of elements in the presented view.
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
        self.layout.count()
    }

    /// Whether two erased values present the same stored payload.
    ///
    /// A metadata-only view answers `true`, and an independent copy answers
    /// `false`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![1.0_f64])?);
    /// assert!(erased.shares_payload_with(&erased.clone()));
    /// assert!(!erased.shares_payload_with(&erased.duplicate()));
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn shares_payload_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.payload, &other.payload)
    }

    /// Copy the payload into an independent value with the same view.
    ///
    /// The copy is the caller's own storage, so later mutation of either value
    /// leaves the other unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![1], vec![1.0_f64])?);
    /// let mut copy = erased.duplicate();
    /// copy.downcast_mut::<f64>().unwrap().as_mut_slice()[0] = 5.0;
    /// assert_eq!(erased.as_dense::<f64>().unwrap().0, &[1.0]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn duplicate(&self) -> Self {
        Self {
            payload: Arc::from(self.payload.clone_payload()),
            type_id: self.type_id,
            element: self.element,
            layout: self.layout.clone(),
            payload_elements: self.payload_elements,
        }
    }

    /// Present the same elements under a permuted axis order.
    ///
    /// This is metadata only: the payload is shared, no element is moved, and the
    /// result's shape and strides follow `axes`.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::InvalidPermutationLength`] when `axes` does not
    /// have one entry per axis, [`ValidationError::AxisOutOfBounds`] when an axis
    /// is out of range, and [`ValidationError::DuplicateAxis`] when an axis
    /// repeats.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6])?);
    /// let permuted = erased.permuted(&[1, 0])?;
    /// assert_eq!(permuted.shape(), &[3, 2]);
    /// assert_eq!(permuted.strides(), &[2, 1]);
    /// assert!(permuted.shares_payload_with(&erased));
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub fn permuted(&self, axes: &[usize]) -> Result<Self, ValidationError> {
        Ok(Self {
            layout: self.layout.permuted(axes)?,
            ..self.clone()
        })
    }

    /// Materialize the presented view into a dense column-major payload.
    ///
    /// The result owns its elements in the view's axis order, so a subsequent
    /// read with the dense accessors returns the same logical values in a
    /// contiguous buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::ShapeDataLengthMismatch`] when the view names
    /// storage the payload does not have.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?);
    /// let contiguous = erased.permuted(&[1, 0])?.to_contiguous()?;
    /// assert_eq!(contiguous.shape(), &[2, 2]);
    /// assert!(contiguous.is_contiguous());
    /// assert_eq!(contiguous.as_dense::<f64>().unwrap().0, &[1.0, 3.0, 2.0, 4.0]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub fn to_contiguous(&self) -> Result<Self, ValidationError> {
        let payload = self.payload.gather(&self.layout)?;
        let layout = Layout::dense(&self.layout.shape);
        let payload_elements = layout.count();
        Ok(Self {
            payload: Arc::from(payload),
            type_id: self.type_id,
            element: self.element,
            layout,
            payload_elements,
        })
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

    /// Borrow the whole payload when it is dense and has element type `T`.
    ///
    /// A strided view answers `None` rather than presenting the payload as if it
    /// were the view. Use [`ErasedHostTensor::element_at`] or
    /// [`ErasedHostTensor::to_contiguous`] for a view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?);
    /// assert_eq!(erased.downcast_ref::<f64>().unwrap().shape(), &[2, 2]);
    ///
    /// // A strided view is not the dense payload, so the dense borrow refuses it.
    /// assert_eq!(erased.permuted(&[1, 0])?.downcast_ref::<f64>(), None);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn downcast_ref<T: Scalar>(&self) -> Option<&HostTensor<T>> {
        if !self.is_contiguous() {
            return None;
        }
        self.payload.as_any().downcast_ref::<HostTensor<T>>()
    }

    /// Mutably borrow the whole payload when it is dense, unique, and has
    /// element type `T`.
    ///
    /// A strided view or a payload shared with another value answers `None`, so
    /// aliasing is never reachable through this entry point. Use
    /// [`ErasedHostTensor::element_at_mut`] for a view.
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
        if !self.is_contiguous() {
            return None;
        }
        Arc::get_mut(&mut self.payload)?
            .as_any_mut()
            .downcast_mut::<HostTensor<T>>()
    }

    /// Take the whole payload when it is dense and has element type `T`.
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
    pub fn into_typed<T: Scalar>(mut self) -> Option<HostTensor<T>> {
        if !self.layout.is_dense() {
            return None;
        }
        // A shared payload cannot be taken out of its reference count, so taking
        // the elements requires the caller to be the only holder.
        let payload = Arc::get_mut(&mut self.payload)?;
        let tensor = payload.as_any_mut().downcast_mut::<HostTensor<T>>()?;
        let empty = HostTensor::from_vec_col_major(vec![0], Vec::<T>::new()).ok()?;
        Some(core::mem::replace(tensor, empty))
    }

    /// Number of elements the stored payload holds.
    ///
    /// A strided view may reach fewer than all of them, so this is the extent a
    /// buffer-length check must use rather than the view's own count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?);
    /// assert_eq!(erased.payload_element_count(), 4);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn payload_element_count(&self) -> usize {
        self.payload_elements
    }

    /// Borrow the dense element slice and its shape when the view is contiguous
    /// and has element type `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?);
    /// assert_eq!(erased.as_dense::<f64>().unwrap().0, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn as_dense<T: Scalar>(&self) -> Option<(&[T], &[usize])> {
        if !self.is_contiguous() {
            return None;
        }
        let payload = self.payload.as_any().downcast_ref::<HostTensor<T>>()?;
        Some((payload.as_slice(), payload.shape()))
    }

    /// Borrow one element of the presented view by logical index.
    ///
    /// This applies the view's strides and offset, so it reads the element the
    /// view names even when the view is not contiguous.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?);
    /// let permuted = erased.permuted(&[1, 0])?;
    /// assert_eq!(permuted.element_at::<f64>(&[1, 0]), Some(&3.0));
    /// assert_eq!(permuted.element_at::<f64>(&[2, 0]), None);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    #[must_use]
    pub fn element_at<T: Scalar>(&self, index: &[usize]) -> Option<&T> {
        let payload = self.payload.as_any().downcast_ref::<HostTensor<T>>()?;
        let linear = usize::try_from(self.layout.linear_index(index)?).ok()?;
        payload.as_slice().get(linear)
    }

    /// Mutably borrow one element of the presented view by logical index.
    ///
    /// This applies the view's strides and offset. It answers `None` when the
    /// payload is shared with another value, so two live views never produce two
    /// mutable borrows of one element. A writer therefore holds the only
    /// reference: build the view, release the value it came from, and mutate
    /// through the view, or call [`ErasedHostTensor::duplicate`] for a payload of
    /// its own.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
    ///
    /// let erased = ErasedHostTensor::new(HostTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?);
    ///
    /// // A view that shares its payload refuses a mutable element borrow.
    /// let mut shared = erased.permuted(&[1, 0])?;
    /// assert!(shared.element_at_mut::<f64>(&[1, 0]).is_none());
    ///
    /// // An independent copy accepts it, and the original stays unchanged.
    /// let mut owned = erased.permuted(&[1, 0])?.duplicate();
    /// *owned.element_at_mut::<f64>(&[1, 0]).unwrap() = 20.0;
    /// assert_eq!(owned.element_at::<f64>(&[1, 0]), Some(&20.0));
    /// assert_eq!(erased.as_dense::<f64>().unwrap().0, &[1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub fn element_at_mut<T: Scalar>(&mut self, index: &[usize]) -> Option<&mut T> {
        let layout = &self.layout;
        let linear = usize::try_from(layout.linear_index(index)?).ok()?;
        let payload = Arc::get_mut(&mut self.payload)?;
        let payload = payload.as_any_mut().downcast_mut::<HostTensor<T>>()?;
        payload.as_mut_slice().get_mut(linear)
    }
}

#[cfg(test)]
mod tests;
