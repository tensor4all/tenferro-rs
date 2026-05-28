//! Lightweight host tensor data model and metadata-only views.
//!
//! `tenferro-tensor-core` owns backend-independent tensor metadata and
//! host-resident contiguous tensor storage. It does not own execution backends,
//! backend buffers, GPU handles, provider selection, or materializing kernels.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor_core::{SliceSpec, TypedTensor};
//!
//! let tensor = TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! let view = tensor
//!     .as_view()
//!     .slice_view(&[
//!         SliceSpec { start: 0, end: 2, step: 1 },
//!         SliceSpec { start: 1, end: 3, step: 1 },
//!     ])?;
//!
//! assert_eq!(view.shape(), &[2, 2]);
//! assert_eq!(view.as_slice()?, &[3.0, 4.0, 5.0, 6.0]);
//! # Ok::<(), tenferro_tensor_core::Error>(())
//! ```

use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;

mod layout;
mod rank;

pub use layout::TensorLayout;
pub use rank::{DynRank, Rank, TensorRank};

/// Small tensor shape vector with inline capacity for common dynamic ranks.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::ShapeVec;
///
/// let shape = ShapeVec::from_vec(vec![2, 3]);
/// assert_eq!(shape.as_slice(), &[2, 3]);
/// ```
pub type ShapeVec = SmallVec<[usize; 8]>;

/// Small tensor stride vector with signed element strides.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::StrideVec;
///
/// let strides = StrideVec::from_vec(vec![1, 2]);
/// assert_eq!(strides.as_slice(), &[1, 2]);
/// ```
pub type StrideVec = SmallVec<[isize; 8]>;

/// Result type for tensor data-model operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Error, Result};
///
/// let result: Result<()> = Err(Error::RankMismatch { expected: 2, actual: 1 });
/// assert!(result.is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;

/// Data-model validation errors.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::Error;
///
/// let err = Error::ReshapeElementCountMismatch { from: 4, to: 5 };
/// assert!(err.to_string().contains("reshape"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("shape product {expected} does not match data length {actual}")]
    ShapeDataLengthMismatch { expected: usize, actual: usize },
    #[error("rank mismatch: expected {expected}, actual {actual}")]
    RankMismatch { expected: usize, actual: usize },
    #[error("axis {axis} out of bounds for rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    #[error("duplicate axis {axis} in permutation")]
    DuplicateAxis { axis: usize },
    #[error("invalid permutation length: expected {expected}, actual {actual}")]
    InvalidPermutationLength { expected: usize, actual: usize },
    #[error("invalid slice step {step}; v1 requires a positive step")]
    InvalidSliceStep { step: isize },
    #[error(
        "slice bounds are invalid or unsupported: start={start}, end={end}, axis_len={axis_len}"
    )]
    InvalidSliceBounds {
        start: isize,
        end: isize,
        axis_len: usize,
    },
    #[error("reshape element-count mismatch: from {from} to {to}")]
    ReshapeElementCountMismatch { from: usize, to: usize },
    #[error("view is not slice-contiguous")]
    NonContiguousViewAsSlice,
    #[error("dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("view metadata is out of borrowed-slice bounds")]
    ViewOutOfBounds,
    #[error("integer overflow while validating tensor metadata")]
    IntegerOverflow,
}

/// Runtime scalar dtype tag.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::DType;
///
/// assert_eq!(DType::F64, DType::F64);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    C32,
    C64,
}

/// Sealed trait for scalar types supported by the core tensor data model.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{DType, TensorScalar};
///
/// assert_eq!(f64::dtype(), DType::F64);
/// assert_eq!(num_complex::Complex64::dtype(), DType::C64);
/// ```
pub trait TensorScalar: Copy + Clone + Send + Sync + 'static + private::Sealed {
    /// Real-valued counterpart of this scalar type.
    type Real: TensorScalar;

    /// Return the scalar dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, TensorScalar};
    ///
    /// assert_eq!(i64::dtype(), DType::I64);
    /// ```
    fn dtype() -> DType;

    fn into_tensor(shape: ShapeVec, data: Vec<Self>) -> Tensor;
    fn tensor_slice(tensor: &Tensor) -> Option<&[Self]>;
    fn tensor_mut_slice(tensor: &mut Tensor) -> Option<&mut [Self]>;
    fn into_typed(tensor: Tensor) -> Option<TypedTensor<Self>>;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for bool {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

macro_rules! impl_scalar {
    ($ty:ty, $real:ty, $dtype:expr, $variant:ident) => {
        impl TensorScalar for $ty {
            type Real = $real;

            fn dtype() -> DType {
                $dtype
            }

            fn into_tensor(shape: ShapeVec, data: Vec<Self>) -> Tensor {
                Tensor::$variant(TypedTensor { data, shape })
            }

            fn tensor_slice(tensor: &Tensor) -> Option<&[Self]> {
                match tensor {
                    Tensor::$variant(typed) => Some(typed.as_slice()),
                    _ => None,
                }
            }

            fn tensor_mut_slice(tensor: &mut Tensor) -> Option<&mut [Self]> {
                match tensor {
                    Tensor::$variant(typed) => Some(typed.as_mut_slice()),
                    _ => None,
                }
            }

            fn into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
                match tensor {
                    Tensor::$variant(typed) => Some(typed),
                    _ => None,
                }
            }
        }
    };
}

impl_scalar!(f32, f32, DType::F32, F32);
impl_scalar!(f64, f64, DType::F64, F64);
impl_scalar!(i32, i32, DType::I32, I32);
impl_scalar!(i64, i64, DType::I64, I64);
impl_scalar!(bool, bool, DType::Bool, Bool);
impl_scalar!(Complex32, f32, DType::C32, C32);
impl_scalar!(Complex64, f64, DType::C64, C64);

/// Explicit positive-step slice descriptor.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::SliceSpec;
///
/// let spec = SliceSpec { start: 1, end: 4, step: 2 };
/// assert_eq!(spec.step, 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceSpec {
    pub start: isize,
    pub end: isize,
    pub step: isize,
}

/// Owned contiguous host tensor in column-major order.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::TypedTensor;
///
/// let tensor = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// assert_eq!(tensor.as_slice(), &[1.0, 2.0]);
/// # Ok::<(), tenferro_tensor_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TypedTensor<T> {
    data: Vec<T>,
    shape: ShapeVec,
}

/// Dynamic owned host tensor over the supported dtype set.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{DType, Tensor};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// assert_eq!(tensor.dtype(), DType::F64);
/// # Ok::<(), tenferro_tensor_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    I32(TypedTensor<i32>),
    I64(TypedTensor<i64>),
    Bool(TypedTensor<bool>),
    C32(TypedTensor<Complex32>),
    C64(TypedTensor<Complex64>),
}

/// Borrowed host tensor view with shape, strides, and offset metadata.
///
/// This type intentionally does not implement `PartialEq` because view
/// equality is ambiguous between metadata identity, storage identity, and
/// logical element equality.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::TypedTensor;
///
/// let tensor = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let view = tensor.as_view();
/// assert_eq!(view.shape(), &[2]);
/// # Ok::<(), tenferro_tensor_core::Error>(())
/// ```
///
/// ```compile_fail
/// # use tenferro_tensor_core::TypedTensor;
/// # let tensor = TypedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
/// let a = tensor.as_view();
/// let b = tensor.as_view();
/// let _ = a == b;
/// ```
#[derive(Clone, Debug)]
pub struct TypedTensorView<'a, T> {
    data: &'a [T],
    shape: ShapeVec,
    strides: StrideVec,
    offset: isize,
}

/// Dynamic borrowed host tensor view.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{DType, Tensor};
///
/// let tensor = Tensor::from_vec_col_major(vec![1], vec![true])?;
/// let view = tensor.as_view();
/// assert_eq!(view.dtype(), DType::Bool);
/// # Ok::<(), tenferro_tensor_core::Error>(())
/// ```
///
/// ```compile_fail
/// # use tenferro_tensor_core::Tensor;
/// # let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
/// let a = tensor.as_view();
/// let b = tensor.as_view();
/// let _ = a == b;
/// ```
#[derive(Clone, Debug)]
pub enum TensorView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool(TypedTensorView<'a, bool>),
    C32(TypedTensorView<'a, Complex32>),
    C64(TypedTensorView<'a, Complex64>),
}

/// Core-neutral tensor input reference.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Tensor, TensorRef};
///
/// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f32])?;
/// let reference = TensorRef::Tensor(&tensor);
/// assert_eq!(reference.shape(), &[1]);
/// # Ok::<(), tenferro_tensor_core::Error>(())
/// ```
#[derive(Clone, Debug)]
pub enum TensorRef<'a> {
    Tensor(&'a Tensor),
    View(TensorView<'a>),
}

fn checked_product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(Error::IntegerOverflow)
    })
}

fn checked_shape_len(shape: &[usize], data_len: usize) -> Result<usize> {
    validate_shape_metadata(shape)?;
    let expected = checked_product(shape)?;
    if expected != data_len {
        return Err(Error::ShapeDataLengthMismatch {
            expected,
            actual: data_len,
        });
    }
    Ok(expected)
}

fn validate_shape_metadata(shape: &[usize]) -> Result<()> {
    checked_product(shape)?;
    col_major_strides(shape)?;
    Ok(())
}

fn compact_col_major_strides(shape: &[usize]) -> StrideVec {
    let mut strides = StrideVec::new();
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        stride *= extent as isize;
    }
    strides
}

/// Return compact column-major strides for a shape.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::col_major_strides;
///
/// assert_eq!(col_major_strides(&[2, 3])?.as_slice(), &[1, 2]);
/// # Ok::<(), tenferro_tensor_core::Error>(())
/// ```
pub fn col_major_strides(shape: &[usize]) -> Result<StrideVec> {
    let mut strides = StrideVec::new();
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        let extent = isize::try_from(extent).map_err(|_| Error::IntegerOverflow)?;
        stride = stride.checked_mul(extent).ok_or(Error::IntegerOverflow)?;
    }
    Ok(strides)
}

fn linear_offset_checked(shape: &[usize], indices: &[usize]) -> Option<usize> {
    if shape.len() != indices.len() {
        return None;
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&extent, &index) in shape.iter().zip(indices) {
        if index >= extent {
            return None;
        }
        offset = offset.checked_add(index.checked_mul(stride)?)?;
        stride = stride.checked_mul(extent)?;
    }
    Some(offset)
}

fn row_major_offset(shape: &[usize], indices: &[usize]) -> Option<usize> {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&extent, &index) in shape.iter().rev().zip(indices.iter().rev()) {
        if index >= extent {
            return None;
        }
        offset = offset.checked_add(index.checked_mul(stride)?)?;
        stride = stride.checked_mul(extent)?;
    }
    Some(offset)
}

fn for_each_col_major_index(
    shape: &[usize],
    mut f: impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    if shape.is_empty() {
        f(&[])?;
        return Ok(());
    }
    if shape.iter().any(|&extent| extent == 0) {
        return Ok(());
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        f(&index)?;
        let mut axis = 0usize;
        loop {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
            if axis == shape.len() {
                return Ok(());
            }
        }
    }
}

fn for_each_row_major_index(
    shape: &[usize],
    mut f: impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    if shape.is_empty() {
        f(&[])?;
        return Ok(());
    }
    if shape.iter().any(|&extent| extent == 0) {
        return Ok(());
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        f(&index)?;
        let mut axis = shape.len();
        loop {
            axis -= 1;
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            if axis == 0 {
                return Ok(());
            }
        }
    }
}

fn row_major_to_col_major<T: Clone>(shape: &[usize], data: Vec<T>) -> Result<Vec<T>> {
    checked_shape_len(shape, data.len())?;
    let mut out = Vec::with_capacity(data.len());
    for_each_col_major_index(shape, |index| {
        let offset = row_major_offset(shape, index).ok_or(Error::IntegerOverflow)?;
        out.push(data[offset].clone());
        Ok(())
    })?;
    Ok(out)
}

fn col_major_to_row_major<T: Clone>(shape: &[usize], data: Vec<T>) -> Result<Vec<T>> {
    checked_shape_len(shape, data.len())?;
    let mut out = Vec::with_capacity(data.len());
    for_each_row_major_index(shape, |index| {
        let offset = linear_offset_checked(shape, index).ok_or(Error::IntegerOverflow)?;
        out.push(data[offset].clone());
        Ok(())
    })?;
    Ok(out)
}

fn validate_permutation(rank: usize, axes: &[usize]) -> Result<()> {
    if axes.len() != rank {
        return Err(Error::InvalidPermutationLength {
            expected: rank,
            actual: axes.len(),
        });
    }
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::AxisOutOfBounds { axis, rank });
        }
        if seen[axis] {
            return Err(Error::DuplicateAxis { axis });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_view_bounds<T>(
    data: &[T],
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<()> {
    if shape.len() != strides.len() {
        return Err(Error::RankMismatch {
            expected: shape.len(),
            actual: strides.len(),
        });
    }
    if offset < 0 || strides.iter().any(|&stride| stride < 0) {
        return Err(Error::ViewOutOfBounds);
    }
    let offset = usize::try_from(offset).map_err(|_| Error::IntegerOverflow)?;
    if shape.iter().any(|&extent| extent == 0) {
        return if offset <= data.len() {
            Ok(())
        } else {
            Err(Error::ViewOutOfBounds)
        };
    }

    let mut max_offset = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let stride = usize::try_from(stride).map_err(|_| Error::IntegerOverflow)?;
        let axis_span = extent
            .checked_sub(1)
            .ok_or(Error::IntegerOverflow)?
            .checked_mul(stride)
            .ok_or(Error::IntegerOverflow)?;
        max_offset = max_offset
            .checked_add(axis_span)
            .ok_or(Error::IntegerOverflow)?;
    }
    if max_offset < data.len() {
        Ok(())
    } else {
        Err(Error::ViewOutOfBounds)
    }
}

fn is_slice_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    let mut expected = 1isize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent <= 1 {
            continue;
        }
        if stride != expected {
            return false;
        }
        let Ok(extent) = isize::try_from(extent) else {
            return false;
        };
        let Some(next) = expected.checked_mul(extent) else {
            return false;
        };
        expected = next;
    }
    true
}

impl<T> TypedTensor<T> {
    /// Create an owned tensor from a column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2], vec![1_i64, 2])?;
    /// assert_eq!(tensor.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn from_vec_col_major(shape: impl Into<ShapeVec>, data: Vec<T>) -> Result<Self> {
        let shape = shape.into();
        checked_shape_len(&shape, data.len())?;
        Ok(Self { data, shape })
    }

    /// Borrow this tensor's shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2], vec![true, false])?;
    /// assert_eq!(tensor.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the tensor rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f32, 2.0])?;
    /// assert_eq!(tensor.rank(), 2);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns `true` when this tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::<f64>::from_vec_col_major(vec![0], vec![])?;
    /// assert!(tensor.is_empty());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the contiguous column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![1], vec![7_i32])?;
    /// assert_eq!(tensor.as_slice(), &[7]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the contiguous column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let mut tensor = TypedTensor::from_vec_col_major(vec![1], vec![7_i32])?;
    /// tensor.as_mut_slice()[0] = 8;
    /// assert_eq!(tensor.as_slice(), &[8]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow this tensor as a compact zero-offset view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(tensor.as_view().is_zero_offset_col_major());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn as_view(&self) -> TypedTensorView<'_, T> {
        TypedTensorView {
            data: &self.data,
            shape: self.shape.clone(),
            strides: compact_col_major_strides(&self.shape),
            offset: 0,
        }
    }

    /// Consume this tensor into its shape and column-major buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// assert_eq!(tensor.into_vec_col_major().1, vec![3.0]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn into_vec_col_major(self) -> (ShapeVec, Vec<T>) {
        (self.shape, self.data)
    }

    /// Consume this tensor into the same data with a different shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// assert_eq!(tensor.into_reshaped(vec![2, 2])?.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn into_reshaped(self, shape: impl Into<ShapeVec>) -> Result<Self> {
        let shape = shape.into();
        let from = self.data.len();
        let to = checked_product(&shape)?;
        if from != to {
            return Err(Error::ReshapeElementCountMismatch { from, to });
        }
        validate_shape_metadata(&shape)?;
        Ok(Self {
            data: self.data,
            shape,
        })
    }
}

impl<T: Clone> TypedTensor<T> {
    /// Create an owned tensor from row-major data by converting to column-major storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn from_vec_row_major(shape: impl Into<ShapeVec>, data: Vec<T>) -> Result<Self> {
        let shape = shape.into();
        let data = row_major_to_col_major(&shape, data)?;
        Self::from_vec_col_major(shape, data)
    }

    /// Consume this tensor into row-major data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// assert_eq!(tensor.into_vec_row_major()?.1, vec![1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn into_vec_row_major(self) -> Result<(ShapeVec, Vec<T>)> {
        let data = col_major_to_row_major(&self.shape, self.data)?;
        Ok((self.shape, data))
    }
}

impl<'a, T> TypedTensorView<'a, T> {
    /// Create a typed view from explicit metadata and validate bounds eagerly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensorView;
    ///
    /// let data = [1.0_f64, 2.0, 3.0, 4.0];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 1, &data)?;
    /// assert_eq!(view.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn from_slice(
        shape: impl Into<ShapeVec>,
        strides: impl Into<StrideVec>,
        offset: isize,
        data: &'a [T],
    ) -> Result<Self> {
        let shape = shape.into();
        let strides = strides.into();
        validate_view_bounds(data, &shape, &strides, offset)?;
        Ok(Self {
            data,
            shape,
            strides,
            offset,
        })
    }

    /// Borrow this view's shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(tensor.as_view().shape(), &[2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow this view's signed element strides.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 3], vec![0_i32; 6])?;
    /// assert_eq!(tensor.as_view().strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Return this view's signed element offset into the backing slice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![1], vec![true])?;
    /// assert_eq!(tensor.as_view().offset(), 0);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Return the view rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
    /// assert_eq!(tensor.as_view().rank(), 2);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns `true` when this view has zero logical elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensorView;
    ///
    /// let data = [1.0_f64];
    /// let view = TypedTensorView::from_slice(vec![0], vec![1], 0, &data)?;
    /// assert!(view.is_empty());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.shape.iter().any(|&extent| extent == 0)
    }

    /// Return whether this view has compact column-major logical strides.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![0_i32; 4])?;
    /// assert!(tensor.as_view().is_compact_col_major());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_compact_col_major(&self) -> bool {
        is_slice_contiguous(&self.shape, &self.strides)
    }

    /// Return whether this view is compact column-major and starts at offset zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert!(tensor.as_view().is_zero_offset_col_major());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_zero_offset_col_major(&self) -> bool {
        self.offset == 0 && self.is_compact_col_major()
    }

    /// Alias for [`TypedTensorView::is_compact_col_major`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert!(tensor.as_view().is_contiguous_col_major());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_contiguous_col_major(&self) -> bool {
        self.is_compact_col_major()
    }

    /// Borrow the slice-contiguous backing region for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 1, &data)?;
    /// assert_eq!(view.as_slice()?, &[2, 3]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn as_slice(&self) -> Result<&'a [T]> {
        if !is_slice_contiguous(&self.shape, &self.strides) {
            return Err(Error::NonContiguousViewAsSlice);
        }
        let len = checked_product(&self.shape)?;
        let start = usize::try_from(self.offset).map_err(|_| Error::IntegerOverflow)?;
        let end = start.checked_add(len).ok_or(Error::IntegerOverflow)?;
        self.data.get(start..end).ok_or(Error::ViewOutOfBounds)
    }

    /// Return a metadata-only reshape of this compact column-major view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// assert_eq!(tensor.as_view().reshape_view(vec![2, 2])?.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn reshape_view(&self, shape: impl Into<ShapeVec>) -> Result<Self> {
        if !self.is_compact_col_major() {
            return Err(Error::NonContiguousViewAsSlice);
        }
        let shape = shape.into();
        let from = checked_product(&self.shape)?;
        let to = checked_product(&shape)?;
        if from != to {
            return Err(Error::ReshapeElementCountMismatch { from, to });
        }
        Self::from_slice(
            shape.clone(),
            col_major_strides(&shape)?,
            self.offset,
            self.data,
        )
    }

    /// Return a metadata-only axis permutation of this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::TypedTensor;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 3], vec![0_i32; 6])?;
    /// let view = tensor.as_view().permute_view(&[1, 0])?;
    /// assert_eq!(view.shape(), &[3, 2]);
    /// assert_eq!(view.strides(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn permute_view(&self, axes: &[usize]) -> Result<Self> {
        validate_permutation(self.rank(), axes)?;
        let shape = axes
            .iter()
            .map(|&axis| self.shape[axis])
            .collect::<ShapeVec>();
        let strides = axes
            .iter()
            .map(|&axis| self.strides[axis])
            .collect::<StrideVec>();
        Self::from_slice(shape, strides, self.offset, self.data)
    }

    /// Return a metadata-only positive-step slice of this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{SliceSpec, TypedTensor};
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![4], vec![1_i64, 2, 3, 4])?;
    /// let view = tensor
    ///     .as_view()
    ///     .slice_view(&[SliceSpec { start: 1, end: 4, step: 2 }])?;
    /// assert_eq!(view.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn slice_view(&self, spec: &[SliceSpec]) -> Result<Self> {
        if spec.len() != self.rank() {
            return Err(Error::RankMismatch {
                expected: self.rank(),
                actual: spec.len(),
            });
        }
        let mut shape = ShapeVec::new();
        let mut strides = StrideVec::new();
        let mut offset = self.offset;
        for ((&axis_len, &stride), slice) in self.shape.iter().zip(self.strides.iter()).zip(spec) {
            if slice.step <= 0 {
                return Err(Error::InvalidSliceStep { step: slice.step });
            }
            if slice.start < 0 || slice.end < 0 {
                return Err(Error::InvalidSliceBounds {
                    start: slice.start,
                    end: slice.end,
                    axis_len,
                });
            }
            let start = usize::try_from(slice.start).map_err(|_| Error::IntegerOverflow)?;
            let end = usize::try_from(slice.end).map_err(|_| Error::IntegerOverflow)?;
            if start > axis_len || end > axis_len {
                return Err(Error::InvalidSliceBounds {
                    start: slice.start,
                    end: slice.end,
                    axis_len,
                });
            }
            let step = usize::try_from(slice.step).map_err(|_| Error::IntegerOverflow)?;
            let extent = if start >= end {
                0
            } else {
                end.checked_sub(start)
                    .and_then(|span| span.checked_add(step - 1))
                    .ok_or(Error::IntegerOverflow)?
                    / step
            };
            let start_offset = isize::try_from(start)
                .map_err(|_| Error::IntegerOverflow)?
                .checked_mul(stride)
                .ok_or(Error::IntegerOverflow)?;
            offset = offset
                .checked_add(start_offset)
                .ok_or(Error::IntegerOverflow)?;
            let new_stride = stride
                .checked_mul(slice.step)
                .ok_or(Error::IntegerOverflow)?;
            shape.push(extent);
            strides.push(new_stride);
        }
        Self::from_slice(shape, strides, offset, self.data)
    }
}

impl Tensor {
    /// Create a dynamic tensor from a column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, Tensor};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f32])?;
    /// assert_eq!(tensor.dtype(), DType::F32);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn from_vec_col_major<T: TensorScalar>(
        shape: impl Into<ShapeVec>,
        data: Vec<T>,
    ) -> Result<Self> {
        let shape = shape.into();
        checked_shape_len(&shape, data.len())?;
        Ok(T::into_tensor(shape, data))
    }

    /// Create a dynamic tensor from a row-major host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_row_major(vec![1, 2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(tensor.shape(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn from_vec_row_major<T: TensorScalar + Clone>(
        shape: impl Into<ShapeVec>,
        data: Vec<T>,
    ) -> Result<Self> {
        let shape = shape.into();
        let data = row_major_to_col_major(&shape, data)?;
        Self::from_vec_col_major(shape, data)
    }

    /// Return the tensor dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, Tensor};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![false])?;
    /// assert_eq!(tensor.dtype(), DType::Bool);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    /// Borrow the tensor shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2])?;
    /// assert_eq!(tensor.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
            Self::C32(t) => t.shape(),
            Self::C64(t) => t.shape(),
        }
    }

    /// Return the tensor rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// assert_eq!(tensor.rank(), 2);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return whether the tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new())?;
    /// assert!(tensor.is_empty());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            Self::F32(t) => t.is_empty(),
            Self::F64(t) => t.is_empty(),
            Self::I32(t) => t.is_empty(),
            Self::I64(t) => t.is_empty(),
            Self::Bool(t) => t.is_empty(),
            Self::C32(t) => t.is_empty(),
            Self::C64(t) => t.is_empty(),
        }
    }

    /// Borrow the typed host slice when the dtype matches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// assert_eq!(tensor.as_slice::<f64>()?, &[3.0]);
    /// assert!(tensor.as_slice::<f32>().is_err());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn as_slice<T: TensorScalar>(&self) -> Result<&[T]> {
        T::tensor_slice(self).ok_or(Error::DTypeMismatch {
            expected: T::dtype(),
            actual: self.dtype(),
        })
    }

    /// Mutably borrow the typed host slice when the dtype matches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// tensor.as_mut_slice::<f64>()?[0] = 4.0;
    /// assert_eq!(tensor.as_slice::<f64>()?, &[4.0]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn as_mut_slice<T: TensorScalar>(&mut self) -> Result<&mut [T]> {
        let actual = self.dtype();
        T::tensor_mut_slice(self).ok_or(Error::DTypeMismatch {
            expected: T::dtype(),
            actual,
        })
    }

    /// Borrow this tensor as a dynamic zero-offset view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, Tensor};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert_eq!(tensor.as_view().dtype(), DType::I64);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn as_view(&self) -> TensorView<'_> {
        match self {
            Self::F32(t) => TensorView::F32(t.as_view()),
            Self::F64(t) => TensorView::F64(t.as_view()),
            Self::I32(t) => TensorView::I32(t.as_view()),
            Self::I64(t) => TensorView::I64(t.as_view()),
            Self::Bool(t) => TensorView::Bool(t.as_view()),
            Self::C32(t) => TensorView::C32(t.as_view()),
            Self::C64(t) => TensorView::C64(t.as_view()),
        }
    }

    /// Consume this tensor and return typed column-major data when the dtype matches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f32])?;
    /// assert_eq!(tensor.into_vec_col_major::<f32>()?.1, vec![2.0]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn into_vec_col_major<T: TensorScalar>(self) -> Result<(ShapeVec, Vec<T>)> {
        let actual = self.dtype();
        T::into_typed(self)
            .map(TypedTensor::into_vec_col_major)
            .ok_or(Error::DTypeMismatch {
                expected: T::dtype(),
                actual,
            })
    }
}

macro_rules! impl_dynamic_view {
    ($self:ident, $method:ident($($arg:ident),*) => $inner:ident) => {
        match $self {
            TensorView::F32(view) => TensorView::F32(view.$method($($arg),*)?),
            TensorView::F64(view) => TensorView::F64(view.$method($($arg),*)?),
            TensorView::I32(view) => TensorView::I32(view.$method($($arg),*)?),
            TensorView::I64(view) => TensorView::I64(view.$method($($arg),*)?),
            TensorView::Bool(view) => TensorView::Bool(view.$method($($arg),*)?),
            TensorView::C32(view) => TensorView::C32(view.$method($($arg),*)?),
            TensorView::C64(view) => TensorView::C64(view.$method($($arg),*)?),
        }
    };
}

impl<'a> TensorView<'a> {
    /// Return this view's dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, Tensor};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f32])?;
    /// assert_eq!(tensor.as_view().dtype(), DType::F32);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    /// Borrow this view's shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert_eq!(tensor.as_view().shape(), &[1]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::F64(view) => view.shape(),
            Self::I32(view) => view.shape(),
            Self::I64(view) => view.shape(),
            Self::Bool(view) => view.shape(),
            Self::C32(view) => view.shape(),
            Self::C64(view) => view.shape(),
        }
    }

    /// Return the view rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// assert_eq!(tensor.as_view().rank(), 2);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return whether this view has zero logical elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new())?;
    /// assert!(tensor.as_view().is_empty());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            Self::F32(view) => view.is_empty(),
            Self::F64(view) => view.is_empty(),
            Self::I32(view) => view.is_empty(),
            Self::I64(view) => view.is_empty(),
            Self::Bool(view) => view.is_empty(),
            Self::C32(view) => view.is_empty(),
            Self::C64(view) => view.is_empty(),
        }
    }

    /// Return a metadata-only reshape of this dynamic view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![4], vec![1_i32, 2, 3, 4])?;
    /// assert_eq!(tensor.as_view().reshape_view(vec![2, 2])?.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn reshape_view(&self, shape: impl Into<ShapeVec>) -> Result<Self> {
        let shape = shape.into();
        Ok(impl_dynamic_view!(self, reshape_view(shape) => view))
    }

    /// Return a metadata-only axis permutation of this dynamic view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::Tensor;
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1, 2], vec![1_i64, 2])?;
    /// assert_eq!(tensor.as_view().permute_view(&[1, 0])?.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn permute_view(&self, axes: &[usize]) -> Result<Self> {
        Ok(impl_dynamic_view!(self, permute_view(axes) => view))
    }

    /// Return a metadata-only positive-step slice of this dynamic view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{SliceSpec, Tensor};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![3], vec![1_i64, 2, 3])?;
    /// assert_eq!(
    ///     tensor.as_view().slice_view(&[SliceSpec { start: 1, end: 3, step: 1 }])?.shape(),
    ///     &[2],
    /// );
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn slice_view(&self, spec: &[SliceSpec]) -> Result<Self> {
        Ok(impl_dynamic_view!(self, slice_view(spec) => view))
    }
}

impl<'a> TensorRef<'a> {
    /// Return the referenced dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, Tensor, TensorRef};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert_eq!(TensorRef::Tensor(&tensor).dtype(), DType::I64);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    /// Borrow the referenced shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Tensor, TensorRef};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert_eq!(TensorRef::Tensor(&tensor).shape(), &[1]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    /// Return the referenced rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Tensor, TensorRef};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// assert_eq!(TensorRef::Tensor(&tensor).rank(), 2);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return whether the referenced tensor/view is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Tensor, TensorRef};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new())?;
    /// assert!(TensorRef::Tensor(&tensor).is_empty());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Tensor(tensor) => tensor.is_empty(),
            Self::View(view) => view.is_empty(),
        }
    }
}
