use num_complex::{Complex32, Complex64};

use super::{materialize_typed_view_col_major, DType, Tensor, TypedTensorView, TypedTensorViewMut};
use crate::Result;

/// One-axis slice specification for typed view slicing.
///
/// This follows ndarray's range-with-step model: negative `start` or `end`
/// values count from the end of the axis, `end` is exclusive, and negative
/// `step` reverses the selected range before stepping.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::StridedSliceSpec;
///
/// let every_other_reversed = StridedSliceSpec::new(0, None, -2);
/// assert_eq!(every_other_reversed.step(), -2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StridedSliceSpec {
    start: isize,
    end: Option<isize>,
    step: isize,
}

impl StridedSliceSpec {
    /// Create a slice from a signed start, optional exclusive end, and step.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// let slice = StridedSliceSpec::new(1, Some(-1), 1);
    /// assert_eq!(slice.start(), 1);
    /// assert_eq!(slice.end(), Some(-1));
    /// ```
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Self {
        Self { start, end, step }
    }

    /// Select the full axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::all(), StridedSliceSpec::new(0, None, 1));
    /// ```
    pub fn all() -> Self {
        Self::new(0, None, 1)
    }

    /// Select the full axis in reverse order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::reverse(), StridedSliceSpec::new(0, None, -1));
    /// ```
    pub fn reverse() -> Self {
        Self::new(0, None, -1)
    }

    /// Return the signed start bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::new(-3, None, 1).start(), -3);
    /// ```
    pub fn start(&self) -> isize {
        self.start
    }

    /// Return the optional signed exclusive end bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::new(0, Some(2), 1).end(), Some(2));
    /// ```
    pub fn end(&self) -> Option<isize> {
        self.end
    }

    /// Return the signed step.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::reverse().step(), -1);
    /// ```
    pub fn step(&self) -> isize {
        self.step
    }
}

/// Dynamic borrowed strided host tensor view.
///
/// The dynamic view supports all dtypes represented by tenferro's compute
/// [`Tensor`], including `bool` and `i32`.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, StridedTensorView};
///
/// let data = [true, false];
/// let view = StridedTensorView::bool(&[2], &[1], 0, &data).unwrap();
///
/// assert_eq!(view.dtype(), DType::Bool);
/// ```
#[derive(Clone, Debug)]
pub enum StridedTensorView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool(TypedTensorView<'a, bool>),
    C32(TypedTensorView<'a, Complex32>),
    C64(TypedTensorView<'a, Complex64>),
}

macro_rules! strided_view_ctor {
    ($name:ident, $variant:ident, $ty:ty, $dtype:ident) => {
        #[doc = concat!("Create a dynamic `", stringify!($dtype), "` strided view.")]
        pub fn $name(
            shape: &[usize],
            strides: &[isize],
            offset: isize,
            data: &'a [$ty],
        ) -> Result<Self> {
            Ok(Self::$variant(TypedTensorView::from_slice(
                shape.to_vec(),
                strides.to_vec(),
                offset,
                data,
            )?))
        }
    };
}

impl<'a> StridedTensorView<'a> {
    strided_view_ctor!(f32, F32, f32, F32);
    strided_view_ctor!(f64, F64, f64, F64);
    strided_view_ctor!(i32, I32, i32, I32);
    strided_view_ctor!(i64, I64, i64, I64);
    strided_view_ctor!(bool, Bool, bool, Bool);
    strided_view_ctor!(c32, C32, Complex32, C32);
    strided_view_ctor!(c64, C64, Complex64, C64);

    /// Return the view dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, StridedTensorView};
    ///
    /// let data = [1_i32, 2];
    /// let view = StridedTensorView::i32(&[2], &[1], 0, &data).unwrap();
    /// assert_eq!(view.dtype(), DType::I32);
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

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorView;
    ///
    /// let data = [1.0_f64, 2.0];
    /// let view = StridedTensorView::f64(&[2], &[1], 0, &data).unwrap();
    /// assert_eq!(view.shape(), &[2]);
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

    /// Return the logical strides measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorView;
    ///
    /// let data = [1_i64, 2];
    /// let view = StridedTensorView::i64(&[2], &[1], 0, &data).unwrap();
    /// assert_eq!(view.strides(), &[1]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        match self {
            Self::F32(t) => t.strides(),
            Self::F64(t) => t.strides(),
            Self::I32(t) => t.strides(),
            Self::I64(t) => t.strides(),
            Self::Bool(t) => t.strides(),
            Self::C32(t) => t.strides(),
            Self::C64(t) => t.strides(),
        }
    }

    /// Return the physical starting offset measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorView;
    ///
    /// let data = [1.0_f32, 2.0];
    /// let view = StridedTensorView::f32(&[1], &[1], 1, &data).unwrap();
    /// assert_eq!(view.offset(), 1);
    /// ```
    pub fn offset(&self) -> isize {
        match self {
            Self::F32(t) => t.offset(),
            Self::F64(t) => t.offset(),
            Self::I32(t) => t.offset(),
            Self::I64(t) => t.offset(),
            Self::Bool(t) => t.offset(),
            Self::C32(t) => t.offset(),
            Self::C64(t) => t.offset(),
        }
    }

    /// Materialize this dynamic view into tenferro's compute [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorView;
    ///
    /// let data = [1.0_f64, 2.0];
    /// let view = StridedTensorView::f64(&[2], &[1], 0, &data).unwrap();
    /// let tensor = view.to_tensor().unwrap();
    ///
    /// assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn to_tensor(&self) -> Result<Tensor> {
        match self {
            Self::F32(t) => Ok(Tensor::F32(materialize_typed_view_col_major(
                t,
                "StridedTensorView::to_tensor",
            )?)),
            Self::F64(t) => Ok(Tensor::F64(materialize_typed_view_col_major(
                t,
                "StridedTensorView::to_tensor",
            )?)),
            Self::I32(t) => Ok(Tensor::I32(materialize_typed_view_col_major(
                t,
                "StridedTensorView::to_tensor",
            )?)),
            Self::I64(t) => Ok(Tensor::I64(materialize_typed_view_col_major(
                t,
                "StridedTensorView::to_tensor",
            )?)),
            Self::Bool(t) => Ok(Tensor::Bool(materialize_typed_view_col_major(
                t,
                "StridedTensorView::to_tensor",
            )?)),
            Self::C32(t) => Ok(Tensor::C32(materialize_typed_view_col_major(
                t,
                "StridedTensorView::to_tensor",
            )?)),
            Self::C64(t) => Ok(Tensor::C64(materialize_typed_view_col_major(
                t,
                "StridedTensorView::to_tensor",
            )?)),
        }
    }
}

/// Dynamic borrowed mutable strided host tensor view.
///
/// The dynamic mutable view supports all dtypes represented by tenferro's
/// compute [`Tensor`]. Like [`TypedTensorViewMut`], constructors reject
/// layouts where two logical indices can refer to the same backing element.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, StridedTensorViewMut};
///
/// let mut data = [true, false];
/// let view = StridedTensorViewMut::bool(&[2], &[1], 0, &mut data).unwrap();
///
/// assert_eq!(view.dtype(), DType::Bool);
/// ```
#[derive(Debug)]
pub enum StridedTensorViewMut<'a> {
    F32(TypedTensorViewMut<'a, f32>),
    F64(TypedTensorViewMut<'a, f64>),
    I32(TypedTensorViewMut<'a, i32>),
    I64(TypedTensorViewMut<'a, i64>),
    Bool(TypedTensorViewMut<'a, bool>),
    C32(TypedTensorViewMut<'a, Complex32>),
    C64(TypedTensorViewMut<'a, Complex64>),
}

macro_rules! strided_view_mut_ctor {
    ($name:ident, $variant:ident, $ty:ty, $dtype:ident) => {
        #[doc = concat!("Create a dynamic mutable `", stringify!($dtype), "` strided view.")]
        pub fn $name(
            shape: &[usize],
            strides: &[isize],
            offset: isize,
            data: &'a mut [$ty],
        ) -> Result<Self> {
            Ok(Self::$variant(TypedTensorViewMut::from_slice(
                shape.to_vec(),
                strides.to_vec(),
                offset,
                data,
            )?))
        }
    };
}

impl<'a> StridedTensorViewMut<'a> {
    strided_view_mut_ctor!(f32, F32, f32, F32);
    strided_view_mut_ctor!(f64, F64, f64, F64);
    strided_view_mut_ctor!(i32, I32, i32, I32);
    strided_view_mut_ctor!(i64, I64, i64, I64);
    strided_view_mut_ctor!(bool, Bool, bool, Bool);
    strided_view_mut_ctor!(c32, C32, Complex32, C32);
    strided_view_mut_ctor!(c64, C64, Complex64, C64);

    /// Return the view dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, StridedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2];
    /// let view = StridedTensorViewMut::i32(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.dtype(), DType::I32);
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

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorViewMut;
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let view = StridedTensorViewMut::f64(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.shape(), &[2]);
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

    /// Return the logical strides measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorViewMut;
    ///
    /// let mut data = [1_i64, 2];
    /// let view = StridedTensorViewMut::i64(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.strides(), &[1]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        match self {
            Self::F32(t) => t.strides(),
            Self::F64(t) => t.strides(),
            Self::I32(t) => t.strides(),
            Self::I64(t) => t.strides(),
            Self::Bool(t) => t.strides(),
            Self::C32(t) => t.strides(),
            Self::C64(t) => t.strides(),
        }
    }

    /// Return the physical starting offset measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorViewMut;
    ///
    /// let mut data = [1.0_f32, 2.0];
    /// let view = StridedTensorViewMut::f32(&[1], &[1], 1, &mut data).unwrap();
    /// assert_eq!(view.offset(), 1);
    /// ```
    pub fn offset(&self) -> isize {
        match self {
            Self::F32(t) => t.offset(),
            Self::F64(t) => t.offset(),
            Self::I32(t) => t.offset(),
            Self::I64(t) => t.offset(),
            Self::Bool(t) => t.offset(),
            Self::C32(t) => t.offset(),
            Self::C64(t) => t.offset(),
        }
    }

    /// Borrow this mutable dynamic view as a read-only dynamic view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorViewMut;
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let view = StridedTensorViewMut::f64(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.as_read_only().to_tensor().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn as_read_only(&self) -> StridedTensorView<'_> {
        match self {
            Self::F32(t) => StridedTensorView::F32(t.as_read_only()),
            Self::F64(t) => StridedTensorView::F64(t.as_read_only()),
            Self::I32(t) => StridedTensorView::I32(t.as_read_only()),
            Self::I64(t) => StridedTensorView::I64(t.as_read_only()),
            Self::Bool(t) => StridedTensorView::Bool(t.as_read_only()),
            Self::C32(t) => StridedTensorView::C32(t.as_read_only()),
            Self::C64(t) => StridedTensorView::C64(t.as_read_only()),
        }
    }

    /// Return two mutable dynamic slices when their physical offset ranges are disjoint.
    ///
    /// This returns `None` instead of panicking when either slice spec is
    /// invalid or the selected physical ranges overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StridedSliceSpec, StridedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = StridedTensorViewMut::i32(&[4], &[1], 0, &mut data).unwrap();
    /// let (left, right) = view
    ///     .try_multi_slice_mut(
    ///         &[StridedSliceSpec::new(0, Some(2), 1)],
    ///         &[StridedSliceSpec::new(2, Some(4), 1)],
    ///     )
    ///     .unwrap();
    /// assert_eq!(left.shape(), &[2]);
    /// assert_eq!(right.shape(), &[2]);
    /// ```
    pub fn try_multi_slice_mut(
        &mut self,
        first: &[StridedSliceSpec],
        second: &[StridedSliceSpec],
    ) -> Option<(StridedTensorViewMut<'_>, StridedTensorViewMut<'_>)> {
        match self {
            Self::F32(t) => t
                .try_multi_slice_mut(first, second)
                .map(|(a, b)| (StridedTensorViewMut::F32(a), StridedTensorViewMut::F32(b))),
            Self::F64(t) => t
                .try_multi_slice_mut(first, second)
                .map(|(a, b)| (StridedTensorViewMut::F64(a), StridedTensorViewMut::F64(b))),
            Self::I32(t) => t
                .try_multi_slice_mut(first, second)
                .map(|(a, b)| (StridedTensorViewMut::I32(a), StridedTensorViewMut::I32(b))),
            Self::I64(t) => t
                .try_multi_slice_mut(first, second)
                .map(|(a, b)| (StridedTensorViewMut::I64(a), StridedTensorViewMut::I64(b))),
            Self::Bool(t) => t
                .try_multi_slice_mut(first, second)
                .map(|(a, b)| (StridedTensorViewMut::Bool(a), StridedTensorViewMut::Bool(b))),
            Self::C32(t) => t
                .try_multi_slice_mut(first, second)
                .map(|(a, b)| (StridedTensorViewMut::C32(a), StridedTensorViewMut::C32(b))),
            Self::C64(t) => t
                .try_multi_slice_mut(first, second)
                .map(|(a, b)| (StridedTensorViewMut::C64(a), StridedTensorViewMut::C64(b))),
        }
    }

    /// Materialize this dynamic mutable view into tenferro's compute [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedTensorViewMut;
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let view = StridedTensorViewMut::f64(&[2], &[1], 0, &mut data).unwrap();
    /// let tensor = view.to_tensor().unwrap();
    ///
    /// assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn to_tensor(&self) -> Result<Tensor> {
        self.as_read_only().to_tensor()
    }
}
