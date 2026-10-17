//! Backend-independent rank/layout, dtype and scalar metadata.
//!
//! `tenferro-tensor-core` owns backend-independent tensor metadata: scalar
//! tags and promotion facts, rank/layout metadata, and layout validation. It
//! does not own tensor storage, execution backends, backend buffers, GPU
//! handles, provider selection, or materializing kernels. The tensor families
//! that carry storage live in `tenferro-tensor`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor_core::{DType, Rank, TensorLayout};
//!
//! let layout = TensorLayout::<Rank<2>>::compact([2, 3])?;
//! let transposed = layout.transpose_view([1, 0])?;
//! assert_eq!(transposed.shape(), &[3, 2]);
//!
//! assert_eq!(DType::F64.spec().width, 64);
//! # Ok::<(), tenferro_tensor_core::ValidationError>(())
//! ```

use num_complex::{Complex32, Complex64};
use tinyvec::TinyVec;

mod error;
mod layout;
mod rank;
mod scalar;
#[macro_use]
mod scalar_set;

pub use error::{ErrorKind, ShapeMismatch, ValidationError, ValidationKind};
pub use layout::TensorLayout;
pub use rank::{DynRank, IntoRankShape, Rank, TensorRank};
pub use scalar::{ad_admission, AdAdmissionError, Scalar, ScalarArithmetic, ScalarDomain};
pub use scalar_set::{promote_in_set, promote_specs, MemberKind, MemberSpec};

/// Maximum rank whose shape/stride metadata stays inline without a heap
/// allocation.
///
/// Mirrors `strided_view::INLINE_RANK`; `tenferro-cpu-basic` carries a
/// compile-time assertion that the two values agree. At rank `<= INLINE_RANK`
/// [`ShapeVec`] and [`StrideVec`] keep their elements inline; a higher rank
/// spills to the heap gracefully.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::INLINE_RANK;
///
/// assert_eq!(INLINE_RANK, 8);
/// ```
pub const INLINE_RANK: usize = 8;

/// Small tensor shape vector with inline capacity for common dynamic ranks.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::ShapeVec;
///
/// let shape: ShapeVec = vec![2, 3].into_iter().collect();
/// assert_eq!(shape.as_slice(), &[2, 3]);
/// ```
pub type ShapeVec = TinyVec<[usize; INLINE_RANK]>;

/// Convert a common shape container into the dynamic owned shape type.
///
/// Arrays, vectors, slices, and [`ShapeVec`] implement this trait through
/// their `AsRef<[usize]>` representation.
pub trait IntoShapeVec {
    /// Convert this shape container into an owned [`ShapeVec`].
    fn into_shape_vec(self) -> ShapeVec;
}

impl<S> IntoShapeVec for S
where
    S: AsRef<[usize]>,
{
    fn into_shape_vec(self) -> ShapeVec {
        self.as_ref().iter().copied().collect()
    }
}

/// Small tensor stride vector with signed element strides.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::StrideVec;
///
/// let strides: StrideVec = vec![1, 2].into_iter().collect();
/// assert_eq!(strides.as_slice(), &[1, 2]);
/// ```
pub type StrideVec = TinyVec<[isize; INLINE_RANK]>;

/// Result type for tensor data-model operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Result, ValidationError};
///
/// let result: Result<()> = Err(ValidationError::RankMismatch { expected: 2, actual: 1 });
/// assert!(result.is_err());
/// ```
pub type Result<T> = std::result::Result<T, ValidationError>;

define_scalar_tag! {
    /// Runtime scalar dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::DType;
    ///
    /// assert_eq!(DType::F64, DType::F64);
    /// ```
    pub enum DType {
        /// 32-bit floating point.
        F32 => f32 : Float 0 32,
        /// 64-bit floating point.
        F64 => f64 : Float 1 64,
        /// 32-bit signed integer.
        I32 => i32 : Integer 0 32,
        /// 64-bit signed integer.
        I64 => i64 : Integer 1 64,
        /// Boolean.
        Bool => bool : Boolean 0 0,
        /// 32-bit complex floating point.
        C32 => Complex32 : Complex 0 32,
        /// 64-bit complex floating point.
        C64 => Complex64 : Complex 1 64,
    }
    external External(core::any::TypeId);
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
pub trait TensorScalar: Scalar + Copy + Clone + Send + Sync + 'static + private::Sealed {
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

macro_rules! scalar_domain {
    (float) => {
        ScalarDomain::Field
    };
    (complex) => {
        ScalarDomain::Field
    };
    (integer) => {
        ScalarDomain::Field
    };
    (boolean) => {
        ScalarDomain::NonField
    };
}

macro_rules! impl_scalar_arithmetic {
    ($ty:ty, float) => {
        impl ScalarArithmetic for $ty {
            fn scalar_zero() -> Self {
                0.0
            }

            fn scalar_one() -> Self {
                1.0
            }

            fn scalar_add(self, rhs: Self) -> Self {
                std::ops::Add::add(self, rhs)
            }

            fn scalar_sub(self, rhs: Self) -> Self {
                std::ops::Sub::sub(self, rhs)
            }

            fn scalar_mul(self, rhs: Self) -> Self {
                std::ops::Mul::mul(self, rhs)
            }
        }
    };
    ($ty:ty, complex) => {
        impl ScalarArithmetic for $ty {
            fn scalar_zero() -> Self {
                <$ty>::new(0.0, 0.0)
            }

            fn scalar_one() -> Self {
                <$ty>::new(1.0, 0.0)
            }

            fn scalar_add(self, rhs: Self) -> Self {
                std::ops::Add::add(self, rhs)
            }

            fn scalar_sub(self, rhs: Self) -> Self {
                std::ops::Sub::sub(self, rhs)
            }

            fn scalar_mul(self, rhs: Self) -> Self {
                std::ops::Mul::mul(self, rhs)
            }
        }
    };
    ($ty:ty, integer) => {
        impl ScalarArithmetic for $ty {
            fn scalar_zero() -> Self {
                0
            }

            fn scalar_one() -> Self {
                1
            }

            fn scalar_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            fn scalar_sub(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            fn scalar_mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
        }
    };
    ($ty:ty, boolean) => {};
}

macro_rules! impl_scalar {
    ($ty:ty, $real:ty, $dtype:expr, $variant:ident, $kind:ident) => {
        impl Scalar for $ty {
            const DOMAIN: ScalarDomain = scalar_domain!($kind);
        }

        impl_scalar_arithmetic!($ty, $kind);

        impl TensorScalar for $ty {
            type Real = $real;

            fn dtype() -> DType {
                $dtype
            }
        }
    };
}

// The single preset table: tag, real counterpart, erased variant, and algebra
// kind are declared once here and expanded into every contract the preset
// scalars implement.
impl_scalar!(f32, f32, DType::F32, F32, float);
impl_scalar!(f64, f64, DType::F64, F64, float);
impl_scalar!(i32, i32, DType::I32, I32, integer);
impl_scalar!(i64, i64, DType::I64, I64, integer);
impl_scalar!(bool, bool, DType::Bool, Bool, boolean);
impl_scalar!(Complex32, f32, DType::C32, C32, complex);
impl_scalar!(Complex64, f64, DType::C64, C64, complex);

/// Explicit slice descriptor.
///
/// A zero step is invalid. Layout metadata APIs support signed steps when
/// reachable-range validation proves the view stays inside the backing
/// allocation.
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

// A valid empty slice. `tinyvec`'s storage (`Array::Item: Default`) needs a
// default value to initialize unused inline slots; unlike a derived default,
// `step` stays nonzero so the value is a legal `SliceSpec`.
impl Default for SliceSpec {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            step: 1,
        }
    }
}

fn checked_product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(ValidationError::IntegerOverflow)
    })
}

fn checked_logical_element_count(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    checked_product(shape)
}

/// Return compact column-major strides for a shape.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::col_major_strides;
///
/// assert_eq!(col_major_strides(&[2, 3])?.as_slice(), &[1, 2]);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
///
/// # Errors
///
/// Returns [`ValidationError::IntegerOverflow`] when a stride or extent
/// cannot be represented by the metadata arithmetic.
pub fn col_major_strides(shape: &[usize]) -> Result<StrideVec> {
    let mut strides = StrideVec::new();
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        let extent = isize::try_from(extent).map_err(|_| ValidationError::IntegerOverflow)?;
        stride = stride
            .checked_mul(extent)
            .ok_or(ValidationError::IntegerOverflow)?;
    }
    Ok(strides)
}

fn validate_permutation(rank: usize, axes: &[usize]) -> Result<()> {
    if axes.len() != rank {
        return Err(ValidationError::InvalidPermutationLength {
            expected: rank,
            actual: axes.len(),
        });
    }
    // Common ranks track seen axes in a bitmask; only very high ranks allocate.
    if rank <= u128::BITS as usize {
        let mut seen = 0u128;
        for &axis in axes {
            if axis >= rank {
                return Err(ValidationError::AxisOutOfBounds { axis, rank });
            }
            let bit = 1u128 << axis;
            if seen & bit != 0 {
                return Err(ValidationError::DuplicateAxis {
                    axis,
                    role: "permutation",
                });
            }
            seen |= bit;
        }
        return Ok(());
    }
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(ValidationError::AxisOutOfBounds { axis, rank });
        }
        if seen[axis] {
            return Err(ValidationError::DuplicateAxis {
                axis,
                role: "permutation",
            });
        }
        seen[axis] = true;
    }
    Ok(())
}
