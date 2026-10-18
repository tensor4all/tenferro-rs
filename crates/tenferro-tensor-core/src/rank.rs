use crate::{IntoShapeVec, Result, ShapeVec, StrideVec, ValidationError};
use std::fmt::Debug;

/// Convert a shape container into the representation required by a tensor rank.
///
/// Dynamic rank accepts any [`IntoShapeVec`] input. For a static [`Rank<N>`],
/// arrays and array references of exactly `N` elements convert without a
/// runtime rank check; vectors, slices, and [`ShapeVec`] values are checked by
/// [`TensorRank::shape_from_vec`].
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{IntoRankShape, Rank};
///
/// let shape = <[usize; 2] as IntoRankShape<Rank<2>>>::into_rank_shape([2, 3])?;
/// assert_eq!(shape, [2, 3]);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
///
/// ```compile_fail
/// use tenferro_tensor_core::{IntoRankShape, Rank};
///
/// let _ = <[usize; 2] as IntoRankShape<Rank<3>>>::into_rank_shape([2, 3]);
/// ```
pub trait IntoRankShape<R: TensorRank> {
    /// Convert this shape container into `R`'s shape representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor_core::{IntoRankShape, Rank};
    /// let shape = <Vec<usize> as IntoRankShape<Rank<2>>>::into_rank_shape(vec![2, 3])?;
    /// assert_eq!(shape, [2, 3]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::RankMismatch`] when a vector or slice has
    /// a length different from the static rank.
    fn into_rank_shape(self) -> Result<R::Shape>;
}

/// Rank contract for tensor metadata shapes and strides.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Rank, TensorRank};
/// let shape = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3].into_iter().collect())?;
/// assert_eq!(shape.as_ref(), &[2, 3]);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
pub trait TensorRank: private::Sealed + Clone + Copy + Debug + Eq + Send + Sync + 'static {
    /// Static rank when known at compile time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DynRank, Rank, TensorRank};
    ///
    /// assert_eq!(DynRank::RANK, None);
    /// assert_eq!(Rank::<2>::RANK, Some(2));
    /// ```
    const RANK: Option<usize>;

    /// Shape representation for this rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DynRank, TensorRank};
    ///
    /// let shape: <DynRank as TensorRank>::Shape = vec![2, 3].into_iter().collect();
    /// assert_eq!(shape.as_ref(), &[2, 3]);
    /// ```
    type Shape: Clone + Debug + PartialEq + Eq + AsRef<[usize]> + IntoRankShape<Self>;

    /// Stride representation for this rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DynRank, TensorRank};
    ///
    /// let strides: <DynRank as TensorRank>::Strides = vec![1, 2].into_iter().collect();
    /// assert_eq!(strides.as_ref(), &[1, 2]);
    /// ```
    type Strides: Clone + Debug + PartialEq + Eq + AsRef<[isize]>;

    /// Convert a dynamic shape vector into this rank's shape representation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorRank};
    ///
    /// let shape = <Rank<1> as TensorRank>::shape_from_vec(vec![4].into_iter().collect())?;
    /// assert_eq!(shape.as_ref(), &[4]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::RankMismatch`] when the shape length does
    /// not equal the compile-time rank.
    fn shape_from_vec(shape: ShapeVec) -> Result<Self::Shape>;

    /// Convert this rank's shape representation into a dynamic shape vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorRank};
    ///
    /// let shape = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3].into_iter().collect())?;
    /// assert_eq!(<Rank<2> as TensorRank>::shape_into_vec(shape).as_slice(), &[2, 3]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    fn shape_into_vec(shape: Self::Shape) -> ShapeVec;

    /// Convert a dynamic stride vector into this rank's stride representation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorRank};
    ///
    /// let strides = <Rank<2> as TensorRank>::strides_from_vec(vec![1, 2].into_iter().collect())?;
    /// assert_eq!(strides.as_ref(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::RankMismatch`] when the stride length does
    /// not equal the compile-time rank.
    fn strides_from_vec(strides: StrideVec) -> Result<Self::Strides>;

    /// Convert this rank's stride representation into a dynamic stride vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorRank};
    ///
    /// let strides = <Rank<2> as TensorRank>::strides_from_vec(vec![1, 2].into_iter().collect())?;
    /// assert_eq!(<Rank<2> as TensorRank>::strides_into_vec(strides).as_slice(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    fn strides_into_vec(strides: Self::Strides) -> StrideVec;
}

/// Dynamic tensor rank marker.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{DynRank, TensorRank};
///
/// assert_eq!(DynRank::RANK, None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynRank;

/// Static tensor rank marker.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Rank, TensorRank};
///
/// assert_eq!(Rank::<3>::RANK, Some(3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rank<const N: usize>;

impl<S> IntoRankShape<DynRank> for S
where
    S: IntoShapeVec,
{
    fn into_rank_shape(self) -> Result<ShapeVec> {
        Ok(self.into_shape_vec())
    }
}

impl<const N: usize> IntoRankShape<Rank<N>> for [usize; N] {
    fn into_rank_shape(self) -> Result<[usize; N]> {
        Ok(self)
    }
}

impl<const N: usize> IntoRankShape<Rank<N>> for &[usize; N] {
    fn into_rank_shape(self) -> Result<[usize; N]> {
        Ok(*self)
    }
}

impl<const N: usize> IntoRankShape<Rank<N>> for Vec<usize> {
    fn into_rank_shape(self) -> Result<[usize; N]> {
        Rank::<N>::shape_from_vec(self.into_iter().collect())
    }
}

impl<const N: usize> IntoRankShape<Rank<N>> for &[usize] {
    fn into_rank_shape(self) -> Result<[usize; N]> {
        Rank::<N>::shape_from_vec(self.into_shape_vec())
    }
}

impl<const N: usize> IntoRankShape<Rank<N>> for ShapeVec {
    fn into_rank_shape(self) -> Result<[usize; N]> {
        Rank::<N>::shape_from_vec(self)
    }
}

impl TensorRank for DynRank {
    const RANK: Option<usize> = None;

    type Shape = ShapeVec;
    type Strides = StrideVec;

    fn shape_from_vec(shape: ShapeVec) -> Result<Self::Shape> {
        Ok(shape)
    }

    fn shape_into_vec(shape: Self::Shape) -> ShapeVec {
        shape
    }

    fn strides_from_vec(strides: StrideVec) -> Result<Self::Strides> {
        Ok(strides)
    }

    fn strides_into_vec(strides: Self::Strides) -> StrideVec {
        strides
    }
}

impl<const N: usize> TensorRank for Rank<N> {
    const RANK: Option<usize> = Some(N);

    type Shape = [usize; N];
    type Strides = [isize; N];

    fn shape_from_vec(shape: ShapeVec) -> Result<Self::Shape> {
        let actual = shape.len();
        let array: core::result::Result<&[usize; N], _> = shape.as_slice().try_into();
        array.copied().map_err(|_| ValidationError::RankMismatch {
            expected: N,
            actual,
        })
    }

    fn shape_into_vec(shape: Self::Shape) -> ShapeVec {
        ShapeVec::from_iter(shape)
    }

    fn strides_from_vec(strides: StrideVec) -> Result<Self::Strides> {
        let actual = strides.len();
        let array: core::result::Result<&[isize; N], _> = strides.as_slice().try_into();
        array.copied().map_err(|_| ValidationError::RankMismatch {
            expected: N,
            actual,
        })
    }

    fn strides_into_vec(strides: Self::Strides) -> StrideVec {
        StrideVec::from_iter(strides)
    }
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::DynRank {}
    impl<const N: usize> Sealed for super::Rank<N> {}
}
