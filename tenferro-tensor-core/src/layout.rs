use crate::{col_major_strides, DynRank, Error, Result, TensorRank};

/// Storage-neutral tensor layout metadata.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Rank, TensorLayout};
///
/// let layout = TensorLayout::<Rank<2>>::compact([2, 3])?;
/// assert_eq!(layout.shape(), &[2, 3]);
/// assert_eq!(layout.strides(), &[1, 2]);
/// # Ok::<(), tenferro_tensor_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLayout<R: TensorRank = DynRank> {
    shape: R::Shape,
    strides: R::Strides,
    offset: isize,
}

impl<R: TensorRank> TensorLayout<R> {
    /// Create a compact column-major layout with zero offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<2>>::compact([2, 3])?;
    /// assert_eq!(layout.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn compact(shape: R::Shape) -> Result<Self> {
        let strides = R::strides_from_vec(col_major_strides(shape.as_ref())?)?;
        Ok(Self {
            shape,
            strides,
            offset: 0,
        })
    }

    /// Create a layout from shape, strides, and element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DynRank, TensorLayout};
    ///
    /// let layout = TensorLayout::<DynRank>::from_parts(vec![2, 3].into(), vec![1, 2].into(), 0)?;
    /// assert!(layout.is_compact_col_major());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn from_parts(shape: R::Shape, strides: R::Strides, offset: isize) -> Result<Self> {
        if shape.as_ref().len() != strides.as_ref().len() {
            return Err(Error::RankMismatch {
                expected: shape.as_ref().len(),
                actual: strides.as_ref().len(),
            });
        }
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    /// Return the layout shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<1>>::compact([4])?;
    /// assert_eq!(layout.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }

    /// Return the layout strides in element units.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<2>>::compact([2, 3])?;
    /// assert_eq!(layout.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.strides.as_ref()
    }

    /// Return the layout element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DynRank, TensorLayout};
    ///
    /// let layout = TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![1].into(), 2)?;
    /// assert_eq!(layout.offset(), 2);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Return whether the layout has compact column-major strides.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<2>>::compact([2, 3])?;
    /// assert!(layout.is_compact_col_major());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn is_compact_col_major(&self) -> bool {
        col_major_strides(self.shape())
            .map(|strides| strides.as_slice() == self.strides())
            .unwrap_or(false)
    }
}
