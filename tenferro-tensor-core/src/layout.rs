use crate::{col_major_strides, DynRank, Error, Result, TensorRank};

pub(crate) fn reachable_offset_range(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<Option<(isize, isize)>> {
    if shape.iter().any(|&extent| extent == 0) {
        return Ok(None);
    }

    let mut min = offset;
    let mut max = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let last = isize::try_from(extent.saturating_sub(1)).map_err(|_| Error::IntegerOverflow)?;
        let delta = last.checked_mul(stride).ok_or(Error::IntegerOverflow)?;
        if delta < 0 {
            min = min.checked_add(delta).ok_or(Error::IntegerOverflow)?;
        } else {
            max = max.checked_add(delta).ok_or(Error::IntegerOverflow)?;
        }
    }
    Ok(Some((min, max)))
}

pub(crate) fn validate_reachable_bounds(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    buffer_len: usize,
) -> Result<()> {
    if shape.len() != strides.len() {
        return Err(Error::RankMismatch {
            expected: shape.len(),
            actual: strides.len(),
        });
    }

    match reachable_offset_range(shape, strides, offset)? {
        Some((min, max)) => {
            if min < 0 {
                return Err(Error::ViewOutOfBounds);
            }
            let max = usize::try_from(max).map_err(|_| Error::IntegerOverflow)?;
            if max < buffer_len {
                Ok(())
            } else {
                Err(Error::ViewOutOfBounds)
            }
        }
        None => {
            if offset < 0 {
                return Err(Error::ViewOutOfBounds);
            }
            let offset = usize::try_from(offset).map_err(|_| Error::IntegerOverflow)?;
            if offset <= buffer_len {
                Ok(())
            } else {
                Err(Error::ViewOutOfBounds)
            }
        }
    }
}

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

    /// Create a layout from shape, strides, element offset, and backing buffer length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DynRank, TensorLayout};
    ///
    /// let layout = TensorLayout::<DynRank>::from_parts(
    ///     vec![2, 3].into(),
    ///     vec![1, 2].into(),
    ///     0,
    ///     6,
    /// )?;
    /// assert!(layout.is_compact_col_major());
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn from_parts(
        shape: R::Shape,
        strides: R::Strides,
        offset: isize,
        buffer_len: usize,
    ) -> Result<Self> {
        validate_reachable_bounds(shape.as_ref(), strides.as_ref(), offset, buffer_len)?;
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
    /// let layout = TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![1].into(), 2, 5)?;
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
