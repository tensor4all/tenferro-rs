use crate::{
    checked_product, col_major_strides, validate_permutation, DynRank, Error, Result, ShapeVec,
    SliceSpec, StrideVec, TensorRank,
};
use smallvec::SmallVec;
use std::collections::HashSet;

/// Maximum logical elements for exact mutable-overlap validation.
///
/// Larger layouts must pass the sufficient stride-span proof. This keeps the
/// fallback bounded because it enumerates logical elements and stores visited
/// physical offsets.
const MUTABLE_NO_OVERLAP_EXACT_ELEMENT_LIMIT: usize = 4096;

pub(crate) fn reachable_offset_range(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<Option<(isize, isize)>> {
    if shape.contains(&0) {
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

fn layout_from_vecs<R: TensorRank>(
    shape: ShapeVec,
    strides: StrideVec,
    offset: isize,
    buffer_len: usize,
) -> Result<TensorLayout<R>> {
    TensorLayout::from_parts(
        R::shape_from_vec(shape)?,
        R::strides_from_vec(strides)?,
        offset,
        buffer_len,
    )
}

fn positive_ceil_div(numerator: isize, denominator: isize) -> Result<usize> {
    debug_assert!(numerator >= 0);
    debug_assert!(denominator > 0);
    let extent = if numerator == 0 {
        0
    } else {
        1 + (numerator - 1) / denominator
    };
    usize::try_from(extent).map_err(|_| Error::IntegerOverflow)
}

fn normalize_slice(slice: SliceSpec, axis_len: usize) -> Result<(isize, usize)> {
    if slice.step == 0 {
        return Err(Error::InvalidSliceStep { step: slice.step });
    }

    let axis_len = isize::try_from(axis_len).map_err(|_| Error::IntegerOverflow)?;
    if slice.step > 0 {
        let start = if slice.start < 0 {
            slice
                .start
                .checked_add(axis_len)
                .ok_or(Error::IntegerOverflow)?
        } else {
            slice.start
        };
        let end = if slice.end < 0 {
            slice
                .end
                .checked_add(axis_len)
                .ok_or(Error::IntegerOverflow)?
        } else {
            slice.end
        };
        if start < 0 || start > axis_len || end < 0 || end > axis_len {
            return Err(Error::InvalidSliceBounds {
                start: slice.start,
                end: slice.end,
                axis_len: usize::try_from(axis_len).map_err(|_| Error::IntegerOverflow)?,
            });
        }
        if start >= end {
            return Ok((start, 0));
        }
        return Ok((start, positive_ceil_div(end - start, slice.step)?));
    }

    let start = if slice.start < 0 {
        slice
            .start
            .checked_add(axis_len)
            .ok_or(Error::IntegerOverflow)?
    } else {
        slice.start
    };
    let end = slice.end;
    if start < 0 || start >= axis_len || end < -1 || end >= axis_len {
        return Err(Error::InvalidSliceBounds {
            start: slice.start,
            end: slice.end,
            axis_len: usize::try_from(axis_len).map_err(|_| Error::IntegerOverflow)?,
        });
    }
    if start <= end {
        return Ok((start, 0));
    }
    let step = slice.step.checked_neg().ok_or(Error::IntegerOverflow)?;
    Ok((start, positive_ceil_div(start - end, step)?))
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

    /// Validate that the layout can be used for mutable access without aliasing.
    ///
    /// Empty logical views are accepted. Non-empty layouts are accepted when a
    /// conservative stride-span proof succeeds, or when exact enumeration of a
    /// small bounded view proves that all logical elements map to distinct
    /// physical offsets.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DynRank, TensorLayout};
    ///
    /// let layout = TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![-1].into(), 2, 3)?;
    /// layout.validate_mutable_no_overlap()?;
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn validate_mutable_no_overlap(&self) -> Result<()> {
        if self.shape().contains(&0) {
            return Ok(());
        }

        for (&extent, &stride) in self.shape().iter().zip(self.strides()) {
            if extent > 1 && stride == 0 {
                return Err(Error::OverlappingMutableLayout);
            }
        }

        let element_count = checked_product(self.shape())?;

        let mut axes = self
            .shape()
            .iter()
            .zip(self.strides())
            .filter(|&(&extent, _)| extent > 1)
            .map(|(&extent, &stride)| (extent, stride.unsigned_abs()))
            .collect::<SmallVec<[(usize, usize); 8]>>();
        axes.sort_by_key(|&(_, stride)| stride);

        let mut span = 0usize;
        for (extent, stride) in axes {
            if stride <= span {
                return self.validate_mutable_no_overlap_exact_or_reject(element_count);
            }
            span = span
                .checked_add(
                    (extent - 1)
                        .checked_mul(stride)
                        .ok_or(Error::IntegerOverflow)?,
                )
                .ok_or(Error::IntegerOverflow)?;
        }

        Ok(())
    }

    fn validate_mutable_no_overlap_exact_or_reject(&self, element_count: usize) -> Result<()> {
        if element_count > MUTABLE_NO_OVERLAP_EXACT_ELEMENT_LIMIT {
            return Err(Error::OverlappingMutableLayout);
        }

        let mut seen = HashSet::with_capacity(element_count);
        let rank = self.shape().len();
        let mut indices = vec![0usize; rank];

        loop {
            let mut physical_offset = self.offset;
            for (&index, &stride) in indices.iter().zip(self.strides()) {
                let index = isize::try_from(index).map_err(|_| Error::IntegerOverflow)?;
                let delta = index.checked_mul(stride).ok_or(Error::IntegerOverflow)?;
                physical_offset = physical_offset
                    .checked_add(delta)
                    .ok_or(Error::IntegerOverflow)?;
            }

            if !seen.insert(physical_offset) {
                return Err(Error::OverlappingMutableLayout);
            }

            let mut axis = 0;
            while axis < rank {
                indices[axis] += 1;
                if indices[axis] < self.shape()[axis] {
                    break;
                }
                indices[axis] = 0;
                axis += 1;
            }
            if axis == rank {
                return Ok(());
            }
        }
    }

    /// Return a metadata-only axis permutation of this layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<2>>::compact([2, 3])?;
    /// let transposed = layout.transpose_view([1, 0])?;
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// assert_eq!(transposed.strides(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn transpose_view(&self, axes: impl AsRef<[usize]>) -> Result<Self> {
        let axes = axes.as_ref();
        validate_permutation(self.shape().len(), axes)?;
        let shape = axes
            .iter()
            .map(|&axis| self.shape()[axis])
            .collect::<ShapeVec>();
        let strides = axes
            .iter()
            .map(|&axis| self.strides()[axis])
            .collect::<StrideVec>();
        Ok(Self {
            shape: R::shape_from_vec(shape)?,
            strides: R::strides_from_vec(strides)?,
            offset: self.offset,
        })
    }

    /// Return a metadata-only slice of this layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, SliceSpec, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<1>>::compact([4])?;
    /// let view = layout.slice_view([SliceSpec { start: 3, end: -1, step: -2 }], 4)?;
    /// assert_eq!(view.shape(), &[2]);
    /// assert_eq!(view.strides(), &[-2]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn slice_view(&self, spec: impl AsRef<[SliceSpec]>, buffer_len: usize) -> Result<Self> {
        let spec = spec.as_ref();
        if spec.len() != self.shape().len() {
            return Err(Error::RankMismatch {
                expected: self.shape().len(),
                actual: spec.len(),
            });
        }

        let mut shape = ShapeVec::new();
        let mut strides = StrideVec::new();
        let mut offset = self.offset;
        for ((&axis_len, &stride), &slice) in self
            .shape()
            .iter()
            .zip(self.strides().iter())
            .zip(spec.iter())
        {
            let (start, extent) = normalize_slice(slice, axis_len)?;
            let start_offset = start.checked_mul(stride).ok_or(Error::IntegerOverflow)?;
            offset = offset
                .checked_add(start_offset)
                .ok_or(Error::IntegerOverflow)?;
            shape.push(extent);
            strides.push(
                stride
                    .checked_mul(slice.step)
                    .ok_or(Error::IntegerOverflow)?,
            );
        }
        layout_from_vecs(shape, strides, offset, buffer_len)
    }

    /// Return a metadata-only reshape of this compact column-major layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<2>>::compact([2, 3])?;
    /// let reshaped = layout.reshape_view_as::<Rank<1>>([6], 6)?;
    /// assert_eq!(reshaped.shape(), &[6]);
    /// assert_eq!(reshaped.strides(), &[1]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn reshape_view_as<R2: TensorRank>(
        &self,
        shape: R2::Shape,
        buffer_len: usize,
    ) -> Result<TensorLayout<R2>> {
        if !self.is_compact_col_major() {
            return Err(Error::NonContiguousViewAsSlice);
        }
        let from = checked_product(self.shape())?;
        let to = checked_product(shape.as_ref())?;
        if from != to {
            return Err(Error::ReshapeElementCountMismatch { from, to });
        }
        let strides = R2::strides_from_vec(col_major_strides(shape.as_ref())?)?;
        TensorLayout::from_parts(shape, strides, self.offset, buffer_len)
    }

    /// Return a metadata-only explicit broadcast of this layout into a target rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{Rank, TensorLayout};
    ///
    /// let layout = TensorLayout::<Rank<1>>::compact([3])?;
    /// let broadcast = layout.broadcast_in_dim_view::<Rank<2>>([2, 3], [1], 3)?;
    /// assert_eq!(broadcast.shape(), &[2, 3]);
    /// assert_eq!(broadcast.strides(), &[0, 1]);
    /// # Ok::<(), tenferro_tensor_core::Error>(())
    /// ```
    pub fn broadcast_in_dim_view<R2: TensorRank>(
        &self,
        shape: R2::Shape,
        broadcast_dims: impl AsRef<[usize]>,
        buffer_len: usize,
    ) -> Result<TensorLayout<R2>> {
        let broadcast_dims = broadcast_dims.as_ref();
        if broadcast_dims.len() != self.shape().len() {
            return Err(Error::RankMismatch {
                expected: self.shape().len(),
                actual: broadcast_dims.len(),
            });
        }

        let output_rank = shape.as_ref().len();
        let mut seen = vec![false; output_rank];
        let mut strides = StrideVec::new();
        strides.resize(output_rank, 0);
        for (input_axis, &output_axis) in broadcast_dims.iter().enumerate() {
            if output_axis >= output_rank {
                return Err(Error::AxisOutOfBounds {
                    axis: output_axis,
                    rank: output_rank,
                });
            }
            if seen[output_axis] {
                return Err(Error::DuplicateAxis { axis: output_axis });
            }
            seen[output_axis] = true;

            let input_extent = self.shape()[input_axis];
            let output_extent = shape.as_ref()[output_axis];
            if input_extent != output_extent && input_extent != 1 {
                return Err(Error::ShapeDataLengthMismatch {
                    expected: input_extent,
                    actual: output_extent,
                });
            }
            if input_extent == output_extent {
                strides[output_axis] = self.strides()[input_axis];
            }
        }

        TensorLayout::from_parts(
            shape,
            R2::strides_from_vec(strides)?,
            self.offset,
            buffer_len,
        )
    }
}
