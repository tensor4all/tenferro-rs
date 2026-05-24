use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;

use super::{for_each_index, DType, Tensor, TypedTensor};
use crate::{Error, Result};

type ShapeVec = SmallVec<[usize; 8]>;
type StrideVec = SmallVec<[isize; 8]>;

const NEW_OP: &str = "TypedStridedTensorView::new";
const MUT_NEW_OP: &str = "TypedStridedTensorViewMut::new";

fn invalid_config(op: &'static str, message: impl Into<String>) -> Error {
    Error::InvalidConfig {
        op,
        message: message.into(),
    }
}

fn checked_element_count(shape: &[usize], op: &'static str) -> Result<usize> {
    let mut product = 1usize;
    for &dim in shape {
        if dim == 0 {
            return Ok(0);
        }
        product = product.checked_mul(dim).ok_or_else(|| {
            invalid_config(op, format!("shape product overflows for shape {shape:?}"))
        })?;
    }
    Ok(product)
}

fn checked_isize(value: usize, op: &'static str, role: &'static str) -> Result<isize> {
    isize::try_from(value)
        .map_err(|_| invalid_config(op, format!("{role} value {value} does not fit in isize")))
}

fn checked_col_major_strides(shape: &[usize], op: &'static str) -> Result<StrideVec> {
    let mut strides = StrideVec::with_capacity(shape.len());
    if shape.is_empty() {
        return Ok(strides);
    }

    strides.push(1);
    let mut stride = 1isize;
    for axis in 1..shape.len() {
        let prev_extent = checked_isize(shape[axis - 1], op, "shape")?;
        stride = stride.checked_mul(prev_extent).ok_or_else(|| {
            invalid_config(
                op,
                format!("column-major stride overflows for shape {shape:?}"),
            )
        })?;
        strides.push(stride);
    }
    Ok(strides)
}

fn validate_parts(
    data_len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    op: &'static str,
) -> Result<usize> {
    if shape.len() != strides.len() {
        return Err(Error::RankMismatch {
            op,
            expected: shape.len(),
            actual: strides.len(),
        });
    }

    let element_count = checked_element_count(shape, op)?;
    let data_bound = checked_isize(data_len, op, "data length")?;

    if element_count == 0 {
        if (0..=data_bound).contains(&offset) {
            return Ok(0);
        }
        return Err(invalid_config(
            op,
            format!("empty view offset {offset} is outside 0..={data_len}"),
        ));
    }

    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let steps = checked_isize(extent - 1, op, "shape")?;
        let end = stride.checked_mul(steps).ok_or_else(|| {
            invalid_config(
                op,
                format!("stride span overflows for extent {extent} and stride {stride}"),
            )
        })?;
        let (axis_min, axis_max) = if end < 0 { (end, 0) } else { (0, end) };
        min_offset = min_offset
            .checked_add(axis_min)
            .ok_or_else(|| invalid_config(op, "minimum reachable offset overflows"))?;
        max_offset = max_offset
            .checked_add(axis_max)
            .ok_or_else(|| invalid_config(op, "maximum reachable offset overflows"))?;
    }

    if min_offset < 0 || max_offset >= data_bound {
        return Err(invalid_config(
            op,
            format!(
                "reachable offsets {min_offset}..={max_offset} are outside data length {data_len}"
            ),
        ));
    }

    Ok(element_count)
}

fn reachable_offset_span(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    op: &'static str,
) -> Result<Option<(usize, usize)>> {
    let element_count = checked_element_count(shape, op)?;
    if element_count == 0 {
        return Ok(None);
    }

    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let steps = checked_isize(extent - 1, op, "shape")?;
        let end = stride.checked_mul(steps).ok_or_else(|| {
            invalid_config(
                op,
                format!("stride span overflows for extent {extent} and stride {stride}"),
            )
        })?;
        let (axis_min, axis_max) = if end < 0 { (end, 0) } else { (0, end) };
        min_offset = min_offset
            .checked_add(axis_min)
            .ok_or_else(|| invalid_config(op, "minimum reachable offset overflows"))?;
        max_offset = max_offset
            .checked_add(axis_max)
            .ok_or_else(|| invalid_config(op, "maximum reachable offset overflows"))?;
    }

    let min_offset = usize::try_from(min_offset)
        .map_err(|_| invalid_config(op, "minimum reachable offset is negative"))?;
    let max_offset = usize::try_from(max_offset)
        .map_err(|_| invalid_config(op, "maximum reachable offset is negative"))?;
    Ok(Some((min_offset, max_offset)))
}

fn split_two_mut_ranges<T>(
    data: &mut [T],
    first: (usize, usize),
    second: (usize, usize),
) -> Option<(&mut [T], &mut [T])> {
    if first.1 < second.0 {
        let (_, after_first_start) = data.split_at_mut(first.0);
        let (first_slice, after_first) = after_first_start.split_at_mut(first.1 - first.0 + 1);
        let (_, after_gap) = after_first.split_at_mut(second.0 - first.1 - 1);
        let (second_slice, _) = after_gap.split_at_mut(second.1 - second.0 + 1);
        Some((first_slice, second_slice))
    } else if second.1 < first.0 {
        let (_, after_second_start) = data.split_at_mut(second.0);
        let (second_slice, after_second) = after_second_start.split_at_mut(second.1 - second.0 + 1);
        let (_, after_gap) = after_second.split_at_mut(first.0 - second.1 - 1);
        let (first_slice, _) = after_gap.split_at_mut(first.1 - first.0 + 1);
        Some((first_slice, second_slice))
    } else {
        None
    }
}

fn adjusted_offset(offset: isize, span_start: usize, op: &'static str) -> Option<isize> {
    let span_start = checked_isize(span_start, op, "span start").ok()?;
    offset.checked_sub(span_start)
}

fn strides_overlap(shape: &[usize], strides: &[isize]) -> bool {
    let mut axes = SmallVec::<[usize; 8]>::from_iter(0..shape.len());
    axes.sort_by_key(|&axis| strides[axis].unsigned_abs());

    let mut sum_prev_offsets = 0usize;
    for axis in axes {
        let extent = shape[axis];
        if extent == 0 {
            return false;
        }
        if extent <= 1 {
            continue;
        }

        let stride = strides[axis].unsigned_abs();
        if stride <= sum_prev_offsets {
            return true;
        }

        let Some(axis_span) = (extent - 1).checked_mul(stride) else {
            return true;
        };
        let Some(next_sum) = sum_prev_offsets.checked_add(axis_span) else {
            return true;
        };
        sum_prev_offsets = next_sum;
    }

    false
}

fn validate_mut_parts(
    data_len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    op: &'static str,
) -> Result<usize> {
    let element_count = validate_parts(data_len, shape, strides, offset, op)?;
    if element_count > 0 && strides_overlap(shape, strides) {
        return Err(invalid_config(
            op,
            "mutable strided views must not alias logical elements",
        ));
    }
    Ok(element_count)
}

fn checked_strided_offset(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    indices: &[usize],
) -> Option<usize> {
    if indices.len() != shape.len() {
        return None;
    }

    let mut offset = base_offset;
    for ((&idx, &extent), &stride) in indices.iter().zip(shape).zip(strides) {
        if idx >= extent {
            return None;
        }
        let idx = isize::try_from(idx).ok()?;
        let delta = stride.checked_mul(idx)?;
        offset = offset.checked_add(delta)?;
    }

    usize::try_from(offset).ok()
}

fn validate_permutation(axes: &[usize], rank: usize, op: &'static str) -> Result<()> {
    if axes.len() != rank {
        return Err(Error::RankMismatch {
            op,
            expected: rank,
            actual: axes.len(),
        });
    }

    let mut seen = SmallVec::<[bool; 8]>::from_elem(false, rank);
    for &axis in axes {
        if axis >= rank {
            return Err(Error::AxisOutOfBounds { op, axis, rank });
        }
        if seen[axis] {
            return Err(Error::DuplicateAxis {
                op,
                axis,
                role: "permutation",
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn is_col_major_contiguous(shape: &[usize], strides: &[isize], op: &'static str) -> Result<bool> {
    let mut expected = 1isize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent == 0 {
            return Ok(true);
        }
        if extent == 1 {
            continue;
        }
        if stride != expected {
            return Ok(false);
        }
        let extent = checked_isize(extent, op, "shape")?;
        expected = expected.checked_mul(extent).ok_or_else(|| {
            invalid_config(
                op,
                format!("column-major stride overflows for shape {shape:?}"),
            )
        })?;
    }
    Ok(true)
}

fn normalize_bound(
    bound: isize,
    extent: usize,
    op: &'static str,
    role: &'static str,
) -> Result<isize> {
    let extent = checked_isize(extent, op, "shape")?;
    let bound = if bound < 0 {
        extent
            .checked_add(bound)
            .ok_or_else(|| invalid_config(op, format!("{role} {bound} overflows")))?
    } else {
        bound
    };
    if !(0..=extent).contains(&bound) {
        return Err(invalid_config(
            op,
            format!("{role} {bound} is outside 0..={extent}"),
        ));
    }
    Ok(bound)
}

fn normalized_slice(
    slice: StridedSliceSpec,
    extent: usize,
    op: &'static str,
) -> Result<(usize, isize, isize)> {
    if slice.step == 0 {
        return Err(invalid_config(op, "slice step must not be zero"));
    }

    let start = normalize_bound(slice.start, extent, op, "slice start")?;
    let end = match slice.end {
        Some(end) => normalize_bound(end, extent, op, "slice end")?,
        None => checked_isize(extent, op, "shape")?,
    };
    if start >= end {
        return Ok((0, start, slice.step));
    }

    let range_len = usize::try_from(end - start)
        .map_err(|_| invalid_config(op, "slice range length overflows"))?;
    let step_abs = slice.step.unsigned_abs();
    let out_extent = range_len.div_ceil(step_abs);
    let first = if slice.step < 0 { end - 1 } else { start };
    Ok((out_extent, first, slice.step))
}

fn permuted_parts(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    axes: &[usize],
    op: &'static str,
) -> Result<(ShapeVec, StrideVec, isize)> {
    validate_permutation(axes, shape.len(), op)?;

    let mut next_shape = ShapeVec::with_capacity(axes.len());
    let mut next_strides = StrideVec::with_capacity(axes.len());
    for &axis in axes {
        next_shape.push(shape[axis]);
        next_strides.push(strides[axis]);
    }
    Ok((next_shape, next_strides, offset))
}

fn sliced_parts(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    slices: &[StridedSliceSpec],
    op: &'static str,
) -> Result<(ShapeVec, StrideVec, isize)> {
    let rank = shape.len();
    if slices.len() != rank {
        return Err(Error::RankMismatch {
            op,
            expected: rank,
            actual: slices.len(),
        });
    }

    let mut offset = base_offset;
    let mut next_shape = ShapeVec::with_capacity(rank);
    let mut next_strides = StrideVec::with_capacity(rank);
    for (axis, slice) in slices.iter().copied().enumerate() {
        let (out_extent, first, step) = normalized_slice(slice, shape[axis], op)?;
        let start_delta = strides[axis]
            .checked_mul(first)
            .ok_or_else(|| invalid_config(op, "slice offset delta overflows"))?;
        offset = offset
            .checked_add(start_delta)
            .ok_or_else(|| invalid_config(op, "slice offset overflows"))?;
        let stride = strides[axis]
            .checked_mul(step)
            .ok_or_else(|| invalid_config(op, "slice stride overflows"))?;
        next_shape.push(out_extent);
        next_strides.push(stride);
    }

    Ok((next_shape, next_strides, offset))
}

fn slice_axis_specs(
    rank: usize,
    axis: usize,
    slice: StridedSliceSpec,
    op: &'static str,
) -> Result<SmallVec<[StridedSliceSpec; 8]>> {
    if axis >= rank {
        return Err(Error::AxisOutOfBounds { op, axis, rank });
    }

    let mut slices = SmallVec::<[StridedSliceSpec; 8]>::from_elem(StridedSliceSpec::all(), rank);
    slices[axis] = slice;
    Ok(slices)
}

fn reshaped_strides(
    current_shape: &[usize],
    current_strides: &[isize],
    current_n_elements: usize,
    next_shape: &[usize],
    op: &'static str,
) -> Result<StrideVec> {
    let next_count = checked_element_count(next_shape, op)?;
    if current_n_elements != next_count {
        return Err(invalid_config(
            op,
            format!(
                "element count mismatch: current shape {:?} has {}, requested shape {:?} has {}",
                current_shape, current_n_elements, next_shape, next_count
            ),
        ));
    }
    if current_n_elements > 1 && !is_col_major_contiguous(current_shape, current_strides, op)? {
        return Err(invalid_config(
            op,
            "reshape without materialization requires a contiguous column-major view",
        ));
    }

    checked_col_major_strides(next_shape, op)
}

/// One-axis slice specification for [`TypedStridedTensorView::try_slice`].
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

/// Borrowed read-only view of host tensor data with arbitrary logical strides.
///
/// This is an adapter type for external host layouts such as row-major,
/// sliced, transposed, or reversed arrays. It does not change tenferro's
/// canonical compute representation: owned [`TypedTensor`] values and
/// [`super::TypedTensorView`] remain contiguous column-major.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::TypedStridedTensorView;
///
/// let row_major = [1.0_f64, 2.0, 3.0, 4.0];
/// let view = TypedStridedTensorView::new(&[2, 2], &[2, 1], 0, &row_major).unwrap();
///
/// assert_eq!(view.get(&[1, 0]), Some(&3.0));
/// ```
#[derive(Clone, Debug)]
pub struct TypedStridedTensorView<'a, T> {
    data: &'a [T],
    shape: ShapeVec,
    strides: StrideVec,
    offset: isize,
    n_elements: usize,
}

impl<'a, T> TypedStridedTensorView<'a, T> {
    /// Create a borrowed strided host view.
    ///
    /// `offset` and `strides` are measured in elements, not bytes. Negative
    /// strides are supported when every reachable element remains inside
    /// `data`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4, 5, 6];
    /// let view = TypedStridedTensorView::new(&[2, 3], &[3, 1], 0, &data).unwrap();
    ///
    /// assert_eq!(view.get(&[1, 2]), Some(&6));
    /// ```
    pub fn new(shape: &[usize], strides: &[isize], offset: isize, data: &'a [T]) -> Result<Self> {
        let n_elements = validate_parts(data.len(), shape, strides, offset, NEW_OP)?;
        Ok(Self {
            data,
            shape: ShapeVec::from_slice(shape),
            strides: StrideVec::from_slice(strides),
            offset,
            n_elements,
        })
    }

    /// Create a strided view over contiguous column-major data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1.0_f64, 2.0, 3.0, 4.0];
    /// let view = TypedStridedTensorView::from_col_major(&[2, 2], &data).unwrap();
    ///
    /// assert_eq!(view.strides(), &[1, 2]);
    /// ```
    pub fn from_col_major(shape: &[usize], data: &'a [T]) -> Result<Self> {
        let strides = checked_col_major_strides(shape, "TypedStridedTensorView::from_col_major")?;
        Self::new(shape, &strides, 0, data)
    }

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [true, false];
    /// let view = TypedStridedTensorView::new(&[2], &[1], 0, &data).unwrap();
    /// assert_eq!(view.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the logical strides measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2, 3];
    /// let view = TypedStridedTensorView::new(&[3], &[-1], 2, &data).unwrap();
    /// assert_eq!(view.strides(), &[-1]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Return the physical starting offset measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2, 3];
    /// let view = TypedStridedTensorView::new(&[2], &[1], 1, &data).unwrap();
    /// assert_eq!(view.offset(), 1);
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Return the borrowed physical host slice backing this view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2];
    /// let view = TypedStridedTensorView::new(&[2], &[1], 0, &data).unwrap();
    /// assert_eq!(view.as_physical_slice(), &[1, 2]);
    /// ```
    pub fn as_physical_slice(&self) -> &'a [T] {
        self.data
    }

    /// Return the number of logical elements in the view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [0; 6];
    /// let view = TypedStridedTensorView::new(&[2, 3], &[1, 2], 0, &data).unwrap();
    /// assert_eq!(view.n_elements(), 6);
    /// ```
    pub fn n_elements(&self) -> usize {
        self.n_elements
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2, 3];
    /// let view = TypedStridedTensorView::new(&[3], &[-1], 2, &data).unwrap();
    /// assert_eq!(view.try_linear_offset(&[2]), Some(0));
    /// ```
    pub fn try_linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_strided_offset(&self.shape, &self.strides, self.offset, indices)
    }

    /// Borrow one element by logical multi-index.
    ///
    /// Returns `None` when the rank or any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2, 3];
    /// let view = TypedStridedTensorView::new(&[3], &[1], 0, &data).unwrap();
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// assert_eq!(view.get(&[3]), None);
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.try_linear_offset(indices)?;
        self.data.get(offset)
    }

    /// Explicit alias for [`TypedStridedTensorView::get`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2];
    /// let view = TypedStridedTensorView::new(&[2], &[1], 0, &data).unwrap();
    /// assert_eq!(view.try_get(&[0]), Some(&1));
    /// ```
    pub fn try_get(&self, indices: &[usize]) -> Option<&T> {
        self.get(indices)
    }

    /// Return a metadata-only view with axes permuted.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2, 3, 4, 5, 6];
    /// let view = TypedStridedTensorView::new(&[2, 3], &[3, 1], 0, &data).unwrap();
    /// let transposed = view.try_permute_axes(&[1, 0]).unwrap();
    ///
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// assert_eq!(transposed.get(&[2, 1]), Some(&6));
    /// ```
    pub fn try_permute_axes(&self, axes: &[usize]) -> Result<Self> {
        const OP: &str = "TypedStridedTensorView::try_permute_axes";
        let (shape, strides, offset) =
            permuted_parts(&self.shape, &self.strides, self.offset, axes, OP)?;
        Self::new(&shape, &strides, offset, self.data)
    }

    /// Return a metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StridedSliceSpec, TypedStridedTensorView};
    ///
    /// let data = [0, 1, 2, 3];
    /// let view = TypedStridedTensorView::new(&[4], &[1], 0, &data).unwrap();
    /// let sliced = view.try_slice(&[StridedSliceSpec::new(0, None, -2)]).unwrap();
    ///
    /// assert_eq!(sliced.shape(), &[2]);
    /// assert_eq!(sliced.materialize_col_major().unwrap().as_slice(), &[3, 1]);
    /// ```
    pub fn try_slice(&self, slices: &[StridedSliceSpec]) -> Result<Self> {
        const OP: &str = "TypedStridedTensorView::try_slice";
        let (shape, strides, offset) =
            sliced_parts(&self.shape, &self.strides, self.offset, slices, OP)?;
        Self::new(&shape, &strides, offset, self.data)
    }

    /// Return a metadata-only slice along a single axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StridedSliceSpec, TypedStridedTensorView};
    ///
    /// let data = [1, 2, 3, 4];
    /// let view = TypedStridedTensorView::new(&[2, 2], &[1, 2], 0, &data).unwrap();
    /// let sliced = view.try_slice_axis(1, StridedSliceSpec::reverse()).unwrap();
    ///
    /// assert_eq!(sliced.get(&[0, 0]), Some(&3));
    /// ```
    pub fn try_slice_axis(&self, axis: usize, slice: StridedSliceSpec) -> Result<Self> {
        const OP: &str = "TypedStridedTensorView::try_slice_axis";
        let slices = slice_axis_specs(self.shape.len(), axis, slice, OP)?;
        self.try_slice(&slices)
    }

    /// Return a metadata-only reshape when the current view is representable
    /// as contiguous column-major logical storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let data = [1, 2, 3, 4];
    /// let view = TypedStridedTensorView::from_col_major(&[2, 2], &data).unwrap();
    /// let reshaped = view.try_reshape(&[4]).unwrap();
    ///
    /// assert_eq!(reshaped.shape(), &[4]);
    /// ```
    pub fn try_reshape(&self, shape: &[usize]) -> Result<Self> {
        const OP: &str = "TypedStridedTensorView::try_reshape";
        let strides = reshaped_strides(&self.shape, &self.strides, self.n_elements, shape, OP)?;
        Self::new(shape, &strides, self.offset, self.data)
    }
}

impl<T: Clone> TypedStridedTensorView<'_, T> {
    /// Materialize this adapter view into tenferro's canonical column-major
    /// owned tensor representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorView;
    ///
    /// let row_major = [1.0_f64, 2.0, 3.0, 4.0];
    /// let view = TypedStridedTensorView::new(&[2, 2], &[2, 1], 0, &row_major).unwrap();
    /// let tensor = view.materialize_col_major().unwrap();
    ///
    /// assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn materialize_col_major(&self) -> Result<TypedTensor<T>> {
        let mut data = Vec::with_capacity(self.n_elements);
        let mut error = None;
        for_each_index(&self.shape, |index| match self.get(index) {
            Some(value) => data.push(value.clone()),
            None => {
                if error.is_none() {
                    error = Some(invalid_config(
                        "TypedStridedTensorView::materialize_col_major",
                        format!("validated index {index:?} was not reachable"),
                    ));
                }
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        Ok(TypedTensor::from_vec_col_major(self.shape.to_vec(), data))
    }
}

/// Borrowed mutable view of host tensor data with arbitrary logical strides.
///
/// Mutable strided views reject layouts where two logical indices can refer to
/// the same backing element. This mirrors ndarray's mutable view invariant:
/// read-only views may alias, but mutable views must not expose overlapping
/// logical elements.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::TypedStridedTensorViewMut;
///
/// let mut row_major = [1_i32, 2, 3, 4];
/// let mut view = TypedStridedTensorViewMut::new(&[2, 2], &[2, 1], 0, &mut row_major).unwrap();
///
/// *view.get_mut(&[1, 0]).unwrap() = 40;
/// assert_eq!(view.as_physical_slice(), &[1, 2, 40, 4]);
/// ```
#[derive(Debug)]
pub struct TypedStridedTensorViewMut<'a, T> {
    data: &'a mut [T],
    shape: ShapeVec,
    strides: StrideVec,
    offset: isize,
    n_elements: usize,
}

impl<'a, T> TypedStridedTensorViewMut<'a, T> {
    /// Create a borrowed mutable strided host view.
    ///
    /// `offset` and `strides` are measured in elements, not bytes. Negative
    /// strides are supported when every reachable element remains inside
    /// `data`. Layouts with overlapping logical elements are rejected.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let mut view = TypedStridedTensorViewMut::new(&[3], &[-1], 2, &mut data).unwrap();
    ///
    /// *view.get_mut(&[2]).unwrap() = 10;
    /// assert_eq!(view.as_physical_slice(), &[10, 2, 3]);
    /// ```
    pub fn new(
        shape: &[usize],
        strides: &[isize],
        offset: isize,
        data: &'a mut [T],
    ) -> Result<Self> {
        let n_elements = validate_mut_parts(data.len(), shape, strides, offset, MUT_NEW_OP)?;
        Ok(Self {
            data,
            shape: ShapeVec::from_slice(shape),
            strides: StrideVec::from_slice(strides),
            offset,
            n_elements,
        })
    }

    /// Create a mutable view over contiguous column-major data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1.0_f64, 2.0, 3.0, 4.0];
    /// let view = TypedStridedTensorViewMut::from_col_major(&[2, 2], &mut data).unwrap();
    ///
    /// assert_eq!(view.strides(), &[1, 2]);
    /// ```
    pub fn from_col_major(shape: &[usize], data: &'a mut [T]) -> Result<Self> {
        let strides =
            checked_col_major_strides(shape, "TypedStridedTensorViewMut::from_col_major")?;
        Self::new(shape, &strides, 0, data)
    }

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [true, false];
    /// let view = TypedStridedTensorViewMut::new(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the logical strides measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2, 3];
    /// let view = TypedStridedTensorViewMut::new(&[3], &[-1], 2, &mut data).unwrap();
    /// assert_eq!(view.strides(), &[-1]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Return the physical starting offset measured in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2, 3];
    /// let view = TypedStridedTensorViewMut::new(&[2], &[1], 1, &mut data).unwrap();
    /// assert_eq!(view.offset(), 1);
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Return the borrowed physical host slice backing this view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2];
    /// let view = TypedStridedTensorViewMut::new(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.as_physical_slice(), &[1, 2]);
    /// ```
    pub fn as_physical_slice(&self) -> &[T] {
        self.data
    }

    /// Mutably borrow the physical host slice backing this view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2];
    /// let mut view = TypedStridedTensorViewMut::new(&[2], &[1], 0, &mut data).unwrap();
    /// view.as_physical_slice_mut()[0] = 3;
    /// assert_eq!(view.get(&[0]), Some(&3));
    /// ```
    pub fn as_physical_slice_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Return the number of logical elements in the view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [0; 6];
    /// let view = TypedStridedTensorViewMut::new(&[2, 3], &[1, 2], 0, &mut data).unwrap();
    /// assert_eq!(view.n_elements(), 6);
    /// ```
    pub fn n_elements(&self) -> usize {
        self.n_elements
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2, 3];
    /// let view = TypedStridedTensorViewMut::new(&[3], &[-1], 2, &mut data).unwrap();
    /// assert_eq!(view.try_linear_offset(&[2]), Some(0));
    /// ```
    pub fn try_linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_strided_offset(&self.shape, &self.strides, self.offset, indices)
    }

    /// Borrow one element by logical multi-index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2, 3];
    /// let view = TypedStridedTensorViewMut::new(&[3], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.try_linear_offset(indices)?;
        self.data.get(offset)
    }

    /// Mutably borrow one element by logical multi-index.
    ///
    /// Returns `None` when the rank or any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2, 3];
    /// let mut view = TypedStridedTensorViewMut::new(&[3], &[1], 0, &mut data).unwrap();
    /// *view.get_mut(&[1]).unwrap() = 20;
    /// assert_eq!(view.as_physical_slice(), &[1, 20, 3]);
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let offset = self.try_linear_offset(indices)?;
        self.data.get_mut(offset)
    }

    /// Explicit alias for [`TypedStridedTensorViewMut::get`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2];
    /// let view = TypedStridedTensorViewMut::new(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.try_get(&[0]), Some(&1));
    /// ```
    pub fn try_get(&self, indices: &[usize]) -> Option<&T> {
        self.get(indices)
    }

    /// Explicit alias for [`TypedStridedTensorViewMut::get_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2];
    /// let mut view = TypedStridedTensorViewMut::new(&[2], &[1], 0, &mut data).unwrap();
    /// *view.try_get_mut(&[0]).unwrap() = 3;
    /// assert_eq!(view.as_physical_slice(), &[3, 2]);
    /// ```
    pub fn try_get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        self.get_mut(indices)
    }

    /// Borrow this mutable view as a read-only strided view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2];
    /// let view = TypedStridedTensorViewMut::new(&[2], &[1], 0, &mut data).unwrap();
    /// assert_eq!(view.as_read_only().get(&[1]), Some(&2));
    /// ```
    pub fn as_read_only(&self) -> TypedStridedTensorView<'_, T> {
        TypedStridedTensorView {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            n_elements: self.n_elements,
        }
    }

    /// Convert this mutable view into a read-only strided view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2];
    /// let view = TypedStridedTensorViewMut::new(&[2], &[1], 0, &mut data).unwrap();
    /// let read_only = view.into_read_only();
    /// assert_eq!(read_only.get(&[0]), Some(&1));
    /// ```
    pub fn into_read_only(self) -> TypedStridedTensorView<'a, T> {
        TypedStridedTensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            offset: self.offset,
            n_elements: self.n_elements,
        }
    }

    /// Return a mutable metadata-only view with axes permuted.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2, 3, 4, 5, 6];
    /// let mut view = TypedStridedTensorViewMut::new(&[2, 3], &[3, 1], 0, &mut data).unwrap();
    /// {
    ///     let mut transposed = view.try_permute_axes(&[1, 0]).unwrap();
    ///     *transposed.get_mut(&[2, 1]).unwrap() = 60;
    /// }
    /// assert_eq!(view.get(&[1, 2]), Some(&60));
    /// ```
    pub fn try_permute_axes(&mut self, axes: &[usize]) -> Result<TypedStridedTensorViewMut<'_, T>> {
        const OP: &str = "TypedStridedTensorViewMut::try_permute_axes";
        let (shape, strides, offset) =
            permuted_parts(&self.shape, &self.strides, self.offset, axes, OP)?;
        TypedStridedTensorViewMut::new(&shape, &strides, offset, &mut *self.data)
    }

    /// Return a mutable metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StridedSliceSpec, TypedStridedTensorViewMut};
    ///
    /// let mut data = [0, 1, 2, 3];
    /// let mut view = TypedStridedTensorViewMut::new(&[4], &[1], 0, &mut data).unwrap();
    /// {
    ///     let mut sliced = view.try_slice(&[StridedSliceSpec::new(0, None, -2)]).unwrap();
    ///     *sliced.get_mut(&[0]).unwrap() = 30;
    /// }
    /// assert_eq!(view.as_physical_slice(), &[0, 1, 2, 30]);
    /// ```
    pub fn try_slice(
        &mut self,
        slices: &[StridedSliceSpec],
    ) -> Result<TypedStridedTensorViewMut<'_, T>> {
        const OP: &str = "TypedStridedTensorViewMut::try_slice";
        let (shape, strides, offset) =
            sliced_parts(&self.shape, &self.strides, self.offset, slices, OP)?;
        TypedStridedTensorViewMut::new(&shape, &strides, offset, &mut *self.data)
    }

    /// Return two mutable metadata-only slices when their physical offset
    /// ranges are disjoint.
    ///
    /// This returns `None` instead of panicking when either slice spec is
    /// invalid or the selected physical ranges overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StridedSliceSpec, TypedStridedTensorViewMut};
    ///
    /// let mut data = [1, 2, 3, 4];
    /// let mut view = TypedStridedTensorViewMut::new(&[4], &[1], 0, &mut data).unwrap();
    /// {
    ///     let (mut left, mut right) = view
    ///         .try_multi_slice_mut(
    ///             &[StridedSliceSpec::new(0, Some(2), 1)],
    ///             &[StridedSliceSpec::new(2, Some(4), 1)],
    ///         )
    ///         .unwrap();
    ///     *left.get_mut(&[1]).unwrap() = 20;
    ///     *right.get_mut(&[0]).unwrap() = 30;
    /// }
    /// assert_eq!(view.as_physical_slice(), &[1, 20, 30, 4]);
    /// ```
    pub fn try_multi_slice_mut(
        &mut self,
        first: &[StridedSliceSpec],
        second: &[StridedSliceSpec],
    ) -> Option<(
        TypedStridedTensorViewMut<'_, T>,
        TypedStridedTensorViewMut<'_, T>,
    )> {
        const OP: &str = "TypedStridedTensorViewMut::try_multi_slice_mut";
        let (first_shape, first_strides, first_offset) =
            sliced_parts(&self.shape, &self.strides, self.offset, first, OP).ok()?;
        let (second_shape, second_strides, second_offset) =
            sliced_parts(&self.shape, &self.strides, self.offset, second, OP).ok()?;

        validate_mut_parts(
            self.data.len(),
            &first_shape,
            &first_strides,
            first_offset,
            OP,
        )
        .ok()?;
        validate_mut_parts(
            self.data.len(),
            &second_shape,
            &second_strides,
            second_offset,
            OP,
        )
        .ok()?;

        match (
            reachable_offset_span(&first_shape, &first_strides, first_offset, OP).ok()?,
            reachable_offset_span(&second_shape, &second_strides, second_offset, OP).ok()?,
        ) {
            (Some(first_span), Some(second_span)) => {
                let first_offset = adjusted_offset(first_offset, first_span.0, OP)?;
                let second_offset = adjusted_offset(second_offset, second_span.0, OP)?;
                let (first_data, second_data) =
                    split_two_mut_ranges(&mut *self.data, first_span, second_span)?;
                let first_view = TypedStridedTensorViewMut::new(
                    &first_shape,
                    &first_strides,
                    first_offset,
                    first_data,
                )
                .ok()?;
                let second_view = TypedStridedTensorViewMut::new(
                    &second_shape,
                    &second_strides,
                    second_offset,
                    second_data,
                )
                .ok()?;
                Some((first_view, second_view))
            }
            (None, Some(second_span)) => {
                let (_, after_second_start) = self.data.split_at_mut(second_span.0);
                let (second_data, _) =
                    after_second_start.split_at_mut(second_span.1 - second_span.0 + 1);
                let second_offset = adjusted_offset(second_offset, second_span.0, OP)?;
                let first_view =
                    TypedStridedTensorViewMut::new(&first_shape, &first_strides, 0, &mut [])
                        .ok()?;
                let second_view = TypedStridedTensorViewMut::new(
                    &second_shape,
                    &second_strides,
                    second_offset,
                    second_data,
                )
                .ok()?;
                Some((first_view, second_view))
            }
            (Some(first_span), None) => {
                let (_, after_first_start) = self.data.split_at_mut(first_span.0);
                let (first_data, _) =
                    after_first_start.split_at_mut(first_span.1 - first_span.0 + 1);
                let first_offset = adjusted_offset(first_offset, first_span.0, OP)?;
                let first_view = TypedStridedTensorViewMut::new(
                    &first_shape,
                    &first_strides,
                    first_offset,
                    first_data,
                )
                .ok()?;
                let second_view =
                    TypedStridedTensorViewMut::new(&second_shape, &second_strides, 0, &mut [])
                        .ok()?;
                Some((first_view, second_view))
            }
            (None, None) => {
                let first_view =
                    TypedStridedTensorViewMut::new(&first_shape, &first_strides, 0, &mut [])
                        .ok()?;
                let second_view =
                    TypedStridedTensorViewMut::new(&second_shape, &second_strides, 0, &mut [])
                        .ok()?;
                Some((first_view, second_view))
            }
        }
    }

    /// Return a mutable metadata-only slice along a single axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{StridedSliceSpec, TypedStridedTensorViewMut};
    ///
    /// let mut data = [1, 2, 3, 4];
    /// let mut view = TypedStridedTensorViewMut::new(&[2, 2], &[1, 2], 0, &mut data).unwrap();
    /// {
    ///     let mut sliced = view.try_slice_axis(1, StridedSliceSpec::reverse()).unwrap();
    ///     *sliced.get_mut(&[0, 0]).unwrap() = 30;
    /// }
    /// assert_eq!(view.get(&[0, 1]), Some(&30));
    /// ```
    pub fn try_slice_axis(
        &mut self,
        axis: usize,
        slice: StridedSliceSpec,
    ) -> Result<TypedStridedTensorViewMut<'_, T>> {
        const OP: &str = "TypedStridedTensorViewMut::try_slice_axis";
        let slices = slice_axis_specs(self.shape.len(), axis, slice, OP)?;
        self.try_slice(&slices)
    }

    /// Return a mutable metadata-only reshape for contiguous column-major views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut data = [1, 2, 3, 4];
    /// let mut view = TypedStridedTensorViewMut::from_col_major(&[2, 2], &mut data).unwrap();
    /// let reshaped = view.try_reshape(&[4]).unwrap();
    ///
    /// assert_eq!(reshaped.shape(), &[4]);
    /// ```
    pub fn try_reshape(&mut self, shape: &[usize]) -> Result<TypedStridedTensorViewMut<'_, T>> {
        const OP: &str = "TypedStridedTensorViewMut::try_reshape";
        let strides = reshaped_strides(&self.shape, &self.strides, self.n_elements, shape, OP)?;
        TypedStridedTensorViewMut::new(shape, &strides, self.offset, &mut *self.data)
    }
}

impl<T: Clone> TypedStridedTensorViewMut<'_, T> {
    /// Materialize this mutable adapter view into tenferro's canonical
    /// column-major owned tensor representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedStridedTensorViewMut;
    ///
    /// let mut row_major = [1.0_f64, 2.0, 3.0, 4.0];
    /// let view = TypedStridedTensorViewMut::new(&[2, 2], &[2, 1], 0, &mut row_major).unwrap();
    /// let tensor = view.materialize_col_major().unwrap();
    ///
    /// assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn materialize_col_major(&self) -> Result<TypedTensor<T>> {
        self.as_read_only().materialize_col_major()
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
    F32(TypedStridedTensorView<'a, f32>),
    F64(TypedStridedTensorView<'a, f64>),
    I32(TypedStridedTensorView<'a, i32>),
    I64(TypedStridedTensorView<'a, i64>),
    Bool(TypedStridedTensorView<'a, bool>),
    C32(TypedStridedTensorView<'a, Complex32>),
    C64(TypedStridedTensorView<'a, Complex64>),
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
            Ok(Self::$variant(TypedStridedTensorView::new(
                shape, strides, offset, data,
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
            Self::F32(t) => Ok(Tensor::F32(t.materialize_col_major()?)),
            Self::F64(t) => Ok(Tensor::F64(t.materialize_col_major()?)),
            Self::I32(t) => Ok(Tensor::I32(t.materialize_col_major()?)),
            Self::I64(t) => Ok(Tensor::I64(t.materialize_col_major()?)),
            Self::Bool(t) => Ok(Tensor::Bool(t.materialize_col_major()?)),
            Self::C32(t) => Ok(Tensor::C32(t.materialize_col_major()?)),
            Self::C64(t) => Ok(Tensor::C64(t.materialize_col_major()?)),
        }
    }
}

/// Dynamic borrowed mutable strided host tensor view.
///
/// The dynamic mutable view supports all dtypes represented by tenferro's
/// compute [`Tensor`]. Like [`TypedStridedTensorViewMut`], constructors reject
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
    F32(TypedStridedTensorViewMut<'a, f32>),
    F64(TypedStridedTensorViewMut<'a, f64>),
    I32(TypedStridedTensorViewMut<'a, i32>),
    I64(TypedStridedTensorViewMut<'a, i64>),
    Bool(TypedStridedTensorViewMut<'a, bool>),
    C32(TypedStridedTensorViewMut<'a, Complex32>),
    C64(TypedStridedTensorViewMut<'a, Complex64>),
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
            Ok(Self::$variant(TypedStridedTensorViewMut::new(
                shape, strides, offset, data,
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
