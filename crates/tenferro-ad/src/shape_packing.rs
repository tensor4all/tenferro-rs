use std::ops::Range;

use tenferro_tensor::{GatherConfig, SliceConfig, Tensor, TypedTensor};

use crate::eager::{EagerSession, EagerTensor};
use crate::error::{Error, Result};

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        rank.checked_sub(axis.unsigned_abs()).ok_or_else(|| {
            tenferro_tensor::Error::axis_out_of_bounds(op, axis.unsigned_abs(), rank)
        })?
    };
    if normalized >= rank {
        return Err(
            tenferro_tensor::Error::axis_out_of_bounds(op, axis.unsigned_abs(), rank).into(),
        );
    }
    Ok(normalized)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let insert_rank = rank
        .checked_add(1)
        .ok_or_else(|| tenferro_tensor::Error::axis_out_of_bounds(op, axis.unsigned_abs(), rank))?;
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        insert_rank
            .checked_sub(axis.unsigned_abs())
            .ok_or_else(|| {
                tenferro_tensor::Error::axis_out_of_bounds(op, axis.unsigned_abs(), insert_rank)
            })?
    };
    if normalized > rank {
        return Err(tenferro_tensor::Error::axis_out_of_bounds(
            op,
            axis.unsigned_abs(),
            insert_rank,
        )
        .into());
    }
    Ok(normalized)
}

fn index_select_config(
    shape: &[usize],
    axis: isize,
    positions: &[usize],
) -> Result<(Tensor, GatherConfig)> {
    let axis = normalize_existing_axis("index_select", axis, shape.len())?;
    let axis_extent = shape[axis];
    for &position in positions {
        if position >= axis_extent {
            return Err(tenferro_tensor::Error::invalid_argument(
                "index_select",
                "position",
                format!(
                    "position {position} out of bounds for axis {axis} with extent {axis_extent}"
                ),
            )
            .into());
        }
    }

    let mut slice_sizes = shape.to_vec();
    slice_sizes[axis] = 1;

    let offset_dims = (0..shape.len()).filter(|&dim| dim != axis).collect();
    let index_data = positions
        .iter()
        .map(|&position| {
            i64::try_from(position).map_err(|_| {
                tenferro_tensor::Error::invalid_argument(
                    "index_select",
                    "position",
                    format!("position {position} cannot be represented as i64"),
                )
            })
        })
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    let indices = Tensor::from_typed::<i64>(TypedTensor::from_vec_col_major(
        vec![positions.len(), 1],
        index_data,
    )?);

    let config = GatherConfig {
        offset_dims,
        collapsed_slice_dims: vec![axis],
        start_index_map: vec![axis],
        index_vector_dim: 1,
        slice_sizes,
    };

    Ok((indices, config))
}

fn validate_stack_shapes(op: &'static str, shapes: &[&[usize]]) -> Result<()> {
    let Some(first) = shapes.first() else {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "inputs",
            "stack requires at least one input",
        )
        .into());
    };
    for shape in shapes.iter().skip(1) {
        if *shape != *first {
            return Err(tenferro_tensor::Error::shape_mismatch(op, *first, *shape).into());
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
enum AxisSelection {
    Slice {
        axis: usize,
        range: Range<usize>,
        step: usize,
    },
    Take {
        axis: usize,
        indices: Vec<usize>,
    },
}

fn validate_axis_selection(
    op: &'static str,
    rank: usize,
    seen: &mut [bool],
    axis: usize,
) -> Result<()> {
    if axis >= rank {
        return Err(tenferro_tensor::Error::axis_out_of_bounds(op, axis, rank).into());
    }
    if seen[axis] {
        return Err(tenferro_tensor::Error::duplicate_axis(op, axis, "selection").into());
    }
    seen[axis] = true;
    Ok(())
}

fn apply_slice_axis_config(
    op: &'static str,
    shape: &[usize],
    selections: &[AxisSelection],
) -> Result<Option<SliceConfig>> {
    let mut starts = vec![0; shape.len()];
    let mut limits = shape.to_vec();
    let mut strides = vec![1; shape.len()];
    let mut has_slice = false;
    for selection in selections {
        let AxisSelection::Slice { axis, range, step } = selection else {
            continue;
        };
        if *step == 0 {
            return Err(tenferro_tensor::Error::invalid_argument(
                op,
                "step",
                format!("axis {axis} has zero step"),
            )
            .into());
        }
        let extent = shape[*axis];
        if range.start > range.end || range.end > extent {
            return Err(tenferro_tensor::Error::invalid_argument(
                op,
                "range",
                format!(
                    "axis {axis} range {}..{} is out of bounds for extent {extent}",
                    range.start, range.end
                ),
            )
            .into());
        }
        starts[*axis] = range.start;
        limits[*axis] = range.end;
        strides[*axis] = *step;
        has_slice = true;
    }
    Ok(has_slice.then_some(SliceConfig {
        starts,
        limits,
        strides,
    }))
}

/// Rank-preserving eager tensor slicing builder.
///
/// Unspecified axes are kept whole. Range selections become one `Slice`
/// operation; host-known position selections become `Gather`/`index_select`
/// operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new()?;
/// let x = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![0.0_f64; 12]).unwrap(),
///     ctx,
/// ).unwrap();
/// let y = x.runtime().with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| x.slice_builder().axis(0, 0..2).axis_step(1, 0..4, 2).apply(s))?;
/// assert_eq!(y.shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct EagerSliceBuilder<'a> {
    tensor: &'a EagerTensor,
    selections: Vec<AxisSelection>,
}

impl<'a> EagerSliceBuilder<'a> {
    fn new(tensor: &'a EagerTensor) -> Self {
        Self {
            tensor,
            selections: Vec::new(),
        }
    }

    /// Add an exclusive-end range selection for one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.runtime().with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| x.slice_builder().axis(0, 1..3).apply(s))?;
    /// assert_eq!(y.shape(), &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn axis(mut self, axis: usize, range: Range<usize>) -> Self {
        self.selections.push(AxisSelection::Slice {
            axis,
            range,
            step: 1,
        });
        self
    }

    /// Add an exclusive-end strided range selection for one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.runtime().with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| x.slice_builder().axis_step(0, 0..5, 2).apply(s))?;
    /// assert_eq!(y.shape(), &[3]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn axis_step(mut self, axis: usize, range: Range<usize>, step: usize) -> Self {
        self.selections
            .push(AxisSelection::Slice { axis, range, step });
        self
    }

    /// Add a host-known position selection for one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.runtime().with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| x.slice_builder().take_axis(0, &[2, 0]).apply(s))?;
    /// assert_eq!(y.shape(), &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn take_axis(mut self, axis: usize, indices: &[usize]) -> Self {
        self.selections.push(AxisSelection::Take {
            axis,
            indices: indices.to_vec(),
        });
        self
    }

    /// Build and apply the requested slice/take operations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.runtime().with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| x.slice_builder().axis(0, 1..4).apply(s))?;
    /// assert_eq!(y.shape(), &[3]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::ValidationError::AxisOutOfBounds`] or
    /// `DuplicateAxis` when selections address an invalid/repeated axis,
    /// `InvalidArgument` for zero steps or out-of-bounds ranges, or a typed
    /// backend/runtime-state error while applying the selections.
    pub fn apply(self, session: &mut EagerSession<'_>) -> Result<EagerTensor> {
        session.ensure_runtime(self.tensor)?;
        let shape = self.tensor.shape().to_vec();
        let mut seen = vec![false; shape.len()];
        for selection in &self.selections {
            let axis = match selection {
                AxisSelection::Slice { axis, .. } | AxisSelection::Take { axis, .. } => *axis,
            };
            validate_axis_selection("slice_builder", shape.len(), &mut seen, axis)?;
        }

        let mut output = self.tensor.clone();
        if let Some(config) = apply_slice_axis_config("slice_builder", &shape, &self.selections)? {
            output = session.slice(&output, config)?;
        }
        for selection in self.selections {
            if let AxisSelection::Take { axis, indices } = selection {
                output = session.take_axis(&output, axis, &indices)?;
            }
        }
        Ok(output)
    }
}

impl EagerTensor {
    /// Start a rank-preserving slicing builder for this tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.runtime().with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| x.slice_builder().axis(0, 0..2).apply(s))?;
    /// assert_eq!(y.shape(), &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn slice_builder(&self) -> EagerSliceBuilder<'_> {
        EagerSliceBuilder::new(self)
    }
}

impl EagerSession<'_> {
    /// Select positions from one axis using a borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?)?;
    ///     s.index_select(&x, -1, &[2, 0])
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[3.0, 1.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns a typed foreign-runtime, invalid-axis/index, or backend error.
    pub fn index_select(
        &mut self,
        tensor: &EagerTensor,
        axis: isize,
        positions: &[usize],
    ) -> Result<EagerTensor> {
        self.ensure_runtime(tensor)?;
        let (indices, config) = index_select_config(tensor.shape(), axis, positions)?;
        let indices = self.constant_from_host(indices)?;
        self.gather(tensor, &indices, config)
    }

    /// Select entries from an axis by host-known positions.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    ///     s.take_axis(&x, 0, &[1])
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns a typed foreign-runtime, invalid-axis/index, or backend error.
    pub fn take_axis(
        &mut self,
        tensor: &EagerTensor,
        axis: usize,
        positions: &[usize],
    ) -> Result<EagerTensor> {
        self.ensure_runtime(tensor)?;
        let axis = isize::try_from(axis).map_err(|_| {
            Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
                "take_axis",
                "axis",
                format!("{axis} cannot be represented as isize"),
            ))
        })?;
        self.index_select(tensor, axis, positions)
    }

    /// Select matrix rows by host-known positions.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?)?;
    ///     s.take_rows(&x, &[1])
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns a typed foreign-runtime, invalid-row, or backend error.
    pub fn take_rows(&mut self, tensor: &EagerTensor, rows: &[usize]) -> Result<EagerTensor> {
        self.take_axis(tensor, 0, rows)
    }

    /// Select matrix columns by host-known positions.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0])?)?;
    ///     s.take_cols(&x, &[1])
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns a typed foreign-runtime, invalid-column, or backend error.
    pub fn take_cols(&mut self, tensor: &EagerTensor, cols: &[usize]) -> Result<EagerTensor> {
        self.take_axis(tensor, 1, cols)
    }

    /// Select a matrix block by host-known row and column positions.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?)?;
    ///     s.take_block(&x, &[1], &[0])
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns a typed foreign-runtime, invalid-row/column, or backend error.
    pub fn take_block(
        &mut self,
        tensor: &EagerTensor,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<EagerTensor> {
        let rows = self.take_rows(tensor, rows)?;
        self.take_cols(&rows, cols)
    }

    /// Slice one axis using an exclusive-end range in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?)?;
    ///     s.slice_axis(&x, 0, 1..3)
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns a typed foreign-runtime, invalid-axis/range, or backend error.
    pub fn slice_axis(
        &mut self,
        tensor: &EagerTensor,
        axis: usize,
        range: Range<usize>,
    ) -> Result<EagerTensor> {
        tensor.slice_builder().axis(axis, range).apply(self)
    }

    /// Stack eager tensors along a new axis in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major(vec![], vec![1.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64])?)?;
    ///     s.stack(&[&a, &b], -1)
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns a typed empty-input, invalid-axis/shape, foreign-runtime, or backend error.
    pub fn stack(&mut self, tensors: &[&EagerTensor], dim: isize) -> Result<EagerTensor> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
                "stack",
                "inputs",
                "stack requires at least one input",
            ))
        })?;
        let shapes = tensors
            .iter()
            .map(|tensor| tensor.shape())
            .collect::<Vec<_>>();
        validate_stack_shapes("stack", &shapes)?;
        let axis = normalize_insert_axis("stack", dim, first.shape().len())?;
        let mut expanded_shape = first.shape().to_vec();
        expanded_shape.insert(axis, 1);
        let expanded = tensors
            .iter()
            .map(|tensor| self.reshape(tensor, &expanded_shape))
            .collect::<Result<Vec<_>>>()?;
        let refs = expanded.iter().collect::<Vec<_>>();
        self.concatenate(&refs, axis)
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_existing_axis, normalize_insert_axis};

    #[test]
    fn axis_normalization_handles_ranks_larger_than_isize_max() {
        assert_eq!(normalize_existing_axis("test", 0, usize::MAX).unwrap(), 0);
        assert_eq!(
            normalize_existing_axis("test", -1, usize::MAX).unwrap(),
            usize::MAX - 1
        );
        assert_eq!(
            normalize_insert_axis("test", -1, usize::MAX - 1).unwrap(),
            usize::MAX - 1
        );
        assert!(normalize_insert_axis("test", -1, usize::MAX).is_err());
    }
}
