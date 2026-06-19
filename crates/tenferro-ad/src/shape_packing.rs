use tenferro_tensor::{GatherConfig, Tensor, TensorDeviceTransfer, TypedTensor};

use crate::eager::EagerTensor;
use crate::error::{Error, Result};

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 { rank as isize + axis } else { axis };
    if normalized < 0 || normalized >= rank as isize {
        return Err(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank,
        }
        .into());
    }
    Ok(normalized as usize)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        rank as isize + 1 + axis
    } else {
        axis
    };
    if normalized < 0 || normalized > rank as isize {
        return Err(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank: rank + 1,
        }
        .into());
    }
    Ok(normalized as usize)
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
            return Err(tenferro_tensor::Error::InvalidConfig {
                op: "index_select",
                message: format!(
                    "position {position} out of bounds for axis {axis} with extent {axis_extent}"
                ),
            }
            .into());
        }
    }

    let mut slice_sizes = shape.to_vec();
    slice_sizes[axis] = 1;

    let offset_dims = (0..shape.len()).filter(|&dim| dim != axis).collect();
    let index_data = positions
        .iter()
        .map(|&position| {
            i64::try_from(position).map_err(|_| tenferro_tensor::Error::InvalidConfig {
                op: "index_select",
                message: format!("position {position} cannot be represented as i64"),
            })
        })
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    let indices = Tensor::I64(TypedTensor::from_vec_col_major(
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
        return Err(tenferro_tensor::Error::InvalidConfig {
            op,
            message: "stack requires at least one input".into(),
        }
        .into());
    };
    for shape in shapes.iter().skip(1) {
        if *shape != *first {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op,
                lhs: first.to_vec(),
                rhs: shape.to_vec(),
            }
            .into());
        }
    }
    Ok(())
}

impl EagerTensor {
    /// Select entries from one axis using host-known indices.
    ///
    /// The index list is primal metadata: gradients flow to `self`, including
    /// accumulation for repeated indices, but not to the selected positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.take_axis(0, &[2, 0]).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn take_axis(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        let axis = isize::try_from(axis).map_err(|_| {
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "take_axis",
                message: format!("axis {axis} cannot be represented as isize"),
            })
        })?;
        self.index_select(axis, indices)
    }

    /// Select matrix rows using host-known row indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.take_rows(&[1]).unwrap();
    ///
    /// assert_eq!(y.shape(), &[1, 2]);
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    pub fn take_rows(&self, rows: &[usize]) -> Result<Self> {
        self.take_axis(0, rows)
    }

    /// Select matrix columns using host-known column indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.take_cols(&[1]).unwrap();
    ///
    /// assert_eq!(y.shape(), &[2, 1]);
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);
    /// ```
    pub fn take_cols(&self, cols: &[usize]) -> Result<Self> {
        self.take_axis(1, cols)
    }

    /// Select a matrix block using host-known row and column indices.
    ///
    /// This is a convenience wrapper over row selection followed by column
    /// selection. The row and column lists, plus the approximation rank implied
    /// by their lengths, are fixed primal metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.take_block(&[1], &[0]).unwrap();
    ///
    /// assert_eq!(y.shape(), &[1, 1]);
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[2.0]);
    /// ```
    pub fn take_block(&self, rows: &[usize], cols: &[usize]) -> Result<Self> {
        self.take_rows(rows)?.take_cols(cols)
    }

    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let y = x.index_select(-1, &[2, 0]).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self> {
        let (indices, config) = index_select_config(self.shape(), axis, positions)?;
        let indices = {
            let mut backend = self
                .ctx
                .backend
                .lock()
                .map_err(|_| Error::Internal("backend lock poisoned".to_string()))?;
            backend.upload_host_tensor(&indices)?
        };
        let indices = self.ctx.constant_from(indices)?;
        self.gather(&indices, config)
    }

    /// Stack tensors along a newly inserted axis.
    ///
    /// The returned tensor uses the context of the first input, matching
    /// [`Self::concatenate`]. All inputs must belong to that same context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap(), ctx).unwrap();
    /// let out = EagerTensor::stack(&[&a, &b], -1).unwrap();
    ///
    /// assert_eq!(out.shape(), &[2]);
    /// assert_eq!(out.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn stack(tensors: &[&Self], dim: isize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "stack",
                message: "stack requires at least one input".into(),
            })
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
            .map(|tensor| tensor.reshape(&expanded_shape))
            .collect::<Result<Vec<_>>>()?;
        let refs = expanded.iter().collect::<Vec<_>>();
        Self::concatenate(&refs, axis)
    }
}
