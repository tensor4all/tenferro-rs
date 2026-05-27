use crate::{GatherConfig, TensorBackend};

use super::{Tensor, TypedTensor};

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> crate::Result<usize> {
    let normalized = if axis < 0 { rank as isize + axis } else { axis };
    if normalized < 0 || normalized >= rank as isize {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank,
        });
    }
    Ok(normalized as usize)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> crate::Result<usize> {
    let normalized = if axis < 0 {
        rank as isize + 1 + axis
    } else {
        axis
    };
    if normalized < 0 || normalized > rank as isize {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank: rank + 1,
        });
    }
    Ok(normalized as usize)
}

fn index_select_parts(
    shape: &[usize],
    axis: isize,
    positions: &[usize],
) -> crate::Result<(Tensor, GatherConfig)> {
    let axis = normalize_existing_axis("index_select", axis, shape.len())?;
    let axis_extent = shape[axis];
    for &position in positions {
        if position >= axis_extent {
            return Err(crate::Error::InvalidConfig {
                op: "index_select",
                message: format!(
                    "position {position} out of bounds for axis {axis} with extent {axis_extent}"
                ),
            });
        }
    }

    let mut slice_sizes = shape.to_vec();
    slice_sizes[axis] = 1;

    let offset_dims = (0..shape.len()).filter(|&dim| dim != axis).collect();
    let index_data = positions
        .iter()
        .map(|&position| {
            i64::try_from(position).map_err(|_| crate::Error::InvalidConfig {
                op: "index_select",
                message: format!("position {position} cannot be represented as i64"),
            })
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let indices = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![positions.len(), 1],
        index_data,
    ));

    let config = GatherConfig {
        offset_dims,
        collapsed_slice_dims: vec![axis],
        start_index_map: vec![axis],
        index_vector_dim: 1,
        slice_sizes,
    };

    Ok((indices, config))
}

impl Tensor {
    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, Tensor};
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]);
    /// let y = x.index_select(-1, &[2, 0], &mut backend).unwrap();
    ///
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn index_select(
        &self,
        axis: isize,
        positions: &[usize],
        ctx: &mut impl TensorBackend,
    ) -> crate::Result<Self> {
        let (indices, config) = index_select_parts(self.shape(), axis, positions)?;
        ctx.with_backend_session(|exec| exec.gather(self, &indices, &config))
    }

    /// Stack tensors along a newly inserted axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, Tensor};
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![], vec![1.0_f64]);
    /// let b = Tensor::from_vec_col_major(vec![], vec![2.0_f64]);
    /// let out = Tensor::stack(&[&a, &b], -1, &mut backend).unwrap();
    ///
    /// assert_eq!(out.shape(), &[2]);
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn stack(
        tensors: &[&Self],
        dim: isize,
        ctx: &mut impl TensorBackend,
    ) -> crate::Result<Self> {
        let first = tensors
            .first()
            .copied()
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "stack",
                message: "stack requires at least one input".into(),
            })?;
        let axis = normalize_insert_axis("stack", dim, first.shape().len())?;

        for tensor in tensors.iter().copied().skip(1) {
            if tensor.shape() != first.shape() {
                return Err(crate::Error::ShapeMismatch {
                    op: "stack",
                    lhs: first.shape().to_vec(),
                    rhs: tensor.shape().to_vec(),
                });
            }
        }

        let mut expanded_shape = first.shape().to_vec();
        expanded_shape.insert(axis, 1);

        ctx.with_backend_session(|exec| {
            let mut expanded = Vec::with_capacity(tensors.len());
            for tensor in tensors {
                expanded.push(exec.reshape(tensor, &expanded_shape)?);
            }
            let refs = expanded.iter().collect::<Vec<_>>();
            exec.concatenate(&refs, axis)
        })
    }
}
