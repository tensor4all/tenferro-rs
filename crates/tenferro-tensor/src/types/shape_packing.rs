use crate::{GatherConfig, TensorBackend};

use super::{Tensor, TypedTensor};

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> crate::Result<usize> {
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        rank.checked_sub(axis.unsigned_abs())
            .ok_or(crate::Error::AxisOutOfBounds {
                op,
                axis: axis.unsigned_abs(),
                rank,
            })?
    };
    if normalized >= rank {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank,
        });
    }
    Ok(normalized)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> crate::Result<usize> {
    let insert_rank = rank.checked_add(1).ok_or(crate::Error::AxisOutOfBounds {
        op,
        axis: axis.unsigned_abs(),
        rank,
    })?;
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        insert_rank
            .checked_sub(axis.unsigned_abs())
            .ok_or(crate::Error::AxisOutOfBounds {
                op,
                axis: axis.unsigned_abs(),
                rank: insert_rank,
            })?
    };
    if normalized > rank {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank: insert_rank,
        });
    }
    Ok(normalized)
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

impl Tensor {
    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend};
    ///
    /// fn select_last_axis<B: TensorBackend>(
    ///     backend: &mut B,
    ///     x: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     x.index_select(-1, &[2, 0], backend)
    /// }
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
    /// use tenferro_tensor::{Tensor, TensorBackend};
    ///
    /// fn stack_scalars<B: TensorBackend>(
    ///     backend: &mut B,
    ///     a: &Tensor,
    ///     b: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     Tensor::stack(&[a, b], -1, backend)
    /// }
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
