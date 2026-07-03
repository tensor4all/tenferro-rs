use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueRef};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{GatherConfig, Tensor, TypedTensor};

use crate::checkpoint::CheckpointNode;
use crate::error::{Error, Result};
use crate::metadata::{register_scoped_value_metadata, tensor_meta, MetadataScopeChain};
use crate::shape_infer::promote_dtypes;
use crate::sym_dim::SymDim;
use crate::traced::{
    apply_binary_preserve_input_dtypes, infer_traced_single_output_shape, merge_traced_inputs_map,
    next_traced_id, try_concrete_shape,
};
use crate::TracedTensor;

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        rank.checked_sub(axis.unsigned_abs())
            .ok_or(tenferro_tensor::Error::AxisOutOfBounds {
                op,
                axis: axis.unsigned_abs(),
                rank,
            })?
    };
    if normalized >= rank {
        return Err(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank,
        }
        .into());
    }
    Ok(normalized)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let insert_rank = rank
        .checked_add(1)
        .ok_or(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank,
        })?;
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        insert_rank.checked_sub(axis.unsigned_abs()).ok_or(
            tenferro_tensor::Error::AxisOutOfBounds {
                op,
                axis: axis.unsigned_abs(),
                rank: insert_rank,
            },
        )?
    };
    if normalized > rank {
        return Err(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank: insert_rank,
        }
        .into());
    }
    Ok(normalized)
}

fn index_select_config(
    shape: &[usize],
    axis: isize,
    positions: &[usize],
) -> Result<(Tensor, GatherConfig, Vec<usize>)> {
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

    let mut out_shape = shape.to_vec();
    out_shape[axis] = positions.len();

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

    Ok((indices, config, out_shape))
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

impl TracedTensor {
    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::from_tensor_concrete_shape(
    ///     Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap(),
    /// )
    /// .unwrap();
    /// let y = x.index_select(-1, &[2, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    ///
    /// assert_eq!(
    ///     out.as_slice::<f64>().unwrap(),
    ///     &[30.0, 10.0],
    /// );
    /// ```
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self> {
        let shape = try_concrete_shape(self).ok_or_else(|| Error::InvalidGraphBuild {
            op: "index_select",
            message: "index_select requires a concrete shape hint".into(),
        })?;
        let (indices_tensor, config, out_shape) = index_select_config(&shape, axis, positions)?;
        let indices = TracedTensor::from_tensor_concrete_shape(indices_tensor)?;
        Ok(apply_binary_preserve_input_dtypes(
            StdTensorOp::Gather(config),
            self,
            &indices,
            out_shape.len(),
            Some(out_shape.into_iter().map(SymDim::from).collect()),
            self.dtype,
        ))
    }

    /// Stack tensors along a newly inserted axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap()).unwrap();
    /// let b = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap()).unwrap();
    /// let stacked = TracedTensor::stack(&[&a, &b], -1).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&stacked).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    ///
    /// assert_eq!(
    ///     out.as_slice::<f64>().unwrap(),
    ///     &[1.0, 2.0],
    /// );
    /// ```
    pub fn stack(tensors: &[&Self], dim: isize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "stack",
                message: "stack requires at least one input".into(),
            })
        })?;
        let first_shape = try_concrete_shape(first).ok_or_else(|| Error::InvalidGraphBuild {
            op: "stack",
            message: "stack requires concrete shape hints".into(),
        })?;
        let mut shapes = Vec::with_capacity(tensors.len());
        shapes.push(first_shape.as_slice());
        let mut owned_shapes = Vec::with_capacity(tensors.len().saturating_sub(1));
        for tensor in tensors.iter().copied().skip(1) {
            owned_shapes.push(try_concrete_shape(tensor).ok_or_else(|| {
                Error::InvalidGraphBuild {
                    op: "stack",
                    message: "stack requires concrete shape hints".into(),
                }
            })?);
        }
        shapes.extend(owned_shapes.iter().map(Vec::as_slice));
        validate_stack_shapes("stack", &shapes)?;

        let axis = normalize_insert_axis("stack", dim, first.rank)?;
        let mut expanded_shape = first_shape;
        expanded_shape.insert(axis, 1);
        let mut out_shape = expanded_shape.clone();
        out_shape[axis] = tensors.len();
        let expanded = tensors
            .iter()
            .map(|tensor| tensor.reshape(&expanded_shape))
            .collect::<Result<Vec<_>>>()?;
        let refs = expanded.iter().collect::<Vec<_>>();
        Ok(apply_nary_concatenate(
            &refs,
            axis,
            out_shape.into_iter().map(SymDim::from).collect(),
        ))
    }

    /// Concatenate tensors along one existing axis.
    pub fn concatenate(tensors: &[&Self], axis: usize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "concatenate",
                message: "concatenate requires at least one input".into(),
            })
        })?;
        if axis >= first.rank {
            return Err(tenferro_tensor::Error::AxisOutOfBounds {
                op: "concatenate",
                axis,
                rank: first.rank,
            }
            .into());
        }
        for tensor in tensors.iter().copied().skip(1) {
            if tensor.rank != first.rank {
                return Err(tenferro_tensor::Error::RankMismatch {
                    op: "concatenate",
                    expected: first.rank,
                    actual: tensor.rank,
                }
                .into());
            }
        }

        let op = StdTensorOp::Concatenate {
            axis,
            input_count: tensors.len(),
        };
        let (_, out_shape_hint) =
            infer_traced_single_output_shape("TracedTensor::concatenate", &op, tensors)?;
        let out_shape = out_shape_hint.ok_or_else(|| {
            Error::Internal("concatenate shape inference returned no shape hint".into())
        })?;
        Ok(apply_nary_concatenate(tensors, axis, out_shape))
    }
}

fn apply_nary_concatenate(
    tensors: &[&TracedTensor],
    axis: usize,
    out_shape: Vec<SymDim>,
) -> TracedTensor {
    let out_dtype = promote_dtypes(tensors.iter().map(|tensor| tensor.dtype));
    let tensors = tensors
        .iter()
        .map(|tensor| {
            if tensor.dtype != out_dtype {
                tensor.cast(out_dtype)
            } else {
                (*tensor).clone()
            }
        })
        .collect::<Vec<_>>();

    let mut builder = GraphBuilder::new();
    for tensor in &tensors {
        builder.add_parent(tensor.graph.clone());
    }
    let input_refs = tensors
        .iter()
        .map(|tensor| ValueRef::External(tensor.graph.values()[tensor.val].key.clone()))
        .collect::<Vec<_>>();
    let outputs = builder.add_operation(
        StdTensorOp::Concatenate {
            axis,
            input_count: tensors.len(),
        },
        input_refs,
        OperationRole::Primary,
    );
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    // Callers route through shape inference before graph construction.
    let metadata_scope = register_scoped_value_metadata(
        graph.values()[outputs[0]].key.clone(),
        tensor_meta(out_dtype, out_shape.clone()),
    )
    .expect("fresh concatenate output metadata registration failed");

    let inputs_map = merge_traced_inputs_map(tensors.iter());
    let mut extra_roots = Vec::new();
    let mut checkpoint_chain = None;
    for tensor in &tensors {
        extra_roots.extend(tensor.extra_roots.iter().cloned());
        checkpoint_chain =
            CheckpointNode::merge_chains(checkpoint_chain, tensor.checkpoint_chain.clone());
    }
    TracedTensor {
        id: next_traced_id(),
        rank: out_shape.len(),
        dtype: out_dtype,
        graph,
        val: outputs[0],
        data: None,
        shape_hint: Some(out_shape),
        inputs_map,
        extra_roots,
        checkpoint_chain,
        metadata_scopes: MetadataScopeChain::with_new(
            metadata_scope,
            tensors.iter().map(|tensor| &tensor.metadata_scopes),
        ),
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
